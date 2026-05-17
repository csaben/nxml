"""BC trainer.

Loss = stick MSE + button BCE-with-logits, weighted per-sample by
``action_weight`` on frames where the human did something (any stick
above ``stick_deadzone`` or any button pressed).

The trainer is algorithm-agnostic at the registry level — it just
implements the ``train_epoch`` / ``val_epoch`` / ``save_checkpoint``
surface that the launcher drives.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from nxml_core.checkpoint import save_checkpoint
from tqdm import tqdm

from nxrl.algorithms.bc.config import BCAlgorithmConfig
from nxrl.core.registry import algorithm_registry


@algorithm_registry.register("bc", config_cls=BCAlgorithmConfig)
class BCTrainer:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        train_loader,
        val_loader,
        device: torch.device,
        config: BCAlgorithmConfig,
        policy_name: str,
        policy_config,
        is_main_node: bool,
        run_dir: Path,
        checkpoint: dict | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.policy_name = policy_name
        self.policy_config = policy_config
        self.is_main_node = is_main_node
        self.run_dir = run_dir
        self._wandb_run = None

        if self.is_main_node:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._init_wandb()

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.scaler = torch.amp.GradScaler("cuda")  # type: ignore[attr-defined]

        if checkpoint is not None and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if is_main_node:
                print("Loaded optimizer state from checkpoint")

        self.best_loss = float("inf")
        self.best_val_loss = float("inf")

    def _init_wandb(self) -> None:
        try:
            import wandb  # type: ignore[import-not-found]
        except ImportError:
            print("(wandb not installed; skipping logging)")
            return
        from dataclasses import asdict, is_dataclass

        if is_dataclass(self.policy_config) and not isinstance(self.policy_config, type):
            policy_cfg_dict = asdict(self.policy_config)
        else:
            policy_cfg_dict = dict(self.policy_config)  # type: ignore[arg-type]
        cfg_dict = {
            "policy": {"name": self.policy_name, "config": policy_cfg_dict},
            "algorithm": asdict(self.config),
            "run_dir": str(self.run_dir),
        }
        self._wandb_run = wandb.init(project=self.config.project_name, config=cfg_dict)

    def _wandb_log(self, payload: dict) -> None:
        if self._wandb_run is None:
            return
        import wandb  # type: ignore[import-not-found]

        wandb.log(payload)

    def _compute_losses(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        stick_loss = F.mse_loss(pred[:, :4], target[:, :4], reduction="none").mean(dim=1)
        button_loss = F.binary_cross_entropy_with_logits(
            pred[:, 4:], target[:, 4:], reduction="none"
        ).mean(dim=1)
        raw_loss = stick_loss + self.config.button_loss_weight * button_loss
        is_active = (target[:, :4].abs() > self.config.stick_deadzone).any(dim=1) | (
            target[:, 4:] > 0.5
        ).any(dim=1)
        return stick_loss, button_loss, raw_loss, is_active

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", disable=not self.is_main_node)

        epoch_loss_sum = 0.0
        num_batches = 0
        for batch in pbar:
            obs = batch["observations"].to(self.device)
            target = batch["action"].to(self.device)

            with torch.autocast("cuda"):
                pred = self.model(obs)
                stick_loss, button_loss, raw_loss, is_active = self._compute_losses(pred, target)
                weights = torch.where(
                    is_active,
                    self.config.action_weight,
                    1.0,
                )
                loss = (raw_loss * weights).mean()

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            epoch_loss_sum += loss.item()
            num_batches += 1

            if self.is_main_node:
                active_frac = is_active.float().mean().item()
                self._wandb_log(
                    {
                        "train/loss": loss.item(),
                        "train/stick_loss": stick_loss.mean().item(),
                        "train/button_loss": button_loss.mean().item(),
                        "train/grad_norm": grad_norm.item(),
                        "train/active_frac": active_frac,
                        "train/epoch": epoch,
                    }
                )
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "act": f"{active_frac:.0%}"})

        avg_loss = epoch_loss_sum / max(num_batches, 1)
        if self.is_main_node:
            print(f"  Epoch {epoch} avg train loss: {avg_loss:.6f}")
        return avg_loss

    @torch.no_grad()
    def val_epoch(self, epoch: int) -> float:
        self.model.eval()
        if len(self.val_loader) == 0:
            if self.is_main_node:
                print(
                    f"  Val skipped: val loader is empty "
                    f"(dataset size {len(self.val_loader.dataset)}). "
                    f"Likely sequence_length > available val frames with align_starts=True."
                )
            self.model.train()
            return float("nan")
        val_loss_sum = 0.0
        val_stick_sum = 0.0
        val_button_sum = 0.0
        val_active_loss_sum = 0.0
        val_active_count = 0
        num_batches = 0
        last_target_size = 0

        for batch in self.val_loader:
            obs = batch["observations"].to(self.device)
            target = batch["action"].to(self.device)

            with torch.autocast("cuda"):
                pred = self.model(obs)
                stick_loss, button_loss, raw_loss, is_active = self._compute_losses(pred, target)
                n_active = is_active.sum().item()
                if n_active > 0:
                    val_active_loss_sum += raw_loss[is_active].sum().item()
                    val_active_count += n_active

            val_loss_sum += raw_loss.mean().item()
            val_stick_sum += stick_loss.mean().item()
            val_button_sum += button_loss.mean().item()
            num_batches += 1
            last_target_size = target.size(0)

        avg_val_loss = val_loss_sum / max(num_batches, 1)
        avg_val_stick = val_stick_sum / max(num_batches, 1)
        avg_val_button = val_button_sum / max(num_batches, 1)
        avg_active_loss = (
            val_active_loss_sum / val_active_count if val_active_count > 0 else 0.0
        )

        if self.is_main_node:
            self._wandb_log(
                {
                    "val/loss": avg_val_loss,
                    "val/stick_loss": avg_val_stick,
                    "val/button_loss": avg_val_button,
                    "val/active_loss": avg_active_loss,
                    "val/active_frac": val_active_count
                    / max(num_batches * max(last_target_size, 1), 1),
                    "val/epoch": epoch,
                }
            )
            print(
                f"  Val loss: {avg_val_loss:.6f} "
                f"(stick: {avg_val_stick:.6f}, button: {avg_val_button:.6f}, "
                f"active: {avg_active_loss:.6f})"
            )
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self._save("best_val.pt", epoch=epoch, loss=avg_val_loss)
                print(f"  New best val checkpoint saved (val_loss: {avg_val_loss:.6f})")

        self.model.train()
        return avg_val_loss

    def save_checkpoint(self, epoch: int, train_loss: float) -> None:
        if not self.is_main_node:
            return
        self._save("latest.pt", epoch=epoch, loss=train_loss)
        if train_loss < self.best_loss:
            self.best_loss = train_loss
            self._save("best.pt", epoch=epoch, loss=train_loss)
            print(f"New best checkpoint saved (loss: {train_loss:.6f})")

    def _save(self, name: str, *, epoch: int, loss: float) -> None:
        raw_model = self.model.module if hasattr(self.model, "module") else self.model  # type: ignore[union-attr]
        save_checkpoint(
            architecture=self.policy_name,
            config=self.policy_config,
            state_dict=raw_model.state_dict(),
            path=self.run_dir / name,
            extra={
                "optimizer_state_dict": self.optimizer.state_dict(),
                "training_metadata": {"epoch": epoch, "loss": loss},
                "algorithm": "bc",
            },
        )
