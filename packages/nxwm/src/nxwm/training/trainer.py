"""WorldModelTrainer composing the pure primitives in flow_matching/unrolled.

Precision modes, gradient clipping, checkpoint resume, and wandb logging
live here; the math lives in pure functions (``unrolled_forward``,
``flow_match_loss``, ``apply_noise_augmentation``, ``lpips_step_loss``) so
it can be reused for evals and smoke tests.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
import torch.optim as optim
from nxml_core.checkpoint import save_checkpoint

from .config import TrainingConfig
from .flow_matching import fa_dit_flow_match_loss, sample_logit_normal
from .lpips import lpips_step_loss
from .noise_aug import apply_noise_augmentation, apply_noise_augmentation_with_taus
from .unrolled import unrolled_forward, unrolled_forward_fa_dit


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Strip a DDP wrapper if present (``model.module``)."""
    return model.module if hasattr(model, "module") else model


class WorldModelTrainer:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        train_loader,
        val_loader,
        device,
        config: TrainingConfig,
        architecture: str,
        is_main_node: bool = True,
        checkpoint: dict | None = None,
        vae: torch.nn.Module | None = None,
        lpips_fn: torch.nn.Module | None = None,
    ):
        self.device = device
        self.is_main_node = is_main_node
        self.config = config
        self.architecture = architecture
        self.train_loader = train_loader
        self.val_loader = val_loader

        if config.precision not in ("fp16", "bf16"):
            raise ValueError(f"precision must be 'fp16' or 'bf16', got {config.precision!r}")
        self._use_bf16 = config.precision == "bf16"
        self._autocast_dtype = torch.bfloat16 if self._use_bf16 else torch.float16

        if config.compile_model:
            if is_main_node:
                print("torch.compile enabled — first batch will be slow (compiling)")
            model = torch.compile(model)  # type: ignore[assignment]
        self.model = model

        self.run_dir = Path(config.run_dir)
        if is_main_node:
            self.run_dir.mkdir(parents=True, exist_ok=True)

        self.vae = vae
        self.lpips_fn = lpips_fn
        if config.lpips_weight > 0 and (vae is None or lpips_fn is None):
            raise ValueError("lpips_weight > 0 requires both vae and lpips_fn")

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=not self._use_bf16)

        if checkpoint is not None:
            opt_state = checkpoint.get("optimizer_state_dict") or checkpoint.get(
                "training_metadata", {}
            ).get("optimizer_state_dict")
            if opt_state is not None:
                self.optimizer.load_state_dict(opt_state)
                if is_main_node:
                    print("Loaded optimizer state from checkpoint")

        self.best_loss = float("inf")
        self.best_val_loss = float("inf")

        # Lazy wandb import — training extra is optional.
        self._wandb = None
        if is_main_node:
            try:
                import wandb

                self._wandb = wandb
                wandb.init(
                    project=config.project_name,
                    config={"architecture": architecture, **asdict(config)},
                )
            except ImportError:
                if is_main_node:
                    print("wandb not installed; continuing without logging")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _active_K(self, epoch: int, start_epoch: int) -> int:
        c = self.config
        if c.unroll_ramp_epochs <= 0 or c.unroll_steps <= c.unroll_min:
            return c.unroll_steps
        elapsed = epoch - start_epoch
        frac = min(elapsed / c.unroll_ramp_epochs, 1.0)
        return max(
            c.unroll_min,
            int(c.unroll_min + (c.unroll_steps - c.unroll_min) * frac),
        )

    def _to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def _build_extra_step_loss(self):
        """Closure that yields LPIPS loss on k==0 only (compute-economy)."""
        if self.config.lpips_weight <= 0:
            return None
        weight = self.config.lpips_weight
        n_subset = self.config.lpips_subset_size
        autocast_dtype = self._autocast_dtype
        vae = self.vae
        lpips_fn = self.lpips_fn

        def extra(*, k: int, v_pred, x0, x1):
            if k != 0:
                return None
            return weight * lpips_step_loss(
                v_pred, x0, x1, vae, lpips_fn, n_subset=n_subset, autocast_dtype=autocast_dtype
            )

        return extra

    # ------------------------------------------------------------------
    # Train / validate
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int, *, start_epoch: int = 0) -> tuple[float, dict | None]:
        self.model.train()
        K = self._active_K(epoch, start_epoch)
        extra = self._build_extra_step_loss()
        total_loss_sum = 0.0
        n_batches = 0
        last_batch = None

        for batch in self.train_loader:
            last_batch = batch
            batch = self._to_device(batch)

            if self.architecture == "fa_dit":
                # fa_dit consumes the per-frame taus directly (model conditions
                # on noise level via its noise_embed), so use the variant that
                # surfaces them.
                noised, past_taus = apply_noise_augmentation_with_taus(
                    batch["observations"],
                    prob=self.config.noise_aug_prob,
                    scale=self.config.noise_aug_scale,
                )
                batch["observations"] = noised
                batch["past_taus"] = past_taus
            else:
                batch["observations"] = apply_noise_augmentation(
                    batch["observations"],
                    prob=self.config.noise_aug_prob,
                    scale=self.config.noise_aug_scale,
                )

            with torch.autocast("cuda", dtype=self._autocast_dtype):
                if self.architecture == "fa_dit":
                    out = unrolled_forward_fa_dit(
                        self.model,
                        batch,
                        K,
                        cfg_dropout_prob=self.config.cfg_dropout_prob,
                        tbptt_window=self.config.tbptt_window,
                        plan_loss_weight=self.config.plan_loss_weight,
                    )
                else:
                    out = unrolled_forward(
                        self.model,
                        batch,
                        K,
                        cfg_dropout_prob=self.config.cfg_dropout_prob,
                        tbptt_window=self.config.tbptt_window,
                        extra_step_loss=extra,
                    )
                loss = out["loss"]

            self.optimizer.zero_grad()
            if self._use_bf16:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )
                self.optimizer.step()
            else:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()

            total_loss_sum += float(loss.detach())
            n_batches += 1

            if self.is_main_node and self._wandb is not None:
                log = {
                    "train/loss": float(loss.detach()),
                    "train/mse_loss": float(out["mse"].detach()),
                    "train/grad_norm": float(grad_norm),
                    "train/epoch": epoch,
                    "train/unroll_steps": K,
                }
                if self.config.lpips_weight > 0:
                    log["train/lpips_loss"] = float(out["extra"].detach())
                if self.architecture == "fa_dit" and out.get("plan_count", 0) > 0:
                    log["train/plan_loss"] = float(out["plan"].detach())
                self._wandb.log(log)

        avg = total_loss_sum / max(n_batches, 1)
        if self.is_main_node:
            print(f"  Epoch {epoch} avg train loss: {avg:.6f} (K={K})")
        return avg, last_batch

    @torch.no_grad()
    def val_epoch(self, epoch: int, train_batch=None) -> float:
        self.model.eval()
        loss_sum = 0.0
        n_batches = 0
        for batch in self.val_loader:
            batch = self._to_device(batch)
            with torch.autocast("cuda", dtype=self._autocast_dtype):
                if self.architecture == "fa_dit":
                    _, _, loss, _ = fa_dit_flow_match_loss(
                        self.model,
                        batch["targets"][:, 0],
                        batch["observations"],
                        batch["actions"],
                        batch["future_latents"],
                        batch["future_poses"],
                        batch["future_actions"][:, 0],
                        cfg_dropout_prob=0.0,
                    )
                else:
                    obs = batch["observations"]
                    x1 = batch["targets"][:, 0]
                    actions = batch["actions"]
                    goal = batch["goals"][:, 0]
                    B = x1.shape[0]
                    x0 = torch.randn_like(x1)
                    t = sample_logit_normal(B, self.device)
                    t_view = t.view(B, 1, 1, 1)
                    xt = t_view * x1 + (1 - t_view) * x0
                    v_target = x1 - x0
                    v_pred = self.model(xt, t, obs, actions, goal)
                    loss = torch.mean((v_pred - v_target) ** 2)
            loss_sum += float(loss)
            n_batches += 1
        avg = loss_sum / max(n_batches, 1)
        if self.is_main_node:
            print(f"  Val loss: {avg:.6f}")
            if self._wandb is not None:
                self._wandb.log({"val/loss": avg, "val/epoch": epoch})
            if avg < self.best_val_loss:
                self.best_val_loss = avg
                self._save("best_val.pt", epoch=epoch, loss=avg)
            self._maybe_log_eval_gif(epoch)
        self.model.train()
        return avg

    # ------------------------------------------------------------------
    # Eval-rollout GIF (wandb panel)
    # ------------------------------------------------------------------

    def _maybe_log_eval_gif(self, epoch: int) -> None:
        cfg = self.config
        every = cfg.eval_gif_every_n_epochs
        if every <= 0 or self.vae is None:
            return
        if (epoch + 1) % every != 0:
            return

        from nxwm.training.eval_gif import generate_eval_gif

        out_path = self.run_dir / "rollouts" / f"epoch_{epoch:04d}.gif"
        try:
            generate_eval_gif(
                model=_unwrap(self.model),
                vae=self.vae,
                val_loader=self.val_loader,
                device=self.device,
                output_path=out_path,
                frames=cfg.eval_gif_frames,
                flow_steps=cfg.eval_gif_flow_steps,
                cfg_scale=cfg.eval_gif_cfg_scale,
                mode=cfg.eval_gif_mode,
                episode_idx=cfg.eval_gif_episode_idx,
                start_frame=cfg.eval_gif_start_frame,
            )
        except Exception as e:
            print(f"  eval-gif failed: {type(e).__name__}: {e}")
            return

        print(f"  eval-gif: {out_path}")
        if self._wandb is not None:
            self._wandb.log(
                {
                    "eval/rollout": self._wandb.Video(str(out_path), fps=30, format="gif"),
                    "eval/epoch": epoch,
                }
            )

    # ------------------------------------------------------------------
    # Checkpoints (self-describing, via nxml_core)
    # ------------------------------------------------------------------

    def _save(self, name: str, *, epoch: int, loss: float) -> None:
        if not self.is_main_node:
            return
        raw = _unwrap(self.model)
        config = getattr(raw, "config", None)
        if config is None:
            raise RuntimeError("model is missing .config; cannot self-describe checkpoint")
        save_checkpoint(
            architecture=self.architecture,
            config=config,
            state_dict=raw.state_dict(),
            path=self.run_dir / name,
            extra={
                "training_metadata": {
                    "epoch": epoch,
                    "loss": loss,
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scaler_state_dict": self.scaler.state_dict() if not self._use_bf16 else None,
                },
            },
        )

    def save_checkpoint(self, epoch: int, epoch_loss: float) -> None:
        self._save("latest.pt", epoch=epoch, loss=epoch_loss)
        if epoch_loss < self.best_loss:
            self.best_loss = epoch_loss
            self._save("best.pt", epoch=epoch, loss=epoch_loss)
            if self.is_main_node:
                print(f"New best checkpoint saved (loss: {epoch_loss:.6f})")
