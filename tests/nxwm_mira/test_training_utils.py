"""Ported training utilities: LR schedule, EMAs, DistributedMetric, periodic_event."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from nxwm_mira.training.ema import DistributedEMA, ModelEMA
from nxwm_mira.training.lr_schedule import WarmupConstantCosineDecayLR
from nxwm_mira.training.metrics import DistributedMetric
from nxwm_mira.training.tracker import periodic_event


def _optimizer(lr: float = 1e-4) -> torch.optim.Optimizer:
    return torch.optim.AdamW([nn.Parameter(torch.zeros(1))], lr=lr)


class TestLRSchedule:
    def test_warmup_then_constant(self) -> None:
        opt = _optimizer(lr=1e-4)
        sched = WarmupConstantCosineDecayLR(
            opt, warmup_steps=10, constant_steps=5, decay_steps=0, min_lr=1e-6
        )
        lrs = []
        for _ in range(20):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
            sched.step()
        assert lrs[0] < 1e-5  # warmup starts low
        assert lrs[9] < lrs[10] or lrs[10] == pytest.approx(1e-4, rel=0.2)
        assert lrs[-1] == pytest.approx(1e-4)  # decay disabled: stays at base

    def test_cosine_decays_to_min(self) -> None:
        opt = _optimizer(lr=1e-4)
        sched = WarmupConstantCosineDecayLR(
            opt, warmup_steps=2, constant_steps=0, decay_steps=10, min_lr=1e-6
        )
        for _ in range(30):
            opt.step()
            sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(1e-6, rel=0.01)


class TestModelEMA:
    def test_average_parameters_swaps_and_restores(self) -> None:
        model = nn.Linear(2, 2)
        ema = ModelEMA(model, decay=0.5)
        # Two steps with a weight change in between: the unbiased EMA now lags the live
        # weights (after a single step it equals them exactly).
        ema.step()
        with torch.no_grad():
            model.weight += 1.0
        ema.step()
        live = model.weight.detach().clone()
        with ema.average_parameters():
            swapped = model.weight.detach().clone()
        assert torch.allclose(model.weight, live)  # restored after context
        assert not torch.allclose(swapped, live)  # EMA weights differ from live

    def test_state_roundtrip(self) -> None:
        model = nn.Linear(2, 2)
        ema = ModelEMA(model, decay=0.9)
        ema.step()
        state = ema.state_dict()
        ema2 = ModelEMA(nn.Linear(2, 2), decay=0.9)
        ema2.load_state_dict(state)


class TestDistributedEMA:
    def test_converges_to_constant(self) -> None:
        ema = DistributedEMA(decay=0.5)
        for _ in range(30):
            ema.update(torch.tensor([3.0]))
        assert ema.value == pytest.approx(3.0, rel=0.01)


class TestDistributedMetric:
    def test_mean_and_reset(self) -> None:
        metric = DistributedMetric()
        metric.update(torch.tensor([1.0, 2.0, 3.0]))
        metric.update(torch.tensor([4.0]))
        assert metric.compute_and_reset().item() == pytest.approx(2.5)
        metric.update(torch.tensor([10.0]))
        assert metric.compute_and_reset().item() == pytest.approx(10.0)


class TestPeriodicEvent:
    def test_integer_interval(self) -> None:
        assert periodic_event(0, 5)
        assert not periodic_event(3, 5)
        assert periodic_event(5, 5)
        assert not periodic_event(0, 5, include_0=False)

    def test_percent_interval(self) -> None:
        assert periodic_event(10, "10%", total_steps=100)
        assert not periodic_event(11, "10%", total_steps=100)
