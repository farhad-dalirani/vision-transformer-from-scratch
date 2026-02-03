import math

import pytest
import torch
from torch.optim import AdamW

from vision_transformer.training.lr_scheduler import (
    cosine_lr_scheduler_with_warmup_fn,
    get_lr_scheduler,
    linear_lr_scheduler_with_warmup_fn,
)


def _dummy_optimizer(base_lr: float = 1e-3):
    # Any parameter tensor works; keep it simple.
    p = torch.nn.Parameter(torch.tensor(1.0))
    return AdamW([p], lr=base_lr)


def test_get_lr_scheduler_raises_for_unknown_name():
    optimizer = _dummy_optimizer()
    with pytest.raises(ValueError):
        get_lr_scheduler(
            optimizer=optimizer,
            scheduler_name="unknown",
            warmup_steps=1,
            total_steps=5,
        )


def test_cosine_warmup_linear_ramp_values():
    total_steps = 10
    warmup_steps = 3
    fn = cosine_lr_scheduler_with_warmup_fn(
        total_steps=total_steps, warmup_steps=warmup_steps
    )

    # Warmup: step 0 -> 0, step 1 -> 1/3, step 2 -> 2/3
    assert fn(0) == pytest.approx(0.0)
    assert fn(1) == pytest.approx(1 / 3)
    assert fn(2) == pytest.approx(2 / 3)

    # At step == warmup_steps, cosine starts; progress=0 -> factor=1
    assert fn(3) == pytest.approx(1.0)


def test_cosine_endpoints_and_monotonic_decay_post_warmup():
    total_steps = 20
    warmup_steps = 4
    fn = cosine_lr_scheduler_with_warmup_fn(
        total_steps=total_steps, warmup_steps=warmup_steps
    )

    # Start of cosine phase -> 1.0
    assert fn(warmup_steps) == pytest.approx(1.0)

    # End of training (step == total_steps) -> progress=1 -> cos(pi)=-1 -> factor=0
    assert fn(total_steps) == pytest.approx(0.0)

    # Sample a few points after warmup; should be non-increasing
    vals = [float(fn(s)) for s in range(warmup_steps, total_steps + 1)]
    for a, b in zip(vals, vals[1:]):
        assert b <= a + 1e-12


def test_cosine_matches_closed_form_example_midpoint():
    total_steps = 100
    warmup_steps = 10
    fn = cosine_lr_scheduler_with_warmup_fn(
        total_steps=total_steps, warmup_steps=warmup_steps
    )

    # Pick a step halfway through cosine phase
    step = warmup_steps + (total_steps - warmup_steps) // 2
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    expected = 0.5 * (1.0 + math.cos(math.pi * progress))
    assert float(fn(step)) == pytest.approx(expected, rel=1e-6, abs=1e-8)


def test_linear_warmup_and_linear_decay_values():
    total_steps = 10
    warmup_steps = 2
    fn = linear_lr_scheduler_with_warmup_fn(
        total_steps=total_steps, warmup_steps=warmup_steps
    )

    # Warmup
    assert fn(0) == pytest.approx(0.0)
    assert fn(1) == pytest.approx(0.5)
    # step==warmup -> progress=0 -> 1.0
    assert fn(2) == pytest.approx(1.0)

    # End -> 0.0
    assert fn(total_steps) == pytest.approx(0.0)


def test_linear_is_non_increasing_after_warmup():
    total_steps = 50
    warmup_steps = 5
    fn = linear_lr_scheduler_with_warmup_fn(
        total_steps=total_steps, warmup_steps=warmup_steps
    )

    vals = [float(fn(s)) for s in range(warmup_steps, total_steps + 1)]
    for a, b in zip(vals, vals[1:]):
        assert b <= a + 1e-12


def test_warmup_zero_behaviour_cosine():
    total_steps = 10
    warmup_steps = 0
    fn = cosine_lr_scheduler_with_warmup_fn(
        total_steps=total_steps, warmup_steps=warmup_steps
    )

    # With warmup=0, step 0 is cosine start (progress 0 -> 1.0)
    assert fn(0) == pytest.approx(1.0)
    assert fn(total_steps) == pytest.approx(0.0)


def test_warmup_zero_behaviour_linear():
    total_steps = 10
    warmup_steps = 0
    fn = linear_lr_scheduler_with_warmup_fn(
        total_steps=total_steps, warmup_steps=warmup_steps
    )

    # With warmup=0, linear decay starts immediately at 1.0
    assert fn(0) == pytest.approx(1.0)
    assert fn(total_steps) == pytest.approx(0.0)


def test_get_lr_scheduler_returns_lambda_lr_and_steps_change_lr_cosine():
    optimizer = _dummy_optimizer(base_lr=1e-3)
    total_steps = 10
    warmup_steps = 2

    scheduler = get_lr_scheduler(
        optimizer=optimizer,
        scheduler_name="cosine",
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )
    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)

    lr_initial = optimizer.param_groups[0]["lr"]
    assert lr_initial == pytest.approx(1e-3 * 0.0)

    optimizer.step()
    scheduler.step()
    lr_after = optimizer.param_groups[0]["lr"]
    assert lr_after == pytest.approx(1e-3 * (1 / warmup_steps))


def test_get_lr_scheduler_returns_lambda_lr_and_steps_change_lr_linear():
    optimizer = _dummy_optimizer(base_lr=2e-3)
    total_steps = 10
    warmup_steps = 2

    scheduler = get_lr_scheduler(
        optimizer=optimizer,
        scheduler_name="linear",
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )
    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)

    optimizer.step()
    scheduler.step()
    lr0 = optimizer.param_groups[0]["lr"]
    assert lr0 == pytest.approx(2e-3 * (1 / warmup_steps))

    optimizer.step()
    scheduler.step()
    lr1 = optimizer.param_groups[0]["lr"]
    assert lr1 == pytest.approx(2e-3 * (2 / warmup_steps))


def test_get_lr_scheduler_accepts_mixed_case_names():
    optimizer = _dummy_optimizer()
    scheduler = get_lr_scheduler(
        optimizer=optimizer,
        scheduler_name="CoSiNe",
        warmup_steps=1,
        total_steps=5,
    )
    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)
