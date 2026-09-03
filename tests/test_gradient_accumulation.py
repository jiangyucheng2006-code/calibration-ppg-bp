import copy

import pytest


torch = pytest.importorskip("torch")

from pulsedb_fewshot.train import _finish_optimizer_step  # noqa: E402


def _one_step(
    initial: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    microbatch_size: int,
) -> torch.nn.Module:
    model = copy.deepcopy(initial)
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, weight_decay=1e-4)
    optimizer.zero_grad(set_to_none=True)
    accumulated_examples = 0
    for start in range(0, len(inputs), microbatch_size):
        batch_inputs = inputs[start : start + microbatch_size]
        batch_targets = targets[start : start + microbatch_size]
        loss = torch.nn.functional.mse_loss(model(batch_inputs), batch_targets)
        (loss * len(batch_inputs)).backward()
        accumulated_examples += len(batch_inputs)
    _finish_optimizer_step(
        parameters,
        optimizer,
        accumulated_examples=accumulated_examples,
    )
    return model


@pytest.mark.parametrize("microbatch_size", [2, 4])
def test_accumulated_step_matches_one_physical_batch(microbatch_size: int) -> None:
    torch.manual_seed(7)
    initial = torch.nn.Linear(3, 2)
    inputs = torch.randn(8, 3)
    targets = torch.randn(8, 2)
    full = _one_step(initial, inputs, targets, microbatch_size=8)
    accumulated = _one_step(
        initial, inputs, targets, microbatch_size=microbatch_size
    )
    for full_parameter, accumulated_parameter in zip(
        full.parameters(), accumulated.parameters(), strict=True
    ):
        assert torch.allclose(
            full_parameter, accumulated_parameter, atol=1e-7, rtol=1e-6
        )


def test_short_final_accumulation_window_uses_actual_example_count() -> None:
    torch.manual_seed(11)
    initial = torch.nn.Linear(4, 1)
    inputs = torch.randn(6, 4)
    targets = torch.randn(6, 1)
    full = _one_step(initial, inputs, targets, microbatch_size=6)
    short_final_window = _one_step(initial, inputs, targets, microbatch_size=4)
    for full_parameter, accumulated_parameter in zip(
        full.parameters(), short_final_window.parameters(), strict=True
    ):
        assert torch.allclose(
            full_parameter, accumulated_parameter, atol=1e-7, rtol=1e-6
        )
