"""Real-encoder unit tests; set PPG_TEST_DEVICE=cuda inside a GPU allocation."""
import os
import copy
import pytest
import torch
from pulsedb_fewshot.personal_feature_models import FEATURE_MODELS, PersonalFeatureRegressor


@pytest.mark.parametrize("candidate", list(FEATURE_MODELS))
def test_personal_state_forward_backward_and_roundtrip(candidate):
    device = torch.device(os.environ.get("PPG_TEST_DEVICE", "cpu"))
    torch.manual_seed(10)
    spec = FEATURE_MODELS[candidate]
    m = PersonalFeatureRegressor(spec, subject_count=3).to(device)
    params = m.personal_parameters()
    assert sum(p.numel() for p in params) == 3 * spec.participant_trainable_parameters
    x = torch.randn(3, 1, 1250, device=device)
    anchor = torch.randn(3, 2, device=device)
    ids = torch.arange(3, device=device)
    m.eval()
    assert torch.equal(m(x, anchor, subject_index=ids), anchor)
    optimizer = torch.optim.AdamW(m.parameters(), lr=0.003)
    # Exercise more than one step: zero-initialized output heads block early encoder gradients.
    for _ in range(5):
        m.train()
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = m(x, anchor, subject_index=ids)
            loss = (output - anchor - 1).square().mean()
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        optimizer.step()
    if params:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in params)
    m.eval()
    with torch.no_grad():
        before = m(x, anchor, subject_index=ids)
        restored = copy.deepcopy(m)
        restored.load_state_dict(m.state_dict(), strict=True)
        assert torch.equal(before, restored(x, anchor, subject_index=ids))
        if params:
            for p in params:
                p[0].add_(0.2)
            after = m(x, anchor, subject_index=ids)
            assert torch.equal(before[1:], after[1:])
            assert not torch.allclose(before[0], after[0])
        else:
            assert torch.equal(before, m(x, anchor, subject_index=ids.flip(0)))


def test_unknown_subject_is_not_silently_reassigned():
    m = PersonalFeatureRegressor(FEATURE_MODELS["shared_bilinear32"], subject_count=2)
    with pytest.raises(ValueError, match="registered"):
        m(torch.randn(2, 1, 1250), torch.zeros(2, 2))
