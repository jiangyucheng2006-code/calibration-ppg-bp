"""Warm-start equivalence, personal-state isolation and frozen-base protection."""
import copy
import os
from types import SimpleNamespace
import pytest
import torch
from pulsedb_fewshot.lora_prs_models import PRS_MODELS, LoraPRSRegressor


@pytest.mark.parametrize('candidate', list(PRS_MODELS))
def test_continuation(candidate):
    device = torch.device(os.environ.get('PPG_TEST_DEVICE', 'cpu'))
    torch.manual_seed(43)
    spec = PRS_MODELS[candidate]
    m = LoraPRSRegressor(spec, subject_count=3).to(device)
    # Simulate a nonzero pretrained model, not the trivial initial BP anchor.
    with torch.no_grad():
        m.base.residual_head[-1].weight.normal_(std=.03)
        m.base.lora_b.weight.normal_(std=.02)
    x = torch.randn(3, 1, 1250, device=device)
    anchor = torch.randn(3, 2, device=device)
    ids = torch.arange(3, device=device)
    m.eval()
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        expected = m.base(x, anchor, subject_index=ids)
        assert torch.equal(expected, m(x, anchor, subject_index=ids))
    frozen_state = copy.deepcopy(m.base.state_dict())
    optimizer = m.optimizer(SimpleNamespace(base_learning_rate=1e-5, learning_rate=3e-4, weight_decay=1e-4))
    assert optimizer.param_groups[-1]['lr'] == (1e-5 if candidate == 'lora_continue' else 3e-4)
    personal = [p for n, p in m.named_parameters() if 'lora_a.' in n or 'lora_b.' in n or n.startswith('subject_')]
    assert sum(p.numel() for p in personal) == 3 * spec.stored_personal_parameters
    assert sum(p.numel() for p in personal if p.requires_grad) == 3 * spec.participant_trainable_parameters
    for _ in range(3):
        m.train()
        if spec.freeze_base:
            assert not any(child.training for child in m.base.modules())
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            pred = m(x, anchor, subject_index=ids)
            loss = (pred - anchor - 1).float().square().mean() + m.penalty()
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        optimizer.step()
    if spec.freeze_base:
        assert all(torch.equal(v, m.base.state_dict()[k]) for k, v in frozen_state.items())
    m.eval()
    with torch.no_grad():
        before = m(x, anchor, subject_index=ids)
        restored = copy.deepcopy(m)
        restored.load_state_dict(m.state_dict(), strict=True)
        assert torch.equal(before, restored(x, anchor, subject_index=ids))
        if candidate != 'lora_continue':
            m.subject_bias.weight[0].add_(.25)
            after = m(x, anchor, subject_index=ids)
            assert torch.equal(before[1:], after[1:])
            assert not torch.equal(before[0], after[0])


def test_requires_personal_identity():
    m = LoraPRSRegressor(PRS_MODELS['lora_prs_bias'], subject_count=2)
    with pytest.raises(ValueError, match='registered'):
        m(torch.randn(2, 1, 1250), torch.zeros(2, 2))
