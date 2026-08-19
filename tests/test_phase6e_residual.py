import numpy as np
import pandas as pd
import pytest
import torch

from pulsedb_fewshot.phase6e_residual import (
    CausalResidualGRU,
    ClusterResidualExperts,
    ResidualMLP,
    SupervisedMoE,
    _cluster_stability,
    _source_mixing_entropy,
)


def test_residual_models_preserve_batch_and_bp_dimensions() -> None:
    x=torch.randn(7,22)
    assert ResidualMLP(22)(x).shape==(7,2)
    assert ResidualMLP(22,gated=True)(x).shape==(7,2)
    residual,weights=SupervisedMoE(22,experts=4)(x)
    assert residual.shape==(7,2)
    assert weights.shape==(7,4)
    assert torch.allclose(weights.sum(1),torch.ones(7))
    membership=torch.softmax(torch.randn(7,8),1)
    assert ClusterResidualExperts(22,8)(x,membership).shape==(7,2)


def test_causal_gru_prefix_is_invariant_to_future_features() -> None:
    model=CausalResidualGRU(22).eval()
    prefix=torch.randn(1,4,22)
    first=model(torch.cat([prefix,torch.randn(1,3,22)],dim=1))[:,:4]
    second=model(torch.cat([prefix,torch.randn(1,5,22)],dim=1))[:,:4]
    assert torch.allclose(first,second,atol=1e-6,rtol=0)


def test_cluster_stability_is_deterministic_and_has_nonempty_clusters() -> None:
    rng=np.random.default_rng(17)
    x=np.concatenate([rng.normal(-3,0.1,(50,3)),rng.normal(3,0.1,(50,3))]).astype(np.float32)
    c1,a1=_cluster_stability(x,2,23)
    c2,a2=_cluster_stability(x,2,23)
    assert np.allclose(c1,c2)
    assert a1==a2
    assert a1["stability"]>0.95
    assert a1["minimum_cluster_fraction"]>0.4
    sources=np.asarray(["MIMIC","VitalDB"]*50)
    assert 0.0<=_source_mixing_entropy(np.asarray([0,1]*50),sources,2)<=1.0
