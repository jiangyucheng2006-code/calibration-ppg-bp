from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from pulsedb_fewshot.models import (  # noqa: E402
    BACKBONE_NAMES,
    MultiScaleResNetEncoder,
    PopulationRegressor,
    SiameseDeltaRegressor,
    SupportConditionedAdapterBank,
    VariableKPersonalizer,
    build_ppg_encoder,
    configure_personal_adaptation,
    model_parameter_counts,
)
from pulsedb_fewshot.train import _load_population_checkpoint  # noqa: E402


ROUND13_PARAMETER_COUNTS = {
    "resnet_small": 665_490,
    "resnet_depth2": 1_333_010,
    "resnet_wide1p5": 1_456_282,
    "inception_time": 512_162,
    "inception_time_wide": 1_123_954,
    "patch_transformer": 710_530,
    "patch_transformer_deep": 1_372_034,
    "patch_transformer_wide": 2_730_498,
    "patch_transformer_highres": 715_522,
    "patch_transformer_longpatch": 716_034,
    "conformer": 1_587_330,
    "conformer_large": 5_230_018,
    "convnext_1d": 5_477_826,
}


def test_all_models_have_expected_shapes_and_gradients() -> None:
    query = torch.randn(2, 1, 1250)
    support = torch.randn(2, 5, 1, 1250)
    support_bp = torch.randn(2, 5, 2)
    mask = torch.tensor([[1, 0, 0, 0, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)

    population = PopulationRegressor()
    assert population(query).shape == (2, 2)
    siamese = SiameseDeltaRegressor()
    assert siamese(query, support[:, 0], support_bp[:, 0]).shape == (2, 2)

    for film, attention in ((False, False), (True, False), (True, True)):
        model = VariableKPersonalizer(use_film=film, query_conditioned_weights=attention)
        output = model(query, support, support_bp, mask)
        assert output.shape == (2, 2)
        output.square().mean().backward()


def test_zero_initialized_main_model_starts_at_residual_offset() -> None:
    model = VariableKPersonalizer(use_film=True, query_conditioned_weights=False)
    model.eval()
    query = torch.randn(2, 1, 1250)
    support = torch.randn(2, 5, 1, 1250)
    support_bp = torch.randn(2, 5, 2)
    mask = torch.tensor([[1, 0, 0, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)
    with torch.no_grad():
        actual = model(query, support, support_bp, mask)
        pop_query = model.population(query)
        batch, maximum_k, channels, length = support.shape
        pop_support = model.population(
            support.reshape(batch * maximum_k, channels, length)
        ).reshape(batch, maximum_k, 2)
        weights = mask.float() / mask.sum(dim=1, keepdim=True)
        expected = pop_query + torch.sum(
            weights[..., None] * (support_bp - pop_support), dim=1
        )
    assert torch.allclose(actual, expected, atol=1e-5)


@pytest.mark.parametrize("basis_count", [5, 10, 15, 20, 25, 30])
def test_support_conditioned_adapter_bank_routing_contract(
    basis_count: int,
) -> None:
    context = torch.randn(4, 32)
    mask = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    dense = SupportConditionedAdapterBank(32, basis_count=basis_count, rank=4)
    dense_weights = dense.routing_weights(context, mask)
    assert dense_weights.shape == (4, basis_count)
    assert torch.all(dense_weights > 0)
    assert torch.allclose(dense_weights.sum(dim=1), torch.ones(4))

    sparse = SupportConditionedAdapterBank(
        32, basis_count=basis_count, rank=4, top_k=5
    )
    sparse_weights = sparse.routing_weights(context, mask)
    assert torch.equal(
        (sparse_weights > 0).sum(dim=1), torch.full((4,), min(5, basis_count))
    )
    assert torch.allclose(sparse_weights.sum(dim=1), torch.ones(4))


def test_five_basis_top5_and_dense_routing_are_identical() -> None:
    dense = SupportConditionedAdapterBank(16, basis_count=5, rank=2)
    top5 = SupportConditionedAdapterBank(16, basis_count=5, rank=2, top_k=5)
    top5.load_state_dict(dense.state_dict())
    context = torch.randn(7, 16)
    mask = torch.tensor([[1, 1, 1, 0, 0]] * 7, dtype=torch.bool)
    assert torch.equal(
        dense.routing_weights(context, mask), top5.routing_weights(context, mask)
    )


def test_adapter_bank_initialization_preserves_m0_and_is_support_order_invariant() -> None:
    model = VariableKPersonalizer(
        use_film=False,
        query_conditioned_weights=False,
        adapter_basis_count=20,
        adapter_rank=4,
        adapter_top_k=5,
    )
    reference = VariableKPersonalizer(
        use_film=False,
        query_conditioned_weights=False,
    )
    # Copy every shared M0 parameter; the adapter output factors remain zero.
    reference.load_state_dict(
        {
            key: value
            for key, value in model.state_dict().items()
            if not key.startswith("adapter_bank.")
        }
    )
    model.eval()
    reference.eval()
    query = torch.randn(3, 1, 1250)
    support = torch.randn(3, 5, 1, 1250)
    support_bp = torch.randn(3, 5, 2)
    mask = torch.ones(3, 5, dtype=torch.bool)
    permutation = torch.tensor([2, 4, 0, 3, 1])
    with torch.no_grad():
        expected = reference(query, support, support_bp, mask)
        actual = model(query, support, support_bp, mask)
        permuted = model(
            query,
            support[:, permutation],
            support_bp[:, permutation],
            mask[:, permutation],
        )
    assert torch.allclose(actual, expected, atol=1e-6)
    assert torch.allclose(actual, permuted, atol=1e-6)


def test_adapter_bank_receives_gradient_without_subject_identity() -> None:
    bank = SupportConditionedAdapterBank(24, basis_count=10, rank=4, top_k=5)
    query = torch.randn(8, 24)
    context = torch.randn(8, 24)
    mask = torch.ones(8, 5, dtype=torch.bool)
    bank(query, context, mask).square().mean().backward()
    assert bank.output_factors.grad is not None
    assert torch.isfinite(bank.output_factors.grad).all()
    assert bank.output_factors.grad.abs().sum() > 0


def test_adaptation_modes_expose_only_intended_parameters() -> None:
    population = PopulationRegressor()
    head = configure_personal_adaptation(population, "head")
    assert any(parameter.requires_grad for parameter in head.head.parameters())
    assert not any(parameter.requires_grad for parameter in head.encoder.parameters())
    lora = configure_personal_adaptation(population, "lora", lora_rank=2)
    trainable = [name for name, parameter in lora.named_parameters() if parameter.requires_grad]
    assert trainable
    assert all(".a." in name or ".b." in name for name in trainable)


def test_masked_median_uses_midpoint_for_even_k_and_ignores_padding() -> None:
    values = torch.tensor(
        [
            [[1.0, 10.0], [9.0, 30.0], [100.0, 100.0], [200.0, 200.0]],
            [[1.0, 5.0], [9.0, 1.0], [3.0, 8.0], [500.0, 500.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
    actual = VariableKPersonalizer._masked_median(values, mask)
    expected = torch.tensor([[5.0, 20.0], [3.0, 5.0]])
    assert torch.equal(actual, expected)


def test_quality_gate_and_demographic_conditioning_accept_expected_inputs() -> None:
    query = torch.randn(2, 1, 1250)
    support = torch.randn(2, 5, 1, 1250)
    support_bp = torch.randn(2, 5, 2)
    mask = torch.tensor([[1, 0, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    demographics = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 1.0], [1.2, 1.0, 1.0, 0.0, 0.0]]
    )
    gate = VariableKPersonalizer(
        use_film=False, query_conditioned_weights=False, use_quality_gate=True
    )
    assert gate(query, support, support_bp, mask).shape == (2, 2)
    demographic = VariableKPersonalizer(
        use_film=False, query_conditioned_weights=False, use_demographics=True
    )
    assert demographic(query, support, support_bp, mask, demographics).shape == (2, 2)
    with pytest.raises(ValueError, match="demographic"):
        demographic(query, support, support_bp, mask)


def test_direct_demographics_are_concatenated_without_expansion() -> None:
    query = torch.randn(2, 1, 1250)
    support = torch.randn(2, 5, 1, 1250)
    support_bp = torch.randn(2, 5, 2)
    mask = torch.ones(2, 5, dtype=torch.bool)
    demographics = torch.zeros(2, 5)
    model = VariableKPersonalizer(
        use_film=False,
        use_demographics=True,
        demographic_mode="direct",
    )
    assert model(query, support, support_bp, mask, demographics).shape == (2, 2)
    assert not hasattr(model, "demographic_encoder")


def test_population_supports_128_dimensional_encoder() -> None:
    model = PopulationRegressor(MultiScaleResNetEncoder(feature_dim=128))
    assert model(torch.randn(2, 1, 1250)).shape == (2, 2)
    assert model.encoder.feature_dim == 128


@pytest.mark.parametrize("backbone", BACKBONE_NAMES)
def test_all_round11_backbones_share_the_same_contract(backbone: str) -> None:
    encoder = build_ppg_encoder(backbone, feature_dim=256)
    output = encoder(torch.randn(2, 1, 1250))
    assert output.shape == (2, 256)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    counts = model_parameter_counts(encoder)
    assert counts["total"] > 0
    assert counts["trainable"] == counts["total"]
    assert encoder.backbone_name == backbone


def test_round11_backbone_factory_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown backbone"):
        build_ppg_encoder("not_a_backbone")


def test_round13_capacity_variants_have_ordered_parameter_counts() -> None:
    families = (
        ("inception_time", "inception_time_wide"),
        ("conformer", "conformer_large"),
    )
    for family in families:
        counts = [
            model_parameter_counts(build_ppg_encoder(name))["total"]
            for name in family
        ]
        assert counts == sorted(counts)
        assert len(set(counts)) == len(counts)

    reference_resnet = model_parameter_counts(build_ppg_encoder("resnet_small"))["total"]
    assert model_parameter_counts(build_ppg_encoder("resnet_depth2"))["total"] > reference_resnet
    assert model_parameter_counts(build_ppg_encoder("resnet_wide1p5"))["total"] > reference_resnet

    reference_transformer = model_parameter_counts(
        build_ppg_encoder("patch_transformer")
    )["total"]
    assert (
        model_parameter_counts(build_ppg_encoder("patch_transformer_deep"))["total"]
        > reference_transformer
    )
    assert (
        model_parameter_counts(build_ppg_encoder("patch_transformer_wide"))["total"]
        > reference_transformer
    )


def test_high_resolution_transformer_uses_more_tokens_at_equal_capacity() -> None:
    inputs = torch.randn(1, 1, 1250)
    medium = build_ppg_encoder("patch_transformer")
    high_resolution = build_ppg_encoder("patch_transformer_highres")
    assert model_parameter_counts(medium)["total"] != model_parameter_counts(
        high_resolution
    )["total"]
    assert high_resolution.patch(inputs).shape[-1] > medium.patch(inputs).shape[-1]


def test_long_patch_transformer_uses_fewer_tokens_than_reference() -> None:
    inputs = torch.randn(1, 1, 1250)
    reference = build_ppg_encoder("patch_transformer")
    long_patch = build_ppg_encoder("patch_transformer_longpatch")
    assert long_patch.patch(inputs).shape[-1] < reference.patch(inputs).shape[-1]


@pytest.mark.parametrize(
    ("backbone", "expected"), ROUND13_PARAMETER_COUNTS.items()
)
def test_round13_population_parameter_count(backbone: str, expected: int) -> None:
    population = PopulationRegressor(build_ppg_encoder(backbone))
    assert model_parameter_counts(population)["total"] == expected


@pytest.mark.parametrize("backbone", ROUND13_PARAMETER_COUNTS)
def test_round13_population_checkpoint_round_trip(
    backbone: str, tmp_path: Path
) -> None:
    torch.manual_seed(13)
    model = PopulationRegressor(build_ppg_encoder(backbone)).eval()
    checkpoint = tmp_path / f"{backbone}.pt"
    scaler = {"mean": [120.0, 70.0], "std": [20.0, 10.0]}
    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_dim": 256,
            "backbone": backbone,
            "target_scaler": scaler,
        },
        checkpoint,
    )
    loaded, loaded_scaler = _load_population_checkpoint(
        checkpoint, torch.device("cpu")
    )
    loaded.eval()
    assert loaded_scaler == scaler
    assert loaded.encoder.backbone_name == backbone
    inputs = torch.randn(1, 1, 1250)
    with torch.no_grad():
        assert torch.equal(model(inputs), loaded(inputs))
