import pytest


torch = pytest.importorskip("torch")

from pulsedb_fewshot.models import (  # noqa: E402
    PopulationRegressor,
    SiameseDeltaRegressor,
    VariableKPersonalizer,
    configure_personal_adaptation,
)


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


def test_adaptation_modes_expose_only_intended_parameters() -> None:
    population = PopulationRegressor()
    head = configure_personal_adaptation(population, "head")
    assert any(parameter.requires_grad for parameter in head.head.parameters())
    assert not any(parameter.requires_grad for parameter in head.encoder.parameters())
    lora = configure_personal_adaptation(population, "lora", lora_rank=2)
    trainable = [name for name, parameter in lora.named_parameters() if parameter.requires_grad]
    assert trainable
    assert all(".a." in name or ".b." in name for name in trainable)
