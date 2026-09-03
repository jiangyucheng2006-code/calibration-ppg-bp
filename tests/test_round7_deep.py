import torch

from pulsedb_fewshot.round7_deep import EmbeddingAutoencoder, EmbeddingCausalGRU, PhenotypeRouter


def test_round7_model_shapes() -> None:
    x=torch.randn(5,256); reconstruction,latent=EmbeddingAutoencoder(256)(x)
    assert reconstruction.shape==(5,256) and latent.shape==(5,32)
    assert PhenotypeRouter(32,8)(latent).shape==(5,8)
    assert EmbeddingCausalGRU(54)(torch.randn(3,7,54)).shape==(3,7,2)
