import pytest
import torch

import explicit_priors as ep


@pytest.mark.parametrize(
    "bank",
    [
        ep.filters.dct(5),
        ep.filters.gabor(),
        ep.filters.gaussian_derivatives(),
        ep.filters.laplacian_of_gaussian(),
        ep.filters.dilate(ep.filters.dct(3), 2),
    ],
)
def test_zero_mean_unit_norm(bank):
    assert torch.allclose(bank.mean((1, 2)), torch.zeros(len(bank)), atol=1e-6)
    assert torch.allclose(bank.flatten(1).norm(dim=1), torch.ones(len(bank)), atol=1e-5)
    assert bank.shape[-1] % 2 == 1


def test_sizes():
    assert ep.filters.dct(7).shape == (48, 7, 7)
    f, g = ep.filters.symmetrize(ep.filters.dct(3))
    assert f.shape == (64, 3, 3) and int(g.max()) == 7
    assert ep.filters.stack(ep.filters.dct(3), ep.filters.dct(5)).shape == (32, 5, 5)


def test_data_banks(natural):
    p, i = ep.filters.pca(natural, 5), ep.filters.ica(natural, 5)
    assert p.shape == i.shape == (24, 5, 5)
    G = p.flatten(1) @ p.flatten(1).T
    assert torch.allclose(G, torch.eye(24, device=G.device), atol=1e-4)  # orthonormal
