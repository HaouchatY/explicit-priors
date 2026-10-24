import pytest
import torch

import explicit_priors as ep

V, S = 0.04, 0.1  # pixel variance, noise std


@pytest.fixture
def gaussian(device):
    """i.i.d. Gaussian pixels: every exact potential is quadratic."""
    g = torch.Generator().manual_seed(0)
    return (V**0.5 * torch.randn(128, 64, 64, generator=g)).to(device)


def _slope(prior):
    """psi(z) / z over the bulk of the data, |z| in [0.25, 2] std."""
    z = V**0.5 * torch.linspace(0.25, 2.0, 50, device=prior.scale.device).view(1, 1, -1)
    return (prior.psi(z) / z).flatten()


@pytest.mark.parametrize(
    "objective, slope", [("score", 1 / V), ("mse", 1 / (V + S**2)), ("map", 1 / V)]
)  # MAP = MMSE for Gaussians: the MAP potential is the clean prior
def test_recovers_quadratic_potentials(gaussian, objective, slope):
    basis = ep.Hat(n=8, eps=0.05)  # few hats: score matching is noisy where cells are tiny
    prior = ep.fit(torch.ones(1, 1, 1), gaussian, sigma=S, objective=objective, basis=basis, shape="free")
    assert torch.allclose(_slope(prior), torch.full((1,), slope, device=gaussian.device), rtol=0.05)


def test_shapes(natural):
    f = ep.filters.dct(3)
    for shape in ("monotone", "convex"):
        prior = ep.fit(f, natural, sigma=0.1, shape=shape)
        z = torch.linspace(1e-3, 1.0, 500, device=natural.device).view(1, 1, -1).expand(1, 8, -1) * prior.scale.view(
            1, -1, 1
        )
        psi = prior.psi(z)
        assert (psi >= -1e-6).all()
        if shape == "convex":
            assert (torch.diff(psi, dim=-1) >= -1e-5).all()


def test_mse_beats_identity(natural):
    prior = ep.fit(ep.filters.dct(3), natural[:24], sigma=0.1)
    x = natural[24:]
    y = x + 0.1 * torch.randn_like(x)
    assert float(((ep.denoise(prior, y) - x) ** 2).mean()) < 0.5 * float(((y - x) ** 2).mean())
