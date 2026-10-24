import torch

import explicit_priors as ep


def _quadratic(v, device):
    """Phi(x) = ||x||^2 / (2 v): Gauss basis, linear term only."""
    m = ep.FoE(torch.ones(1, 1, 1), ep.Gauss())
    m.coef[0, -1] = 1.0 / v
    return m.to(device)


def test_map_estimate(device):
    v, s = 0.05, 0.1
    y = torch.rand(2, 24, 24, device=device)
    x = ep.map_estimate(_quadratic(v, device), y, sigma=s, iters=200)
    assert torch.allclose(x, y / (1 + s**2 / v), atol=1e-4)


def test_one_step(device):
    v, s = 0.05, 0.1
    y = torch.rand(24, 24, device=device)
    assert torch.allclose(ep.one_step(_quadratic(v, device), y, sigma=s), y * (1 - s**2 / v), atol=1e-5)


def test_fista_least_squares(device):
    g = torch.Generator().manual_seed(0)
    A = (torch.eye(20, dtype=torch.float64) + 0.1 * torch.randn(3, 20, 20, generator=g, dtype=torch.float64)).to(device)
    b = torch.randn(3, 20, generator=g, dtype=torch.float64).to(device)

    def fun(x):
        r = torch.einsum("bij,bj->bi", A, x) - b
        return 0.5 * (r * r).sum(1), torch.einsum("bij,bi->bj", A, r)

    x = ep.fista(fun, torch.zeros(3, 20, dtype=torch.float64, device=device), iters=500)
    assert torch.allclose(x, torch.linalg.solve(A, b), atol=1e-6)
