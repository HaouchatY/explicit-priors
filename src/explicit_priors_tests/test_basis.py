import pytest
import torch

import explicit_priors as ep


@pytest.mark.parametrize("basis", [ep.Hat(), ep.Gauss()])
def test_psi_is_linear_in_coef(basis):
    u = torch.linspace(-1.5, 1.5, 301, dtype=torch.float64).view(1, 1, -1).expand(1, 3, -1)
    c = torch.randn(3, basis.size, dtype=torch.float64)
    ref = (basis.features(u) * c[None, :, None]).sum(-1)
    assert torch.allclose(basis.psi(u, c), ref, atol=1e-10)
    assert torch.allclose(basis.psi(-u, c), -basis.psi(u, c), atol=1e-10)


@pytest.mark.parametrize("basis", [ep.Hat(), ep.Gauss()])
def test_phi_integrates_psi(basis):
    u = torch.linspace(0, 1.5, 300001, dtype=torch.float64).view(1, 1, -1)
    c = torch.randn(1, basis.size, dtype=torch.float64)
    phi, psi = basis.phi(u, c)[0, 0], basis.psi(u, c)[0, 0]
    integral = torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumulative_trapezoid(psi, u[0, 0])])
    assert torch.allclose(phi, integral, atol=1e-4 * float(phi.abs().max()))
    assert torch.allclose(basis.phi(-u, c), basis.phi(u, c))


@pytest.mark.parametrize("basis", [ep.Hat(), ep.Gauss()])
def test_derivatives(basis):
    u = (3 * torch.rand(2000, dtype=torch.float64) - 1.5).requires_grad_()
    b = basis.features(u)
    for a in range(basis.size):
        (g,) = torch.autograd.grad(b[:, a].sum(), u, retain_graph=True)
        assert torch.allclose(basis.derivatives(u.detach())[:, a], g, atol=1e-8 * float(g.abs().max()) + 1e-12)


def test_hat_constraints():
    h = ep.Hat()
    u = torch.linspace(1e-4, 1.5, 2000, dtype=torch.float64).view(1, 1, -1)
    for shape in ("monotone", "convex"):
        c = (h.constraint(shape) @ torch.rand(h.size, dtype=torch.float64))[None]
        psi = h.psi(u, c)[0, 0]
        assert (psi >= 0).all()
        if shape == "convex":
            assert (torch.diff(psi) >= -1e-12).all()
