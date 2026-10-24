import torch

import explicit_priors as ep
from explicit_priors.fit import divergence, features


def _prior(device, groups=None, K=4):
    g = torch.Generator().manual_seed(1)
    m = ep.FoE(ep.filters.dct(3)[:K], groups=groups)
    m.scale.uniform_(0.2, 0.5, generator=g)
    m.coef.normal_(generator=g)
    return m.to(device)


def test_grad_is_gradient_of_energy(device):
    m = _prior(device)
    x = torch.rand(2, 20, 20, device=device, requires_grad=True)
    m.energy(x).sum().backward()
    assert torch.allclose(m.grad(x.detach()), x.grad, atol=1e-4)
    e, g = m.energy_grad(x.detach())
    assert torch.allclose(e, m.energy(x.detach())) and torch.allclose(g, x.grad, atol=1e-4)


def test_features_are_basis_gradients(device):
    m = _prior(device)
    x = torch.rand(2, 20, 20, device=device)
    G = features(m, x)
    assert torch.allclose(torch.einsum("bdhw,d->bhw", G, m.coef.reshape(-1)), m.grad(x), atol=1e-4)


def test_divergence_is_laplacian(device):
    m = _prior(device, K=2)
    x = torch.rand(1, 7, 7, device=device)
    lap = 0.0
    for i in range(7):
        for j in range(7):
            xi = x.clone().requires_grad_()
            (g,) = torch.autograd.grad(m.grad(xi)[0, i, j], xi)
            lap += float(g[0, i, j])
    assert abs(float(divergence(m, x).float() @ m.coef.reshape(-1)) - lap) < 1e-3 * abs(lap)


def test_groups_share_potentials(device):
    f, grp = ep.filters.symmetrize(ep.filters.dct(3)[:2])
    tied = ep.FoE(f, groups=grp)
    tied.coef.normal_()
    untied = ep.FoE(f, coef=tied.coef[grp])
    x = torch.rand(1, 16, 16)
    assert torch.allclose(tied.energy(x), untied.energy(x))
    Gt, Gu = features(tied, x), features(untied, x).view(1, 16, -1, 16, 16)
    Gs = torch.zeros(1, 2, Gu.shape[2], 16, 16).index_add_(1, grp, Gu)
    assert torch.allclose(Gt, Gs.view_as(Gt), atol=1e-5)


def test_save_load(tmp_path, device):
    m = _prior(device)
    m.sigma, m.objective = 0.1, "mse"
    m.save(tmp_path / "prior.pt")
    m2 = ep.FoE.load(tmp_path / "prior.pt", map_location=device)
    x = torch.rand(1, 16, 16, device=device)
    assert torch.allclose(m.energy(x), m2.energy(x)) and m2.sigma == 0.1
