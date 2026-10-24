r"""
Filters learned with the potentials, by variable projection.

For filters :math:`W`, the best potentials of the one-step denoiser are one solve,
:math:`c^*(W) = A(W)^{-1} d(W)`. The filters then follow the gradient of the denoising loss at fixed
:math:`c^*` (which is the gradient of the reduced loss, by the envelope theorem):

.. math::

   \min_W \ E \big\| \sigma^2 \nabla \Phi_{W, c^*(W)}(y) - (y - x) \big\|^2 .

Filters are zero-mean and unit-norm, parametrised in whitened patch coordinates. Filters that the
solve barely uses are re-initialised a few times during the descent.
"""

import math

import torch
import torch.nn.functional as F

from .basis import Gauss
from .fit import normal_equations, solve
from .foe import FoE, correlate, correlate_adjoint


def _crops(images, n, size, g):
    out = []
    for i in torch.randint(len(images), (n,), generator=g).tolist():
        x = images[i]
        a = int(torch.randint(x.shape[-2] - size + 1, (1,), generator=g))
        b = int(torch.randint(x.shape[-1] - size + 1, (1,), generator=g))
        out.append(x[a : a + size, b : b + size])
    return torch.stack(out)


def _coordinates(images, size, whiten):
    """(s^2, s^2 - 1) map from parameters to zero-mean filters, whitened by the patch covariance."""
    Z = torch.linalg.svd(torch.eye(size * size, dtype=torch.float64) - 1.0 / size**2)[0][:, : size * size - 1]
    if not whiten:
        return Z
    p = torch.cat([F.unfold(x[None, None].double().cpu(), size, stride=2)[0].T for x in images[:64]])
    p = p - p.mean(1, keepdim=True)
    lam, U = torch.linalg.eigh(Z.T @ (p.T @ p / len(p)) @ Z)
    return Z @ U @ torch.diag(lam.clamp_min(1e-12) ** -0.5) @ U.T


def learn_filters(
    images,
    n_filters: int = 16,
    size: int = 7,
    sigma: float = 25 / 255,
    steps: int = 1000,
    batch: int = 16,
    crop: int = 96,
    lr: float = 0.02,
    whiten: bool = True,
    init=None,
    restarts: int = 3,
    device=None,
    seed: int = 0,
    verbose: bool = False,
):
    r"""
    Learn filters for the one-step denoiser at noise sigma, with the potentials solved at every step.

    Parameters
    ----------
    images: Tensor | list
        Clean training images in [0, 1], at least crop x crop.
    n_filters: int
        Number of filters K.
    size: int
        Filter support s.
    sigma: float
        Noise std.
    steps: int
        Gradient steps (Adam, cosine learning-rate decay).
    batch, crop: int
        Random crops per step and their size.
    lr: float
        Learning rate.
    whiten: bool
        Parametrise the filters in whitened patch coordinates (better conditioned).
    init: Tensor
        (K, s, s) initial filters (default: random).
    restarts: int
        Times the least useful filters (value < 0.1 x max) are re-initialised.

    Returns
    -------
    filters: Tensor
        (K, s, s) zero-mean, unit-norm filters, ready for :py:func:`~explicit_priors.fit`.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    images = [torch.as_tensor(x, dtype=torch.float32).to(device) for x in images]
    g = torch.Generator().manual_seed(seed)
    M = _coordinates(images, size, whiten).float().to(device)
    if init is None:
        V = torch.randn(n_filters, M.shape[1], generator=g).to(device)
    else:
        V = torch.as_tensor(init, dtype=torch.float32).to(device).flatten(1) @ torch.linalg.pinv(M).T
    V = (V / V.norm(dim=1, keepdim=True)).requires_grad_()

    def filters():
        Wf = (V @ M.T).view(-1, size, size)
        return Wf / Wf.flatten(1).norm(dim=1).view(-1, 1, 1)

    basis = Gauss()
    n, m = basis.size, size - 1

    def potentials(Wf, x, y):
        prior = FoE(Wf.detach(), basis).to(device)
        A, d = normal_equations(prior, [(y, (y - x) / sigma**2)])
        return A, d

    def batch_pair(k):
        x = _crops(images, k, crop, g).to(device)
        return x, x + sigma * torch.randn(x.shape, generator=g).to(device)

    xe, ye = batch_pair(32)
    opt = torch.optim.Adam([V], lr=lr)
    at = {int(steps * (j + 1) / (restarts + 1)) for j in range(restarts)}
    for step in range(1, steps + 1):
        for gr in opt.param_groups:
            gr["lr"] = lr * (0.1 + 0.45 * (1 + math.cos(math.pi * step / steps)))
        if step in at:
            with torch.no_grad():
                drop = _useless(filters(), basis, xe, ye, sigma)
                if drop:
                    V[drop] = torch.randn(len(drop), V.shape[1], generator=g).to(device)
                    V[drop] /= V[drop].norm(dim=1, keepdim=True)
                    opt = torch.optim.Adam([V], lr=lr)
            if verbose:
                print(f"learn_filters: step {step}, re-initialised {len(drop)} filters")
        x, y = batch_pair(batch)
        Wf = filters()
        with torch.no_grad():
            A, d = potentials(Wf, x, y)
            c = solve(A, d).float().view(-1, n)
        r = correlate_adjoint(basis.psi(correlate(y, Wf), c), Wf) - (y - x) / sigma**2
        loss = 0.5 * (r[..., m : r.shape[-2] - m, m : r.shape[-1] - m] ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if verbose and step % 100 == 0:
            print(f"learn_filters: step {step}, loss {float(loss):.6g}")
    return filters().detach()


def _useless(Wf, basis, x, y, sigma):
    """Filters whose potential lowers the objective by less than 0.1 x the most useful one."""
    A, d = normal_equations(FoE(Wf, basis).to(Wf.device), [(y, (y - x) / sigma**2)])
    n, K = basis.size, Wf.shape[0]

    def J(keep):
        c = solve(A[keep][:, keep], d[keep])
        return float(0.5 * c @ A[keep][:, keep] @ c - d[keep] @ c)

    j_all = J(torch.arange(K * n, device=Wf.device))
    idx = torch.arange(K * n, device=Wf.device)
    value = [J(idx[idx // n != k]) - j_all for k in range(K)]
    top = max(value)
    return [k for k in range(K) if value[k] < 0.1 * top]
