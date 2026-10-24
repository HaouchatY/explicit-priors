r"""
Denoisers, and the solver they share with inverse problems.

* :py:func:`one_step`: :math:`\hat x = y - \sigma^2 \nabla \Phi(y)`, for priors fitted with
  ``objective="mse"``.
* :py:func:`map_estimate`: :math:`\hat x = \arg\min_x \tfrac12 \|x - y\|^2 + \beta \Phi(x)`,
  :math:`\beta = \sigma^2`, for priors fitted with ``objective="map"`` (or ``"score"``).
* :py:func:`denoise` picks the one that matches the prior.
* :py:func:`fista` minimises any smooth batched objective, e.g.
  :math:`\tfrac12 \|H x - y\|^2 + \lambda \Phi(x)` for an inverse problem.

Images are reflect-padded so that boundary pixels see whole cliques, and cropped back.
"""

import torch
import torch.nn.functional as F

from .foe import batched


def _pad(x, p):
    return F.pad(x[:, None], (p, p, p, p), mode="reflect")[:, 0] if p else x


def _crop(x, p):
    return x[..., p : x.shape[-2] - p, p : x.shape[-1] - p] if p else x


def _sigma(prior, sigma):
    sigma = prior.sigma if sigma is None else sigma
    if sigma is None:
        raise ValueError("pass sigma: the prior was not fitted at a noise level")
    return sigma


def one_step(prior, y, sigma: float = None, pad: int = None):
    r"""
    One-step (Tweedie) denoiser :math:`y - \sigma^2 \nabla \Phi(y)`.

    Parameters
    ----------
    prior: FoE
        Prior, usually fitted with ``objective="mse"``.
    y: Tensor
        (H, W) or (B, H, W) noisy images.
    sigma: float
        Noise std (default: ``prior.sigma``).
    pad: int
        Reflect padding (default: filter size - 1).
    """
    sigma = _sigma(prior, sigma)
    yb, out = batched(y)
    p = prior.size - 1 if pad is None else pad
    yp = _pad(yb, p)
    return out(_crop(yp - sigma**2 * prior.grad(yp), p))


def fista(fun, x0, iters: int = 100, value=None, step: float = 1.0):
    r"""
    Minimise a smooth function of a batch of images: FISTA with backtracking (Beck--Teboulle)
    and momentum restart when the objective rises.

    Parameters
    ----------
    fun: callable
        x -> (value (B,) float64, gradient (B, ...)).
    x0: Tensor
        (B, ...) initial point.
    iters: int
        Number of iterations.
    value: callable
        x -> value (B,), cheaper than ``fun`` (default: ``fun(x)[0]``).
    step: float
        Initial step 1 / L.

    Returns
    -------
    x: Tensor
        (B, ...) minimisers.
    """
    value = value or (lambda u: fun(u)[0])
    B = x0.shape[0]
    view = (B,) + (1,) * (x0.dim() - 1)
    x, zv = x0.clone(), x0.clone()
    fx = value(x)
    t = torch.ones(B, device=x0.device, dtype=torch.float64)
    L = torch.full((B,), 1.0 / step, device=x0.device, dtype=torch.float64)
    for _ in range(iters):
        fz, gz = fun(zv)
        L = 0.9 * L
        for _ in range(40):
            xn = zv - (1.0 / L).view(view).to(x0.dtype) * gz
            fn = value(xn)
            dx = (xn - zv).double().flatten(1)
            # values are float64 sums of float32 terms: compare up to their rounding
            ok = fn <= fz + (gz.double().flatten(1) * dx).sum(1) + 0.5 * L * (dx * dx).sum(1) + 1e-6 * fz.abs()
            if bool(ok.all()):
                break
            L = torch.where(ok, L, 2.0 * L)
        dec = fn <= fx
        xn = torch.where(dec.view(view), xn, x)
        tn = 0.5 * (1 + torch.sqrt(1 + 4 * t * t))
        mom = torch.where(dec, (t - 1) / tn, torch.zeros_like(t)).view(view).to(x0.dtype)
        zv = xn + mom * (xn - x)
        t = torch.where(dec, tn, torch.ones_like(t))
        x, fx = xn, torch.where(dec, fn, fx)
    return x


def map_estimate(prior, y, sigma: float = None, beta: float = None, iters: int = 100, x0=None, pad: int = 16):
    r"""
    MAP denoiser :math:`\arg\min_x \tfrac12 \|x - y\|^2 + \beta \Phi(x)`.

    Parameters
    ----------
    prior: FoE
        Prior, usually fitted with ``objective="map"``.
    y: Tensor
        (H, W) or (B, H, W) noisy images.
    sigma: float
        Noise std (default: ``prior.sigma``).
    beta: float
        Weight of the prior (default: :math:`\sigma^2`, the Bayesian weight; no tuning needed for
        ``objective="map"``).
    iters: int
        FISTA iterations.
    x0: Tensor
        Initial point (default: y).
    pad: int
        Reflect padding.
    """
    if beta is None:
        beta = _sigma(prior, sigma) ** 2
    yb, out = batched(y)
    p = max(pad, prior.size - 1)
    yp = _pad(yb, p)

    def fun(x):
        e, g = prior.energy_grad(x)
        return 0.5 * ((x - yp) ** 2).sum((1, 2), dtype=torch.float64) + beta * e, (x - yp) + beta * g

    def value(x):
        return 0.5 * ((x - yp) ** 2).sum((1, 2), dtype=torch.float64) + beta * prior.energy(x)

    x0 = yp if x0 is None else _pad(batched(x0)[0], p)
    return out(_crop(fista(fun, x0, iters, value), p))


def denoise(prior, y, sigma: float = None, **kwargs):
    """:py:func:`one_step` for ``objective="mse"`` priors, :py:func:`map_estimate` otherwise."""
    if prior.objective == "mse":
        return one_step(prior, y, sigma, **kwargs)
    return map_estimate(prior, y, sigma, **kwargs)
