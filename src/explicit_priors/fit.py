r"""
Closed-form potentials.

The gradient of the prior is linear in the coefficients, :math:`\nabla \Phi(x) = G(x)\, c`, where
the columns of :math:`G(x)` are the gradient fields of the basis potentials. Each objective is
therefore a least-squares problem in :math:`c`, solved by one linear system :math:`A c = d`:

================  =====================================  ===========================================
objective         fits                                   normal equations
================  =====================================  ===========================================
``"score"``       the prior of clean images              :math:`A = E[G(x)^\top G(x)]`,
                  (score matching)                       :math:`d = E[\operatorname{div} G(x)]`
``"mse"``         the one-step denoiser                  :math:`A = E[G(y)^\top G(y)]`,
                  :math:`y - \sigma^2 \nabla \Phi(y)`    :math:`d = E[G(y)^\top (y - x)] / \sigma^2`
``"map"``         the MAP denoiser                       :math:`A = E[G(m)^\top G(m)]`,
                  :math:`\arg\min \tfrac12\|x-y\|^2      :math:`d = E[G(m)^\top (y - m)] / \sigma^2`
                  + \sigma^2 \Phi(x)`
================  =====================================  ===========================================

with :math:`y = x + \sigma n`. For ``"map"``, :math:`m` is the one-step MMSE estimate, so the MAP
denoiser returns :math:`m` (Gribonval 2011); ``refine`` Gauss--Newton steps then use
:math:`m = ` the MAP estimate itself and the target :math:`(y - x)/\sigma^2`.

The ridge and smoothness weights are chosen on held-out images by the same objective. Shape
constraints (``"monotone"``: potentials non-decreasing in :math:`|z|`; ``"convex"``) turn the
solve into a non-negative quadratic program.
"""

import torch
import torch.nn.functional as F

from .denoise import map_estimate, one_step
from .foe import FoE

RIDGES = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3)
SMOOTHS = (0.0, 1.0, 10.0, 100.0, 1e3, 1e4)


# ---------------------------------------------------------------------------- normal equations
def _chunks(prior, x):
    """Filter ranges whose basis fields fit in about 1.6 GB."""
    K, n = prior.filters.shape[0], prior.basis.size
    c = max(1, int(4e8 // (x.shape[0] * n * x.shape[-2] * x.shape[-1])))
    return [(k, min(K, k + c)) for k in range(0, K, c)]


def features(prior, x):
    """(B, H, W) -> (B, G n, H, W): the gradient fields of all basis potentials."""
    n, n_pot = prior.basis.size, prior.coef.shape[0]
    s, _ = prior._per_filter()
    z = prior.responses(x)
    B, _, h, w = z.shape
    H, W = x.shape[-2:]
    spec = prior.spectra((H, W))
    if spec is None:  # small filters: transposed convolutions
        G = z.new_zeros(B, n_pot, n, H, W)
    else:  # large filters: sum the groups in the Fourier domain, one inverse FFT per basis potential
        G = torch.zeros(B, n_pot, n, H, W // 2 + 1, device=z.device, dtype=spec.dtype)
    for k0, k1 in _chunks(prior, x):
        b = prior.basis.features(z[:, k0:k1] / s[:, k0:k1]).permute(0, 1, 4, 2, 3)  # (B, k, n, h, w)
        if spec is None:
            Gk = F.conv_transpose2d(
                b.reshape(B, (k1 - k0) * n, h, w),
                prior.filters[k0:k1].repeat_interleave(n, 0)[:, None],
                groups=(k1 - k0) * n,
            ).view(B, k1 - k0, n, H, W)
        else:
            Gk = torch.fft.rfft2(b, s=(H, W)) * spec[k0:k1, None]
        G.index_add_(1, prior.groups[k0:k1], Gk)
    if spec is not None:
        G = torch.fft.irfft2(G, s=(H, W))
    return G.reshape(B, n_pot * n, H, W)


def divergence(prior, x):
    r"""(G n,) :math:`\sum_{\text{batch}} \Delta` of every basis potential."""
    s, _ = prior._per_filter()
    z = prior.responses(x)
    norms = (prior.filters.double() ** 2).sum((1, 2))
    out = torch.zeros(prior.coef.shape, device=z.device, dtype=torch.float64)
    for k0, k1 in _chunks(prior, x):
        db = prior.basis.derivatives(z[:, k0:k1] / s[:, k0:k1]) / s[:, k0:k1, ..., None]
        out.index_add_(0, prior.groups[k0:k1], db.sum((0, 2, 3), dtype=torch.float64) * norms[k0:k1, None])
    return out.reshape(-1)


def normal_equations(prior, pairs):
    """
    A, d of the least-squares problem, averaged over pixels.

    Parameters
    ----------
    prior: FoE
        Filters, basis and scales.
    pairs: iterable
        (points (B, H, W), targets (B, H, W) or None) batches. Targets t give
        :math:`\\min E\\|G(p) c - t\\|^2` on the pixels that see whole cliques; None gives score
        matching at the points.
    """
    D = prior.coef.numel()
    A = torch.zeros(D, D, device=prior.filters.device, dtype=torch.float64)
    d = torch.zeros(D, device=prior.filters.device, dtype=torch.float64)
    n = 0
    for p, t in pairs:
        G = features(prior, p)
        if t is None:
            d += divergence(prior, p)
        else:
            m = prior.size - 1
            H, W = p.shape[-2:]
            G, t = G[..., m : H - m, m : W - m], t[..., m : H - m, m : W - m]
        G = G.transpose(0, 1).reshape(D, -1).double()
        A += G @ G.T
        if t is not None:
            d += G @ t.reshape(-1).double()
        n += G.shape[1]
    return A / n, d / n


# ---------------------------------------------------------------------------- solves
def nnqp(Q, q, admm_iters: int = 400, pdas_iters: int = 60):
    r""":math:`\min_{\delta \ge 0} \tfrac12 \delta^\top Q \delta - q^\top \delta`: ADMM for the active set, then a
    primal-dual active-set method to the exact KKT point."""
    rho = float(torch.diagonal(Q).mean())
    Lc = torch.linalg.cholesky(Q + rho * torch.eye(len(q), device=q.device, dtype=q.dtype))
    u, w = torch.zeros_like(q), torch.zeros_like(q)
    for _ in range(admm_iters):
        c = torch.cholesky_solve((q + rho * (u - w))[:, None], Lc)[:, 0]
        u = (c + w).clamp_min(0)
        w = w + c - u
    act = u <= 0
    for _ in range(pdas_iters):
        fr = ~act
        c = torch.zeros_like(q)
        c[fr] = torch.linalg.solve(Q[fr][:, fr], q[fr])
        new = torch.where(act, Q @ c - q > 0, c < 0)
        if torch.equal(new, act):
            break
        act = new
    return c.clamp_min(0)


def solve(A, d, ridge: float = 1e-6, smooth: float = 0.0, R=None, T=None):
    r"""
    :math:`(A + \text{ridge}\, \operatorname{diag} A + \text{smooth}\, R)\, c = d`, scaled so that the weights are
    relative; with :math:`c = T \delta,\ \delta \ge 0` if T is given.
    """
    dg = torch.diagonal(A)
    M = A + ridge * torch.diag(dg.clamp_min(1e-6 * float(dg.mean())))
    if smooth > 0:
        M = M + (smooth * torch.trace(A) / torch.trace(R)) * R
    if T is not None:
        return T @ nnqp(T.T @ M @ T, T.T @ d)
    L, info = torch.linalg.cholesky_ex(M)
    if int(info) == 0:
        return torch.cholesky_solve(d[:, None], L)[:, 0]
    return torch.linalg.lstsq(M, d[:, None]).solution[:, 0]


def _select(A, d, Av, dv, R, T, reg):
    """The fit whose held-out objective 1/2 c'Av c - dv'c is lowest."""
    if reg != "auto":
        return solve(A, d, reg[0], reg[1], R, T), tuple(reg)
    grid = [(r, m) for r in RIDGES for m in SMOOTHS]
    if T is not None:  # constrained solves cost more: a coarser grid
        grid = [(r, m) for r in RIDGES[::2] for m in SMOOTHS[::2]]
    best = None
    for r, m in grid:
        c = solve(A, d, r, m, R, T)
        j = float(0.5 * c @ (Av @ c) - dv @ c)
        if torch.isfinite(c).all() and (best is None or j < best[1]):
            best = (c, j, (r, m))
    return best[0], best[2]


# ---------------------------------------------------------------------------- data
def _batches(images, batch, device):
    """(B, H, W) batches of equal-size images, on device."""
    if torch.is_tensor(images) and images.dim() == 3:
        for i in range(0, len(images), batch):
            yield images[i : i + batch].to(device, torch.float32)
        return
    cur = []
    for x in images:
        x = torch.as_tensor(x, dtype=torch.float32)
        if cur and (len(cur) == batch or x.shape != cur[0].shape):
            yield torch.stack(cur).to(device)
            cur = []
        cur.append(x)
    if cur:
        yield torch.stack(cur).to(device)


def _noisy(images, sigma, batch, device, seed):
    """(x, y) batches with a fixed noise realisation."""
    g = torch.Generator(device=device).manual_seed(seed)
    for x in _batches(images, batch, device):
        yield x, x + sigma * torch.randn(x.shape, device=device, generator=g)


def _set_scales(prior, points, quantile: float = 0.999, max_points: int = 2_000_000):
    """scale of every potential: 1.05 x the quantile of |z| over the points, max over its group."""
    K = prior.filters.shape[0]
    zs, n = [], 0
    for p in points:
        z = prior.responses(p).abs().transpose(0, 1).reshape(K, -1)
        z = z[:, :: max(1, z.shape[1] * K // max_points)]
        zs.append(z)
        n += z.shape[1]
        if n * K >= max_points:
            break
    z = torch.cat(zs, 1)
    q = 1.05 * z.kthvalue(max(1, int(quantile * z.shape[1])), dim=1).values
    s = torch.zeros(prior.coef.shape[0], device=q.device).scatter_reduce(0, prior.groups, q, "amax")
    prior.scale.copy_(s.clamp_min(1e-8))


# ---------------------------------------------------------------------------- fit
def fit(
    filters,
    images,
    sigma: float = None,
    objective: str = "mse",
    basis=None,
    groups=None,
    shape: str = None,
    val: float = 0.1,
    reg="auto",
    refine: int = 0,
    batch: int = None,
    device=None,
    seed: int = 0,
    verbose: bool = False,
):
    r"""
    Fit the potentials of a fields of experts in closed form.

    Parameters
    ----------
    filters: Tensor | FoE
        (K, s, s) filters (see :py:mod:`~explicit_priors.filters`), or a prior whose filters,
        basis and groups are reused.
    images: Tensor | list
        Clean training images in [0, 1]: (N, H, W), or a list of (H, W).
    sigma: float
        Noise std of the denoiser ("mse", "map").
    objective: str
        "score" (prior of clean images), "mse" (one-step denoiser) or "map" (MAP denoiser).
    basis: Hat | Gauss
        Potential basis (default: :py:class:`~explicit_priors.Hat`).
    groups: Tensor
        (K,) potential index of every filter, e.g. from :py:func:`~explicit_priors.filters.symmetrize`.
    shape: str
        "free", "monotone" (even, non-decreasing in :math:`|z|`) or "convex". Default: "monotone" for "map",
        "free" otherwise.
    val: float
        Fraction of the images held out to choose the regularisation.
    reg: str | tuple
        "auto", or (ridge, smooth).
    refine: int
        Gauss--Newton steps on the MAP denoiser ("map" only).
    batch: int
        Images per batch (default: from the problem size).
    device: str
        Default: CUDA if available.

    Returns
    -------
    prior: FoE
    """
    if objective not in ("score", "mse", "map"):
        raise ValueError(f"unknown objective '{objective}'")
    if objective != "score" and sigma is None:
        raise ValueError(f"objective '{objective}' needs sigma")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if isinstance(filters, FoE):
        basis, groups, filters = filters.basis, filters.groups, filters.filters
    prior = FoE(filters, basis, groups).to(device)
    shape = shape or ("monotone" if objective == "map" else "free")
    images = list(images) if not (torch.is_tensor(images) and images.dim() == 3) else images
    n_val = int(round(val * len(images)))
    if reg == "auto" and n_val == 0:
        raise ValueError("reg='auto' needs held-out images (val > 0)")
    train, held = images[: len(images) - n_val], images[len(images) - n_val :]
    H, W = images[0].shape[-2:]
    batch = batch or max(1, int(2e8 // (prior.coef.numel() * H * W)))
    log = print if verbose else (lambda *a: None)

    if objective == "map":  # the MMSE estimate the MAP denoiser should return
        mmse = fit(prior, images, sigma, "mse", val=val, reg=reg, batch=batch, device=device)

    def pairs(imgs, seed_):
        if objective == "score":
            return ((x, None) for x in _batches(imgs, batch, device))
        if objective == "mse":
            return ((y, (y - x) / sigma**2) for x, y in _noisy(imgs, sigma, batch, device, seed_))
        return (
            (m, (y - m) / sigma**2) for x, y in _noisy(imgs, sigma, batch, device, seed_) for m in [one_step(mmse, y)]
        )

    with torch.no_grad():
        _set_scales(prior, (p for p, _ in pairs(train, seed)))
        A, d = normal_equations(prior, pairs(train, seed))
        Av, dv = normal_equations(prior, pairs(held, seed + 1)) if n_val else (A, d)
        n_pot = prior.coef.shape[0]
        R = torch.block_diag(*[prior.basis.smoothness().to(device)] * n_pot)
        T = None if shape == "free" else torch.block_diag(*[prior.basis.constraint(shape).to(device)] * n_pot)
        c, reg = _select(A, d, Av, dv, R, T, reg)
        prior.coef.copy_(c.view_as(prior.coef))
        prior.sigma, prior.objective = sigma, objective
        log(f"fit {objective}: {prior.coef.numel()} coefficients, ridge {reg[0]:g}, smooth {reg[1]:g}")
        for it in range(refine if objective == "map" else 0):
            c = _gauss_newton(prior, train, held, sigma, batch, device, seed, R, T, reg)
            if c is None:
                break
            prior.coef.copy_(c.view_as(prior.coef))
            log(f"fit map: Gauss-Newton step {it + 1}")
    return prior


def _gauss_newton(prior, train, held, sigma, batch, device, seed, R, T, reg):
    r"""One damped Gauss--Newton step on :math:`E\|\hat x_{MAP}(y) - x\|^2`: the features at the current
    MAP estimates, the target (y - x) / sigma^2. The step length is chosen on the held-out images."""

    H, W = train[0].shape[-2:]
    mb = min(64, max(1, int(4e8 // (prior.filters.shape[0] * (H + 32) * (W + 32)))))  # images per MAP solve

    def at_map(imgs, seed_):
        out = []
        for x, y in _noisy(imgs, sigma, mb, device, seed_):
            m = map_estimate(prior, y)
            out += [(x[i : i + batch], y[i : i + batch], m[i : i + batch]) for i in range(0, len(x), batch)]
        return out

    tr = at_map(train, seed)
    A, d = normal_equations(prior, ((m, (y - x) / sigma**2) for x, y, m in tr))
    c_gn = solve(A, d, reg[0], reg[1], R, T)
    c0 = prior.coef.reshape(-1).double().clone()
    va = list(_noisy(held if len(held) else train, sigma, mb, device, seed + 1))

    def loss(c):
        prior.coef.copy_(c.view_as(prior.coef))
        return sum(float(((map_estimate(prior, y) - x) ** 2).sum()) for x, y in va)

    best = (loss(c0), c0)
    for a in (1.0, 0.5, 0.25):
        c = c0 + a * (c_gn - c0)
        l_ = loss(c)
        if l_ < best[0]:
            best = (l_, c)
    prior.coef.copy_(c0.view_as(prior.coef))
    return None if best[1] is c0 else best[1]
