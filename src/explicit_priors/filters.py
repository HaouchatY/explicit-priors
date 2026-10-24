r"""
Filter banks.

Analytic banks (:py:func:`dct`, :py:func:`gabor`, :py:func:`gaussian_derivatives`,
:py:func:`laplacian_of_gaussian`) need no data. :py:func:`pca` and :py:func:`ica` are computed in
closed form from image patches. :py:func:`~explicit_priors.learn_filters` learns filters with the
potentials.

Every bank is returned as a (K, s, s) float32 tensor of zero-mean, unit-norm filters. Banks of
different sizes combine with :py:func:`stack`.
"""

import math

import torch
import torch.nn.functional as F

from .foe import batched


def normalize(filters):
    """
    Zero mean and unit norm per filter.

    Parameters
    ----------
    filters: Tensor
        (K, s, s) filters.

    Returns
    -------
    filters: Tensor
        (K, s, s) float32.
    """
    f = torch.as_tensor(filters, dtype=torch.float32)
    f = f - f.mean((1, 2), keepdim=True)
    return f / f.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1)


def stack(*banks):
    """
    Concatenate banks of odd sizes, centre-padded to the largest, then :py:func:`normalize`.

    Parameters
    ----------
    banks: Tensor
        (K_i, s_i, s_i) filters.

    Returns
    -------
    filters: Tensor
        (sum K_i, s, s), s = max s_i.
    """
    s = max(b.shape[-1] for b in banks)
    out = []
    for b in banks:
        p = (s - b.shape[-1]) // 2
        out.append(F.pad(torch.as_tensor(b, dtype=torch.float32), (p, p, p, p)))
    return normalize(torch.cat(out))


def _grid(radius):
    r = torch.arange(-radius, radius + 1, dtype=torch.float64)
    return torch.meshgrid(r, r, indexing="ij")


def _rotated(radius, theta):
    X, Y = _grid(radius)
    return X * math.cos(theta) + Y * math.sin(theta), -X * math.sin(theta) + Y * math.cos(theta)


def dct(size: int = 7):
    """
    The 2-D DCT-II basis without the constant.

    Parameters
    ----------
    size: int
        Filter support s.

    Returns
    -------
    filters: Tensor
        (s^2 - 1, s, s).
    """
    n = torch.arange(size, dtype=torch.float64)
    C = torch.cos(math.pi * (2 * n[:, None] + 1) * n[None] / (2 * size))
    return normalize(torch.stack([torch.outer(C[:, k], C[:, j]) for k in range(size) for j in range(size) if k or j]))


def gabor(wavelengths=(3.0, 4.5, 6.5), orientations: int = 8, aspects=(1.0, 2.0), max_size: int = 31):
    """
    Gabor filters, even and odd.

    Parameters
    ----------
    wavelengths: tuple
        Wavelengths of the carrier, in pixels. The envelope has std wavelength / 2 across the stripes.
    orientations: int
        Orientations, evenly spaced in [0, 180) degrees.
    aspects: tuple
        Envelope std along the stripes, relative to the std across them.
    max_size: int
        Largest filter support.

    Returns
    -------
    filters: Tensor
        (2 len(wavelengths) len(aspects) orientations, s, s).
    """
    out = []
    for lam in wavelengths:
        for asp in aspects:
            sx, sy = 0.5 * lam, 0.5 * lam * asp
            R = min((max_size - 1) // 2, math.ceil(3 * max(sx, sy)))
            for i in range(orientations):
                u, v = _rotated(R, math.pi * i / orientations)
                env = torch.exp(-0.5 * (u / sx) ** 2 - 0.5 * (v / sy) ** 2)
                for ph in (0.0, 0.5 * math.pi):
                    out.append(env * torch.cos(2 * math.pi * u / lam + ph))
    return stack(*[o[None] for o in out])


def gaussian_derivatives(
    sigmas=(0.8, 1.2, 1.8, 2.7, 4.0), orders=(1, 2, 3, 4), orientations: int = 4, aspect: float = 1.0, max_size=31
):
    """
    Directional derivatives of Gaussians.

    Parameters
    ----------
    sigmas: tuple
        Gaussian std across the direction of derivation, in pixels.
    orders: tuple
        Orders of derivation.
    orientations: int
        Directions, evenly spaced in [0, 180) degrees.
    aspect: float
        Gaussian std along the direction, relative to the std across it.
    max_size: int
        Largest filter support.

    Returns
    -------
    filters: Tensor
        (len(sigmas) len(orders) orientations, s, s).
    """
    out = []
    for s in sigmas:
        R = min((max_size - 1) // 2, math.ceil(3.5 * s * max(1.0, aspect)))
        for n in orders:
            for i in range(orientations):
                u, v = _rotated(R, math.pi * i / orientations)
                He = _hermite(n, u / s)
                out.append(He * torch.exp(-0.5 * (u / s) ** 2 - 0.5 * (v / (aspect * s)) ** 2))
    return stack(*[o[None] for o in out])


def _hermite(n, x):
    """Probabilists' Hermite polynomial He_n."""
    h0, h1 = torch.ones_like(x), x
    if n == 0:
        return h0
    for k in range(1, n):
        h0, h1 = h1, x * h1 - k * h0
    return h1


def laplacian_of_gaussian(sigmas=(0.8, 1.2, 1.8, 2.7, 4.0), max_size: int = 31):
    """
    Laplacians of Gaussians.

    Parameters
    ----------
    sigmas: tuple
        Gaussian std, in pixels.
    max_size: int
        Largest filter support.

    Returns
    -------
    filters: Tensor
        (len(sigmas), s, s).
    """
    out = []
    for s in sigmas:
        X, Y = _grid(min((max_size - 1) // 2, math.ceil(3.5 * s)))
        r2 = X**2 + Y**2
        out.append(((r2 - 2 * s**2) * torch.exp(-0.5 * r2 / s**2))[None])
    return stack(*out)


def _patches(images, size, stride):
    """DC-free size x size patches, one (n, size^2) float64 block per image."""
    for x in images:
        xb, _ = batched(torch.as_tensor(x, dtype=torch.float32))
        p = F.unfold(xb[:, None].double(), size, stride=stride).transpose(1, 2).reshape(-1, size * size)
        yield p - p.mean(1, keepdim=True)


def _whitening(images, size, stride):
    C, n = 0, 0
    for p in _patches(images, size, stride):
        C, n = C + p.T @ p, n + len(p)
    lam, V = torch.linalg.eigh(C / n)
    keep = torch.argsort(lam, descending=True)[: size * size - 1]  # drop the constant (zero variance)
    return lam[keep], V[:, keep]


def pca(images, size: int = 7, stride: int = 2):
    """
    Principal components of image patches, without the constant, by decreasing variance.

    Parameters
    ----------
    images: Tensor | list
        (N, H, W) images, or a list of (H, W) images.
    size: int
        Filter support s.
    stride: int
        Patch stride.

    Returns
    -------
    filters: Tensor
        (s^2 - 1, s, s).
    """
    _, V = _whitening(images, size, stride)
    return normalize(V.T.reshape(-1, size, size))


def ica(images, size: int = 7, stride: int = 2):
    """
    Independent components of image patches in closed form (FOBI :cite:`cardoso1989source`): whitened patches,
    then the eigenvectors of their fourth-order moment matrix.

    Parameters
    ----------
    images: Tensor | list
        (N, H, W) images, or a list of (H, W) images.
    size: int
        Filter support s.
    stride: int
        Patch stride.

    Returns
    -------
    filters: Tensor
        (s^2 - 1, s, s).
    """
    lam, V = _whitening(images, size, stride)
    Wh = V / lam.sqrt()
    M, n = 0, 0
    for p in _patches(images, size, stride):
        Z = p.to(Wh) @ Wh
        M, n = M + (Z * (Z * Z).sum(1, keepdim=True)).T @ Z, n + len(Z)
    _, U = torch.linalg.eigh(M / n)
    return normalize((Wh @ U).T.reshape(-1, size, size))


def symmetrize(filters):
    """
    The 8 rotations and reflections of every filter, with groups that make each orbit share one potential.
    Pass both to :py:func:`~explicit_priors.fit`.

    Parameters
    ----------
    filters: Tensor
        (K, s, s) filters.

    Returns
    -------
    filters: Tensor
        (8K, s, s) filters.
    groups: Tensor
        (8K,) potential index.
    """
    f = torch.as_tensor(filters, dtype=torch.float32)
    out = [torch.rot90(g, k, (-2, -1)) for h in f for g in (h, h.T) for k in range(4)]
    return normalize(torch.stack(out)), torch.arange(len(f)).repeat_interleave(8)


def dilate(filters, factor: int):
    """
    The same filters at a coarser scale: taps spread by factor, then a Gaussian low-pass of std factor / 2.

    Parameters
    ----------
    filters: Tensor
        (K, s, s) filters.
    factor: int
        Scale factor.

    Returns
    -------
    filters: Tensor
        (K, s', s').
    """
    f = torch.as_tensor(filters, dtype=torch.float32)
    K, s, _ = f.shape
    up = torch.zeros(K, factor * (s - 1) + 1, factor * (s - 1) + 1)
    up[:, ::factor, ::factor] = f
    R = math.ceil(1.5 * factor)
    g = torch.exp(-0.5 * (torch.arange(-R, R + 1) / (0.5 * factor)) ** 2)
    g2 = torch.outer(g, g) / g.sum() ** 2
    return normalize(F.conv2d(up[:, None], g2[None, None], padding=2 * R)[:, 0])
