"""
Deblurring with a MAP prior, an example of an inverse problem:

    x = argmin_x  1/2 ||k * x - y||^2 + lam Phi(x),   lam = noise^2 (the Bayesian weight),

solved by ep.fista. The blur (circular, by FFT) is defined here: the library provides the prior and
the solver, not the forward operators.

    python examples/deblur.py --data path/to/BSD
"""

import argparse

import torch
from common import DEVICE, load, psnr

import explicit_priors as ep

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--blur", type=float, default=1.6, help="std of the Gaussian blur, in pixels")
p.add_argument("--noise", type=float, default=0.01, help="noise std")
p.add_argument("--sigma", type=float, default=0.06, help="noise std the prior is fitted at")
p.add_argument("--iters", type=int, default=300)
a = p.parse_args()
noise = a.noise

prior = ep.fit(ep.filters.dct(7), load(f"{a.data}/train"), sigma=a.sigma, objective="map")


def blur_operator(shape):
    """Circular Gaussian blur and its adjoint."""
    r = (
        torch.arange(-shape[0] // 2, shape[0] // 2, device=DEVICE),
        torch.arange(-shape[1] // 2, shape[1] // 2, device=DEVICE),
    )
    k = torch.exp(-0.5 * (r[0][:, None] ** 2 + r[1][None] ** 2) / a.blur**2)
    kf = torch.fft.rfft2(torch.fft.ifftshift(k / k.sum()))
    H = lambda x: torch.fft.irfft2(torch.fft.rfft2(x) * kf, s=shape)  # noqa: E731
    Ht = lambda x: torch.fft.irfft2(torch.fft.rfft2(x) * kf.conj(), s=shape)  # noqa: E731
    return H, Ht


g = torch.Generator(device=DEVICE).manual_seed(0)
before, after = [], []
for x in load(f"{a.data}/Set12"):
    H, Ht = blur_operator(x.shape)
    y = H(x) + noise * torch.randn(x.shape, device=DEVICE, generator=g)

    def fun(u):
        r = H(u) - y
        e, gr = prior.energy_grad(u)
        return 0.5 * (r * r).sum((1, 2), dtype=torch.float64) + noise**2 * e, Ht(r) + noise**2 * gr

    with torch.no_grad():
        xh = ep.fista(fun, y[None], a.iters)[0]
    before.append(psnr(y, x))
    after.append(psnr(xh, x))
print(f"Set12, blur {a.blur} px, noise {a.noise}: {sum(before) / 12:.2f} dB -> {sum(after) / 12:.2f} dB")
