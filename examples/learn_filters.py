"""
Learn filters with their potentials, then compare them with the DCT bank under the same fit.

    python examples/learn_filters.py --data path/to/BSD --sigma 0.05 --filters 16
"""

import argparse
import time

import torch
from common import DEVICE, load, psnr

import explicit_priors as ep

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--sigma", type=float, default=0.05, help="noise std")
p.add_argument("--filters", type=int, default=16)
p.add_argument("--size", type=int, default=7)
p.add_argument("--steps", type=int, default=1500)
p.add_argument("--save", default=None, help="where to save the learned filters")
a = p.parse_args()
s = a.sigma

train, test = load(f"{a.data}/train"), load(f"{a.data}/Set68")
t0 = time.perf_counter()
learned = ep.learn_filters(train, a.filters, a.size, sigma=s, steps=a.steps)
print(f"learned {a.filters} filters in {time.perf_counter() - t0:.0f} s")
if a.save:
    torch.save(learned.cpu(), a.save)

for name, f in (
    (f"learned ({a.filters})", learned),
    (f"DCT {a.size}x{a.size} ({a.size**2 - 1})", ep.filters.dct(a.size)),
):
    prior = ep.fit(f, train, sigma=s, basis=ep.Gauss())
    g = torch.Generator(device=DEVICE).manual_seed(0)
    out = []
    for x in test:
        y = x + s * torch.randn(x.shape, device=DEVICE, generator=g)
        with torch.no_grad():
            out.append(psnr(ep.one_step(prior, y), x))
    print(f"{name}: BSD68 {sum(out) / len(out):.2f} dB")
