"""
Fit a prior on BSD400, denoise Set12 and BSD68.

    python examples/denoise.py --data path/to/BSD --sigma 0.1 --objective map

--data holds train/ (400 images), Set12/ and Set68/.
"""

import argparse
import time

import torch
from common import DEVICE, load, psnr

import explicit_priors as ep

BANKS = {
    "dct7": lambda: ep.filters.dct(7),
    "derivatives": lambda: ep.filters.gaussian_derivatives(),
    "gabor": lambda: ep.filters.gabor(),
}

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--sigma", type=float, default=0.1, help="noise std")
p.add_argument("--objective", default="mse", choices=["mse", "map"])
p.add_argument("--filters", default="dct7", help=f"{' | '.join(BANKS)} | path to a (K, s, s) tensor")
p.add_argument("--refine", type=int, default=0, help="Gauss-Newton steps (map)")
a = p.parse_args()
s = a.sigma

filters = BANKS[a.filters]() if a.filters in BANKS else torch.load(a.filters)
train = load(f"{a.data}/train")
t0 = time.perf_counter()
prior = ep.fit(filters, train, sigma=s, objective=a.objective, refine=a.refine)
torch.cuda.synchronize() if DEVICE == "cuda" else None
print(f"{prior}\nfit: {time.perf_counter() - t0:.1f} s")

g = torch.Generator(device=DEVICE).manual_seed(0)
for name in ("Set12", "Set68"):
    out = []
    for x in load(f"{a.data}/{name}"):
        y = x + s * torch.randn(x.shape, device=DEVICE, generator=g)
        with torch.no_grad():
            out.append(psnr(ep.denoise(prior, y), x))
    print(f"{name}: {sum(out) / len(out):.2f} dB")
