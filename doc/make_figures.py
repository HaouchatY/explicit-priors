"""
The figure of the README: filters and their MAP potentials.

    python doc/make_figures.py --data path/to/BSD
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from common import load  # noqa: E402

import explicit_priors as ep  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--filters", default="dct3", help="dct3 | learned | path to a (K, s, s) tensor")
p.add_argument("--sigma", type=float, default=0.1)
p.add_argument("--size", type=int, default=7)
p.add_argument("--steps", type=int, default=1500)
p.add_argument("--out", default=str(Path(__file__).resolve().parent / "potentials.png"))
a = p.parse_args()
s = a.sigma

train = load(f"{a.data}/train")
if a.filters == "learned":
    f = ep.learn_filters(train, 8, a.size, sigma=s, steps=a.steps)
elif a.filters == "dct3":
    f = ep.filters.dct(3)
else:
    f = torch.load(a.filters)
prior = ep.fit(f, train, sigma=s, objective="map")
z, phi = prior.potentials(n=2001, extent=1.0)

K = len(f)
fig, ax = plt.subplots(2, K, figsize=(1.25 * K, 2.6), gridspec_kw=dict(height_ratios=[1, 1.1], hspace=0.1, wspace=0.15))
for k in range(K):
    w = prior.filters[k].cpu()
    ax[0, k].imshow(w, cmap="gray", vmin=-w.abs().max(), vmax=w.abs().max())
    zk, pk = z[k].cpu(), phi[k].cpu() - phi[k].min().cpu()
    half = float(zk[(zk >= 0) & (pk <= 0.95 * pk.max())].max())  # show the core and the start of the plateau
    keep = zk.abs() <= 1.6 * half
    ax[1, k].plot(zk[keep], pk[keep], color="#1d3557", lw=1.6)
    ax[1, k].axhline(0, color="#d0d4da", lw=0.8, zorder=0)
    for a_ in ax[:, k]:
        a_.set_xticks([])
        a_.set_yticks([])
    for sp in ax[0, k].spines.values():
        sp.set_color("#c3c8cf")
    for sp in ax[1, k].spines.values():
        sp.set_visible(False)
fig.savefig(a.out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"saved {a.out}")
