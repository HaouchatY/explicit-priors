"""Figures and tables for afoe (cluster; CPU is enough).

    python afoe_figs.py potentials --specs "W60r3:unf:...:#ab3357;W60r3p:mono:...:#0d8a86" --name pot
    python afoe_figs.py table --rows "map:W60r3:unf:ours unfiltered;map:W60r3p:jmono:ours filtered;chain:W60r3+W60r3c:joint:one-step"
    python afoe_figs.py crops --rows "..." --set set12 --img 08.png,09.png

Potentials are drawn in DATA-TERM units (the penalty added to 1/2||x-y||^2
for one filter response): beta * lambda_k(z) with each row's own beta, and
psi(s_k ||W_k|| z)/s_k^2 for WCRR on its own filters, so all curves on WCRR's
bank are exactly comparable.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import afoe  # noqa: E402

FIG = afoe.OUT / "figs"
FIG.mkdir(parents=True, exist_ok=True)
OLD = Path("/scratch/haouchat/tfp/out")          # WCRR / BM3D reference runs


def beta_rel(name, variant, sigma):
    fp = afoe.OUT / "results" / f"tune_{name}_{variant}_s{sigma}.json"
    return json.load(open(fp))["best"] if fp.exists() else 1.0


def lut(z, zg, lam):
    """the table exactly as the solver reads it: linear inside, linear
    extrapolation with the end-cell slope outside."""
    v = np.interp(z, zg, lam)
    lo, hi = z < zg[0], z > zg[-1]
    v[lo] = lam[0] + (lam[1] - lam[0]) / (zg[1] - zg[0]) * (z[lo] - zg[0])
    v[hi] = lam[-1] + (lam[-1] - lam[-2]) / (zg[-1] - zg[-2]) * (z[hi] - zg[-1])
    return v


def wcrr_curves(z, sigma):
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    sys.path.insert(0, str(afoe.REPO))
    from tf_foe.wcrr_load import load_wcrr
    from tf_foe.explore.decomp import wcrr_filters, wcrr_scaling, wcrr_psi
    model = load_wcrr(device="cpu", seed=0, sn_steps=100, dtype=torch.float64)
    h, nrm = wcrr_filters(model)
    s = wcrr_scaling(model, sigma=float(sigma))
    return h, np.array([wcrr_psi(model, s[k] * nrm[k] * z) / s[k] ** 2 for k in range(h.shape[0])])


def cmd_potentials(specs, sigma, name, n_show=8):
    bcal = (sigma / 255.0) ** 2
    sn = sigma / 255.0
    u = np.linspace(-6, 6, 1201)
    z = u * sn
    curves, h_bank, var = [], None, None
    for sp in specs:
        nm, variant, lab, col = sp.split(":")
        T = afoe.load(nm)
        if h_bank is None:
            h_bank = T["h"]
        assert np.allclose(T["h"], h_bank), f"{nm}: different bank"
        if var is None and "var_clean" in T:
            var = T["var_clean"]
        br = beta_rel(nm, variant, sigma)
        tab = T[f"lam_{variant}"]
        cv = np.stack([br * bcal * lut(z, T["zg"][k], tab[k]) for k in range(len(tab))])
        curves.append((lab, col, cv, br))
    K = h_bank.shape[0]
    hw, Wc = wcrr_curves(z, sigma) if K == 60 else (None, None)
    if hw is not None:
        corr = np.abs((hw.reshape(60, -1) * h_bank.reshape(60, -1)).sum(1))
        assert corr.min() > 0.99, f"bank is not WCRR's: min |corr| {corr.min():.3f}"
    order = np.argsort(var)
    pick = order[np.linspace(0, K - 1, n_show).round().astype(int)]
    fig, axes = plt.subplots(2, n_show // 2, figsize=(4.0 * n_show // 2, 7.4), sharex=True)
    for ax, k in zip(axes.ravel(), pick):
        for lab, col, cv, br in curves:
            ax.plot(u, cv[k], color=col, lw=2.2, label=f"{lab}  (beta = {br:.3g} beta_cal)")
        if Wc is not None:
            ax.plot(u, Wc[k], color="#333333", lw=1.6, ls="--", label="WCRR (trained)")
        ax.set_title(f"filter {k}   std/noise = {np.sqrt(var[k]) / sn:.2f}", fontsize=9)
        ins = ax.inset_axes([0.40, 0.66, 0.2, 0.3])
        ins.imshow(h_bank[k], cmap="RdBu_r", vmin=-np.abs(h_bank[k]).max(),
                   vmax=np.abs(h_bank[k]).max(), interpolation="nearest")
        ins.set_xticks([]); ins.set_yticks([])
        ax.axvline(0, color="0.85", lw=0.8)
        ax.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("filter response  z / sigma_noise")
    for ax in axes[:, 0]:
        ax.set_ylabel("penalty added to 1/2||x-y||^2")
    hs, ls = axes[0, 0].get_legend_handles_labels()
    fig.legend(hs, ls, loc="lower center", ncol=len(ls), fontsize=9, frameon=False)
    fig.suptitle(f"per-filter penalties in data-term units (sigma = {sigma})")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / f"{name}_s{sigma}.png", dpi=140)
    np.savez(FIG / f"{name}_s{sigma}.npz", u=u, pick=pick, h=h_bank, var=var,
             wcrr=(Wc if Wc is not None else np.zeros(0)), labels=np.array([c[0] for c in curves]),
             betas=np.array([c[3] for c in curves]), **{f"c{i}": c[2] for i, c in enumerate(curves)})
    if Wc is not None:
        msk = np.abs(u) <= 4
        err = {lab: float(np.median([np.linalg.norm(cv[k][msk] - Wc[k][msk])
                                     / np.linalg.norm(Wc[k][msk]) for k in range(K)]))
               for lab, _, cv, _ in curves}
        json.dump(err, open(FIG / f"{name}_s{sigma}_dist.json", "w"), indent=1)
        print("median relative L2 distance to WCRR (|z| <= 4 sigma_n):", err)
    print("wrote", FIG / f"{name}_s{sigma}.png")


def cmd_filters(banks, name="filters", n_show=48):
    """Filter galleries side by side: one panel per bank, n_show filters each
    (PCA: by decreasing variance; others: evenly spaced through the bank)."""
    fig, axes = plt.subplots(1, len(banks), figsize=(3.3 * len(banks), 3.6))
    for ax, b in zip(np.atleast_1d(axes), banks):
        h = afoe.get_bank(b)
        idx = np.arange(min(n_show, len(h))) if b.startswith("PCA") else \
            np.linspace(0, len(h) - 1, min(n_show, len(h))).round().astype(int)
        P = h.shape[1]
        g = int(np.ceil(np.sqrt(len(idx))))
        M = np.full((g * (P + 1) + 1, g * (P + 1) + 1), np.nan)
        for j, k in enumerate(idx):
            r, c = divmod(j, g)
            f = h[k] / np.abs(h[k]).max()
            M[1 + r * (P + 1):1 + r * (P + 1) + P, 1 + c * (P + 1):1 + c * (P + 1) + P] = f
        ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(f"{b} ({len(h)} filters)", fontsize=11)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=170, bbox_inches="tight")
    print("wrote", FIG / f"{name}.png")


def per_image(kind, nm, variant, st, sigma):
    if kind in ("wcrr", "bm3d"):
        R = json.load(open(OLD / "results" / f"eval_{kind.upper()}_s{sigma}.json"))
        return R[f"{st}/{kind}"]["per_image"]
    if kind == "chain":
        R = json.load(open(afoe.OUT / "results" / f"chain_s{sigma}.json"))
        return R.get(f"{st}/{nm}:{variant}", {}).get("per_image")
    R = json.load(open(afoe.OUT / "results" / f"map_s{sigma}.json"))
    return R.get(f"{st}/{nm}:{variant}", {}).get("per_image")


def cmd_table(rows, sigma, sets=("set12", "bsd68")):
    out = {}
    rows = [("bm3d", "-", "-", "BM3D"), ("wcrr", "-", "-", "WCRR (trained)")] + \
        [tuple(r.split(":")) for r in rows]
    print(f"{'set':6s} {'method':46s} {'PSNR':>7s} {'vs WCRR':>8s} {'>=WCRR':>7s}")
    for st in sets:
        w = per_image("wcrr", None, None, st, sigma)
        for kind, nm, variant, lab in rows:
            p = per_image(kind, nm, variant, st, sigma)
            if not p or len(p) != len(w):
                continue
            dif = np.array([p[n] - w[n] for n in w])
            m = float(np.mean(list(p.values())))
            out[f"{st}/{lab}"] = dict(mean=m, vs_wcrr=float(dif.mean()),
                                      wins=int((dif > 0).sum()), n=len(dif), per_image=p)
            print(f"{st:6s} {lab:46s} {m:7.3f} {dif.mean():+8.3f} {(dif > 0).sum():3d}/{len(dif)}")
    json.dump(out, open(FIG / f"table_s{sigma}.json", "w"), indent=1)


def image_of(kind, nm, variant, st, sigma, img):
    if kind == "wcrr":
        return np.load(OLD / "images" / f"WCRR_s{sigma}_{st}_wcrr.npz")[img]
    if kind == "chain":
        key = f"{nm}:{variant}".replace(":", "_")
        return np.load(afoe.OUT / "images" / f"chain_{key}_s{sigma}_{st}.npz")[img]
    key = f"{nm}:{variant}".replace(":", "_")
    return np.load(afoe.OUT / "images" / f"map_{key}_s{sigma}_{st}.npz")[img]


def cmd_crops(rows, sigma, st, img, box=None):
    data = {n: (x, y) for n, x, y in afoe.dataset(st, sigma)}
    x, y = data[img]
    ims = [("noisy", y)] + [(r.split(":")[3], image_of(*r.split(":")[:3], st, sigma, img))
                            for r in rows] + [("WCRR (trained)", image_of("wcrr", 0, 0, st, sigma, img)),
                                              ("clean", x)]
    H, W = x.shape
    r0, c0, s = box or (H // 3, W // 3, min(H, W) // 4)
    fig, axes = plt.subplots(2, len(ims), figsize=(2.6 * len(ims), 5.8))
    for j, (k, im) in enumerate(ims):
        im = np.clip(im, 0, 1)
        axes[0, j].imshow(im, cmap="gray", vmin=0, vmax=1)
        axes[0, j].add_patch(plt.Rectangle((c0, r0), s, s, fill=False, ec="#0d8a86", lw=1.2))
        axes[1, j].imshow(im[r0:r0 + s, c0:c0 + s], cmap="gray", vmin=0, vmax=1,
                          interpolation="nearest")
        axes[0, j].set_title(k if k == "clean" else f"{k}\n{afoe.psnr(im, x):.2f} dB", fontsize=8)
        for a in axes[:, j]:
            a.axis("off")
    fig.tight_layout()
    fp = FIG / f"crops_s{sigma}_{st}_{img.replace('.png', '')}.png"
    fig.savefig(fp, dpi=130)
    print("wrote", fp)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["potentials", "table", "crops", "filters"])
    p.add_argument("--banks", default="WCRR60,DCT8,PCA7,PCA13")
    p.add_argument("--specs", default="")
    p.add_argument("--rows", default="")
    p.add_argument("--name", default="potentials")
    p.add_argument("--sigma", type=int, default=25)
    p.add_argument("--set", default="set12")
    p.add_argument("--img", default="08.png")
    p.add_argument("--box", default="", help="r0,c0,size of the zoom")
    a = p.parse_args()
    if a.cmd == "filters":
        cmd_filters(a.banks.split(","))
        return
    if a.cmd == "potentials":
        cmd_potentials(a.specs.split(";"), a.sigma, a.name)
    elif a.cmd == "table":
        cmd_table(a.rows.split(";"), a.sigma)
    else:
        box = tuple(int(v) for v in a.box.split(",")) if a.box else None
        for img in a.img.split(","):
            cmd_crops(a.rows.split(";"), a.sigma, a.set, img, box)


if __name__ == "__main__":
    main()
