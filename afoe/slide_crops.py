"""Zoom panels for the slides: rows = images, columns = methods, PSNR on the full image."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/scratch/haouchat/tfp")
import afoe
from pathlib import Path
A = afoe.OUT / "images"
O = Path("/scratch/haouchat/tfp/out/images")
cols = [("noisy", None), ("unfiltered\n(MAP)", A / "map_W60r3_unf_s25_set12.npz"),
        ("filtered\n(MAP, $\\beta=\\sigma^2$-ish)", A / "map_W60r3p_jmono_s25_set12.npz"),
        ("filtered\n(2-step, no MAP)", A / "chain_U5+U5c_joint_s25_set12.npz"),
        ("WCRR\n(trained)", O / "WCRR_s25_set12_wcrr.npz"), ("clean", None)]
rows = [("05.png", (85, 85, 72)), ("09.png", (40, 330, 112)), ("01.png", (60, 100, 72))]
data = {n: (x, y) for n, x, y in afoe.dataset("set12", 25)}
fig, ax = plt.subplots(len(rows), len(cols), figsize=(2.3 * len(cols), 2.45 * len(rows)))
for i, (img, (r0, c0, s)) in enumerate(rows):
    x, y = data[img]
    for j, (lab, fp) in enumerate(cols):
        im = y if lab == "noisy" else x if lab == "clean" else np.load(fp)[img]
        im = np.clip(im, 0, 1)
        ax[i, j].imshow(im[r0:r0 + s, c0:c0 + s], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        if i == 0:
            ax[i, j].set_title(lab.replace("$\\beta=\\sigma^2$-ish", "tuned $\\beta$"), fontsize=12)
        if lab != "clean":
            ax[i, j].text(0.03, 0.04, f"{afoe.psnr(im, x):.2f} dB", transform=ax[i, j].transAxes,
                          fontsize=11, color="black", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
fig.tight_layout(pad=0.4)
fig.savefig(afoe.OUT / "figs" / "slide_crops.png", dpi=150)
print("ok")
