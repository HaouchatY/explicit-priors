"""Timing benchmark, everything on ONE GPU in one exclusive job.

    python afoe_bench.py

Fit: the three closed-form fits from scratch (step 1, step 2, prox) on WCRR's
60 filters. Inference: one 512x512 (Set12/08) and one 481x321 (BSD68) image,
warm-up once, then the median of 3 synchronised runs, for: our two-step
closed-form estimator, our MAP (500 FISTA iterations), our unfiltered MAP
(4000 iterations), and WCRR's own MAP solver.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import afoe  # noqa: E402

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
sys.path.insert(0, str(afoe.REPO))


def sync():
    if afoe.DEV.type == "cuda":
        torch.cuda.synchronize()


def timed(fn, reps=3):
    fn()                                   # warm-up
    ts = []
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    gpu = torch.cuda.get_device_name(0) if afoe.DEV.type == "cuda" else "cpu"
    res = dict(gpu=gpu, fit={}, infer={})
    afoe.log(f"benchmark on {gpu}")
    # ---------------- fits from scratch (exclusive, sequential) ----------------
    for name, kind, chain in (("bench_s1", "step", ()), ("bench_s2", "step", ("bench_s1",)),
                              ("bench_px", "prox", ("bench_s1", "bench_s2"))):
        p = afoe.model_path(name)
        if p.exists():
            p.unlink()
        sync()
        t0 = time.perf_counter()
        afoe.fit(name, kind, 25, 3, bank="WCRR60" if not chain else
                 ("WCRR60" if kind == "prox" else None), chain=chain)
        sync()
        res["fit"][name] = time.perf_counter() - t0
        afoe.log(f"fit {name}: {res['fit'][name]:.1f}s")
    # ---------------- inference ----------------
    s = 25 / 255.0
    imgs = {"512x512": [d for d in afoe.dataset("set12", 25) if d[0] == "08.png"][0],
            "481x321": [d for d in afoe.dataset("bsd68", 25) if d[1].shape == (321, 481)][0]}
    T1, T2, Tp = afoe.load("bench_s1"), afoe.load("bench_s2"), afoe.load("bench_px")
    from tf_foe.wcrr_load import load_wcrr
    from tf_foe.map_denoise import wcrr_denoise
    wcrr = load_wcrr(device=str(afoe.DEV), seed=0, sn_steps=100, dtype=torch.float64)
    for key, (nm, x, y) in imgs.items():
        H, W = x.shape
        shp = (H + 2 * afoe.PAD, W + 2 * afoe.PAD)
        Es = [afoe.Energy(T1, "joint", shp), afoe.Energy(T2, "joint", shp)]
        yt = afoe.to_t(y)[None]
        r = {}
        r["ours: 2-step closed form"] = timed(lambda: afoe.run_chain(Es, yt, s))
        r["ours: MAP, 1-D+SF, 500 it"] = timed(lambda: afoe.map_denoise(Tp, "jmono", [y], s * s, 500))
        r["ours: MAP, 1-D+SF, 250 it"] = timed(lambda: afoe.map_denoise(Tp, "jmono", [y], s * s, 250))
        r["ours: MAP, 1-D, 500 it"] = timed(lambda: afoe.map_denoise(Tp, "mono", [y], s * s, 500))
        r["unfiltered: MAP, 4000 it"] = timed(lambda: afoe.map_denoise(T1, "unf", [y], 0.0044 * s * s, 4000),
                                              reps=1)
        yw = torch.as_tensor(y, device=afoe.DEV, dtype=torch.float64).view(1, 1, H, W)
        r["WCRR: its own MAP solver (fp64)"] = timed(lambda: wcrr_denoise(wcrr, yw, sigma_cond=25.0))
        res["infer"][key] = r
        for k, v in r.items():
            afoe.log(f"{key} {k}: {v:.3f}s")
    fp = afoe.OUT / "results" / f"bench_{gpu.replace(' ', '_')}.json"
    json.dump(res, open(fp, "w"), indent=1)
    afoe.log(f"wrote {fp}")


if __name__ == "__main__":
    main()
