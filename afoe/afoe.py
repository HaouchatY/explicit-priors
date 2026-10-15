"""afoe -- Analytic Fields of Experts: closed-form potentials for denoising.

Model. For a bank of unit-norm, zero-mean filters h_k, z_k = h_k * x,
    R(x) = sum_k sum_i lambda_k(z_ki)                      1-D experts
         + sum_k (1/m) sum_i F_k(log A_ki)                  scale field (GSM-like)
    A_ki = sum_{j in (2r+1)^2 window of i} z_kj^2 / s2_k,  m = (2r+1)^2.
grad R(x) is LINEAR in the coefficients of lambda_k' (odd piecewise-linear
basis) and F_k' (piecewise-linear in log A):  grad R(x) = sum_a c_a G_a(x).

Every object below is ONE linear least-squares problem
    min_c  E || sum_a c_a G_a(p) - t ||^2        (+ ridge/roughness, + c_1D >= 0)
with points p and targets t built from clean x, noisy y = x + s n, and the
output m of the steps fitted before it:
    step  (DSM, Vincent 2011)   p = m, t = (m - x)/s^2  ->  m <- m - s^2 grad R(m)
          (first step: m = y, i.e. denoising score matching; the estimate is
           the best one-step MMSE (Tweedie) estimator in the class)
    prox  (Gribonval 2011)      p = m, t = (y - m)/s^2  ->  Phi with
          prox_{s^2 Phi}(y) ~ m: the MAP potentials whose minimiser is the
          MMSE estimate, used as  argmin 1/2||x-y||^2 + beta Phi(x), beta = s^2.
The regularisation (ridge, roughness) is chosen on a held-out split by the
same objective (for a step, the held-out objective IS the held-out MSE).
'Unfiltered' = per-channel marginals: lambda_k = -log rho_k (GSM fitted to
the responses of clean images), summed over channels.

Data: BSD400 (180x180): FIT = train[:340] (normal equations), VAL =
train[340:384] (selection), TUNE = train[384:400] (beta, only for tuned
MAP rows). Test: Set12, BSD68. Noise seeded per (set, file, sigma).

Usage (cluster):
    python afoe.py fit   --name U5    --bank WCRR60+V4 --sigma 25 --r 5            # step 1
    python afoe.py fit   --name U5c   --chain U5 --sigma 25 --r 5                   # step 2
    python afoe.py prox  --name U5p   --chain U5,U5c --bank WCRR60+V4 --sigma 25 --r 5
    python afoe.py eval  --chain U5,U5c --sets set12,bsd68                          # one-step chain
    python afoe.py map   --name U5p --variant jmono --sets set12,bsd68 [--tune]
"""
import argparse
import json
import os
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Paths (override with environment variables):
#   AFOE_IMG    folder with train/ (BSD400, 180x180 crops), Set12/, Set68/ -- e.g. the
#               training/data/images/BSD folder of the WCRR repository
#   AFOE_BANKS  filter banks (default: data/banks.npz of this repository)
#   AFOE_OUT    where models, results, images and figures are written
#   TFP_REPO    a clone of the WCRR repository (only needed for the WCRR baseline)
_HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("TFP_REPO",
                           "/scratch/haouchat/tfp/weakly_convex_ridge_regularizer"))
OUT = Path(os.environ.get("AFOE_OUT", "/scratch/haouchat/tfp/afoe"))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG = Path(os.environ.get("AFOE_IMG", REPO / "training/data/images/BSD"))
BANKS = Path(os.environ.get("AFOE_BANKS", _HERE.parent / "data" / "banks.npz"))
if not BANKS.exists():
    BANKS = REPO / "results/round10/bank/banks.npz"
BORD = 10        # interior mask of the normal equations (circular responses)
PAD = 16         # symmetric padding at test time
NFINE = 96001    # 1-D table resolution on [-2c, 2c]
M1, EPS1 = 28, 2e-3   # 1-D basis: knots t_j = geomspace(EPS1, 1, M1) * c_k
M2 = 16               # scale-field basis: knots on the 0.2%..99.8% range of log A


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ===========================================================================
# data
# ===========================================================================
def files(name):
    return sorted((IMG / name).glob("*.png"))


TRAIN = files("train")
FIT, VAL, TUNE = TRAIN[:340], TRAIN[340:384], TRAIN[384:400]
SETS = {"set12": "Set12", "bsd68": "Set68"}


def load_gray(fp):
    return np.asarray(Image.open(fp).convert("L"), dtype=np.float64) / 255.0


def dataset(setname, sigma):
    """[(name, clean, noisy)], noise seeded by (set, file, sigma)."""
    fl = TUNE if setname == "tune" else files(SETS[setname])
    out = []
    for fp in fl:
        x = load_gray(fp)
        seed = zlib.crc32(f"{setname}/{fp.name}/{sigma}".encode())
        n = np.random.default_rng(seed).standard_normal(x.shape) * (sigma / 255.0)
        out.append((fp.name, x, x + n))
    return out


def psnr(x, ref):
    return float(10 * np.log10(1.0 / np.mean((np.asarray(x) - ref) ** 2)))


def to_t(a, dtype=torch.float32):
    return torch.as_tensor(np.asarray(a), device=DEV, dtype=dtype)


# ===========================================================================
# filter banks
# ===========================================================================
BANK_PARTS = {"V4": ["dct8", "steer8x3lo", "steer8x5", "steer12x5"],
              "WCRR60": ["WCRR60"], "ICA": ["ica13"], "DCT107": ["DCT107"], "DCT8": ["dct8"]}


def pca_bank(P, stride=2):
    """Data-analytic filters: the Karhunen-Loeve (PCA) basis of P x P patches of
    the FIT images, DC removed, i.e. the eigenvectors of the patch covariance
    (P^2 - 1 filters, by decreasing variance). For the Gaussian part of the prior
    they are optimal: the best quadratic score-matching prior is the inverse
    patch covariance, whose eigenvectors these are. One eigendecomposition."""
    fp = OUT / "banks" / f"PCA{P}.npz"
    if fp.exists():
        return np.load(fp)["h"]
    C = np.zeros((P * P, P * P))
    n = 0
    for f in FIT:
        x = load_gray(f)
        pt = np.lib.stride_tricks.sliding_window_view(x, (P, P))[::stride, ::stride].reshape(-1, P * P)
        pt = pt - pt.mean(1, keepdims=True)
        C += pt.T @ pt
        n += len(pt)
    lam, V = np.linalg.eigh(C / n)
    order = np.argsort(lam)[::-1][:P * P - 1]          # the last one is the DC null direction
    h = V[:, order].T.reshape(-1, P, P)
    fp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(fp, h=h, lam=lam[order])
    return h


def fobi_bank(P, stride=2):
    """Closed-form ICA (FOBI, Cardoso 1989): whiten the DC-free P x P patches of
    the FIT images with their PCA, then diagonalise the fourth-order matrix
    E[|z|^2 z z^T] of the whitened patches. The unmixing rows, mapped back to
    pixel space, are the filters: the directions of most distinct kurtosis,
    i.e. sparse/non-Gaussian filters from two eigendecompositions."""
    fp = OUT / "banks" / f"FOBI{P}.npz"
    if fp.exists():
        return np.load(fp)["h"]
    pts = []
    for f in FIT:
        x = load_gray(f)
        pt = np.lib.stride_tricks.sliding_window_view(x, (P, P))[::stride, ::stride].reshape(-1, P * P)
        pts.append(pt - pt.mean(1, keepdims=True))
    X = np.concatenate(pts)
    lam, V = np.linalg.eigh(X.T @ X / len(X))
    keep = np.argsort(lam)[::-1][:P * P - 1]
    Wh = V[:, keep] / np.sqrt(lam[keep])                 # whitening: z = X Wh
    Z = X @ Wh
    M4 = (Z * (Z * Z).sum(1, keepdims=True)).T @ Z / len(Z)
    mu, U = np.linalg.eigh(M4)
    h = (Wh @ U).T.reshape(-1, P, P)                      # filters s = <h, x>
    h /= np.sqrt((h ** 2).sum(axis=(1, 2), keepdims=True))
    fp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(fp, h=h, kurt=mu)
    return h


def get_bank(name):
    """'A+B+...' -> (K, P, P) zero-mean unit-norm filters, centred-padded to a
    common size. V4: DCT-8 + steerable pyramids (training-free); WCRR60: the
    60 filters of the trained WCRR (for like-for-like potential comparisons);
    PCA<P>: the patch PCA basis of the training images (data-analytic)."""
    d = np.load(BANKS)
    hs = [pca_bank(int(nm[3:])) if nm.startswith("PCA") else
          fobi_bank(int(nm[4:])) if nm.startswith("FOBI") else d[p]
          for nm in name.split("+") for p in BANK_PARTS.get(nm, [nm])]
    P = max(h.shape[1] for h in hs)
    h = np.concatenate([np.pad(a, ((0, 0), ((P - a.shape[1]) // 2,) * 2,
                                   ((P - a.shape[1]) // 2,) * 2)) for a in hs]).astype(np.float64)
    h = h - h.mean(axis=(1, 2), keepdims=True)
    return np.ascontiguousarray(h / np.sqrt((h ** 2).sum(axis=(1, 2)))[:, None, None])


def bank_fft(h, shape):
    """(K, H, W) complex64 transfer functions of the centred filters."""
    K, P, _ = h.shape
    hp = np.zeros((K,) + tuple(shape))
    hp[:, :P, :P] = h
    hp = np.roll(hp, (-(P // 2), -(P // 2)), axis=(1, 2))
    return torch.fft.fft2(torch.from_numpy(hp)).to(DEV).to(torch.complex64)


def box_sum(v, r):
    """(2r+1)^2 circular box sum over the last two dims (self-adjoint)."""
    sh = v.shape
    v4 = v.reshape(-1, 1, sh[-2], sh[-1])
    ker = torch.ones(1, 1, 2 * r + 1, 2 * r + 1, dtype=v.dtype, device=v.device)
    return F.conv2d(F.pad(v4, (r, r, r, r), mode="circular"), ker).reshape(sh)


# ===========================================================================
# bases
# ===========================================================================
class Basis1D:
    """psi_k(z) = sign(z) f_k(|z|/c_k): f piecewise linear through (0,0) and
    (t_j, v_kj), continued with slope w_k beyond t_M = 1. Coefficients per
    channel [v_k1..v_kM, w_k]. All >= 0  <=>  lambda_k even and
    non-decreasing in |z| (the monotone class)."""

    def __init__(self, c):
        self.c = np.asarray(c, float)
        self.t = np.concatenate([[0.0], np.geomspace(EPS1, 1.0, M1)])
        self.t[-1] = 1.0
        self.dt = np.diff(self.t)
        self.nb = M1 + 1
        self.t_t = torch.tensor(self.t, device=DEV, dtype=torch.float32)
        self.dt_t = torch.tensor(self.dt, device=DEV, dtype=torch.float32)
        self.c_t = torch.tensor(self.c, device=DEV, dtype=torch.float32).view(-1, 1, 1)

    def values(self, z, k0=0):
        """(Kc, nb, H, W) basis images b_ka(z_k) for channels k0..k0+Kc-1."""
        Kc, H, W = z.shape
        s = z.abs() / self.c_t[k0:k0 + Kc]
        sc = s.clamp(max=1.0)
        idx = (torch.searchsorted(self.t_t, sc.reshape(-1).contiguous(), right=True)
               .reshape(s.shape) - 1).clamp(0, M1 - 1)
        frac = (sc - self.t_t[idx]) / self.dt_t[idx]
        sgn = torch.sign(z)
        B = torch.zeros((Kc, M1 + 2, H, W), device=z.device, dtype=z.dtype)
        B.scatter_add_(1, idx.unsqueeze(1), ((1 - frac) * sgn).unsqueeze(1))
        B.scatter_add_(1, (idx + 1).unsqueeze(1), (frac * sgn).unsqueeze(1))
        B[:, M1 + 1] = sgn * (s - 1.0).clamp(min=0.0)
        return B[:, 1:]

    def table(self, coef, zg):
        """lambda_k on grids zg (K, n): c_k * integral of f_k, lambda(0) = 0."""
        coef = np.asarray(coef, float).reshape(len(self.c), self.nb)
        v = np.concatenate([np.zeros((len(self.c), 1)), coef[:, :M1]], 1)
        w = coef[:, M1]
        I = np.zeros((len(self.c), M1 + 1))
        I[:, 1:] = np.cumsum(0.5 * (v[:, :-1] + v[:, 1:]) * self.dt, 1)
        s = np.abs(zg) / self.c[:, None]
        sc = np.minimum(s, 1.0)
        idx = np.clip(np.searchsorted(self.t, sc.ravel(), side="right").reshape(s.shape) - 1,
                      0, M1 - 1)
        vi, vi1 = np.take_along_axis(v, idx, 1), np.take_along_axis(v, idx + 1, 1)
        f_in = vi + (vi1 - vi) * (sc - self.t[idx]) / self.dt[idx]
        Iin = np.take_along_axis(I, idx, 1) + 0.5 * (vi + f_in) * (sc - self.t[idx])
        e = np.maximum(s - 1.0, 0.0)
        Iout = I[:, -1:] + v[:, -1:] * e + 0.5 * w[:, None] * e ** 2
        return self.c[:, None] * np.where(s > 1.0, Iout, Iin)

    def roughness(self, K):
        """sum_j (slope_{j+1} - slope_j)^2 of f, block diagonal over channels."""
        S = np.zeros((M1 + 1, M1 + 1))
        for j in range(1, M1 + 1):
            S[j - 1, j - 1] += 1.0 / self.dt[j - 1]
            if j >= 2:
                S[j - 1, j - 2] -= 1.0 / self.dt[j - 1]
        S[M1, M1] = 1.0
        C = np.diff(np.eye(M1 + 1), axis=0) @ S
        return np.kron(np.eye(K), C.T @ C)


class BasisSF:
    """F_k'(L) piecewise linear on knots in L = log A: b_0 has an exponential
    left arm (F ~ A as A -> 0: the Gaussian limit), b_last a ramp with a
    constant tail (F ~ const * log A: the GSM asymptote). Evaluated through the
    exact cell slopes of the tabulated antiderivatives (consistent energy/grad)."""

    def __init__(self, knots, n=4096, pad=8.0):
        t = np.asarray(knots, float)
        self.nb = len(t)
        self.L = np.linspace(t[0] - pad, t[-1] + pad, n)
        self.dL = float(self.L[1] - self.L[0])
        L = self.L
        b = np.zeros((self.nb, n))
        b[0] = np.where(L <= t[0], np.exp(np.minimum(L - t[0], 0.0)),
                        np.clip(1.0 - (L - t[0]) / (t[1] - t[0]), 0, 1))
        for j in range(1, self.nb - 1):
            b[j] = (np.clip((L - t[j - 1]) / (t[j] - t[j - 1]), 0, 1)
                    - np.clip((L - t[j]) / (t[j + 1] - t[j]), 0, 1))
        b[-1] = np.clip((L - t[-2]) / (t[-1] - t[-2]), 0, 1)
        self.B = np.zeros((self.nb, n))
        self.B[:, 1:] = np.cumsum(0.5 * (b[:, 1:] + b[:, :-1]) * self.dL, 1)
        self.sbar_t = torch.tensor(np.diff(self.B, axis=1) / self.dL, device=DEV,
                                   dtype=torch.float32)

    def values(self, Lv):
        c = ((Lv - float(self.L[0])) / self.dL).floor().long().clamp(0, len(self.L) - 2)
        return self.sbar_t[:, c.reshape(-1)].reshape((self.nb,) + tuple(Lv.shape))

    def table(self, coef):
        return np.asarray(coef, float) @ self.B

    def roughness(self, K):
        Dm = np.diff(np.eye(self.nb), axis=0)
        return np.kron(np.eye(K), Dm.T @ Dm)


# ===========================================================================
# features and normal equations
# ===========================================================================
class Features:
    """G_a(x) for one image: the gradient field of every basis potential.
    1-D block: h_k^T * b_ka(z_k). Scale-field block:
    h_k^T * [(2/(m s2_k)) z_k boxsum(b_a(log A_k)/A_k)]."""

    def __init__(self, h, b1, bsf, s2, r, shape, chunk=16):
        self.K, self.b1, self.bsf, self.r, self.m = h.shape[0], b1, bsf, r, (2 * r + 1) ** 2
        self.Hf = bank_fft(h, shape)
        self.Hc = self.Hf.conj()
        self.s2 = to_t(s2).view(-1, 1, 1)
        self.D1 = self.K * b1.nb
        self.D = self.D1 + (self.K * bsf.nb if bsf is not None else 0)
        self.chunk = chunk

    def __call__(self, x):
        z = torch.fft.ifft2(self.Hf * torch.fft.fft2(x)[None]).real
        shape = x.shape
        G = torch.empty((self.D,) + tuple(shape), device=DEV, dtype=torch.float32)
        nb = self.b1.nb
        for k0 in range(0, self.K, self.chunk):
            k1 = min(k0 + self.chunk, self.K)
            B = self.b1.values(z[k0:k1], k0)
            G[k0 * nb:k1 * nb] = torch.fft.ifft2(self.Hc[k0:k1, None] * torch.fft.fft2(B)) \
                .real.reshape((k1 - k0) * nb, *shape)
        if self.bsf is not None:
            A = box_sum(z * z / self.s2, self.r).clamp_min(1e-20)
            bv = self.bsf.values(torch.log(A)) / A[None]               # (nb2, K, H, W)
            acc = box_sum(bv.permute(1, 0, 2, 3).contiguous(), self.r)  # (K, nb2, H, W)
            P = ((2.0 / self.m) * z / self.s2)[:, None] * acc
            G[self.D1:] = torch.fft.ifft2(self.Hc[:, None] * torch.fft.fft2(P)).real \
                .reshape(-1, *shape)
        return G


def normal_equations(feat, imgs, pairs, seed, bs=16):
    """A = E[G G^T], d = E[G t] over the interior of every image, with
    (points, targets) = pairs(x_batch, y_batch)."""
    shape = imgs[0].shape
    sel = torch.zeros(shape, dtype=torch.bool, device=DEV)
    sel[BORD:-BORD, BORD:-BORD] = True
    sel = sel.reshape(-1)
    A = torch.zeros((feat.D, feat.D), device=DEV, dtype=torch.float64)
    d = torch.zeros(feat.D, device=DEV, dtype=torch.float64)
    g = torch.Generator(device=DEV).manual_seed(seed)
    n = 0
    for i0 in range(0, len(imgs), bs):
        xb = torch.stack([to_t(x) for x in imgs[i0:i0 + bs]])
        p, t = pairs(xb, g)
        for q in range(len(xb)):
            Gs = feat(p[q]).reshape(feat.D, -1)[:, sel].double()
            A += Gs @ Gs.T
            d += Gs @ t[q].double().reshape(-1)[sel]
            n += int(sel.sum())
    return A / n, d / n


def box_qp(Q, d, mask, admm_iters=400, pdas_iters=60):
    """min 1/2 c'Qc - d'c s.t. c_i >= 0 on `mask`: ADMM for the active set, then
    a primal-dual active-set method (Hintermueller-Ito-Kunisch) to the exact
    KKT point."""
    D = Q.shape[0]
    rho = float(torch.diagonal(Q).mean())
    Lc = torch.linalg.cholesky(Q + rho * torch.eye(D, device=DEV, dtype=Q.dtype))
    u = torch.zeros_like(d)
    w = torch.zeros_like(d)
    for _ in range(admm_iters):
        c = torch.cholesky_solve((d + rho * (u - w))[:, None], Lc)[:, 0]
        v = c + w
        u = torch.where(mask, v.clamp_min(0.0), v)
        w = w + c - u
    act = mask & (u <= 0)
    for _ in range(pdas_iters):
        fr = ~act
        c = torch.zeros_like(d)
        c[fr] = torch.linalg.solve(Q[fr][:, fr], d[fr])
        mu = Q @ c - d
        new = mask & torch.where(act, mu > 0, c < 0)
        if torch.equal(new, act):
            break
        act = new
    return torch.where(mask, c.clamp_min(0.0), c)


def select(Af, df, Av, dv, O1, O2, D1, joint, mono, mu2_free=0.0):
    """Fit on FIT, choose (ridge, mu1, mu2) by the held-out objective on VAL.
    Returns (coefficients, held-out objective, chosen hyper-parameters)."""
    D = D1 + (O2.shape[0] if joint else 0)
    At, dt = Af[:D, :D], df[:D]
    tr = torch.trace(At) / D
    grid = [(rg, m1, m2) for rg in (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3)
            for m1 in (0.0, 1.0, 10.0, 100.0, 1e3, 1e4)
            for m2 in ((0.0, 1e-2, 1.0, 1e2, 1e4) if joint else (0.0,))]
    if mono:   # the constrained solves are costlier: ridge x mu1 at the free mu2
        grid = [(rg, m1, mu2_free) for rg in (1e-8, 1e-6, 1e-4) for m1 in (0.0, 10.0, 1e3)]
    mask = torch.zeros(D, dtype=torch.bool, device=DEV)
    mask[:D if mono == "all" else D1] = True
    best = None
    for rg, m1, m2 in grid:
        Aa = At + rg * tr * torch.eye(D, device=DEV, dtype=torch.float64)
        if m1 > 0:
            Aa[:D1, :D1] += m1 * torch.trace(At[:D1, :D1]) / torch.trace(O1) * O1
        if joint and m2 > 0:
            Aa[D1:, D1:] += m2 * torch.trace(At[D1:, D1:]) / torch.trace(O2) * O2
        try:
            c = box_qp(Aa, dt, mask) if mono else torch.linalg.solve(Aa, dt)
        except Exception:
            continue
        jv = float(0.5 * c @ (Av[:D, :D] @ c) - dv[:D] @ c)
        if np.isfinite(jv) and (best is None or jv < best[1]):
            best = (c, jv, (rg, m1, m2))
    return best


# ===========================================================================
# energies: tables, gradient, value
# ===========================================================================
class Energy:
    """R(x) from tables, batched (B, H, W). 1-D part: piecewise-linear table
    with exact cell slopes; scale field: F_k on a log-A grid, exact slopes."""

    def __init__(self, T, variant, shape):
        h = T["h"]
        self.K, self.shape = h.shape[0], shape
        self.Hh = bank_fft(h, shape)[..., : shape[1] // 2 + 1].contiguous()
        variant, _, tail = variant.partition(".")
        zg, lam = T["zg"], T[f"lam_{variant}"]
        if tail == "flat":     # beyond the fitted range |z| > c: constant (signal is not penalised)
            lam = lam.copy()
            for k in range(len(lam)):
                ck = float(T["c"][k])
                lam[k, zg[k] > ck] = np.interp(ck, zg[k], lam[k])
                lam[k, zg[k] < -ck] = np.interp(-ck, zg[k], lam[k])
        self.n = zg.shape[1]
        self.z0 = to_t(zg[:, 0]).view(1, -1, 1, 1)
        self.dz = to_t(zg[:, 1] - zg[:, 0]).view(1, -1, 1, 1)
        self.lam = to_t(lam).reshape(-1)
        self.phi = to_t(np.diff(lam, axis=1) / (zg[:, 1:2] - zg[:, 0:1])).reshape(-1)
        self.koff = (torch.arange(self.K, device=DEV) * self.n).view(1, -1, 1, 1)
        self.koffp = (torch.arange(self.K, device=DEV) * (self.n - 1)).view(1, -1, 1, 1)
        Fk = T.get(f"F_{variant}")
        self.sf = Fk is not None
        if self.sf:
            L = T["Lgrid"]
            self.r = int(T["r"])
            self.m = (2 * self.r + 1) ** 2
            self.nL, self.L0, self.dL = len(L), float(L[0]), float(L[1] - L[0])
            self.F = to_t(Fk).reshape(-1)
            self.Fs = to_t(np.diff(Fk, axis=1) / self.dL).reshape(-1)
            self.koffL = (torch.arange(self.K, device=DEV) * self.nL).view(1, -1, 1, 1)
            self.koffLp = (torch.arange(self.K, device=DEV) * (self.nL - 1)).view(1, -1, 1, 1)
            self.s2 = to_t(T["s2"]).view(1, -1, 1, 1)

    def fwd(self, x):
        return torch.fft.irfft2(self.Hh[None] * torch.fft.rfft2(x)[:, None], s=self.shape)

    def adj(self, p):
        return torch.fft.irfft2((self.Hh.conj()[None] * torch.fft.rfft2(p)).sum(1), s=self.shape)

    def _terms(self, z, need_value=True):
        u = (z - self.z0) / self.dz
        i0 = u.floor().clamp(0, self.n - 2)
        t = u - i0
        i0 = i0.long()
        p = self.phi.take(i0 + self.koffp)
        e = None
        if need_value:
            v0 = self.lam.take(i0 + self.koff)
            e = (v0 + (self.lam.take(i0 + 1 + self.koff) - v0) * t).sum(dim=(1, 2, 3),
                                                                         dtype=torch.float64)
        if self.sf:
            A = box_sum(z * z / self.s2, self.r).clamp_min(1e-20)
            uL = (torch.log(A) - self.L0) / self.dL
            j0 = uL.floor().clamp(0, self.nL - 2)
            tL = uL - j0
            j0 = j0.long()
            p = p + (2.0 / (self.m * self.s2)) * z * box_sum(self.Fs.take(j0 + self.koffLp) / A,
                                                             self.r)
            if need_value:
                f0 = self.F.take(j0 + self.koffL)
                e = e + ((f0 + (self.F.take(j0 + 1 + self.koffL) - f0) * tL) / self.m) \
                    .sum(dim=(1, 2, 3), dtype=torch.float64)
        return e, p

    def grad(self, x):
        return self.adj(self._terms(self.fwd(x), need_value=False)[1])

    def value(self, x):
        return self._terms(self.fwd(x))[0]

    def value_grad(self, x):
        e, p = self._terms(self.fwd(x))
        return e, self.adj(p)


def padded(x):
    return F.pad(x[:, None], (PAD, PAD, PAD, PAD), mode="reflect")[:, 0]


def run_chain(energies, yb, s):
    """m_0 = y, m_{j+1} = m_j - s^2 grad R_j(m_j); energies built for the padded
    shape. Returns the list of every step's output (B, H, W)."""
    outs, m = [], yb
    for E in energies:
        m = m - s * s * E.grad(padded(m))[:, PAD:-PAD, PAD:-PAD]
        outs.append(m)
    return outs


def map_solve(y, E, beta, iters):
    """argmin 1/2||x - y||^2 + beta R(x): FISTA with Beck-Teboulle backtracking
    (L doubled until the quadratic bound holds, relaxed x0.9 per iteration),
    momentum restart only when the energy rises; batched, beta per element.
    Returns (x, energy, number of energy evaluations)."""
    B = y.shape[0]
    b3, bb = beta.view(B, 1, 1).to(y.dtype), beta.view(B).double()

    def Eg(u):
        r, g = E.value_grad(u)
        return 0.5 * ((u - y) ** 2).sum(dim=(1, 2), dtype=torch.float64) + bb * r, (u - y) + b3 * g

    def Ev(u):
        return 0.5 * ((u - y) ** 2).sum(dim=(1, 2), dtype=torch.float64) + bb * E.value(u)

    x = y.clone()
    Ex, zv = Ev(x), x.clone()
    t = torch.ones(B, device=DEV, dtype=torch.float64)
    L = torch.full((B,), 2.0, device=DEV, dtype=torch.float64)
    nev = 0
    for _ in range(iters):
        Ez, gz = Eg(zv)
        L = L * 0.9
        for _bt in range(40):
            nev += 1
            xn = zv - (1.0 / L).view(B, 1, 1).to(y.dtype) * gz
            En = Ev(xn)
            dd = (xn - zv).double()
            # the energies are float64 sums of float32 terms: compare up to that
            # rounding (an exact test fails at random near convergence, and one
            # failing element would make the whole batch backtrack 40 times)
            ok = En <= Ez + (gz.double() * dd).sum(dim=(1, 2)) + 0.5 * L * (dd * dd).sum(dim=(1, 2)) \
                + 1e-6 * Ez.abs()
            if bool(ok.all()):
                break
            L = torch.where(ok, L, 2.0 * L)
        dec = En <= Ex
        xn = torch.where(dec.view(B, 1, 1), xn, x)
        tn = 0.5 * (1 + torch.sqrt(1 + 4 * t * t))
        mom = torch.where(dec, (t - 1) / tn, torch.zeros_like(t)).view(B, 1, 1).to(y.dtype)
        zv = xn + mom * (xn - x)
        t = torch.where(dec, tn, torch.ones_like(t))
        x, Ex = xn, torch.where(dec, En, Ex)
    return x, Ex, nev


# ===========================================================================
# unfiltered marginals
# ===========================================================================
def fit_marginals(z, n_comp=32, iters=300, chunk=16):
    """Per-channel zero-mean GSM on a fixed log-spaced variance grid (EM on the
    weights only: a concave MLE). Smallest variance = 8-bit quantisation."""
    K = z.shape[0]
    vq = 1.0 / (12 * 255.0 ** 2)
    W, V = [], []
    for k0 in range(0, K, chunk):
        zc = z[k0:k0 + chunk].double()
        var = zc.var(1)
        t = torch.linspace(0, 1, n_comp, device=DEV, dtype=torch.float64)
        lo = torch.log(torch.clamp(var * 1e-4, min=vq))
        v = torch.exp(lo[:, None] * (1 - t) + torch.log(var * 50.0)[:, None] * t)
        w = torch.full((zc.shape[0], n_comp), 1.0 / n_comp, device=DEV, dtype=torch.float64)
        logN = -0.5 * (zc * zc)[:, None, :] / v[:, :, None] - 0.5 * torch.log(2 * np.pi * v)[:, :, None]
        for _ in range(iters):
            w = torch.softmax(torch.log(w)[:, :, None] + logN, dim=1).mean(2).clamp_min(1e-300)
            w = w / w.sum(1, keepdim=True)
        W.append(w)
        V.append(v)
        del logN
    return torch.cat(W).cpu().numpy(), torch.cat(V).cpu().numpy()


def neglog_gsm(w, v, zg):
    """-log rho_k(z) + log rho_k(0) on grids zg (K, n)."""
    out = np.empty_like(zg)
    ic = zg.shape[1] // 2
    for k in range(zg.shape[0]):
        a = np.log(w[k])[:, None] - 0.5 * zg[k][None] ** 2 / v[k][:, None] \
            - 0.5 * np.log(2 * np.pi * v[k])[:, None]
        mx = a.max(0)
        lp = mx + np.log(np.exp(a - mx).sum(0))
        out[k] = -(lp - lp[ic])
    return out


# ===========================================================================
# fitting
# ===========================================================================
def model_path(name):
    return OUT / "models" / f"{name}.npz"


def load(name):
    d = np.load(model_path(name))
    return {k: d[k] for k in d.files}


def energies(names, shape, variant="joint"):
    return [Energy(load(n), variant, shape) for n in names]


def responses(h, xs):
    Hf = bank_fft(h, xs[0].shape)
    return torch.cat([torch.fft.ifft2(Hf * torch.fft.fft2(to_t(x))[None]).real.reshape(h.shape[0], -1)
                      for x in xs], 1)


def fit(name, kind, sigma, r, bank=None, chain=(), seed=0):
    """kind 'step': grad R(m) ~ (m - x)/s^2 at the chain output m (first step:
    m = y, DSM). kind 'prox': grad Phi(m) ~ (y - m)/s^2 (Gribonval).
    Writes tables for the 1-D-only and joint classes, free and monotone."""
    t0 = time.perf_counter()
    s = sigma / 255.0
    h = get_bank(bank) if bank else load(chain[0])["h"]
    K = h.shape[0]
    shape0 = (180 + 2 * PAD, 180 + 2 * PAD)
    Es = energies(chain, shape0)

    def pairs_fn(xb, g):
        yb = xb + s * torch.randn(xb.shape, device=DEV, generator=g)
        m = run_chain(Es, yb, s)[-1] if Es else yb
        return m, ((m - xb) if kind == "step" else (yb - m)) / (s * s)

    fit_imgs = [load_gray(f) for f in FIT]
    val_imgs = [load_gray(f) for f in VAL]
    # basis ranges from the points the model will be evaluated at
    g = torch.Generator(device=DEV).manual_seed(seed + 7)
    p0 = torch.cat([pairs_fn(torch.stack([to_t(x) for x in fit_imgs[i:i + 16]]), g)[0]
                    for i in range(0, 64, 16)])
    zp = responses(h, list(p0.cpu().numpy()))
    za = zp.abs()[:, ::5]
    c = (1.05 * za.kthvalue(int(0.999 * za.shape[1]), dim=1).values).double().cpu().numpy()
    s2 = zp.double().var(1).cpu().numpy()
    Ae = box_sum(zp.reshape(K, 64, 180, 180).permute(1, 0, 2, 3) ** 2
                 / to_t(s2).view(1, -1, 1, 1), r).clamp_min(1e-20)
    Ls = torch.log(Ae).flatten()[::97].cpu().numpy()
    del za, zp, Ae
    b1 = Basis1D(c)
    bsf = BasisSF(np.linspace(*np.quantile(Ls, (0.002, 0.998)), M2))
    feat = Features(h, b1, bsf, s2, r, (180, 180))
    Af, df = normal_equations(feat, fit_imgs, pairs_fn, seed + 11)
    Av, dv = normal_equations(feat, val_imgs, pairs_fn, seed + 12)
    log(f"{name}: {kind} forms, K={K} D={feat.D} ({time.perf_counter() - t0:.0f}s)")
    # equalise the two blocks' scales (a reparametrisation; keeps one ridge fair)
    D1 = feat.D1
    sc = torch.sqrt(torch.diagonal(Af)[:D1].mean() / torch.diagonal(Af)[D1:].mean())
    Ds = torch.cat([torch.ones(D1, device=DEV, dtype=torch.float64),
                    torch.full((feat.D - D1,), float(sc), device=DEV, dtype=torch.float64)])
    Af, Av = Af * Ds[:, None] * Ds[None], Av * Ds[:, None] * Ds[None]
    df, dv = df * Ds, dv * Ds
    O1 = torch.as_tensor(b1.roughness(K), device=DEV, dtype=torch.float64)
    O2 = torch.as_tensor(bsf.roughness(K), device=DEV, dtype=torch.float64)
    zg = np.stack([np.linspace(-2 * ci, 2 * ci, NFINE) for ci in c])
    out = dict(h=h, zg=zg, c=c, s2=s2, r=r, Lgrid=bsf.L, sigma=sigma, kind=kind,
               chain=np.array(list(chain)))
    meta = dict(name=name, kind=kind, sigma=sigma, r=r, K=K, bank=bank, chain=list(chain))
    # held-out MSE of the input points (for a step, J_val converts to MSE exactly)
    mse_in = 0.0
    g = torch.Generator(device=DEV).manual_seed(seed + 12)
    for i0 in range(0, len(val_imgs), 16):
        xb = torch.stack([to_t(x) for x in val_imgs[i0:i0 + 16]])
        p, _ = pairs_fn(xb, g)
        mse_in += float(((p - xb)[:, BORD:-BORD, BORD:-BORD] ** 2).mean()) * len(xb)
    mse_in /= len(val_imgs)
    meta["val_psnr_in"] = -10 * np.log10(mse_in)
    mu2_free = 0.0
    # mono: 1-D coefficients >= 0 (lambda_k non-decreasing in |z|); "all": also the
    # scale-field coefficients (F_k non-decreasing in the local energy: no term
    # ever rewards energy, like the -log of a Gaussian scale mixture)
    for variant, joint, mono in (("onedim", False, False), ("joint", True, False),
                                 ("mono", False, True), ("jmono", True, True),
                                 ("jpos", True, "all")):
        cc, jv, hp = select(Af, df, Av, dv, O1, O2, D1, joint, mono, mu2_free)
        if variant == "joint":
            mu2_free = hp[2]
        cc = (cc * Ds[:cc.shape[0]]).cpu().numpy()
        out[f"lam_{variant}"] = b1.table(cc[:D1].reshape(K, b1.nb), zg)
        if joint:
            out[f"F_{variant}"] = bsf.table(cc[D1:].reshape(K, bsf.nb))
        meta[variant] = dict(J_val=jv, ridge=hp[0], mu1=hp[1], mu2=hp[2])
        if kind == "step":
            meta[variant]["val_psnr_out"] = -10 * np.log10(mse_in + 2 * s ** 4 * jv)
        log(f"  {variant:6s} J_val {jv:.5e}  (ridge {hp[0]:g}, mu1 {hp[1]:g}, mu2 {hp[2]:g})"
            + (f"  VAL {meta['val_psnr_in']:.3f} -> {meta[variant]['val_psnr_out']:.3f} dB"
               if kind == "step" else ""))
    if kind == "prox" or not chain:       # unfiltered marginals on this bank
        zc = responses(h, fit_imgs[:120])
        gp = torch.Generator(device=DEV).manual_seed(12345)
        idx = torch.randperm(zc.shape[1], device=DEV, generator=gp)[:200000]
        w, v = fit_marginals(zc[:, idx])
        out.update(lam_unf=neglog_gsm(w, v, zg), gsm_w=w, gsm_v=v,
                   var_clean=zc.double().var(1).cpu().numpy())
        del zc
    meta["seconds"] = time.perf_counter() - t0
    model_path(name).parent.mkdir(parents=True, exist_ok=True)
    tmp = model_path(name).with_suffix(".tmp.npz")
    np.savez(tmp, **out)
    tmp.replace(model_path(name))
    json.dump(meta, open(OUT / "models" / f"{name}.json", "w"), indent=1)
    log(f"{name} written ({meta['seconds']:.0f}s)")


# ===========================================================================
# evaluation
# ===========================================================================
def save_merged(fp, res):
    fp.parent.mkdir(parents=True, exist_ok=True)
    cur = json.load(open(fp)) if fp.exists() else {}
    cur.update(res)
    tmp = fp.with_suffix(f".tmp{os.getpid()}")
    json.dump(cur, open(tmp, "w"), indent=1)
    tmp.replace(fp)


def eval_chain(chain, sigma, sets, variant="joint"):
    """One-step chain (no tuning): PSNR after every step; images of the last."""
    s = sigma / 255.0
    key = "+".join(chain) + f":{variant}"
    fp = OUT / "results" / f"chain_s{sigma}.json"
    done = json.load(open(fp)) if fp.exists() else {}
    for st in sets:
        if f"{st}/{key}" in done:            # preemption-safe: skip finished sets
            continue
        data = dataset(st, sigma)
        per = [dict() for _ in chain]
        imgs = {}
        t0 = time.perf_counter()
        for shp in sorted({d[1].shape for d in data}):
            grp = [d for d in data if d[1].shape == shp]
            Es = energies(chain, (shp[0] + 2 * PAD, shp[1] + 2 * PAD), variant)
            for d in grp:
                outs = run_chain(Es, to_t(d[2])[None], s)
                for j, m in enumerate(outs):
                    per[j][d[0]] = psnr(m[0].double().cpu().numpy(), d[1])
                imgs[d[0]] = outs[-1][0].cpu().numpy()
        means = [float(np.mean(list(p.values()))) for p in per]
        save_merged(fp, {f"{st}/{key}": dict(steps=means, per_image=per[-1],
                                              seconds=time.perf_counter() - t0)})
        (OUT / "images").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT / "images" / f"chain_{key.replace(':', '_')}_s{sigma}_{st}.npz", **imgs)
        log(f"chain {key} {st}: " + "  ".join(f"step{j + 1} {v:.3f}" for j, v in enumerate(means)))


def map_denoise(T, variant, ys, beta, iters):
    """MAP for a list of same-shape images at one absolute beta."""
    H, W = ys[0].shape
    E = Energy(T, variant, (H + 2 * PAD, W + 2 * PAD))
    per = T["h"].shape[0] * (H + 2 * PAD) * (W + 2 * PAD) * (130 if E.sf else 80)
    mb = max(1, int(torch.cuda.get_device_properties(0).total_memory * 0.5 // per)) \
        if DEV.type == "cuda" else 4
    out = []
    for i in range(0, len(ys), mb):
        yb = to_t(np.stack([np.pad(y, PAD, mode="symmetric") for y in ys[i:i + mb]]))
        bt = torch.full((yb.shape[0],), beta, device=DEV)
        xb, _, nev = map_solve(yb, E, bt, iters)
        out += list(xb[:, PAD:PAD + H, PAD:PAD + W].double().cpu().numpy())
        map_denoise.evals = getattr(map_denoise, "evals", 0) + nev
    return out


def iters_for(variant):
    return 4000 if variant == "unf" else 500


def tune_beta(T, variant, sigma, name):
    """beta/beta_cal on TUNE by a 3-round geometric grid (checkpointed)."""
    fp = OUT / "results" / f"tune_{name}_{variant}_s{sigma}.json"
    st = json.load(open(fp)) if fp.exists() else {"tried": {}, "round": 0, "grid": None}
    if "best" in st:
        return st["best"]
    data = dataset("tune", sigma)
    bcal = (sigma / 255.0) ** 2
    lo, hi = (2 ** -12, 2 ** -2) if variant == "unf" else (2 ** -3, 2 ** 3)
    grid = st["grid"] or list(np.geomspace(lo, hi, 9))
    for rnd in range(st["round"], 3):
        for b in grid:
            if f"{b:.6g}" in st["tried"]:
                continue
            xs = map_denoise(T, variant, [d[2] for d in data], b * bcal, iters_for(variant))
            st["tried"][f"{b:.6g}"] = float(np.mean([psnr(x, d[1]) for x, d in zip(xs, data)]))
        ks = sorted(st["tried"], key=float)
        bi = int(np.argmax([st["tried"][k] for k in ks]))
        f0 = float(ks[bi])
        if bi in (0, len(ks) - 1):
            grid = list(np.geomspace(f0 / 16, f0 / 2, 4) if bi == 0 else np.geomspace(f0 * 2, f0 * 16, 4))
        else:
            grid = list(np.geomspace(float(ks[bi - 1]), float(ks[bi + 1]), 7)[1:-1])
        st.update(round=rnd + 1, grid=[float(x) for x in grid])
        save_merged(fp, st)
        log(f"tune {name}:{variant} round {rnd}: beta/beta_cal {f0:.4g} -> {st['tried'][ks[bi]]:.3f} dB")
    ks = sorted(st["tried"], key=float)
    st["best"] = float(ks[int(np.argmax([st["tried"][k] for k in ks]))])
    save_merged(fp, st)
    return st["best"]


def eval_map(name, variant, sigma, sets, tuned):
    """MAP with beta = beta_cal (the MMSE-matched prediction, no tuning) or with
    beta tuned on TUNE."""
    T = load(name)
    b_rel = tune_beta(T, variant, sigma, name) if tuned else 1.0
    key = f"{name}:{variant}" + ("" if tuned else "@cal")
    fp = OUT / "results" / f"map_s{sigma}.json"
    done = json.load(open(fp)) if fp.exists() else {}
    for st in sets:
        if f"{st}/{key}" in done:            # preemption-safe: skip finished sets
            continue
        data = dataset(st, sigma)
        per, imgs = {}, {}
        t0 = time.perf_counter()
        for shp in sorted({d[1].shape for d in data}):
            grp = [d for d in data if d[1].shape == shp]
            xs = map_denoise(T, variant, [d[2] for d in grp], b_rel * (sigma / 255.0) ** 2,
                             iters_for(variant))
            for d, x in zip(grp, xs):
                per[d[0]] = psnr(x, d[1])
                imgs[d[0]] = x.astype(np.float32)
        mean = float(np.mean(list(per.values())))
        save_merged(fp, {f"{st}/{key}": dict(mean=mean, beta_rel=b_rel, per_image=per,
                                              seconds=time.perf_counter() - t0,
                                              gpu=torch.cuda.get_device_name(0) if DEV.type == "cuda" else "cpu")})
        (OUT / "images").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT / "images" / f"map_{key.replace(':', '_')}_s{sigma}_{st}.npz", **imgs)
        log(f"MAP {key} {st}: {mean:.3f} dB (beta = {b_rel:.4g} beta_cal, "
            f"{time.perf_counter() - t0:.0f}s)")


def solver_check(name, variant, sigma, n_img=6):
    """Budget check on BSD68 images: final energy / PSNR / evaluations / time
    against the iteration count (the PSNR must have stopped moving)."""
    T = load(name)
    data = [d for d in dataset("bsd68", sigma) if d[1].shape == (321, 481)][:n_img]
    H, W = data[0][1].shape
    E = Energy(T, variant, (H + 2 * PAD, W + 2 * PAD))
    yb = to_t(np.stack([np.pad(d[2], PAD, mode="symmetric") for d in data]))
    bt = torch.full((len(data),), (sigma / 255.0) ** 2, device=DEV)
    for it in ((250, 500, 1000, 2000) if variant != "unf" else (1000, 2000, 4000, 8000)):
        if variant == "unf":
            bt = torch.full((len(data),), 0.0046 * (sigma / 255.0) ** 2, device=DEV)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        xb, Eb, nev = map_solve(yb, E, bt, it)
        torch.cuda.synchronize()
        xs = xb[:, PAD:PAD + H, PAD:PAD + W].double().cpu().numpy()
        ps = np.mean([psnr(x, d[1]) for x, d in zip(xs, data)])
        log(f"check {name}:{variant} iters {it:5d}: E {float(Eb.sum()):.6f}  PSNR {ps:.4f}  "
            f"evals {nev}  {time.perf_counter() - t0:.1f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["fit", "prox", "eval", "map", "check"])
    p.add_argument("--name")
    p.add_argument("--bank", default=None)
    p.add_argument("--chain", default="")
    p.add_argument("--sigma", type=int, default=25)
    p.add_argument("--r", type=int, default=5)
    p.add_argument("--sets", default="set12,bsd68")
    p.add_argument("--variant", default="jmono")
    p.add_argument("--tune", action="store_true")
    a = p.parse_args()
    chain = [c for c in a.chain.split(",") if c]
    sets = a.sets.split(",")
    log(f"device {DEV} ({torch.cuda.get_device_name(0) if DEV.type == 'cuda' else 'cpu'})")
    if a.cmd in ("fit", "prox"):
        if not model_path(a.name).exists():
            fit(a.name, "step" if a.cmd == "fit" else "prox", a.sigma, a.r, a.bank, chain)
    elif a.cmd == "eval":
        eval_chain(chain, a.sigma, sets, a.variant if a.variant != "jmono" else "joint")
    elif a.cmd == "check":
        for v in a.variant.split(","):
            solver_check(a.name, v, a.sigma)
    elif a.cmd == "map":
        for v in a.variant.split(","):
            eval_map(a.name, v, a.sigma, sets, a.tune)


if __name__ == "__main__":
    main()
