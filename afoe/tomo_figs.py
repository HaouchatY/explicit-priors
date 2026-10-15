"""2-D testbed of the starting idea: reconstruct a density from its 1-D projections.

Target: a 4-mode Gaussian mixture in R^2. Data: its 1-D marginals along K = 64
directions (from 400k samples). Model: log q(x) = sum_k phi_k(<theta_k, x>) (the
maximum-entropy form). Three reconstructions:
  unfiltered back-projection   phi_k = log q_k
  filtered (redundancy)        phi_k = log q_k / rho_k,  rho_k = sum_j cos^2(theta_k - theta_j)
  exact maximum entropy        cyclic IPF from the filtered start
Writes three slide figures (schematic, reconstructions, potentials along one direction).
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
rng = np.random.default_rng(0)
TEAL, WINE, INK, SLATE = "#0D8A86", "#AB3357", "#1F2A30", "#5B6770"


def rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


means = [np.array(m) for m in [(2.3, 0.5), (-2.0, 1.6), (-1.0, -2.2), (0.2, -0.3)]]
covs = [rot(np.pi / 3) @ np.diag([1.2, 0.08]) @ rot(np.pi / 3).T,
        np.array([[0.4, 0.3], [0.3, 0.5]]),
        rot(-np.pi / 6) @ np.diag([1.0, 0.05]) @ rot(-np.pi / 6).T,
        0.05 * np.eye(2)]
wts = np.array([0.35, 0.25, 0.25, 0.15])
L, NG = 6.0, 257
g1 = np.linspace(-L, L, NG)
X, Y = np.meshgrid(g1, g1)
P = np.zeros_like(X)
for w, m, S_ in zip(wts, means, covs):
    Si = np.linalg.inv(S_)
    d0, d1 = X - m[0], Y - m[1]
    P += w * np.exp(-0.5 * (Si[0, 0] * d0 ** 2 + 2 * Si[0, 1] * d0 * d1 + Si[1, 1] * d1 ** 2)) \
        / (2 * np.pi * np.sqrt(np.linalg.det(S_)))
P /= P.sum()
NS = 400_000
comp = rng.choice(4, NS, p=wts)
S = np.concatenate([rng.multivariate_normal(means[c], covs[c], (comp == c).sum()) for c in range(4)])

K = 64
thetas = np.arange(K) * np.pi / K
rho = (np.cos(thetas[:, None] - thetas[None, :]) ** 2).sum(1)          # = K/2
NT = 321
tgrid = np.linspace(-1.5 * L, 1.5 * L, NT)
dt = tgrid[1] - tgrid[0]
edges = np.concatenate([tgrid - dt / 2, [tgrid[-1] + dt / 2]])
q_emp = np.zeros((K, NT))
for k in range(K):
    t = S @ np.array([np.cos(thetas[k]), np.sin(thetas[k])])
    q_emp[k] = gaussian_filter1d(np.histogram(t, bins=edges)[0].astype(float), 2.0) / (NS * dt)
logq = np.log(np.maximum(q_emp, 1e-10))
TT = [X * np.cos(th) + Y * np.sin(th) for th in thetas]


def joint(phi):
    lp = np.zeros_like(X)
    for k in range(K):
        lp += np.interp(TT[k].ravel(), tgrid, phi[k]).reshape(X.shape)
    Q = np.exp(lp - lp.max())
    return Q / Q.sum()


def marg(Q, k):
    return np.maximum(np.histogram(TT[k].ravel(), bins=edges, weights=Q.ravel())[0] / dt, 1e-12)


def KL(Pr, Q):
    m = Pr > 0
    return float((Pr[m] * np.log(Pr[m] / np.maximum(Q[m], 1e-300))).sum())


Q_naive = joint(logq)
Q_rho = joint(logq / rho[:, None])
phi = logq / rho[:, None]
sup = q_emp > 1e-6
phi[~sup] = -30.0
for sweep in range(2):
    for k in range(K):
        r = np.clip(np.log(np.maximum(q_emp[k], 1e-10)) - np.log(marg(joint(phi), k)), -4, 4)
        phi[k][sup[k]] += r[sup[k]]
Q_ipf = joint(phi)
kl = dict(naive=KL(P, Q_naive), rho=KL(P, Q_rho), ipf=KL(P, Q_ipf))
print("KL", kl, "rho", rho[0])

# ---------------- figure 1: projections of a density (the "sinogram" picture)
fig, ax = plt.subplots(figsize=(5.6, 5.6))
ax.imshow(P ** 0.6, origin="lower", extent=[-L, L, -L, L], cmap="Greys", vmax=(P ** 0.6).max() * 0.8)
for kk, col in zip([0, K // 2], [TEAL, WINE]):
    th = thetas[kk]
    u = np.array([np.cos(th), np.sin(th)])
    v = np.array([-np.sin(th), np.cos(th)])
    base = 7.0 * v                                    # detector line outside the density
    sel = np.abs(tgrid) <= 5.0
    prof = q_emp[kk] / q_emp[kk].max() * 2.2
    pts = base[None] + tgrid[sel, None] * u[None] + prof[sel, None] * v[None]
    ax.plot([base[0] - 5 * u[0], base[0] + 5 * u[0]], [base[1] - 5 * u[1], base[1] + 5 * u[1]],
            color=col, lw=0.9)
    ax.fill(np.r_[pts[:, 0], pts[::-1, 0] - prof[sel][::-1] * v[0]],
            np.r_[pts[:, 1], pts[::-1, 1] - prof[sel][::-1] * v[1]], color=col, alpha=0.15)
    ax.plot(pts[:, 0], pts[:, 1], color=col, lw=2.2)
    for t0 in (np.linspace(-4.5, 4.5, 10) if kk == 0 else []):   # rays, one direction
        a0, a1 = t0 * u - 5.5 * v, t0 * u + 6.8 * v
        ax.annotate("", xy=tuple(a1), xytext=tuple(a0),
                    arrowprops=dict(arrowstyle="->", color=col, lw=0.6, alpha=0.5))
ax.set_xlim(-10.5, 6.5); ax.set_ylim(-6.5, 10.5); ax.set_aspect("equal"); ax.axis("off")
fig.savefig(f"{OUT}/tomo_projections.png", dpi=170, bbox_inches="tight")

# ---------------- figure 2: reconstructions
fig, axes = plt.subplots(1, 4, figsize=(13, 3.3))
for a, (Q, ttl) in zip(axes, [(P, "target density\n"),
                              (Q_naive, f"unfiltered back-projection\nKL = {kl['naive']:.1f}"),
                              (Q_rho, f"filtered (divide by redundancy)\nKL = {kl['rho']:.2f}"),
                              (Q_ipf, f"exact max-entropy (IPF)\nKL = {kl['ipf']:.2f}")]):
    a.imshow(Q ** 0.6, origin="lower", extent=[-L, L, -L, L], cmap="magma")
    a.set_xticks([]); a.set_yticks([])                # captions are set in LaTeX
fig.tight_layout(w_pad=0.6)
fig.savefig(f"{OUT}/tomo_recon.png", dpi=170, bbox_inches="tight")

# ---------------- figure 3: potentials along one direction
k = 8
sel = q_emp[k] > 1e-4
fig, a1 = plt.subplots(figsize=(5.2, 3.0))
for f, col, lab in [(logq[k], SLATE, r"unfiltered: $\varphi_k=\log q_k$"),
                    (logq[k] / rho[k], TEAL, r"filtered: $\varphi_k=\log q_k/\rho$")]:
    a1.plot(tgrid[sel], f[sel] - f[sel].max(), color=col, lw=2.4, label=lab)
a1.set_xlabel(r"$t=\langle\theta_k,x\rangle$"); a1.set_ylabel(r"$\varphi_k(t)$")
a1.legend(fontsize=9, frameon=False, loc="lower center"); a1.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f"{OUT}/tomo_potentials.png", dpi=170, bbox_inches="tight")
print("written")
