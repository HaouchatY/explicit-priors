r"""
Bases of the potential derivatives :math:`\psi_k = \phi_k'`.

Every basis function is odd, so every potential is even. A basis acts on normalised responses
:math:`u = z / s_k`, where :math:`s_k` is the scale of filter :math:`k`:

.. math::

   \psi_k(z) = \sum_a c_{ka} \, b_a(z / s_k),
   \qquad
   \phi_k(z) = s_k \sum_a c_{ka} \, B_a(z / s_k),
   \qquad
   B_a' = b_a, \ B_a(0) = 0.

* :py:class:`Hat` is piecewise linear, fast, and supports the shape constraints of
  :py:func:`~explicit_priors.fit`. It is the default.
* :py:class:`Gauss` is smooth. :py:func:`~explicit_priors.learn_filters` uses it.

Coefficients are passed per filter, as a (K, n) tensor, and responses as (B, K, ...) tensors.
"""

import math

import torch


class Hat:
    r"""
    Piecewise-linear :math:`\psi`: hat functions on knots
    :math:`0 = t_0 < t_1 < \dots < t_n = 1`, geometric in :math:`(\epsilon, 1]`, continued beyond
    :math:`|u| = 1` by a linear tail.

    Coefficients :math:`[v_1, \dots, v_n, w]` are the values of :math:`\psi` at the knots and the
    slope of the tail.

    Parameters
    ----------
    n: int
        Number of knots.
    eps: float
        First knot, relative to the scale.
    """

    def __init__(self, n: int = 28, eps: float = 2e-3):
        self.n, self.eps = n, eps
        self.size = n + 1
        self._t = torch.cat([torch.zeros(1), torch.logspace(math.log10(eps), 0.0, n)])
        self._t[-1] = 1.0

    def config(self) -> dict:
        return dict(name="Hat", n=self.n, eps=self.eps)

    def _cell(self, u):
        t = self._t.to(u)
        a = u.abs()
        ac = a.clamp(max=1.0)
        j = (torch.searchsorted(t, ac.contiguous(), right=True) - 1).clamp(0, self.n - 1)
        f = (ac - t[j]) / (t[j + 1] - t[j])
        return t, a, ac, j, f

    def features(self, u):
        """(..., n+1) values :math:`b_a(u)`."""
        _, a, _, j, f = self._cell(u)
        sg = torch.sign(u)
        out = torch.zeros(u.shape + (self.n + 2,), device=u.device, dtype=u.dtype)
        out.scatter_add_(-1, j[..., None], ((1 - f) * sg)[..., None])
        out.scatter_add_(-1, j[..., None] + 1, (f * sg)[..., None])
        out[..., -1] = sg * (a - 1).clamp(min=0)
        return out[..., 1:]

    def derivatives(self, u):
        """(..., n+1) values :math:`b_a'(u)`."""
        t, a, _, j, _ = self._cell(u)
        inv = 1.0 / (t[j + 1] - t[j]) * (a < 1)
        out = torch.zeros(u.shape + (self.n + 2,), device=u.device, dtype=u.dtype)
        out.scatter_add_(-1, j[..., None], -inv[..., None])
        out.scatter_add_(-1, j[..., None] + 1, inv[..., None])
        out[..., -1] = (a > 1).to(u.dtype)
        return out[..., 1:]

    def _knots(self, coef):
        v = torch.cat([torch.zeros_like(coef[:, :1]), coef[:, :-1]], 1)  # psi at t_0 .. t_n
        return v, coef[:, -1]

    def psi(self, u, coef):
        r""":math:`\sum_a c_{ka} b_a(u)` for (B, K, ...) responses and (K, n+1) coefficients."""
        _, a, ac, j, f = self._cell(u)
        v, w = self._knots(coef.to(u))
        K, m = v.shape
        off = (torch.arange(K, device=u.device) * m).view((1, K) + (1,) * (u.dim() - 2))
        vf = v.reshape(-1)
        vj, vj1 = vf[off + j], vf[off + j + 1]
        wk = w.view((1, K) + (1,) * (u.dim() - 2))
        return torch.sign(u) * (vj + (vj1 - vj) * f + wk * (a - 1).clamp(min=0))

    def phi(self, u, coef):
        r""":math:`\sum_a c_{ka} B_a(u)` for (B, K, ...) responses and (K, n+1) coefficients."""
        t, a, ac, j, f = self._cell(u)
        v, w = self._knots(coef.to(u))
        K, m = v.shape
        dt = (t[1:] - t[:-1]).to(v)
        cum = torch.cat([torch.zeros_like(v[:, :1]), torch.cumsum(0.5 * (v[:, 1:] + v[:, :-1]) * dt, 1)], 1)
        shp = (1, K) + (1,) * (u.dim() - 2)
        off = (torch.arange(K, device=u.device) * m).view(shp)
        vf, cf = v.reshape(-1), cum.reshape(-1)
        vj, vj1 = vf[off + j], vf[off + j + 1]
        inside = cf[off + j] + 0.5 * (2 * vj + (vj1 - vj) * f) * (ac - t[j])
        e = (a - 1).clamp(min=0)
        return inside + v[:, -1].view(shp) * e + 0.5 * w.view(shp) * e * e

    def smoothness(self):
        """(n+1, n+1) penalty on the jumps of the slope of :math:`\\psi`."""
        dt = (self._t[1:] - self._t[:-1]).double()
        S = torch.zeros(self.size, self.size, dtype=torch.float64)  # coefficients -> slopes of the cells
        for j in range(self.n):
            S[j, j] = 1.0 / dt[j]
            if j > 0:
                S[j, j - 1] = -1.0 / dt[j]
        S[self.n, self.n] = 1.0
        C = torch.diff(torch.eye(self.size, dtype=torch.float64), dim=0) @ S
        return C.T @ C

    def constraint(self, shape: str):
        """(T, ) with c = T delta, delta >= 0, for 'monotone' or 'convex' potentials."""
        if shape == "monotone":  # psi >= 0 on z > 0
            return torch.eye(self.size, dtype=torch.float64)
        if shape == "convex":  # psi non-decreasing
            T = torch.zeros(self.size, self.size, dtype=torch.float64)
            T[: self.n, : self.n] = torch.tril(torch.ones(self.n, self.n, dtype=torch.float64))
            T[self.n, self.n] = 1.0
            return T
        raise ValueError(f"unknown shape '{shape}'")


class Gauss:
    r"""
    Smooth :math:`\psi`: odd Gaussian bumps
    :math:`e^{-(u - \mu_a)^2 / 2 h_a^2} - e^{-(u + \mu_a)^2 / 2 h_a^2}` on geometric centres
    :math:`\mu_a \in [\epsilon, 1]`, plus a linear term :math:`u`.

    Parameters
    ----------
    n: int
        Number of bumps.
    eps: float
        First centre, relative to the scale.
    """

    def __init__(self, n: int = 24, eps: float = 1.5e-3):
        self.n, self.eps = n, eps
        self.size = n + 1
        r = (1.0 / eps) ** (1.0 / (n - 1))
        self._mu = torch.logspace(math.log10(eps), 0.0, n)
        self._h = 0.5 * self._mu * (r - 1.0 / r)

    def config(self) -> dict:
        return dict(name="Gauss", n=self.n, eps=self.eps)

    def features(self, u):
        mu, h = self._mu.to(u), self._h.to(u)
        uu = u[..., None]
        bumps = torch.exp(-0.5 * ((uu - mu) / h) ** 2) - torch.exp(-0.5 * ((uu + mu) / h) ** 2)
        return torch.cat([bumps, uu], -1)

    def derivatives(self, u):
        mu, h = self._mu.to(u), self._h.to(u)
        uu = u[..., None]
        e1, e2 = torch.exp(-0.5 * ((uu - mu) / h) ** 2), torch.exp(-0.5 * ((uu + mu) / h) ** 2)
        return torch.cat([(-(uu - mu) * e1 + (uu + mu) * e2) / h**2, torch.ones_like(uu)], -1)

    def psi(self, u, coef):
        K = coef.shape[0]
        return (self.features(u) * coef.to(u).view((1, K) + (1,) * (u.dim() - 2) + (-1,))).sum(-1)

    def phi(self, u, coef):
        mu, h = self._mu.to(u), self._h.to(u)
        uu, q = u[..., None], math.sqrt(2.0)
        B = (
            h
            * math.sqrt(0.5 * math.pi)
            * (torch.erf((uu - mu) / (q * h)) - torch.erf((uu + mu) / (q * h)) + 2 * torch.erf(mu / (q * h)))
        )
        B = torch.cat([B, 0.5 * uu * uu], -1)
        K = coef.shape[0]
        return (B * coef.to(u).view((1, K) + (1,) * (u.dim() - 2) + (-1,))).sum(-1)

    def smoothness(self):
        """(n+1, n+1) penalty on the second differences of the bump coefficients."""
        D2 = torch.diff(torch.eye(self.n, dtype=torch.float64), n=2, dim=0)
        R = torch.zeros(self.size, self.size, dtype=torch.float64)
        R[: self.n, : self.n] = D2.T @ D2
        return R

    def constraint(self, shape: str):
        if shape == "monotone":  # every bump and the linear term >= 0: psi >= 0 on z > 0
            return torch.eye(self.size, dtype=torch.float64)
        raise ValueError(f"shape '{shape}' needs the Hat basis")


def from_config(cfg: dict):
    """Basis from :py:meth:`Hat.config` / :py:meth:`Gauss.config`."""
    cfg = dict(cfg)
    return {"Hat": Hat, "Gauss": Gauss}[cfg.pop("name")](**cfg)
