r"""
The fields-of-experts model.

.. math::

   \Phi(x) = \sum_k \sum_i \phi_k\big((f_k \star x)_i\big),
   \qquad
   \nabla \Phi(x) = \sum_k f_k^{\top} \psi_k(f_k \star x),

with :math:`\star` the correlation over valid positions (no padding) and :math:`f_k^{\top}` its
adjoint. Filters may share a potential: ``groups[k]`` is the potential of filter :math:`k`.
"""

import torch
import torch.nn.functional as F

from . import basis as ep_basis


def correlate(x, filters):
    """(B, H, W) images -> (B, K, H-s+1, W-s+1) responses."""
    return F.conv2d(x[:, None], filters[:, None])


def correlate_adjoint(v, filters):
    """(B, K, h, w) fields -> (B, H, W): the adjoint of :py:func:`correlate`, summed over filters."""
    return F.conv_transpose2d(v, filters[:, None])[:, 0]


def batched(x):
    """(H, W) or (B, H, W) -> (B, H, W), and a function undoing it."""
    if x.dim() == 2:
        return x[None], lambda u: u[0]
    return x, lambda u: u


class FoE(torch.nn.Module):
    r"""
    Fields of experts :math:`\Phi(x) = \sum_k \sum_i \phi_k((f_k \star x)_i)`.

    Built by :py:func:`~explicit_priors.fit`. Calling the model returns :math:`\Phi(x)`.

    Parameters
    ----------
    filters: Tensor
        (K, s, s) filters.
    basis: Hat | Gauss
        Basis of :math:`\psi_k = \phi_k'` (default: :py:class:`~explicit_priors.Hat`).
    groups: Tensor
        (K,) potential index of every filter (default: one potential per filter).
    scale: Tensor
        (G,) scale of every potential.
    coef: Tensor
        (G, n) coefficients of every potential.
    """

    def __init__(self, filters, basis=None, groups=None, scale=None, coef=None):
        super().__init__()
        filters = torch.as_tensor(filters, dtype=torch.float32)
        K = filters.shape[0]
        groups = torch.arange(K) if groups is None else torch.as_tensor(groups, dtype=torch.long)
        n_pot = int(groups.max()) + 1
        self.basis = ep_basis.Hat() if basis is None else basis
        self.register_buffer("filters", filters)
        self.register_buffer("groups", groups.to(filters.device))
        self.register_buffer("scale", torch.ones(n_pot) if scale is None else torch.as_tensor(scale).float())
        self.register_buffer(
            "coef", torch.zeros(n_pot, self.basis.size) if coef is None else torch.as_tensor(coef).float()
        )
        self.sigma, self.objective = None, None  # set by fit()

    @property
    def size(self) -> int:
        """Filter support s."""
        return self.filters.shape[-1]

    def responses(self, x):
        """(B, H, W) -> (B, K, H-s+1, W-s+1)."""
        return correlate(x, self.filters)

    def _per_filter(self):
        K = self.filters.shape[0]
        return self.scale[self.groups].view(1, K, 1, 1), self.coef[self.groups]

    def psi(self, z):
        r""":math:`\psi_k(z_k)` for (B, K, ...) responses."""
        s, c = self._per_filter()
        return self.basis.psi(z / s, c)

    def phi(self, z):
        r""":math:`\phi_k(z_k)` for (B, K, ...) responses."""
        s, c = self._per_filter()
        return s * self.basis.phi(z / s, c)

    def energy(self, x):
        r""":math:`\Phi(x)`, (B,) in float64, for (B, H, W) or (H, W) images."""
        xb, out = batched(x)
        return out(self.phi(self.responses(xb)).sum((1, 2, 3), dtype=torch.float64))

    forward = energy

    def grad(self, x):
        r""":math:`\nabla \Phi(x)`, same shape as x."""
        xb, out = batched(x)
        return out(correlate_adjoint(self.psi(self.responses(xb)), self.filters))

    def energy_grad(self, x):
        r""":math:`(\Phi(x), \nabla \Phi(x))` with one correlation."""
        xb, out = batched(x)
        z = self.responses(xb)
        e = self.phi(z).sum((1, 2, 3), dtype=torch.float64)
        return out(e), out(correlate_adjoint(self.psi(z), self.filters))

    def potentials(self, n: int = 401, extent: float = 1.0):
        r"""
        The potentials on a grid.

        Returns
        -------
        z: Tensor
            (G, n) grid of every potential, on :math:`[-\text{extent}\, s_g, \text{extent}\, s_g]`.
        phi: Tensor
            (G, n) values :math:`\phi_g(z)`.
        """
        G = self.coef.shape[0]
        u = torch.linspace(-extent, extent, n, device=self.coef.device).expand(G, n)
        z = u * self.scale[:, None]
        return z, self.scale[:, None] * self.basis.phi(u[None], self.coef)[0]

    def save(self, path):
        torch.save(
            dict(
                filters=self.filters.cpu(),
                groups=self.groups.cpu(),
                scale=self.scale.cpu(),
                coef=self.coef.cpu(),
                basis=self.basis.config(),
                sigma=self.sigma,
                objective=self.objective,
            ),
            path,
        )

    @classmethod
    def load(cls, path, map_location=None):
        d = torch.load(path, map_location=map_location, weights_only=True)
        m = cls(d["filters"], ep_basis.from_config(d["basis"]), d["groups"], d["scale"], d["coef"])
        m.sigma, m.objective = d["sigma"], d["objective"]
        return m

    def extra_repr(self) -> str:
        K, s = self.filters.shape[0], self.size
        return (
            f"filters={K}x{s}x{s}, potentials={self.coef.shape[0]}, basis={self.basis.config()['name']}, "
            f"objective={self.objective}, sigma={self.sigma}"
        )
