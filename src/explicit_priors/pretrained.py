"""
Priors shipped with the package: MAP priors on 485 analytic filters (31 x 31) with monotone potentials, fitted in
closed form on the 400 BSD training images (8 Gauss-Newton steps). Use them with :py:func:`~explicit_priors.denoise`
at their noise level, or as regularisers for inverse problems.

====================  ==========
name                  noise std
====================  ==========
``"map_0.1"``         0.1
====================  ==========
"""

from pathlib import Path

from .foe import FoE

WEIGHTS = Path(__file__).resolve().parent / "weights"


def available() -> list:
    """Names of the shipped priors."""
    return sorted(p.stem for p in WEIGHTS.glob("*.pt"))


def pretrained(name: str, map_location=None) -> FoE:
    """
    A shipped prior.

    Parameters
    ----------
    name: str
        One of :py:func:`available`, e.g. ``"map_0.1"``.
    map_location: str
        Device to load to.

    Returns
    -------
    prior: FoE
    """
    path = WEIGHTS / f"{name}.pt"
    if not path.exists():
        raise ValueError(f"unknown prior '{name}', available: {', '.join(available())}")
    return FoE.load(path, map_location=map_location)
