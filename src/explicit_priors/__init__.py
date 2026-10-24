import importlib.metadata

try:
    __version__ = importlib.metadata.version("explicit_priors")
except importlib.metadata.PackageNotFoundError:  # source tree, not installed
    __version__ = "unknown"

from . import filters as filters
from .basis import (
    Gauss as Gauss,
    Hat as Hat,
)
from .denoise import (
    denoise as denoise,
    fista as fista,
    map_estimate as map_estimate,
    one_step as one_step,
)
from .fit import (
    fit as fit,
    normal_equations as normal_equations,
)
from .foe import FoE as FoE
from .learn import learn_filters as learn_filters
