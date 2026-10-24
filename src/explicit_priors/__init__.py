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
from .foe import FoE as FoE
