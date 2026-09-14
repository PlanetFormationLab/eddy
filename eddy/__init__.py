# Enable JAX double precision before any jax.numpy arrays are created in
# the submodules below. eddy operates on physical quantities (Keplerian
# velocities, beam-scale FFTs) where float32 magnitudes overflow and the
# precision loss is large compared to the typical signal of interest, so
# we match numpy's float64 default. This is a process-global setting; if
# you also use JAX outside of eddy and need float32 there, set
# ``jax.config.update('jax_enable_x64', False)`` after importing eddy.
import jax
jax.config.update('jax_enable_x64', True)

__version__ = "3.2.0"

from .imagecube import imagecube
from .momentmap import momentmap
from .rotationmap import rotationmap
from .linecube import linecube
from .annulus import annulus, Annulus, Annulus2D, Annulus3D
from .linecube import SpectralACF
from .structurefunction import (StructureFunction, StructureFunctionStack,
                                calculate_structure_function,
                                calculate_structure_function_stack,
                                calculate_structure_function_ensemble,
                                draw_polar_field, make_polar_grid)

__all__ = [
    "__version__",
    "imagecube",
    "momentmap",
    "rotationmap",
    "linecube",
    "annulus",
    "Annulus",
    "Annulus2D",
    "Annulus3D",
    "StructureFunction",
    "StructureFunctionStack",
    "SpectralACF",
    "calculate_structure_function",
    "calculate_structure_function_stack",
    "calculate_structure_function_ensemble",
    "draw_polar_field",
    "make_polar_grid",
]


def __getattr__(name):
    """Serve the structure-function names renamed in 3.2.0.

    Delegating to the submodule's own PEP 562 hook keeps the
    ``DeprecationWarning`` (and the eventual 4.0 removal) defined in one
    place, and avoids importing the old spellings eagerly here -- which
    would warn on every ``import eddy``.
    """
    from . import structurefunction
    if name in structurefunction._DEPRECATED_MODULE_NAMES:
        return getattr(structurefunction, name)
    raise AttributeError(
        "module {!r} has no attribute {!r}".format(__name__, name))
