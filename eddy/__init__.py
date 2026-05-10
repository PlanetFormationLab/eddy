# Enable JAX double precision before any jax.numpy arrays are created in
# the submodules below. eddy operates on physical quantities (Keplerian
# velocities, beam-scale FFTs) where float32 magnitudes overflow and the
# precision loss is large compared to the typical signal of interest, so
# we match numpy's float64 default. This is a process-global setting; if
# you also use JAX outside of eddy and need float32 there, set
# ``jax.config.update('jax_enable_x64', False)`` after importing eddy.
import jax
jax.config.update('jax_enable_x64', True)

__version__ = "3.0.0"

from .imagecube import imagecube
from .momentmap import momentmap
from .rotationmap import rotationmap
from .linecube import linecube
from .annulus import annulus, Annulus, Annulus2D, Annulus3D

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
]
