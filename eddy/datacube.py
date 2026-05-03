# -*- coding: utf-8 -*-

# Backward-compatibility shim. The previous ``datacube`` class has been split
# into :class:`imagecube` (the shared base), :class:`momentmap` (2D maps), and
# :class:`linecube` (3D spectral cubes). For new code prefer those classes
# directly.

from .imagecube import imagecube as datacube

__all__ = ["datacube"]
