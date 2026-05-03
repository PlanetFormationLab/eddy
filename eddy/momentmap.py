# -*- coding: utf-8 -*-

import warnings

from .imagecube import imagecube

warnings.filterwarnings("ignore")


class momentmap(imagecube):
    """
    A ``momentmap`` instance for working with 2D moment maps and integrated
    intensity images. This is the natural base class for any analysis that
    operates on a 2D image (peak intensity, line width, velocity centroid)
    rather than a full spectral cube. For Keplerian fitting of velocity maps,
    use the :class:`rotationmap` subclass.

    Args:
        path (str): Path to the moment map to load.
        FOV (Optional[float]): If specified, clip the data down to a
            square field of view with sides of `FOV` [arcsec].
        fill (Optional[float]): Replace all ``NaN`` values with this value.
        force_center (Optional[bool]): If ``True`` define the spatial axes such
            that they describe offset from the array center in [arcsec]. This
            is useful if the FITS header does not contain axis information.
    """

    def __init__(self, path, FOV=None, fill=None, force_center=False):
        super().__init__(path=path, FOV=FOV, fill=fill,
                         force_center=force_center)
        if self.data.ndim != 2:
            raise ValueError(
                "momentmap expects 2D data, got {}D data with shape {}. "
                "Use linecube for 3D spectral cubes."
                .format(self.data.ndim, self.data.shape)
            )

    def clip_cube_spatial(self, FOV):
        """Apply a new FOV to the map, also clipping any associated error and
        mask arrays attached to the instance."""
        if FOV > (self.xaxis.max() - self.xaxis.min()):
            raise ValueError("Cannot apply a larger FOV.")
        xa, xb, ya, yb = self._clip_cube_spatial(FOV / 2.0, False, True)
        self.data = self.data[ya:yb, xa:xb]
        if hasattr(self, 'error') and self.error is not None:
            self.error = self.error[ya:yb, xa:xb]
        if hasattr(self, 'mask') and self.mask is not None:
            self.mask = self.mask[ya:yb, xa:xb]
        self.xaxis = self.xaxis[xa:xb]
        self.yaxis = self.yaxis[ya:yb]
