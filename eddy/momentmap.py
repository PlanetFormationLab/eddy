# -*- coding: utf-8 -*-

import numpy as np

from .imagecube import imagecube
from .annulus import Annulus2D


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

    # -- ANNULUS FUNCTIONS -- #

    def get_annulus(self, r_min, r_max, phi_min=None, phi_max=None,
                    exclude_phi=False, abs_phi=False, x0=0.0, y0=0.0, inc=0.0,
                    PA=0.0, z0=0.0, psi=1.0, r_cavity=0.0, r_taper=np.inf,
                    q_taper=1.0, w_i=None, w_r=None, w_t=None, z_func=None,
                    shadowed=False, mask_frame='disk', user_mask=None,
                    beam_spacing=True, annulus_kwargs=None):
        """Return an :class:`Annulus2D` instance built from the pixels of the
        map that fall inside the requested annulus.

        Mirrors :meth:`linecube.get_annulus` but operates on a 2D map: each
        pixel contributes a single observed velocity (``self.data``) and an
        optional uncertainty (``self.error`` if present, e.g. on a
        :class:`rotationmap`). NaN values are dropped before returning.
        """
        mask = self.get_mask(r_min=r_min, r_max=r_max, phi_min=phi_min,
                             phi_max=phi_max, exclude_phi=exclude_phi,
                             abs_phi=abs_phi, x0=x0, y0=y0, inc=inc, PA=PA,
                             z0=z0, psi=psi, r_cavity=r_cavity,
                             r_taper=r_taper, q_taper=q_taper, w_i=w_i,
                             w_r=w_r, w_t=w_t, z_func=z_func,
                             shadowed=shadowed, mask_frame=mask_frame,
                             user_mask=user_mask)
        if mask.shape != self.data.shape:
            raise ValueError("mask is incorrect shape: {}.".format(mask.shape))
        mask = mask.flatten()

        vobs = self.data.flatten()[mask]
        verror = None
        if hasattr(self, 'error') and self.error is not None:
            error = np.asarray(self.error)
            if error.shape == self.data.shape:
                verror = error.flatten()[mask]

        rvals, pvals = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA, z0=z0,
                                        psi=psi, r_cavity=r_cavity,
                                        r_taper=r_taper, q_taper=q_taper,
                                        w_i=w_i, w_r=w_r, w_t=w_t,
                                        z_func=z_func, shadowed=shadowed,
                                        flatten=True)[:2]
        rvals, pvals = rvals[mask], pvals[mask]

        xsky, ysky = self.disk_coords(x0=0.0, y0=0.0, inc=0.0, PA=0.0,
                                      outframe='cartesian',
                                      flatten=True)[:2]
        xsky, ysky = xsky[mask], ysky[mask]

        iidx, jidx = np.meshgrid(np.arange(self.nypix), np.arange(self.nxpix))
        iidx, jidx = iidx.flatten()[mask], jidx.flatten()[mask]

        # Drop NaN/non-finite pixels in vobs (and verror, if present), then
        # thin to spatially independent samples. We do the thinning inline
        # rather than via ``_independent_samples`` so ``verror`` follows the
        # same permutation as ``vobs``.

        finite = np.isfinite(vobs)
        if verror is not None:
            finite &= np.isfinite(verror) & (verror > 0)
        vobs = vobs[finite]
        if verror is not None:
            verror = verror[finite]
        rvals = rvals[finite]
        pvals = pvals[finite]
        xsky = xsky[finite]
        ysky = ysky[finite]
        jidx = jidx[finite]
        iidx = iidx[finite]

        if beam_spacing and pvals.size:
            order = np.argsort(pvals)
            vobs = vobs[order]
            if verror is not None:
                verror = verror[order]
            rvals = rvals[order]
            pvals = pvals[order]
            xsky = xsky[order]
            ysky = ysky[order]
            jidx = jidx[order]
            iidx = iidx[order]
            sampling = float(beam_spacing) * self.bmaj
            sampling /= np.mean(rvals) * np.median(np.diff(pvals))
            sampling = np.floor(sampling).astype('int')
            if sampling > 1:
                start = np.random.randint(0, pvals.size)
                roll = np.r_[np.arange(start, pvals.size),
                             np.arange(0, start)][::sampling]
                vobs = vobs[roll]
                if verror is not None:
                    verror = verror[roll]
                rvals = rvals[roll]
                pvals = pvals[roll]
                xsky = xsky[roll]
                ysky = ysky[roll]
                jidx = jidx[roll]
                iidx = iidx[roll]

        annulus_kwargs = {} if annulus_kwargs is None else annulus_kwargs
        return Annulus2D(vobs=vobs, pvals=pvals, rvals=rvals, xsky=xsky,
                         ysky=ysky, jidx=jidx, iidx=iidx, inc=inc,
                         verror=verror, **annulus_kwargs)
