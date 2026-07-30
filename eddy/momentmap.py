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

    # -- STRUCTURE FUNCTION -- #

    def compute_structure_function(self, x0=0.0, y0=0.0, inc=0.0, PA=0.0,
                                   z0=None, psi=None, r_taper=None,
                                   q_taper=1.0, r_cavity=0.0, z_func=None,
                                   shadowed=False, rgrid=None, tgrid=None,
                                   griddata_kwargs=None,
                                   max_lag_r=None, max_lag_phi=None,
                                   ref_r=None, ref_band=0.0,
                                   n_bins=50, log_spaced=False,
                                   symmetrize=True):
        """Compute the 2D second-order structure function of this map on
        a polar (r, phi)-deprojected grid.

        The map is deprojected via :meth:`imagecube.polar_deprojection`
        (so all of the standard geometry kwargs apply), and the resulting
        regular grid is fed to the numba kernel in
        :mod:`eddy.structurefunction`. The result is a
        :class:`~eddy.structurefunction.StructureFunction2D` whose two
        lag axes are radial lag in [arcsec] and azimuthal lag in [deg].

        Args:
            x0, y0 (Optional[float]): Source offsets [arcsec].
            inc, PA (Optional[float]): Source inclination / position
                angle [deg]. See :meth:`imagecube.disk_coords` for the
                sign convention.
            z0, psi, r_taper, q_taper, r_cavity, z_func, shadowed
                (Optional): Forwarded to :meth:`imagecube.polar_deprojection`
                to define the emission surface.
            rgrid, tgrid (Optional[ndarray]): Radial grid in [arcsec]
                and azimuthal grid in [rad]. Defaults to the
                :meth:`imagecube.polar_deprojection` defaults.
            griddata_kwargs (Optional[dict]): Forwarded to
                ``scipy.interpolate.griddata`` inside the deprojection.
            max_lag_r (Optional[float]): Maximum radial lag [arcsec].
                Defaults to half the radial span of ``rgrid``.
            max_lag_phi (Optional[float]): Maximum azimuthal lag [deg].
                Defaults to half the azimuthal span of ``tgrid``.
            ref_r (Optional[float]): Reference annulus radius [arcsec].
                If provided, restricts base rows of the structure
                function to a band of half-width ``ref_band`` around
                this radius. If ``None``, averages globally.
            ref_band (float): Half-width of the reference annulus
                [arcsec]. ``0`` selects a single radial bin.
            n_bins (int): Number of radial bins for the azimuthal
                average.
            log_spaced (bool): If ``True``, log-spaced radial bins.
            symmetrize (bool): Only relevant when ``ref_r`` is set. If
                ``True`` (default), the outward (``+l_r``) and inward
                (``-l_r``) halves of the radial-lag axis are combined
                by a pair-count-weighted average — i.e. ``S_2`` becomes
                a direction-agnostic estimator. If ``False``, both
                halves are returned untouched, so the user can inspect
                the inward / outward asymmetry directly.

        Returns:
            :class:`~eddy.structurefunction.StructureFunction2D`
        """
        rgrid_out, tgrid_out, gridded, dr, dphi_deg = (
            self._structure_function_polar_grid(
                x0=x0, y0=y0, inc=inc, PA=PA,
                z0=z0, psi=psi, r_taper=r_taper, q_taper=q_taper,
                r_cavity=r_cavity, z_func=z_func, shadowed=shadowed,
                rgrid=rgrid, tgrid=tgrid, griddata_kwargs=griddata_kwargs,
            )
        )
        return self._structure_function_from_grid(
            rgrid_out, tgrid_out, gridded, dr, dphi_deg,
            max_lag_r=max_lag_r, max_lag_phi=max_lag_phi,
            ref_r=ref_r, ref_band=ref_band,
            n_bins=n_bins, log_spaced=log_spaced,
            symmetrize=symmetrize,
        )

    def compute_structure_function_stack(self, ref_rs, ref_band=0.0,
                                         x0=0.0, y0=0.0, inc=0.0, PA=0.0,
                                         z0=None, psi=None, r_taper=None,
                                         q_taper=1.0, r_cavity=0.0,
                                         z_func=None, shadowed=False,
                                         rgrid=None, tgrid=None,
                                         griddata_kwargs=None,
                                         max_lag_r=None, max_lag_phi=None,
                                         n_bins=50, log_spaced=False,
                                         symmetrize=True):
        """Compute the structure function at a sequence of reference radii.

        The polar deprojection is performed once and shared across all
        ``ref_r`` values; only the (much cheaper) kernel call runs N times.
        Use this when computing S_2 at many reference radii so the
        deprojection cost is paid only once. All other kwargs match
        :meth:`compute_structure_function`.

        Args:
            ref_rs (sequence of float): Reference annulus radii [arcsec].
            ref_band (float): Half-width of each reference annulus
                [arcsec]. Shared across all radii.

        Returns:
            :class:`~eddy.structurefunction.StructureFunction2DStack`
        """
        from .structurefunction import StructureFunction2DStack

        ref_rs = np.asarray(ref_rs, dtype=float)
        if ref_rs.ndim != 1 or ref_rs.size == 0:
            raise ValueError("ref_rs must be a non-empty 1D sequence.")

        rgrid_out, tgrid_out, gridded, dr, dphi_deg = (
            self._structure_function_polar_grid(
                x0=x0, y0=y0, inc=inc, PA=PA,
                z0=z0, psi=psi, r_taper=r_taper, q_taper=q_taper,
                r_cavity=r_cavity, z_func=z_func, shadowed=shadowed,
                rgrid=rgrid, tgrid=tgrid, griddata_kwargs=griddata_kwargs,
            )
        )

        results = [
            self._structure_function_from_grid(
                rgrid_out, tgrid_out, gridded, dr, dphi_deg,
                max_lag_r=max_lag_r, max_lag_phi=max_lag_phi,
                ref_r=float(r0), ref_band=ref_band,
                n_bins=n_bins, log_spaced=log_spaced,
                symmetrize=symmetrize,
            )
            for r0 in ref_rs
        ]
        return StructureFunction2DStack(
            ref_rs=ref_rs, ref_band=float(ref_band), results=results,
            x_grid=rgrid_out, y_grid=tgrid_out, gridded=gridded,
        )

    # -- STRUCTURE FUNCTION INTERNALS -- #

    def _structure_function_polar_grid(self, x0=0.0, y0=0.0, inc=0.0,
                                       PA=0.0, z0=None, psi=None,
                                       r_taper=None, q_taper=1.0,
                                       r_cavity=0.0, z_func=None,
                                       shadowed=False, rgrid=None,
                                       tgrid=None, griddata_kwargs=None):
        """Deproject ``self.data`` onto a polar grid for structure-function
        analysis. Returns ``(rgrid, tgrid, gridded, dr, dphi_deg)`` where
        ``gridded`` is shaped ``(N_r, N_phi)`` — radius is axis 0 so
        ``ref_r`` selects a base row in the kernel.
        """
        rgrid_out, tgrid_out, gridded = self.polar_deprojection(
            self.data, x0=x0, y0=y0, inc=inc, PA=PA,
            z0=z0, psi=psi, r_taper=r_taper, q_taper=q_taper,
            r_cavity=r_cavity, z_func=z_func, shadowed=shadowed,
            rgrid=rgrid, tgrid=tgrid, griddata_kwargs=griddata_kwargs,
        )
        gridded = np.asarray(gridded).T
        for name, axis in (("rgrid", rgrid_out), ("tgrid", tgrid_out)):
            d = np.diff(axis)
            if d.size == 0 or not np.allclose(d, d[0]):
                raise ValueError(
                    "{} must be uniformly spaced for the structure function "
                    "(lag axes are scaled by a single dr/dphi). Pass a linear "
                    "grid such as np.linspace / np.arange.".format(name))
        dr = float(np.diff(rgrid_out)[0])
        dphi_deg = float(np.degrees(np.diff(tgrid_out)[0]))
        return rgrid_out, tgrid_out, gridded, dr, dphi_deg

    @staticmethod
    def _structure_function_from_grid(rgrid_out, tgrid_out, gridded,
                                      dr, dphi_deg, max_lag_r, max_lag_phi,
                                      ref_r, ref_band, n_bins, log_spaced,
                                      symmetrize=True):
        """Run the structure-function kernel on an already-deprojected
        polar grid and build a :class:`StructureFunction2D`.
        """
        from .structurefunction import StructureFunction2D

        max_lag_x = (None if max_lag_r is None
                     else max(1, int(round(max_lag_r / dr))))
        max_lag_y = (None if max_lag_phi is None
                     else max(1, int(round(max_lag_phi / dphi_deg))))
        ref_band_idx = int(round(ref_band / dr)) if ref_band > 0 else 0
        ref_i_idx = (-1 if ref_r is None
                     else int(np.argmin(np.abs(rgrid_out - ref_r))))

        # dx is arcsec and dy is degrees on a polar grid, so the
        # sqrt(lx^2 + ly^2) isotropic bins mix incommensurate units.
        # Pass S2_i=None to suppress the meaningless azimuthal average.
        return StructureFunction2D.from_array(
            gridded, dx=dr, dy=dphi_deg,
            max_lag_x=max_lag_x, max_lag_y=max_lag_y,
            ref_i=ref_i_idx, ref_band=ref_band_idx,
            n_bins=n_bins, log_spaced=log_spaced, symmetrize=symmetrize,
            S2_i=None,
            x_grid=rgrid_out, y_grid=tgrid_out, gridded=gridded,
            ref=(None if ref_r is None else float(rgrid_out[ref_i_idx])),
            x_label="radial lag [arcsec]",
            y_label="azimuthal lag [deg]",
            azimuthal_axis="y",
        )
