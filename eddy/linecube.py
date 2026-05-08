# -*- coding: utf-8 -*-

import os
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from .imagecube import imagecube
from .annulus import annulus


class linecube(imagecube):
    """
    Read in a line cube and initialize the class.

    Args:
        path (str): Relative path to the rotation map you want to fit.
        FOV (Optional[float]): If specified, clip the data down to a
            square field of view with sides of `FOV` [arcsec].
        fill (Optional[float/None]): Value to fill any NaN values.
        velocity_range (Optional[list]): A velocity range in [m/s] to
            clip the data down to: ``[min_velo, max_velo]``.
    """

    def __init__(self, path, FOV=None, fill=0.0, velocity_range=None):
        super().__init__(path=path, FOV=FOV, fill=fill)
        if velocity_range is not None:
            self._clip_cube_velocity(*velocity_range)

    # -- 3D CUBE I/O & DIAGNOSTICS -- #

    def to_momentmap(self, method='zeroth', clip=None,
                     bettermoments_kwargs=None):
        """Collapse the spectral cube to a 2D moment map.

        Returns a :class:`momentmap` instance (or :class:`rotationmap` for
        ``method`` in ``{'first', 'quadratic'}``). When ``bettermoments`` is
        installed and ``method`` matches one of its method names, this
        method delegates to it.

        .. note::
            Stubbed for Phase 4.2; calling this currently raises
            ``NotImplementedError``.
        """
        raise NotImplementedError(
            "linecube.to_momentmap() is scheduled for Phase 4.2 of the "
            "refactor. Use bettermoments directly for now."
        )

    @property
    def rms(self):
        return self.estimate_cube_RMS()

    def _clip_cube_velocity(self, v_min=None, v_max=None):
        """Clip the cube to within ``vmin`` and ``vmax``."""
        if self.data.ndim == 2:
            raise ValueError("Attaced cube has no velocity axis.")
        v_min = self.velax[0] if v_min is None else v_min
        v_max = self.velax[-1] if v_max is None else v_max
        i = abs(self.velax - v_min).argmin()
        i += 1 if self.velax[i] < v_min else 0
        j = abs(self.velax - v_max).argmin()
        j -= 1 if self.velax[j] > v_max else 0
        self.velax = self.velax[i:j+1]
        self.data = self.data[i:j+1]

    def estimate_cube_RMS(self, N=10, r_in=0.0, r_out=1e10):
        """
        Estimate RMS of the cube based on first and last `N` channels and a
        circular area described by an inner and outer radius.

        Args:
            N (int): Number of edge channels to include.
            r_in (float): Inner edge of pixels to consider in [arcsec].
            r_out (float): Outer edge of pixels to consider in [arcsec].

        Returns:
            RMS (float): The RMS based on the requested pixel range.
        """
        r_dep = np.hypot(self.xaxis[None, :], self.yaxis[:, None])
        rmask = np.logical_and(r_dep >= r_in, r_dep <= r_out)
        rms = np.concatenate([self.data[:int(N)], self.data[-int(N):]])
        rms = np.where(rmask[None, :, :], rms, np.nan)
        return np.sqrt(np.nansum(rms**2) / np.sum(np.isfinite(rms)))

    def integrated_spectrum(self, x0=0.0, y0=0.0, inc=0.0, PA=0.0, r_min=None,
                            r_max=None):
        """
        Returns the integrated spectrum over a specified region.

        Args:
            x0 (Optional[float]): Right Ascension offset in [arcsec].
            y0 (Optional[float]): Declination offset in [arcsec].
            inc (Optional[float]): Disk inclination in [deg].
            PA (Optional[float]): Disk position angle in [deg].
            r_min (Optional[float]): Radius to integrate out from in [arcsec].
            r_max (Optional[float]): Radius to integrate out to in [arcsec].

        Returns:
            spectrum, uncertainty (array, array): Something about these.
        """
        rr = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA)[0]
        r_max = rr.max() if r_max is None else r_max
        r_min = 0.0 if r_min is None else r_min
        mask = np.logical_and(rr <= r_max, rr >= r_min)
        nbeams = np.where(mask, 1, 0).sum() / self.pix_per_beam
        spectrum = np.array([np.nansum(c[mask]) for c in self.data])
        spectrum *= self.beams_per_pix
        uncertainty = np.sqrt(nbeams) * self.estimate_cube_RMS()
        return spectrum, uncertainty

    def plot_maximum(self, ax=None, imshow_kwargs=None, return_fig=False):
        """
        Plot the maximum value along each spectrum.

        Args:
            ax (Optional[matplotlib axis]): Axis used for the plotting.
            imshow_kwargs (Optional[dict]): Kwargs to pass to
                ``matplotlib.imshow``.
            return_fig (Optional[bool]): Whether to return the figure instance.
                If an axis was provided, this will always be ``False``.

        Returns:
            fig (matplotlib figure): If ``return_fig=True``, will return the
                figure for continued plotting.
        """

        if ax is None:
            fig, ax = plt.subplots()
        else:
            return_fig = False

        dmax = np.nanmax(self.data, axis=0)
        vmax = np.percentile(dmax, [98])

        imshow_kwargs = {} if imshow_kwargs is None else imshow_kwargs
        imshow_kwargs['interpolation'] = 'nearest'
        imshow_kwargs['extent'] = self.extent
        imshow_kwargs['origin'] = 'lower'
        imshow_kwargs['cmap'] = imshow_kwargs.pop('cmap', 'turbo')
        imshow_kwargs['vmin'] = imshow_kwargs.pop('vmin', 0.0)
        imshow_kwargs['vmax'] = imshow_kwargs.pop('vmax', vmax)

        im = ax.imshow(dmax, **imshow_kwargs)
        cb = plt.colorbar(im, ax=ax, pad=0.03, extend='both')
        cb.set_label('Peak Intensity (Jy/beam)', rotation=270, labelpad=13)
        cb.minorticks_on()
        self._gentrify_plot(ax=ax)

        if return_fig:
            return fig

    def plot_spectrum(self, ax=None, x0=0.0, y0=0.0, inc=0.0, PA=0.0,
                      r_min=None, r_max=None, return_fig=False):
        """
        Plot the integrated spectrum.

        Args:
            x0 (Optional[float]): Right Ascension offset in [arcsec].
            y0 (Optional[float]): Declination offset in [arcsec].
            inc (Optional[float]): Disk inclination in [deg].
            PA (Optional[float]): Disk position angle in [deg].
            r_max (Optional[float]): Radius to integrate out to in [arcsec].
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            return_fig = False
        x = self.velax.copy() / 1e3
        y, dy = self.integrated_spectrum(x0, y0, inc, PA, r_min, r_max)
        ax.axhline(0.0, ls='--', lw=1.0, color='0.9', zorder=-9)
        ax.step(x, y, where='mid', lw=1.0, color='k')
        ax.errorbar(x, y, dy, fmt=' ', lw=1.0, color='k', zorder=-8)
        ax.set_xlabel("Velocity (km/s)")
        ax.set_ylabel("Integrated Flux (Jy)")
        ax.set_xlim(x[0], x[-1])
        ticks = np.diff(ax.xaxis.get_majorticklocs()).mean() / 5.0
        ax.xaxis.set_minor_locator(MultipleLocator(ticks))
        ticks = np.diff(ax.yaxis.get_majorticklocs()).mean() / 5.0
        ax.yaxis.set_minor_locator(MultipleLocator(ticks))

        if return_fig:
            return fig

    # -- ROTATION PROFILE FUNCTIONS -- #

    def get_velocity_profile(self, rbins=None, fit_method='GP', fit_vrad=False,
            fix_vlsr=None, x0=0.0, y0=0.0, inc=0.0, PA=0.0, z0=0.0, psi=1.0,
            r_cavity=0.0, r_taper=np.inf, q_taper=1.0, w_i=None, w_r=None,
            w_t=None, z_func=None, shadowed=False, phi_min=None, phi_max=None,
            exclude_phi=False, abs_phi=False, mask_frame='disk', user_mask=None,
            beam_spacing=True, niter=1, get_vlos_kwargs=None,
            weighted_average=True, return_samples=False, repeat_with_mask=0):
        """
        Returns the rotational and, optionally, radial velocity profiles under
        the assumption that the disk is azimuthally symmetric (at least across
        the regions extracted).

        Several different inference methods can be used through the
        ``fit_method`` argument:

            - ``'GP'``:  Models the aligned and stacked spectrum as a Gaussian
                         Process to remain agnostic about the underlying true
                         line profile. This is the default.
            - ``'dV'``:  Minimize the line width of the aligned and stacked
                         spectrum. Assumes the underyling profile is a
                         Gaussian.
            - ``'SNR'``: Maximizes the SNR of the aligned and stacked spectrum.
                         Assumes the underlying profile is a Gaussian.
            - ``'SHO'``: Fits the azimuthal dependence of the line centroids
                         with a simple harmonic oscillator. Requires a choice
                         of ``centroid_method`` to determine the line centers.

        By default, these methods are applied to annuli across the whole radial
        and azimuthal range of the disk. Masks can be adopted, either limiting
        or excluding azimuthal regions, or a user-defined mask can be provided.

        Boostrapping is possible (multiple iterations using random draws) to
        estimate the uncertainties on the velocities which may be
        underestimated with a single interation.

        Args:
            rbins (Optional[arr]): Array of bin edges of the annuli.
            fit_method (Optional[str]): Method used to infer the velocities.
            fit_vrad (Optional[bool]): Whether to include radial velocities in
                the fit.
            fix_vlsr (optional[bool]): Fix the systemic velocity to calculate
                the deprojected vertical velocities. Only available for
                `fit_method='SHO'`.
            x0 (Optional[float]): Source right ascension offset [arcsec].
            y0 (Optional[float]): Source declination offset [arcsec].
            inc (Optional[float]): Source inclination [degrees]. A positive
                inclination denotes a disk rotating clockwise on the sky, while
                a negative inclination represents a counter-clockwise rotation.
            PA (Optional[float]): Source position angle [degrees]. Measured
                between north and the red-shifted semi-major axis in an
                easterly direction.
            z0 (Optional[float]): Aspect ratio at 1" for the emission surface.
                To get the far side of the disk, make this number negative.
            psi (Optional[float]): Flaring angle for the emission surface.
            r_cavity (Optional[float]): Outer radius of a cavity. Within this
                region the emission surface is taken to be zero.
            w_i (Optional[float]): Warp inclination in [degrees] at the disk
                center.
            w_r (Optional[float]): Scale radius of the warp in [arcsec].
            w_t (Optional[float]): Angle of nodes of the warp in [degrees].
            z_func (Optional[function]): A function which provides
                :math:`z(r)`. Note that no checking will occur to make sure
                this is a valid function.
            shadowed (Optional[bool]): Whether to use the slower, but more
                robust, deprojection method for shadowed disks.
            phi_min (Optional[float]): Minimum polar angle of the segment of
                the annulus in [deg]. Note this is the polar angle, not the
                position angle.
            phi_max (Optional[float]): Maximum polar angle of the segment of
                the annulus in [deg]. Note this is the polar angle, not the
                position angle.
            exclude_phi (Optional[bool]): If ``True``, exclude the provided
                polar angle range rather than include it.
            abs_phi (Optional[bool]): If ``True``, take the absolute value of
                the polar angle such that it runs from 0 [deg] to 180 [deg].
            mask_frame (Optional[str]): Which frame to specify the mask in,
                either ``'disk'``, the default, or ``'sky'``.
            user_mask (Optional[array]): A user-specified mask to include. Must
                have the same shape as ``self.data``.
            beam_spacing (int): Sample pixels separated by roughly
            `beam_spacing * bmaj` in azimuthal distance.
            niter (Optional[int]): Number of iterations to run.
            get_vlos_kwargs=None,
            weighted_average (Optional[bool]): Whether to combine multiple
                iterations with a weighted average and standard deviation,
                ``weighted_average=True``, or the traditional mean and standard
                deviation, ``weighted_average=False``.
            return_samples (Optional[bool]): Whether to return the samples
                instead of combining them.
            repeat_with_mask (Optional[int]):

        Returns:
            samples (array): If ``return_samples=True``. The array of ``niter``
                samples of the velocity profiles.
            rvals, profile, uncertainty (array, array, array): If
                ``return_samples=False``. The arrays of radial locations and
                combined velocity and uncertainties.
        """

        if niter == 0:
            raise ValueError("`niter` must be >= 1.")

        # Single iteration.

        if niter == 1:
            return self._velocity_profile(rbins=rbins,
                                          fit_method=fit_method,
                                          fit_vrad=fit_vrad,
                                          fix_vlsr=fix_vlsr,
                                          x0=x0,
                                          y0=y0,
                                          inc=inc,
                                          PA=PA,
                                          z0=z0,
                                          psi=psi,
                                          r_cavity=r_cavity,
                                          r_taper=r_taper,
                                          q_taper=q_taper,
                                          w_i=w_i,
                                          w_r=w_r,
                                          w_t=w_t,
                                          z_func=z_func,
                                          shadowed=shadowed,
                                          phi_min=phi_min,
                                          phi_max=phi_max,
                                          exclude_phi=exclude_phi,
                                          abs_phi=abs_phi,
                                          mask_frame=mask_frame,
                                          user_mask=user_mask,
                                          beam_spacing=beam_spacing,
                                          get_vlos_kwargs=get_vlos_kwargs,
                                          repeat_with_mask=repeat_with_mask)

        # Multiple iterations.

        if beam_spacing is False:
            raise ValueError("niter must equal 1 when beam_spacing=False.")

        samples = [self._velocity_profile(rbins=rbins,
                                          fit_method=fit_method,
                                          fit_vrad=fit_vrad,
                                          fix_vlsr=fix_vlsr,
                                          x0=x0,
                                          y0=y0,
                                          inc=inc,
                                          PA=PA,
                                          z0=z0,
                                          psi=psi,
                                          r_cavity=r_cavity,
                                          r_taper=r_taper,
                                          q_taper=q_taper,
                                          w_i=w_i,
                                          w_r=w_r,
                                          w_t=w_t,
                                          z_func=z_func,
                                          shadowed=shadowed,
                                          phi_min=phi_min,
                                          phi_max=phi_max,
                                          exclude_phi=exclude_phi,
                                          abs_phi=abs_phi,
                                          mask_frame=mask_frame,
                                          user_mask=user_mask,
                                          beam_spacing=beam_spacing,
                                          get_vlos_kwargs=get_vlos_kwargs,
                                          repeat_with_mask=repeat_with_mask)
                   for _ in range(niter)]

        # Just return the samples if requested.

        if return_samples:
            return samples
        
        rpnts = samples[0][0]
        profiles = np.array([s[1] for s in samples])

        # Calculate weights, making sure they are finite with non-zero sum.

        if weighted_average:
            weights = [1.0 / s[2] for s in samples]
            weights = np.where(np.isfinite(weights), weights, 0.0)
        else:
            weights = np.ones(profiles.shape)

        if np.all(np.sum(weights, axis=0) == 0.0):
            weights = np.ones(profiles.shape)
        
        weights = np.where(np.isfinite(weights), weights, 1.0)
        weights += 1e-10 * np.random.randn(weights.size).reshape(weights.shape)
        
        M = np.sum(weights != 0.0, axis=0)

        # Weighted average.

        profile = np.average(profiles, weights=weights, axis=0)

        # Weighted standard deviation.

        uncertainty = weights * (profiles - profile[None, :, :])**2
        uncertainty = np.sum(uncertainty, axis=0)
        uncertainty /= (M - 1.0) / M * np.sum(weights, axis=0)
        uncertainty = np.sqrt(uncertainty)

        return rpnts, profile, uncertainty

    def _velocity_profile(self, rbins=None, fit_method='GP', fit_vrad=False,
            fix_vlsr=None, x0=0.0, y0=0.0, inc=0.0, PA=0.0, z0=0.0, psi=1.0,
            r_cavity=0.0,  r_taper=np.inf, q_taper=1.0, w_i=None, w_r=None,
            w_t=None, z_func=None, shadowed=False, phi_min=None, phi_max=None,
            exclude_phi=False, abs_phi=False, mask_frame='disk',
            user_mask=None, beam_spacing=True, get_vlos_kwargs=None,
            repeat_with_mask=0):
        """
        Returns the velocity (rotational and radial) profiles.

        Args:
            TBD

        Returns:
            TBD
        """

        # Define the radial binning.

        if rbins is None:
            rbins = np.arange(0, self.xaxis.max(), 0.25 * self.bmaj)
        rpnts = np.mean([rbins[1:], rbins[:-1]], axis=0)

        # Set up the kwargs for the fitting.

        kw = {} if get_vlos_kwargs is None else get_vlos_kwargs
        kw['fit_vrad'] = fit_vrad
        kw['fix_vlsr'] = fix_vlsr
        kw['fit_method'] = fit_method
        kw['repeat_with_mask'] = repeat_with_mask

        # Cycle through the annuli.

        profiles = []
        uncertainties = []
        for r_min, r_max in zip(rbins[:-1], rbins[1:]):
            annulus = self.get_annulus(r_min=r_min,
                                       r_max=r_max,
                                       phi_min=phi_min,
                                       phi_max=phi_max,
                                       exclude_phi=exclude_phi,
                                       abs_phi=abs_phi,
                                       x0=x0,
                                       y0=y0,
                                       inc=inc,
                                       PA=PA,
                                       z0=z0,
                                       psi=psi,
                                       r_cavity=r_cavity,
                                       r_taper=r_taper,
                                       q_taper=q_taper,
                                       w_i=w_i,
                                       w_r=w_r,
                                       w_t=w_t,
                                       z_func=z_func,
                                       shadowed=shadowed,
                                       mask_frame=mask_frame,
                                       user_mask=user_mask,
                                       beam_spacing=beam_spacing)

            output = annulus.get_vlos(**kw)
            profiles += [output[0]]
            uncertainties += [output[1]]

        # Make sure the returned arrays are in the (nparam, nrad) form.

        profiles = np.atleast_2d(profiles).T
        uncertainties = np.atleast_2d(uncertainties).T
        assert profiles.shape[0] == uncertainties.shape[0] == 3
        assert profiles.shape[1] == uncertainties.shape[1] == rpnts.size

        return rpnts, profiles, uncertainties

    # -- ANNULUS FUNCTIONS -- #

    def get_annulus(self, r_min, r_max, phi_min=None, phi_max=None,
            exclude_phi=False, abs_phi=False, x0=0.0, y0=0.0, inc=0.0, PA=0.0,
            z0=0.0, psi=1.0, r_cavity=0.0, r_taper=np.inf, q_taper=1.0,
            w_i=None, w_r=None, w_t=None, z_func=None, shadowed=False,
            mask_frame='disk', user_mask=None, beam_spacing=True,
            annulus_kwargs=None):
        """
        Returns an annulus instance.

        Args:
            r_min (float): Inner radius of the annulus in [arcsec].
            r_max (float): Outer radius of the annulus in [arcsec].
            phi_min (Optional[float]): Minimum polar angle in [X] for the
                annulus. ``phi`` is measured from the red-shifted major axis and
                increases in a clockwise direction, spanning ``-pi`` to ``+pi``.
            phi_max (Optional[float]): Maximum polar angle in [x] for the
                annulus. ``phi`` is measured from the red-shifted major axis and
                increases in a clockwise direction, spanning ``-pi`` to ``+pi``.
            exclude_phi (Optional[bool]): If ``True``, exclude the polar angle
                range rather than to include it.
            abs_phi (Optional[bool]): If ``True``, consider only the absolute
                values of ``phi`` in order to get a symmetric mask.
            x0 (Optional[float]): Source right ascension offset [arcsec].
            y0 (Optional[float]): Source declination offset [arcsec].
            inc (Optional[float]): Source inclination [degrees]. A positive
                inclination denotes a disk rotating clockwise on the sky, while
                a negative inclination represents a counter-clockwise rotation.
            PA (Optional[float]): Source position angle [degrees]. Measured
                between north and the red-shifted semi-major axis in an
                easterly direction.
            z0 (Optional[float]): Aspect ratio at 1" for the emission surface.
                To get the far side of the disk, make this number negative.
            psi (Optional[float]): Flaring angle for the emission surface.
            r_cavity (Optional[float]): Outer radius of a cavity. Within this
                region the emission surface is taken to be zero.
            r_taper (Optional[float]): Radius for tapered emission surface.
            q_taper (Optional[float]): Exponent for tapered emission surface.
            w_i: [coming soon]
            w_r: [coming soon]
            w_t: [coming soon]

        """

        # Calculate and flatten the mask.

        mask = self.get_mask(r_min=r_min,
                             r_max=r_max,
                             phi_min=phi_min,
                             phi_max=phi_max,
                             exclude_phi=exclude_phi,
                             abs_phi=abs_phi,
                             x0=x0,
                             y0=y0,
                             inc=inc,
                             PA=PA,
                             z0=z0,
                             psi=psi,
                             r_cavity=r_cavity,
                             r_taper=r_taper,
                             q_taper=q_taper,
                             w_i=w_i,
                             w_r=w_r,
                             w_t=w_t,
                             z_func=z_func,
                             shadowed=shadowed,
                             mask_frame=mask_frame,
                             user_mask=user_mask)
        if mask.shape != self.data[0].shape:
            raise ValueError("mask is incorrect shape: {}.".format(mask.shape))
        mask = mask.flatten()

        # Flatten the data and get the deprojected pixel coordinates.
        # We will record the on-sky pixels, their deprojected disk-frame polar
        # coordinates and the array indices.

        dvals = self.data.copy().reshape(self.data.shape[0], -1)
        dvals = dvals[:, mask].T

        rvals, pvals = self.disk_coords(x0=x0,
                                        y0=y0,
                                        inc=inc,
                                        PA=PA,
                                        z0=z0,
                                        psi=psi,
                                        r_cavity=r_cavity,
                                        r_taper=r_taper,
                                        q_taper=q_taper,
                                        w_i=w_i,
                                        w_r=w_r,
                                        w_t=w_t,
                                        z_func=z_func,
                                        shadowed=shadowed,
                                        flatten=True)[:2]
        rvals, pvals = rvals[mask], pvals[mask]

        xsky, ysky = self.disk_coords(x0=0.0,
                                      y0=0.0,
                                      inc=0.0,
                                      PA=0.0,
                                      outframe='cartesian',
                                      flatten=True)[:2]
        xsky, ysky = xsky[mask], ysky[mask]
        
        iidx, jidx = np.meshgrid(np.arange(self.nypix), np.arange(self.nxpix))
        iidx, jidx = iidx.flatten()[mask], jidx.flatten()[mask]

        # Thin down to spatially independent pixels.

        thinned = self._independent_samples(beam_spacing=beam_spacing,
                                            rvals=rvals,
                                            pvals=pvals,
                                            dvals=dvals,
                                            xsky=xsky,
                                            ysky=ysky,
                                            jidx=jidx,
                                            iidx=iidx)
        rvals, pvals, dvals, xsky, ysky, jidx, iidx = thinned

        # Return the annulus instance.

        annulus_kwargs = {} if annulus_kwargs is None else annulus_kwargs
        return annulus(spectra=dvals, pvals=pvals, velax=self.velax, inc=inc,
                       rvals=rvals, xsky=xsky, ysky=ysky, jidx=jidx, iidx=iidx,
                       **annulus_kwargs)

    # -- PLOTTING FUNCTIONS -- #

    def plot_mask(self, ax, r_min=None, r_max=None, exclude_r=False,
                  phi_min=None, phi_max=None, exclude_phi=False, abs_phi=False,
                  mask_frame='disk', mask=None, x0=0.0, y0=0.0, inc=0.0,
                  PA=0.0, z0=0.0, psi=1.0, r_cavity=None, r_taper=None,
                  q_taper=None, w_i=None, w_r=None, w_t=None, z_func=None,
                  mask_color='k', mask_alpha=0.5, contour_kwargs=None,
                  contourf_kwargs=None, shadowed=False):
        """
        Plot the boolean mask on the provided axis to check that it makes
        sense.

        Args:
            ax (matplotib axis instance): Axis to plot the mask.
            r_min (Optional[float]): Minimum midplane radius of the annulus in
                [arcsec]. Defaults to minimum deprojected radius.
            r_max (Optional[float]): Maximum midplane radius of the annulus in
                [arcsec]. Defaults to the maximum deprojected radius.
            exclude_r (Optional[bool]): If ``True``, exclude the provided
                radial range rather than include.
            PA_min (Optional[float]): Minimum polar angle of the segment of the
                annulus in [degrees]. Note this is the polar angle, not the
                position angle.
            PA_max (Optional[float]): Maximum polar angle of the segment of the
                annulus in [degrees]. Note this is the polar angle, not the
                position angle.
            exclude_PA (Optional[bool]): If ``True``, exclude the provided
                polar angle range rather than include it.
            abs_PA (Optional[bool]): If ``True``, take the absolute value of
                the polar angle such that it runs from 0 [deg] to 180 [deg].
            x0 (Optional[float]): Source center offset along the x-axis in
                [arcsec].
            y0 (Optional[float]): Source center offset along the y-axis in
                [arcsec].
            inc (Optional[float]): Inclination of the disk in [degrees].
            PA (Optional[float]): Position angle of the disk in [degrees],
                measured east-of-north towards the redshifted major axis.
            z0 (Optional[float]): Emission height in [arcsec] at a radius of
                1".
            psi (Optional[float]): Flaring angle of the emission surface.
            z_func (Optional[function]): A function which provides
                :math:`z(r)`. Note that no checking will occur to make sure
                this is a valid function.
            mask_color (Optional[str]): Color used for the mask lines.
            mask_alpha (Optional[float]): The alpha value of the filled contour
                of the masked regions. Setting ``mask_alpha=0.0`` will remove
                the filling.

            contour_kwargs (Optional[dict]): Kwargs to pass to contour for
                drawing the mask.

        Returns:
            ax : The matplotlib axis instance.
        """
        # Grab the mask.
        if mask is None:
            mask = self.get_mask(r_min=r_min, r_max=r_max, exclude_r=exclude_r,
                                 phi_min=phi_min, phi_max=phi_max,
                                 exclude_phi=exclude_phi, abs_phi=abs_phi,
                                 mask_frame=mask_frame, x0=x0, y0=y0, inc=inc,
                                 PA=PA, z0=z0, psi=psi, r_cavity=r_cavity,
                                 r_taper=r_taper, q_taper=q_taper, w_i=w_i,
                                 w_r=w_r, w_t=w_t, z_func=z_func,
                                 shadowed=shadowed)
        assert mask.shape[0] == self.yaxis.size, "Wrong y-axis shape for mask."
        assert mask.shape[1] == self.xaxis.size, "Wrong x-axis shape for mask."

        # Set the default plotting style.

        contour_kwargs = {} if contour_kwargs is None else contour_kwargs
        contour_kwargs['colors'] = contour_kwargs.pop('colors', mask_color)
        contour_kwargs['linewidths'] = contour_kwargs.pop('linewidths', 1.0)
        contour_kwargs['linestyles'] = contour_kwargs.pop('linestyles', '-')
        contourf_kwargs = {} if contourf_kwargs is None else contourf_kwargs
        contourf_kwargs['alpha'] = contourf_kwargs.pop('alpha', mask_alpha)
        contourf_kwargs['colors'] = contourf_kwargs.pop('colors', mask_color)

        # Plot the contour and return the figure.

        ax.contourf(self.xaxis, self.yaxis, mask, [-.5, .5], **contourf_kwargs)
        ax.contour(self.xaxis, self.yaxis, mask, 1, **contour_kwargs)

    # -- UTILITIES -- #

    def get_spectrum(self, coords, x0=0.0, y0=0.0, inc=0.0, PA=0.0,
                     z0=0.0, psi=1.0, z_func=None, frame='sky',
                     coord_type='cartesian', area=0.0, beam_weighting=False,
                     return_mask=False):
        """
        Return a spectrum at a position defined by a coordinates given either
        in sky-frame position (``frame='sky'``) or a disk-frame location
        (``frame='disk'``). The coordinates can be either in a cartesian system
        (``coord_type='cartesian'``) or cylindrical system
        (``coord_type='cylindrical'``).

        By default the returned spectrum is extracted at the pixel closest to
        the provided coordinates. If ``area`` is set to a positive value, then
        a beam-shaped area is averaged over, where ``area`` sets the size of
        this region in number of beams. For example ``area=2.0`` will result
        in an average over an area twice the size of the beam.

        If an area is averaged over, you can also weight the pixels by the beam
        response with ``beam_weighting=True``. This will reduce the weight of
        pixels that are further away from the beam center.

         Finally, to check that you're extracting what you think you are, you
         can return the mask (and weights) used for the extraction with
         ``return_mask=True``. Note that if ``beam_weighting=False`` then all
         ``weights`` will be 1.

         TODO: Check that the returned uncertainties are reasonable.

        Args:
            coords (tuple): The coordinates from where you want to extract a
                spectrum. Must be a length 2 tuple.
            x0 (Optional[float]): RA offset in [arcsec].
            y0 (Optional[float]): Dec offset in [arcsec].
            inc (Optional[float]): Inclination of source in [deg]. Only
                required for ``frame='disk'``.
            PA (Optional[float]): Position angle of source in [deg]. Only
                required for ``frame='disk'``.
            frame (Optional[str]): The frame that the ``coords`` are given.
                Either ``'disk'`` or ``'sky'``.
            coord_type (Optional[str]): The type of coordinates given, either
                ``'cartesian'`` or ``'cylindrical'``.
            area (Optional[float]): The area to average over in units of the
                beam area. Note that this take into account the beam aspect
                ratio and position angle. For a single pixel extraction use
                ``area=0.0``.
            beam_weighting (Optional[bool]): Whether to use the beam response
                function to weight the averaging of the spectrum.
            return_mask (Optional[bool]): Whether to return the mask and
                weights used to extract the spectrum.

        Retuns (if ``return_mask=False``):
            x, y, dy (arrays): The velocity axis, extracted spectrum and
            associated uncertainties.
        (if ``return_mask=True``):
            mask, weights (arrays): Arrays of the mask used to extract the
            spectrum and the weighted used for the averaging.
        """

        # TODO:
        #   1 - Check if it's three coordinates, or just two.
        #   2 - Check which frame it's in.

        # Convert the input coordinate into on-sky cartesian coordinates
        # relative to the center of the image.

        if frame.lower() == 'sky':
            if inc != 0.0 or PA != 0.0:
                message = "WARNING: You shouldn't need to specify `inc` or "
                message += "`PA` when using `frame='sky'`."
                print(message)
            c1 = np.squeeze(coords[0])
            c2 = np.squeeze(coords[1])
            if coord_type.lower() == 'cartesian':
                x, y = c1 + x0, c2 + y0
            elif coord_type.lower() == 'cylindrical':
                x = x0 + c1 * np.cos(c2 - np.radians(90.0))
                y = y0 - c1 * np.sin(c2 - np.radians(90.0))
        elif frame.lower() == 'disk':
            x, y = self.disk_to_sky(coords=coords, coord_type=coord_type,
                                    inc=inc, PA=PA, x0=x0, y0=y0)
        assert x.size == y.size == 1

        # Define the area to average over.

        if area == 0.0:
            x_pix = abs(self.xaxis - x).argmin()
            y_pix = abs(self.yaxis - y).argmin()
            mask = np.zeros(self.data[0].shape)
            weights = np.zeros(mask.shape)
            mask[y_pix, x_pix] = 1
            weights[y_pix, x_pix] = 1
        elif area > 0.0:
            mask = self._beam_mask(x, y, stretch=area)
            weights = self._beam_mask(x, y, stretch=area, response=True)
        else:
            raise ValueError("`area` must be a non-negative value.")
        weights = weights if beam_weighting else mask

        # If requested, return the mask and the weighting instead.

        if return_mask:
            return mask, weights

        # Otherwise, extract the spectrum and average it.

        y = [np.average(c * mask, weights=weights) for c in self.data]
        dy = max(1.0, mask.sum() * self.beams_per_pix)**-0.5 * self.rms
        return self.velax, np.array(y), np.array([dy for _ in y])
