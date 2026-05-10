# -*- coding: utf-8 -*-

import zeus
import time
import emcee
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from .helper_functions import (plot_walkers, plot_corner, random_p0,
                               _NumpyroSampler)
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit
from scipy.optimize import minimize

import jax
import jax.numpy as jnp
from tinygp import GaussianProcess, kernels

try:
    import numpyro  # noqa: F401
    _HAS_NUMPYRO = True
except ImportError:
    _HAS_NUMPYRO = False

__all__ = ['annulus', 'Annulus', 'Annulus2D', 'Annulus3D']


class Annulus(object):
    """
    Base class for an annulus of pixels at a given disk-frame radius. Stores
    the shared geometry (radial position, polar angle, sky coordinates, source
    inclination, original pixel indices) and exposes a common
    :func:`get_vlos` interface that subclasses must implement.

    The 2D variant :class:`Annulus2D` holds one observed velocity per pixel
    (e.g. extracted from a moment map) while the 3D variant
    :class:`Annulus3D` holds a full spectrum per pixel (extracted from a line
    cube).

    Args:
        pvals (ndarray): Polar angles in [rad] of each pixel. Range -pi to pi.
        rvals (ndarray): Radial position in [arcsec] of each pixel.
        xsky (ndarray): On-sky x-offset in [arcsec] of each pixel.
        ysky (ndarray): On-sky y-offset in [arcsec] of each pixel.
        jidx (ndarray): j-index of the original data array (y-axis).
        iidx (ndarray): i-index of the original data array (x-axis).
        inc (float): Inclination of the disk in [deg]. A positive inclination
            specifies a clockwise rotating disk.
    """

    def __init__(self, pvals, rvals, xsky, ysky, jidx, iidx, inc):
        if inc == 0.0:
            raise ValueError("Disk inclination must be non-zero.")
        self.theta = np.asarray(pvals)
        self.rvals = np.asarray(rvals)
        self.xsky = np.asarray(xsky)
        self.ysky = np.asarray(ysky)
        self.jidx = np.asarray(jidx)
        self.iidx = np.asarray(iidx)
        self.inc = inc
        self.inc_rad = np.radians(inc)
        self.sini = np.sin(self.inc_rad)
        self.cosi = np.cos(self.inc_rad)
        self.rotation = 'clockwise' if inc > 0 else 'anticlockwise'
        self.theta_deg = np.degrees(self.theta)

    def get_vlos(self, *args, **kwargs):
        """Infer the line-of-sight velocity components. Implemented by
        subclasses (:class:`Annulus2D`, :class:`Annulus3D`)."""
        raise NotImplementedError(
            "get_vlos() must be implemented by an Annulus subclass."
        )


class Annulus2D(Annulus):
    """
    A 2D annulus of pixel velocities extracted from a moment map (e.g.,
    a velocity map). Each entry in ``vobs`` is the line-of-sight velocity at
    one pixel within the annulus.

    Args:
        vobs (ndarray): Observed line-of-sight velocities in [m/s] at each
            pixel in the annulus.
        pvals (ndarray): Polar angles in [rad] of each pixel.
        rvals (ndarray): Radial position in [arcsec] of each pixel.
        xsky (ndarray): On-sky x-offset in [arcsec] of each pixel.
        ysky (ndarray): On-sky y-offset in [arcsec] of each pixel.
        jidx (ndarray): j-index of the original data array (y-axis).
        iidx (ndarray): i-index of the original data array (x-axis).
        inc (float): Inclination of the disk in [deg].
        verror (Optional[ndarray]): Per-pixel uncertainty on ``vobs`` in
            [m/s].
    """

    def __init__(self, vobs, pvals, rvals, xsky, ysky, jidx, iidx, inc,
                 verror=None):
        super().__init__(pvals=pvals, rvals=rvals, xsky=xsky, ysky=ysky,
                         jidx=jidx, iidx=iidx, inc=inc)
        self.vobs = np.asarray(vobs)
        self.verror = None if verror is None else np.asarray(verror)
        if self.vobs.shape != self.theta.shape:
            raise ValueError("Mismatch between vobs and pvals shapes.")

    def get_vlos(self, fit_method='SHO', fit_vrad=True, fix_vlsr=None,
                 optimize_kwargs=None):
        """
        Fit a simple-harmonic-oscillator model to the observed pixel
        velocities. Mirrors :meth:`Annulus3D.get_vlos_SHO`: the fit is
        performed on the projected coefficients ``A``, ``B``, ``C`` of
        ``vobs = A*cos(theta) + B*sin(theta) + C`` (or ``A*cos(theta) + C``
        when ``fit_vrad=False``), then deprojected via the disk inclination.

        With ``A = vrot * |sin(i)|`` and ``B = -vrad * sin(i)``, positive
        ``vrad`` corresponds to motion *away* from the star.

        Args:
            fit_method (Optional[str]): Only ``'SHO'`` is supported for 2D.
            fit_vrad (Optional[bool]): If ``True``, include a radial velocity
                component in the fit. Default ``True``.
            fix_vlsr (Optional[float]): If provided, treat the constant offset
                as ``vlsr - vz * cos(i)`` and report ``vz`` instead of
                ``vlsr``.
            optimize_kwargs (Optional[dict]): Extra kwargs forwarded to
                ``scipy.optimize.curve_fit``.

        Returns:
            popt, cvar (ndarray, ndarray): Best-fit deprojected velocities
                and their uncertainties. Length 2 if ``fit_vrad=False``
                (``[vrot, C-or-vz]``), length 3 if ``fit_vrad=True``
                (``[vrot, vrad, C-or-vz]``).
        """
        if fit_method.upper() != 'SHO':
            raise NotImplementedError(
                "Annulus2D currently only supports fit_method='SHO'."
            )

        from .helper_functions import SHO, SHO_double

        finite = np.isfinite(self.vobs)
        if self.verror is not None:
            finite &= np.isfinite(self.verror) & (self.verror > 0)
        theta = self.theta[finite]
        vobs = self.vobs[finite]
        sigma = None if self.verror is None else self.verror[finite]

        A = 0.5 * (vobs.max() - vobs.min()) if vobs.size else 0.0
        B = 0.0
        C = vobs.mean() if vobs.size else 0.0
        p0 = [A, C] if not fit_vrad else [A, B, C]

        kw = {} if optimize_kwargs is None else dict(optimize_kwargs)
        kw['p0'] = p0
        kw['sigma'] = sigma
        kw['absolute_sigma'] = sigma is not None
        kw['maxfev'] = kw.pop('maxfev', 10000)

        try:
            popt, pcov = curve_fit(SHO_double if fit_vrad else SHO,
                                   theta, vobs, **kw)
            cvar = np.sqrt(np.diag(pcov))
        except (TypeError, ValueError, RuntimeError):
            popt = np.full(len(p0), np.nan)
            cvar = np.full(len(p0), np.nan)

        popt[0] /= abs(self.sini)
        cvar[0] /= abs(self.sini)
        if fit_vrad:
            popt[1] /= -self.sini
            cvar[1] /= abs(self.sini)
        if fix_vlsr is not None:
            popt[-1] = (fix_vlsr - popt[-1]) / self.cosi
            cvar[-1] /= abs(self.cosi)
        return popt, cvar


class annulus(Annulus):
    """
    A class containing an annulus of spectra with their associated polar angles
    measured east of north from the redshifted major axis. These range from -pi
    to +pi. It will also store the inclination of the disk which will
    additionally define the rotation direction of the disk. A positive
    inclination will define clockwise (west of north) rotation, while a negative
    inclination will specify anti-clockwise direction.

    Args:
        spectra (ndarray): Array of shape ``[N, M]`` of spectra to shift and
            fit, where ``N`` is the number of spectra and ``M`` is the length
            of the velocity axis.
        pvals (ndarray): Polar angles in [rad] of each of the spectra.
        velax (ndarray): Velocity axis in [m/s] of the spectra.
        inc (float): Inclination of the disk in [deg]. A positive inclination
            specifies a clockwise rotating disk.
        rvals (ndarray): Radial position in [arcsec] of each pixel.
        xsky (ndarray): On-sky x-offset in [arcsec] of each pixel.
        ysky (ndarray): On-sky y-offset in [arcsec] of each pixel.
        jidx (ndarray): j-index of the original data array (y-axis).
        iidx (ndarray): i-index of the original data array (x-axis).
        remove_empty (optional[bool]): Remove empty spectra.
        sort_spectra (optional[bool]): Sorted the spectra into increasing
            ``theta``.
    """

    def __init__(self, spectra, pvals, velax, inc, rvals, xsky, ysky, jidx,
                 iidx, remove_empty=True, sort_spectra=True):

        # Read in the spectra and populate variables.

        self.theta = pvals
        self.rvals = rvals
        self.xsky = xsky
        self.ysky = ysky
        self.jidx = jidx
        self.iidx = iidx
        self.spectra = spectra
        if inc == 0.0:
            raise ValueError("Disk inclination must be non-zero.")
        else:
            self.inc = inc

        # Estimate the RMS.

        self.rms = self._estimate_RMS()

        # Sort the spectra with increasing polar angle.

        if sort_spectra:
            idxs = np.argsort(self.theta)
            self.spectra = self.spectra[idxs]
            self.theta = self.theta[idxs]
            self.rvals = self.rvals[idxs]
            self.xsky = self.xsky[idxs]
            self.ysky = self.ysky[idxs]
            self.jidx = self.jidx[idxs]
            self.iidx = self.iidx[idxs]

        # Remove empty pixels.

        if remove_empty:
            idxa = np.sum(self.spectra, axis=-1) != 0.0
            idxb = np.std(self.spectra, axis=-1) != 0.0
            idxs = idxa & idxb
            self.theta = self.theta[idxs]
            self.spectra = self.spectra[idxs]
            self.rvals = self.rvals[idxs]
            self.xsky = self.xsky[idxs]
            self.ysky = self.ysky[idxs]
            self.jidx = self.jidx[idxs]
            self.iidx = self.iidx[idxs]
        if self.theta.size < 1:
            raise ValueError("No finite spectra. Check for NaNs.")

        # Easier to use variables.

        self.theta_deg = np.degrees(self.theta)
        self.spectra_flat = self.spectra.flatten()
        self.inc_rad = np.radians(self.inc)
        self.sini = np.sin(self.inc_rad)
        self.cosi = np.cos(self.inc_rad)
        self.rotation = 'clockwise' if self.inc > 0 else 'anticlockwise'

        # Velocity axis.

        self.velax = velax
        self.chan = np.diff(velax)[0]
        self.velax_range = (self.velax[0] - 0.5 * self.chan,
                            self.velax[-1] + 0.5 * self.chan)
        self.velax_mask = np.array([self.velax[0], self.velax[-1]])

        # Check the shapes are compatible.

        if self.spectra.shape[0] != self.theta.size:
            raise ValueError("Mismatch in number of angles and spectra.")
        if self.spectra.shape[1] != self.velax.size:
            raise ValueError("Mismatch in the spectra and velocity axis.")

        # Define an empty grid to interpolate the data for plotting.
        # TODO: Check if there's a reasonable way for the user to change this.

        self.theta_grid = np.linspace(-np.pi, np.pi, 60)
        self.velax_grid = self.velax.copy()

    @property
    def extent_grid(self, degrees=True):
        """Extent of the [velax, theta] grid used for plotting."""
        if degrees:
            return [self.velax_grid[0], self.velax_grid[-1],
                    np.degrees(self.theta_grid[0]),
                    np.degrees(self.theta_grid[-1])]
        else:
            return [self.velax_grid[0], self.velax_grid[-1],
                    self.theta_grid[0], self.theta_grid[-1]]

    # -- Measure the Velocity -- #

    def get_vlos(self, p0=None, fit_method='SHO', fit_vrad=False, fix_vlsr=None,
            vrot_mask=None, vlsr_mask=None, vrad_mask=None, dv_mask=None,
            resample=None, optimize=True, nwalkers=32, nburnin=500, nsteps=500,
            scatter=1e-3, signal='int', optimize_kwargs=None, mcmc='emcee',
            mcmc_kwargs=None, centroid_method='quadratic', repeat_with_mask=0):
        """
        Infer the requested velocities by shifting lines back to a common
        center and stacking. The quality of fit is given by the selected
        method which must be ``'dv'``, ``'GP'``, ``'SNR'`` or ``'SHO'``. The
        former, described in `Teague et al. (2018a)`_, minimizes the line width
        of the resulting spectrum via :func:`deprojected_width`. Similarly,
        ``fit_method='SNR'`` aims to maximize the signal to noise of the
        spectrum, as used in `Yen et al. (2016)`_, with specifics of the method
        described in :func:`deprojected_nSNR`. Finally, ``fit_method='GP'``
        uses a Gaussian Process model which relaxes the assumption of an
        analytical line profile, as used in `Teague et al. (2018b)`_. Finally,
        ``fit_method='SHO'`` fits the line centroids with a simple harmonic
        oscillator, as in `Casassus & Perez (2019)`_.

        Different types of resampling can be applied to the spectrum. In
        general it is recommended that ``resample=False`` for the Gaussian
        Process approach, while ``resample=True`` for the other two.

        TWo arrays will be returned: ``v`` and ``dv`` which are the velocity
        profiles and uncertainties, respectively. Both ``v`` and ``dv``
        will have a size of 3, representing the three velocity components, 
        ``v_phi``, ``v_r`` and ``v_z``. As not all methods return the same
        components, those that are unable to be calculated will be populated
        with ``NaN``s.

        Args:
            p0 (optional[list]): Starting positions for the minimization. If
                nothing is provided these will be guessed but this may not
                result in very good starting positions.
            fit_method (optional[str]): Method used to define the quality of
                fit. Must be one of ``'GP'``, ``'dV'``, ``'SNR'`` or ``'SHO'``.
            fit_vrad (bool): Include radial motion in the fit.
            fix_vlsr (optional[bool]): Fix the systemic velocity to calculate
                the deprojected vertical velocities. Only available for
                `fit_method='SHO'`.
            vrot_mask (optional[float]): A rotational velocity to adopt for a
                mask in [m/s].
            vlsr_mask (optional[float]): A systemic velocity to adopt for a mask
                in [m/s].
            vrad_mask (optional[float]): A radial velocity to adopt for a mask
                in [m/s].
            dv_mask (optional[float]): Width of the mask in [m/s].
            resample (optional[bool]): Resampling method to apply. See
                :func:`deprojected_spectrum` for more details.
            optimize (optional[bool]): Optimize the starting positions before
                the MCMC runs. If an integer, the number of iterations to use
                of optimization.
            nwalkers (optional[int]): Number of walkers used for the MCMC runs.
            nburnin (optional[int]): Number of steps used to burn in walkers.
            nsteps (optional[int]): Number of steps taken to sample posteriors.
            scatter (optional[float]): Scatter applied to the starting
                positions before running the MCMC.
            signal (optional[str]): The type of signal to use for the ``'SNR'``
                fit method, either the integral of the line, ``'int'``, the
                default, or the peak of the line, ``'max'``.
            optimize_kwargs (optional[dict]):
            mcmc (optional[str]):
            mcmc_kwargs (optional[dict]):
            centroid_method (optional[str]): Method used to determine the line
                centroids, and must be one of ``'quadratic'``, ``'max'``,
                ``'gaussian'``, ``'doublegauss'`` or ``'doublegauss_fixeddv'``.
            plots (optional[list]):
            repeat_with_mask (optional[int]): Number of iterations to use.
                Currently only works with `fit_method='SHO'`.

        Returns:
            v, dv (array, array): [coming soon]

        .. _Teague et al. (2018a): https://ui.adsabs.harvard.edu/abs/2018ApJ...860L..12T/abstract
        .. _Teague et al. (2018b): https://ui.adsabs.harvard.edu/abs/2018ApJ...868..113T/abstract
        .. _Yen et al. (2016): https://ui.adsabs.harvard.edu/abs/2016ApJ...832..204Y/abstract
        .. _Casassus & Perez (2019): https://ui.adsabs.harvard.edu/abs/2019ApJ...883L..41C/abstract
        """

        # Check the input variables.

        fit_method = fit_method.lower()
        if fit_method not in ['dv', 'gp', 'snr', 'sho']:
            raise ValueError("method must be 'dV', 'GP', 'SNR' or 'SHO'.")
        if fix_vlsr is not None and fit_method != 'sho':
            print("WARNING: fix_vlsr only available for fit_method='SHO'.")

        # Run the appropriate methods.

        if fit_method == 'gp':
            resample = False if resample is None else resample
            popt = self.get_vlos_GP(p0=p0,
                                    fit_vrad=fit_vrad,
                                    vlsr_mask=vlsr_mask,
                                    dv_mask=dv_mask,
                                    nwalkers=nwalkers,
                                    nsteps=nsteps,
                                    nburnin=nburnin,
                                    scatter=scatter,
                                    plots='none',
                                    returns='percentiles',
                                    resample=resample,
                                    mcmc=mcmc,
                                    optimize_kwargs=optimize_kwargs,
                                    mcmc_kwargs=mcmc_kwargs)

            cvar = 0.5 * (popt[:, 2] - popt[:, 0])
            popt = np.array([popt[0, 1],
                             popt[1, 1] if fit_vrad else np.nan,
                             np.nan])
            cvar = np.array([cvar[0],
                             cvar[1] if fit_vrad else np.nan,
                             np.nan])

        elif fit_method == 'dv':
            resample = True if resample is None else resample
            popt = self.get_vlos_dV(p0=p0,
                                    fit_vrad=fit_vrad,
                                    resample=resample,
                                    vlsr_mask=vlsr_mask,
                                    dv_mask=dv_mask,
                                    optimize_kwargs=optimize_kwargs)

            popt = np.array([popt[0], popt[1] if fit_vrad else np.nan, np.nan])
            cvar = np.ones(popt.size) * np.nan

        elif fit_method == 'snr':
            resample = True if resample is None else resample
            popt = self.get_vlos_SNR(p0=p0,
                                     fit_vrad=fit_vrad,
                                     resample=resample,
                                     signal=signal,
                                     vlsr_mask=vlsr_mask,
                                     dv_mask=dv_mask,
                                     optimize_kwargs=optimize_kwargs)

            popt = np.array([popt[0], popt[1] if fit_vrad else np.nan, np.nan])
            cvar = np.ones(popt.size) * np.nan

        elif fit_method == 'sho':
            popt, cvar = self.get_vlos_SHO(p0=p0, 
                                           fit_vrad=fit_vrad,
                                           fix_vlsr=fix_vlsr,
                                           vrot_mask=vrot_mask,
                                           vlsr_mask=vlsr_mask,
                                           vrad_mask=vrad_mask,
                                           dv_mask=dv_mask,
                                           centroid_method=centroid_method,
                                           optimize_kwargs=optimize_kwargs)

            popt = np.array([popt[0],
                             popt[1] if fit_vrad else np.nan,
                             popt[-1]])
            cvar = np.array([cvar[0],
                             cvar[1] if fit_vrad else np.nan,
                             cvar[-1]])
            
        # If required, iterate using these results as a mask. This is only
        # available with SHO given that a systemic velocity is required.

        if repeat_with_mask and fit_method == 'sho':

            vrot_mask = popt[0]
            vrad_mask = popt[1] if fit_vrad else 0.0
            vlsr_mask = popt[2] if fit_vrad else popt[1]

            # Here somtimes the guess is sufficiently bad that it'll mask out
            # all the points. If this is the case, we just skip this and move
            # on.

            try:
                popt, cvar = self.get_vlos(p0=p0,
                                           fit_method=fit_method,
                                           fit_vrad=fit_vrad,
                                           fix_vlsr=fix_vlsr,
                                           vrot_mask=vrot_mask,
                                           vlsr_mask=vlsr_mask,
                                           vrad_mask=vrad_mask,
                                           dv_mask=dv_mask,
                                           resample=resample,
                                           optimize=optimize,
                                           nwalkers=nwalkers,
                                           nburnin=nburnin,
                                           nsteps=nsteps,
                                           scatter=scatter,
                                           signal=signal,
                                           optimize_kwargs=optimize_kwargs,
                                           mcmc=mcmc,
                                           mcmc_kwargs=mcmc_kwargs,
                                           centroid_method=centroid_method,
                                           repeat_with_mask=repeat_with_mask-1)

            except:
                return popt, cvar
        
        return popt, cvar

    # -- Gaussian Processes Approach -- #

    def get_vlos_GP(self, p0=None, fit_vrad=False, vlsr_mask=None, dv_mask=None,
        optimize=False, nwalkers=64,nburnin=50, nsteps=100, scatter=1e-3,
        niter=1, plots=None, returns=None, resample=False, mcmc='emcee',
        optimize_kwargs=None, mcmc_kwargs=None):
        """
        Determine the azimuthally averaged rotational (and optionally radial)
        velocity by finding the greatest overlap between 

        Args:
            p0 (optional[list]): Starting positions.
            fit_vrad (optional[bool]): Whether to also fit for radial
                velocities. Default is ``False``.
            vlsr_mask (optional[float]):
            dv_mask (optional[float]):
            optimize (optional[bool]): Run an optimization step prior to the
                MCMC.
            nwalkers (optional[int]): Number of walkers for the MCMC.
            nburnin (optional[int]): Number of steps to discard for burn-in.
            nsteps (optional[int]): Number of steps used to sample the
                posterior distributions.
            scatter (optional[float]): Scatter of walkers around ``p0``.
            niter (optional[int]): Number of iterations to run, with each run
                adopting the median posterior values from the previous
                iteration as starting points.
            plots (optional[list]): List of diagnostic plots to make. Can be
                ``'walkers'``, ``'corner'`` or ``'none'``.
            returns (optional[list]) List of values to return. Can be
                ``'samples'``, ``'percentiles'`` or ``'none'``.
            mcmc (optional[str]): Which MCMC backend to run, either ``'emcee'``
                or ``'zeus'``.
            optimize_kwargs (optional[dict]): Kwargs to pass to the initial
                optimization of starting parameters.
            mcmc_kwargs (optional[dict]): Kwargs to pass to the MCMC sampler.

        Returns:
            Dependent on what is specified in ``returns``.
        """

        # Starting positions.

        if p0 is None:
            p0 = self._guess_parameters_GP(fit=True)
            if not fit_vrad:
                p0 = np.concatenate([p0[:1], p0[-3:]])
        p0 = np.atleast_1d(p0)

        # Define the parameter labels.

        labels = [r'$v_{\rm \phi,\, proj}$']
        if fit_vrad:
            labels += [r'$v_{\rm r,\, proj}$']
        labels += [r'$\sigma_{\rm rms}}$']
        labels += [r'${\rm ln(\sigma)}$']
        labels += [r'${\rm ln(\rho)}$']

        # Check for NaNs in the starting values.

        if np.any(np.isnan(p0)):
            raise ValueError("WARNING: NaNs in the p0 array.")

        # Optimize the starting positions.

        if optimize:
            if optimize_kwargs is None:
                optimize_kwargs = {}
            p0 = self._optimize_p0_GP(p0,
                                      N=int(optimize),
                                      vlsr_mask=vlsr_mask,
                                      dv_mask=dv_mask,
                                      **optimize_kwargs)

        # Run the sampler.

        nsteps = np.atleast_1d(nsteps)
        nburnin = np.atleast_1d(nburnin)
        nwalkers = np.atleast_1d(nwalkers)
        mcmc_kwargs = {} if mcmc_kwargs is None else mcmc_kwargs

        if mcmc == 'numpyro':
            # The unbinned NUTS likelihood path bypasses
            # _resample_spectra and the velax_mask window — neither is
            # JAX-traceable. Surface clear errors rather than silently
            # ignoring user kwargs.
            if resample is not False and resample != 0:
                raise ValueError(
                    "mcmc='numpyro' requires resample=False (the "
                    "unbinned shifted spectra are fed directly to a "
                    "dense Matern-3/2 GP). Use mcmc='emcee' for "
                    "binned-spectrum fits.")
            if not np.allclose(self.velax_mask,
                               [self.velax[0], self.velax[-1]]):
                raise ValueError(
                    "mcmc='numpyro' does not support a non-default "
                    "velax_mask. Reset self.velax_mask or use "
                    "mcmc='emcee'.")
        else:
            progress = mcmc_kwargs.pop('progress', True)
            moves = mcmc_kwargs.pop('moves', None)
            pool = mcmc_kwargs.pop('pool', None)

        for n in range(int(niter)):

            ns = int(nsteps[n % nsteps.size])
            nb = int(nburnin[n % nburnin.size])
            nw = int(nwalkers[n % nwalkers.size])

            if mcmc == 'numpyro':
                sampler = self._run_nuts_GP(p0=p0,
                                            fit_vrad=fit_vrad,
                                            num_chains=nw,
                                            num_warmup=nb,
                                            num_samples=ns,
                                            **mcmc_kwargs)
            else:
                if mcmc == 'zeus':
                    EnsembleSampler = zeus.EnsembleSampler
                else:
                    EnsembleSampler = emcee.EnsembleSampler

                p0 = random_p0(p0, scatter, nw)

                sampler = EnsembleSampler(nw,
                                          p0.shape[1],
                                          self._lnprobability,
                                          args=(p0[:, 0].mean(),
                                                vlsr_mask,
                                                dv_mask,
                                                resample),
                                          moves=moves,
                                          pool=pool)

                sampler.run_mcmc(p0, nb + ns, progress=progress,
                                 **mcmc_kwargs)

            # Split off the burnt in samples. Numpyro's _NumpyroSampler
            # adapter mirrors emcee's chain shape (nw, nb+ns, ndim) with
            # NaN-padded warmup, so the emcee slicing applies directly.
            if mcmc == 'zeus':
                samples = sampler.chain[-ns:]
            else:
                samples = sampler.chain[:, -ns:]
            samples = samples.reshape(-1, samples.shape[-1])
            p0 = np.median(samples, axis=0)
            time.sleep(0.5)

        # Diagnosis plots if appropriate.

        plots = ['walkers', 'corner'] if plots is None else plots
        plots = [p.lower() for p in np.atleast_1d(plots)]
        if 'walkers' in plots:
            if mcmc == 'zeus':
                # zeus: (nsteps, nwalkers, ndim) -> (ndim, nsteps, nwalkers)
                walkers = np.rollaxis(sampler.chain.copy(), 2)
            else:
                # emcee + numpyro adapter: (nwalkers, nsteps, ndim).T
                walkers = sampler.chain.T
            plot_walkers(walkers, nburnin[-1], labels, True)
        if 'corner' in plots:
            plot_corner(samples, labels)

        # Return the requested values.

        returns = ['percentiles'] if returns is None else returns
        returns = [r.lower() for r in np.atleast_1d(returns)]
        if 'none' in returns:
            return None
        # Resolve all string keys to arrays in one pass so the `in` checks
        # don't trip over a numpy array left in the list by the previous
        # branch (`'samples' in [array, ...]` raises ambiguous-truth).
        wants_percentiles = 'percentiles' in returns
        wants_samples = 'samples' in returns
        if wants_percentiles:
            idx = returns.index('percentiles')
            returns[idx] = np.percentile(samples, [16, 50, 84], axis=0).T
        if wants_samples:
            idx = returns.index('samples')
            returns[idx] = samples
        return returns[0] if len(returns) == 1 else returns

    def _optimize_p0_GP(self, p0, N=1, vlsr_mask=None, dv_mask=None,
                        resample=True, verbose=False, **kwargs):
        """
        Optimize the starting positions, p0. We do this in a slightly hacky way
        because the minimum is not easily found. We first optimize the hyper
        parameters of the GP model, holding the rotation velocity constant,
        then, holding the GP hyperparameters constant, optimizing the rotation
        velocity, before optimizing everything together. This can be run
        multiple times to iteratie to a global optimum. We only update p0 if
        both the minimization converged (res.success == True) and there is an
        improvement in the likelihood.

        One can also pass all the options to optimize.minimize to try different
        minimization techniques. The default values here were based on trial
        and error.

        Args:
            p0 (ndarray): Initial guess of the starting positions.
            N (Optional[int]): Interations of the optimization to run.
            vlsr_mask (Optional[float]):
            dv_mask (Optional[float]):
            resample (Optional[bool/int]): If true, resample the deprojected
                spectra donw to the original velocity resolution. If an integer
                is given, use this as the bew sampling rate relative to the
                original data.

        Returns:
            p0 (ndarray): Optimized array. If scipy.minimize does not converge
                then p0 will not be updated. No warnings are given, however.
        """

        # Default parameters for the minimization.
        # Bit messy to preserve user chosen values.

        kwargs['method'] = kwargs.get('method', 'L-BFGS-B')
        options = kwargs.pop('options', {})
        kwargs['options'] = {'maxiter': options.pop('maxiter', 100000),
                             'maxfun': options.pop('maxfun', 100000),
                             'ftol': options.pop('ftol', 1e-4)}
        for key in options.keys():
            kwargs['options'][key] = options[key]

        # Starting negative log likelihood to test against.

        fit_vrad = len(p0) == 5
        nlnL = self._nlnL(p0=p0,
                          vlsr_mask=vlsr_mask,
                          dv_mask=dv_mask,
                          resample=resample)

        # Cycle through the required number of iterations.

        for _ in range(int(N)):

            # Define the bounds.
            bounds = [(0.8 * p0[0], 1.2 * p0[0])]
            if fit_vrad:
                bounds.append((-0.3 * p0[0], 0.3 * p0[0]))
            bounds += [(0.0, None), (-15.0, 10.0), (0.0, 10.0)]

            # Optimize hyper-parameters, holding vrot and vrad constant.

            res = minimize(self._nlnL_hyper, x0=p0[-3:],
                           args=(p0[0],
                                 p0[1] if fit_vrad else 0.,
                                 vlsr_mask,
                                 dv_mask,
                                 resample),
                           bounds=bounds[-3:], **kwargs)
            if res.success:
                p0_temp = p0
                p0_temp[-3:] = res.x
                nlnL_temp = self._nlnL(p0_temp, vlsr_mask, dv_mask, resample)
                if nlnL_temp < nlnL:
                    p0 = p0_temp
                    nlnL = nlnL_temp
            else:
                if verbose:
                    print('Failed hyper-params mimization: %s' % res.message)

            # Optimize vrot holding the hyper-parameters and vrad constant.

            res = minimize(self._nlnL_vrot, x0=p0[0],
                           args=(p0[-3:],
                                 p0[1] if fit_vrad else 0.,
                                 vlsr_mask,
                                 dv_mask,
                                 resample),
                           bounds=[bounds[0]], **kwargs)
            if res.success:
                p0_temp = p0
                p0_temp[0] = res.x
                nlnL_temp = self._nlnL(p0_temp,
                                       vlsr_mask,
                                       dv_mask,
                                       resample)
                if nlnL_temp < nlnL:
                    p0 = p0_temp
                    nlnL = nlnL_temp
                else:
                    if verbose:
                        print('Failed vrot mimization: %s' % res.message)

            # Optimize vrad holding the hyper-parameters and vrot constant.

            if fit_vrad:
                res = minimize(self._nlnL_vrad, x0=p0[1],
                               args=(p0[0],
                                     p0[-3:],
                                     vlsr_mask,
                                     dv_mask,
                                     resample),
                               bounds=[bounds[1]], **kwargs)
                if res.success:
                    p0_temp = p0
                    p0_temp[1] = res.x
                    nlnL_temp = self._nlnL(p0_temp,
                                           vlsr_mask,
                                           dv_mask,
                                           resample)
                    if nlnL_temp < nlnL:
                        p0 = p0_temp
                        nlnL = nlnL_temp
                    else:
                        if verbose:
                            print('Failed vrad mimization: %s' % res.message)

            # Final minimization with everything.

            res = minimize(self._nlnL,
                           x0=p0,
                           args=(vlsr_mask, dv_mask, resample),
                           bounds=bounds,
                           **kwargs)

            if res.success:
                p0_temp = res.x
                nlnL_temp = self._nlnL(p0_temp, vlsr_mask, dv_mask, resample)
                if nlnL_temp < nlnL:
                    p0 = p0_temp
                    nlnL = nlnL_temp
            else:
                if verbose:
                    print('Failed total mimization: %s' % res.message)

        return p0

    def _guess_parameters_GP(self, fit=True):
        """Guess the starting positions from the spectra."""
        vrot, vrad, _ = self.guess_parameters(fit=fit)
        noise = int(min(10, self.spectra.shape[1] / 3.0))
        noise = np.std([self.spectra[:, :noise], self.spectra[:, -noise:]])
        ln_sig = np.log(np.std(self.spectra))
        ln_rho = np.log(150.)
        return np.array([vrot, vrad, noise, ln_sig, ln_rho])

    @staticmethod
    def _randomize_p0(p0, nwalkers, scatter):
        """Randomize starting positions p0 for nwalkers around p0 with scatter."""
        dp0 = np.random.randn(nwalkers * len(p0)).reshape(nwalkers, len(p0))
        dp0 = np.where(p0 == 0.0, 1.0, p0)[None, :] * (1.0 + scatter * dp0)
        return np.where(p0[None, :] == 0.0, dp0 - 1.0, dp0)

    def _nlnL_vrot(self, vrot, hyperparams, vrad, vlsr_mask=None, dv_mask=None,
                   resample=False):
        """Negative lnlikelihood function with vrot as only argument."""
        theta = np.insert(hyperparams, 0, [vrot, vrad])
        nll = -self._lnlikelihood(theta=theta,
                                  vlsr_mask=vlsr_mask,
                                  dv_mask=dv_mask,
                                  resample=resample)
        return nll if np.isfinite(nll) else 1e15

    def _nlnL_hyper(self, hyperparams, vrot, vrad, vlsr_mask=None, dv_mask=None,
                    resample=False):
        """Negative lnlikelihood function with hyperparams as only argument."""
        theta = np.insert(hyperparams, 0, [vrot, vrad])
        nll = -self._lnlikelihood(theta=theta,
                                  vlsr_mask=vlsr_mask,
                                  dv_mask=dv_mask,
                                  resample=resample)
        return nll if np.isfinite(nll) else 1e15

    def _nlnL_vrad(self, vrad, vrot, hyperparams, vlsr_mask=None, dv_mask=None,
                   resample=False):
        """Negative lnlikelihood function with vrad as only argument."""
        theta = np.insert(hyperparams, 0, [vrot, vrad])
        nll = -self._lnlikelihood(theta=theta,
                                  vlsr_mask=vlsr_mask,
                                  dv_mask=dv_mask,
                                  resample=resample)
        return nll if np.isfinite(nll) else 1e15

    def _nlnL(self, theta, vlsr_mask=None, dv_mask=None, resample=False):
        """Negative log-likelihood function for optimization."""
        nll = -self._lnlikelihood(theta=theta,
                                  vlsr_mask=vlsr_mask,
                                  dv_mask=dv_mask,
                                  resample=resample)
        return nll if np.isfinite(nll) else 1e15

    @staticmethod
    def _build_kernel(x, y, hyperparams):
        """Build the GP. Returns ``None`` if construction fails.

        Replaces the previous celerite implementation with tinygp; the
        kernel parameterisation is preserved exactly:

        - ``noise`` adds a diagonal jitter ``noise**2`` (matches
          ``celerite.terms.JitterTerm(log_sigma=log(noise))``).
        - ``exp(lnsigma)**2 * Matern32(scale=exp(lnrho))`` matches
          ``celerite.terms.Matern32Term(log_sigma=lnsigma,
          log_rho=lnrho)``.
        - The GP mean is the nan-mean of ``y`` (matches the previous
          ``mean=np.nanmean(y), fit_mean=True``).
        """
        noise, lnsigma, lnrho = hyperparams
        sigma2 = jnp.exp(2.0 * lnsigma)
        rho = jnp.exp(lnrho)
        kernel = sigma2 * kernels.Matern32(scale=rho)
        try:
            return GaussianProcess(kernel, jnp.asarray(x),
                                   diag=noise ** 2,
                                   mean=jnp.nanmean(y))
        except Exception:
            return None

    @staticmethod
    def _lnprior(theta, vref):
        """Uninformative log-prior function for MCMC."""
        try:
            vrot, vrad = theta[:-3]
        except ValueError:
            vrot, vrad = theta[0], 0.0
        noise, lnsigma, lnrho = theta[-3:]
        if abs(vrot - vref) / vref > 0.4:
            return -np.inf
        if abs(vrad / vrot) > 1.0:
            return -np.inf
        if vrot <= 0.0:
            return -np.inf
        if noise <= 0.0:
            return -np.inf
        if not -15.0 < lnsigma < 10.:
            return -np.inf
        if not 0.0 <= lnrho <= 10.:
            return -np.inf
        return 0.0

    def _lnlikelihood(self, theta, vlsr_mask=None, dv_mask=None,
                      resample=False):
        """Log-likelihood function for the MCMC."""

        # Unpack the free parameters.

        try:
            vrot, vrad = theta[:-3]
        except ValueError:
            vrot, vrad = theta[0], 0.0
        hyperparams = theta[-3:]

        # Deproject the data and resample if requested.

        x, y = self.deprojected_spectrum(vrot=vrot,
                                         vrad=vrad,
                                         resample=resample,
                                         scatter=False,
                                         vlsr_mask=vlsr_mask,
                                         dv_mask=dv_mask)
        x, y = self._get_masked_spectrum(x, y)

        # Build the GP model and calculate the log-likelihood.

        gp = annulus._build_kernel(x, y, hyperparams)
        if gp is None:
            return -np.inf
        try:
            ll = float(gp.log_probability(jnp.asarray(y)))
        except Exception:
            return -np.inf
        return ll if np.isfinite(ll) else -np.inf

    def _lnprobability(self, theta, vref, vlsr_mask=None, dv_mask=None,
                       resample=False):
        """Log-probability function for the MCMC."""
        if ~np.isfinite(annulus._lnprior(theta, vref)):
            return -np.inf
        return self._lnlikelihood(theta=theta,
                                  vlsr_mask=vlsr_mask,
                                  dv_mask=dv_mask,
                                  resample=resample)

    # -- numpyro NUTS path for the GP fit -- #
    #
    # The emcee path above feeds the resampled, NaN-filtered spectrum to
    # tinygp through ``deprojected_spectrum`` -> ``_resample_spectra``,
    # which use scipy.stats.binned_statistic and boolean indexing — both
    # produce data-dependent shapes that JAX cannot trace. The numpyro
    # path skips resampling entirely and feeds the un-binned shifted
    # spectra directly to a dense Matern32 GP. The pre-shift NaN mask
    # (parameter-independent) is precomputed at first call so the
    # likelihood has a fixed output shape for ``jax.jit``/``jax.grad``.

    def _gp_finite_index(self):
        """Cached (phi_idx, chan_idx, y_finite) over all finite spectrum
        pixels. The masks are parameter-independent so they're computed
        once and reused across NUTS steps."""
        if not hasattr(self, '_gp_finite_cache'):
            spectra = np.asarray(self.spectra)
            finite = np.isfinite(spectra)
            phi_idx, chan_idx = np.where(finite)
            y_finite = spectra[phi_idx, chan_idx]
            self._gp_finite_cache = (
                jnp.asarray(phi_idx),
                jnp.asarray(chan_idx),
                jnp.asarray(y_finite),
            )
        return self._gp_finite_cache

    def _lnlikelihood_jax_GP(self, theta, fit_vrad):
        """JAX-traceable GP log-likelihood on the unbinned, shifted spectra.

        Differs from :meth:`_lnlikelihood` in that it bypasses
        ``deprojected_spectrum`` / ``_resample_spectra`` (numpy-only,
        data-dependent shape) and the ``velax_mask`` window. Only the
        parameter-independent finite-spectrum mask is applied. These
        restrictions are checked upstream in :meth:`get_vlos_GP` before
        the numpyro path is dispatched."""
        if fit_vrad:
            vrot, vrad = theta[0], theta[1]
            noise, lnsigma, lnrho = theta[2], theta[3], theta[4]
        else:
            vrot, vrad = theta[0], 0.0
            noise, lnsigma, lnrho = theta[1], theta[2], theta[3]

        phi_idx, chan_idx, y = self._gp_finite_index()

        theta_pol = jnp.asarray(self.theta)
        inc_rad = jnp.asarray(float(self.inc_rad))
        sin_inc_abs = jnp.sin(jnp.abs(inc_rad))
        sin_inc_signed = jnp.sin(inc_rad)
        vrot_proj = vrot * jnp.cos(theta_pol) * sin_inc_abs
        vrad_proj = -vrad * jnp.sin(theta_pol) * sin_inc_signed
        vlos = vrot_proj + vrad_proj                # (nphi,)

        velax = jnp.asarray(self.velax)
        x = velax[chan_idx] - vlos[phi_idx]         # (N,)

        sigma2 = jnp.exp(2.0 * lnsigma)
        rho = jnp.exp(lnrho)
        kernel = sigma2 * kernels.Matern32(scale=rho)
        gp = GaussianProcess(kernel, x, diag=noise ** 2, mean=jnp.mean(y))
        return gp.log_probability(y)

    def _numpyro_model_GP(self, vref, fit_vrad):
        """numpyro model for :meth:`get_vlos_GP`.

        The emcee path's bespoke ``_lnprior`` (parameter-dependent
        ``|vrad/vrot| < 1`` bound) is translated to structured priors;
        ``vrad`` is given the slightly-looser ``Uniform(-1.4*vref,
        1.4*vref)`` since the original constraint folds vrad's range
        into vrot's, and vrot is bounded by ``Uniform(0.6*vref,
        1.4*vref)`` matching ``|vrot - vref|/vref < 0.4``. The
        (positive-real, no upper bound) constraint on ``noise`` becomes
        ``ImproperUniform`` on the positive cone — the same idiom used
        for unbounded flat priors in :meth:`_numpyro_model_fitmap`."""
        import numpyro
        import numpyro.distributions as dist
        from numpyro.distributions import constraints

        vrot = numpyro.sample('vrot',
                              dist.Uniform(0.6 * vref, 1.4 * vref))
        if fit_vrad:
            vrad = numpyro.sample('vrad',
                                  dist.Uniform(-1.4 * vref, 1.4 * vref))

        noise = numpyro.sample('noise',
                               dist.ImproperUniform(constraints.positive,
                                                    (), ()))
        lnsigma = numpyro.sample('lnsigma', dist.Uniform(-15.0, 10.0))
        lnrho = numpyro.sample('lnrho', dist.Uniform(0.0, 10.0))

        if fit_vrad:
            theta = jnp.stack([vrot, vrad, noise, lnsigma, lnrho])
        else:
            theta = jnp.stack([vrot, noise, lnsigma, lnrho])

        numpyro.factor('lnL', self._lnlikelihood_jax_GP(theta, fit_vrad))

    def _run_nuts_GP(self, p0, fit_vrad, num_chains, num_warmup,
                     num_samples, **kwargs):
        """Run numpyro NUTS over :meth:`_numpyro_model_GP` and return a
        :class:`_NumpyroSampler` adapter so :meth:`get_vlos_GP`'s emcee
        post-processing path consumes it without modification.
        ``num_chains`` / ``num_warmup`` / ``num_samples`` map onto
        ``nwalkers`` / ``nburnin`` / ``nsteps``; ``seed``,
        ``progress``, and ``chain_method`` are forwarded to numpyro."""
        if not _HAS_NUMPYRO:
            raise ImportError(
                "numpyro is required for mcmc='numpyro'. "
                "Install with `pip install numpyro`.")
        import numpyro.infer as ni
        from numpyro.infer.initialization import init_to_value

        seed = kwargs.pop('seed', 0)
        progress = kwargs.pop('progress', True)
        chain_method = kwargs.pop('chain_method', 'sequential')
        kwargs.pop('scatter', None)   # legacy emcee kwarg, not used by NUTS

        p0 = np.asarray(p0).astype(float)
        vref = float(p0[0])

        if fit_vrad:
            free_keys = ('vrot', 'vrad', 'noise', 'lnsigma', 'lnrho')
            init_values = {k: float(p0[i])
                           for i, k in enumerate(free_keys)}
        else:
            free_keys = ('vrot', 'noise', 'lnsigma', 'lnrho')
            init_values = {k: float(p0[i])
                           for i, k in enumerate(free_keys)}

        init_strategy = init_to_value(values=init_values)

        nuts = ni.NUTS(self._numpyro_model_GP,
                       init_strategy=init_strategy, **kwargs)
        mcmc = ni.MCMC(nuts,
                       num_warmup=num_warmup,
                       num_samples=num_samples,
                       num_chains=num_chains,
                       progress_bar=progress,
                       chain_method=chain_method)
        mcmc.run(jax.random.PRNGKey(seed),
                 vref=vref, fit_vrad=fit_vrad,
                 extra_fields=('potential_energy',))

        samples_dict = mcmc.get_samples(group_by_chain=True)
        chain_post = jnp.stack([samples_dict[k] for k in free_keys],
                               axis=-1)
        warmup_pad = jnp.full((num_chains, num_warmup, len(free_keys)),
                              jnp.nan)
        chain = jnp.concatenate([warmup_pad, chain_post], axis=1)

        pe = mcmc.get_extra_fields(group_by_chain=True)['potential_energy']
        lnprob_post = -np.asarray(pe).T
        lnprob_pad = np.full((num_warmup, num_chains), np.nan)
        lnprobability = np.concatenate([lnprob_pad, lnprob_post], axis=0)

        return _NumpyroSampler(np.asarray(chain), lnprobability)

    # -- Minimizing Line Width Approach -- #

    def get_vlos_dV(self, p0=None, fit_vrad=False, resample=False,
            vrot_mask=None, vlsr_mask=None, vrad_mask=None, dv_mask=None, 
            optimize_kwargs=None):
        """
        Infer the rotational (and optically radial) velocity by minimizing the
        width of the shifted-and-stacked azimuthally averaged spectrum.

        Args:
            p0 (optional[list]): Starting positions for the optimization.
            fit_vrad (optional[bool]): Whether to include the radial velocity
                in the fit. Default is ``False``.
            resample (optional[bool]): Resample the shifted spectra by this
                factor. For example, resample = 2 will shift and bin the
                spectrum down to sampling rate twice that of the original data.
            vrot_mask (float): Disk-frame rotational velocity in [m/s].
            vlsr_mask (optional[float]): Systemic velocity in [m/s].
            vrad_mask (optional[float]): Disk-frame radial velocity in [m/s].
            dv_mask (optional[float]): Half-width of the mask in [m/s].
            optimize_kwargs (optional[dict]): Kwargs to pass to ``minimize``.

        Returns:
            Velocities which minimize the line width of the shifted and stacked
            spectrum.
        """

        # Starting positions.

        if p0 is None:
            p0 = self.guess_parameters(fit=True)[:2]
            if not fit_vrad:
                p0 = p0[:1]
        p0 = np.atleast_1d(p0)

        # Populate the kwargs.

        optimize_kwargs = {} if optimize_kwargs is None else optimize_kwargs
        optimize_kwargs['method'] = optimize_kwargs.get('method', 'Nelder-Mead')
        options = optimize_kwargs.pop('options', {})
        options['maxiter'] = options.pop('maxiter', 10000)
        options['maxfev'] = options.pop('maxfev', 10000)
        options['fatol'] = options.pop('fatol', 1e-4)
        optimize_kwargs['options'] = options

        # Run the minimization.

        args = (fit_vrad, resample, vrot_mask, vlsr_mask, vrad_mask, dv_mask)
        res = minimize(self.deprojected_width,
                       x0=p0,
                       args=args,
                       **optimize_kwargs)
        if not res.success:
            print("WARNING: minimize did not converge.")
        return res.x if res.success else np.nan

    def deprojected_width(self, theta, fit_vrad=False, resample=True,
            vrot_mask=None, vlsr_mask=None, vrad_mask=None, dv_mask=None):
        """
        Return the Gaussian width of the deprojected and stacked spectra.

        Args:
            theta (list): Deprojection velocities, ``(vrot[, vrad])``.
            fit_vrad (optional[bool]): Whether ``vrad`` in is ``theta``.
            resample (optional): How to resample the data.  See
                :func:`deprojected_spectrum` for more details.
            vrot_mask (float): Disk-frame rotational velocity in [m/s].
            vlsr_mask (optional[float]): Systemic velocity in [m/s].
            vrad_mask (optional[float]): Disk-frame radial velocity in [m/s].
            dv_mask (optional[float]): Half-width of the mask in [m/s].

        Returns:
            The Doppler width of the average stacked spectrum using the
            velocities to align the individual spectra.
        """
        from .helper_functions import get_gaussian_width
        vrot, vrad = theta if fit_vrad else (theta, 0.0)
        x, y = self.deprojected_spectrum(vrot=vrot,
                                         vrad=vrad,
                                         resample=resample,
                                         scatter=False,
                                         vrot_mask=vrot_mask,
                                         vlsr_mask=vlsr_mask,
                                         vrad_mask=vrad_mask,
                                         dv_mask=dv_mask)
        return get_gaussian_width(*self._get_masked_spectrum(x, y))

    # -- Rotation Velocity by Fitting a SHO -- #

    def get_vlos_SHO(self, p0=None, fit_vrad=False, fix_vlsr=None,
            vrot_mask=None, vlsr_mask=None, vrad_mask=None, dv_mask=None,
            centroid_method='quadratic', optimize_kwargs=None):
        """
        Infer the disk-frame rotational (and, optionally, radial) velocity by
        finding velocity which best describes the azimuthal dependence of the
        line centroid modelled as a simple harmonic oscillator.

        Args:
            p0 (optional[list]): Starting positions for the optimization.
            fit_vrad (optional[bool]): Whether to include the radial velocity
                in the fit. Default is ``False``.
            fix_vlsr (optional[float]): If provided, use this value to deproject
                the vertical velocity component.
            vrot_mask (float): Disk-frame rotational velocity in [m/s].
            vlsr_mask (optional[float]): Systemic velocity in [m/s].
            vrad_mask (optional[float]): Disk-frame radial velocity in [m/s].
            dv_mask (optional[float]): Half-width of the mask in [m/s].
            centroid_method (optional[str]): Method used to determine the line
                centroids, and must be one of ``'quadratic'``, ``'max'``,
                ``'gaussian'``, ``'doublegauss'`` or ``'doublegauss_fixeddv'``.
            optimize_kwargs (optional[dict]): Kwargs to pass to ``curve_fit``.

        Returns:
            pop, cvar (array, array): Arrays of the best-fit parameter values
            and their uncertainties returned from ``curve_fit``.
        """
        from .helper_functions import SHO, SHO_double
        v0, dv0 = self.line_centroids(method=centroid_method,
                                      vrot_mask=vrot_mask,
                                      vlsr_mask=vlsr_mask,
                                      vrad_mask=vrad_mask,
                                      dv_mask=dv_mask)
        assert v0.size == self.theta.size

        # Starting positions. Here we're using projected velocities such that
        # A = vrot * sin(|i|), B = -vrad * sin(i) and C = vlsr - vz * cos(i).

        if p0 is None:
            A, B, C = 0.5 * (v0.max() - v0.min()), 0.0, v0.mean()
            p0 = [A, C] if not fit_vrad else [A, B, C]
        assert len(p0) == 3 if fit_vrad else 2

        # Set up curve_fit. TODO: Is there a better fitting routine than this?

        optimize_kwargs = {} if optimize_kwargs is None else optimize_kwargs
        optimize_kwargs['p0'] = p0
        optimize_kwargs['sigma'] = dv0
        optimize_kwargs['absolute_sigma'] = True
        optimize_kwargs['maxfev'] = optimize_kwargs.pop('maxfev', 10000)

        # Run the optimization.

        try:
            popt, cvar = curve_fit(SHO_double if fit_vrad else SHO,
                                   self.theta, v0, **optimize_kwargs)
        except TypeError:
            popt = np.empty(3 if fit_vrad else 2)
            cvar = popt[:, None] * popt[None, :]
        cvar = np.diag(cvar)**0.5

        # Convert from projected velocities into disk-frame velocities.
        # Note that C is only converted to vertical velocities if the systemic
        # velocity is provided through `fix_vlsr`.

        popt[0] /= abs(self.sini)
        cvar[0] /= abs(self.sini)
        if fit_vrad:
            popt[1] /= -self.sini
            cvar[1] /= abs(self.sini)
        if fix_vlsr is not None:
            popt[-1] = (fix_vlsr - popt[-1]) / self.cosi
            cvar[-1] /= self.cosi

        # Return the optimized, disk-frame values.

        return popt, cvar

    # -- Rotation Velocity by Maximizing SNR -- #

    def get_vlos_SNR(self, p0=None, fit_vrad=False, resample=False,
            signal='weighted', vrot_mask=None, vlsr_mask=None, vrad_mask=None,
            dv_mask=None, optimize_kwargs=None):
        """
        Infer the rotation (and, optically, the radial) velocity by finding the
        rotation velocity (and radial velocities) which, after shifting all
        spectra to a common velocity, results in the maximum signal-to=noise
        ratio of the stacked profile. This is an implementation of the method
        described in Yen et al. (2016, 2018).

        Args:
            p0 (optional[list]): Starting positions for the optimization.
            fit_vrad (optional[bool]): Whether to include the radial velocity
                in the fit. Default is ``False``.
            resample (optional[bool]): Resample the shifted spectra by this
                factor. For example, resample = 2 will shift and bin the
                spectrum down to sampling rate twice that of the original data.
            signal (Optional[str]): Method used to calculate the signal, either
                'max' for the line peak or 'int' for the integrated intensity or
                'weighted' for a Gaussian weighted integrated intensity.
            vrot_mask (float): Disk-frame rotational velocity in [m/s].
            vlsr_mask (optional[float]): Systemic velocity in [m/s].
            vrad_mask (optional[float]): Disk-frame radial velocity in [m/s].
            dv_mask (optional[float]): Half-width of the mask in [m/s].
            optimize_kwargs (optional[dict]): Kwargs to pass to ``minimize``.

        Returns:
            Velocities which maximizese signal to noise of the shifted and
            stacked spectrum.

        .. _Yen et al. (2016): https://ui.adsabs.harvard.edu/abs/2016ApJ...832..204Y/abstract

        """

        # Make sure the signal is defined.

        if signal not in ['max', 'int', 'weighted']:
            raise ValueError("'signal' must be either 'max', 'int', "
                             + "or 'weighted'.")

        # Starting positions.

        if p0 is None:
            p0 = self.guess_parameters(fit=True)[:2]
            if not fit_vrad:
                p0 = p0[:1]
        p0 = np.atleast_1d(p0)

        # Populate the kwargs. For some reason L-BFGS-B doesn't play nicely.

        optimize_kwargs = {} if optimize_kwargs is None else optimize_kwargs
        optimize_kwargs['method'] = optimize_kwargs.get('method', 'Powell')
        options = optimize_kwargs.pop('options', {})
        options['maxiter'] = options.pop('maxiter', 10000)
        options['maxfev'] = options.pop('maxfev', 10000)
        options['ftol'] = options.pop('ftol', 1e-4)
        optimize_kwargs['options'] = options

        # Run the minimization.

        args = (fit_vrad, resample, signal, vrot_mask, vlsr_mask, vrad_mask, dv_mask)
        res = minimize(self.deprojected_nSNR,
                       x0=p0,
                       args=args,
                       **optimize_kwargs)
        if not res.success:
            print("WARNING: minimize did not converge.")
        return res.x if res.success else np.nan

    def deprojected_nSNR(self, theta, fit_vrad=False, resample=False, signal='weighted',
            vrot_mask=None, vlsr_mask=None, vrad_mask=None, dv_mask=None):
        """
        Return the negative SNR of the deprojected spectrum. There are three
        ways to calculate the signal of the data. ``signal='max'`` will use the
        Gaussian peak relative to the noise, ``signal='int'`` will use the
        integrated spectrum as the signal, while ``signal='weighted'`` will
        additionally use a Gaussian shape weighting so that noise in the line
        wings are minimized, as in `Yen et al. (2016)`_. The default method is
        ``signal='weighted'``.

        Args:
            theta (list): Disk-frame velocities, ``(vrot[, vrad])``.
            fit_vrad (optional[bool]): Whether ``vrad`` in is ``theta``.
            resample (optional): How to resample the data. See
                :func:`deprojected_spectrum` for more details.
            signal (optional[str]): Definition of SNR to use.
            vrot_mask (float): Disk-frame rotational velocity in [m/s].
            vlsr_mask (optional[float]): Systemic velocity in [m/s].
            vrad_mask (optional[float]): Disk-frame radial velocity in [m/s].
            dv_mask (optional[float]): Half-width of the mask in [m/s].

        Returns:
            Negative of the signal-to-noise ratio.
        """
        from .helper_functions import gaussian, fit_gaussian
        vrot, vrad = theta if fit_vrad else (theta, 0.0)
        x, y = self.deprojected_spectrum(vrot=vrot,
                                         vrad=vrad,
                                         resample=resample,
                                         scatter=False,
                                         vrot_mask=vrot_mask,
                                         vlsr_mask=vlsr_mask,
                                         vrad_mask=vrad_mask,
                                         dv_mask=dv_mask)
        x0, dx, A = fit_gaussian(x, y)

        noise = self._estimate_RMS() / np.sqrt(self.theta.size)

        if signal == 'max':
            SNR = A / noise
        else:
            if signal == 'weighted':
                w = gaussian(x, x0, dx, (np.sqrt(np.pi) * abs(dx))**-1)
            else:
                w = np.ones(x.size)
            mask = abs(x - x0) / dx <= 3.0
            SNR = np.trapezoid((y * w)[mask], x=x[mask])
        return -SNR

    def _estimate_RMS(self, N=15, iterative=False, nsigma=3.0):
        """Estimate the RMS of the data."""
        if iterative:
            std = np.nanmax(self.spectra_flat)
            for _ in range(5):
                mask = abs(self.spectra_flat) <= nsigma * std
                std_new = np.nanstd(self.spectra_flat[mask])
                if std_new == std or np.isnan(std_new) or std_new == 0.0:
                    return std
                std = std_new
        else:
            std = np.nanstd([self.spectra[:, :N], self.spectra[:, -N:]])
        return std

    # -- Deprojection Functions -- #

    def calc_vlos(self, vrot, vrad=0.0, vlsr=0.0):
        """
        Calculate the line of sight velocity for each spectrum given the
        rotational and radial velocities at the attached polar angles.

        Note that the rotational and radial velocities are specified in the disk
        frame and do not take into account the projection along the line of
        sight. Remember that a positive radial velocity is moving away from the
        star.

        Args:
            vrot (float): Disk-frame rotation velocity in [m/s].
            vrad (optional[float]): Disk-frame radial velocity in [m/s].
            vlsr (optional[float]): Systemtic velocity in [m/s].

        Returns:
            ndarray: Array of projected line of sight velocities in [m/s] at
                each polar angle.
        """
        vrot_proj = vrot * np.cos(self.theta) * np.sin(abs(self.inc_rad))
        vrad_proj = -vrad * np.sin(self.theta) * np.sin(self.inc_rad)
        return vrot_proj + vrad_proj + vlsr

    def deprojected_spectra(self, vrot, vrad=0.0, kind='linear', smooth=None,
            vrot_mask=None, vlsr_mask=0.0, vrad_mask=None, dv_mask=None):
        """
        Returns all deprojected points as an ensemble. If a velocity mask is
        defined then these pixels will be converted to NaNs such that the
        returned array is the same shape as ``self.spectra``. The shifted
        spectra will be on the same velocity axis as the attached cube.

        Args:
            vrot (float): Disk-frame rotation velocity in [m/s].
            vrad (optional[float]): Disk-frame radial velocity in [m/s].
            kind (optional[str]): Interpolation kind to use.
            smooth (optional): The weights used to smooth the data prior to
                shifting. If a ``float`` or ``int`` is provided, will interpret
                this as a top-hat function with that width.
            vrot_mask (float): Disk-frame rotational velocity in [m/s].
            vlsr_mask (optional[float]): Systemic velocity in [m/s].
            vrad_mask (optional[float]): Disk-frame radial velocity in [m/s].
            dv_mask (optional[float]): Half-width of the mask in [m/s].

        Returns:
            A ``(M, N)`` shaped array of ``M`` spectra over ``N`` velocity points.
        """

        # Smooth the spectra before interpolating.
        
        spectra = self.spectra.copy()
        if smooth is not None:
            from scipy.ndimage import convolve1d
            if isinstance(smooth, (float, int)):
                smooth = np.ones(int(smooth)) / float(smooth)
            spectra_a = convolve1d(spectra, smooth, axis=1)
            spectra_b = convolve1d(spectra[:, ::-1], smooth, axis=1)[:, ::-1]
            spectra = np.nanmean([spectra_a, spectra_b], axis=0)

        # Apply the mask if necessary.

        velocity_mask = self.get_velocity_mask(vrot_mask=vrot_mask,
                                               vlsr_mask=vlsr_mask,
                                               vrad_mask=vrad_mask,
                                               dv_mask=dv_mask)

        spectra = np.where(velocity_mask, spectra, np.nan)

        # Interpolate the data onto the new grid.

        from scipy.interpolate import interp1d

        s = []
        vlos = self.calc_vlos(vrot=vrot, vrad=vrad)
        for dv, spectrum in zip(vlos, spectra):
            mask = np.isfinite(spectrum)
            s += [interp1d(x=self.velax[mask]-dv,
                           y=spectrum[mask],
                           kind=kind,
                           fill_value=np.nan,
                           bounds_error=False)(self.velax)]
        s = np.array(s)
        if s.shape == self.spectra.shape:
            return s
        else:
            raise ValueError("Incorrect deprojected spectra shape.")

    def get_river(self, vrot=0.0, vrad=0.0, kind='linear', weights=None,
                  method='nearest'):
        """
        Returns the deprojected spectra, but interpolated onto a reguar grid
        defined by ``annulus.theta_grid`` in [rad] and ``annulus.velax_grid``
        in [m/s].

        Args:
            vrot (float): Rotational velocity in [m/s].
            vrad (optional[float]): Radial velocity in [m/s].
            kind (optional[str]): Interpolation kind to use when shifting
                spectra.
            weights (optional): The weights used to smooth the data prior to
                shifting. If a ``float`` or ``int`` is provided, will interpret
                this as a top-hat function with that width.
            method (optional[str]): Interpolation method to use when gridding
                data.

        Returns:
            The polar angle grid, the velocity grid and the river.
        """
        river = self.deprojected_spectra(vrot, vrad, kind, weights)
        river = self._grid_river(river, method)
        return self.theta_grid, self.velax_grid, river

    def deprojected_spectrum(self, vrot, vrad=0.0, resample=True, scatter=True,
            vrot_mask=None, vlsr_mask=None, vrad_mask=None, dv_mask=None):
        """
        Returns ``(x, y[, dy])`` of the collapsed and deprojected spectrum
        using the provided velocities to deproject the data. When collapsing the
        spectra the velocity grid can be changed with the ``resample`` argument.
        
        Note that the rotational and radial velocities are specified in the disk
        frame and do not take into account the projection along the line of
        sight. Remember that a positive radial velocity is moving away from the
        star.
        
        Different methods to resample the data can also be applied.

            ``reasmple=False`` - returns the unbinned, shifted pixels.

            ``resample=True`` - shifted pixels are binned onto the
            attached velocity axis.

            ``resample=int(N)`` - shifted pixels are binned onto a velocity
            axis which is a factor of N times coarser than the attached
            velocity axis. ``N=1`` is the same as ``resample=True``.

            ``resample=float(N)`` - shifted pixels are binned onto a velocity
            axis with a spacing of N [m/s]. If the velocity spacing is too fine
            this may result in empty bins and thus NaNs in the spectrum.

        It is important to disgintuish between ``float`` and ``int`` arguments
        for ``resample``.

        A mask can also be applied to the data assuming azimuthally symmetric
        rotational and radial velocities.

        Args:
            vrot (float): Disk-frame rotational velocity in [m/s].
            vrad (optional[float]): Disk-frame radial velocity in [m/s].
            resample (optional): Type of resampling to be applied.
            scatter (optional[bool]): If the spectrum is resampled, whether to
                return the scatter in each velocity bin.
            vrot_mask (float): Disk-frame rotational velocity in [m/s].
            vlsr_mask (optional[float]): Systemic velocity in [m/s].
            vrad_mask (optional[float]): Disk-frame radial velocity in [m/s].
            dv_mask (optional[float]): Half-width of the mask in [m/s].

        Returns:
            A deprojected spectrum, resampled using the provided method.
        """
        
        vlos = self.calc_vlos(vrot=vrot, vrad=vrad)
        vpnts = self.velax[None, :] - vlos[:, None]
        spnts = self.spectra.copy()

        # Apply the velocity mask.

        velocity_mask = self.get_velocity_mask(vrot_mask=vrot_mask,
                                               vlsr_mask=vlsr_mask,
                                               vrad_mask=vrad_mask,
                                               dv_mask=dv_mask)

        # Order the spectra in increasing velocity and then resample them.

        vpnts, spnts = self._order_spectra(vpnts=vpnts[velocity_mask],
                                           spnts=spnts[velocity_mask])
        
        x, y, dy = self._resample_spectra(vpnts=vpnts,
                                          spnts=spnts,
                                          resample=resample,
                                          scatter=True)

        mask = np.isfinite(y)
        if scatter:
            return x[mask], y[mask], dy[mask]
        return x[mask], y[mask]
    
    def get_velocity_mask(self, vrot_mask=None, vlsr_mask=0.0, vrad_mask=None, dv_mask=None):
        """
        Returns a mask based on an assumed rotational and radial velocity
        component. Note that all velocities are provided in the disk-frame.
        By default the half-width of the mask if 5 times the channel spacing.

        Args:
            vrot_mask (optional[float]): Disk-frame rotational velocity in [m/s].
            vlsr_mask (optional[float]): Systemic velocity in [m/s].
            vrad_mask (optional[float]): Disk-frame radial velocity in [m/s].
            dv_mask (optional[float]): Half-width of the mask in [m/s].

        Returns:
            mask (array): A 2D mask for to be applied to ``self.spectra``.
        """
        if vrot_mask is None:
            return np.ones(self.spectra.shape).astype('bool')
        if vlsr_mask is None:
            vlsr_mask = 0.0
        if vrad_mask is None:
            vrad_mask = 0.0
        if dv_mask is None:
            dv_mask = 5.0 * self.chan
        if dv_mask < self.chan:
            raise ValueError("`dv_mask` must be at least the channel spacing.")
        mask = self.calc_vlos(vrot=vrot_mask,
                              vrad=vrad_mask,
                              vlsr=vlsr_mask)
        mask = abs(mask[:, None] - self.velax[None, :]) <= dv_mask
        return np.array(mask)

    def get_masked_spectra(self, vrot_mask, vlsr_mask=0.0, vrad_mask=None,
                           dv_mask=None):
        """
        Returns spectra masked only around the line peaks based on the provided
        velocity profile. Note that all mask velocities are provided in the
        disk-frame.

        Args:
            vrot_mask (float): Disk-frame rotational velocity in [m/s].
            vlsr_mask (optional[float]): Systemic velocity in [m/s].
            vrad_mask (optional[float]): Disk-frame radial velocity in [m/s].
            dv_mask (optional[float]): Half-width of the mask in [m/s].

        Returns:
            velax, spectra (array, array): A 2D array of the velocity points and
                the spectra.
        """
        mask = self.get_velocity_mask(vrot_mask=vrot_mask,
                                      vlsr_mask=vlsr_mask,
                                      vrad_mask=vrad_mask,
                                      dv_mask=dv_mask)
        velax = np.array([self.velax[m] for m in mask])
        spectra = np.array([s[m] for s, m in zip(self.spectra, mask)])
        return velax, spectra

    def line_centroids(self, method='quadratic', vrot_mask=None, vlsr_mask=None,
                       vrad_mask=None, dv_mask=None):
        """
        Returns the line centroid for each of the spectra in the annulus.
        Various methods of determining the centroid are possible accessible
        through the ``method`` argument.

        Args:
            method (str): Method used to determine the line centroid. Must be
                in ['max', 'quadratic', 'gaussian', 'gaussthick', 'doublegauss', 
                'doublegauss_fixeddv]. The former returns the pixel of maximum
                value, 'quadratic' fits a quadratic function to the pixel of
                maximum value and its two neighbouring pixels (see Teague &
                Foreman-Mackey 2018 for details) and 'gaussian', 'gaussthick'
                'doublegauss' and 'doublegauss_fixeddv' fit (an) analytical
                Gaussian profile(s) to the line.
            vrot_mask (float): Disk-frame rotational velocity in [m/s].
            vlsr_mask (optional[float]): Systemic velocity in [m/s].
            vrad_mask (optional[float]): Disk-frame radial velocity in [m/s].
            dv_mask (optional[float]): Half-width of the mask in [m/s].

        Returns:
            vmax, dvmax (array, array): Line centroids and associated
                uncertainties.
        """

        # Get the spectra to fit, using the mask if appropriate.

        velax, spectra = self.get_masked_spectra(vrot_mask=vrot_mask,
                                                 vlsr_mask=vlsr_mask,
                                                 vrad_mask=vrad_mask,
                                                 dv_mask=dv_mask)

        # Cycle through the methods and apply.

        method = method.lower()
       
        if method == 'max':
            vmax = np.array([v[np.argmax(s)] for v, s in zip(velax, spectra)])
            dvmax = np.ones(vmax.size) * self.chan
       
        elif method == 'quadratic':
            from bettermoments.quadratic import quadratic
            vmax = [quadratic(s, uncertainty=self.rms,
                              x0=v[0], dx=self.chan)
                    for v, s in zip(velax, spectra)]
            vmax, dvmax = np.array(vmax).T[:2]
       
        elif method == 'gaussian':
            from .helper_functions import get_gaussian_center
            vmax = [get_gaussian_center(v, s, self.rms)
                    for v, s in zip(velax, spectra)]
            vmax, dvmax = np.array(vmax).T
       
        elif method == 'gaussthick':
            from .helper_functions import get_gaussthick_center
            vmax = [get_gaussthick_center(v, s, self.rms)
                    for v, s in zip(velax, spectra)]
            vmax, dvmax = np.array(vmax).T
        
        elif method == 'doublegauss':
            from .helper_functions import get_doublegauss_center
            vmax = [get_doublegauss_center(v, s, self.rms)
                    for v, s in zip(velax, spectra)]
            vmax, dvmax = np.array(vmax).T
        
        elif method == 'doublegauss_fixeddv':
            from .helper_functions import get_doublegauss_fixeddV_center
            vmax = [get_doublegauss_fixeddV_center(v, s, self.rms)
                    for v, s in zip(velax, spectra)]
            vmax, dvmax = np.array(vmax).T
        
        else:
            raise ValueError(f"Unknown method, {method}.")
        
        return vmax, dvmax

    def _order_spectra(self, vpnts, spnts=None):
        """Return velocity ordered spectra removing any NaNs."""
        spnts = self.spectra_flat if spnts is None else spnts
        nan_mask = np.isfinite(spnts)
        vpnts, spnts = vpnts[nan_mask], spnts[nan_mask]
        if len(spnts) != len(vpnts):
            raise ValueError("Wrong size in 'vpnts' and 'spnts'.")
        idxs = np.argsort(vpnts)
        return vpnts[idxs], spnts[idxs]

    def _resample_spectra(self, vpnts, spnts, resample=False, scatter=False):
        """
        Resample the spectra to a given velocity axis. The scatter is estimated
        as the standard deviation of the bin (note that this is not rescaled by
        the square root of the number of samples).

        Args:
            vpnts (ndarray): Array of the velocity values.
            spnts (ndarray): Array of the spectrum values.
            resample (bool/int/float): Describes the resampling method. If
                False, no resample is done and the vpnts and spnts are returned
                as is. If an integer (where True = 1), this samples vpnts on a
                velocity grid at a sampling rate 'resample' times that of the
                originally supplied velocity axis. If a float, this will
                describe the spectral resolution of the sampled grid.
            scatter (bool): If True, return the standard deviation in each bin.

        Returns:
            x (ndarray): Velocity bin centers.
            y (ndarray): Mean of the bin.
            dy (ndarray/None): Standard error on the mean of the bin.
        """
        if type(resample) is bool:
            x = vpnts.copy()
            y = spnts.copy()
            mask = np.logical_and(np.isfinite(y), y != 0.0)
            if not resample:
                if not scatter:
                    return x[mask], y[mask]
                return x[mask], y[mask], np.ones(x[mask].size) * np.nan
        if isinstance(resample, (int, bool)):
            bins = int(self.velax.size * int(resample) + 1)
            bins = np.linspace(self.velax[0], self.velax[-1], bins)
        elif isinstance(resample, float):
            bins = np.arange(self.velax[0], self.velax[-1], resample)
            bins += 0.5 * (vpnts.max() - bins[-1])
        elif isinstance(resample, np.ndarray):
            bins = 0.5 * np.diff(resample).mean()
            bins = np.linspace(resample[0] - bins,
                               resample[-1] + bins,
                               resample.size + 1)
        else:
            raise TypeError("Resample must be a boolean, int, float or array.")
        idxs = np.isfinite(spnts)
        vpnts, spnts = vpnts[idxs], spnts[idxs]
        y = binned_statistic(vpnts, spnts, statistic='mean', bins=bins)[0]
        x = np.average([bins[1:], bins[:-1]], axis=0)
        mask = np.logical_and(np.isfinite(y), y != 0.0)
        if not scatter:
            return x[mask], y[mask]
        dy = binned_statistic(vpnts, spnts, statistic='std', bins=bins)[0]
        mask = np.logical_and(dy > 0.0, mask)
        return x[mask], y[mask], dy[mask]

    def _get_masked_spectrum(self, x, y):
        """Return the masked spectrum for fitting."""
        mask = np.logical_and(x >= self.velax_mask[0], x <= self.velax_mask[1])
        return x[mask], y[mask]

    def guess_parameters(self, method='quadratic', fit=True):
        """
        Guess the starting positions by fitting the SHO equation to the line
        peaks. These include the rotational and radial velocities and the
        combined systemtic velocities and projected vertical motions.

        Args:
            method (optional[str]): Method used to measure the velocities of
                the line peaks. Must be in ['max', 'quadradtic', 'gaussian'].
            fit (optional[bool]): Use ``scipy.curve_fit`` to fit the line peaks
                as a function of velocity.

        Returns:
            The rotational, radial and systemic velocities all in [m/s].
        """
        vpeaks, _ = self.line_centroids(method=method)
        vlsr = np.mean(vpeaks)

        vrot_p = vpeaks[abs(self.theta).argmin()]
        vrot_p -= vpeaks[abs(self.theta - np.pi).argmin()]
        vrot_p *= 0.5

        vrad_p = vpeaks[abs(self.theta - 0.5 * np.pi).argmin()]
        vrad_p -= vpeaks[abs(self.theta + 0.5 * np.pi).argmin()]
        vrad_p *= -0.5

        if fit:
            try:
                from .helper_functions import SHO_double
                vrot_p, vrad_p = curve_fit(f=SHO_double,
                                           xdata=self.theta,
                                           ydata=vpeaks,
                                           p0=[vrot_p, vrad_p, vlsr],
                                           maxfev=10000)[0][:2]
            except Exception:
                print("Fitting failed...")

        vrot = vrot_p / np.sin(abs(self.inc_rad))
        vrad = -vrad_p / np.sin(self.inc_rad)

        return vrot, vrad, vlsr

    # -- River Functions -- #

    def _grid_river(self, spnts, method='nearest'):
        """
        Grid the data to plot as a river as a regular grid. The grids can be
        changed with through the ``theta_grid`` and ``velax_grid`` attributes.

        Args:
            spnts (ndarray): Array of spectra to grid with shape ``(N, M)``
                where ``N`` is the number of spectra and ``M`` is the number of
                velocity channels.
            method (optional[str]): Interpolation method to use.

        Returns:
            river (ndarray): The regularly gridded river.
        """
        from scipy.interpolate import griddata
        spnts = np.vstack([spnts[-1:], spnts, spnts[:1]])
        vpnts = self.velax[None, :] * np.ones(spnts.shape)
        tpnts = np.concatenate([self.theta[-1:] - 2.0 * np.pi,
                                self.theta,
                                self.theta[:1] + 2.0 * np.pi])
        tpnts = tpnts[:, None] * np.ones(spnts.shape)
        sgrid = griddata((vpnts.flatten(), tpnts.flatten()), spnts.flatten(),
                         (self.velax_grid[None, :], self.theta_grid[:, None]),
                         method=method)
        sgrid = np.where(self.theta_grid[:, None] > tpnts.max(), np.nan, sgrid)
        sgrid = np.where(self.theta_grid[:, None] < tpnts.min(), np.nan, sgrid)
        return sgrid

    # -- Plotting Functions -- #

    def plot_spectra(self, ax=None, return_fig=False, step_kwargs=None,
                     vrot_mask=None, vlsr_mask=None, vrad_mask=0.0,
                     dv_mask=200.0):
        """
        Plot the attached spectra on the same velocity axis. You can include the
        velocity mask to see how that affects the spectra.

        Args:
            ax (optional[matplotlib axis]): Axis onto which the data will be
                plotted. If ``None``, a new figure will be created.
            return_fig (optional[bool]): Whether to return the figure. Default
                is ``False``.
            step_kwargs (optional[dict]): Kwargs to pass to ``matplotlib.step``.
            vrot_mask (optional[float]): Disk-frame rotational velocity used for
                the mask in [m/s].
            vlsr_mask (optional[float]): Systemic velocity used for the mask in
                [m/s].
            vrad_mask (optional[float]): Disk-frame radial velocity used for the
                mask in [m/s].
            dv_mask (optional[float]): Half-width of the mask in [m/s].

        Returns:
            fig (matplotlib figure): Returned if ``return_fig=True``.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(5.0, 3.1), constrained_layout=True)
        else:
            return_fig = False
        step_kwargs = {} if step_kwargs is None else step_kwargs
        step_kwargs['where'] = step_kwargs.pop('where', 'mid')
        step_kwargs['lw'] = step_kwargs.pop('lw', 1.0)
        step_kwargs['c'] = step_kwargs.pop('c', 'k')

        velax, spectra = self.get_masked_spectra(vrot_mask=vrot_mask,
                                                 vlsr_mask=vlsr_mask,
                                                 vrad_mask=vrad_mask,
                                                 dv_mask=dv_mask)
        for v, s in zip(velax, spectra):
            ax.step(v, s, **step_kwargs)
        ax.set_xlabel('Velocity (m/s)')
        ax.set_ylabel('Intensity (Jy/beam)')
        ax.set_xlim(self.velax[0], self.velax[-1])
        if return_fig:
            return fig

    def plot_spectrum(self, vrot=0.0, vrad=0.0, resample=True, plot_fit=False,
                      ax=None, return_fig=False, plot_kwargs=None):
        """
        Plot the aligned and stacked spectrum. Can also include a fit to the
        averaged spectrum, by default a Gaussian profile.

        Args:
            vrot (Optional[float]): Rotation velocity in [m/s].
            vrad (Optional[float]): Radial velocity in [m/s].
            resample (Optional[int/float/bool]): Resampling option. See
                ``annulus.deprojected_spectrum`` for more details.
            plot_fit (Optional[bool/str]): Whether to overplot a fit to the
                data. If ``True``, will fit a Gaussian profile, if
                ``'gaussthick'``, will fit an optically thick Gaussian profile.
            ax (Optional[matplotlib axis]): Axis onto which the data will be
                plotted. If ``None``, a new matplotlib figure will be made.
            return_fig (Optional[bool]): Whether to return the new matplotlib
                figure if an ``ax`` was not provided.
            plot_kwargs (Optional[dict]): Kwargs to be passed to
                ``matplotlib.errorbar`` and ``matplotlib.step``.

        Returns:
            fig (matplotlib figure) if ``return_fig=True``.
        """

        # Get the deprojected spectrum and transform to mJy/beam.
        # Remove all points which have a zero uncertainty if resampled.

        x, y, dy = self.deprojected_spectrum(vrot=vrot,
                                             vrad=vrad,
                                             resample=resample)
        y, dy = y * 1e3, dy * 1e3

        # Matplotlib Axes

        if ax is None:
            fig, ax = plt.subplots()
        else:
            return_fig = False

        # Set the plotting defaults.

        kw = {}
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        kw['c'] = plot_kwargs.pop('c', plot_kwargs.pop('color', None))
        kw['c'] = '0.8' if plot_fit else '0.0' if kw['c'] is None else kw['c']
        kw['lw'] = plot_kwargs.pop('lw', plot_kwargs.pop('linewidth', 1.0))
        cs = plot_kwargs.pop('capsize', kw['lw'] * 2.0)
        ct = plot_kwargs.pop('capthick', kw['lw'])
        xlim = plot_kwargs.pop('xlim', (x.min(), x.max()))
        ylim = plot_kwargs.pop('ylim', None)

        # Plot the spectrum.

        L = ax.errorbar(x, y, dy, fmt=' ', capsize=cs, capthick=ct, **kw)
        ax.step(x, y, where='mid', zorder=L[0].get_zorder()+1, **kw)
        ax.set_xlabel('Velocity (m/s)')
        ax.set_ylabel('Intensity (mJy/beam)')
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)

        # Fit the data if requested.

        if plot_fit:
            labels = [r'$v_0$', r'$\Delta V$', r'$I_{\nu}$']
            units = ['(m/s)', '(m/s)', '(mJy/bm)']
            plot_fit = 'gaussian' if type(plot_fit) is bool else plot_fit
            if plot_fit == 'gaussian':
                from .helper_functions import gaussian
                from .helper_functions import fit_gaussian
                popt, cvar = fit_gaussian(x, y, dy, True)
                y_mod = gaussian(x, *popt)
            else:
                from .helper_functions import gaussian_thick
                from .helper_functions import fit_gaussian_thick
                labels += [r'$\tau$']
                units += ['']
                popt, cvar = fit_gaussian_thick(x, y, dy, True)
                y_mod = gaussian_thick(x, *popt)

            ax.plot(x, y_mod, color='r', lw=kw['lw'], ls='-',
                    zorder=L[0].get_zorder()+2)

            # Include the best-fit parameters.

            for lidx, label in enumerate(labels):
                annotation = label + ' = {:.0f}'.format(popt[lidx])
                annotation += ' +/- {:.0f} {}'.format(cvar[lidx], units[lidx])
                ax.text(0.975, 0.95 - 0.075 * lidx, annotation,
                        ha='right', va='top', color='r',
                        transform=ax.transAxes)

        # Return fig.
        if return_fig:
            return fig

    def plot_river(self, vrot=None, vrad=0.0, residual=False, method='nearest',
                   vrot_mask=None, vlsr_mask=None, vrad_mask=None, dv_mask=None,
                   plot_kwargs=None, return_fig=False):
        """
        Make a river plot, showing how the spectra change around the azimuth.
        This is a nice way to search for structure within the data.

        Args:
            vrot (optional[float]): Rotational velocity in [m/s] used to
                deproject the spectra. If ``None``, no deprojection is applied.
            vrad (optional[float]): Radial velocity in [m/s] used to deproject
                the spectra. Default is ``0.0``.
            residual (optional[bool]): If ``True``, subtract the azimuthally
                averaged line profile before plotting. Default is ``False``.
            method (optional[str]): Interpolation method passed to
                ``scipy.interpolate.griddata``. Default is ``'nearest'``.
            vrot_mask (optional[float]): Disk-frame rotational velocity in
                [m/s] used to overlay a velocity mask on the plot.
            vlsr_mask (optional[float]): Systemic velocity in [m/s] used for
                the mask. Required if ``vrot_mask`` is provided.
            vrad_mask (optional[float]): Disk-frame radial velocity in [m/s]
                used for the mask.
            dv_mask (optional[float]): Half-width of the mask in [m/s].
            plot_kwargs (optional[dict]): Kwargs passed to ``pcolormesh``.
                Accepted keys include ``'vmin'``, ``'vmax'``, ``'cmap'`` and
                ``'xlim'``.
            return_fig (optional[bool]): Whether to return the figure.
                Default is ``False``.

        Returns:
            fig (matplotlib figure): Returned if ``return_fig=True``. To
                access the axis use ``fig.axes[0]``.
        """

        # Imports.

        from matplotlib.ticker import MultipleLocator
        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

        # Deproject and grid the spectra.

        if vrot is None:
            spectra = self.spectra
            vrot = 0.0
        else:
            spectra = self.deprojected_spectra(vrot=vrot, vrad=vrad)
        spectra = self._grid_river(spectra, method=method)

        # Get the residual if necessary.

        mean_spectrum = np.nanmean(spectra, axis=0)
        if residual:
            spectra -= mean_spectrum

        # Estimate the RMS. Here we try an iterative clip but if this seems to
        # remove all the points we revert to a standard deviation.

        rms = np.nanstd(spectra)
        for _ in range(5):
            rms = np.nanstd(spectra[abs(spectra) <= 3.0 * rms])
        if np.isnan(rms):
            rms = np.nanstd(spectra)

        # Define the min and max for plotting.

        kw = {} if plot_kwargs is None else plot_kwargs
        xlim = kw.pop('xlim', None)
        kw['vmax'] = kw.pop('vmax', np.nanmax(abs(spectra)))
        kw['vmin'] = kw.pop('vmin', -kw['vmax'] if residual else -rms)
        kw['cmap'] = kw.pop('cmap', 'RdBu_r' if residual else 'turbo')

        # Plot the data.

        fig, ax = plt.subplots(figsize=(6.0, 2.25), constrained_layout=True)
        ax_divider = make_axes_locatable(ax)
        im = ax.pcolormesh(self.velax_grid,
                           np.degrees(self.theta_grid),
                           spectra, **kw)
        ax.set_ylim(-180, 180)
        ax.yaxis.set_major_locator(MultipleLocator(60.0))
        ax.set_xlim(xlim)
        ax.set_xlabel('Velocity (m/s)')
        ax.set_ylabel(r'$\phi$' + ' (deg)')

        # Include the proposed mask.

        if vrot_mask is not None:
            if vlsr_mask is None:
                raise ValueError("Must specify `vlsr_mask`.")
            if vrad_mask is None:
                vrad_mask = 0.0
            if dv_mask is None:
                dv_mask = 2.0 * self.chan

            # Note here that we allow for the mask to be deprojected in case
            # the the shifting isn't the same as the mask...

            mask = (vrot_mask - vrot) * np.cos(self.theta_grid) * abs(self.sini)
            mask += (vrad - vrad_mask) * np.sin(self.theta_grid) * self.sini
            mask += vlsr_mask

            ax.fill_betweenx(np.degrees(self.theta_grid),
                             self.velax_grid[0] - np.diff(self.velax_grid)[0],
                             mask - dv_mask,
                             color='k' if residual else 'w', lw=0.0,
                             alpha=0.3 if residual else 0.7)
            ax.fill_betweenx(np.degrees(self.theta_grid),
                             mask + dv_mask,
                             self.velax_grid[-1] + np.diff(self.velax_grid)[0],
                             color='k' if residual else 'w', lw=0.0,
                             alpha=0.3 if residual else 0.7)
            ax.plot(mask - dv_mask, np.degrees(self.theta_grid),
                    color='k' if residual else 'w')
            ax.plot(mask + dv_mask, np.degrees(self.theta_grid),
                    color='k' if residual else 'w')

        # Add the mean spectrum panel.

        if not residual:
            fig.set_size_inches(6.0, 2.5, forward=True)
            ax1 = ax_divider.append_axes('top', size='25%', pad='2%')
            ax1.step(self.velax_grid, mean_spectrum,
                     where='mid', lw=1., c='k')
            ax1.fill_between(self.velax_grid, mean_spectrum,
                             step='mid', color='.7')
            ax1.set_ylim(3*kw['vmin'], kw['vmax'])
            ax1.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1])
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.tick_params(which='both', left=0, bottom=0, right=0, top=0)
            for side in ['left', 'right', 'top', 'bottom']:
                ax1.spines[side].set_visible(False)

        # Add the colorbar.

        cb_ax = ax_divider.append_axes('right', size='2%', pad='1%')
        cb = plt.colorbar(im, cax=cb_ax)
        if residual:
            cb.set_label('Residual (mJy/beam)',
                         rotation=270, labelpad=13)
        else:
            cb.set_label('Intensity (Jy/beam)',
                         rotation=270, labelpad=13)

        if return_fig:
            return fig

    def plot_centroids(self, centroid_method='quadratic', plot_fit=None,
                       fit_vrad=False, fix_vlsr=None, vrot_mask=None,
                       vlsr_mask=None, vrad_mask=None, dv_mask=None, ax=None,
                       return_fig=False, plot_kwargs=None):
        """
        Plot the measured line centroids as a function of polar angle.

        Args:
            centroid_method (optional[str]): Method used to determine the line
                centroid. Must be one of ``'max'``, ``'quadratic'``,
                ``'gaussian'``, ``'doublegauss'`` or
                ``'doublegauss_fixeddv'``. Default is ``'quadratic'``.
            plot_fit (optional[bool]): Whether to overplot a SHO fit to the
                centroids. Default is ``False``.
            fit_vrad (optional[bool]): Whether to include a radial velocity
                component in the SHO fit. Default is ``False``.
            fix_vlsr (optional[float]): If provided, fix the systemic velocity
                to this value and compute the deprojected vertical velocity.
            vrot_mask (optional[float]): Disk-frame rotational velocity in
                [m/s] used to apply a mask before fitting centroids.
            vlsr_mask (optional[float]): Systemic velocity in [m/s] used for
                the mask.
            vrad_mask (optional[float]): Disk-frame radial velocity in [m/s]
                used for the mask.
            dv_mask (optional[float]): Half-width of the mask in [m/s].
            ax (optional[matplotlib axis]): Axis onto which the data will be
                plotted. If ``None``, a new figure will be created.
            return_fig (optional[bool]): Whether to return the figure.
                Default is ``False``.
            plot_kwargs (optional[dict]): Kwargs passed to
                ``matplotlib.errorbar``.

        Returns:
            fig (matplotlib figure): Returned if ``return_fig=True``. To
                access the axis use ``fig.axes[0]``.
        """

        from .helper_functions import SHO_double
        if ax is None:
            fig, ax = plt.subplots()
        else:
            return_fig = False

        # Calculate the line centroids, including any masking necessary.

        v0, dv0 = self.line_centroids(method=centroid_method,
                                      vrot_mask=vrot_mask,
                                      vlsr_mask=vlsr_mask,
                                      vrad_mask=vrad_mask,
                                      dv_mask=dv_mask)
        dv0 = abs(dv0)

        # Set the defaults for plotting.

        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        plot_kwargs['fmt'] = plot_kwargs.pop('fmt', 'o')
        plot_kwargs['c'] = plot_kwargs.pop('c', 'k')
        plot_kwargs['ms'] = plot_kwargs.pop('ms', 4)
        plot_kwargs['lw'] = plot_kwargs.pop('lw', 1.25)
        plot_kwargs['capsize'] = plot_kwargs.pop('capsize', 2.5)

        # Plot the data.

        L = ax.errorbar(self.theta_deg, v0, dv0, **plot_kwargs)
        ax.set_xlim(-180, 180)
        ax.xaxis.set_major_locator(MultipleLocator(60.0))
        ax.xaxis.set_minor_locator(MultipleLocator(10.0))
        ax.tick_params(which='minor', left=1)
        ax.set_xlabel(r'$\phi$' + ' (deg)')
        ax.set_ylabel(r'$v_0$' + ' (m/s)')

        if plot_fit:

            # Fit the data.

            popt, cvar = self.get_vlos_SHO(fit_vrad=fit_vrad,
                                           fix_vlsr=fix_vlsr,
                                           vrot_mask=vrot_mask,
                                           vlsr_mask=vlsr_mask,
                                           vrad_mask=vrad_mask,
                                           centroid_method=centroid_method)

            v_p, dv_p = popt[0] * abs(self.sini), cvar[0] * abs(self.sini)
            if fit_vrad:
                v_r, dv_r = popt[1] * self.sini, cvar[1] * abs(self.sini)
            else:
                v_r, dv_r = 0.0, 0.0
            if fix_vlsr:
                vlsr, dvlsr = fix_vlsr, 0.0
                v_z, dv_z = popt[-1] * self.cosi, cvar[-1] * self.cosi
            else:
                vlsr, dvlsr = popt[-1], cvar[-1]
                v_z, dv_z = None, None

            # Note that as get_vlos_SHO returns the true values, so we need to
            # reproject them.

            v0mod = SHO_double(self.theta_grid, v_p, v_r, vlsr)
            ax.plot(np.degrees(self.theta_grid), v0mod, lw=1.0, ls='--',
                    color='r', zorder=L[0].get_zorder()-10)

            # Add in the labels. If the systemic velocity has a small error then
            # we can skip the uncertainty. If a vlsr is not provided then we can
            # skip the v_z label.

            label = r'$v_{\rm LSR}$' + ' = {:.0f} '.format(vlsr)
            if dvlsr > 0.5:
                label += r'$\pm$' + ' {:.0f} '.format(dvlsr)
            label += 'm/s'
            ax.text(0.975, 0.975, label, va='top', ha='right', color='r',
                    transform=ax.transAxes)

            label = r'$v_{\phi,\, proj}$' + ' = {:.0f} '.format(v_p)
            label += r'$\pm$' + ' {:.0f} m/s'.format(dv_p)
            ax.text(0.975, 0.900, label, va='top', ha='right', color='r',
                    transform=ax.transAxes)

            if fit_vrad:
                label = r'$v_{r,\, proj}$' + ' = {:.0f} '.format(v_r)
                label += r'$\pm$' + ' {:.0f} m/s'.format(dv_r)
                ax.text(0.975, 0.825, label, va='top', ha='right', color='r',
                        transform=ax.transAxes)
                
            if v_z is not None:
                label = r'$v_{z,\, proj}$' + ' = {:.0f} '.format(v_z)
                label += r'$\pm$' + ' {:.0f} m/s'.format(dv_z)
                ax.text(0.975, 0.750 if fit_vrad else 0.825,
                        label, va='top', ha='right', color='r',
                        transform=ax.transAxes)

            ylim = max(ax.get_ylim()[1] - vlsr, vlsr - ax.get_ylim()[0])
            ax.set_ylim(vlsr - ylim, vlsr + ylim)

        if return_fig:
            return fig

    @staticmethod
    def cmap_RdGy():
        """Return a custom red-gray diverging colormap for river plots."""
        import matplotlib.colors as mcolors
        c2 = plt.cm.Reds(np.linspace(0.0, 0.9, 16))
        c1 = plt.cm.gray(np.linspace(0.2, 1.0, 16))
        colors = np.vstack((c1, np.ones((2, 4)), c2))
        return mcolors.LinearSegmentedColormap.from_list('eddymap', colors)


# The 3D annulus is identical to the existing ``annulus`` class.
Annulus3D = annulus
