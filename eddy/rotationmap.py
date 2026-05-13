# -*- coding: utf-8 -*-

import time
import yaml
import zeus
import emcee
import numpy as np
import jax
import jax.numpy as jnp
import scipy.constants as sc
from .imagecube import imagecube
from .momentmap import momentmap
from .helper_functions import plot_walkers, plot_corner, random_p0, _NumpyroSampler
import matplotlib.pyplot as plt
import warnings

try:
    import numpyro  # noqa: F401
    _HAS_NUMPYRO = True
except ImportError:
    _HAS_NUMPYRO = False

# Default MCMC backend for fit_map / fit_annuli. Kept on emcee even when
# numpyro is installed so existing scripts retain identical walker semantics
# and posterior shapes; users opt into NUTS via sampler='numpyro'.
_default_sampler = 'emcee'


class rotationmap(momentmap):
    """
    Read in the velocity maps and initialize the class. To make the fitting
    quicker, we can clip the cube to a smaller region, or downsample the
    image to get a quicker initial fit.

    Args:
        path (str): Relative path to the rotation map you want to fit.
        FOV (Optional[float]): If specified, clip the data down to a
            square field of view with sides of `FOV` [arcsec].
        uncertainty (Optional[str]): Relative path to the map of v0
            uncertainties. Must be a FITS file with the same shape as the
            data. If nothing is specified, will tried to find an uncerainty
            file following the ``bettermoments`` format. If this is not found,
            it will assume a 10% on all pixels.
        downsample (Optional[int]): Downsample the image by this factor for
            quicker fitting. For example, using ``downsample=4`` on an image
            which is 1000 x 1000 pixels will reduce it to 250 x 250 pixels.
            If you use ``downsample='beam'`` it will sample roughly
            spatially independent pixels using the beam major axis as the
            spacing.
        fill (Optional[float]): Replace all ``NaN`` values with this value.
        force_center (Optional[bool]): If ``True`` define the spatial axes such
            that they describe offset from the array center in [arcsec]. This
            is useful if the FITS header does not contain axis information.
    """

    priors = {}
    priors_jax = {}
    SHO_priors = {}
    _vortex_layers = 2

    def __init__(self, path, FOV=None, uncertainty=None, downsample=None,
                 fill=None, force_center=False):
        super().__init__(path=path, FOV=FOV, fill=fill,
                         force_center=force_center)

        # Check to see what unit the velocities are in.

        if self.header['bunit'].lower() not in ['m/s', 'km/s']:
            msg = "What is the velocity unit? Either `m/s` or `km/s`."
            self.velocity_unit = input(msg)
        else:
            self.velocity_unit = self.header['bunit'].lower()

        self.data *= 1e3 if self.velocity_unit == 'km/s' else 1.0
        self.mask = np.isfinite(self.data)
        self._readuncertainty(uncertainty=uncertainty, FOV=FOV)

        if downsample is not None:
            self.downsample_cube(downsample)

        self.default_parameters = self._load_default_parameters()
        self._set_default_priors()

    @property
    def vlsr(self):
        """Median line-of-sight velocity of the data in [m/s]."""
        return np.nanmedian(self.data)

    @property
    def vlsr_kms(self):
        """Median line-of-sight velocity of the data in [km/s]."""
        return self.vlsr / 1e3

    # -- FITTING FUNCTIONS -- #

    def fit_map(self, p0, params, r_min=None, r_max=None, optimize=True,
                nwalkers=None, nburnin=300, nsteps=100, scatter=1e-3,
                plots=None, returns=None, pool=None, mcmc='emcee',
                mcmc_kwargs=None, niter=1):
        """
        Fit a rotation profile to the data. Note that for a disk with
        a non-zero height, the sign of the inclination dictates the direction
        of rotation: a positive inclination denotes a clockwise rotation, while
        a negative inclination denotes an anti-clockwise rotation.

        The function must be provided with a dictionary of parameters,
        ``params`` where key is the parameter and its value is either the fixed
        value this will take,

            ``params['PA'] = 45.0``

        would fix position angle to 45 degrees. Or, if an integeter, the index
        in the starting positions, ``p0``, such that,

            ``params['mstar'] = 0``

        would mean that the first value in ``p0`` is the guess for ``'mstar'``.

        For a list of the geometrical parameters, included flared emission
        surfaces or warps, see :func:`disk_coords`. In addition to these
        geometrical properties ``params`` also requires ``'mstar'``,
        ``'vlsr'``, ``'dist'``.

        To include a spatial convolution of the model rotation map you may also
        set ``params['beam']=True``, however this is only really necessary for
        low spatial resolution data, or for rotation maps made via the
        intensity-weighted average velocity.

        .. _Rosenfeld et al. (2013): https://ui.adsabs.harvard.edu/abs/2013ApJ...774...16R/abstract

        Args:
            p0 (list): List of the free parameters to fit.
            params (dictionary): Dictionary of the model parameters.
            r_min (optional[float]): Inner radius to fit in [arcsec].
            r_max (optional[float]): Outer radius to fit in [arcsec].
            optimize (optional[bool]): Use ``scipy.optimize`` to find the
                ``p0`` values which maximize the likelihood. Better results
                will likely be found. Note that for the masking the default
                ``p0`` and ``params`` values are used for the deprojection, or
                those foundvfrom the optimization. If this results in a poor
                initial mask, try with ``optimise=False`` or with a ``niter``
                value larger than 1.
            nwalkers (optional[int]): Number of walkers to use for the MCMC.
            nburnin (optional[int]): Number of steps to discard for burn-in.
            nsteps (optional[int]): Number of steps to use to sample the
                posterior distributions.
            scatter (optional[float]): Scatter used in distributing walker
                starting positions around the initial ``p0`` values.
            plots (optional[list]): List of the diagnostic plots to make. This
                can include ``'mask'``, ``'walkers'``, ``'corner'``,
                ``'bestfit'``, ``'residual'``, or ``'none'`` if no plots are to
                be plotted. By default, all are plotted.
            returns (optional[list]): List of items to return. Can contain
                ``'samples'``, ``'sampler'``, ``'percentiles'``, ``'dict'``,
                ``'model'``, ``'residuals'`` or ``'none'``. By default only
                ``'percentiles'`` are returned.
            pool (optional): An object with a `map` method.
            mcmc_kwargs (Optional[dict]): Dictionary to pass to the MCMC
                ``EnsembleSampler``.
            niter (optional[int]): Number of iterations to perform using the
                median PDF values from the previous MCMC run as starting
                positions. This is probably only useful if you have no idea
                about the starting positions for the emission surface or if you
                want to remove walkers stuck in local minima.

        Returns:
            to_return (list): Depending on the returns list provided.
                ``'samples'`` will be all the samples of the posteriors (with
                the burn-in periods removed). ``'percentiles'`` will return the
                16th, 50th and 84th percentiles of each posterior functions.
                ``'dict'`` will return a dictionary of the median parameters
                which can be directly input to other functions.
        """

        # Check the dictionary. May need some more work.

        if r_min is not None:
            if 'r_min' in params.keys():
                print("Found `r_min` in `params`. Overwriting value.")
            params['r_min'] = r_min
        if r_max is not None:
            if 'r_max' in params.keys():
                print("Found `r_max` in `params`. Overwriting value.")
            params['r_max'] = r_max

        params_tmp = self.verify_params_dictionary(params.copy())

        # Generate the mask for fitting based on the params.

        p0 = np.squeeze(p0).astype(float)
        temp = rotationmap._populate_dictionary(p0, params_tmp)
        self.ivar = self._calc_ivar(temp)

        # Check what the parameters are.

        labels = rotationmap._get_labels(params_tmp)
        labels_raw = []
        for label in labels:
            label_raw = label.replace('$', '').replace('{', '')
            label_raw = label_raw.replace(r'\rm ', '').replace('}', '')
            labels_raw += [label_raw]
        if len(labels) != len(p0):
            raise ValueError("Mismatch in labels and p0. Check for integers.")
        print("Assuming:\n\tp0 = [%s]." % (', '.join(labels_raw)))

        # Run an initial optimization using scipy.minimize. Recalculate the
        # inverse variance mask.

        if optimize:
            p0 = self._optimize_p0(p0, params_tmp)

        # Set up and run the MCMC with emcee.

        nsteps = np.atleast_1d(nsteps)
        nburnin = np.atleast_1d(nburnin)
        nwalkers = np.atleast_1d(nwalkers)

        mcmc_kwargs = {} if mcmc_kwargs is None else mcmc_kwargs
        mcmc_kwargs['scatter'], mcmc_kwargs['pool'] = scatter, pool

        for n in range(int(niter)):

            # Make the mask for fitting.

            temp = rotationmap._populate_dictionary(p0, params_tmp)
            temp = self.verify_params_dictionary(temp)
            self.ivar = self._calc_ivar(temp)

            # Run the sampler.

            sampler = self._run_mcmc(p0=p0, params=params_tmp,
                                     nwalkers=nwalkers[n % nwalkers.size],
                                     nburnin=nburnin[n % nburnin.size],
                                     nsteps=nsteps[n % nsteps.size],
                                     mcmc=mcmc, **mcmc_kwargs)

            # Split off the samples.

            samples = sampler.get_chain(discard=nburnin[-1], flat=True)
            if type(params_tmp['PA']) is int:
                samples[:, params_tmp['PA']] %= 360.0
            p0 = np.median(samples, axis=0)
            medians = rotationmap._populate_dictionary(p0, params.copy())
            medians = self.verify_params_dictionary(medians)

        # Diagnostic plots.

        if plots is None:
            plots = ['walkers', 'corner', 'bestfit', 'residual']
        plots = np.atleast_1d(plots)
        if 'none' in plots:
            plots = []
        if 'walkers' in plots:
            if mcmc == 'emcee':
                # emcee 3.x deprecates sampler.chain in favour of get_chain;
                # transpose (nsteps, nwalkers, ndim) -> (ndim, nsteps, nwalkers)
                walkers = np.transpose(sampler.get_chain(), (2, 0, 1))
            else:
                # zeus and the numpyro adapter expose a writable .chain;
                # zeus.chain is (nsteps, nwalkers, ndim) and the adapter is
                # (num_chains, total_steps, ndim) -- both reduce to
                # (ndim, ..., nwalkers/num_chains) via rollaxis(2).
                walkers = np.rollaxis(sampler.chain.copy(), 2)
            plot_walkers(walkers, nburnin[-1], labels)
        if 'corner' in plots:
            plot_corner(samples, labels)
        if 'bestfit' in plots:
            self.plot_model(samples=samples,
                            params=params,
                            mask=self.ivar,
                            draws=10)
        if 'residual' in plots:
            self.plot_model_residual(samples=samples,
                                     params=params,
                                     mask=self.ivar,
                                     draws=10)

        # Generate the output.

        to_return = []

        if returns is None:
            returns = ['samples']
        returns = np.atleast_1d(returns)

        if 'none' in returns:
            return None
        if 'samples' in returns:
            to_return += [samples]
        if 'sampler' in returns:
            to_return += [sampler]
        if 'lnprob' in returns:
            to_return += [sampler.lnprobability[nburnin:]]
        if 'percentiles' in returns:
            to_return += [np.percentile(samples, [16, 50, 84], axis=0)]
        if 'dict' in returns:
            to_return += [medians]
        if 'model' in returns or 'residual' in returns:
            model = self.evaluate_models(samples, params)
            if 'model' in returns:
                to_return += [model]
            if 'residual' in returns:
                to_return += [self.data * 1e3 - model]

        return to_return if len(to_return) > 1 else to_return[0]

    def fit_annuli(self, rpnts=None, rbins=None, x0=0.0, y0=0.0, inc=0.0,
                   PA=0.0, z0=0.0, psi=1.0, r_cavity=0.0, r_taper=np.inf,
                   q_taper=1.0, z_func=None, shadowed=False, phi_min=None,
                   phi_max=None, exclude_phi=False, abs_phi=False,
                   mask_frame='disk', user_mask=None, fit_vrad=True,
                   fix_vlsr=None, beam_spacing=0, niter=1, plots=None,
                   returns=None, optimize_kwargs=None, MCMC=False):
        r"""
        Splits the map into concentric annuli based on the geometrical
        parameters, then fits each annnulus with a simple harmonic oscillator
        model,

        .. math::

            v_0(\phi) = v_{\phi} \cos(\phi) \sin(|i|) - v_{\rm r} \sin(\phi) \sin(i) - v_{\rm z} \cos(i) + v_{\rm lsr}

        where :math:`i` is the inclination of the disk.

        .. note::
            If you find negative :math:`v_{\phi}` values then your chosen
            position angle is likely off by 180 degrees.

        Args:
            rpnts (Optional[array]): Array of radial position in [arcsec] to
                center the annuli on. Only ``rpnts`` or ``rbins`` need to be
                set.
            rbins (Optional[array]): Array of annuli edges in [arcsec] to use.
                Only ``rpnts`` or ``rbins`` need to be set.
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
            z_func (Optional[callable]): A user-defined emission surface
                function that will return ``z`` in [arcsec] for a given ``r``
                in [arcsec]. This will override the analytical form.
            shadowed (Optional[bool]): Whether to use the slower, but more
                robust method for deprojecting pixel values.
            phi_min (Optional[float]): Minimum polar angle of the segment of
                the annulus in [degrees]. Note this is the polar angle, not the
                position angle.
            phi_max (Optional[float]): Maximum polar angle of the segment of
                the annulus in [degrees]. Note this is the polar angle, not the
                position angle.
            exclude_phi (Optional[bool]): If ``True``, exclude the provided
                polar angle range rather than include it.
            abs_phi (Optional[bool]): If ``True``, take the absolute value of
                the polar angle such that it runs from 0 [deg] to 180 [deg].
            mask_frame (Optional[str]): Which frame the radial and azimuthal
                mask is specified in, either ``'disk'`` or ``'sky'``.
            user_mask (Optional[ndarray]): A 2D mask to use.
            fit_vrad (Optional[bool]): Whether to include radial velocities in
                the fit. Default is ``True``.
            fix_vlsr (Optional[float]): Fix the systemic velocity to this value
                such that returned velocity component is the deprojected
                vertical velocity.
            beam_spacing (Optional[int/float]): If provided, sample pixels that
                are roughly this fraction of a beam separated.
            niter (Optional[int]): Run ``niter`` iterations. Should only be
                used when `beam_spacing > 0`.
            plots (Optional[list]): Plots to generate after the fitting. Can be
                either of ``'model'`` and ``'residual'``. Default is both.
            returns (Optional[list]): List of objects to return. Can be any of
                ``'profiles'``, ``'model'`` or ``'residual'``.
            optimize_kwargs (Optional[dict]): Kwargs to pass to
                ``scipy.optimize.curve_fit``.

        Returns:
            Depends on the value of ``returns``.
        """

        # Remove possbility to run niter > 1 with beam_spacing = 0.

        if niter > 1 and beam_spacing == 0:
            print("WARNING: Can't run multiple iterations using all pixels.")
            print("\t Setting niter = 1 and continuing.")

        # Get the radial binning and deprojected radius and phi values.

        rpnts, rbins = self._get_radial_bins(rpnts=rpnts,
                                             rbins=rbins)
        
        rvals, pvals = self.disk_coords(x0=x0,
                                        y0=y0,
                                        inc=inc,
                                        PA=PA,
                                        z0=z0,
                                        psi=psi,
                                        r_cavity=r_cavity,
                                        r_taper=r_taper,
                                        q_taper=q_taper,
                                        z_func=z_func,
                                        shadowed=shadowed)[:2]

        # Empty lists to hold the results. `velo_proj` are the projected
        # velocities, i.e., the {A, B, C} parameters from a SHO fit. `velo` are
        # these values deprojected accounting for the disk inclination and
        # rotation.

        velo, dvelo = [], []
        empty = [np.nan, np.nan, np.nan, np.nan]

        # Cycle through each annulus to include the fit.

        if MCMC:
            raise NotImplementedError("MCMC fitting in fit_annuli is not "
                                      "supported.")

        for r_min, r_max in zip(rbins[:-1], rbins[1:]):

            # If `niter > 1` then the value and uncertainty returned will be
            # the uncertainty-weighted average and standard deviation of the
            # samples. Each iteration draws an independent random rotation
            # in get_annulus()'s beam-spacing thinning step.

            velo_tmp = []
            dvelo_tmp = []

            for _ in range(niter):
                try:
                    ann = self.get_annulus(r_min=r_min, r_max=r_max,
                                           phi_min=phi_min, phi_max=phi_max,
                                           exclude_phi=exclude_phi,
                                           abs_phi=abs_phi, x0=x0, y0=y0,
                                           inc=inc, PA=PA, z0=z0, psi=psi,
                                           r_cavity=r_cavity,
                                           r_taper=r_taper, q_taper=q_taper,
                                           z_func=z_func, shadowed=shadowed,
                                           mask_frame=mask_frame,
                                           user_mask=user_mask,
                                           beam_spacing=beam_spacing)
                except ValueError:
                    velo_tmp += [empty]
                    dvelo_tmp += [empty]
                    continue

                if ann.theta.size < 2:
                    velo_tmp += [empty]
                    dvelo_tmp += [empty]
                    continue

                # Fit on projected coefficients, deproject via the inclination
                # inside Annulus2D.get_vlos. ``vrad`` follows the convention
                # that positive values describe motion away from the star.

                try:
                    popt, cvar = ann.get_vlos(fit_method='SHO',
                                              fit_vrad=fit_vrad,
                                              fix_vlsr=fix_vlsr,
                                              optimize_kwargs=optimize_kwargs)
                except ValueError:
                    velo_tmp += [empty]
                    dvelo_tmp += [empty]
                    continue

                # Pad to four components: [v_rot, v_rad, v_alt, v_lsr].

                velo_tmp += [[popt[0],
                              popt[1] if fit_vrad else 0.0,
                              0.0 if fix_vlsr is None else popt[-1],
                              popt[-1] if fix_vlsr is None else fix_vlsr]]
                dvelo_tmp += [[cvar[0],
                               cvar[1] if fit_vrad else 0.0,
                               0.0 if fix_vlsr is None else cvar[-1],
                               cvar[-1] if fix_vlsr is None else 0.0]]

            # Combine the values using a weighted average if niter > 1.
            # velo_tmp.shape = [niter, 4]
    
            velo_tmp = np.array(velo_tmp)
            dvelo_tmp = np.array(dvelo_tmp)
            if niter == 1:
                velo += [velo_tmp[0]]
                dvelo += [dvelo_tmp[0]]
            elif niter > 1:
                scatter = 1e-10 * np.random.randn(dvelo_tmp.size)
                weights = np.where(dvelo_tmp != 0.0, 1.0 / dvelo_tmp, 0.0)
                weights = weights + scatter.reshape(weights.shape)
                w_mu = np.average(velo_tmp, weights=weights, axis=0)
                wstd = np.sum(weights * (velo_tmp - w_mu[None, :])**2, axis=0)
                wstd *= niter / (niter - 1.0) / np.sum(weights, axis=0)
                wstd = np.sqrt(wstd)
                velo += [w_mu]
                dvelo += [wstd]
            else:
                raise ValueError("Unknown `niter` value.")
            
        # Combine all the results into [4, rpnts] shaped arrays to deproject.

        velo = np.atleast_2d(np.squeeze(velo)).T
        dvelo = np.atleast_2d(np.squeeze(dvelo)).T
        assert dvelo.shape == velo.shape
        assert velo.shape[0] == 4

        velo = np.where(np.isfinite(velo), velo, np.nan)
        dvelo = np.where(np.isfinite(dvelo), dvelo, np.nan)

        # Build the linearly interpolated model noting that the velocities need
        # to be projected into the sky.

        velo_proj = np.array([velo[0] * abs(np.sin(np.radians(inc))),
                              velo[1] * -np.sin(np.radians(inc)),
                              velo[2] * -np.cos(np.radians(inc)),
                              velo[3]])
        
        model = self._evaluate_annuli_model(rpnts=rpnts,
                                            velo_proj=velo_proj,
                                            rvals=rvals,
                                            pvals=pvals)

        # Make the plots.

        plots = ['profiles', 'model', 'residual'] if plots is None else plots
        plots = np.atleast_1d(plots)
        if 'profiles' in plots:
            self.plot_velocity_profiles(rpnts=rpnts,
                                        velo=velo,
                                        dvelo=dvelo)
        if 'model' in plots:
            self.plot_model(model=model)
        if 'residual' in plots:
            self.plot_model_residual(model=model)

        # Generate the retuns.

        to_return = []
        returns = ['profiles'] if returns is None else np.atleast_1d(returns)
        returns = np.atleast_1d(returns)
        if 'profiles' in returns:
            to_return += [rpnts, velo, dvelo]
        if 'model' in returns:
            to_return += [model]
        if 'residual' in returns:
            to_return += [self.data - model]
        return to_return

    def _evaluate_annuli_model(self, rpnts, velo_proj, rvals, pvals):
        """
        Evaluate the annuli models onto a 2D map. The velocity profiles must be
        projected into the plane of the sky.

        Args:
            rpnts (array): A size M array of radial positions.
            velo_proj (array): A [4xM] array containing the inferred velocity
                coefficients from a double SHO fit.
            rvals (array): A 2D array of on-sky radial positions in [arcsec].
            pvals (array): A 2D array of on-sky polar angle in [radians].

        Returns:
            v0 (array): Model of the projected velocity in [m/s].
        """
        from scipy.interpolate import interp1d
        from .helper_functions import SHO_double
        A = interp1d(rpnts, velo_proj[0], bounds_error=False)(rvals)
        B = interp1d(rpnts, velo_proj[1], bounds_error=False)(rvals)
        C = interp1d(rpnts, velo_proj[2] + velo_proj[3], bounds_error=False)(rvals)
        return SHO_double(pvals, A, B, C)

    def _get_radial_bins(self, rpnts=None, rbins=None):
        """Return default radial bins."""
        if rpnts is None and rbins is None:
            rbins = np.arange(self.bmaj / 2, self.xaxis[0], self.bmaj / 4)
            rpnts = np.mean([rbins[1:], rbins[:-1]], axis=0)
        elif rpnts is None:
            rpnts = np.mean([rbins[1:], rbins[:-1]], axis=0)
        else:
            dr = np.diff(rpnts) / 2.0
            dr = np.insert(dr, -1, dr[-1])
            if not np.all(np.isclose(dr, dr[0])):
                print("WARNING: Non-linear `rpnts` found. Check results!")
            rbins = np.insert(rpnts + dr, 0, rpnts[0] - dr[0])
        return rpnts, rbins

    def set_prior(self, param, args, type='flat'):
        """
        Set the prior for the given parameter used in ``fit_map``. Overwrites
        any previously set prior (including the default) for that parameter.
        There are two types of priors currently usable, ``'flat'`` which
        requires ``args=[min, max]`` while for ``'gaussian'`` you need to
        specify ``args=[mu, sig]``.

        Args:
            param (str): Name of the parameter to set the prior for.
            args (list): Boundary values ``[min, max]`` for a flat prior or
                ``[mu, sigma]`` for a Gaussian prior.
            type (Optional[str]): Type of prior, either ``'flat'`` or
                ``'gaussian'``. Default is ``'flat'``.
        """
        type = type.lower()
        if type not in ['flat', 'gaussian']:
            raise ValueError("type must be 'flat' or 'gaussian'.")
        if type == 'flat':
            lo, hi = float(min(args)), float(max(args))
            width = hi - lo
            # Improper-uniform priors (one bound at +/- inf) have an
            # undefined normalisation; fall back to the same -100 floor
            # used for very wide proper priors.
            if np.isfinite(width) and width > 0:
                log_density = max(-100.0, -np.log(width))
            else:
                log_density = -100.0
            def prior(p):
                if not lo <= p <= hi:
                    return -np.inf
                return log_density
            def prior_jax(p):
                in_bounds = jnp.logical_and(p >= lo, p <= hi)
                return jnp.where(in_bounds, log_density, -jnp.inf)
            rotationmap.priors_jax[param] = ('flat', lo, hi, prior_jax)
        else:
            mu, sigma = float(args[0]), float(args[1])
            def prior(p):
                return -0.5 * ((mu - p) / sigma)**2
            def prior_jax(p):
                return -0.5 * ((mu - p) / sigma)**2
            rotationmap.priors_jax[param] = ('gaussian', mu, sigma, prior_jax)
        rotationmap.priors[param] = prior

    def set_SHO_prior(self, param, args, type='flat'):
        """
        Set the prior for the given parameter used in ``fit_annuli``. Overwrites
        any previously set prior (including the default) for that parameter.
        There are two types of priors currently usable, ``'flat'`` which
        requires ``args=[min, max]`` while for ``'gaussian'`` you need to
        specify ``args=[mu, sig]``. The three ``params`` which are available are
        ``'vrot'``, ``'vrad'`` and ``'vlsr'``.

        Args:
            param (str): Name of the parameter to set the prior for. Must be
                one of ``'vrot'``, ``'vrad'``, or ``'vlsr'``.
            args (list): Boundary values ``[min, max]`` for a flat prior or
                ``[mu, sigma]`` for a Gaussian prior.
            type (Optional[str]): Type of prior, either ``'flat'`` or
                ``'gaussian'``. Default is ``'flat'``.
        """
        type = type.lower()
        if type not in ['flat', 'gaussian']:
            raise ValueError("type must be 'flat' or 'gaussian'.")
        if type == 'flat':
            def prior(p):
                if not min(args) <= p <= max(args):
                    return -np.inf
                return np.log(1.0 / (args[1] - args[0]))
        else:
            def prior(p):
                return -0.5 * ((args[0] - p) / args[1])**2
        rotationmap.SHO_priors[param] = prior

    def plot_data(self, vmin=None, vmax=None, ivar=None, return_fig=False):
        """
        Plot the first moment map. By default will clip the velocity contours
        such that the velocities around the systemic velocity are highlighted.

        Args:
            levels (optional[list]): List of contour levels to use.
            ivar (optional[ndarray]): Inverse variances for each pixel. Will
                draw a solid contour around the regions with finite ``ivar``
                values and fill regions not considered.
            return_fig (optional[bool]): Return the figure.

        Returns:
            fig (Matplotlib figure): If ``return_fig`` is ``True``. Can access
                the axes through ``fig.axes`` for additional plotting.
        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
    
        if vmin is None or vmax is None:
            vmin_tmp, vmax_tmp = np.nanpercentile(self.data, [2, 98])
            vmax_tmp = max(abs(vmin_tmp - self.vlsr), abs(vmax_tmp - self.vlsr))
            vmin_tmp = (self.vlsr - vmax_tmp) / 1e3
            vmax_tmp = (self.vlsr + vmax_tmp) / 1e3
            if vmin is None:
                vmin = vmin_tmp
            if vmax is None:
                vmax = vmax_tmp

        im = ax.imshow(self.data / 1e3, origin='lower', extent=self.extent,
                       vmin=vmin, vmax=vmax,
                       cmap=rotationmap.cmap(), zorder=-9)
        cb = plt.colorbar(im, pad=0.03, extend='both', format='%.2f')
        cb.minorticks_on()
        cb.set_label(r'${\rm v_{0} \quad (km\,s^{-1})}$',
                     rotation=270, labelpad=15)
        if ivar is not None:
            ax.contour(self.xaxis, self.yaxis, ivar,
                       [0.0], colors='k')
            ax.contourf(self.xaxis, self.yaxis, ivar,
                        [-1.0, 0.0], colors='k', alpha=0.5)
        self._gentrify_plot(ax)
        if return_fig:
            return fig

    # -- MCMC Functions -- #

    def _optimize_p0(self, theta, params, **kwargs):
        """Optimize the initial starting positions.

        Uses ``scipy.optimize.minimize`` with method ``L-BFGS-B`` and an
        analytic gradient computed via ``jax.grad`` of a JAX-traceable
        negative log-probability. Bounds are derived from any flat priors
        on free parameters; Gaussian-prior parameters are unbounded.
        Falls back to the older finite-difference path with a default
        method of ``L-BFGS-B`` if the autodiff path raises (e.g. because
        a user-installed prior is not JAX-traceable).
        """
        from scipy.optimize import minimize

        free_keys = self._free_parameter_keys(params)
        bounds = self._optimize_bounds(free_keys)

        method = kwargs.pop('method', 'L-BFGS-B')
        options = kwargs.pop('options', {})
        options['maxiter'] = options.pop('maxiter', 10000)
        options['ftol'] = options.pop('ftol', 1e-3)

        try:
            grad_fn = jax.jit(jax.grad(
                lambda t: self._neg_ln_prob_jax(t, params)
            ))
            # Warm the JIT cache and verify the gradient is finite at p0,
            # so we fail fast into the fallback path if autodiff cannot be
            # applied to this parameter set.
            g0 = np.asarray(grad_fn(jnp.asarray(theta)))
            if not np.all(np.isfinite(g0)):
                raise ValueError(
                    "Initial gradient contains non-finite values."
                )
            def nlnL(theta_np):
                return float(self._neg_ln_prob_jax(jnp.asarray(theta_np),
                                                   params))
            def jac(theta_np):
                return np.asarray(grad_fn(jnp.asarray(theta_np)))
            res = minimize(nlnL, x0=np.asarray(theta), jac=jac,
                           method=method, bounds=bounds, options=options)
        except Exception as exc:
            warnings.warn(
                "Falling back to finite-difference optimisation; "
                "autodiff path raised: {}".format(exc)
            )
            def nlnL(theta_np):
                return -self._ln_probability(theta_np, params)
            res = minimize(nlnL, x0=np.asarray(theta), method=method,
                           bounds=bounds, options=options)
        if res.success:
            theta = res.x
            print("Optimized starting positions:")
        else:
            print("WARNING: scipy.optimize did not converge.")
            print("Starting positions:")
        print('\tp0 =', ['%.2e' % t for t in theta])
        return theta

    def _run_mcmc(self, p0, params, nwalkers, nburnin, nsteps, mcmc, **kwargs):
        """Run the MCMC sampling. Returns the sampler (or sampler-shaped
        adapter when ``mcmc='numpyro'``)."""

        if mcmc == 'numpyro':
            kwargs.pop('pool', None)   # numpyro handles parallelism via JAX
            kwargs.pop('moves', None)  # ensemble-only concept
            return self._run_nuts(p0=p0, params=params,
                                  num_chains=int(nwalkers),
                                  num_warmup=int(nburnin),
                                  num_samples=int(nsteps),
                                  **kwargs)

        if mcmc == 'zeus':
            EnsembleSampler = zeus.EnsembleSampler
        else:
            EnsembleSampler = emcee.EnsembleSampler

        p0 = random_p0(p0, kwargs.pop('scatter', 1e-3), nwalkers)
        moves = kwargs.pop('moves', None)
        pool = kwargs.pop('pool', None)

        # Single-process: build a vmap'd JIT closure that handles the
        # whole walker batch in one XLA dispatch. emcee calls the
        # log-prob once per step (via ``vectorize=True``) instead of
        # nwalkers times in a Python loop, and XLA vectorises the
        # model evaluation across walkers. ~5x speedup per walker-eval
        # on top of the per-call JIT win.
        #
        # ``Pool``: the JIT'd closure isn't picklable (ForkingPickler
        # requires top-level functions / bound methods). Fall back to
        # the per-walker ``self._ln_probability`` method which is
        # picklable; cross-core parallelism then replaces JIT'd batch
        # vectorisation. (Tutorial 2 is the canonical example.)
        if pool is None:
            ln_prob_fn = self._build_vectorized_ln_probability(params)
            sampler = EnsembleSampler(nwalkers,
                                      p0.shape[1],
                                      ln_prob_fn,
                                      args=[params, np.nan],
                                      moves=moves,
                                      pool=pool,
                                      vectorize=True)
        else:
            sampler = EnsembleSampler(nwalkers,
                                      p0.shape[1],
                                      self._ln_probability,
                                      args=[params, np.nan],
                                      moves=moves,
                                      pool=pool)

        progress = kwargs.pop('progress', True)

        sampler.run_mcmc(p0, nburnin + nsteps, progress=progress, **kwargs)

        return sampler

    def _build_vectorized_ln_probability(self, params):
        """Return a ``(coords, *unused) -> ndarray`` callable for
        ``emcee.EnsembleSampler(..., vectorize=True)`` that JIT-compiles
        a vmap'd log-posterior once and reuses the compiled function
        across all walker batches.

        ``coords`` has shape ``(nwalkers_subset, ndim)`` -- emcee's
        red-blue move feeds half the walker complement per call. XLA
        vectorises the model evaluation across the batch axis, so one
        compiled call handles every walker in the subset.

        The free-parameter values vary per call; everything else in
        ``params`` (vfunc, boolean flags like ``beam`` / ``vortex``,
        fixed scalars like ``inc`` / ``dist``, the deprojection
        callables, and ``self.data`` / ``self.mask`` / ``self.ivar``)
        is captured at trace time and baked in as constants. As long as
        the dict's structural keys don't change (they don't, inside a
        single ``_run_mcmc``), JAX hits the cached trace on every
        iteration. The prior is applied inside the trace via
        :meth:`_ln_prior_jax`, which propagates ``-inf`` cleanly through
        the sum so out-of-bounds proposals are rejected without an
        explicit Python short-circuit."""

        free_keys = rotationmap._free_parameter_keys(params)
        static_params = {k: v for k, v in params.items()
                         if k not in free_keys}

        def single_ln_prob(theta):
            full = dict(static_params)
            for i, k in enumerate(free_keys):
                full[k] = theta[i]
            return self._ln_prior_jax(full) + self._ln_likelihood(full)

        jit_lnP_batch = jax.jit(jax.vmap(single_ln_prob))

        def vectorized_ln_probability(coords, *_params_in):
            return np.asarray(jit_lnP_batch(jnp.asarray(coords)))

        return vectorized_ln_probability

    def _run_nuts(self, p0, params, num_chains, num_warmup, num_samples,
                  **kwargs):
        """Run numpyro NUTS over :meth:`_numpyro_model_fitmap` and return a
        :class:`_NumpyroSampler` adapter. ``num_chains`` / ``num_warmup`` /
        ``num_samples`` map onto the legacy ``nwalkers`` / ``nburnin`` /
        ``nsteps`` kwargs of :meth:`fit_map`. ``scatter`` controls the
        per-chain spread of ``p0`` (matching emcee semantics); ``seed`` and
        ``chain_method`` are forwarded to numpyro."""
        if not _HAS_NUMPYRO:
            raise ImportError(
                "numpyro is required for mcmc='numpyro'. "
                "Install with `pip install numpyro`."
            )
        import numpyro.infer as ni
        from numpyro.infer.initialization import init_to_value

        free_keys = self._free_parameter_keys(params)
        ndim = len(free_keys)

        kwargs.pop('scatter', None)   # legacy emcee kwarg, not needed for NUTS
        seed = kwargs.pop('seed', 0)
        progress = kwargs.pop('progress', True)
        chain_method = kwargs.pop('chain_method', 'sequential')

        # Seed every chain at the (already-optimised) p0 in constrained
        # parameter space; numpyro will transform to the unconstrained
        # space internally. NUTS adapts the step size during warmup, so
        # we don't need per-chain scatter the way emcee/zeus do.
        p0 = np.asarray(p0).astype(float)
        init_strategy = init_to_value(
            values={k: float(p0[idx]) for idx, k in enumerate(free_keys)}
        )

        nuts = ni.NUTS(self._numpyro_model_fitmap,
                       init_strategy=init_strategy, **kwargs)
        mcmc = ni.MCMC(nuts,
                       num_warmup=num_warmup,
                       num_samples=num_samples,
                       num_chains=num_chains,
                       progress_bar=progress,
                       chain_method=chain_method)
        mcmc.run(jax.random.PRNGKey(seed),
                 params=params,
                 extra_fields=('potential_energy',))

        samples_dict = mcmc.get_samples(group_by_chain=True)
        chain_post = jnp.stack([samples_dict[k] for k in free_keys], axis=-1)
        warmup_pad = jnp.full((num_chains, num_warmup, ndim), jnp.nan)
        chain = jnp.concatenate([warmup_pad, chain_post], axis=1)

        pe = mcmc.get_extra_fields(group_by_chain=True)['potential_energy']
        lnprob_post = -np.asarray(pe).T
        lnprob_pad = np.full((num_warmup, num_chains), np.nan)
        lnprobability = np.concatenate([lnprob_pad, lnprob_post], axis=0)

        return _NumpyroSampler(np.asarray(chain), lnprobability)

    def _ln_likelihood(self, params):
        """Log-likelihood function. Simple chi-squared likelihood.

        ``self.data`` contains NaN outside the disk; ``where(mask, ...)``
        masks them out forward, but autodiff still propagates the NaN
        from ``data - model`` at masked pixels and poisons the gradient.
        Replacing NaN with 0 *before* the difference keeps the diff
        finite everywhere; ``self.ivar`` is already 0 at masked pixels
        so the contribution stays correct."""
        model = self._make_model(params)
        data = jnp.where(self.mask, jnp.asarray(self.data), 0.0)
        lnx2 = -0.5 * jnp.sum((data - model) ** 2 * self.ivar)
        return jnp.where(jnp.isfinite(lnx2), lnx2, -jnp.inf)

    def _ln_prior_jax(self, params):
        """JAX-traceable log-prior. Mirrors :meth:`_ln_prior` but uses
        ``rotationmap.priors_jax`` so the result composes with
        ``jax.grad``. Out-of-bounds flat priors return ``-jnp.inf``."""
        lnp = jnp.array(0.0)
        for key in params.keys():
            if key in rotationmap.priors_jax and params[key] is not None:
                _, _, _, prior_jax = rotationmap.priors_jax[key]
                lnp = lnp + prior_jax(params[key])
        return lnp

    def _neg_ln_prob_jax(self, theta, params):
        """JAX-traceable negative log-posterior. Used by
        :meth:`_optimize_p0` as the objective for ``jax.grad``."""
        model = rotationmap._populate_dictionary(theta, params)
        lnp = self._ln_prior_jax(model)
        ll = self._ln_likelihood(model)
        return -(lnp + ll)

    @staticmethod
    def _free_parameter_keys(params):
        """Return the parameter names that are free (placeholder ints) in
        the order matching the ``theta`` index they reference."""
        free = [(v, k) for k, v in params.items()
                if isinstance(v, int) and not isinstance(v, bool)]
        free.sort()
        return [k for _, k in free]

    @staticmethod
    def _optimize_bounds(free_keys):
        """Build ``L-BFGS-B`` bounds from the registered priors. Flat
        priors give finite bounds; Gaussian (unbounded) priors and
        params without a registered prior get ``(None, None)``."""
        bounds = []
        for key in free_keys:
            spec = rotationmap.priors_jax.get(key)
            if spec is not None and spec[0] == 'flat':
                bounds.append((spec[1], spec[2]))
            else:
                bounds.append((None, None))
        return bounds

    def _ln_probability(self, theta, *params_in):
        """Log-probablility function."""
        model = rotationmap._populate_dictionary(theta, params_in[0])
        lnp = self._ln_prior(model)
        if np.isfinite(lnp):
            return lnp + self._ln_likelihood(model)
        return -np.inf

    def _numpyro_model_fitmap(self, params):
        """numpyro model for :meth:`fit_map`. Draws each free parameter from
        its registered ``priors_jax`` entry, builds the velocity model via
        :meth:`_make_model`, and observes the data with per-pixel Gaussian
        noise derived from ``self.ivar`` (which already has the fitting mask
        baked in). Pixels with ``ivar == 0`` or non-finite data are masked
        out of the likelihood.

        Flat priors with one or both bounds at +/- inf cannot be expressed
        as ``dist.Uniform`` (numpyro can't sample from an unbounded uniform
        — its log_prob is improper). They become ``dist.ImproperUniform``
        on the matching constraint, which preserves the emcee/zeus
        semantics (constant log-density inside the support)."""
        import numpyro
        import numpyro.distributions as dist
        from numpyro.distributions import constraints

        free_keys = self._free_parameter_keys(params)
        theta = []
        for name in free_keys:
            spec = rotationmap.priors_jax[name]
            if spec[0] == 'flat':
                _, lo, hi, _ = spec
                lo_finite = np.isfinite(lo)
                hi_finite = np.isfinite(hi)
                if lo_finite and hi_finite:
                    prior = dist.Uniform(lo, hi)
                elif lo_finite:
                    prior = dist.ImproperUniform(
                        constraints.greater_than(lo), (), ())
                elif hi_finite:
                    prior = dist.ImproperUniform(
                        constraints.less_than(hi), (), ())
                else:
                    prior = dist.ImproperUniform(constraints.real, (), ())
                theta.append(numpyro.sample(name, prior))
            else:
                _, mu, sigma, _ = spec
                theta.append(numpyro.sample(name, dist.Normal(mu, sigma)))
        theta = jnp.stack(theta)

        populated = rotationmap._populate_dictionary(theta, params)
        model = self._make_model(populated)

        ivar = self.ivar
        valid = jnp.logical_and(jnp.asarray(self.mask), ivar > 0)
        sigma = 1.0 / jnp.sqrt(jnp.where(ivar > 0, ivar, 1.0))
        data = jnp.where(valid, jnp.asarray(self.data), 0.0)
        numpyro.sample(
            'obs',
            dist.Normal(model, sigma).mask(valid),
            obs=data,
        )

    def _load_default_parameters(self, path='default_parameters.yml'):
        """Load the default parameters."""
        with open(__file__.replace('rotationmap.py', path)) as stream:
            parameters = yaml.safe_load(stream)
        for p in parameters.keys():
            if parameters[p]['prior_type'] is not None:
                values = parameters[p]['prior_values']
                if len(values) == 1:
                    parameters[p]['prior_values'] = [-values[0], values[0]]
        return parameters

    def print_default_prior(self, parameter):
        """
        Print the default prior for a given parameter to stdout.

        Args:
            parameter (str): Name of the parameter whose default prior to
                display. If the parameter is not a recognized free parameter, a
                warning message is printed instead.
        """
        try:
            prior_type = self.default_parameters[parameter]['prior_type']
            prior_values = self.default_parameters[parameter]['prior_values']
            str = '`{}` has a '
            if prior_type == 'flat':
                str += '{} prior with a minimum value of {}'
                str += ' and a maximum value {}.'
            elif prior_type == 'gaussian':
                str += '{} prior with mean of {}'
                str += ' and standard deviation of {}.'
            print(str.format(parameter, prior_type, *prior_values))
        except KeyError:
            print('`{}` is not a free parameter.'.format(parameter))

    def _set_default_priors(self):
        """Set the default priors."""

        # fit_map functions

        for k in self.default_parameters.keys():
            p = self.default_parameters[k]
            if p['prior_type'] is not None:
                self.set_prior(k, p['prior_values'], p['prior_type'])

        # SHO functions.

        self.set_SHO_prior('vrot', [-3e3, 3e3], 'flat')
        self.set_SHO_prior('vrad', [-1e2, 1e2], 'flat')
        self.set_SHO_prior('vlsr', [-1e4, 1e4], 'flat')

    def _ln_prior(self, params):
        """Log-priors."""
        lnp = 0.0
        for key in params.keys():
            if key in rotationmap.priors.keys() and params[key] is not None:
                try:
                    lnp += rotationmap.priors[key](params[key])
                except:
                    print(key, params[key])
                if not np.isfinite(lnp):
                    return lnp
        return lnp

    def _calc_ivar(self, params):
        """Calculate the inverse variance including radius mask."""

        # Check the error array is the same shape as the data.

        try:
            assert self.error.shape == self.data.shape
        except AttributeError:
            self.error = self.error * np.ones(self.data.shape)

        # Cast data and error to jnp once. astropy reads FITS data in the
        # file's native (typically big-endian) byte order, which JAX
        # otherwise refuses to trace.

        data = jnp.asarray(self.data)
        error = jnp.asarray(self.error)

        # Deprojected coordinates.

        r, t = self.disk_coords(**params)[:2]
        t = jnp.abs(t) if params['abs_phi'] else t

        # Radial mask.

        mask_r = jnp.logical_and(r >= params['r_min'], r <= params['r_max'])
        mask_r = jnp.logical_not(mask_r) if params['exclude_r'] else mask_r

        # Azimuthal mask.

        mask_p = jnp.logical_and(t >= jnp.radians(params['phi_min']),
                                 t <= jnp.radians(params['phi_max']))
        mask_p = jnp.logical_not(mask_p) if params['exclude_phi'] else mask_p

        # Finite value mask.

        mask_f = jnp.logical_and(jnp.isfinite(data), error > 0.0)

        # Velocity mask.

        v_min = params.get('v_min', np.nanmin(self.data))
        v_max = params.get('v_max', np.nanmax(self.data))
        mask_v = jnp.logical_and(data >= v_min, data <= v_max)
        mask_v = jnp.logical_not(mask_v) if params['exclude_v'] else mask_v

        # Combine with the user_mask.

        mask = jnp.logical_and(jnp.logical_and(mask_v, mask_f),
                               jnp.logical_and(mask_r, mask_p))
        mask = jnp.logical_and(mask, params['user_mask'])
        return jnp.where(mask, jnp.power(error, -2.0), 0.0)

    @staticmethod
    def _get_labels(params):
        """Return the labels of the parameters to fit."""
        idxs, labs = [], []
        for k in params.keys():
            if isinstance(params[k], int):
                if not isinstance(params[k], bool):
                    idxs.append(params[k])
                    try:
                        idx = k.index('_') + 1
                        label = k[:idx] + '{{' + k[idx:] + '}}'
                    except ValueError:
                        label = k
                    label = r'${{\rm {}}}$'.format(label)
                    labs.append(label)
        return np.array(labs)[np.argsort(idxs)]

    @staticmethod
    def _populate_dictionary(theta, dictionary_in):
        """Populate the dictionary of free parameters."""
        dictionary = dictionary_in.copy()
        for key in dictionary.keys():
            if isinstance(dictionary[key], int):
                if not isinstance(dictionary[key], bool):
                    dictionary[key] = theta[dictionary[key]]
        return dictionary

    def verify_params_dictionary(self, params):
        """
        Check that the minimum number of parameters are provided for the
        fitting and fill in defaults for any that are missing. Sets the
        rotation velocity function (``'vfunc'``) based on whether a power-law
        profile or Keplerian profile is requested, and flags whether a vortex
        component should be included.

        Args:
            params (dict): Dictionary of model parameters, as passed to
                ``fit_map``. Missing parameters are filled with their default
                values from ``default_parameters.yml``.

        Returns:
            params (dict): The verified and completed parameter dictionary,
                ready for use in model evaluation.
        """

        for p in self.default_parameters.keys():
            params[p] = params.pop(p, self.default_parameters[p]['default'])

        # Rotation profile.

        if params['vp_100'] is not None:
            if params['mstar'] is not None:
                params['vfunc'] = self._vpow
            else:
                raise ValueError("Cannot specify both `vp_100` and `mstar`.")
        else:
            params['vfunc'] = self._vkep

        # Vortex model.

        if params['r0_vortex'] is not None:
            if params['p0_vortex'] is not None:
                params['vortex'] = True
            else:
                raise ValueError("Must specify by `r0_vortex` and `p0_vortex'.")
        else:
            params['vortex'] = False

        # Deprojection properties.

        params['z_func'] = params.pop('z_func', None)
        params['shadowed'] = params.pop('shadowed', False)
        params['user_mask'] = params.pop('user_mask', np.ones(self.data.shape))

        return params

    def evaluate_models(self, samples=None, params=None, draws=50,
                        collapse_func=np.median, coords_only=False,
                        profile_only=False):
        """
        Evaluate models based on the samples provided and the parameter
        dictionary. If ``draws`` is an integer, it represents the number of
        random draws from ``samples`` which are then averaged to return the
        model. Alternatively ``draws`` can be a float between 0 and 1,
        representing the percentile of the samples to use to evaluate the model
        at.

        Args:
            samples (ndarray): An array of samples returned from ``fit_map``.
            params (dict): The parameter dictionary passed to ``fit_map``.
            draws (Optional[int/float]): If an integer, describes the number of
                random draws averaged to form the returned model. If a float,
                represents the percentile used from the samples. Must be
                between 0 and 1 if a float.
            collapse_func (Optional[callable]): How to collapse the random
                number of samples. Must be a function which allows an ``axis``
                argument (as with most Numpy functions).
            coords_only (Optional[bool]): Return the deprojected coordinates
                rather than the v0 model. Default is False.
            profile_only (Optional[bool]): Return the radial profiles of the
                velocity profiles rather than the v0 model. Default is False.

        Returns:
            model (ndarray): The sampled model, either the v0 model, or, if
                ``coords_only`` is True, the deprojected cylindrical
                coordinates, (r, t, z).
        """

        # Check the input.

        if params is None:
            raise ValueError("Must provide model parameters dictionary.")

        # Model is fully specified.

        if samples is None:
            verified_params = self.verify_params_dictionary(params.copy())
            if coords_only:
                return self.disk_coords(**verified_params)
            else:
                if profile_only:
                    return self._make_profile(verified_params)
                else:
                    return self._make_model(verified_params)

        nparam = np.sum([type(params[k]) is int for k in params.keys()])
        if samples.shape[1] != nparam:
            warning = "Invalid number of free parameters in 'samples': {:d}."
            raise ValueError(warning.format(nparam))
        if not callable(collapse_func):
            raise ValueError("'collapse_func' must be callable.")
        verified_params = self.verify_params_dictionary(params.copy())

        # Avearge over a random draw of models.

        if isinstance(int(draws) if draws > 1.0 else draws, int):
            rvals, models = [], []
            for idx in np.random.randint(0, samples.shape[0], draws):
                tmp = self._populate_dictionary(samples[idx], verified_params)
                if coords_only:
                    models += [self.disk_coords(**tmp)]
                elif profile_only:
                    _rval, _model = self._make_profile(tmp)
                    rvals += [_rval]
                    models += [_model]
                else:
                    models += [self._make_model(tmp)]
            models = collapse_func(models, axis=0)
            if profile_only:
                rvals = collapse_func(rvals, axis=0)
                return rvals, models
            return models

        # Take a percentile of the samples.

        elif isinstance(draws, float):
            tmp = np.percentile(samples, draws, axis=0)
            tmp = self._populate_dictionary(tmp, verified_params)
            if coords_only:
                return self.disk_coords(**tmp)
            elif profile_only:
                return self._make_profile(tmp)
            else:
                return self._make_model(tmp)

        else:
            raise ValueError("'draws' must be a float or integer.")
        
    def evaluate_models_vortex(self, samples=None, params=None, draws=50,
            collapse_func=np.median, frame=None):
        """
        Evaluate the vortex model. Same functionality as ``evaluate_models`` but
        only for evaluating the vortex model. The frame of reference can be
        chosen with the options being ``None``, the default which returns just
        the on-sky projection, ``'sky'``, ``'face-on'``, ``'polar'`` and
        ``'vortex'``. The ``'sky'`` frame includes the projection of the
        velocity along the line of sight, while the ``'face-on'``, ``'polar'``
        and ``'vortex'`` frames are the intrinsic velocities. Note that for all
        frames other than ``'sky'`` the model is on an unstructured grid so will
        require gridding if used for a figure.

        Args:
            samples (ndarray): An array of samples returned from ``fit_map``.
            params (dict): The parameter dictionary passed to ``fit_map``.
            draws (Optional[int/float]): If an integer, describes the number of
                random draws averaged to form the returned model. If a float,
                represents the percentile used from the samples. Must be
                between 0 and 1 if a float.
            collapse_func (Optional[callable]): How to collapse the random
                number of samples. Must be a function which allows an ``axis``
                argument (as with most Numpy functions).
            frame (Optional[None/str]): The frame for the projection of the
                velocity components. If ``frame=None`` then the on-sky
                projection is used and no associated coordinates are returned.

        Returns:
            [x, y,] v ([array, array,] array): The vortex velocity model along
                with the associated coordinate values if ``frame`` is specified.
        """

        # Check the input -- must provide at least a `params` dictionary.

        if params is None:
            raise ValueError("Must provide model parameters dictionary.")
        
        # NOTE: Calculate the coordinates needed for this. Note that this won't
        # be exactly the same draws (if draws > 1) but for a well sampled
        # posterior and a large enough draw value this should be OK...
        
        rvals, tvals, _ = self.evaluate_models(samples=samples,
                                               params=params,
                                               draws=draws,
                                               coords_only=True)

        # Model is fully specified and no draws are needed.

        if samples is None:
            verified_params = self.verify_params_dictionary(params.copy())
            return self._make_model_vortex(rvals=rvals, 
                                           tvals=tvals,
                                           params=verified_params,
                                           frame=frame)
        
        # Now do a random number of draws. Check to make sure the `params`
        # dictionary has the same number of free parameters as there are in
        # `samples`.

        nparam = np.sum([type(params[k]) is int for k in params.keys()])
        if samples.shape[1] != nparam:
            warning = "Invalid number of free parameters in 'samples': {:d}."
            raise ValueError(warning.format(nparam))
        if not callable(collapse_func):
            raise ValueError("'collapse_func' must be callable.")
        verified_params = self.verify_params_dictionary(params.copy())

        # Average over draw of random model samples.

        if isinstance(int(draws) if draws > 1.0 else draws, int):
            models = []
            for idx in np.random.randint(0, samples.shape[0], draws):
                tmp = self._populate_dictionary(samples[idx], verified_params)
                models += [self._make_model_vortex(rvals=rvals, 
                                                   tvals=tvals,
                                                   params=tmp,
                                                   frame=frame)]
            return collapse_func(models, axis=0)
        
        # Take a percentile of the samples.

        elif isinstance(draws, float):
            tmp = np.percentile(samples, draws, axis=0)
            tmp = self._populate_dictionary(tmp, verified_params)
            self._make_model(rvals=rvals, 
                             tvals=tvals,
                             params=tmp,
                             frame=frame)

        # Otherwise `draws` is invalid.

        else:
            raise ValueError("'draws' must be a float or integer.")
        
    def save_model(self, samples=None, params=None, model=None, filename=None,
                   overwrite=True):
        """
        Save the model as a FITS file. If you have used ``downsample`` when
        loading the cube data, _this will not work_.

        Args:
            samples (Optional[ndarray]): An array of samples returned from
                ``fit_map``. Used to generate the model if ``model`` is not
                provided.
            params (Optional[dict]): The parameter dictionary passed to
                ``fit_map``. Used to generate the model if ``model`` is not
                provided.
            model (Optional[ndarray]): A pre-computed model array. If not
                provided, ``evaluate_models`` is called with ``samples`` and
                ``params``.
            filename (Optional[str]): Output filename. Defaults to the input
                path with ``'_model.fits'`` replacing ``.fits``.
            overwrite (Optional[bool]): If ``True``, overwrite any existing
                file with the same name. Default is ``True``.
        """
        from astropy.io import fits
        if model is None:
            model = self.evaluate_models(samples, params)
        if self.header['naxis1'] > self.nypix:
            canvas = np.ones(self._original_shape) * np.nan
            canvas[self._ya:self._yb, self._xa:self._xb] = model
            model = canvas.copy()
        if filename is None:
            filename = self.path.replace('.fits', '_model.fits')
        fits.writeto(filename, model, self.header, overwrite=overwrite)

    def mirror_residual(self, samples, params, mirror_velocity_residual=True,
                        mirror_axis='minor', return_deprojected=False,
                        deprojected_dpix_scale=1.0):
        """
        Return the residuals after subtracting a mirror image of either the
        rotation map, as in _Huang et al. 2018, or the residuals, as in
        _Izquierdo et al. 2021.

        .. _Huang et al. 2018: https://ui.adsabs.harvard.edu/abs/2018ApJ...867....3H/abstract
        .. _Izquierdo et al. 2021: https://ui.adsabs.harvard.edu/abs/2021arXiv210409530V/abstract

        Args:
            samples (ndarray): An array of samples returned from ``fit_map``.
            params (dict): The parameter dictionary passed to ``fit_map``.
            mirror_velocity_residual (Optional[bool]): If ``True``, the
                default, mirror the velocity residuals, otherwise use the line
                of sight velocity corrected for the systemic velocity.
            mirror_axis (Optional[str]): Which axis to mirror the image along,
                either ``'minor'`` or ``'major'``.
            return_deprojected (Optional[bool]): If ``True``, return the
                deprojected image, or, if ``False``, reproject the data onto
                the sky.

        Returns:
            x, y, residual (array, array, array): The x- and y-axes of the
                residual (either the sky axes or deprojected, depending on the
                chosen arguments) and the residual.
        """

        # Calculate the image to mirror.

        if mirror_velocity_residual:
            to_mirror = self.data * 1e3 - self.evaluate_models(samples, params)
        else:
            to_mirror = self.data.copy() * 1e3
            if isinstance(type(params['vlsr']), int):
                to_mirror -= params['vlsr']
            else:
                to_mirror -= np.median(samples, axis=0)[params['vlsr']]

        # Generate the axes for the deprojected image.

        r, t, _ = self.evaluate_models(samples, params, coords_only=True)
        t += np.pi / 2.0 if mirror_axis.lower() == 'minor' else 0.0
        x = np.nanmax(np.where(np.isfinite(to_mirror), r, np.nan))
        x = np.arange(-x, x, deprojected_dpix_scale * self.dpix)
        x -= 0.5 * (x[0] + x[-1])

        xs = (r * np.cos(t)).flatten()
        ys = (r * np.sin(t)).flatten()

        # Deproject the image.

        from scipy.interpolate import griddata
        d = griddata((xs, ys), to_mirror.flatten(), (x[:, None], x[None, :]))

        # Either subtract or add the mirrored image. Only want to add when
        # mirroring the line-of-sight velocity and mirroring about the minor
        # axis.

        if not mirror_velocity_residual and mirror_axis == 'minor':
            d += d[:, ::-1]
        else:
            d -= d[:, ::-1]

        if return_deprojected:
            return x, x, d

        # Reproject the residuals onto the sky plane.

        dd = d.flatten()
        mask = np.isfinite(dd)
        xx, yy = np.meshgrid(x, x)
        xx, yy = xx.flatten(), yy.flatten()
        xx, yy, dd = xx[mask], yy[mask], dd[mask]

        from scipy.interpolate import interp2d

        f = interp2d(xx, yy, dd, kind='linear')
        f = np.squeeze([f(xd, yd) for xd, yd in zip(xs, ys)])
        f = f.reshape(to_mirror.shape)
        f = np.where(np.isfinite(to_mirror), f, np.nan)

        return self.xaxis, self.yaxis, f

    # -- VELOCITY PROJECTION -- #

    def _vkep(self, rvals, tvals, zvals, params):
        """Keplerian rotation velocity."""
        r_m = rvals * sc.au * params['dist']
        z_m = zvals * sc.au * params['dist']
        mtotal = params['mstar'] + self._calc_mdisk(rvals, params)
        vkep = sc.G * mtotal * self.msun * jnp.power(r_m, 2)
        return jnp.sqrt(vkep * jnp.power(jnp.hypot(r_m, z_m), -3))

    def _calc_mdisk(self, rvals, params):
        """Psuedo disk self-gravity component."""
        if params['mdisk'] is None:
            return 0.0
        exponent = 2.0 - params['gamma']
        rscale = rvals**exponent - params['r_in']**exponent
        rscale /= params['r_out']**exponent - params['r_in']**exponent
        return params['mdisk'] * jnp.clip(rscale, 0.0, 1.0)

    def _vkep_pressure(self, rvals, tvals, zvals, params):
        """Keplerian rotation velocity with pressure term."""
        vkep = self._vkep(rvals, tvals, zvals, params)
        r_p = params['r_pressure']
        idx = jnp.unravel_index(jnp.argmin(jnp.abs(rvals - r_p)), rvals.shape)
        dvprs = (1.0 - 1.5 * r_p**2 / (r_p**2 + zvals[idx]**2))
        dvprs = ((rvals - r_p) / r_p) * dvprs + 1.0
        vkep = jnp.where(rvals <= r_p, vkep, vkep[idx] * dvprs)
        vkep = jnp.clip(vkep, 0.0, None)
        if params['w_pressure'] > 0.0:
            taper = (rvals - r_p) / params['w_pressure']
            taper = jnp.exp(-jnp.power(taper, 2.0))
            vkep *= jnp.where(rvals <= r_p, 1.0, taper)
        return vkep

    def _vpow(self, rvals, tvals, zvals, params):
        """Power-law rotation velocity profile."""
        vpow = (rvals * params['dist'] / 100.)**params['vp_q']
        return params['vp_100'] * vpow

    def _vpow_pressure(self, rvals, tvals, zvals, params):
        """Power-law rotation with pressure term."""
        vpow = self._vpow(rvals, tvals, zvals, params)
        r_p = params['r_pressure']
        idx = jnp.unravel_index(jnp.argmin(jnp.abs(rvals - r_p)), rvals.shape)
        dvprs = ((rvals - r_p) / r_p) * params['vp_q'] + 1.0
        vpow = jnp.where(rvals <= r_p, vpow, vpow[idx] * dvprs)
        if params['w_pressure'] > 0.0:
            taper = (rvals - r_p) / params['w_pressure']
            taper = jnp.exp(-jnp.power(taper, 2.0))
            vpow *= jnp.where(rvals <= r_p, 1.0, taper)
        return vpow


    def _eliptical_orbit(self, xvals, yvals, ac, ec):
        """
        Define the elliptical orbit properties.
        xf is the distance from the focal point of the cavity.

        Args:
            xvals - disk frame cartesian coordinates
            yvals - disk frame cartesian coordinates
            xf - x-offset from disk center of ellipse focal point
        """
        bc2 = ac**2 * (1-ec**2)
        c2 = ac**2 - bc2
        x  = xvals - jnp.sqrt(c2)
        a2 = (x**2 + yvals**2 + c2) / 2 + jnp.sqrt((x**2 + yvals**2 + c2)**2 / 4 - x**2 * c2)
        return jnp.sqrt(c2), jnp.sqrt(c2 / a2), jnp.sqrt(a2)


    def _make_model_vortex(self, rvals, tvals, params, frame=None):
        """
        Vortex velocity profile projected onto the requested frame. Can return
        the model in a range of frames-of-reference through the ``frame``
        argument. This accepts ``None``, ``'vortex'``, ``'polar'``,
        ``'face-on'`` and ``'sky'``. If ``frame=None`` (default) then the on-sky
        projection is returned without associated coordinates.

        Args:
            rvals (array): 2D array of the deprojected radial disk coordinates
                in [arcsec].
            tvals (array): 2D array of the deprojected polar disk coordinates
                in [radians].
            params (dict): Dictionary of model parameters.
            frame (Optional[str]): If provided, the frame of the vortex velocity
                model to be returned in along with associated coordinates. If no
                frame is specified just the on-sky projected velocity will be
                returned.

        Returns:
            [x, y,] v ([array, array,] array): The vortex velocity model along
                with the associated coordinate values if ``frame`` is specified.
        """

        # Loop through (at least 1) layers to extent the azimuthal map for
        # vortices which overlap.

        x_vortex_layers, y_vortex_layers, v_vortex_layers = [], [], []
        v_disk_stack, v_proj_stack = [], []

        for i in range(-(self._vortex_layers-1), self._vortex_layers):

            # Shift in the polar angle.

            dtheta = i * 2.0 * jnp.pi
            tvals_tmp = tvals + dtheta

            # (x_tmp, y_tmp) describe the vortex cartesian frame.
            # TODO: Should change x_tmp to be rvals * (stuff).

            x_tmp = params['r0_vortex'] * (tvals_tmp - jnp.radians(params['p0_vortex']))
            y_tmp = rvals - params['r0_vortex']
            x_vortex_layers.append(x_tmp)
            y_vortex_layers.append(y_tmp)

            # (r_tmp, p_tmp) describe the vortex polar frame.

            r_tmp = jnp.hypot(x_tmp, params['chi_vortex'] * y_tmp)
            p_tmp = jnp.arctan2(x_tmp, params['chi_vortex'] * y_tmp)
            p_tmp = jnp.clip(p_tmp, -jnp.pi, jnp.pi)

            # Model the vortex radial velocity profile as a Gaussian.

            v_tmp = ((r_tmp - params['r_vortex']) / params['w_vortex'])**2
            v_tmp = params['v_vortex'] * jnp.exp(-v_tmp)
            v_vortex_layers.append(v_tmp)
            v_disk_stack += [v_tmp]

            # Project the vortex velocity onto the sky.

            v_proj_tmp = v_tmp * jnp.cos(tvals_tmp + p_tmp)
            v_proj_tmp *= jnp.sin(abs(jnp.radians(params['inc'])))
            v_proj_stack += [v_proj_tmp]

        # Combine the different layers by summing them up. Note that this will
        # give rise to odd effects if the vortex tails are overlapping.

        v_disk = jnp.sum(jnp.stack(v_disk_stack, axis=0), axis=0)
        v_proj = jnp.sum(jnp.stack(v_proj_stack, axis=0), axis=0)

        # Return the vortex model along with the appropriate coordinates.
        # Note that in the `vortex`, `polar` or  `face-on` frame the velocity
        # isn't projected along the line of sight.

        if frame is None:
            return v_proj
        elif frame == 'vortex':
            x_vortex = jnp.concatenate([a.flatten() for a in x_vortex_layers])
            y_vortex = jnp.concatenate([a.flatten() for a in y_vortex_layers])
            v_vortex = jnp.concatenate([a.flatten() for a in v_vortex_layers])
            assert x_vortex.shape == y_vortex.shape == v_vortex.shape
            return x_vortex, y_vortex, v_vortex
        elif frame == 'polar':
            assert rvals.shape == tvals.shape == v_disk.shape
            return rvals, tvals, v_disk
        elif frame == 'face-on':
            x_disk = rvals * jnp.cos(tvals)
            y_disk = rvals * jnp.sin(tvals)
            assert x_disk.shape == y_disk.shape == v_disk.shape
            return x_disk, y_disk, v_disk
        elif frame == 'sky':
            x_sky, y_sky = jnp.meshgrid(self.xaxis, self.yaxis)
            assert x_sky.shape == y_sky.shape == v_proj.shape
            return x_sky, y_sky, v_proj

    def _proj_vphi(self, v_phi, tvals, params):
        """Project the rotational velocity onto the sky."""
        return v_phi * jnp.cos(tvals) * jnp.sin(abs(jnp.radians(params['inc'])))

    def _proj_vrad(self, v_rad, tvals, params):
        """Project the radial velocity onto the sky."""
        return v_rad * jnp.sin(tvals) * jnp.sin(-jnp.radians(params['inc']))

    def _proj_valt(self, v_alt, tvals, params):
        """Project the vertical velocity onto the sky."""
        return -v_alt * jnp.cos(jnp.radians(params['inc']))

    def _make_model(self, params):
        """Build the velocity model from the dictionary of parameters."""

        # Calculate the deprojected pixel values including ellipticity.
        # Read via ``.get`` not ``.pop`` so the input dict is not mutated;
        # mutation breaks repeated JIT calls that re-use the captured
        # dict (the second call would see missing keys). ``disk_coords``
        # absorbs unknown kwargs via ``**_``, so leaving the extras in
        # place is harmless.

        ac = params.get('ac', None)
        ec = params.get('ec', None)
        om = params.get('pericenter_phase', 0.0)

        rvals, tvals, zvals = self.disk_coords(**params)
        if ec is not None and ac is not None:
            xx, yy, _ = self.disk_coords(**params, frame='cartesian')
            xvals = xx * jnp.cos(jnp.radians(om)) - yy * jnp.sin(jnp.radians(om))
            yvals = xx * jnp.sin(jnp.radians(om)) + yy * jnp.cos(jnp.radians(om))
            a = self._eliptical_orbit(xvals, yvals, ac, ec)[-1]
            rvals = 1.0 / (2.0 / rvals - 1.0 / a)
            #tvals = jnp.arcsin(xvals / rvals)

        # Calculate the velocity profile and project. This includes an
        # additional component from the vortex.

        vphi = params['vfunc'](rvals, tvals, zvals, params)
        vphi_proj = self._proj_vphi(vphi, tvals, params)
        if params['vortex']:
            vvor_proj = self._make_model_vortex(rvals, tvals, params)
        else:
            vvor_proj = 0.0

        v0 = vphi_proj + vvor_proj + params['vlsr']

        # Convolve if necessary.

        if params['beam']:
            v0 = imagecube._convolve_image(v0, self._beamkernel())

        # Return.

        return v0

    def _make_profile(self, params):
        """Build the velocity profile from the dictionary of parameters."""
        rvals, _, zvals = self.disk_coords(**params)
        rvals, zvals = rvals.flatten(), zvals.flatten()
        idx = jnp.argsort(rvals)
        rvals, zvals = rvals[idx], zvals[idx]
        tvals = jnp.zeros(rvals.size)
        return rvals, params['vfunc'](rvals, tvals, zvals, params)

    def deproject_model_residuals(self, samples, params):
        """
        Deproject the residuals into cylindrical velocity components. Takes the
        median value of the samples to determine the model.

        Args:
            samples (ndarray): An array of samples returned from ``fit_map``.
            params (dict): The parameter dictionary passed to ``fit_map``.

        Returns:
            v_p, v_r, v_z (array, array, array): The deprojected rotational,
                radial, and vertical velocity residuals in [m/s], each as a
                2D array with the same shape as the data.
        """
        median_samples = np.median(samples, axis=0)
        verified_params = self.verify_params_dictionary(params.copy())
        model = self._populate_dictionary(median_samples, verified_params)
        v_res = self.data - self._make_model(model)
        rvals, tvals, zvals = self.disk_coords(**model)
        v_p = v_res / np.cos(tvals) / np.sin(-np.radians(model['inc']))
        v_r = v_res / np.sin(tvals) / np.sin(abs(np.radians(model['inc'])))
        v_z = v_res / np.cos(abs(np.radians(model['inc'])))
        v_z = np.where(zvals >= 0.0, -v_z, v_z)
        return v_p, v_r, v_z

    # -- UTILITIES -- #

    def remove_hot_pixels(self, npix=2, nsigma=1.0, niter=1, replace=True):
        """
        Remove hot pixels from the data. Hot pixels are identified by deviating
        from the mean of the region +/- `npix` by an amount of at least `nsigma`
        times the standard deviation of the region. These hot pixels are
        replaced by interpolated (using a box kernel convolution) values.

        Args:
            npix (Optional[int]): The number of pixels from the pixel of
                interest to consider part of the region.
            nsigma (Optional[float]): The threshold for considering a pixel a
                'hot' pixel. Smaller values identify more hot pixels.
            niter (Optional[int]): How many times to repeat this smoothing. Note
                that with `niter > 1` some features may be washed out.
            replace (Optional[bool]): If `True`, replace the attached dataset,
                otherwise, return as an array.

        Returns:
            corrected_data (array): The correced data if `replace=False`.
        """
        from astropy.convolution import convolve, Box2DKernel

        data_tmp = self.data.copy()

        for _ in range(niter):
        
            # Cycle through each pixel and identify the hot pixels.

            coldpix = np.ones(data_tmp.shape) * np.nan
            for xi in np.arange(npix, self.nxpix - npix):
                for yi in np.arange(npix, self.nypix - npix):
                    point = data_tmp[yi, xi]
                    region = data_tmp[yi-npix:yi+npix+1, xi-npix:xi+npix+1]
                    region_mu = np.nanmean(region)
                    region_std = np.nanstd(region)
                    if abs(point - region_mu) < (nsigma * region_std):
                        coldpix[yi, xi] = point
                        
            # Convolve, interpolating the NaN value, and re-mask based on the
            # old data. 
            
            hotpix = np.logical_and(np.isfinite(self.data), np.isnan(coldpix))
            coldpix = convolve(coldpix, Box2DKernel(2*npix+1))
            data_tmp = np.where(hotpix, coldpix, data_tmp)
        
        # Either replace the attached data or return as an array.

        if not replace:
            return data_tmp
        self.data = data_tmp
        self.mask = np.isfinite(self.data)

    # -- Functions to help determine the emission height. -- #

    def find_maxima(self, x0=0.0, y0=0.0, PA=0.0, vlsr=None, r_max=None,
                    r_min=None, smooth=False, through_center=True):
        """
        Find the node of velocity maxima along the major axis of the disk.

        Args:
            x0 (Optional[float]): Source center offset along x-axis in
                [arcsec].
            y0 (Optional[float]): Source center offset along y-axis in
                [arcsec].
            PA (Optioanl[float]): Source position angle in [deg].
            vlsr (Optional[float]): Systemic velocity in [m/s].
            r_max (Optional[float]): Maximum offset to consider in [arcsec].
            r_min (Optional[float]): Minimum offset to consider in [arcsec].
            smooth (Optional[bool/float]): Smooth the line of nodes. If
                ``True``, smoth with the beam kernel, otherwise ``smooth``
                describes the FWHM of the Gaussian convolution kernel in
                [arcsec].
            through_center (Optional[bool]): If ``True``, force the central
                pixel to go through ``(0, 0)``.

        Returns:
            array, array: Arrays of ``x_sky`` and ``y_sky`` values of the
            maxima.
        """

        # Default parameters.
        vlsr = np.nanmedian(self.data) if vlsr is None else vlsr
        r_max = 0.5 * self.xaxis.max() if r_max is None else r_max
        r_min = 0.0 if r_min is None else r_min

        # Shift and rotate the image.
        data = self._shift_center(dx=x0, dy=y0, save=False)
        data = self._rotate_image(PA=PA, data=data, save=False)

        # Find the maximum values. Apply some clipping to help.
        mask = np.maximum(0.3 * abs(self.xaxis), self.bmaj)
        mask = abs(self.yaxis)[:, None] > mask[None, :]
        resi = np.where(mask, 0.0, abs(data - vlsr))
        resi = np.take(self.yaxis, np.argmax(resi, axis=0))

        # Gentrification.
        if through_center:
            resi[abs(self.xaxis).argmin()] = 0.0
        if smooth:
            if isinstance(smooth, bool):
                kernel = np.hanning(self.bmaj / self.dpix)
                kernel /= kernel.sum()
            else:
                from astropy.convolution import Gaussian1DKernel
                kernel = Gaussian1DKernel(smooth / self.fwhm / self.dpix)
            resi = np.convolve(resi, kernel, mode='same')
        mask = np.logical_and(abs(self.xaxis) <= r_max,
                              abs(self.xaxis) >= r_min)
        x, y = self.xaxis[mask], resi[mask]

        # Rotate back to sky-plane and return.
        x, y = self._rotate_coords(x, -y, PA)
        return x + x0, y + y0

    def find_minima(self, x0=0.0, y0=0.0, PA=0.0, vlsr=None, r_max=None,
                    r_min=None, smooth=False, through_center=True):
        """
        Find the line of nodes where v0 = v_LSR along the minor axis.

        Args:
            x0 (Optional[float]): Source center offset along x-axis in
                [arcsec].
            y0 (Optional[float]): Source center offset along y-axis in
                [arcsec].
            PA (Optioanl[float]): Source position angle in [deg].
            vlsr (Optional[float]): Systemic velocity in [m/s].
            r_max (Optional[float]): Maximum offset to consider in [arcsec].
            r_min (Optional[float]): Minimum offset to consider in [arcsec].
            smooth (Optional[bool/float]): Smooth the line of nodes. If
                ``True``, smoth with the beam kernel, otherwise ``smooth``
                describes the FWHM of the Gaussian convolution kernel in
                [arcsec].
            through_center (Optional[bool]): If ``True``, force the central
                pixel to go through ``(0, 0)``.

        Returns:
            array, array: Arrays of ``x_sky`` and ``y_sky`` values of the
            velocity minima.
        """

        # Default parameters.
        vlsr = np.nanmedian(self.data) if vlsr is None else vlsr
        r_max = 0.5 * self.xaxis.max() if r_max is None else r_max
        r_min = 0.0 if r_min is None else r_min

        # Shift and rotate the image.
        data = self._shift_center(dx=x0, dy=y0, save=False)
        data = self._rotate_image(PA=PA, data=data, save=False)

        # Find the maximum values. Apply some clipping to help.
        mask = np.maximum(0.3 * abs(self.yaxis), self.bmaj)
        mask = abs(self.xaxis)[None, :] > mask[:, None]
        resi = np.where(mask, 1e10, abs(data - vlsr))
        resi = np.take(-self.yaxis, np.argmin(resi, axis=1))

        # Gentrification.
        if through_center:
            resi[abs(self.yaxis).argmin()] = 0.0
        if smooth:
            if isinstance(smooth, bool):
                kernel = np.hanning(self.bmaj / self.dpix)
                kernel /= kernel.sum()
            else:
                from astropy.convolution import Gaussian1DKernel
                kernel = Gaussian1DKernel(smooth / self.fwhm / self.dpix)
            resi = np.convolve(resi, kernel, mode='same')
        mask = np.logical_and(abs(self.yaxis) <= r_max,
                              abs(self.yaxis) >= r_min)
        x, y = resi[mask], self.yaxis[mask]

        # Rotate back to sky-plane and return.
        x, y = self._rotate_coords(x, -y, PA)
        return x + x0, y + y0

    def _fit_surface(self, inc, PA, x0=None, y0=None, r_min=None, r_max=None,
                     fit_z1=False, ycut=20.0, **kwargs):
        """
        Fit the emission surface with the parametric model based on the pixels
        of peak velocity.

        Args:
            inc (float): Disk inclination in [degrees].
            PA (float): Disk position angle in [degrees].
            x0 (Optional[float]): Disk center x-offset in [arcsec].
            y0 (Optional[float]): Disk center y-offset in [arcsec].
            r_min (Optional[float]): Minimum radius to fit in [arcsec].
            r_max (Optional[float]): Maximum radius to fit in [arcsec].
            fit_z1 (Optional[bool]): Include (z1, phi) in the fit.
            y_cut (Optional[float]): Only use the `ycut` fraction of the minor
                axis for fitting.
            kwargs (Optional[dict]): Additional kwargs for curve_fit.

        Returns:
            popt (list): Best-fit parameters of (z0, psi[, z1, phi]).
        """

        from scipy.optimize import curve_fit

        def z_func(x, z0, psi, z1=0.0, phi=1.0):
            """Parametric emission surface model: z = z0*r^psi + z1*r^phi."""
            return z0 * x**psi + z1 * x**phi

        # Get the coordinates to fit.
        r, z = self.get_peak_pix(PA=PA, inc=inc, x0=x0, y0=y0,
                                 frame='disk', ycut=ycut)

        # Mask the coordinates.
        r_min = r_min if r_min is not None else r[0]
        r_max = r_max if r_max is not None else r[-1]
        m1 = np.logical_and(np.isfinite(r), np.isfinite(z))
        m2 = (r >= r_min) & (r <= r_max)
        mask = m1 & m2
        r, z = r[mask], z[mask]

        # Initial guesses for the free params.
        p0 = [z[abs(r - 1.0).argmin()], 1.0]
        if fit_z1:
            p0 += [-0.05, 3.0]
        maxfev = kwargs.pop('maxfev', 100000)

        # First fit the inner half to get a better p0 value.
        mask = r <= 0.5 * r.max()
        p_in = curve_fit(z_func, r[mask], z[mask], p0=p0, maxfev=maxfev,
                         **kwargs)[0]

        # Remove the pixels which are 'negative' depending on the inclination.
        mask = z > 0.0 if np.sign(p_in[0]) else z < 0.0
        popt = curve_fit(z_func, r[mask], z[mask], p0=p_in, maxfev=maxfev,
                         **kwargs)[0]
        return popt

    # -- Axes Functions -- #

    def downsample_cube(self, N, randomize=False):
        """
        Downsample the cube to make faster calculations.

        Args:
            N (int or str): Downsampling factor. If ``N='beam'``, uses the beam
                major axis divided by the pixel size as the factor.
            randomize (Optional[bool]): If ``True``, choose a random starting
                pixel offset rather than centering the sampling grid.
        """
        N = int(np.ceil(self.bmaj / self.dpix)) if N == 'beam' else N
        if randomize:
            N0x, N0y = np.random.randint(0, N, 2)
        else:
            N0x, N0y = int(N / 2), int(N / 2)
        if N > 1:
            self.xaxis = self.xaxis[N0x::N]
            self.yaxis = self.yaxis[N0y::N]
            self.data = self.data[N0y::N, N0x::N]
            self.error = self.error[N0y::N, N0x::N]
            self.mask = self.mask[N0y::N, N0x::N]

    def _shift_center(self, dx=0.0, dy=0.0, data=None, save=True):
        """
        Shift the center of the image.

        Args:
            dx (optional[float]): shift along x-axis [arcsec].
            dy (optional[float]): Shift along y-axis [arcsec].
            data (optional[ndarray]): Data to shift. If nothing is provided,
                will shift the attached ``rotationmap.data``.
            save (optional[bool]): If True, overwrite ``rotationmap.data``.
        """
        from scipy.ndimage import shift
        data = self.data.copy() if data is None else data
        to_shift = np.where(np.isfinite(data), data, 0.0)
        data = shift(to_shift, [-dy / self.dpix, dx / self.dpix])
        if save:
            self.data = data
        return data

    def _rotate_image(self, PA, data=None, save=True):
        """
        Rotate the image anticlockwise about the center.

        Args:
            PA (float): Rotation angle in [degrees].
            data (optional[ndarray]): Data to shift. If nothing is provided,
                will shift the attached ``rotationmap.data``.
            save (optional[bool]): If True, overwrite ``rotationmap.data``.
        """
        from scipy.ndimage import rotate
        data = self.data.copy() if data is None else data
        to_rotate = np.where(np.isfinite(data), data, 0.0)
        data = rotate(to_rotate, PA - 90.0, reshape=False)
        if save:
            self.data = data
        return data

    # -- DATA I/O -- #

    def _readuncertainty(self, uncertainty, FOV=None):
        """Reads the uncertainties."""
        if uncertainty is not None:
            self.error = imagecube(uncertainty, FOV=FOV, fill=None)
            self.error = self.error.data.copy()
        else:
            try:
                uncertainty = '_'.join(self.path.split('_')[:-1])
                uncertainty += '_d' + self.path.split('_')[-1]
                print("Assuming uncertainties in {}.".format(uncertainty))
                self.error = imagecube(uncertainty, FOV=FOV, fill=None)
                self.error = self.error.data.copy()
            except FileNotFoundError:
                print("No uncertainties found, assuming uncertainties of 10%.")
                print("Change this at any time with `rotationmap.error`.")
                self.error = 0.1 * (self.data - np.nanmedian(self.data))
        self.error = np.where(np.isnan(self.error), 0.0, abs(self.error))
        self.error *= 1e3 if self.velocity_unit == 'km/s' else 1.0
        assert self.data.shape == self.error.shape

    # -- PLOTTING -- #

    def plot_velocity_profiles(self, rpnts, velo, dvelo):
        """
        Plot the velocity profiles. The `velo` array must specify
        ``[v_phi, v_r, v_z, v_lsr]`` in that order.

        Args:
            rpnts (array): Array of the annulus centers in [arcsec].
            velo (ndarray): Array of the velocity profiles with a shape of
                ``(nparam, nannuli)`` in [m/s].
            dvelo (ndarray): Array of the uncertainties of the velocity
                profiles with the same shape as ``velo`` in [m/s].

        Returns:
            None: The figure is displayed but not returned.
        """

        # Make the axes.

        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(6.75, 6.25))

        # Plot the rotational velocities.

        v_phi, dv_phi = velo[0], dvelo[0]
        v_phi_ylim = np.nanpercentile(v_phi / 1e3, [2, 98])

        axs[0].errorbar(rpnts, v_phi / 1e3, dv_phi / 1e3, fmt='-o', ms=3)
        axs[0].set_xlabel(r'Radius (arcsec)', labelpad=8)
        axs[0].xaxis.set_label_position('top')
        axs[0].xaxis.tick_top()
        axs[0].set_ylabel(r'$v_{\rm \phi}$' + ' (km/s)')
        axs[0].set_ylim(v_phi_ylim)

        # Plot the radial velocities.

        v_rad, dv_rad = velo[1], dvelo[1]
        std = np.nanpercentile(v_rad, [16, 84])
        std = 0.5 * (std[1] - std[0])
        v_rad_ylim = (-3.0 * std, 3.0 * std)
 
        axs[1].errorbar(rpnts, v_rad, dv_rad, fmt='-o', ms=3)
        axs[1].set_xticklabels([])
        axs[1].set_ylabel(r'$v_{\rm r}$' + ' (m/s)')
        axs[1].set_ylim(v_rad_ylim)
        axs[1].text(0.975, 0.925, 'away from star', ha='right', va='top',
                    transform=axs[1].transAxes, color='0.5')
        axs[1].text(0.975, 0.075, 'towards star', ha='right', va='bottom',
                    transform=axs[1].transAxes, color='0.5')

        # Plot the vertical / systemic velocities. Note the change between the
        # y-axis label depending of if we're plotting just the vertical velocity
        # components, or the combined values.

        if np.nanmean(velo[2]) == 0.0 and np.nanstd(velo[2]) < 1e-4:
            v_alt, dv_alt = velo[3], dvelo[3]
            axs[2].set_ylabel(r'$v_{\rm LSR} - v_{\rm z} / \cos(i)$' + ' (m/s)')
            mu = np.nanmedian(v_alt)
        elif np.nanstd(velo[3]) < 1e-4:
            v_alt, dv_alt = velo[2], dvelo[2]
            axs[2].set_ylabel(r'$v_{\rm z}$' + ' (m/s)')
            label = r'$v_{\rm LSR} = $' + ' {:.0f} m/s'.format(velo[3, 0])
            axs[2].text(0.025, 0.925, label, ha='left', va='top',
                        transform=axs[2].transAxes, color='0.5')
            axs[2].text(0.975, 0.925, 'away from midplane', ha='right',
                        va='top', transform=axs[2].transAxes, color='0.5')
            axs[2].text(0.975, 0.075, 'towards midplane', ha='right',
                        va='bottom', transform=axs[2].transAxes, color='0.5')
            mu = 0.0
        else:
            raise ValueError("Struggling to parse the `velo` arrays.")

        std = np.nanpercentile(v_alt, [16, 84])
        std = 0.5 * (std[1] - std[0])
        v_alt_ylim = (mu - 3.0 * std, mu + 3.0 * std)

        axs[2].errorbar(rpnts, v_alt, dv_alt, fmt='-o', ms=3)
        axs[2].set_xlabel(r'Radius (arcsec)')
        axs[2].set_ylim(v_alt_ylim)

        for ax in axs:
            ax.grid(ls='--', color='0.9', lw=1.0)
            ax.set_xlim(rpnts.min(), rpnts.max())
            ax.tick_params(which='both', bottom=True, right=True, top=True)
        fig.align_labels(axs)

    def plot_model(self, samples=None, params=None, model=None, draws=0.5,
                   mask=None, ax=None, imshow_kwargs=None, cb_label=None,
                   return_fig=False):
        """
        Plot a v0 model using the same scalings as the plot_data() function.

        Args:
            samples (Optional[ndarray]): An array of samples returned from
                ``fit_map``.
            params (Optional[dict]): The parameter dictionary passed to
                ``fit_map``.
            model (Optional[ndarry]): A model array from ``evaluate_models``.
            draws (Optional[int/float]): If an integer, describes the number of
                random draws averaged to form the returned model. If a float,
                represents the percentile used from the samples. Must be
                between 0 and 1 if a float.
            mask (Optional[ndarray]): The mask used for the fitting to plot as
                a shaded region.
            ax (Optional[AxesSubplot]): Axis to plot onto.
            imshow_kwargs (Optional[dict]): Dictionary of imshow kwargs.
            cb_label (Optional[str]): Colorbar label.
            return_fig (Optional[bool]): Return the figure.

        Returns:
            fig (Matplotlib figure): If ``return_fig`` is ``True``. Can access
                the axes through ``fig.axes`` for additional plotting.
        """

        # Dummy axis for the plotting.

        if ax is None:
            fig, ax = plt.subplots()

        # Make the model and calculate the plotting limits. Note that we assume
        # the input model is in [m/s] and we convert to [km/s].

        if model is None:
            model = self.evaluate_models(samples, params.copy(), draws=draws)
        vmin, vmax = np.nanpercentile(model / 1e3, [2, 98])
        vmax = max(abs(vmin - self.vlsr / 1e3), abs(vmax - self.vlsr / 1e3))
        vmin = self.vlsr / 1e3 - vmax
        vmax = self.vlsr / 1e3 + vmax

        # Initialize the plotting parameters.

        imshow_kwargs = {} if imshow_kwargs is None else imshow_kwargs
        imshow_kwargs['cmap'] = imshow_kwargs.pop('cmap', imagecube.cmap())
        imshow_kwargs['zorder'] = imshow_kwargs.pop('zorder', -9)
        imshow_kwargs['extent'] = self.extent
        imshow_kwargs['origin'] = 'lower'
        imshow_kwargs['vmin'] = imshow_kwargs.pop('vmin', vmin)
        imshow_kwargs['vmax'] = imshow_kwargs.pop('vmax', vmax)
        im = ax.imshow(model / 1e3, **imshow_kwargs)

        # Overplot the mask if necessary.

        if mask is not None:
            ax.contour(self.xaxis, self.yaxis, mask,
                       [0.0], colors='k')
            ax.contourf(self.xaxis, self.yaxis, mask,
                        [-1.0, 0.0], colors='k', alpha=0.5)

        if cb_label is None:
            cb_label = r'${\rm v_{0,\,{\rm mod}} \quad (km\,s^{-1})}$'
        if cb_label != '':
            cb = plt.colorbar(im, pad=0.03, format='%.2f', extend='both')
            cb.set_label(cb_label, rotation=270, labelpad=15)
            cb.minorticks_on()

        self._gentrify_plot(ax)

        if return_fig:
            return fig

    def plot_model_residual(self, samples=None, params=None, model=None,
                            draws=0.5, mask=None, ax=None, imshow_kwargs=None,
                            return_fig=False):
        """
        Plot the residual from the provided model.

        Args:
            samples (Optional[ndarray]): An array of samples returned from
                ``fit_map``.
            params (Optional[dict]): The parameter dictionary passed to
                ``fit_map``.
            model (Optional[ndarry]): A model array from ``evaluate_models``.
            draws (Optional[int/float]): If an integer, describes the number of
                random draws averaged to form the returned model. If a float,
                represents the percentile used from the samples. Must be
                between 0 and 1 if a float.
            mask (Optional[ndarray]): The mask used for the fitting to plot as
                a shaded region.
            ax (Optional[AxesSubplot]): Axis to plot onto.
            imshow_kwargs (Optional[dict]): Dictionary of imshow kwargs.
            return_fig (Optional[bool]): Return the figure.

        Returns:
            fig (Matplotlib figure): If ``return_fig`` is ``True``. Can access
                the axes through ``fig.axes`` for additional plotting.
        """

        # Dummy axis to overplot.

        if ax is None:
            fig, ax = plt.subplots()

        # Make the model and calculate the plotting limits.

        if model is None:
            model = self.evaluate_models(samples, params.copy(), draws=draws)
        vres = self.data - model
        mask = np.ones(vres.shape) if mask is None else mask
        masked_vres = np.where(mask, vres, np.nan)
        vmin, vmax = np.nanpercentile(masked_vres, [2, 98])
        vmax = max(abs(vmin), abs(vmax))

        # Plot the data.

        imshow_kwargs = {} if imshow_kwargs is None else imshow_kwargs
        imshow_kwargs['origin'] = 'lower'
        imshow_kwargs['extent'] = self.extent
        imshow_kwargs['cmap'] = 'RdBu_r'
        imshow_kwargs['vmin'] = imshow_kwargs.pop('vmin', -vmax)
        imshow_kwargs['vmax'] = imshow_kwargs.pop('vmax', vmax)
        im = ax.imshow(vres, **imshow_kwargs)

        # Overplot the mask is necessary.

        if mask is not None:
            ax.contour(self.xaxis, self.yaxis, mask,
                       [0.0], colors='k')
            ax.contourf(self.xaxis, self.yaxis, mask,
                        [-1.0, 0.0], colors='k', alpha=0.5)

        cb = plt.colorbar(im, pad=0.02, format='%d', ax=ax, extend='both')
        cb.set_label(r'${\rm  v_{0} - v_{0,\,{\rm mod}} \quad (m\,s^{-1})}$',
                     rotation=270, labelpad=15)
        cb.minorticks_on()

        self._gentrify_plot(ax)

        if return_fig:
            return fig

    def plot_model_surface(self, samples, params,  plot_surface_kwargs=None,
                           mask_with_data=True, return_fig=True):
        """
        Overplot the emission surface onto the provided axis. Takes the median
        value of the samples as the model to plot.

        Args:
            samples (ndarray): An array of samples returned from ``fit_map``.
            params (dict): The parameter dictionary passed to ``fit_map``.
            plot_surface_kwargs (Optional[dict]): Dictionary of kwargs to pass
                to ``plot_surface``.
            mask_with_data (Optional[bool]): If ``True``, mask the surface to
                regions where the data is finite valued.
            return_fig (Optional[bool]): Return the figure.

        Returns:
            fig (Matplotlib figure): If ``return_fig`` is ``True``. Can access
                the axes through ``fig.axes`` for additional plotting.
        """

        # Check the input.

        nparam = np.sum([type(params[k]) is int for k in params.keys()])
        if samples.shape[1] != nparam:
            warning = "Invalid number of free parameters in 'samples': {:d}."
            raise ValueError(warning.format(nparam))
        if plot_surface_kwargs is None:
            plot_surface_kwargs = {}
        plot_surface_kwargs['return_fig'] = True

        # Populate the model with the median values.

        model = self.verify_params_dictionary(params.copy())
        model = self._populate_dictionary(np.median(samples, axis=0), model)
        model['mask'] = np.isfinite(self.data) if mask_with_data else None
        model.pop('r_max')
        fig = self.plot_surface(**model, **plot_surface_kwargs)
        self._gentrify_plot(ax=fig.axes[0])

        if return_fig:
            return fig

    def plot_model_surface_2D(self, samples, params, draws=50, return_fig=True):
        """
        Plot the model surface in the (r, z) plane.

        Args:
            samples (ndarray): An array of samples returned from ``fit_map``.
            params (dict): The parameter dictionary passed to ``fit_map``.
            drawns (Optional[int]): Number of draws from the posterior to plot.
            return_fig (Optional[bool]): Return the figure.

        Returns:
            fig (Matplotlib figure): If ``return_fig`` is ``True``. Can access
                the axes through ``fig.axes`` for additional plotting.
        """

        r, _, z = self.evaluate_models(samples=samples,
                                       params=params,
                                       draws=draws,
                                       coords_only=True,
                                       )
        rvals, rbins = self._get_radial_bins()
        ridxs = np.digitize(rbins, r)
        zvals = np.array([np.mean(z[ridxs == r]) for r in ridxs])

        fig, ax = plt.subplots()
        ax.plot(rvals, zvals)
        ax.set_xlabel('Radius (arcsec')
        ax.set_ylabel('Height (arcsec)')

        if return_fig:
            return fig

    def plot_disk_axes(self, x0=0.0, y0=0.0, inc=0.0, PA=0.0, major=1.0,
                       ax=None, plot_kwargs=None, return_ax=True):
        """
        Plot the major and minor axes on the provided axis.

        Args:
            ax (Matplotlib axes): Axes instance to plot onto.
            x0 (Optional[float]): Relative x-location of the center [arcsec].
            y0 (Optional[float]): Relative y-location of the center [arcsec].
            inc (Optional[float]): Inclination of the disk in [degrees].
            PA (Optional[float]): Position angle of the disk in [degrees].
            major (Optional[float]): Size of the major axis line in [arcsec].
            plot_kwargs (Optional[dict]): Dictionary of parameters to pass to
                ``matplotlib.plot``.

        Returns:
            matplotlib axis: Matplotlib ax with axes drawn.
        """

        # Dummy axis to plot.

        if ax is None:
            fig, ax = plt.subplots()

        x = np.array([major, -major,
                      0.0, 0.0])
        y = np.array([0.0, 0.0,
                      major * np.cos(np.radians(inc)),
                      -major * np.cos(np.radians(inc))])
        x, y = self._rotate_coords(x, y, PA=PA)
        x, y = x + x0, y + y0
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        c = plot_kwargs.pop('c', plot_kwargs.pop('color', 'k'))
        ls = plot_kwargs.pop('ls', plot_kwargs.pop('linestyle', ':'))
        lw = plot_kwargs.pop('lw', plot_kwargs.pop('linewidth', 1.0))
        ax.plot(x[:2], y[:2], c=c, ls=ls, lw=lw)
        ax.plot(x[2:], y[2:], c=c, ls=ls, lw=lw)

        if return_ax:
            ax

    def plot_maxima(self, x0=0.0, y0=0.0, inc=0.0, PA=0.0, vlsr=None,
                    r_max=1.0, r_min=None, smooth=False, through_center=True,
                    plot_axes_kwargs=None, plot_kwargs=None,
                    return_fig=False):
        """
        Mark the position of the maximum velocity as a function of radius. This
        can help demonstrate if there is an appreciable bend in velocity
        contours indicative of a flared emission surface. This function pulls
        the results from fit_cube.get_peak_pix() for plotting.

        Args:
            x0 (Optional[float]): Source center x-offset in [arcsec].
            y0 (Optional[float]): Source center y-offset in [arcsec].
            inc (Optional[float]): Disk inclination in [degrees].
            PA (Optional[float]): Disk position angle in [degrees].
            vlsr (Optional[float]): Systemic velocity in [m/s].
            r_max (Optional[float]): Maximum offset to consider in [arcsec].
            r_min (Optional[float]): Minimum offset to consider in [arcsec].
                This is useful to skip large regions where beam convolution
                dominates the map.
            smooth (Optional[bool/float]): Smooth the line of nodes. If
                ``True``, smoth with the beam kernel, otherwise ``smooth``
                describes the FWHM of the Gaussian convolution kernel in
                [arcsec].
            through_center (Optional[bool]): If ``True``, force the central
                pixel to go through ``(0, 0)``.
            levels (Optional[array]): Levels to pass to ``plot_data``.
            plot_axes_kwargs (Optional[dict]): Dictionary of kwargs for the
                plot_axes function.
            plot_kwargs (Optional[dict]): Dictionary of kwargs for the
                plot of the maximum and minimum pixels.
            return_fig (Optional[bool]): If True, return the figure.

        Returns:
            (Matplotlib figure): If return_fig is ``True``. Can access the axis
                through ``fig.axes[0]``.
        """

        # Background figure.
        fig = self.plot_data(return_fig=True)
        ax = fig.axes[0]
        if plot_axes_kwargs is None:
            plot_axes_kwargs = dict()
        self.plot_disk_axes(ax=ax, x0=x0, y0=y0, inc=inc, PA=PA,
                            major=plot_axes_kwargs.pop('major', r_max),
                            **plot_axes_kwargs)

        # Get the pixels for the maximum and minimum values.
        x_maj, y_maj = self.find_maxima(x0=x0, y0=y0, PA=PA, vlsr=vlsr,
                                        r_max=r_max, r_min=r_min,
                                        smooth=smooth,
                                        through_center=through_center)
        r_max = r_max * np.cos(np.radians(inc))
        x_min, y_min = self.find_minima(x0=x0, y0=y0, PA=PA, vlsr=vlsr,
                                        r_max=r_max, r_min=r_min,
                                        smooth=smooth,
                                        through_center=through_center)

        # Plot the lines.
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        c = plot_kwargs.pop('c', plot_kwargs.pop('color', 'k'))
        lw = plot_kwargs.pop('lw', plot_kwargs.pop('linewidth', 1.0))

        ax.plot(x_maj, y_maj, c=c, lw=lw, **plot_kwargs)
        ax.plot(x_min, y_min, c=c, lw=lw, **plot_kwargs)

        if return_fig:
            return fig
