Fitting Rotation Maps FAQ
=========================

Common questions about fitting rotation maps with
:class:`eddy.rotationmap.rotationmap`. For a walkthrough, start with
:doc:`Fitting Rotation Maps </tutorials/tutorial_1>`.


What should I pass to ``rotationmap``?
--------------------------------------

A map of line centroids (a "rotation map") and, ideally, a matching map of
its uncertainties. Both can be made with `bettermoments`_:

.. code-block:: python

    from eddy import rotationmap
    cube = rotationmap(path='disk_v0.fits', uncertainty='disk_dv0.fits',
                       FOV=8.0, downsample='beam')

``FOV`` clips the field of view in [arcsec]. ``downsample='beam'`` keeps
roughly one pixel per beam, so neighbouring pixels are closer to
independent and the posterior widths are less over-confident.

.. _bettermoments: https://github.com/richteague/bettermoments


How do I choose which parameters are free and which are fixed?
--------------------------------------------------------------

Through the ``params`` dictionary passed to ``fit_map``. An **integer**
value is the index of that parameter's starting guess in ``p0``; a
**float** fixes the parameter at that value:

.. code-block:: python

    params = {'x0': 0, 'y0': 1, 'PA': 2, 'mstar': 3, 'vlsr': 4}
    p0 = [0.0, 0.0, 151., 0.81, 2.8e3]

    params['inc'] = 5.8     # fixed, degrees
    params['dist'] = 60.1   # fixed, parsec

Because the type carries the meaning, fixed values must be floats:
``params['inc'] = 5`` would be read as "free, guess at ``p0[5]``". Any
parameter you leave out takes its default value.


How do I change the priors?
---------------------------

Use :meth:`~eddy.rotationmap.rotationmap.set_prior`, which takes the
parameter name, the prior's values and its type. Flat priors take
``[min, max]``; Gaussian priors take ``[mean, std]``:

.. code-block:: python

    cube.set_prior('vlsr', [-20e3, 20e3], 'flat')
    cube.set_prior('x0', [0.0, 0.05], 'gaussian')

This replaces any earlier prior on that parameter, including the default.
To see the default, call ``cube.print_default_prior('vlsr')``.


Which way is the position angle measured, and what does the sign of the inclination mean?
-----------------------------------------------------------------------------------------

The position angle is measured east of north to the **red-shifted** semi-major
axis. The sign of the inclination sets the direction of rotation: positive is
clockwise on the sky, negative is anti-clockwise. The sign only matters once
the emission surface is elevated, because it decides which side of the disk
is the near side.


How do I fit an elevated emission surface?
------------------------------------------

Free the surface parameters ``z0``, ``psi``, ``r_taper`` and ``q_taper``
(see :doc:`Disks with Elevated Emission Surfaces </tutorials/tutorial_2>`).
To trace the back side of the disk instead of the front, make ``z0``
negative. If the analytical form is too restrictive, pass your own
``z_func`` that returns ``z`` in [arcsec] for ``r`` in [arcsec].


How do I fit only part of the disk?
-----------------------------------

Set ``r_min`` and ``r_max`` (in [arcsec]), either as ``fit_map`` arguments
or in ``params``. Masking the inner beam or two (``r_min = 2 * cube.bmaj``)
is usually a good idea, as beam smearing is strongest there. For arbitrary
regions, pass a 2D boolean ``user_mask`` in ``params``.


Do I need to convolve the model with the beam?
----------------------------------------------

Usually not. Setting ``params['beam'] = True`` convolves the model
rotation map with the beam, which mainly matters for low spatial
resolution data or for maps made with the intensity-weighted average
velocity (moment 1).


Can I add radial motions or a non-Keplerian rotation curve?
-----------------------------------------------------------

Yes. ``params['vr_100']`` (and ``params['vr_q']``) add a radial velocity
``v_r(r) = vr_100 * (r / 100)^vr_q`` on top of the rotation curve. Setting
``params['vp_100']`` instead of ``mstar`` swaps the Keplerian curve for a
power law; you cannot set both.


Which MCMC sampler should I use?
--------------------------------

``fit_map`` supports three backends through the ``mcmc`` argument:
``'emcee'`` (the default), ``'zeus'`` and ``'numpyro'``, which uses the
No-U-Turn Sampler (NUTS). Use emcee unless you have a reason not to: on a
CPU it is usually much faster in wall time for the fits ``eddy`` does.
NUTS follows the gradient of the likelihood, so each sample is less
correlated with the last, but each one is far more expensive. It is mainly
worth trying on a GPU, which JAX uses automatically.

With ``mcmc='numpyro'``, the usual arguments map onto NUTS: ``nwalkers``
sets the number of chains, ``nburnin`` the warmup and ``nsteps`` the
number of samples. Anything else in ``mcmc_kwargs`` is passed to
``numpyro.infer.NUTS``:

.. code-block:: python

    cube.set_prior('r_taper', [0.0, 50.0], 'flat')
    samples = cube.fit_map(p0=p0, params=params, mcmc='numpyro',
                           nwalkers=2, nburnin=500, nsteps=500,
                           mcmc_kwargs={'seed': 0, 'dense_mass': True,
                                        'max_tree_depth': 8})

A few things make NUTS behave better:

- Give every free parameter a finite prior range. The default ``r_taper``
  prior is unbounded above, which NUTS handles poorly, hence the
  ``set_prior`` call above.
- Use ``dense_mass=True`` when parameters are strongly correlated, as
  ``psi``, ``r_taper`` and ``q_taper`` are for an elevated surface. The
  default diagonal mass matrix cannot follow those correlations, so NUTS
  takes tiny steps and makes slow progress.
- Watch the progress bar. If the number of steps per iteration sits at
  the maximum (``2**max_tree_depth - 1``), every trajectory is being cut
  short and the chain is mixing poorly.
- Run at least two chains, so you can check that they agree.


My velocities come out negative. What went wrong?
-------------------------------------------------

The position angle is most likely 180 degrees off. Remember it points to
the red-shifted side of the disk.
