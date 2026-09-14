.. structurefunction:

Structure Functions
===================

The :mod:`eddy.structurefunction` module computes the second-order structure
function,

.. math::

    S_2(\ell) = \langle\,[f(x+\ell) - f(x)]^2\,\rangle,

on a polar ``(radius, azimuth)`` grid. The :class:`StructureFunction` class
holds a single 2D structure function — either global or anchored at a reference
annulus — together with its radial and azimuthal slices, while
:class:`StructureFunctionStack` collects results across a range of reference
radii for radius-resolved analyses. Both provide tools to denoise (``subtract``
a noise model), ``combine`` realizations, ``collapse`` to a global statistic,
and reduce to scalar summaries (``plateau``, ``half_power_lag``). A stack can
also be fit with a parametric anisotropic Gaussian-random-field model
(``fit_GRF``) for the correlation lengths and their radial scaling.

Both bare-array entry points take a ``grid`` argument declaring the geometry
of the field you pass, since the kernel is a generic regular-grid lag
estimator and cannot infer it. ``grid='polar'`` (the default) is the
:meth:`eddy.imagecube.imagecube.polar_deprojection` layout — axis 0 = radius
[arcsec], axis 1 = azimuth [deg] — and suppresses the mixed-units azimuthal
average ``S2_i``. ``grid='cartesian'`` is for a field whose axes share units
(a sky-plane image, a simulation slice); ``S2_i`` is then meaningful, but the
radius/azimuth analyses raise. See :data:`eddy.structurefunction.GRID_TYPES`.

A worked example is given in the
:doc:`structure function tutorial </tutorials/tutorial_7_structurefunction>`.

For real (sky-plane) data, build a stack directly from a map with
:meth:`eddy.momentmap.momentmap.calculate_structure_function_stack`, which
deprojects onto the polar grid first.


Building a structure function
------------------------------

.. autofunction:: eddy.structurefunction.calculate_structure_function

.. autofunction:: eddy.structurefunction.calculate_structure_function_stack


The 2D structure function
--------------------------

.. autoclass:: eddy.structurefunction.StructureFunction
   :members:


The radius-resolved stack
-------------------------

.. autoclass:: eddy.structurefunction.StructureFunctionStack
   :members:


Module functions
----------------

.. autodata:: eddy.structurefunction.GRID_TYPES

.. autofunction:: eddy.structurefunction.calculate_structure_function_ensemble

.. autofunction:: eddy.structurefunction.gaussian_beam_s2

.. autofunction:: eddy.structurefunction.calculate_s2

.. autofunction:: eddy.structurefunction.setup_lag_coords

.. autofunction:: eddy.structurefunction.extract_basic_profiles

.. autofunction:: eddy.structurefunction.combine_s2_weighted


Drawing realizations
--------------------

The forward direction of the analysis: draw fields with a known ``S_2`` to
calibrate an estimator, or build a noise null to subtract from a
measurement. :func:`~eddy.structurefunction.draw_polar_field` draws the
parametric anisotropic GRF that ``fit_GRF`` models, on a grid from
:func:`~eddy.structurefunction.make_polar_grid`.

.. autofunction:: eddy.structurefunction.draw_polar_field

.. autofunction:: eddy.structurefunction.make_polar_grid

.. autofunction:: eddy.structurefunction.polar_covariance

For a *noise* null there are two interchangeable backends, both sharing the
beam frame of :func:`~eddy.structurefunction.gaussian_beam_s2`: the
parametric :func:`~eddy.structurefunction.gaussian_beam_realization`
(beam-convolved white noise, carrying only the PSF correlation), and the
empirical :meth:`~eddy.structurefunction.StructureFunction.draw_realization`,
which synthesizes from a measured ``S_2`` — typically
:meth:`eddy.linecube.linecube.noise_structure_function` — and so also
reproduces the imaging pipeline's extra correlated structure. Differencing
ensembles built from each isolates how much apparent structure the naive PSF
model misses. :meth:`eddy.imagecube.imagecube.noise_realization` is the
cube-level entry point to both, filling the shape, ``dpix`` and beam from the
object.

.. autofunction:: eddy.structurefunction.gaussian_beam_realization

.. automethod:: eddy.imagecube.imagecube.noise_realization


Azimuthal spiral model
----------------------

.. autofunction:: eddy.structurefunction.S2phi

.. autofunction:: eddy.structurefunction.S2phi_singlemodel


Theoretical structure functions
-------------------------------

Forward models for the second-order structure function of an anisotropic,
non-stationary Gaussian random field (the model fit by
:meth:`StructureFunctionStack.fit_GRF`), with an optional deterministic
grand-design spiral contribution.

.. autofunction:: eddy.structurefunction.grf_s2_slices

.. autofunction:: eddy.structurefunction.grf_s2_2d_global

.. autofunction:: eddy.structurefunction.predict_s2_slices

.. autofunction:: eddy.structurefunction.predict_s2_2d

.. autofunction:: eddy.structurefunction.predict_spiral_s2_slices

.. autofunction:: eddy.structurefunction.predict_spiral_s2_2d

.. autofunction:: eddy.structurefunction.ell_r

.. autofunction:: eddy.structurefunction.ell_phi


Renamed in 3.2.0
----------------

The structure-function API was aligned on the ``calculate_`` / ``fit_`` /
``plot_`` verb convention in 3.2.0, and the classes dropped their ``2D``
suffix. The old spellings below still work but emit a
``DeprecationWarning``; they will be removed in 4.0.

====================================================  ============================================================
Old name (3.1.x)                                      New name (3.2.0)                                            
====================================================  ============================================================
``StructureFunction2D``                               ``StructureFunction``
``StructureFunction2DStack``                          ``StructureFunctionStack``
``StructureFunction2D.from_array``                    ``StructureFunction.calculate``
``StructureFunction2DStack.from_array``               ``StructureFunctionStack.calculate``
``momentmap.compute_structure_function``              ``momentmap.calculate_structure_function``
``momentmap.compute_structure_function_stack``        ``momentmap.calculate_structure_function_stack``
``StructureFunction2DStack.measure_heuristics``       ``StructureFunctionStack.calculate_heuristics``
``StructureFunction2DStack.pairwise_error_heatmaps``  ``StructureFunctionStack.calculate_pairwise_error_heatmaps``
``compute_s2``                                        ``calculate_s2``
``structure_function_ensemble``                       ``calculate_structure_function_ensemble``
====================================================  ============================================================
