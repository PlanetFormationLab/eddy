Structure Functions FAQ
=======================

Common questions about the :mod:`eddy.structurefunction` module. A
worked :doc:`structure function tutorial </tutorials/structure_functions_coming_soon>`
is coming soon.


What does the module compute?
-----------------------------

The second-order structure function,

.. math::

    S_2(\ell) = \langle\,[f(x+\ell) - f(x)]^2\,\rangle,

which measures how different a field is at two points separated by the lag
``ℓ``. It is computed on a polar ``(radius, azimuth)`` grid.
``StructureFunction`` holds a single 2D structure function, either global or
anchored at a reference annulus, along with its radial and azimuthal
slices. ``StructureFunctionStack`` collects results over a range of
reference radii for radius-resolved analyses.


How do I compute a structure function from real data?
-----------------------------------------------------

Start from a map and use
``momentmap.calculate_structure_function`` or
``momentmap.calculate_structure_function_stack``. These deproject the map
onto the polar grid for you, given the disk geometry (``x0``, ``y0``,
``inc``, ``PA``, and the surface parameters).

The bare-array functions ``calculate_structure_function`` and
``calculate_structure_function_stack`` are for fields that are already on
a regular grid.


What does the ``grid`` argument do?
-----------------------------------

The bare-array functions estimate lags on any regular grid, so they cannot
infer the geometry; ``grid`` tells them:

- ``grid='polar'`` (default): the
  ``imagecube.polar_deprojection`` layout, with axis 0 = radius [arcsec]
  and axis 1 = azimuth [deg]. The azimuthally averaged ``S2_i`` is
  suppressed because it would mix units.
- ``grid='cartesian'``: both axes share units (a sky-plane image, a
  simulation slice). ``S2_i`` is then meaningful, but the radius and
  azimuth analyses raise an error.


How do I remove the noise contribution?
---------------------------------------

Independent components add in ``S_2``, so a noise model can be subtracted
lag by lag with ``StructureFunction.subtract`` (or
``StructureFunctionStack.subtract``). Noise ``S_2`` is lag dependent: it
rises from zero and only levels off at ``2σ²`` beyond the noise
correlation scale. Subtracting a flat offset is therefore wrong at small
lags. Two noise models are available:

- ``gaussian_beam_s2``: the analytic ``S_2`` of beam-convolved white noise.
- ``linecube.noise_structure_function``: an empirical noise ``S_2``
  measured from line-free channels, which also captures correlated
  structure added by the imaging pipeline.


How do I draw noise realizations?
---------------------------------

``imagecube.noise_realization`` is the simplest entry point: it fills in the
shape, pixel size and beam from the cube. It wraps two interchangeable
backends:

- ``gaussian_beam_realization``: beam-convolved white noise, carrying only
  the beam's correlation.
- ``StructureFunction.draw_realization``: fields synthesized from a measured
  ``S_2``, typically the output of ``linecube.noise_structure_function``, so
  they also reproduce the pipeline's extra correlated structure.

Comparing ensembles from the two shows how much apparent structure the
simple beam model misses. To draw *signal* fields with a known ``S_2``, for
example to test an estimator, use ``draw_polar_field`` on a grid from
``make_polar_grid``.


``calculate_heuristics`` and ``fit_GRF`` give different correlation lengths. Why?
---------------------------------------------------------------------------------

Since 3.2.0 they shouldn't. ``calculate_heuristics`` now returns the kernel
correlation lengths that ``fit_GRF`` reports. Before 3.2.0 it returned
half-power lags, which are larger by
``HALF_POWER_FACTOR = sqrt(2 ln 2) ≈ 1.177``. Pass
``length_scale='halfpower'`` to get the old numbers back, and keep the factor
in mind when comparing with results made with 3.1.x or earlier.


Can I measure the pitch of the correlation pattern?
---------------------------------------------------

Only from the global surface, with
``StructureFunction.fit_GRF(pitch=True, r_axis=...)``. The slice-based
measurements (``StructureFunctionStack.fit_GRF`` and
``calculate_heuristics``) cut the correlation ellipse along the radial and
azimuthal axes, so they see its chords rather than its principal axes. When
the pattern is pitched, this biases the anisotropy ``ell0phi / ell0r`` low.
For a 30° pitch and a true anisotropy of 3, the slices give about 1.5.


Some function names changed. What are the new ones?
---------------------------------------------------

The API was aligned on the ``calculate_`` / ``fit_`` / ``plot_`` naming
convention in 3.2.0, and the classes dropped their ``2D`` suffix. The old
names still work but raise a ``DeprecationWarning``, and will be removed in
4.0.

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
