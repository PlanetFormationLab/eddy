.. structurefunction:

Structure Functions
===================

The :mod:`eddy.structurefunction` module computes the second-order structure
function,

.. math::

    S_2(\ell) = \langle\,[f(x+\ell) - f(x)]^2\,\rangle,

on a polar ``(radius, azimuth)`` grid. The :class:`StructureFunction2D` class
holds a single 2D structure function — either global or anchored at a reference
annulus — together with its radial and azimuthal slices, while
:class:`StructureFunction2DStack` collects results across a range of reference
radii for radius-resolved analyses. Both provide tools to denoise (``subtract``
a noise model), ``combine`` realizations, ``collapse`` to a global statistic,
and reduce to scalar summaries (``plateau``, ``half_power_lag``). A stack can
also be fit with a parametric anisotropic Gaussian-random-field model
(``fit_GRF``) for the correlation lengths and their radial scaling.

A worked example is given in the
:doc:`structure function tutorial </tutorials/tutorial_7_structurefunction>`.

For real (sky-plane) data, build a stack directly from a map with
:meth:`eddy.momentmap.momentmap.compute_structure_function_stack`, which
deprojects onto the polar grid first.


The 2D structure function
--------------------------

.. autoclass:: eddy.structurefunction.StructureFunction2D
   :members:


The radius-resolved stack
-------------------------

.. autoclass:: eddy.structurefunction.StructureFunction2DStack
   :members:


Module functions
----------------

.. autofunction:: eddy.structurefunction.structure_function_ensemble

.. autofunction:: eddy.structurefunction.gaussian_beam_s2

.. autofunction:: eddy.structurefunction.compute_s2

.. autofunction:: eddy.structurefunction.setup_lag_coords

.. autofunction:: eddy.structurefunction.extract_basic_profiles

.. autofunction:: eddy.structurefunction.combine_s2_weighted


Azimuthal spiral model
----------------------

.. autofunction:: eddy.structurefunction.S2phi

.. autofunction:: eddy.structurefunction.S2phi_singlemodel


Theoretical structure functions
-------------------------------

Forward models for the second-order structure function of an anisotropic,
non-stationary Gaussian random field (the model fit by
:meth:`StructureFunction2DStack.fit_GRF`), with an optional deterministic
grand-design spiral contribution.

.. autofunction:: eddy.structurefunction.grf_s2_slices

.. autofunction:: eddy.structurefunction.grf_s2_2d_global

.. autofunction:: eddy.structurefunction.predict_s2_slices

.. autofunction:: eddy.structurefunction.predict_s2_2d

.. autofunction:: eddy.structurefunction.predict_spiral_s2_slices

.. autofunction:: eddy.structurefunction.predict_spiral_s2_2d

.. autofunction:: eddy.structurefunction.ell_r

.. autofunction:: eddy.structurefunction.ell_phi
