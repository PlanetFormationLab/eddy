Velocity Profiles FAQ
=====================

Common questions about extracting velocities from annuli of spectra with
:class:`eddy.linecube.linecube` and :class:`eddy.annulus.annulus`. For a
walkthrough, start with
:doc:`An Introduction to Working with Annuli </tutorials/tutorial_3>`.


What is an annulus, and why fit one?
------------------------------------

An annulus is the set of spectra extracted from a narrow range of disk
radii, made with ``linecube.get_annulus``. If the disk is close to
azimuthally symmetric, every spectrum in the annulus has the same
intrinsic profile, Doppler shifted by the projected velocity at its polar
angle. Undoing that shift for the right velocity lines the spectra up.
Because this uses the full spectra rather than a single centroid per
pixel, it can recover velocities from weaker or more complex lines than a
rotation-map fit can.

.. raw:: html

   <div class="eddy-figure">
     <div class="eddy-figure-body">
       <eddy-deprojection vtrue="400" nspec="40"></eddy-deprojection>
       <noscript>
         <img src="../_static/deprojection_example.png"
              alt="Incorrect versus correct rotation velocity, with residuals"
              style="width:100%">
       </noscript>
     </div>
     <div class="eddy-figure-caption">
       Sweep the assumed projected rotation velocity: the deprojected spectra
       smear at the wrong value and snap together at the right one. The lower
       panel is the scatter about the stacked profile, which is the quantity
       <code class="literal">annulus.get_vlos()</code> minimises. Synthetic
       annulus, illustrative.
     </div>
   </div>


Which ``fit_method`` should I use in ``get_vlos``?
--------------------------------------------------

``annulus.get_vlos`` offers four ways of judging when the spectra are
aligned (compared in
:doc:`Different Methods for Inferring Velocities </tutorials/tutorial_4>`):

- ``'SHO'`` (default): fit the line centroids with a simple harmonic
  oscillator. Fast, and the only method that can return ``v_z``.
- ``'dV'``: minimise the width of the stacked line.
- ``'SNR'``: maximise the signal to noise of the stacked line.
- ``'GP'``: model the stacked line with a Gaussian process, which drops the
  assumption of a Gaussian line profile and returns posterior
  uncertainties via MCMC.

The ``'dV'`` and ``'SNR'`` methods currently return no uncertainty, and both
assume a roughly Gaussian line. Use ``resample=False`` with ``'GP'`` and
``resample=True`` with the others.


How is the polar angle defined?
-------------------------------

The polar angle is measured from the red-shifted semi-major axis (the
position angle's direction), going east of north. So ``phi = 0`` is the
red-shifted side and ``phi = ±180`` degrees is the blue-shifted side.


What are the sign conventions for the velocities?
-------------------------------------------------

The returned velocities are deprojected. Positive ``v_r`` points away from
the star and positive ``v_z`` points away from the midplane. Each annulus
is modelled as

.. math::

    v_0(\phi) = v_{\phi} \cos(\phi) \sin(|i|)
              - v_{\rm r} \sin(\phi) \sin(i)
              - v_{\rm z} \cos(i) + v_{\rm lsr}.

If ``v_phi`` comes out negative, the position angle is probably 180
degrees off.


How do I get vertical velocities?
---------------------------------

``v_z`` and ``v_lsr`` both shift the whole annulus by a constant, so they
can't be told apart from one annulus alone. Fix the systemic velocity with
``fix_vlsr`` (in [m/s]) and the constant offset is read as ``v_z``. Without
``fix_vlsr``, the third component returned is ``v_lsr`` and ``v_z = 0`` is
assumed. From a rotation map, ``rotationmap.fit_annuli(fit_vz=True,
fix_vlsr=...)`` does the same thing.


Do I need the full cube, or will a rotation map do?
---------------------------------------------------

If you already have a map of line centroids,
:meth:`~eddy.rotationmap.rotationmap.fit_annuli` splits it into annuli and
fits the same harmonic-oscillator model without recomputing centroids from
the cube. It is equivalent to ``fit_method='SHO'``. Use the cube when you
want the spectrum-based methods.


Why are my uncertainties so small?
----------------------------------

Neighbouring pixels within a beam are correlated, but the fit treats them as
independent, so the formal errors mostly reflect how stiff the model is.
Setting ``beam_spacing`` draws pixels roughly that fraction of a beam
apart. ``niter`` then repeats the fit with a fresh random draw of pixels each
time and combines the results, which gives a more honest spread.
``niter > 1`` only makes sense with ``beam_spacing > 0``.


Can I resample the stacked spectrum to finer channels?
------------------------------------------------------

Yes. Because each spectrum's Doppler shift is not tied to the channel grid,
the stacked spectrum samples the underlying line more finely than the data
do. In ``deprojected_spectrum``, ``resample`` given as a float sets the new
channel spacing in [m/s], and given as an integer it sets how many times
finer than the native spacing to sample.
