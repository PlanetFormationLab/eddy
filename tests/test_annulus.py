"""Smoke tests for the :class:`eddy.annulus` family.

Exercises both :class:`Annulus2D` (built from a moment map via
``rotationmap.get_annulus``) and :class:`Annulus3D` (built from a line
cube via ``linecube.get_annulus``), and runs each ``get_vlos`` fit
method at a budget low enough to keep the suite fast. The goal is
regression coverage of the dispatch + return shapes; numerical
accuracy is established by the tutorial-scale validations recorded in
``REFACTORING_PLAN.md``.
"""

import numpy as np
import pytest

from eddy import Annulus2D, Annulus3D


def _roughly_finite(popt, cvar, n=3):
    assert popt.shape == (n,)
    assert cvar.shape == (n,)
    # Some entries are intentionally NaN (e.g. vz when fix_vlsr is None);
    # at least the rotation component must be finite.
    assert np.isfinite(popt[0])


# ---------------------------------------------------------------------------
# Annulus2D — built from the HD163296 rotation map.
# ---------------------------------------------------------------------------


def test_annulus2d_construction_and_sho(hd163296_rotationmap):
    cube = hd163296_rotationmap
    ann = cube.get_annulus(
        r_min=1.0, r_max=1.0 + cube.bmaj,
        inc=46.7, PA=312.0,
    )
    assert isinstance(ann, Annulus2D)
    assert ann.vobs.size > 0
    assert ann.theta.shape == ann.vobs.shape

    popt, cvar = ann.get_vlos(fit_method='SHO', fit_vrad=True)
    # Annulus2D returns length-3 arrays when fit_vrad=True (vrot, vrad, C).
    assert popt.shape == (3,)
    assert cvar.shape == (3,)
    assert np.isfinite(popt[0])     # vrot
    assert np.isfinite(popt[2])     # C / vlsr-component


def test_annulus2d_rejects_non_sho(hd163296_rotationmap):
    cube = hd163296_rotationmap
    ann = cube.get_annulus(r_min=1.0, r_max=1.0 + cube.bmaj,
                           inc=46.7, PA=312.0)
    with pytest.raises(NotImplementedError):
        ann.get_vlos(fit_method='GP')


# ---------------------------------------------------------------------------
# Annulus3D — built from the TWHya CO cube.
# ---------------------------------------------------------------------------


def _twhya_annulus(cube):
    """Tutorial-3 setup: thin annulus around r=1.0".

    inc is positive (small disk) and PA matches the published value;
    these are needed so the sin(i) deprojection in get_vlos doesn't
    blow up. ``beam_spacing=False`` to keep enough spectra per annulus
    for the fit methods to converge at low budget.
    """
    return cube.get_annulus(
        r_min=1.0, r_max=1.0 + cube.bmaj,
        inc=6.5, PA=151.0,
        beam_spacing=False,
    )


def test_annulus3d_construction(twhya_linecube):
    cube = twhya_linecube
    ann = _twhya_annulus(cube)
    assert isinstance(ann, Annulus3D)
    assert ann.spectra.ndim == 2
    assert ann.spectra.shape[0] == ann.theta.size
    assert ann.spectra.shape[1] == ann.velax.size


def test_annulus3d_get_vlos_sho(twhya_linecube):
    ann = _twhya_annulus(twhya_linecube)
    popt, cvar = ann.get_vlos(fit_method='SHO')
    _roughly_finite(popt, cvar)


def test_annulus3d_get_vlos_dv(twhya_linecube):
    ann = _twhya_annulus(twhya_linecube)
    popt, cvar = ann.get_vlos(fit_method='dV', resample=True)
    assert popt.shape == (3,)
    assert np.isfinite(popt[0])


def test_annulus3d_get_vlos_snr(twhya_linecube):
    ann = _twhya_annulus(twhya_linecube)
    popt, cvar = ann.get_vlos(fit_method='SNR', resample=True)
    assert popt.shape == (3,)
    assert np.isfinite(popt[0])


@pytest.mark.slow
def test_annulus3d_get_vlos_gp(twhya_linecube):
    """The GP path runs an MCMC over a tinygp Matérn-3/2 kernel; even at
    minimum budget it takes ~10–30 s due to JAX compilation. Marked slow
    so the default suite skips it; opt in with ``pytest -m slow``."""
    ann = _twhya_annulus(twhya_linecube)
    popt, cvar = ann.get_vlos(
        fit_method='GP',
        nwalkers=8, nburnin=20, nsteps=20,
        mcmc_kwargs={'progress': False},
    )
    assert popt.shape == (3,)
    assert np.isfinite(popt[0])
