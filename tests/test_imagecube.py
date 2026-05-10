"""Smoke tests for :class:`eddy.imagecube`.

Covers FITS load, axis/shape consistency, ``disk_coords`` sanity, and
the ``to_fits`` round-trip. The 3D cube exercises the spectral axis path
through ``_consistent_header``; the 2D map exercises the
spatial-only path.
"""

import os

import numpy as np
import pytest
from astropy.io import fits

from eddy import imagecube, momentmap, linecube


def test_imagecube_loads_2d(hd163296_v0_path):
    cube = imagecube(hd163296_v0_path, FOV=8.0)
    assert cube.data.ndim == 2
    assert cube.data.shape == (cube.yaxis.size, cube.xaxis.size)
    # FOV=8" => spatial extent should be roughly 8" across (allow a couple
    # of pixels of slack — the clipper rounds outward from the requested FOV).
    assert cube.xaxis.max() - cube.xaxis.min() == pytest.approx(8.0, abs=2 * cube.dpix)
    # x-axis runs east -> west (decreasing) in arcsec offsets.
    assert cube.xaxis[0] > cube.xaxis[-1]


def test_imagecube_loads_3d(twhya_cube_path):
    cube = linecube(twhya_cube_path, FOV=4.0)
    assert cube.data.ndim == 3
    nchan, ny, nx = cube.data.shape
    assert nx == cube.xaxis.size
    assert ny == cube.yaxis.size
    assert nchan == cube.velax.size
    # Velocity axis monotonic.
    diffs = np.diff(cube.velax)
    assert np.all(diffs > 0) or np.all(diffs < 0)


def test_disk_coords_midplane_consistency(hd163296_v0_path):
    """At inc=0, PA=0 the deprojected radius matches the on-sky radius and
    flatten=True returns 1D arrays of the right size."""
    cube = momentmap(hd163296_v0_path, FOV=8.0)
    r2d, phi2d, _ = cube.disk_coords(inc=0.0, PA=0.0)
    assert r2d.shape == cube.data.shape
    assert phi2d.shape == cube.data.shape
    assert r2d.min() >= 0.0
    # Flat path returns 1D arrays of the right length.
    rflat, pflat, _ = cube.disk_coords(inc=0.0, PA=0.0, flatten=True)
    assert rflat.ndim == pflat.ndim == 1
    assert rflat.size == cube.data.size
    # At inc=0 PA=0, deprojected radius equals sky radius.
    xs, ys = np.meshgrid(cube.xaxis, cube.yaxis)
    rsky = np.hypot(xs, ys)
    np.testing.assert_allclose(np.asarray(r2d), rsky, atol=1e-9)


def test_disk_coords_flared(hd163296_v0_path):
    """The flared (analytic) surface produces non-zero z and finite r/phi."""
    cube = imagecube(hd163296_v0_path, FOV=8.0)
    r, phi, z = cube.disk_coords(inc=46.7, PA=312.0, z0=0.3, psi=1.25)
    assert np.all(np.isfinite(np.asarray(r)))
    assert np.all(np.isfinite(np.asarray(phi)))
    # z is non-zero somewhere outside the cavity.
    assert np.any(np.asarray(z) > 0.0)


def test_to_fits_round_trip_2d(hd163296_v0_path, tmp_path):
    cube = momentmap(hd163296_v0_path, FOV=8.0)
    out = tmp_path / "roundtrip_2d.fits"
    cube.to_fits(str(out))
    assert out.exists()

    cube2 = momentmap(str(out))
    np.testing.assert_allclose(np.asarray(cube2.data), np.asarray(cube.data),
                               equal_nan=True)
    np.testing.assert_allclose(cube2.xaxis, cube.xaxis, atol=1e-9)
    np.testing.assert_allclose(cube2.yaxis, cube.yaxis, atol=1e-9)


def test_to_fits_round_trip_3d(twhya_cube_path, tmp_path):
    cube = linecube(twhya_cube_path, FOV=4.0)
    out = tmp_path / "roundtrip_3d.fits"
    cube.to_fits(str(out))

    cube2 = linecube(str(out))
    assert cube2.data.shape == cube.data.shape
    np.testing.assert_allclose(cube2.velax, cube.velax, rtol=1e-9, atol=1e-3)
    np.testing.assert_allclose(np.asarray(cube2.data), np.asarray(cube.data),
                               equal_nan=True)


def test_to_fits_header_consistency_after_clip(hd163296_v0_path, tmp_path):
    """After FOV clipping, the rebuilt header's NAXIS and CRPIX match the
    new live shape."""
    cube = momentmap(hd163296_v0_path, FOV=8.0)
    out = tmp_path / "clipped.fits"
    cube.to_fits(str(out))
    with fits.open(str(out)) as hdul:
        hdr = hdul[0].header
    assert hdr['NAXIS1'] == cube.xaxis.size
    assert hdr['NAXIS2'] == cube.yaxis.size
    assert hdr['CRPIX1'] == pytest.approx(0.5 * (cube.xaxis.size + 1))
    assert hdr['CRPIX2'] == pytest.approx(0.5 * (cube.yaxis.size + 1))


def test_momentmap_rejects_3d(twhya_cube_path):
    with pytest.raises(ValueError, match="momentmap expects 2D data"):
        momentmap(twhya_cube_path)
