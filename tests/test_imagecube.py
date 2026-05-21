"""Smoke tests for :class:`eddy.imagecube`.

Covers FITS load, axis/shape consistency, ``disk_coords`` sanity, and
the ``to_fits`` round-trip. The 3D cube exercises the spectral axis path
through ``_consistent_header``; the 2D map exercises the
spatial-only path.
"""


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


# ---------------------------------------------------------------------------
# linecube.to_momentmap (Phase 4.2). Each method is a dispatch + numerics
# smoke test; the actual moment-map maths is bettermoments' job.
# ---------------------------------------------------------------------------


def test_to_momentmap_zeroth_returns_momentmap(twhya_cube_path):
    pytest.importorskip("bettermoments")
    from eddy import rotationmap
    cube = linecube(twhya_cube_path, FOV=4.0)
    out = cube.to_momentmap(method='zeroth')
    assert isinstance(out, momentmap)
    assert not isinstance(out, rotationmap)
    assert out.data.shape == (cube.yaxis.size, cube.xaxis.size)
    assert np.all(np.isfinite(np.asarray(out.data)))
    # bettermoments' collapse_zeroth returns (M0, dM0); we attach dM0.
    assert hasattr(out, 'error') and out.error is not None
    assert out.error.shape == out.data.shape


def test_to_momentmap_first_returns_rotationmap(twhya_cube_path):
    """method='first' returns a rotationmap because the output is a
    velocity centroid in the same units as ``self.velax`` (m/s)."""
    pytest.importorskip("bettermoments")
    from eddy import rotationmap
    cube = linecube(twhya_cube_path, FOV=4.0)
    out = cube.to_momentmap(method='first')
    assert isinstance(out, rotationmap)
    assert out.data.shape == (cube.yaxis.size, cube.xaxis.size)
    assert out.header['BUNIT'].lower() == 'm/s'
    # Output values should sit inside the cube's velocity range.
    finite = np.asarray(out.data)[np.isfinite(np.asarray(out.data))]
    assert finite.min() >= cube.velax.min() - abs(cube.chan)
    assert finite.max() <= cube.velax.max() + abs(cube.chan)


def test_to_momentmap_quadratic_unpacks_stacked_return(twhya_cube_path):
    """``collapse_quadratic`` returns a stacked 3D ndarray (4, ny, nx)
    rather than a tuple. This test exercises the unpack path."""
    pytest.importorskip("bettermoments")
    from eddy import rotationmap
    cube = linecube(twhya_cube_path, FOV=4.0)
    out = cube.to_momentmap(method='quadratic')
    assert isinstance(out, rotationmap)
    assert out.data.shape == (cube.yaxis.size, cube.xaxis.size)
    # dv0 attached as error.
    assert hasattr(out, 'error') and out.error is not None
    assert out.error.shape == out.data.shape


def test_to_momentmap_clip_zeroes_low_signal(twhya_cube_path):
    """clip=N replaces |data| < N*rms with 0 before collapsing. Check
    that a very large clip (N=1e6) zeroes out everything — the moment
    map should be identically zero except where bettermoments' RMS
    estimate is itself 0 (which would zero the threshold)."""
    pytest.importorskip("bettermoments")
    cube = linecube(twhya_cube_path, FOV=4.0)
    out = cube.to_momentmap(method='zeroth', clip=1e6)
    arr = np.asarray(out.data)
    assert np.all(arr == 0.0)


def test_to_momentmap_unknown_method_raises(twhya_cube_path):
    pytest.importorskip("bettermoments")
    cube = linecube(twhya_cube_path, FOV=4.0)
    with pytest.raises(ValueError, match="Unknown bettermoments method"):
        cube.to_momentmap(method='not_a_real_method')


# ---------------------------------------------------------------------------
# linecube.to_momentmap product= selection. The product suffix picks out
# a specific output of the collapse method, and the returned class is
# decided by the suffix's bettermoments unit (m/s -> rotationmap).
# ---------------------------------------------------------------------------


def test_to_momentmap_product_v0_from_quadratic(twhya_cube_path):
    """product='v0' from quadratic returns a rotationmap with dv0 attached."""
    pytest.importorskip("bettermoments")
    from eddy import rotationmap
    cube = linecube(twhya_cube_path, FOV=4.0)
    out = cube.to_momentmap(method='quadratic', product='v0')
    assert isinstance(out, rotationmap)
    assert out.header['BUNIT'].lower() == 'm/s'
    assert hasattr(out, 'error') and out.error is not None
    assert out.error.shape == out.data.shape


def test_to_momentmap_product_fnu_from_quadratic(twhya_cube_path):
    """product='Fnu' from quadratic returns a momentmap (intensity unit)
    with dFnu attached as the uncertainty."""
    pytest.importorskip("bettermoments")
    from eddy import rotationmap
    cube = linecube(twhya_cube_path, FOV=4.0)
    out = cube.to_momentmap(method='quadratic', product='Fnu')
    assert isinstance(out, momentmap)
    assert not isinstance(out, rotationmap)
    assert out.header['BUNIT'].lower() != 'm/s'
    assert hasattr(out, 'error') and out.error is not None
    assert out.error.shape == out.data.shape


def test_to_momentmap_product_wp50_from_percentiles(twhya_cube_path):
    """percentiles returns 8 products; product='wp50' picks the median
    velocity (rotationmap) and 'dwp50' is attached as .error."""
    pytest.importorskip("bettermoments")
    from eddy import rotationmap
    cube = linecube(twhya_cube_path, FOV=4.0)
    out = cube.to_momentmap(method='percentiles', product='wp50')
    assert isinstance(out, rotationmap)
    assert out.header['BUNIT'].lower() == 'm/s'
    assert hasattr(out, 'error') and out.error is not None
    assert out.error.shape == out.data.shape


def test_to_momentmap_unknown_product_raises(twhya_cube_path):
    pytest.importorskip("bettermoments")
    cube = linecube(twhya_cube_path, FOV=4.0)
    with pytest.raises(ValueError, match="Unknown product"):
        cube.to_momentmap(method='quadratic', product='not_a_real_product')


def test_to_momentmap_uncertainty_product_rejected(twhya_cube_path):
    """Asking for a 'd…' uncertainty suffix as the primary product is
    rejected with a hint to use the value suffix instead."""
    pytest.importorskip("bettermoments")
    cube = linecube(twhya_cube_path, FOV=4.0)
    with pytest.raises(ValueError, match="uncertainty product"):
        cube.to_momentmap(method='quadratic', product='dv0')
