"""Smoke tests for :mod:`eddy.structurefunction`.

Cover the numba kernel via :func:`compute_s2` (analytic recovery, NaN
handling, reference-annulus mode), the ``StructureFunction2D`` result
container (combine, fit_spiral), and the end-to-end path through
``momentmap.compute_structure_function``.

The whole module is skipped if numba is not installed.
"""

import numpy as np
import pytest

numba = pytest.importorskip("numba")  # noqa: F841 - module-level skip

from eddy import StructureFunction2D, StructureFunction2DStack, momentmap
import eddy.structurefunction as sf
from eddy.structurefunction import (
    compute_s2,
    extract_basic_profiles,
    combine_s2_weighted,
    grf_s2_2d_global,
    S2phi,
)


def test_compute_s2_constant_field():
    """A constant field has zero structure function at all lags."""
    f = np.full((32, 40), 7.5)
    S2, counts, mlx, mly = compute_s2(f)
    assert mlx == 16 and mly == 20
    assert S2.shape == (2 * mlx + 1, 2 * mly + 1)
    np.testing.assert_allclose(S2, 0.0)
    # Every lag had at least some valid pairs.
    assert np.all(counts > 0)


def test_compute_s2_linear_gradient_axis0():
    """For ``f(i, j) = i``, ``S_2`` along axis 0 is exactly ``di**2``
    and along axis 1 is exactly zero."""
    N, M = 40, 30
    i_idx = np.arange(N, dtype=float)[:, None]
    f = np.broadcast_to(i_idx, (N, M)).copy()
    S2, _, mlx, mly = compute_s2(f, max_lag_x=10, max_lag_y=8)

    # Axis 0 slice at zero azimuthal lag.
    S2_x = S2[mlx:, mly]
    di = np.arange(mlx + 1, dtype=float)
    np.testing.assert_allclose(S2_x, di**2)

    # Axis 1 slice at zero radial lag (f doesn't vary in j).
    S2_y = S2[mlx, mly:]
    np.testing.assert_allclose(S2_y, 0.0, atol=1e-12)


def test_compute_s2_symmetry():
    """``S_2(-l_x, -l_y) == S_2(l_x, l_y)`` from the mirror-fill."""
    rng = np.random.default_rng(0)
    f = rng.standard_normal((24, 26))
    S2, counts, mlx, mly = compute_s2(f, max_lag_x=8, max_lag_y=9)
    np.testing.assert_allclose(S2, S2[::-1, ::-1])
    np.testing.assert_array_equal(counts, counts[::-1, ::-1])


def test_compute_s2_nan_handling():
    """Masking pixels with NaN gives the same S_2 as a manual loop that
    excludes those pairs."""
    rng = np.random.default_rng(1)
    f = rng.standard_normal((12, 10))
    f[3, 4] = np.nan
    f[7, 1] = np.nan
    S2, counts, mlx, mly = compute_s2(f, max_lag_x=3, max_lag_y=3)

    # Manual reference for one specific lag.
    di, dj = 2, 1
    N, M = f.shape
    acc = 0.0
    cnt = 0
    for i in range(0, N - di):
        for j in range(max(0, -dj), min(M, M - dj)):
            a = f[i, j]
            b = f[i + di, j + dj]
            if np.isnan(a) or np.isnan(b):
                continue
            acc += (b - a) ** 2
            cnt += 1
    ref = acc / cnt
    assert counts[mlx + di, mly + dj] == cnt
    np.testing.assert_allclose(S2[mlx + di, mly + dj], ref)


def test_compute_s2_reference_band():
    """With ``ref_i`` set, ``ref_band=0`` and ``symmetrize=False`` the
    base rows are pinned to a single index. The positive-``l_r`` half
    is the outward statistic and the negative-``l_r`` half is the
    inward statistic; each should match a manual loop in its own
    direction."""
    rng = np.random.default_rng(2)
    f = rng.standard_normal((20, 18))
    mlx, mly = 5, 5
    S2, counts, _, _ = compute_s2(f, max_lag_x=mlx, max_lag_y=mly,
                                  ref_i=10, ref_band=0, symmetrize=False)

    N, M = f.shape
    # Outward: base row 10, partner row 12.
    di, dj = 2, 3
    j_lo, j_hi = max(0, -dj), min(M, M - dj)
    a = f[10, j_lo:j_hi]
    b = f[10 + di, j_lo + dj:j_hi + dj]
    ref_out = float(np.mean((b - a) ** 2))
    np.testing.assert_allclose(S2[mlx + di, mly + dj], ref_out)
    assert counts[mlx + di, mly + dj] == j_hi - j_lo

    # Inward: base row 10, partner row 8 (same dj).
    a = f[10, j_lo:j_hi]
    b = f[10 - di, j_lo + dj:j_hi + dj]
    ref_in = float(np.mean((b - a) ** 2))
    np.testing.assert_allclose(S2[mlx - di, mly + dj], ref_in)
    assert counts[mlx - di, mly + dj] == j_hi - j_lo


def test_compute_s2_symmetrize_averages_inward_outward():
    """``symmetrize=True`` collapses the outward and inward halves into
    a pair-count-weighted average. With equal counts (interior ref_i)
    that's the simple mean of the two manual statistics."""
    rng = np.random.default_rng(3)
    f = rng.standard_normal((20, 18))
    mlx, mly = 5, 5

    S2_raw, _, _, _ = compute_s2(f, max_lag_x=mlx, max_lag_y=mly,
                                 ref_i=10, ref_band=0, symmetrize=False)
    S2_sym, counts_sym, _, _ = compute_s2(f, max_lag_x=mlx, max_lag_y=mly,
                                          ref_i=10, ref_band=0,
                                          symmetrize=True)

    di, dj = 2, 3
    expected = 0.5 * (S2_raw[mlx + di, mly + dj] + S2_raw[mlx - di, mly - dj])
    np.testing.assert_allclose(S2_sym[mlx + di, mly + dj], expected)
    # Symmetrized array equals its (-l_r, -l_y) flip by construction.
    np.testing.assert_allclose(S2_sym, S2_sym[::-1, ::-1])
    # Counts are summed.
    assert counts_sym[mlx + di, mly + dj] == (
        counts_sym[mlx - di, mly - dj]
    )


def test_compute_s2_reference_band_near_edge():
    """For a ``ref_i`` near the outer edge the outward direction has
    very few valid lags but the inward direction has many — exactly
    the case the previous mirror-only kernel could not handle."""
    rng = np.random.default_rng(4)
    N, M = 20, 18
    f = rng.standard_normal((N, M))
    mlx, mly = 8, 3
    S2, counts, _, _ = compute_s2(f, max_lag_x=mlx, max_lag_y=mly,
                                  ref_i=N - 2, ref_band=0,
                                  symmetrize=False)
    # Outward from ref_i=N-2=18: only di=0, 1 reach inside the grid.
    assert counts[mlx + 5, mly] == 0   # not reachable outward
    # Inward is fine: di = -7 leaves partner at row 11 in [0, N).
    assert counts[mlx - 7, mly] > 0
    assert np.isfinite(S2[mlx - 7, mly])


def test_extract_basic_profiles_shapes():
    rng = np.random.default_rng(3)
    f = rng.standard_normal((30, 40))
    S2, _, mlx, mly = compute_s2(f, max_lag_x=8, max_lag_y=10)
    lags_x, lags_y, lags_i, S2_x, S2_y, S2_i = extract_basic_profiles(
        S2, mlx, mly, dx=0.1, dy=0.5, n_bins=20,
    )
    assert lags_x.shape == (mlx + 1,)
    assert lags_y.shape == (mly + 1,)
    assert lags_i.shape == S2_i.shape == (20,)
    assert S2_x.shape == (mlx + 1,)
    assert S2_y.shape == (mly + 1,)


def test_combine_s2_weighted_two_realizations():
    rng = np.random.default_rng(4)
    f1 = rng.standard_normal((16, 16))
    f2 = rng.standard_normal((16, 16))
    S2a, ca, mlx, mly = compute_s2(f1, max_lag_x=5, max_lag_y=5)
    S2b, cb, _, _ = compute_s2(f2, max_lag_x=5, max_lag_y=5)

    S2c, S2e, S2s = combine_s2_weighted([S2a, S2b], [ca, cb])
    assert S2c.shape == S2a.shape == S2e.shape == S2s.shape

    # The combined S_2 must lie between the two inputs at every cell.
    mn = np.minimum(S2a, S2b)
    mx = np.maximum(S2a, S2b)
    assert np.all(S2c >= mn - 1e-9) and np.all(S2c <= mx + 1e-9)


def test_structurefunction2d_from_array_and_combine():
    rng = np.random.default_rng(5)
    f1 = rng.standard_normal((20, 22))
    f2 = rng.standard_normal((20, 22))
    r1 = StructureFunction2D.from_array(f1, dx=0.1, dy=0.5,
                                        max_lag_x=6, max_lag_y=7)
    r2 = StructureFunction2D.from_array(f2, dx=0.1, dy=0.5,
                                        max_lag_x=6, max_lag_y=7)
    combined = r1.combine(r2)
    assert combined.S2.shape == r1.S2.shape
    assert combined.counts.shape == r1.counts.shape
    # combine_s2_weighted attaches error/std arrays on the result.
    assert combined.combined_error.shape == r1.S2.shape
    assert combined.combined_std.shape == r1.S2.shape


def test_structurefunction2d_fit_spiral_m1_recovers_amplitude():
    """Synthesise an ``S_2_y(dphi)`` profile from a known m=1 spiral and
    check that ``fit_spiral`` recovers the amplitude within a few %."""
    dphi = np.linspace(0.0, 180.0, 61)
    A_true, N_true = 0.7, 0.05
    S2_y = S2phi(dphi, N_true, A_true)

    # Build a minimal StructureFunction2D where lags_y / S2_y carry the
    # signal and the 2D arrays are placeholders of the right shape.
    mlx, mly = 3, len(dphi) - 1
    S2 = np.zeros((2 * mlx + 1, 2 * mly + 1))
    counts = np.ones_like(S2, dtype=int)
    result = StructureFunction2D(
        S2=S2, counts=counts, dx=1.0, dy=float(dphi[1] - dphi[0]),
        lags_x=np.arange(mlx + 1), lags_y=dphi, lags_i=dphi[:20],
        S2_x=np.zeros(mlx + 1), S2_y=S2_y, S2_i=np.zeros(20),
        azimuthal_axis="y",
    )
    popt, perr, model_fn = result.fit_spiral(modes=(1,))
    N_fit, A_fit = popt
    # m=1 amplitude enters as A**2 in S2phi, so the sign is degenerate.
    assert abs(abs(A_fit) - A_true) < 0.02
    assert abs(N_fit - N_true) < 0.02
    # The returned model evaluates without error.
    assert np.allclose(model_fn(dphi), S2_y, atol=1e-6)


@pytest.mark.slow
def test_momentmap_compute_structure_function_stack(hd163296_v0_path):
    """Stack over three reference radii on the HD163296 fixture; check
    shapes, that ``ref_r`` is recorded per-result, and that a
    single-element stack matches a direct ``compute_structure_function``
    call at the same radius."""
    cube = momentmap(hd163296_v0_path, FOV=6.0)
    rgrid = np.linspace(0.5, 2.5, 60)
    tgrid = np.linspace(-np.pi, np.pi, 90)
    geom = dict(inc=46.7, PA=312.0, rgrid=rgrid, tgrid=tgrid,
                max_lag_r=1.0, max_lag_phi=120.0, n_bins=20)

    ref_rs = np.array([1.0, 1.5, 2.0])
    stack = cube.compute_structure_function_stack(
        ref_rs=ref_rs, ref_band=0.05, **geom,
    )
    assert isinstance(stack, StructureFunction2DStack)
    assert len(stack) == 3
    assert stack.S2_stack.shape == (3,) + stack[0].S2.shape
    assert stack.S2_y_stack.shape == (3, stack[0].S2_y.size)
    assert stack.S2_x_stack.shape == (3, stack[0].S2_x.size)
    # Polar pipeline (arcsec radial / deg azimuthal) leaves S2_i undefined.
    assert stack.S2_i_stack is None
    assert all(r.S2_i is None for r in stack)

    # The deprojected grid is shared (computed once).
    assert stack.gridded is not None
    assert stack.x_grid is not None and stack.y_grid is not None

    # Per-result ``ref`` should be close to the requested ``ref_r``
    # (within one grid spacing — ``argmin`` snaps to the nearest bin).
    for r0, res in zip(ref_rs, stack):
        assert abs(res.ref - r0) <= (rgrid[1] - rgrid[0])

    # Single-element stack should match a direct call at the same radius.
    one = cube.compute_structure_function(ref_r=1.5, ref_band=0.05, **geom)
    np.testing.assert_allclose(one.S2, stack[1].S2)
    np.testing.assert_array_equal(one.counts, stack[1].counts)


def test_structurefunction2dstack_fit_spiral_smoke():
    """``StructureFunction2DStack.fit_spiral`` returns (popt, perr, model_fns)
    where popt/perr have shape ``(N_ref, 1 + len(modes))``."""
    dphi = np.linspace(0.0, 180.0, 41)
    mlx = 3
    mly = dphi.size - 1

    def _make_result(A_true, N_true=0.05):
        S2 = np.zeros((2 * mlx + 1, 2 * mly + 1))
        counts = np.ones_like(S2, dtype=int)
        return StructureFunction2D(
            S2=S2, counts=counts, dx=1.0,
            dy=float(dphi[1] - dphi[0]),
            lags_x=np.arange(mlx + 1), lags_y=dphi,
            lags_i=dphi[:10],
            S2_x=np.zeros(mlx + 1),
            S2_y=S2phi(dphi, N_true, A_true),
            S2_i=np.zeros(10),
            azimuthal_axis="y",
        )

    amps_true = np.array([0.3, 0.5, 0.7])
    results = [_make_result(A) for A in amps_true]
    stack = StructureFunction2DStack(ref_rs=[1.0, 1.5, 2.0],
                                     ref_band=0.05, results=results)
    popt, perr, model_fns = stack.fit_spiral(modes=(1,))
    assert popt.shape == (3, 2)
    assert perr.shape == (3, 2)
    assert len(model_fns) == 3
    # Amplitudes recovered within a few %.
    np.testing.assert_allclose(np.abs(popt[:, 1]), amps_true, atol=0.02)


@pytest.mark.parametrize("kw", [
    dict(sigma=1.3, alphar=0.7, ell0r=0.4, alphaphi=1.1, ell0phi=0.3,
         r0=1.0, pitch=15.0),
    dict(sigma=0.8, alphar=-0.5, ell0r=0.6, alphaphi=None, ell0phi=None,
         r0=1.5, pitch=-30.0),
    dict(sigma=1.0, alphar=0.5, ell0r=0.5, alphaphi=0.5, ell0phi=0.5,
         r0=1.0, pitch=45.0),
])
def test_grf_s2_2d_global_numba_matches_python(monkeypatch, kw):
    """The numba kernel path of ``grf_s2_2d_global`` must reproduce the
    pure-Python reference to machine precision, including the NaN pattern
    for empty base rows and non-positive radii."""
    r_axis = np.linspace(0.2, 4.0, 45)
    lags_x = np.linspace(0.0, 2.0, 18)
    lags_y_deg = np.linspace(-90.0, 90.0, 19)

    numba_out = np.asarray(grf_s2_2d_global(r_axis, lags_x, lags_y_deg, **kw))
    monkeypatch.setattr(sf, "_HAS_NUMBA", False)
    python_out = np.asarray(grf_s2_2d_global(r_axis, lags_x, lags_y_deg, **kw))

    # NaN masks (empty base rows) must line up exactly.
    np.testing.assert_array_equal(np.isnan(numba_out), np.isnan(python_out))
    fin = np.isfinite(numba_out)
    np.testing.assert_allclose(numba_out[fin], python_out[fin],
                               rtol=0, atol=1e-12)


def test_grf_s2_2d_global_physical_limits():
    """``S_2`` of the GRF surface is 0 at zero lag, non-negative, and
    saturates towards ``2 sigma**2`` as the azimuthal lag grows."""
    r_axis = np.linspace(0.5, 3.0, 60)
    lags_x = np.array([0.0, 0.5, 1.0])
    lags_y_deg = np.linspace(0.0, 179.0, 40)
    sigma = 1.2
    S2 = np.asarray(grf_s2_2d_global(
        r_axis, lags_x, lags_y_deg, sigma=sigma, alphar=0.5, ell0r=0.3,
        alphaphi=0.5, ell0phi=0.2, r0=1.0, pitch=0.0))

    # Zero radial and azimuthal lag => perfectly correlated => S_2 = 0.
    assert abs(S2[0, 0]) < 1e-9
    # Non-negative and bounded by 2 sigma**2 (+ small roundoff).
    assert np.all(S2 >= -1e-9)
    assert np.all(S2 <= 2.0 * sigma ** 2 + 1e-6)
    # Along the zero-radial-lag row, S_2 grows monotonically with |dphi|
    # and approaches the 2 sigma**2 decorrelation plateau.
    row = S2[0]
    assert row[-1] > row[1]
    assert row[-1] > 1.5 * sigma ** 2


@pytest.mark.slow
def test_momentmap_compute_structure_function_smoke(hd163296_v0_path):
    """End-to-end smoke test: deproject HD163296 v0 onto a polar grid,
    compute the structure function, check shapes and finiteness."""
    cube = momentmap(hd163296_v0_path, FOV=6.0)
    rgrid = np.linspace(0.5, 2.5, 60)
    tgrid = np.linspace(-np.pi, np.pi, 90)
    result = cube.compute_structure_function(
        inc=46.7, PA=312.0, rgrid=rgrid, tgrid=tgrid,
        max_lag_r=1.0, max_lag_phi=120.0, n_bins=25,
    )
    assert isinstance(result, StructureFunction2D)
    assert result.S2.ndim == 2
    assert result.lags_x.size == result.S2_x.size
    assert result.lags_y.size == result.S2_y.size
    # Polar pipeline drops the mixed-units azimuthal average; bin axis is
    # still built so the shape contract for ``n_bins`` is observable.
    assert result.lags_i.size == 25
    assert result.S2_i is None
    # The deprojected field is real-valued so S_2 should be finite and
    # non-negative everywhere it has any pair support.
    has_pairs = result.counts > 0
    assert np.all(result.S2[has_pairs] >= 0.0)
    assert np.all(np.isfinite(result.S2[has_pairs]))
    # x_grid / y_grid attached on the polar path.
    assert result.x_grid is not None and result.y_grid is not None
    assert result.azimuthal_axis == "y"
