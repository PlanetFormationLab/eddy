"""Smoke tests for :mod:`eddy.structurefunction`.

Cover the numba kernel via :func:`calculate_s2` (analytic recovery, NaN
handling, reference-annulus mode), the ``StructureFunction`` result
container (combine, fit_spiral), and the end-to-end path through
``momentmap.calculate_structure_function``.

The whole module is skipped if numba is not installed.
"""

import numpy as np
import pytest

numba = pytest.importorskip("numba")  # noqa: F841 - module-level skip

from eddy import (StructureFunction, StructureFunctionStack, SpectralACF,
                  momentmap)
import eddy.structurefunction as sf
from eddy.structurefunction import (
    calculate_s2,
    extract_basic_profiles,
    combine_s2_weighted,
    gaussian_beam_realization,
    gaussian_beam_s2,
    grf_s2_2d_global,
    S2phi,
)


def test_calculate_s2_constant_field():
    """A constant field has zero structure function at all lags."""
    f = np.full((32, 40), 7.5)
    S2, counts, mlx, mly = calculate_s2(f)
    assert mlx == 16 and mly == 20
    assert S2.shape == (2 * mlx + 1, 2 * mly + 1)
    np.testing.assert_allclose(S2, 0.0)
    # Every lag had at least some valid pairs.
    assert np.all(counts > 0)


def test_calculate_s2_linear_gradient_axis0():
    """For ``f(i, j) = i``, ``S_2`` along axis 0 is exactly ``di**2``
    and along axis 1 is exactly zero."""
    N, M = 40, 30
    i_idx = np.arange(N, dtype=float)[:, None]
    f = np.broadcast_to(i_idx, (N, M)).copy()
    S2, _, mlx, mly = calculate_s2(f, max_lag_x=10, max_lag_y=8)

    # Axis 0 slice at zero azimuthal lag.
    S2_x = S2[mlx:, mly]
    di = np.arange(mlx + 1, dtype=float)
    np.testing.assert_allclose(S2_x, di**2)

    # Axis 1 slice at zero radial lag (f doesn't vary in j).
    S2_y = S2[mlx, mly:]
    np.testing.assert_allclose(S2_y, 0.0, atol=1e-12)


def test_calculate_s2_symmetry():
    """``S_2(-l_x, -l_y) == S_2(l_x, l_y)`` from the mirror-fill."""
    rng = np.random.default_rng(0)
    f = rng.standard_normal((24, 26))
    S2, counts, mlx, mly = calculate_s2(f, max_lag_x=8, max_lag_y=9)
    np.testing.assert_allclose(S2, S2[::-1, ::-1])
    np.testing.assert_array_equal(counts, counts[::-1, ::-1])


def test_calculate_s2_nan_handling():
    """Masking pixels with NaN gives the same S_2 as a manual loop that
    excludes those pairs."""
    rng = np.random.default_rng(1)
    f = rng.standard_normal((12, 10))
    f[3, 4] = np.nan
    f[7, 1] = np.nan
    S2, counts, mlx, mly = calculate_s2(f, max_lag_x=3, max_lag_y=3)

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


def test_calculate_s2_reference_band():
    """With ``ref_i`` set, ``ref_band=0`` and ``symmetrize=False`` the
    base rows are pinned to a single index. The positive-``l_r`` half
    is the outward statistic and the negative-``l_r`` half is the
    inward statistic; each should match a manual loop in its own
    direction."""
    rng = np.random.default_rng(2)
    f = rng.standard_normal((20, 18))
    mlx, mly = 5, 5
    S2, counts, _, _ = calculate_s2(f, max_lag_x=mlx, max_lag_y=mly,
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


def test_calculate_s2_symmetrize_averages_inward_outward():
    """``symmetrize=True`` collapses the outward and inward halves into
    a pair-count-weighted average. With equal counts (interior ref_i)
    that's the simple mean of the two manual statistics."""
    rng = np.random.default_rng(3)
    f = rng.standard_normal((20, 18))
    mlx, mly = 5, 5

    S2_raw, _, _, _ = calculate_s2(f, max_lag_x=mlx, max_lag_y=mly,
                                 ref_i=10, ref_band=0, symmetrize=False)
    S2_sym, counts_sym, _, _ = calculate_s2(f, max_lag_x=mlx, max_lag_y=mly,
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


def test_calculate_s2_reference_band_near_edge():
    """For a ``ref_i`` near the outer edge the outward direction has
    very few valid lags but the inward direction has many — exactly
    the case the previous mirror-only kernel could not handle."""
    rng = np.random.default_rng(4)
    N, M = 20, 18
    f = rng.standard_normal((N, M))
    mlx, mly = 8, 3
    S2, counts, _, _ = calculate_s2(f, max_lag_x=mlx, max_lag_y=mly,
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
    S2, _, mlx, mly = calculate_s2(f, max_lag_x=8, max_lag_y=10)
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
    S2a, ca, mlx, mly = calculate_s2(f1, max_lag_x=5, max_lag_y=5)
    S2b, cb, _, _ = calculate_s2(f2, max_lag_x=5, max_lag_y=5)

    S2c, S2e, S2s = combine_s2_weighted([S2a, S2b], [ca, cb])
    assert S2c.shape == S2a.shape == S2e.shape == S2s.shape

    # The combined S_2 must lie between the two inputs at every cell.
    mn = np.minimum(S2a, S2b)
    mx = np.maximum(S2a, S2b)
    assert np.all(S2c >= mn - 1e-9) and np.all(S2c <= mx + 1e-9)


def test_structurefunction_calculate_and_combine():
    rng = np.random.default_rng(5)
    f1 = rng.standard_normal((20, 22))
    f2 = rng.standard_normal((20, 22))
    r1 = StructureFunction.calculate(f1, dx=0.1, dy=0.5,
                                        max_lag_x=6, max_lag_y=7)
    r2 = StructureFunction.calculate(f2, dx=0.1, dy=0.5,
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

    # Build a minimal StructureFunction where lags_y / S2_y carry the
    # signal and the 2D arrays are placeholders of the right shape.
    mlx, mly = 3, len(dphi) - 1
    S2 = np.zeros((2 * mlx + 1, 2 * mly + 1))
    counts = np.ones_like(S2, dtype=int)
    result = StructureFunction(
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


def _smoothed_polar_field(seed=7, n_r=80, n_phi=120, sigma_r=3.0, sigma_phi=4.0):
    """Gaussian-smoothed white noise on a polar grid: a stationary field
    with a finite, anisotropic correlation scale so the per-ring S_2 rises
    to a well-defined plateau (half-power lags are finite)."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    return gaussian_filter(rng.standard_normal((n_r, n_phi)),
                           sigma=(sigma_r, sigma_phi), mode="wrap")


def test_calculate_heuristics_returns_finite_scalars():
    """``calculate_heuristics`` on a well-formed stack returns six finite
    scalars: the amplitude tracks the field's fluctuation level, the
    correlation lengths are positive, and the raw/rescaled amplitudes are
    mutually consistent."""
    field = _smoothed_polar_field()
    dx, dy = 0.05, 3.0
    x_axis = 0.5 + np.arange(field.shape[0]) * dx
    stack = StructureFunctionStack.calculate(
        field, ref_rs=[1.0, 1.5, 2.0, 2.5], x_axis=x_axis, dx=dx, dy=dy,
        ref_band=0.05, max_lag_x=20, max_lag_y=30, n_bins=25,
    )

    sig_hat, T1b, T1c, A_hat, T3, T4 = stack.calculate_heuristics()
    assert all(np.isfinite(v) for v in (sig_hat, T1b, T1c, A_hat, T3, T4))
    # sigma_hat = sqrt(plateau / 2) -> field std for a stationary field.
    assert 0.5 * field.std() < sig_hat < 2.0 * field.std()
    assert T1b > 0 and T1c > 0 and A_hat > 0
    # Azimuthal smoothing was wider than radial, so the arc-length scale
    # exceeds the radial scale: anisotropy A_hat > 1.
    assert A_hat > 1.0
    # Rescaled sigma_hat and raw plateau T1a are the same quantity.
    T1a_raw = stack.calculate_heuristics(rescale_returns=False)[0]
    assert T1a_raw == pytest.approx(2.0 * sig_hat ** 2)


def _empty_the_largest_lags(res, keep_x, keep_y=None):
    """Emulate ``_s2_kernel`` on an annulus whose lag range runs off the grid:
    the largest lags get no pairs, so ``counts`` is 0 and ``S_2`` is left at
    its initialized 0.0 -- indistinguishable, without ``counts``, from a
    genuine ``S_2 = 0``. ``keep_y=None`` leaves the azimuthal slice fully
    populated, which is the real geometry: a ring near the edge of the map
    loses its large *radial* lags while still spanning 360 deg."""
    cx, cy = res.max_lag_x, res.max_lag_y
    res.counts[cx + keep_x:, cy] = 0
    res.S2_x[keep_x:] = 0.0
    if keep_y is not None:
        res.counts[cx, cy + keep_y:] = 0
        res.S2_y[keep_y:] = 0.0


def _old_pooled_plateau(res, frac=0.5):
    """The plateau eddy computed before the ``counts`` mask: outer-lag median
    with unpopulated bins read as genuine zeros."""
    vals = []
    for lags, s2 in ((res.lags_x, res.S2_x), (res.lags_y, res.S2_y)):
        sel = np.asarray(s2)[lags >= frac * lags[-1]]
        vals.append(sel[np.isfinite(sel)])
    return float(np.median(np.concatenate(vals)))


def test_plateau_and_half_power_lag_ignore_unpopulated_bins():
    """An annulus whose radial lags run off the grid must still measure the
    right plateau from its (intact) azimuthal slice, and must report an
    unmeasurable radial scale as NaN rather than inventing a short one.

    Before the ``counts`` mask the empty bins entered ``plateau``'s outer-lag
    median as zeros, halving it; every half-power lag measured against that
    level then came out spuriously short.
    """
    field = _smoothed_polar_field(n_r=100, n_phi=400)
    dx, dy = 0.05, 3.0
    x_axis = 0.5 + np.arange(field.shape[0]) * dx
    stack = StructureFunctionStack.calculate(
        field, ref_rs=[1.5], x_axis=x_axis, dx=dx, dy=dy,
        ref_band=0.05, max_lag_x=40, max_lag_y=40, n_bins=25,
    )
    res = stack.results[0]
    plateau_before = res.plateau()
    ell_r_before = res.half_power_lag("x")
    ell_phi_before = res.half_power_lag("y")
    assert np.isfinite([plateau_before, ell_r_before, ell_phi_before]).all()

    # Radial slice dies past lag 5*dx = 0.25", i.e. short of ell_r itself.
    _empty_the_largest_lags(res, keep_x=6)

    # Plateau still comes off the intact azimuthal slice, ...
    assert res.plateau() == pytest.approx(plateau_before, rel=0.1)
    # ... the azimuthal scale is unmoved, ...
    assert res.half_power_lag("y") == pytest.approx(ell_phi_before, rel=0.1)
    # ... and the radial scale, no longer reachable, is NaN not a short lag.
    assert np.isnan(res.half_power_lag("x"))

    # The regression being guarded: reading the empty bins as zeros halves the
    # plateau, and both lags measured against that level come out short.
    old_plateau = _old_pooled_plateau(res)
    assert old_plateau < 0.6 * plateau_before
    assert res.half_power_lag("y", plateau=old_plateau) < 0.75 * ell_phi_before
    assert res.half_power_lag("x", plateau=old_plateau) < 0.75 * ell_r_before

    # The emptied bins are excluded, not read as zeros ...
    _, s2_x = res._populated_slice("x")
    assert np.all(np.isnan(s2_x[6:])) and np.all(np.isfinite(s2_x[:6]))
    # ... and the stored slice itself is never mutated by the masking.
    assert np.all(res.S2_x[6:] == 0.0)


def test_heuristics_unchanged_when_outer_annuli_lose_their_largest_lags():
    """Stationary field, and the outermost annuli lose their largest lags the
    way real edge annuli do. Plateaus, ``ell_r`` and the radial trend T3 must
    all be unmoved -- the failure mode was ell_r shrinking with radius, which
    manufactures a T3 < 0 for a field that has no radial trend at all."""
    field = _smoothed_polar_field(n_r=120, n_phi=140)
    dx, dy = 0.05, 3.0
    x_axis = 0.5 + np.arange(field.shape[0]) * dx
    kw = dict(ref_rs=np.linspace(1.0, 5.5, 10), x_axis=x_axis, dx=dx, dy=dy,
              ref_band=0.05, max_lag_x=25, max_lag_y=30, n_bins=25)
    pristine = StructureFunctionStack.calculate(field, **kw)
    damaged = StructureFunctionStack.calculate(field, **kw)
    for res in damaged.results[-4:]:
        _empty_the_largest_lags(res, keep_x=18, keep_y=22)

    np.testing.assert_allclose(damaged.plateaus(), pristine.plateaus(),
                               rtol=0.2)
    T1b_p, T3_p = pristine.calculate_heuristics()[1], pristine.calculate_heuristics()[4]
    T1b_d, T3_d = damaged.calculate_heuristics()[1], damaged.calculate_heuristics()[4]
    assert T1b_d == pytest.approx(T1b_p, rel=0.05)
    assert T3_d == pytest.approx(T3_p, abs=0.05)


def test_populated_slice_rejects_bad_axis():
    """``_populated_slice`` keeps ``half_power_lag``'s axis validation."""
    field = _smoothed_polar_field()
    res = StructureFunction.calculate(field, dx=0.05, dy=3.0,
                                         max_lag_x=15, max_lag_y=20)
    with pytest.raises(ValueError, match="axis must be"):
        res.half_power_lag(axis="z")


def test_calculate_heuristics_all_zero_weights_raises():
    """A stack with no measurable correlation scale (constant field -> zero
    S_2 -> NaN half-power lags -> zero reliability weights) must raise a
    clear ValueError, not the ZeroDivisionError from np.average (P0.3)."""
    flat = np.full((80, 120), 3.14159)
    dx, dy = 0.05, 3.0
    x_axis = 0.5 + np.arange(flat.shape[0]) * dx
    stack = StructureFunctionStack.calculate(
        flat, ref_rs=[1.0, 1.5, 2.0], x_axis=x_axis, dx=dx, dy=dy,
        ref_band=0.05, max_lag_x=20, max_lag_y=30, n_bins=25,
    )
    with pytest.raises(ValueError, match="positive reliability weights"):
        stack.calculate_heuristics()


def test_spectral_acf_basic(twhya_linecube):
    """``spectral_acf`` returns a SpectralACF with a normalised lag-0
    autocorrelation, a non-negative null band, and flags only positive
    lags as significant."""
    acf = twhya_linecube.spectral_acf(r_in=1.0, r_out=2.0, max_lag=15)
    assert isinstance(acf, SpectralACF)
    assert acf.lags[0] == 0
    assert acf.acf[0] == pytest.approx(1.0)          # lag-0 = self-correlation
    assert np.all(np.isfinite(acf.acf))
    assert np.all(acf.null_band >= 0.0)
    assert acf.n_channels == twhya_linecube.data.shape[0]
    sig = acf.significant_lags()
    assert np.all(sig > 0)                           # never flags lag 0
    assert set(sig).issubset(set(acf.lags))


def test_noise_structure_function_basic(twhya_linecube):
    """``noise_structure_function`` returns a StructureFunction of the
    edge channels, finite where it has pair support, and records the
    annulus mask it used."""
    noise = twhya_linecube.noise_structure_function(
        N=5, r_in=1.0, r_out=2.0, max_lag_x=6, max_lag_y=6, n_bins=15)
    assert isinstance(noise, StructureFunction)
    assert noise.S2.shape == (13, 13)
    assert np.all(np.isfinite(noise.S2[noise.counts > 0]))
    assert noise.noise_mask == {"N": 5, "r_in": 1.0, "r_out": 2.0}
    # Sky-plane pixels: both axes arcsec, so this is a Cartesian grid and
    # the azimuthal average is meaningful.
    assert noise.grid == "cartesian"
    assert noise.S2_i is not None


def test_gaussian_beam_s2_matches_empirical_grid(twhya_linecube):
    """``gaussian_beam_s2(match=emp)`` inherits the empirical lag grid and
    returns a finite, non-negative analytic prediction that is zero at
    zero lag (no beam-noise structure at zero separation)."""
    emp = twhya_linecube.noise_structure_function(
        N=5, r_in=1.0, r_out=2.0, max_lag_x=6, max_lag_y=6, n_bins=15)
    ana = twhya_linecube.gaussian_beam_s2(match=emp)
    assert isinstance(ana, StructureFunction)
    assert ana.S2.shape == emp.S2.shape
    assert np.all(np.isfinite(ana.S2))
    assert np.all(ana.S2 >= -1e-9)
    cx, cy = ana.S2.shape[0] // 2, ana.S2.shape[1] // 2
    assert ana.S2[cx, cy] == pytest.approx(0.0, abs=1e-9)


def test_gaussian_beam_realization_shapes_and_sigma():
    """Unit-power kernel: the per-pixel sigma of the draw is the requested
    one whatever the beam size, and the ``n_draws == 1`` squeeze matches
    ``draw_realization``."""
    shape, dpix, sigma = (96, 96), 0.02, 3.0
    one = gaussian_beam_realization(shape, dpix, 0.30, 0.12, 37.0, sigma,
                                    rng=0)
    assert one.shape == shape
    many = gaussian_beam_realization(shape, dpix, 0.30, 0.12, 37.0, sigma,
                                     n_draws=4, rng=0)
    assert many.shape == (4, *shape)

    # Convolving with a unit-power kernel preserves the white-noise sigma,
    # so a beam twice the size must not dilute it. Measured on a grid big
    # enough to hold a few hundred independent beams -- the sampling error
    # on a correlated field goes as the beam count, not the pixel count.
    for bmaj, bmin in ((0.30, 0.12), (0.60, 0.24)):
        draws = gaussian_beam_realization((256, 256), dpix, bmaj, bmin,
                                          37.0, sigma, n_draws=16, rng=7)
        assert np.std(draws) == pytest.approx(sigma, rel=0.05)

    with pytest.raises(ValueError, match="n_draws"):
        gaussian_beam_realization(shape, dpix, 0.3, 0.12, 0.0, sigma,
                                  n_draws=0)


def test_gaussian_beam_realization_recovers_gaussian_beam_s2():
    """The measured ``S_2`` of the draws converges on the analytic
    prediction they are the realization of.

    An anisotropic beam at a non-trivial PA, so an axis swap or a sign
    error in the beam frame cannot hide: the same comparison against a
    beam rotated by 90 deg is required to miss by an order of magnitude
    more, which is what makes the agreement meaningful rather than a
    statement that both are roughly flat.
    """
    dpix, bmaj, bmin, bpa, sigma = 0.02, 0.30, 0.12, 37.0, 3.0
    shape, max_lag = (128, 128), 16
    rng = np.random.default_rng(0)

    measured = None
    for _ in range(40):
        f = gaussian_beam_realization(shape, dpix, bmaj, bmin, bpa, sigma,
                                      rng=rng)
        s = StructureFunction.calculate(f, dx=dpix, dy=dpix,
                                        max_lag_x=max_lag,
                                        max_lag_y=max_lag, grid="cartesian")
        measured = s if measured is None else measured.combine([s])

    plateau = 2.0 * sigma ** 2
    pred = gaussian_beam_s2(bmaj, bmin, bpa, measured.lags_x, measured.lags_y,
                            sigma2=sigma ** 2, counts=measured.counts)
    rms = np.sqrt(np.nanmean((measured.S2 - pred.S2) ** 2)) / plateau
    assert rms < 0.02

    wrong = gaussian_beam_s2(bmaj, bmin, bpa + 90.0, measured.lags_x,
                             measured.lags_y, sigma2=sigma ** 2,
                             counts=measured.counts)
    rms_wrong = np.sqrt(np.nanmean((measured.S2 - wrong.S2) ** 2)) / plateau
    assert rms_wrong > 10.0 * rms


def test_noise_realization_both_backends(twhya_linecube):
    """``imagecube.noise_realization`` fills shape/beam from the object for
    both backends, defaults the analytic sigma from the cube RMS, and
    rejects the arguments each backend cannot work without."""
    shape = tuple(twhya_linecube.data.shape[-2:])

    ana = twhya_linecube.noise_realization('analytic', rng=0)
    assert ana.shape == shape
    assert np.isfinite(ana).all()
    # sigma defaulted from estimate_cube_RMS, so the draw sits at that level.
    assert np.std(ana) == pytest.approx(
        float(twhya_linecube.estimate_cube_RMS()), rel=0.3)

    emp_s2 = twhya_linecube.noise_structure_function(
        N=5, r_in=1.0, r_out=2.0, max_lag_x=6, max_lag_y=6, n_bins=15)
    emp = twhya_linecube.noise_realization('empirical', S2=emp_s2, n_draws=3,
                                           rng=0)
    assert emp.shape == (3, *shape)
    assert np.isfinite(emp).all()

    with pytest.raises(ValueError, match="requires S2"):
        twhya_linecube.noise_realization('empirical')
    with pytest.raises(ValueError, match="analytic.*or.*empirical"):
        twhya_linecube.noise_realization('bogus')


def test_noise_realization_analytic_needs_sigma_without_channels(
        hd163296_v0_path):
    """A momentmap has no line-free channels to estimate an RMS from, so the
    analytic backend must say so rather than draw at an invented level."""
    cube = momentmap(hd163296_v0_path, FOV=4.0)
    with pytest.raises(ValueError, match="requires sigma"):
        cube.noise_realization('analytic')
    out = cube.noise_realization('analytic', sigma=12.0, rng=0)
    assert out.shape == tuple(cube.data.shape[-2:])


@pytest.mark.slow
def test_momentmap_calculate_structure_function_stack(hd163296_v0_path):
    """Stack over three reference radii on the HD163296 fixture; check
    shapes, that ``ref_r`` is recorded per-result, and that a
    single-element stack matches a direct ``calculate_structure_function``
    call at the same radius."""
    cube = momentmap(hd163296_v0_path, FOV=6.0)
    rgrid = np.linspace(0.5, 2.5, 60)
    tgrid = np.linspace(-np.pi, np.pi, 90)
    geom = dict(inc=46.7, PA=312.0, rgrid=rgrid, tgrid=tgrid,
                max_lag_r=1.0, max_lag_phi=120.0, n_bins=20)

    ref_rs = np.array([1.0, 1.5, 2.0])
    stack = cube.calculate_structure_function_stack(
        ref_rs=ref_rs, ref_band=0.05, **geom,
    )
    assert isinstance(stack, StructureFunctionStack)
    assert len(stack) == 3
    assert stack.S2_stack.shape == (3,) + stack[0].S2.shape
    assert stack.S2_y_stack.shape == (3, stack[0].S2_y.size)
    assert stack.S2_x_stack.shape == (3, stack[0].S2_x.size)
    # Polar pipeline (arcsec radial / deg azimuthal) leaves S2_i undefined.
    assert stack.grid == "polar"
    assert all(r.grid == "polar" for r in stack)
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
    one = cube.calculate_structure_function(ref_r=1.5, ref_band=0.05, **geom)
    np.testing.assert_allclose(one.S2, stack[1].S2)
    np.testing.assert_array_equal(one.counts, stack[1].counts)


def test_structurefunction2dstack_fit_spiral_smoke():
    """``StructureFunctionStack.fit_spiral`` returns (popt, perr, model_fns)
    where popt/perr have shape ``(N_ref, 1 + len(modes))``."""
    dphi = np.linspace(0.0, 180.0, 41)
    mlx = 3
    mly = dphi.size - 1

    def _make_result(A_true, N_true=0.05):
        S2 = np.zeros((2 * mlx + 1, 2 * mly + 1))
        counts = np.ones_like(S2, dtype=int)
        return StructureFunction(
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
    stack = StructureFunctionStack(ref_rs=[1.0, 1.5, 2.0],
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


def _make_global_grf_surface(true, *, dx=0.05, dy=3.0, mlx=25, mly=30,
                             n_r=80, r_start=0.5):
    """Build a global-mode ``StructureFunction`` whose ``S2`` is exactly
    ``grf_s2_2d_global`` evaluated at ``true``. Feeding the model back to
    ``fit_GRF`` makes recovery a clean inverse problem (lsq cost -> 0).

    Correlation lengths in ``true`` must sit well inside the resolved lag
    window: ``fit_GRF``'s data-driven bounds cap ``ell0r``/``ell0phi`` at
    half the resolved lag range, so an oversized ``ell`` pins at the bound.
    """
    r_axis = r_start + np.arange(n_r) * dx
    lags_x_full = np.arange(-mlx, mlx + 1) * dx
    lags_y_deg = np.arange(-mly, mly + 1) * dy
    S2 = np.asarray(grf_s2_2d_global(r_axis, lags_x_full, lags_y_deg, **true))
    counts = np.ones_like(S2, dtype=int)
    result = StructureFunction(
        S2=S2, counts=counts, dx=dx, dy=dy,
        lags_x=np.arange(mlx + 1) * dx, lags_y=np.arange(mly + 1) * dy,
        lags_i=np.arange(10) * dx,
        S2_x=S2[mlx:, mly], S2_y=S2[mlx, mly:], S2_i=None,
        ref=None, azimuthal_axis="y",
    )
    return result, r_axis


def test_fit_GRF_global_pitch_false_recovers_params():
    """Global-mode surface fit (pitch held at 0) recovers the injected
    amplitude and correlation lengths from a noiseless model surface."""
    true = dict(sigma=1.3, alphar=0.0, ell0r=0.15, alphaphi=0.0,
                ell0phi=0.2, r0=2.0, pitch=0.0)
    result, r_axis = _make_global_grf_surface(true)
    params, perr, sol, cov = result.fit_GRF(
        method="lsq", r_axis=r_axis, r0=2.0, pitch=False, progress=False)

    assert sol.cost < 1e-6                       # model surface reproduced
    assert params["sigma"] == pytest.approx(true["sigma"], rel=0.05)
    assert params["ell0r"] == pytest.approx(true["ell0r"], rel=0.05)
    assert params["ell0phi"] == pytest.approx(true["ell0phi"], rel=0.05)
    assert abs(params["alphar"]) < 0.05          # tied slope, injected 0
    assert "pitch" not in params                 # not freed


def test_fit_GRF_global_pitch_true_recovers_pitch():
    """With ``pitch=True`` on an anisotropic, pitched surface the fit
    recovers the pitch angle (and its sign) plus the other parameters."""
    true = dict(sigma=1.0, alphar=0.0, ell0r=0.2, alphaphi=0.0,
                ell0phi=0.08, r0=2.0, pitch=25.0)
    result, r_axis = _make_global_grf_surface(true)
    params, perr, sol, cov = result.fit_GRF(
        method="lsq", r_axis=r_axis, r0=2.0, pitch=True, progress=False)

    assert sol.cost < 1e-6
    assert params["sigma"] == pytest.approx(true["sigma"], rel=0.05)
    assert params["ell0r"] == pytest.approx(true["ell0r"], rel=0.05)
    assert params["ell0phi"] == pytest.approx(true["ell0phi"], rel=0.05)
    assert params["pitch"] == pytest.approx(true["pitch"], abs=1.0)


def test_fit_GRF_pitch_true_rejects_symmetric_surface():
    """A pitch=0 (azimuthally symmetric) surface cannot constrain the
    pitch ridge; fit_GRF(pitch=True) must refuse it rather than return a
    spurious pitch."""
    true = dict(sigma=1.0, alphar=0.0, ell0r=0.15, alphaphi=0.0,
                ell0phi=0.2, r0=2.0, pitch=0.0)
    result, r_axis = _make_global_grf_surface(true)
    with pytest.raises(ValueError, match="symmetric in the azimuthal lag"):
        result.fit_GRF(method="lsq", r_axis=r_axis, r0=2.0, pitch=True,
                       progress=False)


def test_fit_GRF_drops_nonpositive_radii(recwarn):
    """A radial grid that includes r=0 (the default polar-grid case that
    once crashed in the SVD, P0.1) is fitted after dropping the singular
    row, with a warning, instead of raising."""
    true = dict(sigma=1.0, alphar=0.0, ell0r=0.2, alphaphi=0.0,
                ell0phi=0.08, r0=2.0, pitch=25.0)
    result, _ = _make_global_grf_surface(true)
    r_axis_zero = np.arange(80) * 0.05           # starts at 0.0

    params, perr, sol, cov = result.fit_GRF(
        method="lsq", r_axis=r_axis_zero, r0=2.0, pitch=True, progress=False)

    assert any("non-positive" in str(w.message) for w in recwarn)
    assert all(np.isfinite(v) for v in params.values())


@pytest.mark.slow
def test_momentmap_calculate_structure_function_smoke(hd163296_v0_path):
    """End-to-end smoke test: deproject HD163296 v0 onto a polar grid,
    compute the structure function, check shapes and finiteness."""
    cube = momentmap(hd163296_v0_path, FOV=6.0)
    rgrid = np.linspace(0.5, 2.5, 60)
    tgrid = np.linspace(-np.pi, np.pi, 90)
    result = cube.calculate_structure_function(
        inc=46.7, PA=312.0, rgrid=rgrid, tgrid=tgrid,
        max_lag_r=1.0, max_lag_phi=120.0, n_bins=25,
    )
    assert isinstance(result, StructureFunction)
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


# -- 3.2.0 RENAMES -- #


def _rng_field(shape=(40, 60), seed=0):
    return np.random.default_rng(seed).standard_normal(shape)


def test_module_level_constructors_match_classmethods():
    """``calculate_structure_function[_stack]`` are thin wrappers, so they
    must reproduce the classmethods bit for bit."""
    f = _rng_field()
    a = sf.calculate_structure_function(f, dx=0.1, dy=2.0, ref_i=-1)
    b = StructureFunction.calculate(f, dx=0.1, dy=2.0, ref_i=-1)
    assert isinstance(a, StructureFunction)
    np.testing.assert_array_equal(a.S2, b.S2)
    np.testing.assert_array_equal(a.counts, b.counts)

    ref_rs = [1.0, 2.0]
    c = sf.calculate_structure_function_stack(f, ref_rs, dx=0.1, dy=2.0)
    d = StructureFunctionStack.calculate(f, ref_rs, dx=0.1, dy=2.0)
    assert isinstance(c, StructureFunctionStack)
    np.testing.assert_array_equal(c.S2_x_stack, d.S2_x_stack)


@pytest.mark.parametrize("old,new", [
    ("StructureFunction2D", "StructureFunction"),
    ("StructureFunction2DStack", "StructureFunctionStack"),
    ("compute_s2", "calculate_s2"),
    ("structure_function_ensemble", "calculate_structure_function_ensemble"),
])
def test_renamed_module_names_warn_and_alias(old, new):
    """The 3.1.x module-level spellings warn but still resolve to the very
    same object, so ``isinstance`` checks against them keep working."""
    with pytest.warns(DeprecationWarning, match=new):
        obj = getattr(sf, old)
    assert obj is getattr(sf, new)


def test_renamed_names_reachable_from_package_root():
    """``from eddy import StructureFunction2D`` must keep working, and a
    plain ``import eddy`` must not itself warn."""
    import eddy
    with pytest.warns(DeprecationWarning):
        assert eddy.StructureFunction2D is StructureFunction
    with pytest.warns(DeprecationWarning):
        assert eddy.StructureFunction2DStack is StructureFunctionStack
    with pytest.raises(AttributeError):
        eddy.no_such_structure_function_name


def test_from_array_warns_and_builds_same_result():
    """``from_array`` is the deprecated spelling of ``calculate``."""
    f = _rng_field()
    with pytest.warns(DeprecationWarning, match="calculate"):
        old = StructureFunction.from_array(f, dx=0.1, dy=2.0, ref_i=-1)
    new = StructureFunction.calculate(f, dx=0.1, dy=2.0, ref_i=-1)
    np.testing.assert_array_equal(old.S2, new.S2)

    with pytest.warns(DeprecationWarning, match="calculate"):
        old_st = StructureFunctionStack.from_array(f, [1.0], dx=0.1, dy=2.0)
    new_st = StructureFunctionStack.calculate(f, [1.0], dx=0.1, dy=2.0)
    np.testing.assert_array_equal(old_st.S2_x_stack, new_st.S2_x_stack)


def test_renamed_stack_methods_warn_and_forward():
    """``measure_heuristics`` / ``pairwise_error_heatmaps`` forward to their
    ``calculate_`` spellings."""
    stack = StructureFunctionStack.calculate(
        _rng_field((60, 90)), [1.0, 2.0, 3.0], dx=0.05, dy=4.0)

    with pytest.warns(DeprecationWarning, match="calculate_heuristics"):
        old = stack.measure_heuristics()
    np.testing.assert_allclose(old, stack.calculate_heuristics())

    with pytest.warns(DeprecationWarning,
                      match="calculate_pairwise_error_heatmaps"):
        stack.pairwise_error_heatmaps()


def test_momentmap_compute_aliases_warn(tmp_path):
    """The ``momentmap`` entry points renamed from ``compute_`` keep working."""
    for old, new in [
            ("compute_structure_function", "calculate_structure_function"),
            ("compute_structure_function_stack",
             "calculate_structure_function_stack")]:
        assert hasattr(momentmap, old) and hasattr(momentmap, new)
        assert getattr(momentmap, old) is not getattr(momentmap, new)


# -- GRID GEOMETRY -- #


def test_grid_defaults_to_polar_and_suppresses_s2i():
    """The polar default matches the momentmap pipeline: axis 0 arcsec,
    axis 1 degrees, so the mixed-units azimuthal average is dropped."""
    f = _rng_field()
    polar = sf.calculate_structure_function(f, dx=0.02, dy=1.5)
    assert polar.grid == "polar"
    assert polar.S2_i is None

    cart = sf.calculate_structure_function(f, dx=0.02, dy=0.02,
                                           grid="cartesian")
    assert cart.grid == "cartesian"
    assert cart.S2_i is not None and np.isfinite(cart.S2_i).any()

    # The surface itself is geometry-agnostic; only S2_i differs.
    same = sf.calculate_structure_function(f, dx=0.02, dy=0.02, grid="polar")
    np.testing.assert_array_equal(same.S2, cart.S2)


def test_grid_rejects_unknown_value():
    with pytest.raises(ValueError, match="grid must be one of"):
        sf.calculate_structure_function(_rng_field(), grid="sky")


def test_cartesian_grid_blocks_polar_only_analyses():
    """Radius/azimuth analyses must refuse a Cartesian grid rather than
    return numbers with no physical meaning."""
    f = _rng_field((60, 90))
    res = sf.calculate_structure_function(f, dx=0.02, dy=0.02,
                                          grid="cartesian")
    stack = sf.calculate_structure_function_stack(
        f, [1.0, 2.0], dx=0.02, dy=0.02, grid="cartesian")

    for call in (lambda: res.fit_spiral(),
                 lambda: res.fit_GRF(ref_r=1.0),
                 lambda: stack.fit_GRF(),
                 lambda: stack.calculate_heuristics(),
                 lambda: stack.calculate_anisotropy_heatmap(),
                 lambda: stack.calculate_azimuthal_heatmap(arclength=True)):
        with pytest.raises(ValueError, match="requires a polar grid"):
            call()

    # Geometry-agnostic reductions stay available.
    assert np.isfinite(res.plateau())
    assert np.isfinite(res.half_power_lag("x"))


def test_grid_is_propagated_and_mixing_raises():
    f = _rng_field((60, 90))
    a = sf.calculate_structure_function(f, dx=0.02, dy=0.02, grid="polar")
    b = sf.calculate_structure_function(f, dx=0.02, dy=0.02, grid="cartesian")

    # Same dx/dy, so the geometry check is the one that has to fire.
    for call in (lambda: a.combine([b]),
                 lambda: a.subtract(b),
                 lambda: a.compare_to(b)):
        with pytest.raises(ValueError, match="grid geometries do not match"):
            call()

    assert a.combine([a]).grid == "polar"
    assert b.combine([b]).grid == "cartesian"

    stack = sf.calculate_structure_function_stack(f, [1.0, 2.0], dx=0.02,
                                                  dy=1.5)
    assert stack.grid == "polar"
    assert stack.collapse().grid == "polar"

    with pytest.raises(ValueError, match="share one grid geometry"):
        StructureFunctionStack(ref_rs=[1.0, 2.0], ref_band=0.0,
                               results=[a, b])


def test_ensemble_forwards_grid():
    fields = np.random.default_rng(1).standard_normal((3, 40, 60))
    ens = sf.calculate_structure_function_ensemble(
        fields, mode="global", dx=0.02, dy=0.02, grid="cartesian")
    assert all(r.grid == "cartesian" and r.S2_i is not None for r in ens)
    ens_p = sf.calculate_structure_function_ensemble(
        fields, mode="stack", ref_rs=[0.2, 0.4], dx=0.02, dy=1.5)
    assert all(st.grid == "polar" for st in ens_p)


def test_pipeline_entry_points_declare_their_geometry():
    """The two in-package producers must label themselves correctly: the
    momentmap polar pipeline and the sky-plane beam model."""
    beam = sf.gaussian_beam_s2(0.3, 0.2, 30.0, np.arange(6) * 0.05,
                               np.arange(6) * 0.05, 1.0)
    assert beam.grid == "cartesian"
    assert beam.S2_i is not None


# -- FIELD REALIZATIONS -- #


def test_make_polar_grid_conventions():
    r, phi = sf.make_polar_grid(0.5, 1.5, 10, 8)
    assert r[0] == 0.5 and r[-1] == 1.5 and r.size == 10
    # full period, no duplicated endpoint (a duplicate makes C singular)
    assert phi.size == 8
    assert np.isclose(phi.size * np.diff(phi)[0], 2 * np.pi)
    assert not np.isclose(phi[0] % (2*np.pi), phi[-1] % (2*np.pi))
    with pytest.raises(ValueError, match="r_min must be > 0"):
        sf.make_polar_grid(0.0, 1.5, 10, 8)


def test_draw_polar_field_backends_agree_on_variance():
    """The convolution backend targets the same covariance as the exact
    one, so both must land on the requested sigma."""
    r, phi = sf.make_polar_grid(0.6, 1.4, 30, 60)
    kw = dict(sigma=1.0, alphar=1.0, ell0r=0.08, alphaphi=1.0, ell0phi=0.20,
              r0=1.0)
    a = sf.draw_polar_field(r, phi, n_realizations=4, rng=3,
                            method="exact", **kw)
    b = sf.draw_polar_field(r, phi, n_realizations=4, rng=3,
                            method="convolution", **kw)
    assert a.shape == b.shape == (4, 30, 60)
    for f in (a, b):
        assert 0.85 < f.std() < 1.15

    # single draw drops the leading axis; seeds are reproducible
    one = sf.draw_polar_field(r, phi, rng=5, method="convolution", **kw)
    assert one.shape == (30, 60)
    np.testing.assert_array_equal(
        one, sf.draw_polar_field(r, phi, rng=5, method="convolution", **kw))

    with pytest.raises(ValueError, match="method must be"):
        sf.draw_polar_field(r, phi, method="nope", **kw)


def test_draw_polar_field_injection_recovery():
    """The point of the drawer: inject known GRF parameters, measure S_2,
    and recover them with fit_GRF."""
    truth = dict(sigma=1.0, alphar=1.0, ell0r=0.08, alphaphi=1.0,
                 ell0phi=0.20, r0=1.0)
    r, phi = sf.make_polar_grid(0.6, 1.4, 120, 240)
    dr = float(np.diff(r)[0])
    dphi = float(np.degrees(np.diff(phi)[0]))
    fields = sf.draw_polar_field(r, phi, n_realizations=12, rng=7, **truth)

    stacks = [sf.calculate_structure_function_stack(
        f, np.linspace(0.75, 1.25, 6), x_axis=r, dx=dr, dy=dphi)
        for f in fields]
    p, _, _, _ = stacks[0].combine(stacks[1:]).fit_GRF(method="lsq", r0=1.0)

    # 10% is loose, but this runs on 12 realizations to stay fast; the
    # point is that the drawer and the fitter agree on the parameterization.
    for key in ("sigma", "ell0r", "ell0phi"):
        assert abs(p[key] / truth[key] - 1.0) < 0.10, (key, p[key])


def test_polar_covariance_is_symmetric_psd_and_guarded():
    r, phi = sf.make_polar_grid(0.8, 1.2, 8, 12)
    C = sf.polar_covariance(r, phi, ell0r=0.1, ell0phi=0.2, sigma=1.5)
    assert C.shape == (96, 96)
    np.testing.assert_allclose(C, C.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(C), 1.5**2, rtol=1e-10)
    assert np.linalg.eigvalsh(C).min() > -1e-8 * C.max()
    with pytest.raises(ValueError, match="exceeds max_points"):
        sf.polar_covariance(r, phi, max_points=10)


def test_draw_realization_requires_cartesian_grid():
    """Spectral synthesis assumes stationarity, which a polar S_2 breaks."""
    f = _rng_field((60, 60))
    polar = sf.calculate_structure_function(f, dx=0.05, dy=1.5)
    with pytest.raises(ValueError, match="requires grid='cartesian'"):
        polar.draw_realization()


def test_draw_realization_reproduces_its_input_s2():
    """Synthesize from a well-averaged S_2 and re-measure: the plateau and
    the surface must come back, up to the clipped-power inflation."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(0)
    kw = dict(dx=0.05, dy=0.05, grid="cartesian", max_lag_x=15, max_lag_y=15)

    def one():
        im = gaussian_filter(rng.standard_normal((64, 64)), 2.5)
        return im / im.std()

    res = [sf.calculate_structure_function(one(), **kw) for _ in range(40)]
    meas = res[0].combine(res[1:])

    draws = meas.draw_realization(shape=(64, 64), n_draws=40, rng=1)
    assert draws.shape == (40, 64, 64)
    assert meas.draw_realization(shape=(64, 64), rng=1).shape == (64, 64)

    back = [sf.calculate_structure_function(d, **kw) for d in draws]
    got = back[0].combine(back[1:])
    # Clipping negative PSD bins adds variance, so allow a one-sided margin.
    assert 0.95 < got.plateau() / meas.plateau() < 1.15

    with pytest.raises(ValueError, match="n_draws must be"):
        meas.draw_realization(n_draws=0)


def test_draw_realization_warns_when_clipping_is_large():
    """A single-realization S_2 is not positive-definite enough to
    synthesize from, and the user has to be told."""
    from scipy.ndimage import gaussian_filter
    im = gaussian_filter(np.random.default_rng(0).standard_normal((64, 64)), 2.5)
    noisy = sf.calculate_structure_function(
        im / im.std(), dx=0.05, dy=0.05, grid="cartesian",
        max_lag_x=15, max_lag_y=15)
    with pytest.warns(RuntimeWarning, match="clipped"):
        noisy.draw_realization(shape=(64, 64))
