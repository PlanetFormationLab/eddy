"""
Second-order structure function analysis for 2D maps.

The structure function

    S_2(l) = < [f(x + l) - f(x)]^2 >

probes the variance of fluctuations as a function of separation ``l``.
Applied to (r, phi)-deprojected velocity residuals, it reveals coherent
non-Keplerian motions in protoplanetary disks: spiral wakes appear as
power at specific azimuthal lags, while isotropic turbulence produces
a single radius-independent power-law in |l|.

The user-facing entry point in eddy is
:meth:`eddy.momentmap.momentmap.compute_structure_function`. The kernel
and helpers below can also be used directly for analyses that don't
start from an eddy map (e.g. simulations).

Numba is an optional dependency. Install with

    pip install astro-eddy[structurefunction]

or simply ``pip install numba``. ``eddy`` imports without numba; only
the structure-function entry points raise.
"""

import warnings

import numpy as np

try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:  # pragma: no cover - exercised only without numba
    _HAS_NUMBA = False


__all__ = [
    "StructureFunction2D",
    "StructureFunction2DStack",
    "compute_s2",
    "setup_lag_coords",
    "extract_basic_profiles",
    "combine_s2_weighted",
    "structure_function_ensemble",
    "gaussian_beam_s2",
    "grf_s2_slices",
    "grf_s2_2d_global",
    "ell_r",
    "ell_phi",
    "predict_s2_slices",
    "predict_s2_2d",
    "predict_spiral_s2_slices",
    "predict_spiral_s2_2d",
    "S2phi",
    "S2phi_singlemodel",
]


_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


_NUMBA_INSTALL_MSG = (
    "Structure function computation requires numba. "
    "Install with `pip install astro-eddy[structurefunction]` "
    "or `pip install numba`."
)


def _require_numba():
    if not _HAS_NUMBA:
        raise ImportError(_NUMBA_INSTALL_MSG)


# -- KERNEL -- #

if _HAS_NUMBA:

    @njit(parallel=True, cache=True)
    def _s2_kernel(f, max_lag_x, max_lag_y, ref_i, ref_band):
        """Numba kernel for the 2D second-order structure function.

        Parameters are fully typed (no ``None`` defaults). Wrapped by
        :func:`compute_s2`, which handles defaults and dtype promotion.

        NaN values in ``f`` are excluded from both the sum and the pair
        count, so the average is always over finite pairs.

        Two code paths:

        * **Global mode** (``ref_i < 0``): the base index ranges over the
          whole grid, so ``S_2`` has full ``(l_x, l_y) -> (-l_x, -l_y)``
          symmetry; the kernel iterates ``l_x >= 0`` and mirror-fills the
          conjugate half.
        * **Reference-annulus mode** (``ref_i >= 0``): the base row is
          pinned at ``ref_i``, so ``S_2(+l_r)`` (outward pairs) and
          ``S_2(-l_r)`` (inward pairs) are *different* physical
          statistics. Both signs of ``l_x`` are iterated explicitly; only
          the azimuthal mirror is applied (phi-translation symmetry
          survives the radial pin). To collapse the two halves into a
          single direction-agnostic estimator, post-process with
          :func:`_symmetrize_s2` (see the ``symmetrize`` kwarg on
          :func:`compute_s2`).
        """
        N, M = f.shape

        nlx = 2 * max_lag_x + 1
        nly = 2 * max_lag_y + 1
        S2 = np.zeros((nlx, nly), dtype=np.float64)
        counts = np.zeros((nlx, nly), dtype=np.int64)

        use_ref = ref_i >= 0
        ref_lo = 0
        ref_hi = N
        if use_ref:
            ref_lo = max(0, ref_i - ref_band)
            ref_hi = min(N, ref_i + ref_band + 1)
            if ref_lo >= ref_hi:
                use_ref = False

        if use_ref:
            # Iterate both signs of di because the radial direction
            # loses translation symmetry under the reference-annulus
            # pin. dj still has phi-translation symmetry, so iterate
            # dj >= 0 only and mirror.
            for idx in prange(2 * max_lag_x + 1):
                di = idx - max_lag_x
                for dj in range(0, max_lag_y + 1):
                    # i in [ref_lo, ref_hi); i + di must also be in [0, N).
                    i_min = max(ref_lo, -di)
                    i_max = min(ref_hi, N - di)
                    if i_min >= i_max:
                        continue
                    # dj >= 0 in this branch (phi-mirror covers dj < 0).
                    j_min = 0
                    j_max = M - dj

                    acc = 0.0
                    cnt = 0
                    for i in range(i_min, i_max):
                        for j in range(j_min, j_max):
                            fi = f[i, j]
                            fid = f[i + di, j + dj]
                            if np.isnan(fi) or np.isnan(fid):
                                continue
                            diff = fid - fi
                            acc += diff * diff
                            cnt += 1

                    val = acc / cnt if cnt > 0 else 0.0
                    li = idx
                    lj = dj + max_lag_y
                    S2[li, lj] = val
                    counts[li, lj] = cnt

                    if dj != 0:
                        S2[li, -dj + max_lag_y] = val
                        counts[li, -dj + max_lag_y] = cnt
        else:
            for di in prange(0, max_lag_x + 1):
                dj_start = 0 if di == 0 else -max_lag_y

                for dj in range(dj_start, max_lag_y + 1):

                    i_min = 0
                    i_max = N - di
                    j_min = max(0, -dj)
                    j_max = min(M, M - dj)

                    acc = 0.0
                    cnt = 0
                    for i in range(i_min, i_max):
                        for j in range(j_min, j_max):
                            fi = f[i, j]
                            fid = f[i + di, j + dj]
                            if np.isnan(fi) or np.isnan(fid):
                                continue
                            diff = fid - fi
                            acc += diff * diff
                            cnt += 1

                    val = acc / cnt if cnt > 0 else 0.0
                    li = di + max_lag_x
                    lj = dj + max_lag_y
                    S2[li, lj] = val
                    counts[li, lj] = cnt

                    if di != 0 or dj != 0:
                        S2[-di + max_lag_x, -dj + max_lag_y] = val
                        counts[-di + max_lag_x, -dj + max_lag_y] = cnt

        return S2, counts

else:  # pragma: no cover

    def _s2_kernel(*_args, **_kwargs):
        _require_numba()


def _symmetrize_s2(S2, counts):
    """Pair-count-weighted average of ``S_2`` with its ``(-l_x, -l_y)`` conjugate.

    In reference-annulus mode the kernel returns a genuinely two-sided
    ``S_2``: ``S_2(+l_r)`` is the outward statistic (base pixel at
    ``ref_i``, partner at ``ref_i + l_r``) and ``S_2(-l_r)`` is the
    inward statistic. Averaging the two weighted by their pair counts
    collapses them into a single direction-agnostic estimator.
    """
    counts_flip = counts[::-1, ::-1]
    total = counts + counts_flip
    weighted = S2 * counts + S2[::-1, ::-1] * counts_flip
    S2_sym = np.where(total > 0, weighted / np.maximum(total, 1), 0.0)
    return S2_sym, total


def compute_s2(f, max_lag_x=None, max_lag_y=None, ref_i=-1, ref_band=0,
               symmetrize=True):
    """Compute the 2D second-order structure function on a regular grid.

    Args:
        f (ndarray): 2D field with shape (N, M). NaN values are excluded
            from the pair averages.
        max_lag_x (Optional[int]): Maximum lag along axis 0. Defaults to
            ``N // 2``.
        max_lag_y (Optional[int]): Maximum lag along axis 1. Defaults to
            ``M // 2``.
        ref_i (int): Row index of the reference annulus center. If < 0
            (default), averages over all valid base rows (global mode).
            If >= 0, restricts base rows to ``[ref_i - ref_band,
            ref_i + ref_band]``.
        ref_band (int): Half-width in rows of the reference annulus.
            ``0`` selects a single row.
        symmetrize (bool): Only relevant when ``ref_i >= 0``. If
            ``True`` (default), the outward (``+l_r``) and inward
            (``-l_r``) halves are combined by a pair-count-weighted
            average, i.e. ``S_2`` becomes a direction-agnostic
            estimator. If ``False``, both halves are returned
            untouched, so the user can inspect the inward / outward
            asymmetry. Ignored in global mode (``ref_i < 0``), where
            ``S_2`` is already symmetric by construction.

    Returns:
        S2 (ndarray): Structure function with shape
            ``(2*max_lag_x+1, 2*max_lag_y+1)``. Zero lag is at index
            ``[max_lag_x, max_lag_y]``.
        counts (ndarray): Number of finite pairs contributing to each
            lag bin (or the sum thereof when ``symmetrize=True``).
        max_lag_x (int): The resolved value of ``max_lag_x``.
        max_lag_y (int): The resolved value of ``max_lag_y``.
    """
    _require_numba()
    f = np.ascontiguousarray(np.asarray(f, dtype=np.float64))
    N, M = f.shape
    if max_lag_x is None:
        max_lag_x = N // 2
    if max_lag_y is None:
        max_lag_y = M // 2
    S2, counts = _s2_kernel(f, int(max_lag_x), int(max_lag_y),
                            int(ref_i), int(ref_band))
    if symmetrize and ref_i >= 0:
        S2, counts = _symmetrize_s2(S2, counts)
    return S2, counts, max_lag_x, max_lag_y


# -- HELPERS -- #


def setup_lag_coords(max_lag_x, max_lag_y, dx=1.0, dy=1.0):
    """Build physical lag coordinates that match :func:`compute_s2` output.

    Returns:
        lag_x, lag_y (ndarray): 1D lag axes.
        LAG_X, LAG_Y (ndarray): 2D meshgrids.
        lag_mag (ndarray): ``|l|`` in mixed units.
        lag_angle (ndarray): ``arctan2(LAG_Y, LAG_X)`` in [-pi, pi].
    """
    lag_x = np.arange(-max_lag_x, max_lag_x + 1) * dx
    lag_y = np.arange(-max_lag_y, max_lag_y + 1) * dy
    LAG_X, LAG_Y = np.meshgrid(lag_x, lag_y, indexing="ij")
    lag_mag = np.sqrt(LAG_X**2 + LAG_Y**2)
    lag_angle = np.arctan2(LAG_Y, LAG_X)
    return lag_x, lag_y, LAG_X, LAG_Y, lag_mag, lag_angle


def extract_basic_profiles(S2, max_lag_x, max_lag_y, dx=1.0, dy=1.0,
                           n_bins=50, log_spaced=False):
    """Extract x-axis, y-axis, and azimuthally-averaged S_2 profiles.

    The azimuthal average ``S_2_i`` is built by sampling ``S_2`` on circles
    of radius ``|l| = sqrt((l_x/dx)^2 + (l_y/dy)^2)`` in *index* space, then
    labelling the bins with ``r = sqrt(l_x^2 + l_y^2)`` using the raw ``dx``
    and ``dy``. It is only physically meaningful when ``dx`` and ``dy``
    share units (e.g. both arcsec); for the momentmap polar pipeline where
    ``dx`` is arcsec and ``dy`` is degrees the bins mix incommensurate
    units and the result has no physical interpretation. That pipeline
    therefore discards the computed ``S_2_i`` and stores ``None`` on the
    result; this function still returns the array so direct callers can
    decide.

    Args:
        S2 (ndarray): 2D structure function from :func:`compute_s2`.
        max_lag_x, max_lag_y (int): Maximum lags used to build ``S2``.
        dx, dy (float): Physical pixel spacing along axis 0/1.
        n_bins (int): Number of radial bins for the azimuthal average.
        log_spaced (bool): If ``True``, log-spaced radial bins.

    Returns:
        lags_x (ndarray): Positive lags along axis 0 in physical units.
        lags_y (ndarray): Positive lags along axis 1 in physical units.
        lags_i (ndarray): Bin centers for the azimuthally averaged profile.
        S2_x (ndarray): ``S_2`` slice along axis 0 (at ``l_y = 0``).
        S2_y (ndarray): ``S_2`` slice along axis 1 (at ``l_x = 0``).
        S2_i (ndarray): Azimuthally averaged ``S_2(|l|)``. Only physically
            meaningful when ``dx`` and ``dy`` share units (see above).
    """
    from scipy.interpolate import RegularGridInterpolator

    center_x = max_lag_x
    center_y = max_lag_y

    S2_x = S2[center_x:, center_y]
    lags_x = np.arange(0, max_lag_x + 1) * dx

    S2_y = S2[center_x, center_y:]
    lags_y = np.arange(0, max_lag_y + 1) * dy

    lag_x_arr = np.arange(-max_lag_x, max_lag_x + 1) * dx
    lag_y_arr = np.arange(-max_lag_y, max_lag_y + 1) * dy

    max_phys = min(max_lag_x * dx, max_lag_y * dy)
    min_phys = min(dx, dy)
    if log_spaced:
        ell_bins = np.geomspace(min_phys, max_phys, n_bins + 1)
    else:
        ell_bins = np.linspace(min_phys, max_phys, n_bins + 1)
    ell_centers = 0.5 * (ell_bins[:-1] + ell_bins[1:])

    # Interpolate-then-sample rather than histogram-bin: avoids the
    # lattice artifacts that make hard-binned azimuthal averages jagged.
    interp = RegularGridInterpolator(
        (lag_x_arr, lag_y_arr), S2,
        method="linear", bounds_error=False, fill_value=np.nan,
    )
    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    cos_a, sin_a = np.cos(angles), np.sin(angles)

    xs = np.outer(ell_centers, cos_a)           # (n_bins, 360)
    ys = np.outer(ell_centers, sin_a)
    pts_all = np.column_stack([xs.ravel(), ys.ravel()])   # (n_bins*360, 2)
    vals_all = interp(pts_all).reshape(n_bins, 360)
    with np.errstate(invalid="ignore"):
        any_finite = np.any(np.isfinite(vals_all), axis=1)
        S2_i = np.where(any_finite, np.nanmean(vals_all, axis=1), np.nan)

    return lags_x, lags_y, ell_centers, S2_x, S2_y, S2_i


def combine_s2_weighted(S2_list, counts_list):
    """Combine S_2 estimates from multiple realizations by pair-count.

    Args:
        S2_list (list of ndarray): K arrays of shape ``(Nx, Ny)``.
        counts_list (list of ndarray): K arrays of shape ``(Nx, Ny)``.

    Returns:
        S2_combined (ndarray): Weighted mean ``S_2``.
        S2_error (ndarray): Standard error on the mean across
            realizations.
        S2_std (ndarray): Standard deviation across realizations
            (intrinsic scatter).
    """
    K = len(S2_list)
    S2_stack = np.asarray(S2_list)
    counts_stack = np.asarray(counts_list)

    total_counts = np.sum(counts_stack, axis=0)
    weights = counts_stack / np.maximum(total_counts, 1.0)

    S2_combined = np.sum(weights * S2_stack, axis=0)

    residuals = S2_stack - S2_combined[np.newaxis, :, :]
    S2_var = np.sum(weights * residuals**2, axis=0)

    sum_w2 = np.sum(weights**2, axis=0)
    correction = 1.0 / np.maximum(1.0 - sum_w2, 1e-12)
    S2_std = np.sqrt(S2_var * np.minimum(correction, K))
    S2_error = S2_std / np.sqrt(K)

    return S2_combined, S2_error, S2_std


# -- PLOTTING HELPERS -- #


def _resolve_ax(ax):
    """Return ``(fig, ax)``, creating a new figure if ``ax`` is ``None``."""
    import matplotlib.pyplot as plt
    if ax is None:
        return plt.subplots()
    return ax.figure, ax


def _plot_heatmap(ax, X, Y, C, xlabel, cbar_label, return_fig, **kwargs):
    """Draw a ``pcolormesh`` heatmap with a labeled colorbar.

    Standard layout for the three stack heatmap methods: X/Y are the lag
    and ref-radius grids, C is the value surface. The y-axis is always
    labeled ``r_ref (arcsec)``; callers supply the x-axis and colorbar
    labels.
    """
    fig, ax = _resolve_ax(ax)
    kw = dict(shading="auto", rasterized=True)
    kw.update(kwargs)
    pcm = ax.pcolormesh(X, Y, C, **kw)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$r_{\rm ref}$ (arcsec)")
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.ax.set_ylabel(cbar_label, rotation=270, labelpad=13)
    return fig if return_fig else None


# -- ANALYTIC BEAM PREDICTION -- #


def gaussian_beam_s2(bmaj, bmin, bpa, lags_x, lags_y, sigma2,
                     counts=None, n_bins=50, log_spaced=False,
                     x_label="lag_x", y_label="lag_y"):
    """Analytic ``S_2`` for white pixel noise convolved with a 2D Gaussian beam.

    For per-pixel noise with variance ``sigma2`` smoothed by a Gaussian
    beam B, the noise covariance is ``C(l) = sigma2 * rho(l)``, where
    ``rho`` is the normalized beam-beam autocorrelation: a Gaussian with
    the same orientation as the beam but with axis sigmas of
    ``sqrt(2) * sigma_beam`` along major and minor. The structure
    function is then

        S_2(l) = 2 * sigma2 * (1 - rho(l))

    so ``S_2 -> 0`` at zero lag and ``S_2 -> 2*sigma2`` at lags much
    larger than the beam.

    This is the "naive PSF" prediction; the difference between this and
    an empirical ``S_2`` computed from real noise channels exposes any
    extra correlated structure introduced by the imaging pipeline
    (CLEAN residuals, sidelobe leakage, deconvolution bias, etc.).

    Args:
        bmaj, bmin (float): Beam FWHM in the same units as ``lags_x``
            and ``lags_y`` (typically arcsec).
        bpa (float): Beam position angle in degrees. Standard FITS
            convention: measured from the axis-0 direction (image y /
            declination) toward axis-1 (image x / RA). The
            autocorrelation is symmetric under PA -> PA + 180.
        lags_x (ndarray): Positive lags along axis 0, shape
            ``(max_lag_x + 1,)``, as produced by
            :meth:`StructureFunction2D.lags_x`.
        lags_y (ndarray): Positive lags along axis 1, shape
            ``(max_lag_y + 1,)``.
        sigma2 (float): Per-pixel noise variance (e.g.
            ``cube.rms ** 2``).
        counts (Optional[ndarray]): Pair-count grid to attach to the
            returned :class:`StructureFunction2D`, e.g. copied from a
            companion empirical result for like-for-like weighting in
            :meth:`StructureFunction2D.combine`. Defaults to ones.
        n_bins, log_spaced: Forwarded to :func:`extract_basic_profiles`
            for the 1D profile extraction.
        x_label, y_label (str): Lag-axis labels for the returned
            :class:`StructureFunction2D` (default ``"lag_x"`` /
            ``"lag_y"``).

    Returns:
        :class:`StructureFunction2D` whose ``S2`` is the analytic
        prediction on the supplied lag grid.
    """
    lags_x = np.asarray(lags_x, dtype=float)
    lags_y = np.asarray(lags_y, dtype=float)
    if lags_x.ndim != 1 or lags_y.ndim != 1:
        raise ValueError("lags_x and lags_y must be 1D positive-lag arrays.")
    if lags_x[0] != 0.0 or lags_y[0] != 0.0:
        raise ValueError("lags_x and lags_y must start at zero.")

    max_lag_x = lags_x.size - 1
    max_lag_y = lags_y.size - 1
    dx = float(lags_x[1] - lags_x[0]) if max_lag_x > 0 else 1.0
    dy = float(lags_y[1] - lags_y[0]) if max_lag_y > 0 else 1.0

    # Build two-sided lag grids that match compute_s2's output layout.
    lag_x_full = np.arange(-max_lag_x, max_lag_x + 1) * dx
    lag_y_full = np.arange(-max_lag_y, max_lag_y + 1) * dy
    LX, LY = np.meshgrid(lag_x_full, lag_y_full, indexing="ij")

    # Rotate (lag_axis0, lag_axis1) into (l_major, l_minor) in the
    # beam's principal-axis frame. PA is the FITS beam position angle:
    # measured east of north. eddy stores cubes with axis 0 = +DEC
    # (north) and axis 1 = -RA (west), since xaxis is flipped to be
    # monotonically decreasing (imagecube enforces this). So east =
    # -axis1, and the major-axis unit vector at PA east-of-north is
    # (cos PA, -sin PA) in (axis0, axis1) components.
    phi = np.radians(float(bpa))
    cos_p, sin_p = np.cos(phi), np.sin(phi)
    l_maj = LX * cos_p - LY * sin_p
    l_min = LX * sin_p + LY * cos_p

    # Beam autocorrelation sigma is sqrt(2) * beam sigma along each
    # axis -- the convolution-of-Gaussian-with-itself rule.
    sigma_maj = float(bmaj) * _FWHM_TO_SIGMA
    sigma_min = float(bmin) * _FWHM_TO_SIGMA
    sigma_maj_auto2 = 2.0 * sigma_maj * sigma_maj
    sigma_min_auto2 = 2.0 * sigma_min * sigma_min

    rho = np.exp(-(l_maj * l_maj) / (2.0 * sigma_maj_auto2)
                 - (l_min * l_min) / (2.0 * sigma_min_auto2))
    S2 = 2.0 * float(sigma2) * (1.0 - rho)

    if counts is None:
        counts = np.ones_like(S2, dtype=np.int64)
    else:
        counts = np.asarray(counts)
        if counts.shape != S2.shape:
            raise ValueError(
                "counts shape {} does not match S2 shape {}."
                .format(counts.shape, S2.shape)
            )

    lx, ly, lags_i, S2_x, S2_y, S2_i = extract_basic_profiles(
        S2, max_lag_x, max_lag_y, dx=dx, dy=dy,
        n_bins=n_bins, log_spaced=log_spaced,
    )
    return StructureFunction2D(
        S2=S2, counts=counts, dx=dx, dy=dy,
        lags_x=lx, lags_y=ly, lags_i=lags_i,
        S2_x=S2_x, S2_y=S2_y, S2_i=S2_i,
        x_label=x_label, y_label=y_label,
        symmetrized=True,
    )


# -- 1D AZIMUTHAL SPIRAL MODEL -- #


def S2phi_singlemodel(dphi, A, m=1):
    """Single-mode spiral contribution to ``S_2(dphi)``.

    ``A^2 * (1 - cos(m * dphi))`` with ``dphi`` in [deg].
    """
    return A**2 * (1.0 - np.cos(m * np.radians(dphi)))


def S2phi(dphi, Nphi, A1, A2=None, A3=None):
    """Multi-mode spiral model with optional ``m=2`` and ``m=3`` modes."""
    s2 = S2phi_singlemodel(dphi, A1, m=1)
    if A2 is not None:
        s2 = s2 + S2phi_singlemodel(dphi, A2, m=2)
    if A3 is not None:
        s2 = s2 + S2phi_singlemodel(dphi, A3, m=3)
    return s2 + Nphi


def _make_spiral_model(modes):
    """Build a residual function for ``scipy.optimize.least_squares``
    that sums ``S2phi_singlemodel`` over the requested mode list, with a
    constant ``Nphi`` offset as the first parameter.
    """
    def model(params, dphi):
        Nphi = params[0]
        amplitudes = params[1:]
        out = np.full_like(dphi, Nphi, dtype=float)
        for m, A in zip(modes, amplitudes):
            out = out + S2phi_singlemodel(dphi, A, m=m)
        return out
    return model


# -- ANISOTROPIC GAUSSIAN RANDOM FIELD MODEL -- #
#
# Theoretical S_2 for the non-stationary, anisotropic Gaussian random field
# whose radial correlation length is ell_r(r) = ell0 (r/r0)**alpha and whose
# azimuthal (arc-length) correlation length is ell_phi = A ell_r. The
# covariance is the Paciorek & Schervish (2006) non-stationary kernel, the
# unique smooth, positive-definite extension that reduces exactly to
#   C(dr, dphi) = sigma^2 exp(-dr^2 / 2 ell_r^2 - (r dphi)^2 / 2 ell_phi^2)
# whenever the two points share a radius. A ``pitch`` angle tilts the local
# anisotropy ellipse toward the radial direction (flocculent, random-phase
# spiral arms). These are the forward models fit by
# :meth:`StructureFunction2DStack.fit_GRF`; a coherent grand-design spiral is
# instead a deterministic mean (see :func:`predict_spiral_s2_slices`).


def ell_r(r, ell0r=1.0, alphar=1.0, r0=1.0):
    """Radial correlation length ``ell0r * (r / r0)**alphar`` [arcsec]."""
    return ell0r * (np.asarray(r, dtype=float) / r0) ** alphar


def ell_phi(r, ell0phi=1.0, alphaphi=1.0, r0=1.0):
    """Azimuthal correlation length (arc length) ``ell0phi * (r / r0)**alphaphi``
    [arcsec].

    Parametrised independently of the radial scale: its own normalization
    ``ell0phi = ell_phi(r0)`` and radial exponent ``alphaphi``. The
    (radius-dependent) anisotropy is the derived ratio
    ``A(r) = ell_phi(r) / ell_r(r) = (ell0phi/ell0r) (r/r0)**(alphaphi - alphar)``,
    constant only when ``alphaphi == alphar``. ``ell_phi`` is a physical arc
    length (the azimuthal kernel term is ``(r dphi)^2``).
    """
    return ell0phi * (np.asarray(r, dtype=float) / r0) ** alphaphi


def _resolve_phi(ell0r, alphar, ell0phi, alphaphi):
    """Fill in azimuthal correlation-length params from the radial ones.

    ``ell0phi`` / ``alphaphi`` default to ``ell0r`` / ``alphar`` when ``None``,
    recovering an isotropic, radius-independent-anisotropy field (``A = 1`` at
    all radii). Pass either to decouple the azimuthal amplitude and/or slope.
    """
    if ell0phi is None:
        ell0phi = ell0r
    if alphaphi is None:
        alphaphi = alphar
    return ell0phi, alphaphi


# -- GRF FIT SAMPLING SPEC -- #
#
# The GRF fits sample a subset of (sigma, alphar, ell0r, ell0phi, alphaphi,
# pitch, s) depending on configuration. ``sigma``/``ell0r``/``ell0phi``/``s``
# are sampled in natural log (kept positive); ``alphar``/``alphaphi``/``pitch``
# are linear. ``alphaphi`` is only sampled when ``fit_alphaphi=True`` (otherwise
# tied to ``alphar`` for a radius-independent anisotropy); ``pitch`` only for the
# global pitched fit; ``s`` (jitter) only with ``jitter=True`` and always last.
# A single ordered key list drives bounds/priors/labels/unpacking so they all
# stay aligned regardless of which optional dimensions are active.

_GRF_LOG_SAMPLED = frozenset({"sigma", "ell0r", "ell0phi", "s"})
# Hard default bounds (linear space). Kept deliberately wide so they do not
# silently clamp real disks: the correlation lengths and pitch were the
# parameters most likely to hit a bound, and the ``ell`` entries here are
# only fallbacks. In practice :meth:`fit_GRF` overrides ``ell0r`` / ``ell0phi``
# with data-driven bounds (half the resolved lag range, see
# :func:`_grf_data_bounds`) so they cannot run away past the scales the
# structure function actually constrains, and the slopes / pitch stay open
# enough that a true value rarely pins. Users can still narrow any of these
# via ``bounds=`` per call.
_GRF_BOUND_DEFAULTS = {
    "sigma": (1e-3, 1e3), "alphar": (-5.0, 5.0), "ell0r": (1e-4, 1e2),
    "ell0phi": (1e-4, 1e2), "alphaphi": (-5.0, 5.0), "pitch": (-89.0, 89.0),
    "s": (1e-3, 1e3),
}
_GRF_LABELS = {
    "sigma": (r"$\log\sigma$", r"$\sigma$"),
    "alphar": (r"$\alpha_r$", r"$\alpha_r$"),
    "ell0r": (r"$\log\ell_{0r}$", r"$\ell_{0r}$"),
    "ell0phi": (r"$\log\ell_{0\phi}$", r"$\ell_{0\phi}$"),
    "alphaphi": (r"$\alpha_\phi$", r"$\alpha_\phi$"),
    "pitch": (r"$\mathrm{pitch}$", r"$\mathrm{pitch}$"),
    "s": (r"$\log s$", r"$s$"),
}


def _grf_sample_keys(*, pitch=False, fit_alphaphi=False, jitter=False):
    """Ordered sampled-parameter keys for a GRF fit (see module note above)."""
    keys = ["sigma", "alphar", "ell0r", "ell0phi"]
    if fit_alphaphi:
        keys.append("alphaphi")
    if pitch:
        keys.append("pitch")
    if jitter:
        keys.append("s")
    return keys


def _grf_unpack(theta, keys):
    """Sampling vector -> linear model-parameter dict.

    Exponentiates the log-sampled entries, ties ``alphaphi`` to ``alphar``
    when it is not itself sampled, and defaults ``pitch`` to 0 so every GRF
    residual / log-likelihood closure can call the forward model
    parameterisation-agnostically.
    """
    d = {}
    for k, v in zip(keys, theta):
        d[k] = float(np.exp(v)) if k in _GRF_LOG_SAMPLED else float(v)
    d.setdefault("alphaphi", d["alphar"])
    d.setdefault("pitch", 0.0)
    return d


def _grf_x0(keys, est):
    """Initial sampling vector from a dict of linear-space estimates ``est``.

    Log-sampled keys are stored as ``log(est[k])``; ``s`` seeds at ``0`` (so
    ``s = 1``); a missing linear key (e.g. ``pitch``) seeds at ``0``.
    """
    x0 = []
    for k in keys:
        if k == "s":
            x0.append(0.0)
        elif k in _GRF_LOG_SAMPLED:
            x0.append(np.log(est[k]))
        else:
            x0.append(float(est.get(k, 0.0)))
    return np.array(x0, dtype=float)


def _grf_data_bounds(extents):
    """Data-driven linear-space ``ell`` bounds: ``[1/2 min_lag, 1/2 max_lag]``.

    ``extents`` maps a key (``ell0r`` / ``ell0phi``) to its
    ``(min_positive_lag, max_lag)`` in the parameter's own linear units
    (arcsec for ``ell0r``; arc-length arcsec for ``ell0phi``). The correlation
    length is only constrained between roughly half the smallest resolved lag
    (below it the field is sub-grid) and half the largest (above it the
    structure function has not turned over), so both ends are clamped there.
    Degenerate / non-finite extents are dropped so the wide module fallbacks
    apply instead.
    """
    out = {}
    for k, (lo, hi) in extents.items():
        if np.isfinite(lo) and np.isfinite(hi) and 0.0 < lo < hi:
            out[k] = (0.5 * lo, 0.5 * hi)
    return out


def _grf_bounds_from_keys(keys, bounds, defaults=None):
    """``(lo, hi)`` sampling-space flat-prior / hard bounds aligned to ``keys``.

    Precedence (each overrides the previous): the module-level
    :data:`_GRF_BOUND_DEFAULTS`, the data-driven ``defaults`` (e.g. the
    half-lag-range ``ell`` bounds from :func:`_grf_data_bounds`), then the
    caller's ``bounds`` (``jitter`` accepted as an alias for ``s``).
    """
    b = dict(_GRF_BOUND_DEFAULTS)
    if defaults:
        b.update(defaults)
    if bounds:
        bb = dict(bounds)
        if "jitter" in bb:
            bb["s"] = bb.pop("jitter")
        b.update(bb)
    lo, hi = [], []
    for k in keys:
        x0, x1 = b[k]
        if k in _GRF_LOG_SAMPLED:
            lo.append(np.log(x0))
            hi.append(np.log(x1))
        else:
            lo.append(float(x0))
            hi.append(float(x1))
    return np.array(lo), np.array(hi)


def _grf_prior_arrays_from_keys(keys, priors):
    """Gaussian log-prior ``(mu, sd)`` in sampling space, aligned to ``keys``.

    ``priors`` maps a key to a ``(mu, sigma)`` Gaussian in *sampling* space
    (natural-log for ``sigma``/``ell0r``/``ell0phi``/``s``, linear for
    ``alphar``/``alphaphi``/``pitch``); omitted keys get infinite width (flat
    within :func:`_grf_bounds_from_keys`). ``jitter`` aliases ``s``.
    """
    mu = np.zeros(len(keys))
    sd = np.full(len(keys), np.inf)
    if priors:
        pp = dict(priors)
        if "jitter" in pp:
            pp["s"] = pp.pop("jitter")
        for k, (m, s) in pp.items():
            if k not in keys:
                raise ValueError(
                    "Unknown prior key {!r}; expected {}.".format(k, keys))
            i = keys.index(k)
            mu[i], sd[i] = float(m), float(s)
    return mu, sd


def _grf_perr(keys, mp, cov):
    """Formal 1-sigma errors in linear space from the sampling-space ``cov``.

    Log-sampled params propagate as ``value * sqrt(cov_ii)``; linear params are
    ``sqrt(cov_ii)``. ``mp`` is the unpacked linear-parameter dict.
    """
    perr = {}
    for i, k in enumerate(keys):
        sd = np.sqrt(cov[i, i])
        perr[k] = mp[k] * sd if k in _GRF_LOG_SAMPLED else sd
    return perr


def _grf_labels(keys):
    """``(log_labels, lin_labels)`` for corner/walker plots, aligned to ``keys``."""
    return ([_GRF_LABELS[k][0] for k in keys],
            [_GRF_LABELS[k][1] for k in keys])


def _grf_warn_pinned(keys, x, lo, hi):
    """Emit one warning per parameter whose lsq solution sits on a bound.

    A pinned parameter is reporting "I am at the edge of the allowed range",
    not "I am measured to this value with the reported uncertainty"; the
    Gauss-Newton ``perr`` from the bounded ``trf`` solve does not reflect
    that. Used by :func:`_grf_fit_core` after the LM step.
    """
    for i, k in enumerate(keys):
        on_lo = np.isclose(x[i], lo[i])
        on_hi = np.isclose(x[i], hi[i])
        if not (on_lo or on_hi):
            continue
        side = "lower" if on_lo else "upper"
        bound = float(lo[i] if on_lo else hi[i])
        if k in _GRF_LOG_SAMPLED:
            bound = float(np.exp(bound))
        warnings.warn(
            "{0!r} pinned at the {1} bound ({2:.4g}); the reported perr is "
            "unreliable -- the structure function does not resolve this "
            "parameter, or the true value lies outside the bounds. Pass "
            "bounds={{{0!r}: (lo, hi)}} to widen.".format(k, side, bound),
            stacklevel=3)


def _grf_fit_core(parts, est, data_bounds, *, pitch, fit_alphaphi, method,
                  bounds, priors, jitter, nwalkers, nburnin, nsteps, scatter,
                  plots, returns, pool, progress, plot_bestfit=None,
                  user_p0=None):
    """LM + (optional) MCMC pipeline shared by the GRF fits.

    ``parts(mp)`` is the per-fit residual closure: given a linear
    model-parameter dict (see :func:`_grf_unpack`) it returns
    ``(resid_raw, weights)``: the un-normalized ``model - data`` array and
    the matching weight array, both already concatenated/masked. ``est`` is
    the linear-space initial-guess dict; ``data_bounds`` is the
    half-lag-range bounds overlay for ``ell0r``/``ell0phi`` from
    :func:`_grf_data_bounds`. ``plot_bestfit(mp)`` is called when
    ``'bestfit'`` is in the MCMC plot list. ``user_p0`` is the user's raw
    ``p0`` dict (or ``None``), used only to label which entries warrant a
    ``p0``-clipped warning. All other args mirror the public ``fit_GRF``
    signatures.
    """
    from scipy.optimize import least_squares

    model_keys = _grf_sample_keys(pitch=pitch, fit_alphaphi=fit_alphaphi)
    lo_m, hi_m = _grf_bounds_from_keys(model_keys, bounds, data_bounds)
    x0_raw = _grf_x0(model_keys, est)
    x0 = np.clip(x0_raw, lo_m, hi_m)

    # P1.2: warn only when an *explicit* user p0 entry was clipped onto a
    # bound (heuristic-derived starts may move silently).
    if user_p0:
        for i, k in enumerate(model_keys):
            if k in user_p0 and not np.isclose(x0_raw[i], x0[i]):
                lo_lin = (float(np.exp(lo_m[i])) if k in _GRF_LOG_SAMPLED
                          else float(lo_m[i]))
                hi_lin = (float(np.exp(hi_m[i])) if k in _GRF_LOG_SAMPLED
                          else float(hi_m[i]))
                clipped_lin = (float(np.exp(x0[i])) if k in _GRF_LOG_SAMPLED
                               else float(x0[i]))
                warnings.warn(
                    "p0[{0!r}] = {1!r} is outside the {0!r} bounds "
                    "[{2:.4g}, {3:.4g}]; clipped to {4:.4g}. Pass "
                    "bounds={{{0!r}: (lo, hi)}} to widen.".format(
                        k, user_p0[k], lo_lin, hi_lin, clipped_lin),
                    stacklevel=3)

    def resid(theta):
        r_raw, w = parts(_grf_unpack(theta, model_keys))
        return r_raw / w

    # The optimizer can probe large log-parameters (np.exp -> overflow); those
    # warnings are harmless here, so silence them during the solve.
    with np.errstate(over="ignore", invalid="ignore"):
        sol = least_squares(resid, x0, bounds=(lo_m, hi_m), method="trf")

    # P1.3: warn per pinned parameter -- the Gauss-Newton perr is unreliable
    # when the optimizer stopped at a bound (a true unresolved / non-detected
    # scale rather than a measurement with the reported uncertainty). Applies
    # to both backends since the MCMC walker ball is seeded from this lsq.
    _grf_warn_pinned(model_keys, sol.x, lo_m, hi_m)

    if method == "lsq":
        mp = _grf_unpack(sol.x, model_keys)
        J = sol.jac                                   # Gauss-Newton covariance
        dof = max(J.shape[0] - J.shape[1], 1)
        cov = np.linalg.inv(J.T @ J) * (2.0 * sol.cost / dof)
        params = {k: mp[k] for k in model_keys}
        perr = _grf_perr(model_keys, mp, cov)
        if "alphaphi" not in params:                  # tied: report for clarity
            params["alphaphi"] = mp["alphaphi"]
        return params, perr, sol, cov

    if method != "mcmc":
        raise ValueError(
            "method must be 'lsq' or 'mcmc', got {!r}.".format(method))

    import emcee
    from .helper_functions import random_p0, plot_walkers, plot_corner

    keys = _grf_sample_keys(pitch=pitch, fit_alphaphi=fit_alphaphi,
                            jitter=jitter)
    lo, hi = _grf_bounds_from_keys(keys, bounds, data_bounds)
    pmu, psd = _grf_prior_arrays_from_keys(keys, priors)
    # P1.4: nudge the lsq solution strictly inside the box before the
    # walker ball is scattered around it. Without this, a pinned parameter
    # has ~50 % of walkers start at -inf prior (``random_p0`` scatters
    # symmetrically). The nudge is a small fraction of the box width so it
    # barely shifts well-resolved solutions.
    box_eps = 1e-3 * (hi_m - lo_m)
    theta0_inside = np.clip(sol.x, lo_m + box_eps, hi_m - box_eps)
    theta0 = list(theta0_inside) + ([0.0] if jitter else [])
    ndim = len(theta0)
    log_labels, lin_labels = _grf_labels(keys)
    print("Assuming:\n\tp0 = [%s]."
          % (", ".join(la.replace("$", "") for la in lin_labels)))

    def ln_prob(theta):
        if np.any(theta < lo) or np.any(theta > hi):
            return -np.inf
        mp = _grf_unpack(theta, keys)
        r_raw, w = parts(mp)
        if not np.all(np.isfinite(r_raw)):
            return -np.inf
        var = (mp["s"] ** 2 if jitter else 1.0) * w ** 2
        ln_like = -0.5 * np.sum(r_raw ** 2 / var
                                + np.log(2.0 * np.pi * var))
        ln_prior = -0.5 * np.sum(((theta - pmu) / psd) ** 2)
        return ln_like + ln_prior

    nwalkers = max(int(nwalkers), 2 * ndim)
    p0_ball = random_p0(theta0, scatter, nwalkers)
    with np.errstate(over="ignore", invalid="ignore"):
        sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_prob, pool=pool)
        sampler.run_mcmc(p0_ball, int(nburnin) + int(nsteps),
                         progress=progress)

    # Posterior samples in linear space: exponentiate the log-sampled cols.
    chain = sampler.get_chain(discard=int(nburnin), flat=True)
    cols = [np.exp(chain[:, n]) if k in _GRF_LOG_SAMPLED else chain[:, n]
            for n, k in enumerate(keys)]
    samples = np.column_stack(cols)

    # Diagnostic plots (mirrors rotationmap.fit_map).
    if plots is None:
        plots = ["walkers", "corner"]
    plots = np.atleast_1d(plots)
    if "none" in plots:
        plots = []
    if "walkers" in plots:
        full_chain = sampler.get_chain()          # (nsteps, nwalkers, ndim)
        walkers = np.stack(
            [np.exp(full_chain[:, :, n]) if k in _GRF_LOG_SAMPLED
             else full_chain[:, :, n]
             for n, k in enumerate(keys)],
            axis=0,
        )                                          # (ndim, nsteps, nwalkers)
        plot_walkers(walkers, int(nburnin), lin_labels)
    if "corner" in plots:
        plot_corner(samples, lin_labels)
    if "bestfit" in plots and plot_bestfit is not None:
        plot_bestfit(_grf_unpack(np.median(chain, axis=0), keys))

    # Output (mirrors rotationmap.fit_map's `returns` mechanism).
    if returns is None:
        returns = ["samples"]
    returns = np.atleast_1d(returns)
    if "none" in returns:
        return None
    pct = np.percentile(samples, [16, 50, 84], axis=0)
    medians = {k: pct[1, n] for n, k in enumerate(keys)}
    to_return = []
    if "samples" in returns:
        to_return += [samples]
    if "sampler" in returns:
        to_return += [sampler]
    if "lnprob" in returns:
        to_return += [sampler.get_log_prob(discard=int(nburnin), flat=True)]
    if "percentiles" in returns:
        to_return += [pct]
    if "dict" in returns:
        to_return += [medians]
    return to_return if len(to_return) > 1 else to_return[0]


def _sigma_components(lr, lphi, pitch):
    """Components of the local anisotropy (kernel covariance) matrix Sigma.

    ``Sigma = R(p) diag(2 ell_perp^2, 2 ell_par^2) R(p)^T`` in the local
    ``(dr, r dphi)`` tangent frame, with the long axis ``ell_par = ell_phi``
    along the arm (pitch ``p`` from the azimuthal direction) and the short
    axis ``ell_perp = ell_r`` across it. ``lr`` and ``lphi`` are the radial and
    azimuthal correlation lengths *at this point*, supplied independently (their
    ratio is the local anisotropy). Returns the three independent entries
    ``(s11, s22, s12)``; ``pitch=0`` gives ``s12=0`` and the diagonal
    ``(2 ell_r^2, 2 ell_phi^2)``.
    """
    cp, sp = np.cos(pitch), np.sin(pitch)
    lperp2 = lr ** 2
    lpar2 = lphi ** 2
    s11 = 2.0 * (lperp2 * cp ** 2 + lpar2 * sp ** 2)
    s22 = 2.0 * (lperp2 * sp ** 2 + lpar2 * cp ** 2)
    s12 = 2.0 * sp * cp * (lpar2 - lperp2)
    return s11, s22, s12


def _ps_cov(r_a, phi_a, r_b, phi_b, *, alphar, ell0r, alphaphi, ell0phi,
            sigma, r0, pitch=0.0):
    """Paciorek-Schervish covariance between point sets ``a`` and ``b``.

    Uses the full 2x2 anisotropy form so the correlation ellipse can be tilted
    to a pitch angle (``pitch`` in radians). The radial and azimuthal
    correlation lengths are evaluated per point from their independent power
    laws. All four coordinate inputs broadcast against each other. ``pitch=0``
    recovers the diagonal kernel.
    """
    lr_a = ell_r(r_a, ell0r, alphar, r0)
    lr_b = ell_r(r_b, ell0r, alphar, r0)
    lp_a = ell_phi(r_a, ell0phi, alphaphi, r0)
    lp_b = ell_phi(r_b, ell0phi, alphaphi, r0)
    s11a, s22a, s12a = _sigma_components(lr_a, lp_a, pitch)
    s11b, s22b, s12b = _sigma_components(lr_b, lp_b, pitch)

    # Averaged anisotropy matrix Sigma_bar = (Sigma_a + Sigma_b) / 2.
    a = 0.5 * (s11a + s11b)
    d = 0.5 * (s22a + s22b)
    b = 0.5 * (s12a + s12b)
    det = a * d - b ** 2

    # Displacement (dr, rbar dphi) in the shared tangent frame.
    dphi = phi_a - phi_b
    dphi = (dphi + np.pi) % (2.0 * np.pi) - np.pi  # wrap to [-pi, pi]
    D1 = r_a - r_b
    D2 = 0.5 * (r_a + r_b) * dphi

    # Quadratic form D^T Sigma_bar^{-1} D and the det prefactor.
    Q = (d * D1 ** 2 - 2.0 * b * D1 * D2 + a * D2 ** 2) / det
    det_a = s11a * s22a - s12a ** 2
    det_b = s11b * s22b - s12b ** 2
    pref = det_a ** 0.25 * det_b ** 0.25 / np.sqrt(det)
    return sigma ** 2 * pref * np.exp(-Q)


def _spiral_pitch_wavenumber(m, pitch_angle):
    """Radial phase wavenumber ``p = m / tan(pitch_angle)`` (phase per d ln r).

    ``pitch_angle`` [deg] is the angle between an arm's tangent and the local
    azimuthal direction. ``90 deg`` -> ``p = 0`` (radial spokes, no winding);
    small angles -> tightly wound. ``0`` is rejected (an unwound circle).
    """
    if m < 1:
        raise ValueError("m must be a positive integer (>= 1).")
    if pitch_angle == 0.0:
        raise ValueError(
            "pitch_angle must be nonzero (0 deg is a closed ring, p -> inf); "
            "use 90 deg for the no-winding limit.")
    return m / np.tan(np.radians(pitch_angle))


def _spiral_envelope(envelope, r):
    """Evaluate a radial amplitude envelope ``B(r) / amplitude`` on ``r``.

    ``None`` -> constant 1. Otherwise a callable ``r -> factor``.
    """
    r = np.asarray(r, dtype=float)
    if envelope is None:
        return np.ones_like(r)
    if not callable(envelope):
        raise TypeError(
            "envelope must be a callable r -> amplitude factor, or None.")
    return np.broadcast_to(np.asarray(envelope(r), dtype=float), r.shape)


def _spiral_s2(r_a, r_b, dphi, *, m, p, amplitude, r0, envelope):
    """Azimuthally-averaged S_2 of the deterministic spiral between two radii.

    For ``mu = B(r) cos(m phi - p ln(r/r0) - const)``, averaging the squared
    increment over base azimuth gives, exactly,
    ``(B_a^2 + B_b^2)/2 - B_a B_b cos(m dphi - p ln(r_b/r_a))``, independent of
    the pattern phase. On the azimuthal slice (``r_a = r_b``) this reduces to
    ``B^2 (1 - cos m dphi)`` (oscillating, non-saturating).
    """
    Ba = amplitude * _spiral_envelope(envelope, r_a)
    Bb = amplitude * _spiral_envelope(envelope, r_b)
    dpsi = p * np.log(r_b / r_a)
    return 0.5 * (Ba ** 2 + Bb ** 2) - Ba * Bb * np.cos(m * dphi - dpsi)


def predict_spiral_s2_slices(ref_r, lags_r, lags_phi_deg, *, m, pitch_angle,
                             amplitude=1.0, r0=1.0, envelope=None):
    """Deterministic grand-design contribution to the ``S_2`` slices.

    The expected ``S_2`` of a coherent logarithmic-spiral mean field alone,
    azimuthally averaged, on the same lag axes as :func:`predict_s2_slices`.
    Add to the GRF prediction (or pass via its ``spiral=`` kwarg) for the full
    expected field-plus-spiral ``S_2``.

    Args:
        ref_r (float): Reference annulus radius [arcsec].
        lags_r (ndarray): Radial lags [arcsec], >= 0 (outward).
        lags_phi_deg (ndarray): Azimuthal lags [deg].
        m (int): Azimuthal wavenumber / number of arms (>= 1).
        pitch_angle (float): Pitch angle [deg]; the arm tangent's lean from the
            azimuthal direction (``90`` is no winding). Must be nonzero.
        amplitude (float): Peak amplitude ``B``.
        r0 (float): Reference radius [arcsec] for the radial phase.
        envelope (Optional[callable]): ``r -> amplitude factor`` radial taper;
            ``None`` (default) is constant.

    Returns:
        dict with keys ``S2_r`` (radial slice, ``dphi=0``; oscillates with
        ``ln r`` and does *not* saturate), ``S2_phi`` (azimuthal slice,
        ``dr=0``, equal to ``B(ref_r)^2 (1 - cos m dphi)``), ``B_ref`` (the
        amplitude at ``ref_r``) and ``p`` (the radial phase wavenumber).
    """
    lags_r = np.asarray(lags_r, dtype=float)
    lags_phi_deg = np.asarray(lags_phi_deg, dtype=float)
    m = int(m)
    p = _spiral_pitch_wavenumber(m, pitch_angle)
    kw = dict(m=m, p=p, amplitude=amplitude, r0=r0, envelope=envelope)

    S2_r = _spiral_s2(ref_r, ref_r + lags_r, 0.0, **kw)
    S2_phi = _spiral_s2(ref_r, ref_r, np.radians(lags_phi_deg), **kw)

    return {
        "S2_r": S2_r,
        "S2_phi": S2_phi,
        "B_ref": float(amplitude * _spiral_envelope(envelope, ref_r)),
        "p": float(p),
    }


def predict_spiral_s2_2d(ref_r, lags_r_full, lags_phi_full_deg, *, m,
                         pitch_angle, amplitude=1.0, r0=1.0, envelope=None):
    """Deterministic grand-design 2D ``S_2`` surface, on a two-sided lag grid.

    The expected ``S_2`` of a coherent spiral mean field alone, on the same
    axes as :func:`predict_s2_2d`. Its trough runs along
    ``m dphi = p ln(r_b / r_a)``: the coherent winding ridge. Pairs whose
    partner radius ``ref_r + l_r <= 0`` are set to NaN.

    Args:
        ref_r (float): Reference annulus radius [arcsec].
        lags_r_full (ndarray): Two-sided radial lags [arcsec].
        lags_phi_full_deg (ndarray): Two-sided azimuthal lags [deg].
        m, pitch_angle, amplitude, r0, envelope: Spiral parameters (see
            :func:`predict_spiral_s2_slices`).

    Returns:
        ndarray: 2D ``S_2``, shape
        ``(lags_r_full.size, lags_phi_full_deg.size)``.
    """
    lags_r_full = np.asarray(lags_r_full, dtype=float)
    lags_phi_full_deg = np.asarray(lags_phi_full_deg, dtype=float)
    m = int(m)
    p = _spiral_pitch_wavenumber(m, pitch_angle)

    LR, LP = np.meshgrid(lags_r_full, np.radians(lags_phi_full_deg),
                         indexing="ij")
    r_b = ref_r + LR
    # No pairs exist past the disk center; mask so the log/envelope stay finite.
    r_b_safe = np.where(r_b > 0.0, r_b, np.nan)
    return _spiral_s2(ref_r, r_b_safe, LP, m=m, p=p, amplitude=amplitude,
                      r0=r0, envelope=envelope)


def predict_s2_slices(ref_r, lags_r, lags_phi_deg, *, alphar=1.0, ell0r=1.0,
                      alphaphi=None, ell0phi=None, sigma=1.0, r0=1.0,
                      pitch=0.0, spiral=None):
    """Expected ``S_2`` radial and azimuthal slices at a reference annulus.

    For a Gaussian field, ``S_2(a, b) = C(a, a) + C(b, b) - 2 C(a, b)
    = 2 sigma^2 - 2 C(a, b)``, evaluated with the same Paciorek-Schervish
    covariance the field is drawn from, so the prediction is *exact* for the
    on-axis slices, including the radial slice, whose pairs straddle two
    radii with different correlation lengths. The lag axes match
    :class:`StructureFunction2D`: ``lags_r`` in arcsec (its ``lags_x`` /
    ``S2_x``) and ``lags_phi_deg`` in degrees (its ``lags_y`` / ``S2_y``).

    With ``pitch != 0`` the correlation ellipse is tilted; the on-axis slices
    still match in shape, but the clearest tilt signature is the diagonal ridge
    in the full 2D surface; use :func:`predict_s2_2d`.

    This is the forward model fit by
    :meth:`StructureFunction2DStack.fit_GRF` (which fits ``pitch=0``,
    ``spiral=None``); :func:`grf_s2_slices` is the convenience entry point for
    that common case.

    Args:
        ref_r (float): Reference annulus radius [arcsec].
        lags_r (ndarray): Radial lags [arcsec], >= 0 (outward).
        lags_phi_deg (ndarray): Azimuthal lags [deg], >= 0.
        alphar (float): Radial scaling exponent of ``ell_r``.
        ell0r (float): Radial correlation length at ``r0`` [arcsec].
        alphaphi (Optional[float]): Azimuthal scaling exponent of ``ell_phi``.
            Defaults to ``alphar`` (radius-independent anisotropy).
        ell0phi (Optional[float]): Azimuthal (arc-length) correlation length at
            ``r0`` [arcsec]. Defaults to ``ell0r`` (isotropic). The anisotropy
            ``ell_phi/ell_r`` is then a derived, possibly radius-dependent ratio.
        sigma (float): Per-point standard deviation (plateau is ``2 sigma^2``).
        r0 (float): Reference radius for both correlation-length power laws.
        pitch (float): Pitch angle [deg] of the correlation ellipse.
        spiral (Optional[dict]): If given, the deterministic grand-design
            contribution from :func:`predict_spiral_s2_slices` is added to both
            slices (the field and a deterministic mean add in expectation).
            Keys are that function's keyword args (``m``, ``pitch_angle``,
            ``amplitude``, ``r0``, ``envelope``).

    Returns:
        dict with keys ``S2_r`` (radial slice aligned with ``lags_r``),
        ``S2_phi`` (azimuthal slice aligned with ``lags_phi_deg``), ``ell_r``
        and ``ell_phi`` at ``ref_r`` [arcsec], and ``plateau`` (``2 sigma^2``,
        the GRF large-lag asymptote; the spiral part does not saturate).
    """
    ell0phi, alphaphi = _resolve_phi(ell0r, alphar, ell0phi, alphaphi)
    lags_r = np.asarray(lags_r, dtype=float)
    lags_phi_deg = np.asarray(lags_phi_deg, dtype=float)
    params = dict(alphar=alphar, ell0r=ell0r, alphaphi=alphaphi,
                  ell0phi=ell0phi, sigma=sigma, r0=r0,
                  pitch=np.radians(pitch))

    # Radial slice: pairs (ref_r, 0) and (ref_r + lag, 0) at different radii.
    cov_r = _ps_cov(ref_r, 0.0, ref_r + lags_r, 0.0, **params)
    S2_r = 2.0 * sigma ** 2 - 2.0 * cov_r

    # Azimuthal slice: pairs (ref_r, 0) and (ref_r, dphi) at the same radius.
    dphi_rad = np.radians(lags_phi_deg)
    cov_phi = _ps_cov(ref_r, 0.0, ref_r, dphi_rad, **params)
    S2_phi = 2.0 * sigma ** 2 - 2.0 * cov_phi

    if spiral is not None:
        sp = predict_spiral_s2_slices(ref_r, lags_r, lags_phi_deg, **spiral)
        S2_r = S2_r + sp["S2_r"]
        S2_phi = S2_phi + sp["S2_phi"]

    return {
        "S2_r": S2_r,
        "S2_phi": S2_phi,
        "ell_r": float(ell_r(ref_r, ell0r, alphar, r0)),
        "ell_phi": float(ell_phi(ref_r, ell0phi, alphaphi, r0)),
        "plateau": 2.0 * sigma ** 2,
    }


def predict_s2_2d(ref_r, lags_r_full, lags_phi_full_deg, *, alphar=1.0,
                  ell0r=1.0, alphaphi=None, ell0phi=None, sigma=1.0, r0=1.0,
                  pitch=0.0, spiral=None):
    """Expected 2D ``S_2`` surface at a reference annulus, on a two-sided grid.

    ``S_2 = 2 sigma^2 - 2 C`` between ``(ref_r, 0)`` and
    ``(ref_r + l_r, l_phi)`` using the same Paciorek-Schervish covariance the
    field is drawn from. With ``pitch != 0`` the trough (max correlation) runs
    along a diagonal in the ``(l_r, l_phi)`` plane.

    Pass two-sided lag axes matching
    :attr:`StructureFunction2D.S2`: ``lags_r_full`` = ``S2.lags_x_full``
    [arcsec] and ``lags_phi_full_deg`` =
    ``np.arange(-S2.max_lag_y, S2.max_lag_y + 1) * S2.dy`` [deg].

    Args:
        ref_r (float): Reference annulus radius [arcsec].
        lags_r_full (ndarray): Two-sided radial lags [arcsec].
        lags_phi_full_deg (ndarray): Two-sided azimuthal lags [deg].
        alphar, ell0r, alphaphi, ell0phi, sigma, r0, pitch (float): Field
            parameters. ``alphaphi`` / ``ell0phi`` default to ``alphar`` /
            ``ell0r`` (isotropic).
        spiral (Optional[dict]): If given, adds the deterministic grand-design
            surface from :func:`predict_spiral_s2_2d`. Keys are that function's
            keyword args.

    Returns:
        ndarray: 2D ``S_2``, shape
        ``(lags_r_full.size, lags_phi_full_deg.size)``.
    """
    ell0phi, alphaphi = _resolve_phi(ell0r, alphar, ell0phi, alphaphi)
    lags_r_full = np.asarray(lags_r_full, dtype=float)
    lags_phi_full_deg = np.asarray(lags_phi_full_deg, dtype=float)
    LR, LP = np.meshgrid(lags_r_full, np.radians(lags_phi_full_deg),
                         indexing="ij")
    cov = _ps_cov(ref_r, 0.0, ref_r + LR, LP, alphar=alphar, ell0r=ell0r,
                  alphaphi=alphaphi, ell0phi=ell0phi, sigma=sigma,
                  r0=r0, pitch=np.radians(pitch))
    S2 = 2.0 * sigma ** 2 - 2.0 * cov

    if spiral is not None:
        S2 = S2 + predict_spiral_s2_2d(ref_r, lags_r_full, lags_phi_full_deg,
                                       **spiral)
    return S2


def grf_s2_slices(ref_r, lags_r, lags_phi_deg, *, sigma=1.0, alphar=1.0,
                  ell0r=1.0, alphaphi=None, ell0phi=None, r0=1.0):
    """Expected ``S_2`` slices for the axis-aligned (zero-pitch) GRF.

    Convenience wrapper around :func:`predict_s2_slices` for the common case
    fit by :meth:`StructureFunction2DStack.fit_GRF`: an anisotropic Gaussian
    random field with radial correlation length
    ``ell_r(r) = ell0r (r/r0)**alphar`` and azimuthal (arc-length) length
    ``ell_phi(r) = ell0phi (r/r0)**alphaphi``, no pitch and no deterministic
    spiral. See :func:`predict_s2_slices` for the general form.

    Args:
        ref_r (float): Reference annulus radius [arcsec].
        lags_r (ndarray): Radial lags [arcsec], >= 0 (outward).
        lags_phi_deg (ndarray): Azimuthal lags [deg], >= 0.
        sigma (float): Per-point standard deviation (plateau is ``2 sigma^2``).
        alphar (float): Radial scaling exponent of ``ell_r``.
        ell0r (float): Radial correlation length at ``r0`` [arcsec].
        alphaphi (Optional[float]): Azimuthal scaling exponent; defaults to
            ``alphar`` (radius-independent anisotropy).
        ell0phi (Optional[float]): Azimuthal (arc-length) correlation length at
            ``r0`` [arcsec]; defaults to ``ell0r`` (isotropic).
        r0 (float): Reference radius for both correlation-length power laws.

    Returns:
        dict with keys ``S2_r``, ``S2_phi``, ``ell_r``, ``ell_phi``,
        ``plateau`` (see :func:`predict_s2_slices`).
    """
    return predict_s2_slices(ref_r, lags_r, lags_phi_deg, alphar=alphar,
                             ell0r=ell0r, alphaphi=alphaphi, ell0phi=ell0phi,
                             sigma=sigma, r0=r0, pitch=0.0, spiral=None)


if _HAS_NUMBA:

    @njit(parallel=True, cache=True)
    def _grf_s2_2d_global_kernel(r_axis, di, lp, alphar, ell0r, alphaphi,
                                 ell0phi, sigma, r0, pitch):
        """Numba kernel for :func:`grf_s2_2d_global`.

        All inputs are plain arrays / scalars (no kwargs, no None defaults).
        ``di`` is the integer lag-index array (``round(lags_x / dr)``);
        ``lp`` is the azimuthal lag array in radians.  The GRF physics is
        fully inlined so LLVM can fuse the inner arithmetic.
        """
        n = r_axis.shape[0]
        n_lags = di.shape[0]
        n_phi = lp.shape[0]
        sigma2 = sigma * sigma
        out = np.full((n_lags, n_phi), np.nan)

        cp = np.cos(pitch)
        sp = np.sin(pitch)

        for k in prange(n_lags):                        # parallel over lag bins
            d = di[k]
            i_lo = max(0, -d)
            i_hi = min(n, n - d)
            n_valid = i_hi - i_lo
            if n_valid <= 0:
                continue

            acc = np.zeros(n_phi)

            for ii in range(i_lo, i_hi):                # base-row accumulation
                ra = r_axis[ii]
                rb = r_axis[ii + d]

                # Inline ell_r / ell_phi
                lr_a = ell0r * (ra / r0) ** alphar
                lr_b = ell0r * (rb / r0) ** alphar
                lpa  = ell0phi * (ra / r0) ** alphaphi
                lpb  = ell0phi * (rb / r0) ** alphaphi

                # Inline _sigma_components for point a and b
                lperp2_a = lr_a * lr_a
                lpar2_a  = lpa * lpa
                s11a = 2.0 * (lperp2_a * cp * cp + lpar2_a * sp * sp)
                s22a = 2.0 * (lperp2_a * sp * sp + lpar2_a * cp * cp)
                s12a = 2.0 * sp * cp * (lpar2_a - lperp2_a)

                lperp2_b = lr_b * lr_b
                lpar2_b  = lpb * lpb
                s11b = 2.0 * (lperp2_b * cp * cp + lpar2_b * sp * sp)
                s22b = 2.0 * (lperp2_b * sp * sp + lpar2_b * cp * cp)
                s12b = 2.0 * sp * cp * (lpar2_b - lperp2_b)

                # Averaged anisotropy matrix Sigma_bar and its determinant
                a_m = 0.5 * (s11a + s11b)
                d_m = 0.5 * (s22a + s22b)
                b_m = 0.5 * (s12a + s12b)
                det = a_m * d_m - b_m * b_m

                # Phi-independent displacement and det prefactor
                D1    = ra - rb
                r_bar = 0.5 * (ra + rb)
                det_a = s11a * s22a - s12a * s12a
                det_b = s11b * s22b - s12b * s12b
                pref  = (det_a ** 0.25) * (det_b ** 0.25) / np.sqrt(det)

                for j in range(n_phi):                  # azimuthal lag loop
                    dphi = -lp[j]                       # phi_a=0 - phi_b=lp[j]
                    dphi = (dphi + np.pi) % (2.0 * np.pi) - np.pi
                    D2 = r_bar * dphi
                    Q  = (d_m * D1 * D1 - 2.0 * b_m * D1 * D2
                          + a_m * D2 * D2) / det
                    acc[j] += pref * np.exp(-Q)

            inv_n = 1.0 / n_valid
            for j in range(n_phi):
                out[k, j] = 2.0 * sigma2 * (1.0 - acc[j] * inv_n)

        return out


def grf_s2_2d_global(r_axis, lags_x, lags_y_deg, *, sigma=1.0, alphar=1.0,
                     ell0r=1.0, alphaphi=None, ell0phi=None, r0=1.0, pitch=0.0):
    """Expected GLOBAL-mode (``ref_i=-1``) 2D ``S_2`` surface of the pitched GRF.

    The global estimator averages the pair statistic over *every* base radius,
    so the expected surface is the (equal-weight-per-base-row) mean over base
    radius ``r_i`` of the single-pair surface ``2 sigma^2 - 2 C(r_i -> r_i +
    l_r, l_phi)``, with ``C`` the Paciorek-Schervish covariance
    (:func:`_ps_cov`, the same kernel :func:`predict_s2_2d` uses).

    This is the surface to fit a pitch against. The per-annulus
    (reference-mode) surface that :class:`StructureFunction2DStack` builds
    mirror-fills the azimuthal lag and is therefore *exactly* symmetric in
    ``l_phi``, which averages the antisymmetric pitch ridge, and with it the
    pitch sign, away. The global surface is only point-symmetric
    (``(l_r, l_phi) -> (-l_r, -l_phi)``) and preserves the ridge. See
    :meth:`StructureFunction2D.fit_GRF` with ``pitch=True``.

    Equal weight per valid base row matches the global kernel's pair counts on
    a rectangular polar grid: lag bin ``(di, dj)`` accumulates
    ``(N - |di|) * (M - |dj|)`` pairs, uniform across the ``N - |di|`` valid
    base rows, and the ``(M - |dj|)`` azimuthal factor cancels in the per-bin
    average.

    Args:
        r_axis (ndarray): Absolute radii [arcsec] of axis 0 of the field this
            ``S_2`` was computed from (ascending, uniform spacing). Sets which
            base radii enter the average and the radial scaling of ``ell_r``.
        lags_x (ndarray): Two-sided radial lags [arcsec], i.e.
            :attr:`StructureFunction2D.lags_x_full`.
        lags_y_deg (ndarray): Two-sided azimuthal lags [deg].
        sigma (float): Per-point standard deviation (plateau is ``2 sigma^2``).
        alphar (float): Radial scaling exponent of ``ell_r``.
        ell0r (float): ``ell_r(r0) = ell0r`` [arcsec].
        alphaphi (Optional[float]): Azimuthal scaling exponent; defaults to
            ``alphar``.
        ell0phi (Optional[float]): Azimuthal (arc-length) length at ``r0``;
            defaults to ``ell0r`` (isotropic).
        r0 (float): Reference radius for both correlation-length power laws.
        pitch (float): Pitch angle [deg] of the correlation ellipse's long
            axis from the azimuthal direction (sign sets the winding sense).

    Returns:
        ndarray: Expected ``S_2``, shape ``(lags_x.size, lags_y_deg.size)``.
            Radial lags with no valid base-row pairs are ``NaN``.
    """
    ell0phi, alphaphi = _resolve_phi(ell0r, alphar, ell0phi, alphaphi)
    r_axis = np.asarray(r_axis, dtype=float)
    if r_axis.ndim != 1 or r_axis.size < 2:
        raise ValueError("r_axis must be a 1D radial grid with >= 2 points.")
    dr = float(np.mean(np.diff(r_axis)))
    n = r_axis.size
    di = np.round(np.asarray(lags_x, dtype=float) / dr).astype(int)
    lp = np.radians(np.asarray(lags_y_deg, dtype=float))
    pr = np.radians(pitch)

    if _HAS_NUMBA:
        return _grf_s2_2d_global_kernel(
            r_axis, di, lp,
            float(alphar), float(ell0r), float(alphaphi), float(ell0phi),
            float(sigma), float(r0), float(pr),
        )

    out = np.full((di.size, lp.size), np.nan)
    for k, d in enumerate(di):
        i = np.arange(max(0, -d), min(n, n - d))      # valid base rows
        if i.size == 0:
            continue
        ra = r_axis[i][:, None]
        rb = r_axis[i + d][:, None]
        cov = _ps_cov(ra, 0.0, rb, lp[None, :], alphar=alphar, ell0r=ell0r,
                      alphaphi=alphaphi, ell0phi=ell0phi, sigma=sigma,
                      r0=r0, pitch=pr)                   # (n_valid, n_phi)
        out[k] = 2.0 * sigma ** 2 - 2.0 * cov.mean(axis=0)
    return out


# -- FACTORY FUNCTIONS -- #


def structure_function_ensemble(fields, *, mode="global", dx=1.0, dy=1.0,
                                ref_rs=None, x_axis=None, ref_band=0.0,
                                max_lag_x=None, max_lag_y=None,
                                n_bins=50, log_spaced=False, symmetrize=True,
                                azimuthal_axis="y", x_label="lag_x",
                                y_label="lag_y", y_grid=None):
    """Build a per-realization ensemble from a 3D stack of fields.

    Given ``fields`` of shape ``(N, n_x, n_y)``, returns a list of N results,
    one per field, WITHOUT averaging across realizations. Contrast
    :meth:`StructureFunction2D.combine` (and passing a 3D array to a helper
    that combines), which POOLS the realizations into a single result; this
    keeps them separate so you can take the per-cell / per-statistic scatter
    across the realization axis (np.std / np.percentile).

    Args:
        fields (ndarray): 3D array ``(N, n_x, n_y)`` (axis 1 = radius,
            axis 2 = azimuth).
        mode ({'global', 'stack'}):
            ``'global'`` (default): one :class:`StructureFunction2D` per
                field over the whole field (``ref_i = -1``): N global S_2.
            ``'stack'``: one :class:`StructureFunction2DStack` per field at
                ``ref_rs`` (requires ``ref_rs``): N radius-resolved stacks.
        dx, dy, ref_band, max_lag_x, max_lag_y, n_bins, log_spaced,
            symmetrize, azimuthal_axis, x_label, y_label, y_grid: forwarded to
            the per-field constructor (``max_lag_*`` in pixels).
        ref_rs (sequence of float): Reference radii, required for
            ``mode='stack'``.
        x_axis (Optional[ndarray]): Axis-1 (radial) coordinate for
            ``mode='stack'`` ref-radius matching; defaults to
            ``np.arange(n_x) * dx``.

    Returns:
        list: N :class:`StructureFunction2D` (``mode='global'``) or N
        :class:`StructureFunction2DStack` (``mode='stack'``).
    """
    fields = np.asarray(fields)
    if fields.ndim != 3:
        raise ValueError("fields must be 3D (N, n_x, n_y); got shape {}."
                         .format(fields.shape))

    if mode == "global":
        return [StructureFunction2D.from_array(
                    f, dx=dx, dy=dy, max_lag_x=max_lag_x, max_lag_y=max_lag_y,
                    ref_i=-1, n_bins=n_bins, log_spaced=log_spaced,
                    symmetrize=symmetrize, azimuthal_axis=azimuthal_axis,
                    x_label=x_label, y_label=y_label)
                for f in fields]

    if mode == "stack":
        if ref_rs is None:
            raise ValueError("mode='stack' requires ref_rs.")
        return [StructureFunction2DStack.from_array(
                    f, ref_rs, x_axis=x_axis, dx=dx, dy=dy, ref_band=ref_band,
                    max_lag_x=max_lag_x, max_lag_y=max_lag_y, n_bins=n_bins,
                    log_spaced=log_spaced, symmetrize=symmetrize,
                    azimuthal_axis=azimuthal_axis, x_label=x_label,
                    y_label=y_label, y_grid=y_grid)
                for f in fields]

    raise ValueError("mode must be 'global' or 'stack', got {!r}.".format(mode))


# -- RESULT CONTAINER -- #


class StructureFunction2D:
    """Container for a 2D second-order structure function plus its
    derived 1D profiles.

    Built by :meth:`eddy.momentmap.momentmap.compute_structure_function`
    or by :meth:`from_array` when working from a bare numpy array.

    Attributes:
        S2 (ndarray): 2D structure function, shape
            ``(2*max_lag_x+1, 2*max_lag_y+1)``.
        counts (ndarray): Pair counts, same shape as ``S2``.
        dx, dy (float): Physical grid spacing along axis 0 and 1.
        lags_x, lags_y (ndarray): Positive lags along each axis.
        lags_i (ndarray): Radial bin centers for the azimuthal average.
        S2_x, S2_y (ndarray): 1D profiles along axis 0 and axis 1.
        S2_i (Optional[ndarray]): Azimuthally averaged profile. ``None``
            when ``dx`` and ``dy`` do not share units (e.g. the momentmap
            polar pipeline, where ``dx`` is arcsec and ``dy`` is degrees),
            since the underlying circular bins would mix units. See
            :func:`extract_basic_profiles` for the construction.
        x_grid, y_grid (Optional[ndarray]): Underlying grid axes of the
            field (e.g. radial / azimuthal grid for a polar
            deprojection). ``None`` when constructed from a bare array.
        gridded (Optional[ndarray]): The 2D field that ``S_2`` was
            computed from.
        ref, ref_band: Reference-annulus center / band in physical units
            (e.g. arcsec). ``None`` if global.
        x_label, y_label (str): Labels for the two lag axes; used by the
            plotting helpers.
        azimuthal_axis (Optional[str]): ``'y'`` if axis 1 corresponds
            to an angular coordinate (e.g. azimuth in degrees), in which
            case :meth:`fit_spiral` defaults to that axis. ``None``
            otherwise.
        combined_error (Optional[ndarray]): Per-bin standard error set by
            :meth:`combine`; ``None`` on single-realization results.
        combined_std (Optional[ndarray]): Per-bin intrinsic scatter set by
            :meth:`combine`; ``None`` on single-realization results.
        noise_mask (Optional[dict]): Annulus mask parameters set by
            :meth:`~eddy.linecube.linecube.noise_structure_function` when
            the default channel selection is used. ``None`` otherwise.
            Keys: ``N``, ``r_in``, ``r_out``.
    """

    def __init__(self, *, S2, counts, dx, dy, lags_x, lags_y, lags_i,
                 S2_x, S2_y, S2_i, x_grid=None, y_grid=None,
                 gridded=None, ref=None, ref_band=None,
                 x_label="lag_x", y_label="lag_y", azimuthal_axis=None,
                 symmetrized=True, combined_error=None, combined_std=None,
                 noise_mask=None):
        self.S2 = np.asarray(S2)
        self.counts = np.asarray(counts)
        self.dx = float(dx)
        self.dy = float(dy)
        self.lags_x = np.asarray(lags_x)
        self.lags_y = np.asarray(lags_y)
        self.lags_i = np.asarray(lags_i)
        self.S2_x = np.asarray(S2_x)
        self.S2_y = np.asarray(S2_y)
        # ``S2_i`` is optional: the momentmap polar pipeline passes None
        # because its (dx, dy) = (arcsec, deg) are not commensurate, so an
        # azimuthal average of a mixed-units Euclidean norm is meaningless.
        self.S2_i = None if S2_i is None else np.asarray(S2_i)
        self.x_grid = None if x_grid is None else np.asarray(x_grid)
        self.y_grid = None if y_grid is None else np.asarray(y_grid)
        self.gridded = None if gridded is None else np.asarray(gridded)
        self.ref = ref
        self.ref_band = ref_band
        self.x_label = x_label
        self.y_label = y_label
        self.azimuthal_axis = azimuthal_axis
        # True for global mode and for ref-annulus mode with the
        # symmetrize post-processing applied. False only when the user
        # explicitly asked for the raw two-sided result. Plotting
        # routines branch on this so the displayed lag axis matches the
        # statistic actually computed.
        self.symmetrized = bool(symmetrized)
        self.combined_error = combined_error
        self.combined_std = combined_std
        self.noise_mask = noise_mask

    @property
    def max_lag_x(self):
        return self.S2.shape[0] // 2

    @property
    def max_lag_y(self):
        return self.S2.shape[1] // 2

    @property
    def extent(self):
        """Matplotlib ``extent`` for ``imshow(S2)``: (l_y_min, l_y_max,
        l_x_min, l_x_max). Axis 0 is plotted on the y-axis."""
        return (-self.max_lag_y * self.dy, self.max_lag_y * self.dy,
                -self.max_lag_x * self.dx, self.max_lag_x * self.dx)

    def _check_same_grid(self, other):
        """Raise if ``other`` has a different S2 shape or dx/dy."""
        if self.S2.shape != other.S2.shape:
            raise ValueError(
                "S2 shapes do not match: {} vs {}.".format(
                    self.S2.shape, other.S2.shape))
        if self.dx != other.dx or self.dy != other.dy:
            raise ValueError(
                "dx, dy do not match: ({}, {}) vs ({}, {}).".format(
                    self.dx, self.dy, other.dx, other.dy))

    @property
    def lags_x_full(self):
        """Two-sided radial lag axis, shape ``(2*max_lag_x+1,)``.

        Useful with :attr:`S2_x_full` in ``symmetrize=False`` mode,
        where the negative-lag half is the inward statistic and is
        physically distinct from the outward (positive-lag) half.
        """
        return np.arange(-self.max_lag_x, self.max_lag_x + 1) * self.dx

    @property
    def S2_x_full(self):
        """Full two-sided radial slice ``S_2[:, center_y]``, shape
        ``(2*max_lag_x+1,)``.

        With ``symmetrize=True`` (default) or in global mode this is
        symmetric about ``l_r = 0``. With ``symmetrize=False`` and a
        reference annulus, the positive-lag half is the outward
        statistic and the negative-lag half is the inward statistic.
        """
        return self.S2[:, self.max_lag_y]

    @classmethod
    def from_array(cls, f, dx=1.0, dy=1.0, max_lag_x=None, max_lag_y=None,
                   ref_i=-1, ref_band=0, n_bins=50, log_spaced=False,
                   symmetrize=True, **meta):
        """Compute ``S_2`` from a 2D array on a regular grid.

        Args:
            f (ndarray): 2D field. NaNs are excluded from pair averages.
            dx, dy (float): Physical grid spacing along axis 0 / 1.
            max_lag_x, max_lag_y (Optional[int]): Lag extents. Default to
                half the array along each axis.
            ref_i, ref_band (int): Reference-annulus index and half-width.
            n_bins (int): Number of radial bins for the azimuthal average.
            log_spaced (bool): If ``True``, log-spaced radial bins.
            symmetrize (bool): See :func:`compute_s2`. Recorded on the
                result as :attr:`StructureFunction2D.symmetrized` so the
                plotting routines can pick a sensible default lag axis.
            **meta: Forwarded to the :class:`StructureFunction2D` constructor
                (e.g. ``x_grid``, ``y_grid``, ``gridded``, ``ref``,
                ``ref_band``, ``x_label``, ``y_label``, ``azimuthal_axis``).

        Returns:
            StructureFunction2D
        """
        S2, counts, mlx, mly = compute_s2(
            f, max_lag_x=max_lag_x, max_lag_y=max_lag_y,
            ref_i=ref_i, ref_band=ref_band, symmetrize=symmetrize,
        )
        # In global mode the result is symmetric by construction.
        symmetrized = bool(symmetrize) or ref_i < 0
        lags_x, lags_y, lags_i, S2_x, S2_y, S2_i = extract_basic_profiles(
            S2, mlx, mly, dx=dx, dy=dy,
            n_bins=n_bins, log_spaced=log_spaced,
        )
        # Allow the caller to override S2_i (e.g. pass None for polar
        # results where dx and dy are incommensurate units).
        S2_i = meta.pop('S2_i', S2_i)
        # Convert the pixel-based ref_band to physical units for the
        # constructor unless the caller already supplied a physical value.
        meta.setdefault('ref_band', ref_band * dx if ref_band else 0.0)
        return cls(S2=S2, counts=counts, dx=dx, dy=dy,
                   lags_x=lags_x, lags_y=lags_y, lags_i=lags_i,
                   S2_x=S2_x, S2_y=S2_y, S2_i=S2_i,
                   symmetrized=symmetrized, **meta)

    def combine(self, others, n_bins=50, log_spaced=False):
        """Combine this result with one or more others via pair-count
        weighting (e.g. multiple noise realizations, multiple disks).

        Returns a new :class:`StructureFunction2D` with combined ``S_2``
        and ``counts``, recomputed 1D profiles, and the combined
        per-bin standard error attached as ``combined_error`` and
        intrinsic scatter as ``combined_std``.
        """
        if isinstance(others, StructureFunction2D):
            others = [others]
        all_results = [self, *others]

        for r in all_results[1:]:
            self._check_same_grid(r)

        S2_list = [r.S2 for r in all_results]
        counts_list = [r.counts for r in all_results]
        S2_comb, S2_err, S2_std = combine_s2_weighted(S2_list, counts_list)
        counts_comb = np.sum(counts_list, axis=0)

        lags_x, lags_y, lags_i, S2_x, S2_y, S2_i = extract_basic_profiles(
            S2_comb, self.max_lag_x, self.max_lag_y, dx=self.dx, dy=self.dy,
            n_bins=n_bins, log_spaced=log_spaced,
        )
        # Carry the mixed-units S2_i veto forward: if any input dropped it
        # (typical of the polar pipeline), the combined surface has the same
        # incommensurate axes and the recomputed S2_i is still meaningless.
        if any(r.S2_i is None for r in all_results):
            S2_i = None
        return type(self)(
            S2=S2_comb, counts=counts_comb, dx=self.dx, dy=self.dy,
            lags_x=lags_x, lags_y=lags_y, lags_i=lags_i,
            S2_x=S2_x, S2_y=S2_y, S2_i=S2_i,
            x_grid=self.x_grid, y_grid=self.y_grid,
            ref=self.ref, ref_band=self.ref_band,
            x_label=self.x_label, y_label=self.y_label,
            azimuthal_axis=self.azimuthal_axis,
            symmetrized=self.symmetrized,
            combined_error=S2_err, combined_std=S2_std,
        )

    def subtract(self, other, clip=True, n_bins=50, log_spaced=False):
        """Subtract another ``S_2`` from this one, returning a new result.

        Second-order structure functions add over statistically
        independent components: if the observed field is signal plus an
        independent noise field, ``S_2^obs = S_2^signal + S_2^noise``.
        So a model of the noise ``S_2`` can be removed lag-by-lag to
        recover the signal's structure function,
        ``S_2^signal = S_2^obs - S_2^noise``.

        ``other`` is another :class:`StructureFunction2D` on the *same*
        lag grid, typically the analytic noise prediction from
        :func:`gaussian_beam_s2`, an empirical noise ``S_2`` from
        :meth:`eddy.linecube.linecube.noise_structure_function`, or any
        parametric noise model evaluated on this object's lags. Note the
        noise ``S_2`` is generally *lag-dependent* (it rises from zero
        and saturates to ``2 sigma_noise^2`` only for lags much larger
        than the noise correlation scale), not a constant floor; this
        method subtracts the full lag dependence, which a flat-offset
        subtraction would get wrong at small lags.

        Unlike :meth:`compare_to` (which returns raw difference arrays
        for diagnostics), this returns a fully-formed
        :class:`StructureFunction2D` whose 1D profiles are recomputed
        from the differenced 2D map, so it can feed straight into
        :meth:`fit_spiral`, the heatmap helpers, etc.

        The validity of the subtraction rests on the noise being
        statistically *independent* of the signal (so the cross term
        ``<dS dN>`` vanishes in expectation). In a finite sample the
        cross term is only zero on average, and differencing two similar
        large values near the plateau is noisy, so the result can dip
        below zero there; ``clip=True`` (the default) floors it at zero.

        Args:
            other (StructureFunction2D): Noise model to subtract. Must
                share ``S2`` shape and ``dx, dy``.
            clip (bool): Clip the differenced ``S_2`` (2D and the
                recomputed profiles) at zero. Defaults to ``True``.
            n_bins (int): Radial bins for the azimuthal average of the
                result.
            log_spaced (bool): Log-spaced radial bins for the result.

        Returns:
            StructureFunction2D: ``self.S2 - other.S2``, with metadata
            (grid, reference annulus, labels, azimuthal axis) inherited
            from ``self`` and ``counts`` carried over from ``self``
            (the observed field's pair counts, the right weighting basis
            for the recovered signal).
        """
        self._check_same_grid(other)
        S2_diff = self.S2 - other.S2
        if clip:
            S2_diff = np.clip(S2_diff, 0.0, None)

        # Recompute the 1D profiles from the differenced (and possibly
        # clipped) 2D map so they stay consistent with S2 -- subtracting
        # the stored profiles directly would disagree with the 2D map
        # wherever the clip bites.
        lags_x, lags_y, lags_i, S2_x, S2_y, S2_i = extract_basic_profiles(
            S2_diff, self.max_lag_x, self.max_lag_y, dx=self.dx, dy=self.dy,
            n_bins=n_bins, log_spaced=log_spaced,
        )
        # Carry the mixed-units S2_i veto forward (see ``combine``).
        if self.S2_i is None or other.S2_i is None:
            S2_i = None
        return type(self)(
            S2=S2_diff, counts=self.counts, dx=self.dx, dy=self.dy,
            lags_x=lags_x, lags_y=lags_y, lags_i=lags_i,
            S2_x=S2_x, S2_y=S2_y, S2_i=S2_i,
            x_grid=self.x_grid, y_grid=self.y_grid,
            ref=self.ref, ref_band=self.ref_band,
            x_label=self.x_label, y_label=self.y_label,
            azimuthal_axis=self.azimuthal_axis,
            symmetrized=self.symmetrized,
        )

    def compare_to(self, other, eps=1e-30):
        """Compare this ``S_2`` against another on the same lag grid.

        The natural use case is empirical-vs-analytic: subtract the
        Gaussian-beam prediction from a measured noise ``S_2`` to expose
        the excess correlated structure left over by the imaging
        pipeline.

        Args:
            other (StructureFunction2D): The reference ``S_2`` to
                compare against. Must share ``S2`` shape and ``dx, dy``.
            eps (float): Floor added to ``other.S2`` in the ratio to
                avoid division by zero at zero lag (where the analytic
                prediction is exactly zero).

        Returns:
            dict with keys:
                ``diff`` (ndarray): ``self.S2 - other.S2``.
                ``ratio`` (ndarray): ``self.S2 / (other.S2 + eps)``.
                ``diff_x``, ``diff_y``, ``diff_i`` (ndarray): 1D cuts
                    along axis 0, axis 1, and the azimuthal average.
                ``ratio_x``, ``ratio_y``, ``ratio_i`` (ndarray): same
                    for the ratio.
                ``other`` (StructureFunction2D): reference, for plotting.
        """
        self._check_same_grid(other)
        diff = self.S2 - other.S2
        ratio = self.S2 / (other.S2 + eps)
        has_i = self.S2_i is not None and other.S2_i is not None
        return {
            "diff": diff,
            "ratio": ratio,
            "diff_x": self.S2_x - other.S2_x,
            "diff_y": self.S2_y - other.S2_y,
            "diff_i": (self.S2_i - other.S2_i) if has_i else None,
            "ratio_x": self.S2_x / (other.S2_x + eps),
            "ratio_y": self.S2_y / (other.S2_y + eps),
            "ratio_i": (self.S2_i / (other.S2_i + eps)) if has_i else None,
            "other": other,
        }

    def plot_comparison(self, other, axes=None, return_fig=False,
                        labels=("empirical", "analytic")):
        """Three-panel summary: 1D profile overlays, 2D difference, and
        1D residual profiles.

        Args:
            other (StructureFunction2D): The reference (typically the
                Gaussian-beam analytic prediction).
            axes (Optional[sequence of matplotlib.axes.Axes]): Length-3
                axes to draw into. New figure if ``None``.
            return_fig (bool): Return the figure object.
            labels (tuple of str): Legend labels for ``self`` and
                ``other`` on the profile panels.

        Returns:
            Optional[matplotlib.figure.Figure]
        """
        import matplotlib.pyplot as plt

        if axes is None:
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        else:
            fig = axes[0].figure

        ax0, ax1, ax2 = axes

        # 1D profile overlay: azimuthal average + two principal cuts.
        # The mixed-units polar pipeline leaves ``S2_i`` as ``None`` because
        # ``sqrt(l_x^2 + l_y^2)`` would mix arcsec and degrees; skip the
        # average overlay in that case.
        has_i = self.S2_i is not None and other.S2_i is not None
        if has_i:
            ax0.plot(self.lags_i, self.S2_i,
                     label="{} (avg)".format(labels[0]))
            ax0.plot(other.lags_i, other.S2_i, ls="--",
                     label="{} (avg)".format(labels[1]))
        ax0.plot(self.lags_x, self.S2_x, alpha=0.5,
                 label="{} ({})".format(labels[0], self.x_label))
        ax0.plot(self.lags_y, self.S2_y, alpha=0.5,
                 label="{} ({})".format(labels[0], self.y_label))
        ax0.set_xlabel("lag")
        ax0.set_ylabel(r"$S_2$")
        ax0.legend(fontsize=8)
        ax0.set_title("profiles")

        # 2D difference map.
        diff = self.S2 - other.S2
        vmax = float(np.nanmax(np.abs(diff))) if np.any(np.isfinite(diff)) else 1.0
        im = ax1.imshow(diff, origin="lower", aspect="auto",
                        extent=self.extent, cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax)
        ax1.set_xlabel(self.y_label)
        ax1.set_ylabel(self.x_label)
        ax1.set_title(r"$S_2$ residual ({} - {})".format(*labels))
        fig.colorbar(im, ax=ax1)

        # 1D residual cuts.
        if has_i:
            ax2.plot(self.lags_i, self.S2_i - other.S2_i,
                     label="azimuthal avg")
        ax2.plot(self.lags_x, self.S2_x - other.S2_x, alpha=0.7,
                 label=self.x_label)
        ax2.plot(self.lags_y, self.S2_y - other.S2_y, alpha=0.7,
                 label=self.y_label)
        ax2.axhline(0.0, color="k", lw=0.5, ls=":")
        ax2.set_xlabel("lag")
        ax2.set_ylabel(r"$\Delta S_2$")
        ax2.legend(fontsize=8)
        ax2.set_title("residual")

        fig.tight_layout()
        return fig if return_fig else None

    def plateau(self, frac=0.5, stat="median"):
        """Robust large-lag plateau of ``S_2`` (the ``2 sigma^2`` asymptote).

        Pools the outer-lag portion (lag ``> frac * max_lag``) of the
        radial and azimuthal slices and returns a robust statistic,
        ``median`` by default, so a single noisy outlier (common in a
        denoised / low-pair-count slice) does not set the level the way
        ``np.nanmax`` would.

        Args:
            frac (float): Fraction of the lag range treated as "large lag".
                ``0.5`` -> outer half of each slice.
            stat ({'median', 'mean'}): Reduction over the pooled outer cells.

        Returns:
            float: the plateau estimate (``nan`` if no finite outer cells).
        """
        vals = []
        for lags, s2 in ((self.lags_x, self.S2_x), (self.lags_y, self.S2_y)):
            if lags.size == 0:
                continue
            sel = np.asarray(s2)[lags >= frac * lags[-1]]
            vals.append(sel[np.isfinite(sel)])
        pooled = np.concatenate(vals) if vals else np.array([])
        if pooled.size == 0:
            return float("nan")
        return float(np.median(pooled) if stat == "median" else np.mean(pooled))

    def half_power_lag(self, axis="x", level=0.5, plateau=None):
        """Lag at which a 1D slice first reaches ``level * plateau``.

        A model-free correlation-scale proxy: the lag where ``S_2`` first
        crosses half its plateau (``= 1.18 ell`` for a Gaussian kernel, but
        no Gaussian assumption is made). Located by linear interpolation of
        the FIRST upward crossing, so it is robust to non-monotonic wiggles
        at larger lag.

        Args:
            axis ({'x', 'y'}): ``'x'`` -> radial slice ``S2_x`` (lag in the
                x units, arcsec); ``'y'`` -> azimuthal slice ``S2_y`` (lag
                in the y units, degrees).
            level (float): Fraction of the plateau to cross (``0.5`` = half
                power).
            plateau (Optional[float]): Plateau to use; defaults to
                :meth:`plateau`.

        Returns:
            float: the crossing lag in that axis's units (``nan`` if the
            slice never reaches the level within the lag range).
        """
        if axis == "x":
            lags, s2 = self.lags_x, self.S2_x
        elif axis == "y":
            lags, s2 = self.lags_y, self.S2_y
        else:
            raise ValueError("axis must be 'x' or 'y', got {!r}.".format(axis))
        if plateau is None:
            plateau = self.plateau()
        target = level * plateau
        if not np.isfinite(target) or target <= 0:
            return float("nan")
        above = np.isfinite(s2) & (np.asarray(s2) >= target)
        if not above.any():
            return float("nan")
        i = int(np.argmax(above))                  # first crossing index
        if i == 0:
            return float(lags[0])
        x0, x1, y0, y1 = lags[i - 1], lags[i], s2[i - 1], s2[i]
        if y1 == y0:
            return float(x1)
        return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))

    def reliability_weight(self, kind="counts"):
        """Relative per-annulus reliability for weighting cross-annulus averages.

        A single annulus's measurement is more trustworthy the more
        independent samples it contains. Use this to weight per-annulus
        quantities (e.g. the anisotropy log-ratio in ``T_2``) when
        collapsing them across radius.

        Args:
            kind ({'counts', 'neff'}):
                ``'counts'`` (default): total pair count on the on-axis
                    slices; model-free, more pairs -> more reliable. A
                    relative proxy.
                ``'neff'``: effective number of independent patches in the
                    annulus, ``~ (radial lag range / ell_r) * (360 deg /
                    ell_phi)``; closer to the true statistical weight but
                    assumes a single correlation scale via
                    :meth:`half_power_lag`.

        Returns:
            float: a non-negative relative weight (``0`` if undefined).

        Note:
            The statistically rigorous weight is the inverse variance of
            your specific statistic from the null / bootstrap ensemble;
            these are quick single-annulus proxies.
        """
        if kind == "counts":
            cx = self.counts[self.max_lag_x:, self.max_lag_y]
            cy = self.counts[self.max_lag_x, self.max_lag_y:]
            w = np.nansum(cx) + np.nansum(cy)
            return float(w) if np.isfinite(w) else 0.0
        if kind == "neff":
            ell_r = self.half_power_lag("x")
            ell_phi_deg = self.half_power_lag("y")
            if not (np.isfinite(ell_r) and ell_r > 0
                    and np.isfinite(ell_phi_deg) and ell_phi_deg > 0):
                return 0.0
            n_r = self.lags_x[-1] / ell_r          # patches across the radial lag range
            n_phi = 360.0 / ell_phi_deg            # patches around the full ring
            return float(max(n_r, 1.0) * max(n_phi, 1.0))
        raise ValueError("kind must be 'counts' or 'neff', got {!r}.".format(kind))

    def fit_spiral(self, modes=(1,), axis=None, p0=None):
        """Fit a multi-mode spiral model to a 1D azimuthal slice of S_2.

        Args:
            modes (tuple of int): Spiral modes to include, e.g.
                ``(1, 2, 3)``.
            axis (Optional[str]): Which slice to fit. ``'y'`` fits
                ``S2_y`` (axis 1), ``'x'`` fits ``S2_x`` (axis 0). If
                ``None``, uses :attr:`azimuthal_axis` if set, else
                ``'y'``.
            p0 (Optional[sequence]): Initial guess
                ``[Nphi, A_m1, A_m2, ...]``. If ``None``, a heuristic
                ``[0, sqrt(max(S2))/n, ...]`` is used.

        Returns:
            popt (ndarray): Best-fit ``[Nphi, A_m1, A_m2, ...]``.
            perr (ndarray): 1-sigma uncertainties from the Jacobian.
            model_fn (callable): ``f(dphi_deg) -> S_2`` evaluated at the
                best-fit parameters.
        """
        from scipy.optimize import least_squares

        if axis is None:
            axis = self.azimuthal_axis or "y"
        if axis == "y":
            dphi, S2_slice = self.lags_y, self.S2_y
        elif axis == "x":
            dphi, S2_slice = self.lags_x, self.S2_x
        else:
            raise ValueError("axis must be 'x' or 'y', got {!r}".format(axis))

        finite = np.isfinite(S2_slice)
        dphi_f = np.asarray(dphi)[finite]
        S2_f = np.asarray(S2_slice)[finite]
        if dphi_f.size < len(modes) + 1:
            raise ValueError("Not enough finite points to fit {} modes."
                             .format(len(modes)))

        model = _make_spiral_model(tuple(modes))

        if p0 is None:
            scale = np.sqrt(max(np.nanmax(S2_f), 1e-30)) / max(len(modes), 1)
            p0 = np.array([0.0] + [scale] * len(modes), dtype=float)
        p0 = np.asarray(p0, dtype=float)

        def residuals(params):
            return model(params, dphi_f) - S2_f

        result = least_squares(residuals, p0)
        popt = result.x

        # Covariance from the Jacobian, scaled by residual variance.
        J = result.jac
        dof = max(dphi_f.size - popt.size, 1)
        s_sq = float(np.sum(result.fun**2) / dof)
        try:
            cov = np.linalg.inv(J.T @ J) * s_sq
            perr = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
        except np.linalg.LinAlgError:
            perr = np.full_like(popt, np.nan)

        def model_fn(dphi_deg, popt=popt):
            return model(popt, np.asarray(dphi_deg))

        return popt, perr, model_fn

    # -- PITCHED-GRF FIT -- #

    def _grf_surface_setup(self, r_axis, r0, floor_frac, sigma_map, p0, *,
                           pitch):
        """Shared setup for the surface (global-mode) GRF fit.

        Returns ``(parts, est, data_bounds, plot_bestfit)`` where
        ``parts(mp)`` returns ``(resid_raw, weights)``: the un-normalized
        ``model - data`` array and the matching weights, both flat and
        masked to the populated lag bins; ``est`` is the linear-space
        initial-guess dict, ``data_bounds`` is the half-lag-range
        ``ell0r``/``ell0phi`` overlay for :func:`_grf_bounds_from_keys`, and
        ``plot_bestfit(mp)`` draws the measured/model/residual surfaces.

        With ``pitch=True`` the symmetry guard rejects a reference-annulus
        surface (the pitch ridge is unmeasurable there); ``pitch=False``
        skips the guard since the symmetric pitch=0 fit is valid either way.
        """
        r_axis = np.asarray(r_axis, dtype=float)
        if r_axis.ndim != 1 or r_axis.size < self.max_lag_x + 1:
            raise ValueError(
                "r_axis must be the 1D radial grid (axis 0) of the field this "
                "S_2 was built from; need at least max_lag_x + 1 = {} points "
                "(got shape {}).".format(self.max_lag_x + 1, np.shape(r_axis)))
        ddr = np.diff(r_axis)
        if not np.allclose(ddr, ddr[0]) or not np.isclose(ddr[0], self.dx):
            raise ValueError(
                "r_axis must be uniform with spacing dx={:.6g} (the surface's "
                "radial pixel scale); got mean spacing {:.6g}. Pass the same "
                "radial grid the field was built on.".format(
                    self.dx, float(np.mean(ddr))))

        # Non-positive base radii blow up ``ell_r(r) = ell0r * (r/r0)**alphar``
        # (0 or inf), which propagates to a NaN row in the base-row average.
        # ``polar_deprojection``'s default ``rgrid = np.arange(0, ...)`` lands
        # exactly there, so drop them up front rather than crashing inside the
        # SVD. Keep the filter outside the public ``grf_s2_2d_global`` so its
        # contract (one expected row per supplied radius) is unchanged.
        n_drop = int(np.sum(r_axis <= 0))
        if n_drop:
            r_axis = r_axis[r_axis > 0]
            if r_axis.size < self.max_lag_x + 1:
                raise ValueError(
                    "r_axis has only {} positive radii after dropping "
                    "non-positive entries; need at least max_lag_x + 1 = {}. "
                    "Pass a trimmed grid.".format(
                        r_axis.size, self.max_lag_x + 1))
            warnings.warn(
                "Dropped {} non-positive radii from r_axis before the GRF "
                "surface fit (ell_r(0) is singular for nonzero alphar).".format(
                    n_drop), stacklevel=3)

        lags_x = self.lags_x_full
        lags_y_deg = np.arange(-self.max_lag_y, self.max_lag_y + 1) * self.dy
        S2 = self.S2

        # Empty (zero-count) lag bins are not measurements: exclude them so
        # they neither bias the fit nor inflate the plateau / weights.
        mask = self.counts > 0
        if not mask.any():
            raise ValueError("No populated lag bins to fit.")

        if pitch:
            # Pitch lives entirely in the antisymmetric (off-diagonal) part of
            # the surface. A reference-annulus S_2 mirror-fills the azimuthal
            # lag (S2[:, +dphi] == S2[:, -dphi] exactly), so it is symmetric
            # in l_phi and carries no pitch sign; only a global-mode
            # (ref_i=-1) surface -- point-symmetric but not l_phi-symmetric --
            # preserves the ridge. A single global realization is never
            # exactly l_phi-symmetric, so an exact-mirror test cleanly
            # catches the reference-mode case.
            both = mask & mask[:, ::-1]
            if both.any() and np.allclose(S2[both], S2[:, ::-1][both],
                                          rtol=1e-9, atol=1e-12):
                raise ValueError(
                    "This S_2 is exactly symmetric in the azimuthal lag, i.e. "
                    "a reference-annulus surface (ref={!r}). Its kernel "
                    "mirror-fills the azimuthal lag, averaging the "
                    "antisymmetric pitch ridge -- and the pitch sign -- away, "
                    "so pitch is unmeasurable. Rebuild in global mode: "
                    "StructureFunction2D.from_array(field, ref_i=-1, ...) "
                    "or call fit_GRF(pitch=False).".format(self.ref))

        plateau = float(np.nanmax(np.abs(S2[mask])))
        if not plateau > 0:
            raise ValueError("S_2 is everywhere zero on populated bins.")

        if sigma_map is not None:
            w = np.asarray(sigma_map, dtype=float)
            if w.shape != S2.shape:
                raise ValueError(
                    "sigma_map must match S2.shape {} (got {}).".format(
                        S2.shape, w.shape))
        else:
            w = np.maximum(np.abs(S2), floor_frac * plateau)

        w_masked = w[mask]

        def model(mp):
            return grf_s2_2d_global(
                r_axis, lags_x, lags_y_deg, sigma=mp["sigma"],
                alphar=mp["alphar"], ell0r=mp["ell0r"],
                alphaphi=mp["alphaphi"], ell0phi=mp["ell0phi"],
                r0=r0, pitch=mp["pitch"])

        def parts(mp):
            return (model(mp) - S2)[mask], w_masked

        # Data-driven correlation-length bounds: half the resolved radial
        # (|lag_r| arcsec) and azimuthal (arc-length radians(|dphi|) * r) lag
        # ranges. These override the wide ``ell`` fallbacks so the lengths
        # cannot run away past the scales the surface constrains; the
        # caller's ``bounds`` still take precedence. Degenerate inputs (no
        # positive lags / no positive radii) feed ``np.nan`` placeholders so
        # ``_grf_data_bounds`` drops them and the module fallbacks apply.
        ar = np.abs(lags_x.astype(float))
        ap = np.radians(np.abs(lags_y_deg.astype(float)))
        r_pos = r_axis[r_axis > 0]
        ar_pos = ar[ar > 0]
        ap_pos = ap[ap > 0]
        data_bounds = _grf_data_bounds({
            "ell0r": (ar_pos.min() if ar_pos.size else np.nan,
                      ar.max() if ar.size else np.nan),
            "ell0phi": (ap_pos.min() * r_pos.min()
                        if ap_pos.size and r_pos.size else np.nan,
                        ap.max() * r_axis.max() if r_axis.size else np.nan),
        })

        est = self._grf_surface_initial_guess(r_axis, plateau, p0)

        def plot_bestfit(mp):
            return self._plot_grf_bestfit_surface(
                lags_x, lags_y_deg, S2, mask, model, mp)

        return parts, est, data_bounds, plot_bestfit

    def _grf_surface_initial_guess(self, r_axis, plateau, p0):
        """Heuristic start for the surface fit, as a linear-space dict
        ``{sigma, alphar, ell0r, ell0phi, alphaphi, pitch}``.

        ``sigma`` from the plateau (``2 sigma^2``), ``ell0r`` from the radial
        half-power lag, ``ell0phi`` from the azimuthal half-power lag (arc
        length) at the median radius, and ``alphar = alphaphi = pitch = 0`` (a
        single global surface gives no radial-slope leverage). ``p0`` (any of
        ``sigma, alphar, ell0r, ell0phi, alphaphi, pitch`` in linear space,
        ``pitch`` in deg) overrides the corresponding heuristic.
        """
        g = dict(sigma=None, alphar=None, ell0r=None, ell0phi=None,
                 alphaphi=None, pitch=None)
        if p0:
            g.update(p0)

        sigma0 = (g["sigma"] if g["sigma"] is not None
                  else np.sqrt(max(plateau, 0.0) / 2.0)) or 1.0
        hp_x = self.half_power_lag("x")
        hp_y = self.half_power_lag("y")
        dx = self.dx
        ell0r0 = g["ell0r"]
        if ell0r0 is None:
            ell0r0 = hp_x / 1.177 if np.isfinite(hp_x) and hp_x > 0 else 5.0 * dx
        ell0r0 = max(ell0r0, dx)

        ell0phi0 = g["ell0phi"]
        if ell0phi0 is None:
            r_ref = float(np.median(r_axis))
            ell_phi0 = np.radians(hp_y) * r_ref / 1.177
            ell0phi0 = (ell_phi0 if np.isfinite(ell_phi0) and ell_phi0 > 0
                        else ell0r0)
        ell0phi0 = max(ell0phi0, dx)

        alphar0 = 0.0 if g["alphar"] is None else float(g["alphar"])
        alphaphi0 = alphar0 if g["alphaphi"] is None else float(g["alphaphi"])
        pitch0 = 0.0 if g["pitch"] is None else float(g["pitch"])
        return dict(sigma=float(sigma0), alphar=alphar0, ell0r=float(ell0r0),
                    ell0phi=float(ell0phi0), alphaphi=alphaphi0, pitch=pitch0)

    def fit_GRF(self, *, pitch=False, ref_r=None, r_axis=None, r0=1.0,
                floor_frac=0.05, sigma_map=None, method="lsq", p0=None,
                bounds=None, priors=None, jitter=True, fit_alphaphi=False,
                nwalkers=64, nburnin=500, nsteps=1000, scatter=1e-3,
                plots=None, returns=None, pool=None, progress=True):
        """Fit the anisotropic-GRF parameters to this single ``S_2``.

        Dispatches on ``pitch`` and the construction mode:

        * **Reference-annulus mode** (``ref_r`` provided or :attr:`ref` set),
          ``pitch=False``: fit the radial and azimuthal slices ``S2_x`` /
          ``S2_y`` jointly against :func:`grf_s2_slices` at the single
          ``ref_r``. Cheaper than the surface fit; same forward model as the
          stack's per-annulus fit. Solves for ``(sigma, alphar, ell0r,
          ell0phi)``.
        * **Global mode** (``r_axis`` provided, no reference radius),
          ``pitch=False``: fit the full 2D surface against
          :func:`grf_s2_2d_global` with ``pitch`` held at 0. Same parameters
          as the slice fit.
        * **Global mode**, ``pitch=True``: fit the surface with ``pitch``
          freed. The only configuration that can recover the pitch sign,
          since the antisymmetric ridge is the unique pitch signature
          (slices and reference-annulus surfaces are pitch-symmetric).

        ``pitch=True`` with a reference-annulus surface raises: the
        kernel mirror-fills the azimuthal lag and averages the pitch sign
        away. Use a global-mode :class:`StructureFunction2D` (built with
        ``ref_i=-1``) for that case.

        Args:
            pitch (bool): If ``True``, include ``pitch`` in the fit (global
                mode only). Default ``False``.
            ref_r (Optional[float]): Reference radius [arcsec] for the
                ref-mode slice fit. Falls back to :attr:`ref` when not
                given; the dispatcher picks ref-mode if either is set.
            r_axis (Optional[ndarray]): 1D radial grid (axis 0) of the field
                this ``S_2`` was built from, required for the global-mode
                surface fit. Must be uniform with spacing :attr:`dx`.
            r0 (float): Reference radius [arcsec] in
                ``ell_r(r) = ell0r (r / r0)**alphar``.
            floor_frac (float): Floor for the default weighting, as a
                fraction of the data plateau (ignored when ``sigma_map`` is
                supplied).
            sigma_map: Per-bin uncertainty. For the surface fit, an array
                matching :attr:`S2`. For the slice fit, a
                ``(sigma_r, sigma_phi)`` pair of 1D arrays matching
                :attr:`S2_x` / :attr:`S2_y`. ``None`` uses the heuristic
                ``max(|S2|, floor)`` weighting; supply a real uncertainty
                (e.g. the bootstrap ``S_2`` scatter) for trustworthy
                ``mcmc`` credible intervals.
            method ({'lsq', 'mcmc'}): Back-end. ``'lsq'`` is a
                least-squares point estimate with a Gauss-Newton covariance;
                ``'mcmc'`` is an ``emcee`` sampler initialized from the
                ``lsq`` solution.
            p0 (Optional[dict]): Override the heuristic start for any of
                ``sigma, alphar, ell0r, ell0phi, alphaphi, pitch`` (linear;
                ``pitch`` in deg).
            bounds (Optional[dict]): Hard bounds per key ``sigma, alphar,
                ell0r, ell0phi, alphaphi, pitch, jitter`` (linear/deg).
                Defaults: ``alphar``/``alphaphi`` in ``[-5, 5]``, ``pitch``
                in ``[-89, 89]`` deg, and ``ell0r``/``ell0phi`` set per-fit
                to half the resolved lag range.
            priors (Optional[dict]): Gaussian priors (``mcmc``) keyed the
                same way, each ``(mu, sigma)`` in *sampling* space (log for
                ``sigma, ell0r, ell0phi, jitter``; linear for ``alphar,
                alphaphi, pitch``). Intersected with the flat ``bounds``;
                samples outside the box get ``-inf`` posterior regardless of
                the Gaussian prior, so a prior that pulls walkers past a
                bound is effectively truncated there.
            jitter (bool): ``mcmc`` only. Add a log-scale nuisance ``s``
                scaling every weight.
            fit_alphaphi (bool): If ``True``, free the azimuthal slope
                ``alphaphi``. Default ``False`` ties ``alphaphi = alphar``.
            nwalkers, nburnin, nsteps (int): ``emcee`` sizes.
            scatter (float): Fractional walker-ball scatter about the
                ``lsq`` solution.
            plots (Optional[list]): ``mcmc`` only; any of ``'walkers'``,
                ``'corner'``, ``'bestfit'`` (measured/model/residual or
                slice overlay depending on mode), or ``'none'``. Defaults
                to ``['walkers', 'corner']``.
            returns (Optional[list]): ``mcmc`` only; any of ``'samples'``,
                ``'sampler'``, ``'lnprob'``, ``'percentiles'``, ``'dict'``,
                ``'none'``. Defaults to ``['samples']``.
            pool: Optional worker pool for ``emcee``.
            progress (bool): Show the ``emcee`` progress bar.

        Returns:
            ``method='lsq'``: ``(params, perr, sol, cov)``, the best-fit
            and formal 1-sigma dicts keyed by the active sampled
            parameters, the ``scipy.optimize.least_squares`` result, and
            the sampling-space covariance. A tied ``alphaphi`` is added to
            ``params`` (equal to ``alphar``) without a ``perr`` entry.

            ``method='mcmc'``: whatever ``returns`` selects.
        """
        resolved_ref_r = ref_r if ref_r is not None else self.ref
        ref_mode = resolved_ref_r is not None

        if pitch and ref_mode:
            raise ValueError(
                "pitch=True is unmeasurable on a reference-annulus surface "
                "(ref={!r}): the kernel mirror-fills the azimuthal lag and "
                "averages the antisymmetric pitch ridge away. Rebuild in "
                "global mode (StructureFunction2D.from_array(field, "
                "ref_i=-1, ...)) and pass r_axis instead.".format(
                    resolved_ref_r))

        if ref_mode and r_axis is not None:
            raise ValueError(
                "r_axis is for the global-mode surface fit; ref_r/self.ref "
                "selects the slice fit. Pass only one of the two.")
        if not ref_mode and r_axis is None:
            raise ValueError(
                "Need either ref_r (or self.ref) for the slice fit, or "
                "r_axis for the global-mode surface fit.")

        if ref_mode:
            parts, est, data_bounds, plot_bestfit = self._grf_slice_setup(
                resolved_ref_r, r0, floor_frac, sigma_map, p0)
        else:
            parts, est, data_bounds, plot_bestfit = self._grf_surface_setup(
                r_axis, r0, floor_frac, sigma_map, p0, pitch=pitch)

        return _grf_fit_core(
            parts, est, data_bounds, pitch=pitch, fit_alphaphi=fit_alphaphi,
            method=method, bounds=bounds, priors=priors, jitter=jitter,
            nwalkers=nwalkers, nburnin=nburnin, nsteps=nsteps,
            scatter=scatter, plots=plots, returns=returns, pool=pool,
            progress=progress, plot_bestfit=plot_bestfit, user_p0=p0,
        )

    def _plot_grf_bestfit_surface(self, lags_x, lags_y_deg, S2, mask,
                                  model, mp):
        """Measured / model / residual surfaces for the surface-mode GRF best
        fit. ``mp`` is a linear model-parameter dict (see
        :func:`_grf_unpack`)."""
        import matplotlib.pyplot as plt

        m = model(mp)
        resid = np.where(mask, m - S2, np.nan)
        extent = (lags_y_deg[0], lags_y_deg[-1], lags_x[0], lags_x[-1])
        vmax = np.nanmax(np.abs(S2[mask]))
        fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6),
                                 constrained_layout=True)
        for ax, img, ttl, kw in (
                (axes[0], np.where(mask, S2, np.nan), "measured",
                 dict(vmin=0, vmax=vmax)),
                (axes[1], np.where(mask, m, np.nan), "model",
                 dict(vmin=0, vmax=vmax)),
                (axes[2], resid, "residual",
                 dict(cmap="bwr", vmin=-0.3 * vmax, vmax=0.3 * vmax))):
            im = ax.imshow(img, origin="lower", aspect="auto", extent=extent,
                           **kw)
            ax.set(title=ttl, xlabel=self.y_label, ylabel=self.x_label)
            fig.colorbar(im, ax=ax)
        return axes

    def _grf_slice_setup(self, ref_r, r0, floor_frac, sigma_map, p0):
        """Shared setup for the single-surface slice (ref-mode) GRF fit.

        Same return shape as :meth:`_grf_surface_setup`: ``(parts, est,
        data_bounds, plot_bestfit)``. ``parts(mp)`` fits ``S2_x`` and
        ``S2_y`` of this one annulus against :func:`grf_s2_slices` at the
        bound reference radius ``ref_r``. ``sigma_map`` is a ``(sigma_r,
        sigma_phi)`` pair for the radial/azimuthal slices (analogous to the
        stack's per-annulus entry), or ``None`` for the heuristic weights.
        """
        ref_r = float(ref_r)

        mx = self.counts[self.max_lag_x:, self.max_lag_y] > 0
        my = self.counts[self.max_lag_x, self.max_lag_y:] > 0
        if not (mx.any() or my.any()):
            raise ValueError("No populated lag bins to fit.")

        S2_x = np.asarray(self.S2_x)
        S2_y = np.asarray(self.S2_y)
        plateau = float(max(np.nanmax(np.abs(S2_x[mx])) if mx.any() else 0.0,
                            np.nanmax(np.abs(S2_y[my])) if my.any() else 0.0))
        if not plateau > 0:
            raise ValueError("S_2 slices are everywhere zero on populated bins.")
        floor = floor_frac * plateau

        if sigma_map is not None:
            try:
                wr_full, wp_full = sigma_map
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "sigma_map must be a (sigma_r, sigma_phi) pair of 1D "
                    "weight arrays matching S2_x / S2_y."
                ) from exc
            wr_full = np.asarray(wr_full, dtype=float)
            wp_full = np.asarray(wp_full, dtype=float)
            if wr_full.shape != S2_x.shape or wp_full.shape != S2_y.shape:
                raise ValueError(
                    "sigma_map shapes {} / {} must match S2_x {} and "
                    "S2_y {}.".format(wr_full.shape, wp_full.shape,
                                      S2_x.shape, S2_y.shape))
        else:
            wr_full = np.maximum(np.abs(S2_x), floor)
            wp_full = np.maximum(np.abs(S2_y), floor)

        lags_x = self.lags_x
        lags_y = self.lags_y

        def parts(mp):
            p = grf_s2_slices(ref_r, lags_x, lags_y, sigma=mp["sigma"],
                              alphar=mp["alphar"], ell0r=mp["ell0r"],
                              alphaphi=mp["alphaphi"], ell0phi=mp["ell0phi"],
                              r0=r0)
            resid = np.concatenate([(p["S2_r"] - S2_x)[mx],
                                    (p["S2_phi"] - S2_y)[my]])
            weights = np.concatenate([wr_full[mx], wp_full[my]])
            return resid, weights

        # Half the resolved lag range on each slice (arcsec for ell0r,
        # arc-length arcsec for ell0phi at this single annulus).
        ar = np.asarray(lags_x, dtype=float)
        ap = np.radians(np.asarray(lags_y, dtype=float))
        data_bounds = _grf_data_bounds({
            "ell0r": (ar[ar > 0].min() if np.any(ar > 0) else np.nan,
                      ar.max() if ar.size else np.nan),
            "ell0phi": (ap[ap > 0].min() * ref_r if np.any(ap > 0) else np.nan,
                        ap.max() * ref_r if ap.size else np.nan),
        })

        est = self._grf_slice_initial_guess(ref_r, plateau, p0)

        def plot_bestfit(mp):
            return self._plot_grf_bestfit_slices(ref_r, r0, mx, my, mp)

        return parts, est, data_bounds, plot_bestfit

    def _grf_slice_initial_guess(self, ref_r, plateau, p0):
        """Heuristic start for the single-surface slice fit (linear-space
        dict ``{sigma, alphar, ell0r, ell0phi, alphaphi}``).

        ``sigma`` from the plateau, ``ell0r`` from the radial half-power lag,
        ``ell0phi`` from the azimuthal half-power lag converted to arc length
        at ``ref_r``, and ``alphar = alphaphi = 0`` (single annulus → no
        radial-slope leverage). ``p0`` overrides any heuristic.
        """
        g = dict(sigma=None, alphar=None, ell0r=None, ell0phi=None,
                 alphaphi=None)
        if p0:
            g.update(p0)
        sigma0 = (g["sigma"] if g["sigma"] is not None
                  else np.sqrt(max(plateau, 0.0) / 2.0)) or 1.0
        hp_x = self.half_power_lag("x")
        hp_y = self.half_power_lag("y")
        dx = self.dx
        ell0r0 = g["ell0r"]
        if ell0r0 is None:
            ell0r0 = hp_x / 1.177 if np.isfinite(hp_x) and hp_x > 0 else 5.0 * dx
        ell0r0 = max(ell0r0, dx)

        ell0phi0 = g["ell0phi"]
        if ell0phi0 is None:
            ell_phi0 = np.radians(hp_y) * ref_r / 1.177
            ell0phi0 = (ell_phi0 if np.isfinite(ell_phi0) and ell_phi0 > 0
                        else ell0r0)
        ell0phi0 = max(ell0phi0, dx)

        alphar0 = 0.0 if g["alphar"] is None else float(g["alphar"])
        alphaphi0 = alphar0 if g["alphaphi"] is None else float(g["alphaphi"])
        return dict(sigma=float(sigma0), alphar=alphar0, ell0r=float(ell0r0),
                    ell0phi=float(ell0phi0), alphaphi=alphaphi0)

    def _plot_grf_bestfit_slices(self, ref_r, r0, mx, my, mp, axes=None):
        """Overlay the model on the measured radial and azimuthal slices for
        a single-surface slice fit. Mirror of
        :meth:`StructureFunction2DStack._plot_grf_bestfit` for one annulus.
        """
        import matplotlib.pyplot as plt

        if axes is None:
            _, axes = plt.subplots(1, 2, figsize=(9.0, 3.6),
                                   constrained_layout=True)
        p = grf_s2_slices(ref_r, self.lags_x, self.lags_y, sigma=mp["sigma"],
                          alphar=mp["alphar"], ell0r=mp["ell0r"],
                          alphaphi=mp["alphaphi"], ell0phi=mp["ell0phi"],
                          r0=r0)
        axes[0].plot(self.lags_x[mx], self.S2_x[mx], ".", ms=3)
        axes[0].plot(self.lags_x[mx], p["S2_r"][mx], "-", lw=1.0)
        axes[1].plot(self.lags_y[my], self.S2_y[my], ".", ms=3,
                     label=r"$r={:.2f}$".format(ref_r))
        axes[1].plot(self.lags_y[my], p["S2_phi"][my], "-", lw=1.0)
        axes[0].set(xlabel=r"$\ell_r$ (arcsec)", ylabel=r"$S_2$",
                    title="radial slice")
        axes[1].set(xlabel=r"$\Delta\phi$ (deg)", title="azimuthal slice")
        axes[1].legend(fontsize=8)
        return axes

    # -- PLOTTING -- #

    def plot_2d(self, ax=None, return_fig=False, **imshow_kwargs):
        """Plot the 2D ``S_2`` surface with axis 0 on the vertical axis.

        Args:
            ax (Optional[matplotlib.axes.Axes]): Axes to draw into.
            return_fig (bool): Return the figure.
            **imshow_kwargs: Extra kwargs forwarded to ``imshow``.

        Returns:
            ``matplotlib.figure.Figure`` if ``return_fig=True``, else None.
        """
        fig, ax = _resolve_ax(ax)
        kwargs = dict(origin="lower", aspect="auto", extent=self.extent)
        kwargs.update(imshow_kwargs)
        im = ax.imshow(self.S2, **kwargs)
        ax.set_xlabel(self.y_label)
        ax.set_ylabel(self.x_label)
        fig.colorbar(im, ax=ax, label=r"$S_2$")
        return fig if return_fig else None

    def plot_profiles(self, ax=None, return_fig=False):
        """Plot ``S_2_x``, ``S_2_y`` and (when defined) the azimuthal average.

        The azimuthal average ``S_2_i`` is only physically meaningful when
        ``dx`` and ``dy`` share units (e.g. both arcsec): the underlying
        radial bins are circles of radius ``sqrt(l_x^2 + l_y^2)`` in raw
        index space. For the momentmap polar pipeline (``dx`` in arcsec,
        ``dy`` in degrees), ``S_2_i`` is left ``None`` and this method
        plots only the two axis slices.
        """
        fig, ax = _resolve_ax(ax)
        ax.plot(self.lags_x, self.S2_x, label=self.x_label)
        ax.plot(self.lags_y, self.S2_y, label=self.y_label)
        if self.S2_i is not None:
            ax.plot(self.lags_i, self.S2_i, label="azimuthal average",
                    ls="--")
        ax.set_xlabel("lag")
        ax.set_ylabel(r"$S_2$")
        ax.legend()
        return fig if return_fig else None


# -- STACKED RESULT CONTAINER -- #


class StructureFunction2DStack:
    """A stack of :class:`StructureFunction2D` results computed at
    different reference radii on the same polar grid.

    Built by
    :meth:`eddy.momentmap.momentmap.compute_structure_function_stack`,
    which performs the polar deprojection once and runs the kernel N
    times with different ``ref_r`` values.

    Iteration / indexing yields the individual per-radius
    :class:`StructureFunction2D` results, so the stack behaves like a
    list. Stacked numpy arrays of the most common per-radius outputs
    are exposed as properties (``S2_stack``, ``S2_y_stack``,
    ``S2_x_stack``, ``S2_i_stack``).

    Attributes:
        ref_rs (ndarray): Reference radii [arcsec], shape ``(N_ref,)``.
        ref_band (float): Half-width [arcsec] of each reference annulus.
        results (list of StructureFunction2D): Per-radius results.
        x_grid, y_grid (Optional[ndarray]): Shared polar grid the
            stack was computed on.
        gridded (Optional[ndarray]): Shared deprojected field, shape
            ``(N_r, N_phi)``.
    """

    def __init__(self, ref_rs, ref_band, results, x_grid=None,
                 y_grid=None, gridded=None):
        self.ref_rs = np.asarray(ref_rs, dtype=float)
        self.ref_band = float(ref_band)
        self.results = list(results)
        if len(self.results) != self.ref_rs.size:
            raise ValueError("len(results) must equal len(ref_rs).")
        if self.ref_rs.size == 0:
            raise ValueError("StructureFunction2DStack requires at least one result.")
        self.x_grid = None if x_grid is None else np.asarray(x_grid)
        self.y_grid = None if y_grid is None else np.asarray(y_grid)
        self.gridded = None if gridded is None else np.asarray(gridded)

    def __len__(self):
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def __getitem__(self, idx):
        return self.results[idx]

    @classmethod
    def from_array(cls, field, ref_rs, *, x_axis=None, dx=1.0, dy=1.0,
                   ref_band=0.0, max_lag_x=None, max_lag_y=None,
                   n_bins=50, log_spaced=False, symmetrize=True,
                   azimuthal_axis="y", x_label="lag_x", y_label="lag_y",
                   y_grid=None):
        """Build a stack from one bare 2D polar field at a sequence of radii.

        The bare-array analogue of
        :meth:`eddy.momentmap.momentmap.compute_structure_function_stack`:
        runs :meth:`StructureFunction2D.from_array` at each reference radius
        on the SAME field (axis 0 = radius, axis 1 = azimuth). For real sky
        data that still needs deprojection, use the ``momentmap`` method
        instead; this is for fields already on a polar grid.

        Args:
            field (ndarray): 2D field, axis 0 = radius, axis 1 = azimuth.
            ref_rs (sequence of float): Reference radii, in the units of
                ``x_axis`` (or of ``dx`` if ``x_axis`` is ``None``). Each is
                matched to the nearest ``x_axis`` entry.
            x_axis (Optional[ndarray]): Physical coordinate of axis 0.
                Defaults to ``np.arange(n_x) * dx``.
            dx, dy (float): Grid spacing along axis 0 / 1.
            ref_band (float): Reference-annulus half-width in ``x_axis`` units.
            max_lag_x, max_lag_y (Optional[int]): Lag extents IN PIXELS
                (as in :meth:`StructureFunction2D.from_array`).
            n_bins, log_spaced, symmetrize: Forwarded per annulus.
            azimuthal_axis, x_label, y_label: Result metadata.
            y_grid (Optional[ndarray]): Axis-1 coordinate stored on the stack.

        Returns:
            StructureFunction2DStack
        """
        field = np.asarray(field)
        if field.ndim != 2:
            raise ValueError("field must be 2D (n_x, n_y); got shape {}."
                             .format(field.shape))
        if x_axis is None:
            x_axis = np.arange(field.shape[0]) * dx
        x_axis = np.asarray(x_axis, dtype=float)
        ref_rs = np.atleast_1d(np.asarray(ref_rs, dtype=float))
        ref_band_idx = int(round(ref_band / dx)) if ref_band > 0 else 0

        results = []
        for rr in ref_rs:
            ref_i = int(np.argmin(np.abs(x_axis - rr)))
            results.append(StructureFunction2D.from_array(
                field, dx=dx, dy=dy, max_lag_x=max_lag_x, max_lag_y=max_lag_y,
                ref_i=ref_i, ref_band=ref_band_idx,
                n_bins=n_bins, log_spaced=log_spaced, symmetrize=symmetrize,
                azimuthal_axis=azimuthal_axis, x_label=x_label,
                y_label=y_label, ref=float(x_axis[ref_i]),
            ))
        return cls(ref_rs=ref_rs, ref_band=float(ref_band), results=results,
                   x_grid=x_axis, y_grid=y_grid, gridded=field)

    @property
    def lags_x(self):
        return self.results[0].lags_x

    @property
    def lags_y(self):
        return self.results[0].lags_y

    @property
    def lags_i(self):
        return self.results[0].lags_i

    @property
    def lags_x_full(self):
        """Two-sided radial lag axis, shape ``(2*mlx+1,)``."""
        return self.results[0].lags_x_full

    @property
    def symmetrized(self):
        """Whether the per-ring results were symmetrized (taken from the
        first result; ``compute_structure_function_stack`` uses one
        ``symmetrize`` value for the whole stack)."""
        return self.results[0].symmetrized

    @property
    def S2_stack(self):
        """Full 2D ``S_2`` surfaces, shape ``(N_ref, 2*mlx+1, 2*mly+1)``."""
        return np.stack([r.S2 for r in self.results])

    @property
    def S2_x_stack(self):
        """Outward radial-lag slice at each ``ref_r`` (``l_r >= 0`` only),
        shape ``(N_ref, mlx+1)``. For the two-sided slice that distinguishes
        inward from outward under ``symmetrize=False``, use
        :attr:`S2_x_full_stack`."""
        return np.stack([r.S2_x for r in self.results])

    @property
    def S2_x_full_stack(self):
        """Two-sided radial-lag slice at each ``ref_r``, shape
        ``(N_ref, 2*mlx+1)``. Negative-lag half is the inward statistic
        when ``symmetrize=False``."""
        return np.stack([r.S2_x_full for r in self.results])

    @property
    def S2_y_stack(self):
        """Azimuthal-lag slice at each ``ref_r``, shape ``(N_ref, mly+1)``."""
        return np.stack([r.S2_y for r in self.results])

    @property
    def S2_i_stack(self):
        """Azimuthally-averaged profile at each ``ref_r``, shape
        ``(N_ref, n_bins)``. ``None`` when any constituent surface left
        ``S2_i`` undefined (e.g. the mixed-units polar pipeline)."""
        if any(r.S2_i is None for r in self.results):
            return None
        return np.stack([r.S2_i for r in self.results])

    @property
    def counts_x_stack(self):
        """Pair counts on the outward radial-lag slice at each ``ref_r``,
        shape ``(N_ref, mlx+1)``, aligned cell-for-cell with
        :attr:`S2_x_stack`. Feeds :meth:`pairwise_error_heatmaps`."""
        return np.stack([r.counts[r.max_lag_x:, r.max_lag_y]
                         for r in self.results])

    @property
    def counts_x_full_stack(self):
        """Pair counts on the two-sided radial-lag slice at each ``ref_r``,
        shape ``(N_ref, 2*mlx+1)``, aligned with :attr:`S2_x_full_stack`."""
        return np.stack([r.counts[:, r.max_lag_y] for r in self.results])

    @property
    def counts_y_stack(self):
        """Pair counts on the azimuthal-lag slice at each ``ref_r``, shape
        ``(N_ref, mly+1)``, aligned with :attr:`S2_y_stack`."""
        return np.stack([r.counts[r.max_lag_x, r.max_lag_y:]
                         for r in self.results])

    def extent(self, azimuth_in_degrees=True):
        """Matplotlib ``extent`` for ``imshow(self.gridded, origin='lower')``:
        ``(y_grid_min, y_grid_max, x_grid_min, x_grid_max)``.

        For a polar-deprojected stack the azimuth axis is stored in
        radians (the ``polar_deprojection`` convention). By default
        (``azimuth_in_degrees=True``) the returned y bounds are converted
        to degrees to match :meth:`plot_gridded`'s default display.

        Args:
            azimuth_in_degrees (bool): Convert the azimuth (y) bounds from
                radians to degrees. Default ``True``.
        """
        if self.x_grid is None or self.y_grid is None:
            raise ValueError(
                "extent requires x_grid/y_grid on the stack; the stack "
                "was constructed without them."
            )
        y0, y1 = float(self.y_grid[0]), float(self.y_grid[-1])
        if azimuth_in_degrees:
            y0, y1 = np.degrees(y0), np.degrees(y1)
        return (y0, y1, float(self.x_grid[0]), float(self.x_grid[-1]))

    def fit_spiral(self, modes=(1,), axis=None, p0=None):
        """Fit a multi-mode spiral model at every reference radius.

        Args:
            modes (tuple of int): Spiral modes to fit at each ring.
            axis (Optional[str]): Slice to fit, see
                :meth:`StructureFunction2D.fit_spiral`. Defaults to the
                azimuthal axis set on each result (``'y'`` for stacks
                from ``compute_structure_function_stack``).
            p0 (Optional[sequence]): Shared initial guess used at every
                radius. If ``None``, each ring uses its own heuristic.

        Returns:
            popt (ndarray): Best-fit parameters per ring, shape
                ``(N_ref, 1 + len(modes))``. Column 0 is ``Nphi``;
                remaining columns are mode amplitudes.
            perr (ndarray): 1-sigma uncertainties, same shape.
            model_fns (list): Per-ring model callables, as returned by
                :meth:`StructureFunction2D.fit_spiral`. Each takes a
                scalar azimuthal lag ``dphi`` and returns the evaluated
                model.
        """
        popts, perrs, model_fns = [], [], []
        for r in self.results:
            popt, perr, model_fn = r.fit_spiral(modes=modes, axis=axis, p0=p0)
            popts.append(popt)
            perrs.append(perr)
            model_fns.append(model_fn)
        return np.asarray(popts), np.asarray(perrs), model_fns

    def calculate_modal_power(self, modes=(1,), axis=None, p0=None):
        """Per-annulus azimuthal modal power and fractional power.

        Fits the multi-mode spiral model (:meth:`fit_spiral`) at every
        reference radius and converts the fitted amplitudes into power.
        Each mode contributes ``A_m^2 (1 - cos(m dphi))`` to the azimuthal
        ``S_2``, so the **modal power** is ``A_m^2``, and ``Nphi`` (the fit
        offset, column 0) is the non-oscillating baseline (noise + any
        axisymmetric power).

        Args:
            modes (tuple of int): Spiral modes, as in :meth:`fit_spiral`.
            axis (Optional[str]): Slice to fit; defaults to the azimuthal
                axis.
            p0 (Optional[sequence]): Shared initial guess.

        Returns:
            dict with per-annulus arrays (``N_ref`` rows; mode arrays have
            ``len(modes)`` columns in the order of ``modes``):

            * ``modes``: the mode list, shape ``(n_modes,)``.
            * ``power``: modal power ``A_m^2``, ``(N_ref, n_modes)``.
            * ``power_err``: its 1-sigma (``2 |A_m| sigma_{A_m}``).
            * ``offset``: ``Nphi`` baseline, ``(N_ref,)``.
            * ``modal_total``: ``sum_m power``, ``(N_ref,)``.
            * ``fitted_total``: ``offset + modal_total``, ``(N_ref,)``.
            * ``data_var``: mean observed azimuthal ``S_2`` per annulus
              (``nanmean`` of the azimuthal heatmap), ``(N_ref,)``.
            * ``frac_among_modes``: ``power / modal_total``: each mode's
              share of the *modal* budget, ``(N_ref, n_modes)``.
            * ``frac_of_fit``: ``power / fitted_total``: share of the
              total fitted power (modes + offset), ``(N_ref, n_modes)``.
            * ``frac_offset``: ``offset / fitted_total``: non-modal share
              of the fit, ``(N_ref,)``.
            * ``frac_of_data``: ``power / data_var``: modal power as a
              fraction of the observed azimuthal variance, ``(N_ref, n_modes)``.
            * ``frac_of_data_total``: ``sum_m frac_of_data``, ``(N_ref,)``.
            * ``popt``, ``perr``: the raw fit outputs.
        """
        popt, perr = self.fit_spiral(modes=modes, axis=axis, p0=p0)
        offset = popt[:, 0]
        amps = popt[:, 1:]
        power = amps ** 2
        power_err = 2.0 * np.abs(amps) * perr[:, 1:]
        modal_total = power.sum(axis=1)
        fitted_total = offset + modal_total

        _, _, C = self.calculate_azimuthal_heatmap()
        data_var = np.nanmean(C, axis=1)

        with np.errstate(invalid="ignore", divide="ignore"):
            frac_among_modes = power / modal_total[:, None]
            frac_of_fit = power / fitted_total[:, None]
            frac_offset = offset / fitted_total
            frac_of_data = power / data_var[:, None]

        return dict(
            modes=np.asarray(modes),
            power=power, power_err=power_err, offset=offset,
            modal_total=modal_total, fitted_total=fitted_total,
            data_var=data_var,
            frac_among_modes=frac_among_modes,
            frac_of_fit=frac_of_fit,
            frac_offset=frac_offset,
            frac_of_data=frac_of_data,
            frac_of_data_total=frac_of_data.sum(axis=1),
            popt=popt, perr=perr,
        )

    _MODAL_POWER_LABELS = {
        "power": r"modal power $A_m^2$",
        "frac_of_fit": r"fractional power $A_m^2 / (N_\phi + \sum_m A_m^2)$",
        "frac_of_data": r"$A_m^2 / \langle S_2^\phi\rangle$",
        "frac_among_modes": r"$A_m^2 / \sum_m A_m^2$",
    }

    def plot_modal_power(self, modes=(1,), axis=None, p0=None,
                         which="frac_of_fit", show_offset=True, ax=None,
                         return_fig=False, **plot_kwargs):
        """Plot per-mode (fractional) azimuthal power vs reference radius.

        One line per mode, from :meth:`calculate_modal_power`.

        Args:
            modes (tuple of int): Spiral modes, as in :meth:`calculate_modal_power`.
            axis, p0: Forwarded to :meth:`calculate_modal_power` / :meth:`fit_spiral`.
            which ({'frac_of_fit', 'frac_of_data', 'frac_among_modes',
                'power'}): Which per-mode quantity to plot. Default
                ``'frac_of_fit'`` (each mode's share of the total fitted
                power).
            show_offset (bool): When ``which='frac_of_fit'``, also draw the
                non-modal baseline fraction (``frac_offset``) as a dashed
                line, completing the budget to 1.
            ax (Optional): Matplotlib ``Axes`` to draw into.
            return_fig (bool): Return the figure object.
            **plot_kwargs: Forwarded to ``ax.plot`` for the mode lines.
        """
        if which not in self._MODAL_POWER_LABELS:
            raise ValueError(
                "which must be one of {}; got {!r}.".format(
                    list(self._MODAL_POWER_LABELS), which)
            )

        fig, ax = _resolve_ax(ax)

        mp = self.calculate_modal_power(modes=modes, axis=axis, p0=p0)
        Y = mp[which]
        for j, m in enumerate(mp["modes"]):
            ax.plot(self.ref_rs, Y[:, j], label="$m={}$".format(int(m)),
                    **plot_kwargs)
        if show_offset and which == "frac_of_fit":
            ax.plot(self.ref_rs, mp["frac_offset"], ls="--", color="k",
                    label="offset")

        ax.set_xlabel(r"$r_{\rm ref}$ (arcsec)")
        ax.set_ylabel(self._MODAL_POWER_LABELS[which])
        ax.legend()
        return fig if return_fig else None

    def _check_ref_rs(self, others, what):
        """Validate that ``others`` share this stack's ``ref_rs`` (count
        and values), so per-annulus operations line up ring-for-ring."""
        for o in others:
            if not isinstance(o, StructureFunction2DStack):
                raise TypeError(
                    "{}: expected a StructureFunction2DStack, got {}. "
                    "(A single StructureFunction2D has no reference-radius "
                    "axis; use StructureFunction2D.subtract / .combine for "
                    "single results.)".format(what, type(o).__name__)
                )
            if len(o) != len(self):
                raise ValueError(
                    "{}: stacks have different N_ref ({} vs {}).".format(
                        what, len(self), len(o))
                )
            if not np.allclose(o.ref_rs, self.ref_rs):
                raise ValueError(
                    "{}: stacks have different ref_rs.".format(what)
                )

    def subtract(self, other, clip=True, n_bins=50, log_spaced=False):
        """Subtract another stack from this one, ring by ring.

        Applies :meth:`StructureFunction2D.subtract` at every reference
        radius, so it removes a noise model from an observed stack while
        preserving the per-ring geometry. Because the subtraction happens
        at the ``S_2`` level, *every* heatmap built from the result is
        correctly denoised, including the nonlinear ones (``row_max``
        normalization, and the :meth:`calculate_anisotropy_heatmap`
        ratio) where subtracting two finished heatmaps would be wrong.

        ``other`` is typically the mean noise stack from
        :meth:`combine` over many noise-only realizations, computed with
        the same ``compute_structure_function_stack`` call (same
        ``ref_rs``, ``ref_band`` and deprojection geometry) as this one.

        Args:
            other (StructureFunction2DStack): Noise model to subtract.
                Must share ``ref_rs`` and per-ring lag grids.
            clip (bool): Forwarded to
                :meth:`StructureFunction2D.subtract`; floor the
                differenced ``S_2`` at zero. Default ``True``.
            n_bins (int): Radial bins for the azimuthal average of each
                differenced ring.
            log_spaced (bool): Log-spaced radial bins for the result.

        Returns:
            StructureFunction2DStack: a new stack with the same
            ``ref_rs`` / ``ref_band`` / grid, holding the per-ring
            differences. ``gridded`` is dropped (the denoised stack does
            not correspond to a single field).
        """
        self._check_ref_rs([other], "subtract")
        results = [s.subtract(o, clip=clip, n_bins=n_bins,
                              log_spaced=log_spaced)
                   for s, o in zip(self.results, other.results)]
        return type(self)(
            ref_rs=self.ref_rs, ref_band=self.ref_band, results=results,
            x_grid=self.x_grid, y_grid=self.y_grid,
        )

    def combine(self, others, n_bins=50, log_spaced=False):
        """Pair-count-weighted combination with one or more other stacks,
        ring by ring.

        Applies :meth:`StructureFunction2D.combine` at every reference
        radius. The main use here is averaging many noise-only
        realizations into a single mean noise stack (the expected noise
        ``S_2`` per ring) before passing it to :meth:`subtract`, so a
        single realization's scatter is not injected into the recovered
        signal.

        Args:
            others (StructureFunction2DStack or sequence): One or more
                stacks to combine with this one. Must share ``ref_rs``
                and per-ring lag grids.
            n_bins (int): Radial bins for the azimuthal average of each
                combined ring.
            log_spaced (bool): Log-spaced radial bins for the result.

        Returns:
            StructureFunction2DStack: a new stack with combined per-ring
            ``S_2`` and ``counts``. Each ring carries ``combined_error``
            and ``combined_std`` as set by
            :meth:`StructureFunction2D.combine`.
        """
        if isinstance(others, StructureFunction2DStack):
            others = [others]
        others = list(others)
        self._check_ref_rs(others, "combine")
        results = [
            self.results[i].combine([o.results[i] for o in others],
                                    n_bins=n_bins, log_spaced=log_spaced)
            for i in range(len(self))
        ]
        return type(self)(
            ref_rs=self.ref_rs, ref_band=self.ref_band, results=results,
            x_grid=self.x_grid, y_grid=self.y_grid, gridded=self.gridded,
        )

    def collapse(self, n_bins=50, log_spaced=False):
        """Collapse the reference-radius axis into one global ``StructureFunction2D``.

        Pair-count-weighted combination of the per-annulus results
        (:meth:`StructureFunction2D.combine` across this stack's own
        ``results``), i.e. every lag cell is averaged over reference radii
        weighted by its pair count, equivalently all pairs from every
        annulus are pooled. Returns a single :class:`StructureFunction2D`
        with no reference annulus (``ref=None``).

        Note this differs from :meth:`combine`, which combines *across
        stacks* (e.g. realizations) ring-by-ring and returns a stack. This
        collapses *within* one stack, across ``ref_rs``, and returns a
        single result.

        When the reference annuli tile the field without overlap or gaps
        (e.g. ``ref_band=0`` over every radial ring, as
        ``compute_structure_function_stack`` produces), the on-axis radial
        and azimuthal slices (:attr:`S2_x`, :attr:`S2_y`) and the
        azimuthal-average profile (:attr:`S2_i`) match the true global
        ``S_2`` (``StructureFunction2D.from_array(field, ref_i=-1)``)
        exactly, so the radial/azimuthal heatmaps and 1D profiles collapse
        exactly. The full 2D :attr:`S2` surface, however, differs off the
        axes: the reference-annulus kernel mirror-fills the azimuthal lag
        (``S2[dr, +dphi] = S2[dr, -dphi]``), averaging the diagonal ridge
        away, so any off-axis / 2D structure (e.g. a pitch or grand-design
        ridge) is *not* the global result; compute ``ref_i=-1`` directly
        for that. With overlapping bands shared pixels are over-weighted and
        with gaps some pairs are missing, degrading even the on-axis match.

        The ``counts`` of the collapse reflect the reference-anchoring
        convention (each pair counted per anchoring annulus), so their
        absolute values differ from a direct global; the count-weighted
        ``S_2`` is the meaningful output.

        This collapses the RAW ``S_2``. To collapse the row-normalized or
        anisotropy heatmaps, collapse first and normalize / ratio the
        result; those transforms are nonlinear and must not be averaged
        across radii.

        Args:
            n_bins (int): Radial bins for the azimuthal-average profile.
            log_spaced (bool): Log-spaced radial bins.

        Returns:
            StructureFunction2D: the count-weighted collapse, ``ref=None``,
            carrying ``combined_error`` / ``combined_std`` from
            :meth:`StructureFunction2D.combine`.
        """
        collapsed = self.results[0].combine(self.results[1:], n_bins=n_bins,
                                            log_spaced=log_spaced)
        collapsed.ref = None
        collapsed.ref_band = None
        collapsed.symmetrized = self.symmetrized
        return collapsed

    def plateaus(self, frac=0.5, stat="median"):
        """Per-annulus large-lag plateau, shape ``(N_ref,)``.

        Maps :meth:`StructureFunction2D.plateau` over the stack.
        """
        return np.array([r.plateau(frac=frac, stat=stat) for r in self.results])

    def half_power_lags(self, axis="x", level=0.5, plateau=None):
        """Per-annulus half-power lag along ``axis``, shape ``(N_ref,)``.

        Maps :meth:`StructureFunction2D.half_power_lag` over the stack.
        ``axis='x'`` returns radial lags in arcsec; ``axis='y'`` azimuthal
        lags in degrees (convert to arclength with ``np.radians(.) * ref_rs``).

        Args:
            axis ({'x', 'y'}): Lag axis to query.
            level (float): Fraction of the plateau to cross.
            plateau (Optional[float or array-like]): Plateau override.
                A scalar is shared across all rings; an array of length
                ``N_ref`` uses a per-ring value (e.g. from a previous
                :meth:`StructureFunction2D.plateau` call). Defaults to
                each ring's own :meth:`~StructureFunction2D.plateau`.
        """
        if plateau is None or np.ndim(plateau) == 0:
            plateaus = [plateau] * len(self.results)
        else:
            plateaus = list(plateau)
        return np.array([r.half_power_lag(axis=axis, level=level, plateau=p)
                         for r, p in zip(self.results, plateaus)])

    def reliability_weights(self, kind="counts"):
        """Per-annulus reliability weights, shape ``(N_ref,)``.

        Maps :meth:`StructureFunction2D.reliability_weight` over the stack;
        use to weight per-annulus quantities when collapsing across radius.
        """
        return np.array([r.reliability_weight(kind=kind) for r in self.results])

    def measure_heuristics(self, r_min=None, r_max=None, t1c_arclength=True,
                           rescale_returns=True):
        """Scalar heuristics characterising this structure-function stack.

        Six numbers summarising the field's amplitude, correlation lengths,
        anisotropy and (radial) stationarity, built from the per-annulus
        :meth:`half_power_lags`, :meth:`reliability_weights` (``neff``) and the
        collapsed :meth:`plateau`. They map onto the anisotropic-GRF parameters
        fit by :meth:`fit_GRF` (``T1a`` -> ``sigma``, ``T1b`` -> ``ell0r``,
        ``T1c`` -> ``ell0phi``, ``T2`` -> anisotropy ``A``, ``T3`` -> ``alphar``,
        ``T4`` -> ``alphaphi - 1``), but are model-free per-ring measurements,
        not a fit.

        Args:
            r_min, r_max (Optional[float]): restrict every cross-ring average
                and slope to annuli with ``ref_rs`` in ``[r_min, r_max]``
                (arcsec). Either bound may be ``None`` (open on that side);
                ``None, None`` uses the full stack. Use this to confine the
                heuristics to the radial band the ``neff`` weights concentrate
                on, and to drop poorly-deprojected inner annuli (sparse
                azimuthal sampling, largest ``ell_r / r``, most polar-grid
                aliasing) that otherwise bias the slopes.
            t1c_arclength (bool): units of T1c. If ``True`` (default) report the
                arc length ``s_phi = radians(ell_phi) * ref_rs`` [arcsec],
                commensurate with ``ell_r`` but folding in the
                deprojection-uncertain ring radius. If ``False`` report the
                angular azimuthal scale ``ell_phi`` [deg], measured natively on
                the deprojected grid (no r multiply, so no inherited radial
                systematic). (Only T1c is affected; T2 always uses the arc
                length.)
            rescale_returns (bool): if ``True`` (default) return the
                human-facing magnitudes ``sigma_hat = sqrt(T1a / 2)`` and
                ``A_hat = exp(T2)``; if ``False`` return the raw statistics
                ``T1a = 2 sigma^2`` (plateau) and ``T2 = log A``. Only the
                *display* of T1a/T2 changes; all internal averaging stays in the
                raw (additive) coordinates.

        Returns:
            tuple: ``(T1a, T1b, T1c, T2, T3, T4)``, all floats:

                T1a (float): Plateau ``2 sigma^2`` (raw,
                    ``rescale_returns=False``) or velocity dispersion
                    ``sigma_hat = sqrt(T1a / 2)`` (rescaled, default). Measured
                    on the collapsed stack, independent of the radial selection.
                T1b (float): Radial correlation length ``ell_r`` [arcsec],
                    neff-weighted mean over the selected rings.
                T1c (float): Azimuthal correlation length, neff-weighted mean
                    over the selected rings. Arc length
                    ``s_phi = r * ell_phi`` [arcsec] (default,
                    ``t1c_arclength=True``) or angular scale ``ell_phi`` [deg]
                    (``t1c_arclength=False``).
                T2 (float): Anisotropy, reported as ``log A`` (raw) or
                    ``A_hat = exp(log A)`` (rescaled). Computed as the
                    neff-weighted mean of ``log(s_phi / ell_r)``, where
                    ``s_phi = radians(ell_phi) * ref_rs``. For a
                    radius-dependent anisotropy (``alphaphi != alphar``) this
                    is the band-averaged ``log A(r)``; the radial trend is
                    captured by T3 and T4.
                T3 (float): Radial stationarity: neff-weighted slope of
                    ``log ell_r`` vs ``log r``, equal to ``alphar``.
                T4 (float): Azimuthal stationarity: neff-weighted slope of
                    ``log ell_phi[deg]`` vs ``log r``, equal to
                    ``alphaphi - 1``. Deprojection-robust (no r multiply). A
                    slope near zero implies angularly self-similar modes; a
                    slope near -1 implies fixed physical arc length.
                    ``T3 - T4 == 1`` exactly when the anisotropy is
                    radius-independent (``alphaphi == alphar``).
        """
        plateau = self.collapse().plateau()

        # Compute per-ring plateaus once; pass them through to avoid
        # four redundant plateau() calls per ring (two in half_power_lags
        # and two more inside reliability_weights('neff')).
        pls = self.plateaus()
        ell_rs = self.half_power_lags('x', plateau=pls)     # radial scale [arcsec]
        ell_ps_deg = self.half_power_lags('y', plateau=pls)  # azimuthal scale [deg]
        ell_ps_arc = np.radians(ell_ps_deg) * self.ref_rs    # azimuthal arc length [arcsec]

        # Inline the neff formula from reliability_weight('neff') using the
        # half-power lags already computed above, avoiding 2N more plateau calls.
        lags_x_max = np.array([r.lags_x[-1] for r in self.results])
        n_r = lags_x_max / ell_rs
        n_phi = 360.0 / ell_ps_deg
        ok = (np.isfinite(ell_rs) & (ell_rs > 0)
              & np.isfinite(ell_ps_deg) & (ell_ps_deg > 0))
        weights = np.where(ok, np.maximum(n_r, 1.0) * np.maximum(n_phi, 1.0), 0.0)

        # radial band over which the cross-ring averages/slopes are taken
        in_range = np.ones(self.ref_rs.shape, dtype=bool)
        if r_min is not None:
            in_range = np.logical_and(in_range, self.ref_rs >= r_min)
        if r_max is not None:
            in_range = np.logical_and(in_range, self.ref_rs <= r_max)

        # ``reliability_weight('neff')`` returns finite 0.0 on rings with a
        # NaN half-power lag, so positivity has to be folded in -- otherwise
        # an all-zero ``weights[mask]`` reaches ``np.average`` and raises
        # ZeroDivisionError. (``mask_arc`` is redundant with ``mask_p`` here:
        # ``ell_ps_arc = radians(ell_ps_deg) * ref_rs`` is finite iff
        # ``ell_ps_deg`` is, given finite ``ref_rs``.)
        w_ok = np.isfinite(weights) & (weights > 0.0)
        mask_r = np.isfinite(ell_rs) & w_ok & in_range
        mask_p = np.isfinite(ell_ps_deg) & w_ok & in_range
        if not (mask_r.any() and mask_p.any()):
            raise ValueError(
                "no annuli with positive reliability weights in "
                "[r_min, r_max] = {}".format((r_min, r_max)))

        T1a = plateau
        T1b = np.average(ell_rs[mask_r], weights=weights[mask_r])
        if t1c_arclength:
            T1c = np.average(ell_ps_arc[mask_p], weights=weights[mask_p])       # arc length [arcsec]
        else:
            T1c = np.average(ell_ps_deg[mask_p], weights=weights[mask_p])       # angular [deg]

        # anisotropy -- ratio of physical scales, so the arc length (hence r)
        # is unavoidable. Use the *per-ring* ell_r in the denominator so the
        # ratio is log(A) on every ring rather than (per-ring arc) / (mean ell_r).
        mask_rp = mask_r & mask_p
        T2 = np.average(np.log(ell_ps_arc / ell_rs)[mask_rp], weights=weights[mask_rp])

        # weighted log-log slope of a per-ring scale against ref_rs
        def _slope(ell, mask):
            m = mask & (ell > 0.0) & (self.ref_rs > 0.0)
            if m.sum() < 2:
                return np.nan
            x = np.log(self.ref_rs)[m]
            y = np.log(ell)[m]
            X = np.vstack([x, np.ones_like(x)]).T
            w = weights[m]
            try:
                return np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * y))[0]
            except np.linalg.LinAlgError:
                return np.nan

        T3 = _slope(ell_rs, mask_r)        # radial stationarity     (== alphar)
        T4 = _slope(ell_ps_deg, mask_p)    # azimuthal stationarity  (== alphaphi - 1)

        if rescale_returns:
            return np.sqrt(T1a / 2.0), T1b, T1c, np.exp(T2), T3, T4
        return T1a, T1b, T1c, T2, T3, T4

    def _grf_setup(self, r0, r_range, sigma_map, floor_frac, p0):
        """Shared setup for :meth:`fit_GRF`: select annuli and build the
        ``parts`` / ``plot_bestfit`` closures shared by both back-ends.

        Returns ``(parts, est, data_bounds, plot_bestfit)``. ``parts(mp)``
        takes a linear model-parameter dict (``sigma, alphar, ell0r,
        ell0phi, alphaphi``) and returns ``(resid_raw, weights)``: the
        un-normalized ``model - data`` residual and the matching weight
        array, concatenated over the selected annuli's radial and azimuthal
        slices. ``est`` is the dict of linear-space initial estimates (see
        :meth:`_grf_initial_guess`); ``data_bounds`` is the half-lag-range
        ``ell0r``/``ell0phi`` overlay for :func:`_grf_bounds_from_keys`;
        ``plot_bestfit(mp)`` draws the model-on-slices goodness-of-fit
        panel.
        """
        idx = np.arange(len(self))
        if r_range is not None:
            lo, hi = r_range
            idx = idx[(self.ref_rs >= lo) & (self.ref_rs <= hi)]
        if idx.size == 0:
            raise ValueError("r_range selected no annuli.")

        rr = self.ref_rs
        sel = [self.results[i] for i in idx]

        # Empty lag bins (no finite pairs) are returned as exactly 0 by the
        # kernel -- they are not measurements. Mask them so they neither drag
        # the model toward zero nor count toward the plateau / weighting.
        masks_x = [s.counts[s.max_lag_x:, s.max_lag_y] > 0 for s in sel]
        masks_y = [s.counts[s.max_lag_x, s.max_lag_y:] > 0 for s in sel]
        plateau = np.nanmax([np.nanmax(np.abs(s.S2_x[m])) if m.any() else 0.0
                             for s, m in zip(sel, masks_x)])
        if not plateau > 0:
            raise ValueError(
                "No finite pairs in the selected annuli/lags -- check "
                "`r_range`, the deprojection mask, and `max_lag_x`.")
        floor = floor_frac * plateau

        def parts(mp):
            resid, weights = [], []
            for j, (i, mx, my) in enumerate(zip(idx, masks_x, masks_y)):
                s = self.results[i]
                p = grf_s2_slices(rr[i], s.lags_x, s.lags_y, sigma=mp["sigma"],
                                  alphar=mp["alphar"], ell0r=mp["ell0r"],
                                  alphaphi=mp["alphaphi"], ell0phi=mp["ell0phi"],
                                  r0=r0)
                if sigma_map is not None:
                    wr, wp = sigma_map[j]
                else:
                    wr = np.maximum(np.abs(s.S2_x), floor)
                    wp = np.maximum(np.abs(s.S2_y), floor)
                resid += [(p["S2_r"] - s.S2_x)[mx],
                          (p["S2_phi"] - s.S2_y)[my]]
                weights += [np.asarray(wr)[mx], np.asarray(wp)[my]]
            return np.concatenate(resid), np.concatenate(weights)

        # Data-driven correlation-length bounds: half the resolved radial
        # (arcsec) and azimuthal (arc-length, radians(dphi) * r) lag ranges,
        # pooled over the selected annuli. These override the wide ``ell``
        # fallbacks so the lengths cannot run away past the scales S_2
        # constrains; the caller's ``bounds`` still take precedence.
        rr_sel = self.ref_rs[idx]
        lr = [np.asarray(s.lags_x, dtype=float) for s in sel]
        lp = [np.radians(np.asarray(s.lags_y, dtype=float)) * r
              for s, r in zip(sel, rr_sel)]
        # Empty generators (no annulus with positive lags) yield np.nan so
        # ``_grf_data_bounds`` drops the entry and the module fallback applies.
        data_bounds = _grf_data_bounds({
            "ell0r": (min((a[a > 0].min() for a in lr if np.any(a > 0)),
                          default=np.nan),
                      max((a.max() for a in lr if a.size), default=np.nan)),
            "ell0phi": (min((a[a > 0].min() for a in lp if np.any(a > 0)),
                            default=np.nan),
                        max((a.max() for a in lp if a.size),
                            default=np.nan)),
        })

        est = self._grf_initial_guess(idx, r0, p0)

        def plot_bestfit(mp):
            return self._plot_grf_bestfit(idx, r0, mp)

        return parts, est, data_bounds, plot_bestfit

    def _grf_initial_guess(self, idx, r0, p0):
        """Heuristic starting point for :meth:`fit_GRF`, as a dict of
        linear-space estimates ``{sigma, alphar, ell0r, ell0phi, alphaphi}``.

        Each parameter is seeded from the scalar heuristics rather than a fixed
        guess, which keeps the optimizer away from the ``sigma``-amplitude-scale
        plateau degeneracy:

        * ``sigma`` from the robust plateau (``plateau = 2 sigma^2``), i.e. the
          data standard deviation;
        * ``ell_r`` per annulus from the radial half-power lag
          (``hp ~ 1.177 ell_r`` for a Gaussian), fit in ln-ln vs radius -> the
          slope is ``alphar`` and the value at ``r0`` is ``ell0r``;
        * ``ell_phi`` (arc length) per annulus from the azimuthal half-power
          lag, fit the same way -> ``alphaphi`` and ``ell0phi``.

        When a heuristic is unavailable (too few resolved annuli) it falls back
        to ``ell0r = 5 dx``, ``alphar = 0``, an isotropic ``ell0phi = ell0r``,
        ``alphaphi = alphar`` and ``sigma`` from the plateau. ``p0`` (a dict
        with any of ``sigma``, ``alphar``, ``ell0r``, ``ell0phi``, ``alphaphi``)
        overrides the corresponding heuristic.
        """
        sel = [self.results[i] for i in idx]
        rr = self.ref_rs[idx]
        dx = self.results[0].dx

        guess = dict(sigma=None, alphar=None, ell0r=None,
                     ell0phi=None, alphaphi=None)
        if p0:
            guess.update(p0)

        # sigma from the robust plateau (= 2 sigma^2 = 2 * data variance).
        plateau = np.nanmedian([s.plateau() for s in sel])
        sigma0 = (guess["sigma"] if guess["sigma"] is not None
                  else np.sqrt(max(plateau, 0.0) / 2.0))

        # Per-annulus correlation lengths from the half-power lags.
        hp_x = np.array([s.half_power_lag("x") for s in sel])
        hp_y = np.array([s.half_power_lag("y") for s in sel])
        w = np.array([s.reliability_weight() for s in sel])
        ell_r = hp_x / 1.177                          # arcsec
        ell_phi = np.radians(hp_y) * rr / 1.177       # arcsec (arc length)

        def _loglog(ell, slope_guess, intercept_guess, iso_fallback):
            """(slope, value-at-r0) from a reliability-weighted ln-ln fit."""
            slope, val0 = slope_guess, intercept_guess
            if slope is not None and val0 is not None:
                return slope, max(val0, dx)
            ok = np.isfinite(ell) & (ell > 0) & np.isfinite(w) & (w > 0)
            if ok.sum() >= 2:
                m, c = np.polyfit(np.log(rr[ok]), np.log(ell[ok]), 1,
                                  w=np.sqrt(w[ok]))
                if slope is None:
                    slope = m
                if val0 is None:
                    val0 = np.exp(c + m * np.log(r0))
            else:
                if slope is None:
                    slope = 0.0
                if val0 is None:
                    val0 = ell[ok][0] if ok.any() else iso_fallback
            return slope, max(val0, dx)

        alphar0, ell0r0 = _loglog(ell_r, guess["alphar"], guess["ell0r"],
                                  5.0 * dx)
        # Azimuthal scale: seed from ell_phi(r); fall back to isotropic.
        alphaphi0, ell0phi0 = _loglog(ell_phi, guess["alphaphi"],
                                      guess["ell0phi"], ell0r0)

        return dict(sigma=float(sigma0), alphar=float(alphar0),
                    ell0r=float(ell0r0), ell0phi=float(ell0phi0),
                    alphaphi=float(alphaphi0))

    def fit_GRF(self, *, r0=1.0, r_range=None, sigma_map=None, floor_frac=0.05,
                method="lsq", p0=None, bounds=None, priors=None, jitter=True,
                fit_alphaphi=False, pitch=False, nwalkers=64, nburnin=500,
                nsteps=1000, scatter=1e-3, plots=None, returns=None,
                pool=None, progress=True):
        """Fit the anisotropic-GRF parameters to the stack's ``S_2`` slices.

        Fits :func:`grf_s2_slices` jointly to the measured radial (``S2_x``)
        and azimuthal (``S2_y``) slices across all annuli, solving for the field
        whose radial correlation length is ``ell_r(r) = ell0r (r / r0)**alphar``
        and whose azimuthal (arc-length) length is
        ``ell_phi(r) = ell0phi (r / r0)**alphaphi``. The plateau is
        ``2 sigma^2`` and the (radius-dependent) anisotropy ``ell_phi / ell_r``
        is a derived quantity.

        ``pitch`` is unmeasurable from the slice fit (the on-axis slices are
        symmetric under a pitch sign flip; only the off-diagonal ridge of the
        full 2D surface preserves it). ``pitch=True`` therefore raises here;
        use :meth:`StructureFunction2D.fit_GRF` with ``pitch=True`` on a
        global-mode surface (built with ``ref_i=-1``) instead.

        By default ``alphaphi`` is tied to ``alphar`` (a radius-independent
        anisotropy), so the fit solves for ``(sigma, alphar, ell0r, ell0phi)`` --
        the same dimensionality as the historical ``(sigma, alpha, A, ell0)``
        model, just with the azimuthal amplitude fit directly as ``ell0phi``
        instead of ``A = ell0phi/ell0r``. Set ``fit_alphaphi=True`` to free the
        azimuthal slope and measure a radius-dependent anisotropy.

        Both back-ends work in log space for the positive parameters
        (``log sigma, alphar, log ell0r, log ell0phi[, alphaphi]``) so those
        stay positive and the priors are flat in log.

        * ``method='lsq'`` (default): a fast Levenberg-Marquardt-style
          least-squares point estimate with a Gauss-Newton covariance.
        * ``method='mcmc'``: an ``emcee`` ensemble sampler, initialized from
          the least-squares solution, that returns the full posterior. The
          likelihood is Gaussian in the slice residuals weighted by
          ``sigma_map`` (or the default weighting); with ``jitter=True`` an
          extra log-scale nuisance parameter ``s`` multiplies every weight
          (variance ``(s w)**2``), marginalizing over an imperfect overall
          uncertainty scale so the posterior widths are calibrated. Because
          the default weighting is a heuristic scale rather than a true
          uncertainty, supply ``sigma_map`` (e.g. the bootstrap ``S_2``
          scatter) for trustworthy credible intervals.

        Args:
            r0 (float): Reference radius [arcsec] in
                ``ell_r(r) = ell0r (r / r0)**alphar``.
            r_range (Optional[tuple]): ``(r_lo, r_hi)`` [arcsec] to fit only
                annuli in that range (e.g. to skip unresolved inner radii
                where ``ell_r`` falls below the pixel scale). Defaults to all
                annuli.
            sigma_map (Optional[list]): Per-annulus ``(sigma_r, sigma_phi)``
                1D uncertainty arrays used as residual weights, ordered to
                match the selected annuli. If ``None``, uses
                ``max(|S_2|, floor)`` since the scatter scales with ``S_2``.
            floor_frac (float): Floor for the default weighting, as a fraction
                of the ``S_2`` plateau.
            method ({'lsq', 'mcmc'}): Back-end selector.
            p0 (Optional[dict]): Override the heuristic starting guess for any
                of ``sigma``, ``alphar``, ``ell0r``, ``ell0phi``, ``alphaphi``
                (linear space). Keys left out are seeded from the scalar
                heuristics (plateau, half-power lags, reliability weights); see
                :meth:`_grf_initial_guess`.
            bounds (Optional[dict]): Hard bounds in linear space per key
                ``sigma``, ``alphar``, ``ell0r``, ``ell0phi``, ``alphaphi``,
                ``jitter``; each a ``(lo, hi)`` tuple. Constrain both back-ends
                (the ``lsq`` solve, bound-aware ``trf``, and the ``mcmc`` flat
                prior). Defaults: ``alphar``/``alphaphi`` in ``[-5, 5]`` and
                ``ell0r``/``ell0phi`` set per-fit to half the resolved radial /
                azimuthal-arc-length lag range (pooled over the selected
                annuli), so the lengths cannot run away past the scales ``S_2``
                constrains. Any key given here overrides the default.
            priors (Optional[dict]): Gaussian priors (``mcmc`` only) added on
                top of the flat bounds, keyed the same way; each a
                ``(mu, sigma)`` tuple in *sampling* space (natural-log for
                ``sigma``, ``ell0r``, ``ell0phi``, ``jitter``; linear for
                ``alphar``, ``alphaphi``). Missing keys stay flat. Intersected
                with the flat ``bounds``; samples outside the box get
                ``-inf`` posterior regardless of the Gaussian prior, so a
                prior that pulls walkers past a bound is effectively
                truncated there.
            jitter (bool): ``mcmc`` only. Add the log-scale nuisance parameter
                (recommended). Adds a trailing ``s`` column to ``samples``.
            fit_alphaphi (bool): If ``True``, free the azimuthal slope
                ``alphaphi`` as its own parameter (a radius-dependent
                anisotropy). Default ``False`` ties ``alphaphi = alphar``.
            pitch (bool): Must be ``False`` (the default); ``True`` raises
                because pitch is unmeasurable from the slice fit; see
                :meth:`StructureFunction2D.fit_GRF` for the global-surface
                fit that recovers it.
            nwalkers, nburnin, nsteps (int): ``emcee`` ensemble size and step
                counts (``mcmc`` only). ``nwalkers`` is raised to at least
                ``2 * ndim``.
            scatter (float): Fractional scatter of the initial walker ball
                about the least-squares solution.
            plots (Optional[list]): ``mcmc`` only. Diagnostic plots to draw,
                any of ``'walkers'``, ``'corner'``, ``'bestfit'`` (model vs.
                measured slices), or ``'none'``. Defaults to
                ``['walkers', 'corner']``.
            returns (Optional[list]): ``mcmc`` only. Items to return, any of
                ``'samples'`` (the flattened post-burn-in chain in *linear*
                space, columns aligned to the active keys ``[sigma, alphar,
                ell0r, ell0phi(, alphaphi)(, s)]``), ``'sampler'`` (the
                ``emcee.EnsembleSampler``), ``'lnprob'`` (the flattened
                log-probability), ``'percentiles'`` (the ``(3, ndim)`` 16/50/84
                array), ``'dict'`` (the median parameters keyed by name), or
                ``'none'``. Defaults to ``['samples']``, the full posterior,
                from which percentiles and medians can both be derived.
            pool: Optional worker pool passed to ``emcee.EnsembleSampler``.
            progress (bool): Show the ``emcee`` progress bar.

        Returns:
            For ``method='lsq'``: ``(params, perr, sol, cov)``, the best-fit and
            formal 1-sigma dicts keyed by the active sampled parameters
            (``sigma, alphar, ell0r, ell0phi`` and, if ``fit_alphaphi``,
            ``alphaphi``), the raw ``scipy.optimize.least_squares`` result, and
            the covariance in sampling space. When ``alphaphi`` is tied it is
            also added to ``params`` (equal to ``alphar``) so the dict is a
            complete field spec, but without a ``perr`` entry.

            For ``method='mcmc'``: whatever ``returns`` selects (a single
            object if only one item, else a list in the order above);
            ``None`` if ``returns=['none']``.
        """
        if pitch:
            raise ValueError(
                "pitch=True is unmeasurable from the stack's slice fit: the "
                "on-axis slices are symmetric under a pitch sign flip, so "
                "only the off-diagonal ridge of the global 2D surface "
                "preserves it. Build a global StructureFunction2D "
                "(ref_i=-1) and call StructureFunction2D.fit_GRF(pitch=True) "
                "instead.")

        parts, est, data_bounds, plot_bestfit = self._grf_setup(
            r0, r_range, sigma_map, floor_frac, p0)

        return _grf_fit_core(
            parts, est, data_bounds, pitch=False, fit_alphaphi=fit_alphaphi,
            method=method, bounds=bounds, priors=priors, jitter=jitter,
            nwalkers=nwalkers, nburnin=nburnin, nsteps=nsteps,
            scatter=scatter, plots=plots, returns=returns, pool=pool,
            progress=progress, plot_bestfit=plot_bestfit, user_p0=p0,
        )

    def _plot_grf_bestfit(self, idx, r0, mp, axes=None):
        """Overlay the model (lines) on the measured (points) radial and
        azimuthal ``S_2`` slices for the fitted annuli, the goodness-of-fit
        gate. ``mp`` is a linear model-parameter dict (``sigma, alphar, ell0r,
        ell0phi, alphaphi``; see :func:`_grf_unpack`)."""
        import matplotlib.pyplot as plt

        if axes is None:
            _, axes = plt.subplots(1, 2, figsize=(9.0, 3.6),
                                   constrained_layout=True)
        rr = self.ref_rs
        n = max(len(idx) - 1, 1)
        cmap = plt.get_cmap("viridis")
        for k, i in enumerate(idx):
            s = self.results[i]
            c = cmap(k / n)
            p = grf_s2_slices(rr[i], s.lags_x, s.lags_y, sigma=mp["sigma"],
                              alphar=mp["alphar"], ell0r=mp["ell0r"],
                              alphaphi=mp["alphaphi"], ell0phi=mp["ell0phi"],
                              r0=r0)
            # Hide empty (zero-count) lag bins, as the fit does.
            mx = s.counts[s.max_lag_x:, s.max_lag_y] > 0
            my = s.counts[s.max_lag_x, s.max_lag_y:] > 0
            axes[0].plot(s.lags_x[mx], s.S2_x[mx], ".", color=c, ms=3)
            axes[0].plot(s.lags_x[mx], p["S2_r"][mx], "-", color=c, lw=1.0)
            axes[1].plot(s.lags_y[my], s.S2_y[my], ".", color=c, ms=3)
            axes[1].plot(s.lags_y[my], p["S2_phi"][my], "-", color=c, lw=1.0,
                         label=r"$r={:.2f}$".format(rr[i]))
        axes[0].set(xlabel=r"$\ell_r$ (arcsec)", ylabel=r"$S_2$",
                    title="radial slice")
        axes[1].set(xlabel=r"$\Delta\phi$ (deg)", title="azimuthal slice")
        axes[1].legend(fontsize=6, ncol=2)
        return axes

    _NORMALIZE_LABELS = {
        None: r"$S_2$",
        "row_max": r"$S_2\,\, /\,\, \max_\ell(S_2)$",
    }

    @classmethod
    def _heatmap_normalize_label(cls, normalize):
        """Colorbar label for a given ``normalize`` choice. Raises on
        unknown values so the plot and calculate methods fail together."""
        if normalize not in cls._NORMALIZE_LABELS:
            raise ValueError(
                f"Unknown normalize={normalize!r}; expected one of "
                f"{list(cls._NORMALIZE_LABELS)}."
            )
        return cls._NORMALIZE_LABELS[normalize]

    @classmethod
    def _apply_heatmap_normalize(cls, C, normalize):
        """Apply a per-row normalization to a heatmap ``C`` of shape
        ``(N_ref, N_lag)``. Currently supported:

        * ``None``: no rescaling, raw ``S_2``.
        * ``'row_max'``: each row divided by its (nan-safe) max. Rows
          where the max is zero (e.g. unreachable in the bowtie
          envelope) stay zero.
        """
        # Validate via the label table so calculate_* and plot_* agree.
        cls._heatmap_normalize_label(normalize)
        if normalize is None:
            return C
        if normalize == "row_max":
            row_max = np.nanmax(C, axis=1, keepdims=True)
            safe = np.where(row_max > 0, row_max, 1.0)
            return np.where(row_max > 0, C / safe, 0.0)
        # Unreachable: the validation above already raised for unknowns.
        return C  # pragma: no cover

    def calculate_azimuthal_heatmap(self, arclength=False, normalize=None):
        """Return the ``(X, Y, C)`` arrays for the azimuthal-slice heatmap.

        Args:
            arclength (Optional[bool]): If ``True``, convert the
                azimuthal lag to arc length ``ref_r * dphi_rad`` in
                arcsec. Each row then has its own physical x-scale, so
                ``X`` and ``Y`` are returned as 2D arrays matching ``C``.
            normalize (Optional[str]): Per-row rescaling applied to
                ``C``. See :meth:`_apply_heatmap_normalize` for the
                supported values. Default ``None`` (raw ``S_2``).

        Returns:
            X (ndarray): Azimuthal lag axis. 1D ``(mly+1,)`` in degrees if
                ``arclength=False``; 2D ``(N_ref, mly+1)`` in arcsec if
                ``arclength=True``.
            Y (ndarray): Reference-radius axis. 1D ``(N_ref,)`` if
                ``arclength=False``; 2D ``(N_ref, mly+1)`` if
                ``arclength=True``.
            C (ndarray): ``S2_y_stack`` (possibly rescaled), shape
                ``(N_ref, mly+1)``.
        """
        C = self._apply_heatmap_normalize(self.S2_y_stack, normalize)
        if arclength:
            X = self.ref_rs[:, None] * np.radians(self.lags_y)[None, :]
            Y = np.broadcast_to(self.ref_rs[:, None], X.shape)
            return X, Y, C
        return self.lags_y, self.ref_rs, C

    def evaluate_spiral_heatmap(self, popt, modes, arclength=False,
                                normalize=None):
        """Evaluate a fitted spiral model on every ring's ``lags_y`` axis.

        Returns ``(X, Y, C)`` in the same shape as
        :meth:`calculate_azimuthal_heatmap`, so the result drops
        straight into the same ``pcolormesh`` call for a side-by-side
        data / model comparison.

        Typical use:

            popt, perr = stack.fit_spiral(modes=(1, 2, 3))
            X_d, Y_d, C_d = stack.calculate_azimuthal_heatmap()
            X_m, Y_m, C_m = stack.evaluate_spiral_heatmap(
                popt, modes=(1, 2, 3))
            # Compare C_d (data) and C_m (model) — same shape.

        Args:
            popt (ndarray): Per-ring best-fit parameters, shape
                ``(N_ref, 1 + len(modes))``. Column 0 is ``Nphi``;
                subsequent columns are the mode amplitudes in the same
                order as ``modes``. Usually comes from
                :meth:`fit_spiral`.
            modes (sequence of int): The mode list that produced
                ``popt``. Must match the column count of ``popt``.
            arclength (Optional[bool]): As in
                :meth:`calculate_azimuthal_heatmap`.
            normalize (Optional[str]): Per-row rescaling. Apply the
                **same** normalize choice to both data and model
                heatmaps when comparing.

        Returns:
            X, Y, C: same return contract as
                :meth:`calculate_azimuthal_heatmap`; ``C`` is the model
                evaluated at each ring's parameters, shape
                ``(N_ref, mly+1)``.
        """
        popt = np.asarray(popt, dtype=float)
        modes = tuple(int(m) for m in modes)
        expected = (len(self.results), 1 + len(modes))
        if popt.shape != expected:
            raise ValueError(
                f"popt has shape {popt.shape}; expected {expected} "
                f"for {len(modes)} modes and {len(self.results)} rings."
            )

        phi = self.lags_y
        model = _make_spiral_model(modes)
        C = np.stack([model(p, phi) for p in popt])
        C = self._apply_heatmap_normalize(C, normalize)

        if arclength:
            X = self.ref_rs[:, None] * np.radians(phi)[None, :]
            Y = np.broadcast_to(self.ref_rs[:, None], X.shape)
            return X, Y, C
        return phi, self.ref_rs, C

    def calculate_radial_heatmap(self, two_sided=None, normalize=None):
        """Return the ``(X, Y, C)`` arrays for the radial-slice heatmap.

        Args:
            two_sided (Optional[bool]): If ``None`` (default), pick
                based on :attr:`symmetrized` — positive-only when the
                result is symmetric (the negative half would just
                mirror), two-sided when it isn't (so the inward /
                outward asymmetry from ``symmetrize=False`` is visible).
                Override explicitly to force one shape or the other.
            normalize (Optional[str]): Per-row rescaling applied to
                ``C``. See :meth:`_apply_heatmap_normalize`.

        Returns:
            X (ndarray): Radial lag axis in arcsec — 1D ``(2*mlx+1,)``
                if ``two_sided``, else ``(mlx+1,)``.
            Y (ndarray): ``ref_rs``, 1D ``(N_ref,)`` in arcsec.
            C (ndarray): The radial-slice stack (possibly rescaled),
                shape ``(N_ref, len(X))``.
        """
        if two_sided is None:
            two_sided = not self.symmetrized
        raw = self.S2_x_full_stack if two_sided else self.S2_x_stack
        C = self._apply_heatmap_normalize(raw, normalize)
        X = self.lags_x_full if two_sided else self.lags_x
        return X, self.ref_rs, C

    def calculate_anisotropy_heatmap(self, lag_floor=None, two_sided=None,
                                     log=False):
        """Return ``(X, Y, C)`` for the ``S_2_phi / S_2_r`` anisotropy heatmap.

        Compares the azimuthal and radial structure functions at the same
        *arcsec* lag, per reference annulus. For an isotropic field the
        ratio is 1 at every lag; for an anisotropy
        ``A = ell_phi / ell_r`` it asymptotes to ``1/A^2`` at small lag
        (where both ``S_2`` grow as ``L^2/ell^2``) and to 1 at large lag
        (both reach the ``2 sigma^2`` plateau). So the small-lag value of
        the ratio is a direct, quantitative anisotropy estimator (a
        ratio < 1 means azimuthally-elongated structure, > 1 means
        radially-elongated).

        The azimuthal slice is evaluated as an arclength (``r_ref * dphi``)
        and interpolated, per ``ref_r``, onto the radial lag axis so the
        two ``S_2`` are compared at the same physical lag.

        Args:
            lag_floor (Optional[float]): Mask bins with ``|L| <= lag_floor``
                to avoid the 0/0 column at zero lag (both ``S_2`` vanish
                there). Defaults to half a radial bin.
            two_sided (Optional[bool]): As in
                :meth:`calculate_radial_heatmap`. ``None`` (default)
                follows :attr:`symmetrized`: one-sided for symmetric
                results, two-sided otherwise. In the two-sided case the
                azimuthal slice is evaluated at ``|L|`` so both signs of
                ``dr`` share the same azimuthal normalization; any L/R
                asymmetry in the ratio then comes from the radial
                outward/inward statistics (i.e. non-stationarity).
            log (bool): If ``True``, return ``log10`` of the ratio instead
                of the ratio itself (the name is ``log`` for brevity, but it
                is base-10, not natural). Since the anisotropy is
                multiplicative (``1/A^2`` at small lag, ``1`` at large lag),
                the log is symmetric about ``0`` (isotropic), with negative
                values for azimuthally-elongated and positive for
                radially-elongated structure. Non-positive ratios map to
                ``np.nan``.

        Returns:
            X (ndarray): Radial lag axis [arcsec] — 1D ``(2*mlx+1,)``
                if ``two_sided``, else ``(mlx+1,)``.
            Y (ndarray): ``ref_rs``, 1D ``(N_ref,)`` in arcsec.
            C (ndarray): The anisotropy ratio (or its ``log10`` if
                ``log=True``), shape ``(N_ref, len(X))``, with ``np.nan``
                where ``|L| <= lag_floor``.
        """
        if two_sided is None:
            two_sided = not self.symmetrized

        Xa, _, Ca = self.calculate_azimuthal_heatmap(arclength=True)
        Xr, _, Cr = self.calculate_radial_heatmap(two_sided=two_sided)

        # Azimuthal S2 is intrinsically positive-lag; evaluate at |L| so
        # both signs of dr (when two-sided) share the same azimuthal
        # normalization. Out-of-range -> NaN.
        Ca_on_Xr = np.array([np.interp(np.abs(Xr), Xa[i], Ca[i],
                                       left=np.nan, right=np.nan)
                             for i in range(Ca.shape[0])])

        if lag_floor is None:
            lag_floor = 0.5 * float(np.median(np.diff(Xr)))
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(np.abs(Xr)[None, :] > lag_floor,
                             Ca_on_Xr / Cr, np.nan)
            if log:
                ratio = np.where(ratio > 0, np.log10(ratio), np.nan)
        return Xr, self.ref_rs, ratio

    def pairwise_error_heatmaps(self, two_sided=None, arclength=False):
        """Per-cell 1-sigma uncertainty on the radial and azimuthal ``S_2``
        heatmaps from Gaussian pair statistics: ``sigma = S2 * sqrt(2 / N_pairs)``.

        For a Gaussian field the increment at lag ``l`` is ``N(0, S2(l))``,
        so each squared increment is ``S2 * chi^2_1`` (variance ``2 S2^2``)
        and the pair-count average has variance ``2 S2^2 / N_pairs``. The
        returned arrays share the ``(X, Y)`` axes of
        :meth:`calculate_radial_heatmap` / :meth:`calculate_azimuthal_heatmap`
        (with ``normalize=None``), so they drop onto the same ``pcolormesh``
        as an error layer.

        .. warning::
            This is an **optimistic lower bound**. It assumes the pairs are
            independent, but real pairs share endpoints and, more
            importantly, the field is correlated, so the effective number
            of independent samples is ``N_eff << N_pairs``. The true scatter
            is larger by roughly ``sqrt(N_pairs / N_eff)``: negligible when
            the correlation length is near the pixel scale, but an order of
            magnitude or more once it spans many pixels. For an honest
            uncertainty, use the per-cell standard deviation across many
            realizations (Monte Carlo).

        Args:
            two_sided (Optional[bool]): Controls the radial axis, as in
                :meth:`calculate_radial_heatmap`. ``None`` (default) follows
                :attr:`symmetrized`.
            arclength (Optional[bool]): Controls the azimuthal axis, as in
                :meth:`calculate_azimuthal_heatmap`.

        Returns:
            dict with keys ``'radial'`` and ``'azimuthal'``, each an
            ``(X, Y, sigma)`` tuple matching the corresponding
            ``calculate_*_heatmap`` return (``normalize=None``).
        """
        if two_sided is None:
            two_sided = not self.symmetrized

        Xr, Yr, S2r = self.calculate_radial_heatmap(two_sided=two_sided,
                                                     normalize=None)
        Nr = self.counts_x_full_stack if two_sided else self.counts_x_stack
        sigma_r = np.abs(S2r) * np.sqrt(2.0 / np.maximum(Nr, 1.0))

        Xa, Ya, S2a = self.calculate_azimuthal_heatmap(arclength=arclength,
                                                        normalize=None)
        sigma_a = np.abs(S2a) * np.sqrt(2.0 / np.maximum(self.counts_y_stack, 1.0))

        return {"radial": (Xr, Yr, sigma_r),
                "azimuthal": (Xa, Ya, sigma_a)}

    def plot_azimuthal_heatmap(self, ax=None, return_fig=False,
                               arclength=False, normalize=None,
                               **pcolormesh_kwargs):
        """Heatmap of ``S_2_y`` vs ``(ref_r, dphi)`` — the canonical
        figure for radius-resolved spiral analysis. Thin wrapper around
        :meth:`calculate_azimuthal_heatmap`.

        Args:
            arclength (Optional[bool]): See
                :meth:`calculate_azimuthal_heatmap`.
            normalize (Optional[str]): See
                :meth:`calculate_azimuthal_heatmap`. Also sets the
                colorbar label.
        """
        X, Y, C = self.calculate_azimuthal_heatmap(arclength=arclength,
                                                   normalize=normalize)
        xlabel = (r"$\ell_\phi$ (arcsec)" if arclength
                  else r"$\ell_\phi$ (deg)")
        return _plot_heatmap(ax, X, Y, C, xlabel,
                             self._heatmap_normalize_label(normalize),
                             return_fig, **pcolormesh_kwargs)

    def plot_radial_heatmap(self, ax=None, return_fig=False,
                            two_sided=None, normalize=None,
                            **pcolormesh_kwargs):
        """Heatmap of ``S_2_x`` vs ``(ref_r, radial_lag)``. Thin wrapper
        around :meth:`calculate_radial_heatmap`. Each row is the radial-
        lag slice (``ell_phi = 0``) at one reference radius.

        Args:
            two_sided (Optional[bool]): See
                :meth:`calculate_radial_heatmap`. ``None`` (default)
                picks positive-only for symmetric results and two-sided
                otherwise.
            normalize (Optional[str]): See
                :meth:`calculate_radial_heatmap`. Also sets the colorbar
                label.
        """
        X, Y, C = self.calculate_radial_heatmap(two_sided=two_sided,
                                                normalize=normalize)
        return _plot_heatmap(ax, X, Y, C, r"$\ell_r$ (arcsec)",
                             self._heatmap_normalize_label(normalize),
                             return_fig, **pcolormesh_kwargs)

    def plot_anisotropy_heatmap(self, ax=None, return_fig=False,
                                lag_floor=None, two_sided=None, log=False,
                                **pcolormesh_kwargs):
        """Heatmap of ``S_2_phi / S_2_r`` vs ``(ref_r, lag)``. Thin
        wrapper around :meth:`calculate_anisotropy_heatmap`.

        A direct anisotropy diagnostic: ratio < 1 indicates
        azimuthally-elongated structure (``ell_phi > ell_r``), > 1
        radially-elongated. The small-lag value asymptotes to ``1/A^2``.

        Args:
            lag_floor (Optional[float]): See
                :meth:`calculate_anisotropy_heatmap`.
            two_sided (Optional[bool]): See
                :meth:`calculate_anisotropy_heatmap`.
            log (bool): Plot ``log10`` of the ratio (symmetric about 0,
                isotropic). See :meth:`calculate_anisotropy_heatmap`.
        """
        X, Y, C = self.calculate_anisotropy_heatmap(lag_floor=lag_floor,
                                                    two_sided=two_sided,
                                                    log=log)
        cbar_label = (r"$\log_{10}(S_2^\phi / S_2^r)$" if log
                      else r"$S_2^\phi / S_2^r$")
        return _plot_heatmap(ax, X, Y, C, r"$\ell$ (arcsec)",
                             cbar_label, return_fig, **pcolormesh_kwargs)

    def plot_gridded(self, ax=None, return_fig=False,
                     azimuth_in_degrees=True, show_rings=False,
                     ring_kwargs=None, center=0.0,
                     **pcolormesh_kwargs):
        """Plot the polar-deprojected field that the stack was computed from.

        Useful as a sanity-check companion to the heatmaps: lets you see
        whether the structure picked up at a given ``ref_r`` corresponds
        to a visible feature in the source data.

        By default uses the eddy ``imagecube.cmap()`` (the diverging
        blue-white-red map shared with :meth:`rotationmap.plot_data`)
        and a symmetric color scale around ``center`` — matching the
        rotation-map convention so a velocity residual reads naturally.

        Args:
            ax (Optional): Matplotlib ``Axes`` to draw into.
            return_fig (Optional[bool]): Return the figure.
            azimuth_in_degrees (Optional[bool]): Convert the azimuth axis
                from radians (the ``polar_deprojection`` convention) to
                degrees for display. Default ``True``.
            show_rings (Optional[bool]): Overlay the reference annuli
                from :attr:`ref_rs` as horizontal lines.
            ring_kwargs (Optional[dict]): Kwargs for the ring overlay
                (forwarded to ``ax.axhline``). Defaults to a thin white
                semi-transparent line.
            center (Optional[float]): Centre value for the symmetric
                color scale. Default ``0.0``. Pass ``None`` to fall
                back to matplotlib's auto-scaling. ``vmin``/``vmax`` in
                ``pcolormesh_kwargs`` override this entirely.
            **pcolormesh_kwargs: Forwarded to ``ax.pcolormesh``. Override
                ``cmap`` here if you don't want the eddy diverging map
                (e.g. ``cmap='viridis'`` for an intensity field).

        Raises:
            ValueError: If ``self.gridded`` was not stored on the stack
                (e.g. constructed without the polar grid).
        """
        if self.gridded is None or self.x_grid is None or self.y_grid is None:
            raise ValueError(
                "plot_gridded requires gridded/x_grid/y_grid on the stack; "
                "the stack was constructed without them."
            )

        fig, ax = _resolve_ax(ax)

        phi = (np.degrees(self.y_grid) if azimuth_in_degrees
               else self.y_grid)

        # Defaults — only applied where the user hasn't overridden.
        plot_kwargs = dict(shading="auto", rasterized=True)
        if "cmap" not in pcolormesh_kwargs:
            from .imagecube import imagecube
            plot_kwargs["cmap"] = imagecube.cmap()
        if (center is not None
                and "vmin" not in pcolormesh_kwargs
                and "vmax" not in pcolormesh_kwargs):
            lo, hi = np.nanpercentile(self.gridded, [2, 98])
            half = max(abs(lo - center), abs(hi - center))
            plot_kwargs["vmin"] = center - half
            plot_kwargs["vmax"] = center + half
        plot_kwargs.update(pcolormesh_kwargs)

        pcm = ax.pcolormesh(phi, self.x_grid, self.gridded, **plot_kwargs)
        ax.set_xlabel("azimuth [deg]" if azimuth_in_degrees
                      else "azimuth [rad]")
        ax.set_ylabel("radius [arcsec]")
        fig.colorbar(pcm, ax=ax)

        if show_rings:
            rk = dict(color="white", lw=0.5, alpha=0.6)
            if ring_kwargs:
                rk.update(ring_kwargs)
            for r in self.ref_rs:
                ax.axhline(r, **rk)

        return fig if return_fig else None
