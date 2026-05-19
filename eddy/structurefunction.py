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
    "S2phi",
    "S2phi_singlemodel",
]


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
        ref_i (int): Row index of the reference annulus centre. If < 0
            (default), averages over all valid base rows (global mode).
            If >= 0, restricts base rows to ``[ref_i - ref_band,
            ref_i + ref_band]``.
        ref_band (int): Half-width in rows of the reference annulus.
            ``0`` selects a single row.
        symmetrize (bool): Only relevant when ``ref_i >= 0``. If
            ``True`` (default), the outward (``+l_r``) and inward
            (``-l_r``) halves are combined by a pair-count-weighted
            average — i.e. ``S_2`` becomes a direction-agnostic
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

    Args:
        S2 (ndarray): 2D structure function from :func:`compute_s2`.
        max_lag_x, max_lag_y (int): Maximum lags used to build ``S2``.
        dx, dy (float): Physical pixel spacing along axis 0/1.
        n_bins (int): Number of radial bins for the azimuthal average.
        log_spaced (bool): If ``True``, log-spaced radial bins.

    Returns:
        lags_x (ndarray): Positive lags along axis 0 in physical units.
        lags_y (ndarray): Positive lags along axis 1 in physical units.
        lags_i (ndarray): Bin centres for the azimuthally averaged profile.
        S2_x (ndarray): ``S_2`` slice along axis 0 (at ``l_y = 0``).
        S2_y (ndarray): ``S_2`` slice along axis 1 (at ``l_x = 0``).
        S2_i (ndarray): Azimuthally averaged ``S_2(|l|)``.
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
    # lattice artefacts that make hard-binned azimuthal averages jagged.
    interp = RegularGridInterpolator(
        (lag_x_arr, lag_y_arr), S2,
        method="linear", bounds_error=False, fill_value=np.nan,
    )
    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    cos_a, sin_a = np.cos(angles), np.sin(angles)

    S2_i = np.full(n_bins, np.nan)
    for i, r in enumerate(ell_centers):
        pts = np.column_stack([r * cos_a, r * sin_a])
        vals = interp(pts)
        if np.any(np.isfinite(vals)):
            S2_i[i] = np.nanmean(vals)

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
        lags_i (ndarray): Radial bin centres for the azimuthal average.
        S2_x, S2_y, S2_i (ndarray): 1D profiles along axis 0, axis 1,
            and azimuthally averaged.
        x_grid, y_grid (Optional[ndarray]): Underlying grid axes of the
            field (e.g. radial / azimuthal grid for a polar
            deprojection). ``None`` when constructed from a bare array.
        gridded (Optional[ndarray]): The 2D field that ``S_2`` was
            computed from.
        ref, ref_band: Reference-annulus centre / band in physical units
            (e.g. arcsec). ``None`` if global.
        x_label, y_label (str): Labels for the two lag axes; used by the
            plotting helpers.
        azimuthal_axis (Optional[str]): ``'y'`` if axis 1 corresponds
            to an angular coordinate (e.g. azimuth in degrees), in which
            case :meth:`fit_spiral` defaults to that axis. ``None``
            otherwise.
    """

    def __init__(self, *, S2, counts, dx, dy, lags_x, lags_y, lags_i,
                 S2_x, S2_y, S2_i, x_grid=None, y_grid=None,
                 gridded=None, ref=None, ref_band=None,
                 x_label="lag_x", y_label="lag_y", azimuthal_axis=None,
                 symmetrized=True):
        self.S2 = np.asarray(S2)
        self.counts = np.asarray(counts)
        self.dx = float(dx)
        self.dy = float(dy)
        self.lags_x = np.asarray(lags_x)
        self.lags_y = np.asarray(lags_y)
        self.lags_i = np.asarray(lags_i)
        self.S2_x = np.asarray(S2_x)
        self.S2_y = np.asarray(S2_y)
        self.S2_i = np.asarray(S2_i)
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

        if any(r.S2.shape != self.S2.shape for r in all_results):
            raise ValueError("All results must share the same S2 shape.")
        if any(r.dx != self.dx or r.dy != self.dy for r in all_results):
            raise ValueError("All results must share the same dx, dy.")

        S2_list = [r.S2 for r in all_results]
        counts_list = [r.counts for r in all_results]
        S2_comb, S2_err, S2_std = combine_s2_weighted(S2_list, counts_list)
        counts_comb = np.sum(counts_list, axis=0)

        lags_x, lags_y, lags_i, S2_x, S2_y, S2_i = extract_basic_profiles(
            S2_comb, self.max_lag_x, self.max_lag_y, dx=self.dx, dy=self.dy,
            n_bins=n_bins, log_spaced=log_spaced,
        )
        combined = type(self)(
            S2=S2_comb, counts=counts_comb, dx=self.dx, dy=self.dy,
            lags_x=lags_x, lags_y=lags_y, lags_i=lags_i,
            S2_x=S2_x, S2_y=S2_y, S2_i=S2_i,
            x_grid=self.x_grid, y_grid=self.y_grid,
            ref=self.ref, ref_band=self.ref_band,
            x_label=self.x_label, y_label=self.y_label,
            azimuthal_axis=self.azimuthal_axis,
        )
        combined.combined_error = S2_err
        combined.combined_std = S2_std
        return combined

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
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        kwargs = dict(origin="lower", aspect="auto", extent=self.extent)
        kwargs.update(imshow_kwargs)
        im = ax.imshow(self.S2, **kwargs)
        ax.set_xlabel(self.y_label)
        ax.set_ylabel(self.x_label)
        fig.colorbar(im, ax=ax, label=r"$S_2$")
        return fig if return_fig else None

    def plot_profiles(self, ax=None, return_fig=False):
        """Plot ``S_2_x``, ``S_2_y`` and the azimuthal average on one axes."""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        ax.plot(self.lags_x, self.S2_x, label=self.x_label)
        ax.plot(self.lags_y, self.S2_y, label=self.y_label)
        ax.plot(self.lags_i, self.S2_i, label="azimuthal average", ls="--")
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
        ``(N_ref, n_bins)``."""
        return np.stack([r.S2_i for r in self.results])

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
        """
        popts, perrs = [], []
        for r in self.results:
            popt, perr, _ = r.fit_spiral(modes=modes, axis=axis, p0=p0)
            popts.append(popt)
            perrs.append(perr)
        return np.asarray(popts), np.asarray(perrs)

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

        * ``None`` -- no rescaling, raw ``S_2``.
        * ``'row_max'`` -- each row divided by its (nan-safe) max. Rows
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
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        X, Y, C = self.calculate_azimuthal_heatmap(arclength=arclength,
                                                   normalize=normalize)

        kwargs = dict(shading="auto", rasterized=True)
        kwargs.update(pcolormesh_kwargs)
        pcm = ax.pcolormesh(X, Y, C, **kwargs)
        ax.set_xlabel(r"$\ell_\phi$ (arcsec)" if arclength
                      else r"$\ell_\phi$ (deg)")
        ax.set_ylabel(r"$r_{\rm ref}$ (arcsec)")
        cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
        cbar.ax.set_ylabel(self._heatmap_normalize_label(normalize),
                        rotation=270, labelpad=13)
        return fig if return_fig else None

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
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        X, Y, C = self.calculate_radial_heatmap(two_sided=two_sided,
                                                normalize=normalize)

        kwargs = dict(shading="auto", rasterized=True)
        kwargs.update(pcolormesh_kwargs)
        pcm = ax.pcolormesh(X, Y, C, **kwargs)
        ax.set_xlabel(r"$\ell_r$ (arcsec)")
        ax.set_ylabel(r"$r_{\rm ref}$ (arcsec)")
        cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
        cbar.ax.set_ylabel(self._heatmap_normalize_label(normalize),
                        rotation=270, labelpad=13)
        return fig if return_fig else None

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
        and a symmetric colour scale around ``center`` — matching the
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
                colour scale. Default ``0.0``. Pass ``None`` to fall
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

        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

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
