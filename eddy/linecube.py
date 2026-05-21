# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from .imagecube import imagecube
from .annulus import annulus


class SpectralACF:
    """Pooled spectral auto-correlation function from a signal-free
    region of a cube. Built by :meth:`linecube.spectral_acf`.

    Under the null hypothesis that channels are statistically
    independent, Bartlett's formula gives ``Var(rho_k) ~ 1/N`` for
    ``k > 0``, where ``N`` is the number of independent sample pairs.
    Here ``N = counts / pix_per_beam`` to fold in the spatial
    correlation between pixels in the same beam.

    Attributes:
        lags (ndarray): Spectral lag in channels, ``0..max_lag``.
        acf (ndarray): Pooled (Pearson) ACF averaged across pixels.
            ``acf[0]`` is 1 by construction.
        counts (ndarray): Number of ``(pixel, channel-pair)`` samples
            contributing to each lag.
        null_band (ndarray): Half-width of the 95% confidence interval
            on ``acf[k]`` under the null hypothesis of independent
            channels (``1.96 / sqrt(counts / pix_per_beam)``).
        n_eff (ndarray): Effective independent sample count per lag,
            ``counts / pix_per_beam``.
        pix_per_beam (float): Inherited from the parent cube.
        n_pixels (int): Number of spatial pixels in the annulus.
        n_channels (int): Number of unmasked channels used.
        chan_width (float): Channel width in the cube's velocity units.
    """

    def __init__(self, *, lags, acf, counts, null_band, n_eff,
                 pix_per_beam, n_pixels, n_channels, chan_width):
        self.lags = np.asarray(lags)
        self.acf = np.asarray(acf)
        self.counts = np.asarray(counts)
        self.null_band = np.asarray(null_band)
        self.n_eff = np.asarray(n_eff)
        self.pix_per_beam = float(pix_per_beam)
        self.n_pixels = int(n_pixels)
        self.n_channels = int(n_channels)
        self.chan_width = float(chan_width)

    def significant_lags(self, alpha=0.05):
        """Lags (excluding ``k=0``) at which ``|acf|`` exceeds the
        null band scaled to confidence level ``1 - alpha``.

        Args:
            alpha (float): Two-sided significance level. ``0.05`` (the
                default) reproduces the 95% null band on the plot.

        Returns:
            ndarray of lag values where the null is rejected.
        """
        from scipy.stats import norm
        z = norm.ppf(1.0 - 0.5 * float(alpha))
        band = z / np.sqrt(np.maximum(self.n_eff, 1.0))
        sig = (self.lags > 0) & (np.abs(self.acf) > band)
        return self.lags[sig]

    def plot(self, ax=None, x_unit='channels', skip_zero=True,
             return_fig=False):
        """Plot the ACF with the 95% null band shaded.

        Args:
            ax: Existing matplotlib axis. Created if not supplied.
            x_unit ({'channels', 'velocity'}): Lag axis units.
                ``'velocity'`` multiplies by ``chan_width``.
            skip_zero (bool): Hide ``k=0`` from the plot (it is
                always 1 by construction and obscures the y-range).
            return_fig (bool): If ``True``, return the matplotlib
                figure rather than just the axis.

        Returns:
            ``fig`` if ``return_fig``, else ``ax``.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        sl = slice(1, None) if skip_zero else slice(None)
        lags = self.lags[sl]
        acf = self.acf[sl]
        band = self.null_band[sl]

        if x_unit == 'velocity':
            x = lags * self.chan_width
            xlabel = 'Spectral lag'
        else:
            x = lags
            xlabel = 'Spectral lag [channels]'

        ax.axhline(0.0, color='0.5', lw=0.8)
        ax.fill_between(x, -band, band, color='C0', alpha=0.2,
                        label='95% null band')
        ax.plot(x, acf, color='C0', marker='o', ms=4, lw=1.2,
                label='Empirical ACF')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(r'$\hat{\rho}(\tau)$')
        ax.legend(frameon=False)
        return fig if return_fig else ax


class linecube(imagecube):
    """
    Read in a line cube and initialize the class.

    Args:
        path (str): Relative path to the rotation map you want to fit.
        FOV (Optional[float]): If specified, clip the data down to a
            square field of view with sides of `FOV` [arcsec].
        fill (Optional[float/None]): Value to fill any NaN values.
        velocity_range (Optional[list]): A velocity range in [m/s] to
            clip the data down to: ``[min_velo, max_velo]``.
    """

    def __init__(self, path, FOV=None, fill=0.0, velocity_range=None):
        super().__init__(path=path, FOV=FOV, fill=fill)
        if velocity_range is not None:
            self._clip_cube_velocity(*velocity_range)

    # -- 3D CUBE I/O & DIAGNOSTICS -- #

    def to_momentmap(self, method='zeroth', product=None, clip=None,
                     bettermoments_kwargs=None):
        """Collapse the spectral cube to a 2D moment map via
        ``bettermoments``.

        Args:
            method (str): Name of the bettermoments collapse method,
                without the ``collapse_`` prefix (e.g. ``'zeroth'``,
                ``'first'``, ``'quadratic'``, ``'maximum'``,
                ``'gaussian'``). See :mod:`bettermoments` for the full
                list and what each one returns.
            product (Optional[str]): Suffix of the specific moment
                product to return, as named by ``bettermoments`` (e.g.
                ``'M0'``, ``'v0'``, ``'wp50'``, ``'gv0'``). If
                ``None`` (default), the first product of the method is
                used — matching the historical behaviour. Uncertainty
                suffixes (``'d…'``) are rejected; the matching
                uncertainty array is attached as ``.error`` instead.
            clip (Optional[float]): If set, replace pixels with
                ``|data| < clip * rms`` by zero before collapsing —
                matches ``bettermoments``'s sigma-clipping convention.
                Set to ``None`` (the default) to skip clipping.
            bettermoments_kwargs (Optional[dict]): Extra kwargs forwarded
                to the ``bettermoments`` collapse function. Most simple
                methods (``zeroth``, ``first``, ``quadratic``,
                ``maximum``, ...) accept no extra kwargs; the
                fitting-based methods (``gaussian``, ``gausshermite``,
                ...) accept ``indices`` and ``ncpu``.

        Returns:
            A :class:`rotationmap` instance when the selected product is
            a velocity field (``bettermoments`` unit ``'m/s'``) and a
            :class:`momentmap` instance otherwise. The matching
            uncertainty product (``'d' + product``, if present) is
            attached as ``self.error`` on the returned object.
        """
        try:
            import bettermoments as bm
        except ImportError as exc:
            raise ImportError(
                "linecube.to_momentmap() requires the bettermoments "
                "package. Install with `pip install bettermoments`."
            ) from exc

        method_key = method.lower()
        collapse = getattr(bm, 'collapse_{}'.format(method_key), None)
        if collapse is None or not callable(collapse):
            raise ValueError(
                "Unknown bettermoments method '{}'. See "
                "bettermoments.methods for valid names "
                "(e.g. 'zeroth', 'first', 'quadratic', 'maximum')."
                .format(method)
            )

        # Canonical ordered list of product suffixes for this method,
        # e.g. ``['v0', 'dv0', 'Fnu', 'dFnu']`` for ``quadratic``.
        bm_products = [s.strip() for s in
                       bm.methods.collapse_method_products(method_key)
                       .split(',')]

        if product is None:
            product = bm_products[0]
        elif product not in bm_products:
            raise ValueError(
                "Unknown product '{}' for method '{}'. Valid suffixes: {}."
                .format(product, method_key, ', '.join(bm_products))
            )
        elif product.startswith('d') and product[1:] in bm_products:
            raise ValueError(
                "'{}' is an uncertainty product; pass product='{}' "
                "instead — the uncertainty is attached as .error."
                .format(product, product[1:])
            )

        idx = bm_products.index(product)
        err_key = 'd' + product
        err_idx = bm_products.index(err_key) if err_key in bm_products else None

        rms = float(self.estimate_cube_RMS())
        data = np.asarray(self.data)
        if clip is not None:
            data = np.where(np.abs(data) > float(clip) * rms, data, 0.0)

        bm_kwargs = ({} if bettermoments_kwargs is None
                     else dict(bettermoments_kwargs))
        products = collapse(velax=np.asarray(self.velax),
                            data=data, rms=rms, **bm_kwargs)
        # Normalise the return to a tuple of 2D arrays. Most collapse
        # functions return a tuple; ``collapse_quadratic`` (and a couple
        # of others) instead return a single stacked 3D ndarray whose
        # leading axis enumerates the products.
        if isinstance(products, np.ndarray):
            if products.ndim == 3:
                products = tuple(products[i] for i in range(products.shape[0]))
            else:
                products = (products,)
        moment = np.asarray(products[idx])

        # Use bettermoments' canonical unit string for the chosen
        # product — ``'m/s'`` flags velocity fields (→ rotationmap);
        # everything else is intensity-typed (→ momentmap).
        bunit = bm.io._get_bunits(self.path)[product]
        if bunit == 'm/s':
            from .rotationmap import rotationmap
            out_cls = rotationmap
        else:
            from .momentmap import momentmap
            out_cls = momentmap

        # Round-trip through a temp FITS so the returned instance picks
        # up a consistent header rebuilt from the live xaxis/yaxis. This
        # avoids reimplementing all the per-attribute init logic
        # (beam parsing, axis flips, restfreq, ...) for an in-memory
        # construction path.
        import contextlib
        import io
        import tempfile
        from astropy.io import fits

        header = self._consistent_header(moment)
        header['BUNIT'] = bunit

        with tempfile.NamedTemporaryFile(suffix='.fits',
                                         delete=False) as tmp:
            tmp_path = tmp.name
        try:
            fits.writeto(tmp_path, data=moment,
                         header=header, overwrite=True)
            # Suppress rotationmap.__init__'s "Assuming uncertainties
            # in /tmp/...fits" prints — the temp path has no sibling
            # uncertainty file, and we attach the proper d<product> below.
            with contextlib.redirect_stdout(io.StringIO()):
                out = out_cls(path=tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        if err_idx is not None and err_idx < len(products) \
                and products[err_idx] is not None:
            out.error = np.asarray(products[err_idx])

        return out

    @property
    def rms(self):
        return self.estimate_cube_RMS()

    def _clip_cube_velocity(self, v_min=None, v_max=None):
        """Clip the cube to within ``vmin`` and ``vmax``."""
        if self.data.ndim == 2:
            raise ValueError("Attaced cube has no velocity axis.")
        v_min = self.velax[0] if v_min is None else v_min
        v_max = self.velax[-1] if v_max is None else v_max
        i = abs(self.velax - v_min).argmin()
        i += 1 if self.velax[i] < v_min else 0
        j = abs(self.velax - v_max).argmin()
        j -= 1 if self.velax[j] > v_max else 0
        self.velax = self.velax[i:j+1]
        self.data = self.data[i:j+1]

    def estimate_cube_RMS(self, N=10, r_in=0.0, r_out=1e10):
        """
        Estimate RMS of the cube based on first and last `N` channels and a
        circular area described by an inner and outer radius.

        Args:
            N (int): Number of edge channels to include.
            r_in (float): Inner edge of pixels to consider in [arcsec].
            r_out (float): Outer edge of pixels to consider in [arcsec].

        Returns:
            RMS (float): The RMS based on the requested pixel range.
        """
        r_dep = np.hypot(self.xaxis[None, :], self.yaxis[:, None])
        rmask = np.logical_and(r_dep >= r_in, r_dep <= r_out)
        rms = np.concatenate([self.data[:int(N)], self.data[-int(N):]])
        rms = np.where(rmask[None, :, :], rms, np.nan)
        return np.sqrt(np.nansum(rms**2) / np.sum(np.isfinite(rms)))

    def gaussian_beam_s2(self, lags_x, lags_y=None, sigma2=None,
                         counts=None, N=None, r_in=None, r_out=None,
                         n_bins=50, log_spaced=False,
                         x_label="lag_x", y_label="lag_y"):
        """Analytic Gaussian-beam noise ``S_2`` prediction for this cube.

        Convenience wrapper around
        :func:`eddy.structurefunction.gaussian_beam_s2` that fills the
        beam parameters from the cube header (``bmaj``, ``bmin``,
        ``bpa``) and the per-pixel noise variance from
        :meth:`estimate_cube_RMS`.

        The ergonomic path is to pass the empirical
        :class:`StructureFunction2D` directly -- the lag grid, pair
        counts, and (when available) the ``N`` / ``r_in`` / ``r_out``
        mask used to build the empirical are inherited automatically,
        so the two ``S_2`` instances share the same ``sigma2`` region:

            emp = cube.noise_structure_function(r_in=1.5, r_out=4.0)
            ana = cube.gaussian_beam_s2(emp)  # inherits r_in/r_out/N
            emp.plot_comparison(ana)

        Explicit ``N`` / ``r_in`` / ``r_out`` always override the
        inherited values. Without an empirical and without explicit
        overrides, ``N=10``, ``r_in=0.0``, ``r_out=1e10`` are used.

        Args:
            lags_x: Either a 1D array of positive lags along axis 0,
                or a :class:`StructureFunction2D` -- in which case
                ``lags_x``, ``lags_y``, and ``counts`` are taken from
                that instance.
            lags_y (Optional[ndarray]): 1D positive lags along axis 1.
                Ignored if ``lags_x`` is a ``StructureFunction2D``.
            sigma2 (Optional[float]): Per-pixel noise variance. If
                ``None``, defaults to
                ``estimate_cube_RMS(N=N, r_in=r_in, r_out=r_out)**2``.
                Pass the same ``r_in`` / ``r_out`` you used for the
                empirical ``S_2``; otherwise the two compare against
                inconsistent ``sigma^2`` plateaus and the "excess
                structure" residual is dominated by that mismatch.
            counts (Optional[ndarray]): Pair counts to attach to the
                returned result. Defaults to the empirical's counts if
                ``lags_x`` is a ``StructureFunction2D``, else ones.
            N (int): Edge-channel count for the default RMS estimate.
            r_in, r_out (float): Annulus [arcsec] for the default RMS
                estimate.
            n_bins, log_spaced: Forwarded to
                :func:`extract_basic_profiles` for the 1D profiles.
            x_label, y_label (str): Lag-axis labels.

        Returns:
            :class:`eddy.structurefunction.StructureFunction2D`
        """
        from .structurefunction import (
            StructureFunction2D,
            gaussian_beam_s2 as _gaussian_beam_s2,
        )

        if isinstance(lags_x, StructureFunction2D):
            match = lags_x
            lags_x = match.lags_x
            if lags_y is None:
                lags_y = match.lags_y
            if counts is None:
                counts = match.counts
            nm = getattr(match, "noise_mask", None)
            if nm is not None:
                if N is None:
                    N = nm.get("N")
                if r_in is None:
                    r_in = nm.get("r_in")
                if r_out is None:
                    r_out = nm.get("r_out")
        elif lags_y is None:
            raise ValueError(
                "lags_y is required unless lags_x is a StructureFunction2D."
            )

        if N is None:
            N = 10
        if r_in is None:
            r_in = 0.0
        if r_out is None:
            r_out = 1e10

        if sigma2 is None:
            sigma2 = float(
                self.estimate_cube_RMS(N=N, r_in=r_in, r_out=r_out)
            ) ** 2

        return _gaussian_beam_s2(
            self.bmaj, self.bmin, self.bpa,
            lags_x, lags_y, sigma2=sigma2, counts=counts,
            n_bins=n_bins, log_spaced=log_spaced,
            x_label=x_label, y_label=y_label,
        )

    def noise_structure_function(self, channels=None, N=10,
                                 r_in=0.0, r_out=1e10,
                                 max_lag_x=None, max_lag_y=None,
                                 n_bins=50, log_spaced=False,
                                 return_per_channel=False,
                                 symmetrize=True):
        """Empirical 2D structure function of the noise channels.

        Each requested channel is masked to a circular annulus
        ``r_in <= r <= r_out`` (matching :meth:`estimate_cube_RMS`'s
        convention -- use ``r_in > 0`` to exclude residual emission in
        the centre, ``r_out`` to exclude noisy edges), passed to
        :meth:`eddy.structurefunction.StructureFunction2D.from_array`,
        and the per-channel results are combined via pair-count-weighted
        averaging.

        To compare against the naive Gaussian-beam prediction, pair this
        with :func:`eddy.structurefunction.gaussian_beam_s2` evaluated
        on the same lag grid and ``sigma2 = self.rms ** 2``:

            empirical = cube.noise_structure_function(r_in=1.5)
            from eddy.structurefunction import gaussian_beam_s2
            analytic = gaussian_beam_s2(
                cube.bmaj, cube.bmin, cube.bpa,
                empirical.lags_x, empirical.lags_y,
                sigma2=cube.rms**2, counts=empirical.counts,
            )
            empirical.plot_comparison(analytic)

        Args:
            channels (Optional[sequence of int]): Channel indices to use.
                If ``None``, defaults to the first and last ``N`` channels
                of the cube (matching :meth:`estimate_cube_RMS`).
            N (int): Edge-channel count used when ``channels`` is
                ``None``. Ignored otherwise.
            r_in (float): Inner mask radius [arcsec]. Pixels with
                ``r < r_in`` are NaN-masked and excluded from pair
                averages.
            r_out (float): Outer mask radius [arcsec].
            max_lag_x, max_lag_y (Optional[int]): Maximum lag along
                axis 0 / axis 1 in pixels. Defaults to half the image.
            n_bins (int): Radial bins for the azimuthal average.
            log_spaced (bool): Log-spaced radial bins.
            return_per_channel (bool): If ``True``, also return the
                list of per-channel :class:`StructureFunction2D`
                results.
            symmetrize (bool): Forwarded to
                :class:`StructureFunction2D.from_array`. Noise has
                no preferred radial direction, so the default ``True``
                is almost always what you want.

        Returns:
            ``StructureFunction2D`` (combined across channels), or
            ``(combined, per_channel_list)`` if ``return_per_channel``.

        Notes:
            If the cube was loaded with ``fill=0.0`` (the default) and
            a ``FOV`` clip was applied, off-FOV pixels were filled with
            zero and will contribute spurious zero-difference pairs to
            ``S_2``. Reload with ``fill=np.nan`` or set ``r_out`` to
            exclude that region.
        """
        from .structurefunction import StructureFunction2D, _require_numba
        _require_numba()

        user_channels = channels
        if channels is None:
            N = int(N)
            if N < 1:
                raise ValueError("N must be >= 1.")
            n_total = self.data.shape[0]
            if 2 * N > n_total:
                raise ValueError(
                    "Requested 2*N={} > nchan={}.".format(2 * N, n_total)
                )
            channels = np.concatenate([np.arange(N),
                                       np.arange(n_total - N, n_total)])
        channels = np.asarray(channels, dtype=int).ravel()
        if channels.size == 0:
            raise ValueError("No channels selected.")

        r = np.hypot(self.xaxis[None, :], self.yaxis[:, None])
        keep = np.logical_and(r >= r_in, r <= r_out)
        if not np.any(keep):
            raise ValueError("Mask is empty; check r_in / r_out.")

        dpix = float(abs(self.dpix))

        per_channel = []
        for c in channels:
            chan = np.where(keep, np.asarray(self.data[c], dtype=float),
                            np.nan)
            sf = StructureFunction2D.from_array(
                chan, dx=dpix, dy=dpix,
                max_lag_x=max_lag_x, max_lag_y=max_lag_y,
                n_bins=n_bins, log_spaced=log_spaced,
                symmetrize=symmetrize,
            )
            per_channel.append(sf)

        if len(per_channel) == 1:
            combined = per_channel[0]
        else:
            combined = per_channel[0].combine(per_channel[1:],
                                              n_bins=n_bins,
                                              log_spaced=log_spaced)

        # Stash the mask used so gaussian_beam_s2 can inherit it for a
        # like-for-like sigma2. Only meaningful when channels came from
        # the default first/last-N scheme that estimate_cube_RMS uses.
        if user_channels is None:
            combined.noise_mask = {
                "N": int(N), "r_in": float(r_in), "r_out": float(r_out),
            }

        if return_per_channel:
            return combined, per_channel
        return combined

    def spectral_acf(self, signal_channels=None, signal_velocity=None,
                     r_in=0.0, r_out=1e10, max_lag=None):
        """Channel-to-channel auto-correlation in a signal-free region.

        For every pixel in the spatial annulus ``r_in <= r <= r_out``,
        signal-bearing channels are NaN-masked out; the surviving
        spectrum is mean-subtracted; and a pooled (Pearson-like) ACF
        is accumulated across all pixels at each spectral lag.
        Channel pairs that straddle a masked window are skipped
        (so removing signal channels does not create spurious
        lag-1 correlations across the gap).

        The 95% null band on the returned :class:`SpectralACF` comes
        from Bartlett's formula with the effective sample count
        corrected for spatial pixel correlation by dividing the per-lag
        pair count by ``pix_per_beam``. ``|acf[k]|`` exceeding the band
        at lag ``k > 0`` is evidence of channel-to-channel correlation.

        Args:
            signal_channels (Optional[sequence of int]): Channel
                indices that contain signal. These are NaN-masked
                before pairing. Mutually exclusive with
                ``signal_velocity``.
            signal_velocity (Optional[tuple]): ``(v_min, v_max)``
                velocity range containing signal. Channels with
                ``v_min <= velax <= v_max`` are NaN-masked.
            r_in (float): Inner mask radius [arcsec]. Pixels with
                ``r < r_in`` are excluded -- use this to drop the
                central emission region in a line cube.
            r_out (float): Outer mask radius [arcsec].
            max_lag (Optional[int]): Maximum spectral lag in channels.
                Defaults to ``min(n_used // 4, 50)``.

        Returns:
            :class:`SpectralACF`
        """
        nchan = self.data.shape[0]

        r = np.hypot(self.xaxis[None, :], self.yaxis[:, None])
        keep_xy = np.logical_and(r >= float(r_in), r <= float(r_out))
        if not np.any(keep_xy):
            raise ValueError("Spatial mask is empty; check r_in / r_out.")

        if signal_channels is not None and signal_velocity is not None:
            raise ValueError(
                "Pass signal_channels or signal_velocity, not both."
            )
        chan_mask = np.ones(nchan, dtype=bool)
        if signal_channels is not None:
            chan_mask[np.asarray(signal_channels, dtype=int).ravel()] = False
        elif signal_velocity is not None:
            v_min, v_max = signal_velocity
            chan_mask &= ~np.logical_and(self.velax >= float(v_min),
                                         self.velax <= float(v_max))
        n_used = int(chan_mask.sum())
        if n_used < 4:
            raise ValueError(
                "Fewer than 4 signal-free channels remain; "
                "cannot estimate ACF."
            )

        if max_lag is None:
            max_lag = min(n_used // 4, 50)
        max_lag = int(max_lag)
        if max_lag < 1 or max_lag >= nchan:
            raise ValueError(
                "max_lag must be in [1, nchan); got {}.".format(max_lag)
            )

        pix_y, pix_x = np.where(keep_xy)
        data = np.asarray(self.data[:, pix_y, pix_x], dtype=float).copy()
        # NaN-mask signal channels and any pre-existing NaN pixels.
        data[~chan_mask, :] = np.nan
        # Per-pixel demean over unmasked, finite samples.
        with np.errstate(invalid="ignore"):
            data -= np.nanmean(data, axis=0, keepdims=True)

        lags = np.arange(max_lag + 1)
        acf = np.empty(max_lag + 1, dtype=float)
        counts = np.empty(max_lag + 1, dtype=np.int64)
        for k in lags:
            if k == 0:
                x = data
                y = data
            else:
                x = data[:-k]
                y = data[k:]
            valid = np.isfinite(x) & np.isfinite(y)
            xk = np.where(valid, x, 0.0)
            yk = np.where(valid, y, 0.0)
            num = float(np.sum(xk * yk))
            denom = np.sqrt(float(np.sum(xk * xk))
                            * float(np.sum(yk * yk)))
            counts[k] = int(valid.sum())
            acf[k] = (num / denom) if denom > 0.0 else np.nan

        pix_per_beam = float(self.pix_per_beam)
        n_eff = np.maximum(counts / pix_per_beam, 1.0)
        null_band = 1.96 / np.sqrt(n_eff)

        return SpectralACF(
            lags=lags, acf=acf, counts=counts,
            null_band=null_band, n_eff=n_eff,
            pix_per_beam=pix_per_beam,
            n_pixels=int(keep_xy.sum()),
            n_channels=n_used,
            chan_width=float(abs(self.chan)) if self.chan is not None else 1.0,
        )

    def integrated_spectrum(self, x0=0.0, y0=0.0, inc=0.0, PA=0.0, r_min=None,
                            r_max=None):
        """
        Returns the integrated spectrum over a specified region.

        Args:
            x0 (Optional[float]): Right Ascension offset in [arcsec].
            y0 (Optional[float]): Declination offset in [arcsec].
            inc (Optional[float]): Disk inclination in [deg].
            PA (Optional[float]): Disk position angle in [deg].
            r_min (Optional[float]): Radius to integrate out from in [arcsec].
            r_max (Optional[float]): Radius to integrate out to in [arcsec].

        Returns:
            spectrum, uncertainty (array, array): Something about these.
        """
        rr = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA)[0]
        r_max = rr.max() if r_max is None else r_max
        r_min = 0.0 if r_min is None else r_min
        mask = np.logical_and(rr <= r_max, rr >= r_min)
        nbeams = np.where(mask, 1, 0).sum() / self.pix_per_beam
        spectrum = np.array([np.nansum(c[mask]) for c in self.data])
        spectrum *= self.beams_per_pix
        uncertainty = np.sqrt(nbeams) * self.estimate_cube_RMS()
        return spectrum, uncertainty

    def plot_maximum(self, ax=None, imshow_kwargs=None, return_fig=False):
        """
        Plot the maximum value along each spectrum.

        Args:
            ax (Optional[matplotlib axis]): Axis used for the plotting.
            imshow_kwargs (Optional[dict]): Kwargs to pass to
                ``matplotlib.imshow``.
            return_fig (Optional[bool]): Whether to return the figure instance.
                If an axis was provided, this will always be ``False``.

        Returns:
            fig (matplotlib figure): If ``return_fig=True``, will return the
                figure for continued plotting.
        """

        if ax is None:
            fig, ax = plt.subplots()
        else:
            return_fig = False

        dmax = np.nanmax(self.data, axis=0)
        vmax = np.percentile(dmax, [98])

        imshow_kwargs = {} if imshow_kwargs is None else imshow_kwargs
        imshow_kwargs['interpolation'] = 'nearest'
        imshow_kwargs['extent'] = self.extent
        imshow_kwargs['origin'] = 'lower'
        imshow_kwargs['cmap'] = imshow_kwargs.pop('cmap', 'turbo')
        imshow_kwargs['vmin'] = imshow_kwargs.pop('vmin', 0.0)
        imshow_kwargs['vmax'] = imshow_kwargs.pop('vmax', vmax)

        im = ax.imshow(dmax, **imshow_kwargs)
        cb = plt.colorbar(im, ax=ax, pad=0.03, extend='both')
        cb.set_label('Peak Intensity (Jy/beam)', rotation=270, labelpad=13)
        cb.minorticks_on()
        self._gentrify_plot(ax=ax)

        if return_fig:
            return fig

    def plot_spectrum(self, ax=None, x0=0.0, y0=0.0, inc=0.0, PA=0.0,
                      r_min=None, r_max=None, return_fig=False):
        """
        Plot the integrated spectrum.

        Args:
            x0 (Optional[float]): Right Ascension offset in [arcsec].
            y0 (Optional[float]): Declination offset in [arcsec].
            inc (Optional[float]): Disk inclination in [deg].
            PA (Optional[float]): Disk position angle in [deg].
            r_max (Optional[float]): Radius to integrate out to in [arcsec].
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            return_fig = False
        x = self.velax.copy() / 1e3
        y, dy = self.integrated_spectrum(x0, y0, inc, PA, r_min, r_max)
        ax.axhline(0.0, ls='--', lw=1.0, color='0.9', zorder=-9)
        ax.step(x, y, where='mid', lw=1.0, color='k')
        ax.errorbar(x, y, dy, fmt=' ', lw=1.0, color='k', zorder=-8)
        ax.set_xlabel("Velocity (km/s)")
        ax.set_ylabel("Integrated Flux (Jy)")
        ax.set_xlim(x[0], x[-1])
        ticks = np.diff(ax.xaxis.get_majorticklocs()).mean() / 5.0
        ax.xaxis.set_minor_locator(MultipleLocator(ticks))
        ticks = np.diff(ax.yaxis.get_majorticklocs()).mean() / 5.0
        ax.yaxis.set_minor_locator(MultipleLocator(ticks))

        if return_fig:
            return fig

    # -- ROTATION PROFILE FUNCTIONS -- #

    def get_velocity_profile(self, rbins=None, fit_method='GP', fit_vrad=False,
            fix_vlsr=None, x0=0.0, y0=0.0, inc=0.0, PA=0.0, z0=0.0, psi=1.0,
            r_cavity=0.0, r_taper=np.inf, q_taper=1.0, w_i=None, w_r=None,
            w_t=None, z_func=None, shadowed=False, phi_min=None, phi_max=None,
            exclude_phi=False, abs_phi=False, mask_frame='disk', user_mask=None,
            beam_spacing=True, niter=1, get_vlos_kwargs=None,
            weighted_average=True, return_samples=False, repeat_with_mask=0):
        """
        Returns the rotational and, optionally, radial velocity profiles under
        the assumption that the disk is azimuthally symmetric (at least across
        the regions extracted).

        Several different inference methods can be used through the
        ``fit_method`` argument:

            - ``'GP'``:  Models the aligned and stacked spectrum as a Gaussian
                         Process to remain agnostic about the underlying true
                         line profile. This is the default.
            - ``'dV'``:  Minimize the line width of the aligned and stacked
                         spectrum. Assumes the underyling profile is a
                         Gaussian.
            - ``'SNR'``: Maximizes the SNR of the aligned and stacked spectrum.
                         Assumes the underlying profile is a Gaussian.
            - ``'SHO'``: Fits the azimuthal dependence of the line centroids
                         with a simple harmonic oscillator. Requires a choice
                         of ``centroid_method`` to determine the line centers.

        By default, these methods are applied to annuli across the whole radial
        and azimuthal range of the disk. Masks can be adopted, either limiting
        or excluding azimuthal regions, or a user-defined mask can be provided.

        Boostrapping is possible (multiple iterations using random draws) to
        estimate the uncertainties on the velocities which may be
        underestimated with a single interation.

        Args:
            rbins (Optional[arr]): Array of bin edges of the annuli.
            fit_method (Optional[str]): Method used to infer the velocities.
            fit_vrad (Optional[bool]): Whether to include radial velocities in
                the fit.
            fix_vlsr (optional[bool]): Fix the systemic velocity to calculate
                the deprojected vertical velocities. Only available for
                `fit_method='SHO'`.
            x0 (Optional[float]): Source right ascension offset [arcsec].
            y0 (Optional[float]): Source declination offset [arcsec].
            inc (Optional[float]): Source inclination [degrees]. A positive
                inclination denotes a disk rotating clockwise on the sky, while
                a negative inclination represents a counter-clockwise rotation.
            PA (Optional[float]): Source position angle [degrees]. Measured
                between north and the red-shifted semi-major axis in an
                easterly direction.
            z0 (Optional[float]): Aspect ratio at 1" for the emission surface.
                To get the far side of the disk, make this number negative.
            psi (Optional[float]): Flaring angle for the emission surface.
            r_cavity (Optional[float]): Outer radius of a cavity. Within this
                region the emission surface is taken to be zero.
            w_i (Optional[float]): Warp inclination in [degrees] at the disk
                center.
            w_r (Optional[float]): Scale radius of the warp in [arcsec].
            w_t (Optional[float]): Angle of nodes of the warp in [degrees].
            z_func (Optional[function]): A function which provides
                :math:`z(r)`. Note that no checking will occur to make sure
                this is a valid function.
            shadowed (Optional[bool]): Whether to use the slower, but more
                robust, deprojection method for shadowed disks.
            phi_min (Optional[float]): Minimum polar angle of the segment of
                the annulus in [deg]. Note this is the polar angle, not the
                position angle.
            phi_max (Optional[float]): Maximum polar angle of the segment of
                the annulus in [deg]. Note this is the polar angle, not the
                position angle.
            exclude_phi (Optional[bool]): If ``True``, exclude the provided
                polar angle range rather than include it.
            abs_phi (Optional[bool]): If ``True``, take the absolute value of
                the polar angle such that it runs from 0 [deg] to 180 [deg].
            mask_frame (Optional[str]): Which frame to specify the mask in,
                either ``'disk'``, the default, or ``'sky'``.
            user_mask (Optional[array]): A user-specified mask to include. Must
                have the same shape as ``self.data``.
            beam_spacing (int): Sample pixels separated by roughly
            `beam_spacing * bmaj` in azimuthal distance.
            niter (Optional[int]): Number of iterations to run.
            get_vlos_kwargs=None,
            weighted_average (Optional[bool]): Whether to combine multiple
                iterations with a weighted average and standard deviation,
                ``weighted_average=True``, or the traditional mean and standard
                deviation, ``weighted_average=False``.
            return_samples (Optional[bool]): Whether to return the samples
                instead of combining them.
            repeat_with_mask (Optional[int]):

        Returns:
            samples (array): If ``return_samples=True``. The array of ``niter``
                samples of the velocity profiles.
            rvals, profile, uncertainty (array, array, array): If
                ``return_samples=False``. The arrays of radial locations and
                combined velocity and uncertainties.
        """

        if niter == 0:
            raise ValueError("`niter` must be >= 1.")

        # Single iteration.

        if niter == 1:
            return self._velocity_profile(rbins=rbins,
                                          fit_method=fit_method,
                                          fit_vrad=fit_vrad,
                                          fix_vlsr=fix_vlsr,
                                          x0=x0,
                                          y0=y0,
                                          inc=inc,
                                          PA=PA,
                                          z0=z0,
                                          psi=psi,
                                          r_cavity=r_cavity,
                                          r_taper=r_taper,
                                          q_taper=q_taper,
                                          w_i=w_i,
                                          w_r=w_r,
                                          w_t=w_t,
                                          z_func=z_func,
                                          shadowed=shadowed,
                                          phi_min=phi_min,
                                          phi_max=phi_max,
                                          exclude_phi=exclude_phi,
                                          abs_phi=abs_phi,
                                          mask_frame=mask_frame,
                                          user_mask=user_mask,
                                          beam_spacing=beam_spacing,
                                          get_vlos_kwargs=get_vlos_kwargs,
                                          repeat_with_mask=repeat_with_mask)

        # Multiple iterations.

        if beam_spacing is False:
            raise ValueError("niter must equal 1 when beam_spacing=False.")

        samples = [self._velocity_profile(rbins=rbins,
                                          fit_method=fit_method,
                                          fit_vrad=fit_vrad,
                                          fix_vlsr=fix_vlsr,
                                          x0=x0,
                                          y0=y0,
                                          inc=inc,
                                          PA=PA,
                                          z0=z0,
                                          psi=psi,
                                          r_cavity=r_cavity,
                                          r_taper=r_taper,
                                          q_taper=q_taper,
                                          w_i=w_i,
                                          w_r=w_r,
                                          w_t=w_t,
                                          z_func=z_func,
                                          shadowed=shadowed,
                                          phi_min=phi_min,
                                          phi_max=phi_max,
                                          exclude_phi=exclude_phi,
                                          abs_phi=abs_phi,
                                          mask_frame=mask_frame,
                                          user_mask=user_mask,
                                          beam_spacing=beam_spacing,
                                          get_vlos_kwargs=get_vlos_kwargs,
                                          repeat_with_mask=repeat_with_mask)
                   for _ in range(niter)]

        # Just return the samples if requested.

        if return_samples:
            return samples

        rpnts = samples[0][0]
        profiles = np.array([s[1] for s in samples])

        # Calculate weights, making sure they are finite with non-zero sum.

        if weighted_average:
            weights = [1.0 / s[2] for s in samples]
            weights = np.where(np.isfinite(weights), weights, 0.0)
        else:
            weights = np.ones(profiles.shape)

        if np.all(np.sum(weights, axis=0) == 0.0):
            weights = np.ones(profiles.shape)

        weights = np.where(np.isfinite(weights), weights, 1.0)
        weights += 1e-10 * np.random.randn(weights.size).reshape(weights.shape)

        M = np.sum(weights != 0.0, axis=0)

        # Weighted average.

        profile = np.average(profiles, weights=weights, axis=0)

        # Weighted standard deviation.

        uncertainty = weights * (profiles - profile[None, :, :])**2
        uncertainty = np.sum(uncertainty, axis=0)
        uncertainty /= (M - 1.0) / M * np.sum(weights, axis=0)
        uncertainty = np.sqrt(uncertainty)

        return rpnts, profile, uncertainty

    def _velocity_profile(self, rbins=None, fit_method='GP', fit_vrad=False,
            fix_vlsr=None, x0=0.0, y0=0.0, inc=0.0, PA=0.0, z0=0.0, psi=1.0,
            r_cavity=0.0,  r_taper=np.inf, q_taper=1.0, w_i=None, w_r=None,
            w_t=None, z_func=None, shadowed=False, phi_min=None, phi_max=None,
            exclude_phi=False, abs_phi=False, mask_frame='disk',
            user_mask=None, beam_spacing=True, get_vlos_kwargs=None,
            repeat_with_mask=0):
        """
        Returns the velocity (rotational and radial) profiles.

        Args:
            TBD

        Returns:
            TBD
        """

        # Define the radial binning.

        if rbins is None:
            rbins = np.arange(0, self.xaxis.max(), 0.25 * self.bmaj)
        rpnts = np.mean([rbins[1:], rbins[:-1]], axis=0)

        # Set up the kwargs for the fitting.

        kw = {} if get_vlos_kwargs is None else get_vlos_kwargs
        kw['fit_vrad'] = fit_vrad
        kw['fix_vlsr'] = fix_vlsr
        kw['fit_method'] = fit_method
        kw['repeat_with_mask'] = repeat_with_mask

        # Cycle through the annuli.

        profiles = []
        uncertainties = []
        for r_min, r_max in zip(rbins[:-1], rbins[1:]):
            annulus = self.get_annulus(r_min=r_min,
                                       r_max=r_max,
                                       phi_min=phi_min,
                                       phi_max=phi_max,
                                       exclude_phi=exclude_phi,
                                       abs_phi=abs_phi,
                                       x0=x0,
                                       y0=y0,
                                       inc=inc,
                                       PA=PA,
                                       z0=z0,
                                       psi=psi,
                                       r_cavity=r_cavity,
                                       r_taper=r_taper,
                                       q_taper=q_taper,
                                       w_i=w_i,
                                       w_r=w_r,
                                       w_t=w_t,
                                       z_func=z_func,
                                       shadowed=shadowed,
                                       mask_frame=mask_frame,
                                       user_mask=user_mask,
                                       beam_spacing=beam_spacing)

            output = annulus.get_vlos(**kw)
            profiles += [output[0]]
            uncertainties += [output[1]]

        # Make sure the returned arrays are in the (nparam, nrad) form.

        profiles = np.atleast_2d(profiles).T
        uncertainties = np.atleast_2d(uncertainties).T
        assert profiles.shape[0] == uncertainties.shape[0] == 3
        assert profiles.shape[1] == uncertainties.shape[1] == rpnts.size

        return rpnts, profiles, uncertainties

    # -- ANNULUS FUNCTIONS -- #

    def get_annulus(self, r_min, r_max, phi_min=None, phi_max=None,
            exclude_phi=False, abs_phi=False, x0=0.0, y0=0.0, inc=0.0, PA=0.0,
            z0=0.0, psi=1.0, r_cavity=0.0, r_taper=np.inf, q_taper=1.0,
            w_i=None, w_r=None, w_t=None, z_func=None, shadowed=False,
            mask_frame='disk', user_mask=None, beam_spacing=True,
            annulus_kwargs=None):
        """
        Returns an annulus instance.

        Args:
            r_min (float): Inner radius of the annulus in [arcsec].
            r_max (float): Outer radius of the annulus in [arcsec].
            phi_min (Optional[float]): Minimum polar angle in [X] for the
                annulus. ``phi`` is measured from the red-shifted major axis and
                increases in a clockwise direction, spanning ``-pi`` to ``+pi``.
            phi_max (Optional[float]): Maximum polar angle in [x] for the
                annulus. ``phi`` is measured from the red-shifted major axis and
                increases in a clockwise direction, spanning ``-pi`` to ``+pi``.
            exclude_phi (Optional[bool]): If ``True``, exclude the polar angle
                range rather than to include it.
            abs_phi (Optional[bool]): If ``True``, consider only the absolute
                values of ``phi`` in order to get a symmetric mask.
            x0 (Optional[float]): Source right ascension offset [arcsec].
            y0 (Optional[float]): Source declination offset [arcsec].
            inc (Optional[float]): Source inclination [degrees]. A positive
                inclination denotes a disk rotating clockwise on the sky, while
                a negative inclination represents a counter-clockwise rotation.
            PA (Optional[float]): Source position angle [degrees]. Measured
                between north and the red-shifted semi-major axis in an
                easterly direction.
            z0 (Optional[float]): Aspect ratio at 1" for the emission surface.
                To get the far side of the disk, make this number negative.
            psi (Optional[float]): Flaring angle for the emission surface.
            r_cavity (Optional[float]): Outer radius of a cavity. Within this
                region the emission surface is taken to be zero.
            r_taper (Optional[float]): Radius for tapered emission surface.
            q_taper (Optional[float]): Exponent for tapered emission surface.
            w_i: [coming soon]
            w_r: [coming soon]
            w_t: [coming soon]

        """

        # Calculate and flatten the mask.

        mask = self.get_mask(r_min=r_min,
                             r_max=r_max,
                             phi_min=phi_min,
                             phi_max=phi_max,
                             exclude_phi=exclude_phi,
                             abs_phi=abs_phi,
                             x0=x0,
                             y0=y0,
                             inc=inc,
                             PA=PA,
                             z0=z0,
                             psi=psi,
                             r_cavity=r_cavity,
                             r_taper=r_taper,
                             q_taper=q_taper,
                             w_i=w_i,
                             w_r=w_r,
                             w_t=w_t,
                             z_func=z_func,
                             shadowed=shadowed,
                             mask_frame=mask_frame,
                             user_mask=user_mask)
        if mask.shape != self.data[0].shape:
            raise ValueError("mask is incorrect shape: {}.".format(mask.shape))
        mask = mask.flatten()

        # Flatten the data and get the deprojected pixel coordinates.
        # We will record the on-sky pixels, their deprojected disk-frame polar
        # coordinates and the array indices.

        dvals = self.data.copy().reshape(self.data.shape[0], -1)
        dvals = dvals[:, mask].T

        rvals, pvals = self.disk_coords(x0=x0,
                                        y0=y0,
                                        inc=inc,
                                        PA=PA,
                                        z0=z0,
                                        psi=psi,
                                        r_cavity=r_cavity,
                                        r_taper=r_taper,
                                        q_taper=q_taper,
                                        w_i=w_i,
                                        w_r=w_r,
                                        w_t=w_t,
                                        z_func=z_func,
                                        shadowed=shadowed,
                                        flatten=True)[:2]
        rvals, pvals = rvals[mask], pvals[mask]

        xsky, ysky = self.disk_coords(x0=0.0,
                                      y0=0.0,
                                      inc=0.0,
                                      PA=0.0,
                                      outframe='cartesian',
                                      flatten=True)[:2]
        xsky, ysky = xsky[mask], ysky[mask]

        iidx, jidx = np.meshgrid(np.arange(self.nypix), np.arange(self.nxpix))
        iidx, jidx = iidx.flatten()[mask], jidx.flatten()[mask]

        # Thin down to spatially independent pixels.

        thinned = self._independent_samples(beam_spacing=beam_spacing,
                                            rvals=rvals,
                                            pvals=pvals,
                                            dvals=dvals,
                                            xsky=xsky,
                                            ysky=ysky,
                                            jidx=jidx,
                                            iidx=iidx)
        rvals, pvals, dvals, xsky, ysky, jidx, iidx = thinned

        # Return the annulus instance.

        annulus_kwargs = {} if annulus_kwargs is None else annulus_kwargs
        return annulus(spectra=dvals, pvals=pvals, velax=self.velax, inc=inc,
                       rvals=rvals, xsky=xsky, ysky=ysky, jidx=jidx, iidx=iidx,
                       **annulus_kwargs)

    # -- PLOTTING FUNCTIONS -- #

    def plot_mask(self, ax, r_min=None, r_max=None, exclude_r=False,
                  phi_min=None, phi_max=None, exclude_phi=False, abs_phi=False,
                  mask_frame='disk', mask=None, x0=0.0, y0=0.0, inc=0.0,
                  PA=0.0, z0=0.0, psi=1.0, r_cavity=None, r_taper=None,
                  q_taper=None, w_i=None, w_r=None, w_t=None, z_func=None,
                  mask_color='k', mask_alpha=0.5, contour_kwargs=None,
                  contourf_kwargs=None, shadowed=False):
        """
        Plot the boolean mask on the provided axis to check that it makes
        sense.

        Args:
            ax (matplotib axis instance): Axis to plot the mask.
            r_min (Optional[float]): Minimum midplane radius of the annulus in
                [arcsec]. Defaults to minimum deprojected radius.
            r_max (Optional[float]): Maximum midplane radius of the annulus in
                [arcsec]. Defaults to the maximum deprojected radius.
            exclude_r (Optional[bool]): If ``True``, exclude the provided
                radial range rather than include.
            PA_min (Optional[float]): Minimum polar angle of the segment of the
                annulus in [degrees]. Note this is the polar angle, not the
                position angle.
            PA_max (Optional[float]): Maximum polar angle of the segment of the
                annulus in [degrees]. Note this is the polar angle, not the
                position angle.
            exclude_PA (Optional[bool]): If ``True``, exclude the provided
                polar angle range rather than include it.
            abs_PA (Optional[bool]): If ``True``, take the absolute value of
                the polar angle such that it runs from 0 [deg] to 180 [deg].
            x0 (Optional[float]): Source center offset along the x-axis in
                [arcsec].
            y0 (Optional[float]): Source center offset along the y-axis in
                [arcsec].
            inc (Optional[float]): Inclination of the disk in [degrees].
            PA (Optional[float]): Position angle of the disk in [degrees],
                measured east-of-north towards the redshifted major axis.
            z0 (Optional[float]): Emission height in [arcsec] at a radius of
                1".
            psi (Optional[float]): Flaring angle of the emission surface.
            z_func (Optional[function]): A function which provides
                :math:`z(r)`. Note that no checking will occur to make sure
                this is a valid function.
            mask_color (Optional[str]): Color used for the mask lines.
            mask_alpha (Optional[float]): The alpha value of the filled contour
                of the masked regions. Setting ``mask_alpha=0.0`` will remove
                the filling.

            contour_kwargs (Optional[dict]): Kwargs to pass to contour for
                drawing the mask.

        Returns:
            ax : The matplotlib axis instance.
        """
        # Grab the mask.
        if mask is None:
            mask = self.get_mask(r_min=r_min, r_max=r_max, exclude_r=exclude_r,
                                 phi_min=phi_min, phi_max=phi_max,
                                 exclude_phi=exclude_phi, abs_phi=abs_phi,
                                 mask_frame=mask_frame, x0=x0, y0=y0, inc=inc,
                                 PA=PA, z0=z0, psi=psi, r_cavity=r_cavity,
                                 r_taper=r_taper, q_taper=q_taper, w_i=w_i,
                                 w_r=w_r, w_t=w_t, z_func=z_func,
                                 shadowed=shadowed)
        assert mask.shape[0] == self.yaxis.size, "Wrong y-axis shape for mask."
        assert mask.shape[1] == self.xaxis.size, "Wrong x-axis shape for mask."

        # Set the default plotting style.

        contour_kwargs = {} if contour_kwargs is None else contour_kwargs
        contour_kwargs['colors'] = contour_kwargs.pop('colors', mask_color)
        contour_kwargs['linewidths'] = contour_kwargs.pop('linewidths', 1.0)
        contour_kwargs['linestyles'] = contour_kwargs.pop('linestyles', '-')
        contourf_kwargs = {} if contourf_kwargs is None else contourf_kwargs
        contourf_kwargs['alpha'] = contourf_kwargs.pop('alpha', mask_alpha)
        contourf_kwargs['colors'] = contourf_kwargs.pop('colors', mask_color)

        # Plot the contour and return the figure.

        ax.contourf(self.xaxis, self.yaxis, mask, [-.5, .5], **contourf_kwargs)
        ax.contour(self.xaxis, self.yaxis, mask, 1, **contour_kwargs)

    # -- UTILITIES -- #

    def get_spectrum(self, coords, x0=0.0, y0=0.0, inc=0.0, PA=0.0,
                     z0=0.0, psi=1.0, z_func=None, frame='sky',
                     coord_type='cartesian', area=0.0, beam_weighting=False,
                     return_mask=False):
        """
        Return a spectrum at a position defined by a coordinates given either
        in sky-frame position (``frame='sky'``) or a disk-frame location
        (``frame='disk'``). The coordinates can be either in a cartesian system
        (``coord_type='cartesian'``) or cylindrical system
        (``coord_type='cylindrical'``).

        By default the returned spectrum is extracted at the pixel closest to
        the provided coordinates. If ``area`` is set to a positive value, then
        a beam-shaped area is averaged over, where ``area`` sets the size of
        this region in number of beams. For example ``area=2.0`` will result
        in an average over an area twice the size of the beam.

        If an area is averaged over, you can also weight the pixels by the beam
        response with ``beam_weighting=True``. This will reduce the weight of
        pixels that are further away from the beam center.

         Finally, to check that you're extracting what you think you are, you
         can return the mask (and weights) used for the extraction with
         ``return_mask=True``. Note that if ``beam_weighting=False`` then all
         ``weights`` will be 1.

         TODO: Check that the returned uncertainties are reasonable.

        Args:
            coords (tuple): The coordinates from where you want to extract a
                spectrum. Must be a length 2 tuple.
            x0 (Optional[float]): RA offset in [arcsec].
            y0 (Optional[float]): Dec offset in [arcsec].
            inc (Optional[float]): Inclination of source in [deg]. Only
                required for ``frame='disk'``.
            PA (Optional[float]): Position angle of source in [deg]. Only
                required for ``frame='disk'``.
            frame (Optional[str]): The frame that the ``coords`` are given.
                Either ``'disk'`` or ``'sky'``.
            coord_type (Optional[str]): The type of coordinates given, either
                ``'cartesian'`` or ``'cylindrical'``.
            area (Optional[float]): The area to average over in units of the
                beam area. Note that this take into account the beam aspect
                ratio and position angle. For a single pixel extraction use
                ``area=0.0``.
            beam_weighting (Optional[bool]): Whether to use the beam response
                function to weight the averaging of the spectrum.
            return_mask (Optional[bool]): Whether to return the mask and
                weights used to extract the spectrum.

        Retuns (if ``return_mask=False``):
            x, y, dy (arrays): The velocity axis, extracted spectrum and
            associated uncertainties.
        (if ``return_mask=True``):
            mask, weights (arrays): Arrays of the mask used to extract the
            spectrum and the weighted used for the averaging.
        """

        # TODO:
        #   1 - Check if it's three coordinates, or just two.
        #   2 - Check which frame it's in.

        # Convert the input coordinate into on-sky cartesian coordinates
        # relative to the center of the image.

        if frame.lower() == 'sky':
            if inc != 0.0 or PA != 0.0:
                message = "WARNING: You shouldn't need to specify `inc` or "
                message += "`PA` when using `frame='sky'`."
                print(message)
            c1 = np.squeeze(coords[0])
            c2 = np.squeeze(coords[1])
            if coord_type.lower() == 'cartesian':
                x, y = c1 + x0, c2 + y0
            elif coord_type.lower() == 'cylindrical':
                x = x0 + c1 * np.cos(c2 - np.radians(90.0))
                y = y0 - c1 * np.sin(c2 - np.radians(90.0))
        elif frame.lower() == 'disk':
            x, y = self.disk_to_sky(coords=coords, coord_type=coord_type,
                                    inc=inc, PA=PA, x0=x0, y0=y0)
        assert x.size == y.size == 1

        # Define the area to average over.

        if area == 0.0:
            x_pix = abs(self.xaxis - x).argmin()
            y_pix = abs(self.yaxis - y).argmin()
            mask = np.zeros(self.data[0].shape)
            weights = np.zeros(mask.shape)
            mask[y_pix, x_pix] = 1
            weights[y_pix, x_pix] = 1
        elif area > 0.0:
            mask = self._beam_mask(x, y, stretch=area)
            weights = self._beam_mask(x, y, stretch=area, response=True)
        else:
            raise ValueError("`area` must be a non-negative value.")
        weights = weights if beam_weighting else mask

        # If requested, return the mask and the weighting instead.

        if return_mask:
            return mask, weights

        # Otherwise, extract the spectrum and average it.

        y = [np.average(c * mask, weights=weights) for c in self.data]
        dy = max(1.0, mask.sum() * self.beams_per_pix)**-0.5 * self.rms
        return self.velax, np.array(y), np.array([dy for _ in y])
