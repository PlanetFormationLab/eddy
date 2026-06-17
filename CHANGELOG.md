# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.1.0] – 2026-06-16

### Added
- **Structure function analysis.** New `eddy.structurefunction` module
  with a numba-jitted 2D second-order structure function kernel,
  helpers for lag coordinates, axis/azimuthal profile extraction, and
  pair-count weighted combination across realizations. The 1D
  azimuthal spiral model `S2phi` (with `m=1,2,3` modes) is included.
- **`StructureFunction2D` result class** holding the 2D `S_2` surface,
  pair counts, 1D profiles, and the underlying polar grid. Methods
  include `.combine(...)`, `.subtract(...)`, `.compare_to(...)`,
  `.fit_spiral(modes=...)`, `.plot_2d()`, `.plot_profiles()`, and
  `.plot_comparison(...)`. Surfaced at the top level
  (`from eddy import StructureFunction2D`).
- **`StructureFunction2DStack` container** — list-like over per-radius
  `StructureFunction2D` results, with stacked array properties
  (`S2_stack`, `S2_x_stack`, `S2_y_stack`, `S2_i_stack`), batched
  `.fit_spiral(modes=...)`, `.half_power_lags(axis=...)`, the radius
  vs azimuth/radial-lag heatmaps (`.plot_azimuthal_heatmap()`,
  `.plot_radial_heatmap()`, `.plot_anisotropy_heatmap()`), and
  `.plot_gridded()` for the underlying deprojected field. Surfaced at
  the top level.
- **`momentmap.compute_structure_function(...)`** and
  **`compute_structure_function_stack(ref_rs, ref_band, ...)`** —
  apply the structure function to a moment map after a polar
  deprojection through `imagecube.polar_deprojection`; the stack
  variant sweeps reference radii on a single shared deprojection.
  Inherited by `rotationmap`, so `fit_map` residuals flow through the
  same entry point.
- **Anisotropic-GRF forward models.** `grf_s2_slices`,
  `grf_s2_2d_global`, `predict_s2_slices`, `predict_s2_2d`, and
  `predict_spiral_s2_2d` — the Paciorek-Schervish-kernel structure
  functions used to fit `(sigma, alphar, ell0r, ell0phi, alphaphi,
  pitch)` from data.
- **`StructureFunction2D.fit_GRF(...)`** — dispatches on construction
  mode and a `pitch` flag: reference-annulus slice fit, global-mode
  surface fit, or global-mode surface fit with the pitch freed (the
  only configuration that resolves the pitch sign). LM (`'lsq'`) and
  emcee (`'mcmc'`) back-ends share a single internal driver
  (`_grf_fit_core`).
- **`StructureFunction2DStack.fit_GRF(...)`** — joint anisotropic-GRF
  fit across the per-annulus slices.
- **`StructureFunction2DStack.measure_heuristics(...)`** — six
  model-free per-ring scalars (`T1a..T4`) mapping onto the GRF
  parameters: amplitude, radial/azimuthal correlation lengths,
  anisotropy, and stationarity slopes.
- **Spectral diagnostics on `linecube`.** New `SpectralACF` container
  and `linecube.spectral_acf(...)` — the channel-to-channel
  autocorrelation diagnostic used to spot oversampled spectral axes
  before computing noise statistics.
- **`linecube.noise_structure_function(...)`** — empirical noise
  `S_2` averaged over signal-free channels, returning a
  `StructureFunction2D`-compatible surface.
- **`linecube.gaussian_beam_s2(...)`** — analytic Gaussian-beam noise
  `S_2` prediction matched to an empirical surface via an explicit
  `match=` argument (counts / lag-grid / `noise_mask` inherited).
- **Spectral smoothing in `linecube.to_momentmap`.** New `smooth` and
  `polyorder` kwargs apply a Savitzky-Golay (or top-hat) smoother
  before collapse and re-estimate the RMS from the smoothed cube,
  matching the `bettermoments` CLI flow.
- **`product=` kwarg in `linecube.to_momentmap`** — pick a specific
  bettermoments product suffix (e.g. `'v0'` for `quadratic`) instead
  of always taking the first. The matching `'d'`-prefixed
  uncertainty product is attached to the returned map as `.error`.
- **`rotationmap.fit_map(optimize_kwargs=...)`** — forwarded to the
  pre-MCMC `L-BFGS-B` optimizer. Supports `'method'` and `'options'`
  (a dict of `scipy.optimize.minimize` options).
- **`py.typed` marker** — `eddy` is now PEP-561 type-stubs-compatible,
  so IDE / type-checker tooling no longer skips the package.
- **`structurefunction` optional dependency** — install the numba
  extra via `pip install astro-eddy[structurefunction]`. `import
  eddy` works without numba; only the structure-function entry
  points raise.
- **Tests.** New `tests/test_structurefunction.py` (14 cases)
  covering analytic recovery, NaN handling, reference-annulus mode,
  the result-class `combine` / `fit_spiral` helpers, the radius-sweep
  stack (including single-element-stack equivalence with a direct
  call), and end-to-end smoke tests through
  `momentmap.compute_structure_function` and
  `compute_structure_function_stack`.

### Changed
- **`linecube.to_momentmap` is now keyword-only after `method`.**
  Everything else (`product`, `clip`, `smooth`, `polyorder`,
  `bettermoments_kwargs`) must be passed by name. An old positional
  call (e.g. `cube.to_momentmap('quadratic', 3.0)`) now raises
  `TypeError` immediately instead of binding `product=3.0` and
  failing during product validation.
- **`linecube.to_momentmap` no longer requires a `BUNIT` header
  card** or the private `bettermoments.io._get_bunits` helper. A
  failed lookup falls back to a name-based velocity/intensity
  classification via the new `_BM_VELOCITY_PRODUCTS` set, so
  simulated cubes (which often lack `BUNIT`) and future
  bettermoments releases that move `_get_bunits` both degrade
  gracefully.
- **`rotationmap.fit_map(optimize_kwargs=...)` rejects unknown
  top-level keys.** Previously, anything other than `'method'` and
  `'options'` was silently dropped — including the common mistake of
  passing solver options (`maxiter`, `ftol`, ...) at the top level
  instead of nesting them under `'options'`. The new behavior raises
  `TypeError` with a hint about the nested form.
- **The momentmap polar pipeline leaves `S2_i = None`.** Its `dx` is
  arcsec and its `dy` is degrees, so an azimuthally-averaged
  `sqrt(l_x^2 + l_y^2)` mixes incommensurate units and is not a
  physical average. `StructureFunction2D.S2_i` is now an
  `Optional[ndarray]`; `plot_profiles`, `compare_to`,
  `plot_comparison`, `combine`, `subtract`, and `S2_i_stack` all
  branch on it.
- **Uniform `rgrid` / `tgrid` are required by the structure-function
  pipeline.** `momentmap._structure_function_polar_grid` now raises
  `ValueError` on log-spaced or otherwise non-uniform grids instead
  of silently mislabeling the lag axis with the mean spacing.

### Fixed
- `S2phi(dphi, Nphi, A1, A3=...)` no longer silently drops the m=3
  term when `A2 is None`; the `m=3` branch was previously nested
  inside the `m=2` branch.
- `StructureFunction2DStack.measure_heuristics` no longer raises
  `ZeroDivisionError` on stacks with all-zero reliability weights
  (e.g. mixed rings where every `ell_r` or `ell_phi` half-power lag is
  NaN). Weight positivity is now folded into the radial/azimuthal
  masks and a clear `ValueError` fires when nothing survives. The
  bare `except Exception` around the slope fit was replaced with an
  explicit `m.sum() < 2` guard plus a targeted `LinAlgError` catch.
- Default-grid `fit_GRF(pitch=True, r_axis=...)` no longer crashes
  inside scipy's SVD. `_grf_surface_setup` drops non-positive radii
  (with a `warnings.warn` naming the count) before evaluating
  `ell_r(r) = ell0r * (r/r0)**alphar`, which was singular at `r = 0`.
- `_grf_*_setup` no longer crashes with `min() iterable argument is
  empty` / `zero-size array reduction` when no annulus has positive
  lags. Degenerate cases now feed `np.nan` placeholders to
  `_grf_data_bounds`, whose existing filter drops them and falls
  back to the module-wide defaults.
- `imagecube.polar_deprojection`'s `tgrid` docstring now says
  `[radians]` (matching the implementation, which builds
  `np.linspace(-pi, pi, ...)`); the previous "[degrees]" annotation
  silently produced ~57× inflated azimuthal lags for users who
  followed the docstring.
- `pyproject.toml` version had drifted from `eddy.__version__`; both
  now report `3.1.0`.

### Behaviour notes
- `StructureFunction2D.fit_GRF` / `StructureFunction2DStack.fit_GRF`
  now warn (a) when a user-supplied `p0` entry is clipped onto a
  bound (naming the parameter, the requested value, and the bound),
  and (b) per parameter whose lsq solution sits on a bound (the
  reported `perr` is unreliable). The MCMC walker centre is also
  nudged strictly inside the box before scattering, so walkers no
  longer start with ~50 % of members at `-inf` prior when a
  parameter pins. Gaussian priors are intersected with the flat
  bounds — a prior that pulls walkers past a bound is truncated
  there.

### Backward compatibility
- All public 3.0.x `fit_map`, `fit_annuli`, `get_vlos`, `plot_*`, and
  instantiation call signatures are preserved.
- The `linecube.to_momentmap` signature change is the only positional
  break in 3.1.0. Migration is a one-line edit per call site (add
  `,` after `method` and use keyword names for everything else).

## [3.0.0] – 2026-05-13

A major refactor centred on a JAX-backed model, a new class hierarchy,
and an opt-in gradient-based MCMC sampler. All user-facing call
signatures from 2.x are preserved.

### Added
- **JAX backend.** `rotationmap`'s model and likelihood (`_make_model`,
  `disk_coords`, `_ln_likelihood`, FFT beam convolution) are now
  JIT-compiled, autodifferentiable, and GPU-aware via
  `jax.default_backend()`.
- **`mcmc='numpyro'`** opt-in NUTS sampler in `rotationmap.fit_map()`
  and `Annulus3D.get_vlos_GP()`. emcee remains the default backend.
  Legacy `nwalkers`/`nburnin`/`nsteps` kwargs map onto numpyro's
  `num_chains`/`num_warmup`/`num_samples` internally; numpyro-specific
  controls (`max_tree_depth`, `seed`, `chain_method`) are passed
  through `mcmc_kwargs`. See tutorial 6.
- **`imagecube.to_fits()`** — writes `self.data` (or a provided array)
  back to FITS with a header rebuilt from the live `xaxis`/`yaxis`
  /`velax` so axis keywords stay in sync after FOV clipping etc.
- **`linecube.to_momentmap()`** — collapses a 3D spectral cube into a
  2D moment map via `bettermoments`. Returns a `rotationmap` for
  velocity-typed methods (`first`, `quadratic`, ...) or a `momentmap`
  otherwise.
- **`tests/`** — a pytest smoke suite (24 tests) covering `imagecube`,
  `rotationmap`, and `annulus` basics. Tutorial FITS files are reused
  as fixtures.
- **`.github/workflows/ci.yml`** — GitHub Actions matrix on Python
  3.11 and 3.13, runs the smoke suite on every push to
  `master`/`refactoring` and on PRs.
- **Tutorial 6 (`tutorial_6_numpyro.ipynb`)** — short walkthrough of
  the numpyro path on the standard HD163296 setup, with
  recommended `max_tree_depth=6` and the prior-tightening tip for
  improper uniforms (e.g. `r_taper`).
- **`docs` extra** — `pip install -e ".[docs]"` now installs the
  Sphinx stack listed in `docs/requirements.txt`.

### Changed
- **Class hierarchy.** The single `datacube` class has been split into
  three: `imagecube` (shared FITS I/O, WCS, beam parsing, plotting),
  `momentmap` (2D), `linecube` (3D). `rotationmap` now inherits from
  `momentmap`. The legacy `from eddy.datacube import datacube` import
  still works (re-exports `imagecube as datacube`).
- **Annulus split.** The previous `annulus` class is now
  `Annulus3D`, with a new sibling `Annulus2D` for pixel-velocity
  fits (used by `rotationmap.fit_annuli`). `from eddy import annulus`
  remains as an alias for `Annulus3D`.
- **Gaussian Process backend** swapped from `celerite` to
  `tinygp.quasisep.Matern32`. JAX-compatible; lets the GP annulus
  path compose with the numpyro NUTS sampler.
- **Pre-MCMC optimizer** switched from finite-difference `TNC` to
  `L-BFGS-B` with analytic JAX gradients.
- **Performance.** ~16× cumulative speedup in warm `fit_map` calls
  via JIT-closure + `vmap`'d batch log-probability. emcee path was
  the main beneficiary; numpyro path's per-iteration cost is
  documented in tutorial 6 and is being tracked separately.
- **Minimum Python version** bumped from 3.8 to 3.10.

### Removed
- `celerite` runtime dependency (replaced by `tinygp`).
- Dead `_SHO_*` private helpers in `rotationmap` (`_fit_SHO`,
  `_SHO_chi2`, `_SHO_MCMC`, `_SHO_ln_*`) — orphaned by the
  `fit_annuli → Annulus2D.get_vlos` rewire. ~140 lines.
- Global `warnings.filterwarnings("ignore")` calls at the top of
  each submodule — they were hiding three real bugs (all fixed; see
  below).

### Fixed
- NaN gradient hazards that prevented numpyro from running the
  9-parameter 3D fit: `_analytic_z` `r_eff**psi` at the cavity
  boundary (now wrapped in the double-`where` pattern) and
  `_ln_likelihood`'s NaN-data leakage via `vjp` (now zeros the
  masked data before the subtraction).
- `dist.Uniform(0, inf)` is degenerate in numpyro and aborted NUTS
  init; flat priors with `±inf` bounds now dispatch to
  `dist.ImproperUniform` on the matching constraint.
- emcee 3.x `sampler.chain` deprecation warnings — `fit_map`'s
  walker plot and PA-wrap now use `sampler.get_chain()`.
- `np.log(1.0 / (hi - lo))` RuntimeWarning for improper uniform
  priors.
- Invalid `\-` escape sequence in `remove_hot_pixels` docstring.

### Backward compatibility
- All public `fit_map`, `fit_annuli`, `get_vlos`, `plot_*`, and
  instantiation call signatures from 2.x are preserved.
- Legacy import aliases retained: `from eddy.datacube import datacube`,
  `from eddy import annulus`.
- A few `DeprecationWarning`s previously hidden by the global filter
  may now surface (notably from emcee 3.x when client code touches
  `sampler.chain` directly).

### Migration notes
- If your environment pins `celerite`, you can drop it.
- If you maintain a subclass of `datacube`, it still works via the
  shim. Consider migrating to the appropriate subclass
  (`imagecube`/`momentmap`/`linecube`) for clarity.

---

For releases earlier than 3.0.0, see the [git
history](https://github.com/PlanetFormationLab/eddy/commits/master)
and the corresponding entries on
[PyPI](https://pypi.org/project/astro-eddy/#history).
