# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
