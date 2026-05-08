# eddy Refactoring Plan

## Goals

1. **JAX backend** — GPU acceleration, JIT compilation, autodifferentiation
2. **Better class hierarchy** — cleaner 2D/3D separation, shared base classes
3. **Gradient-based MCMC** — numpyro/NUTS as default, emcee as fallback
4. **New features** — `to_fits()`, `Annulus2D`/`Annulus3D` split
5. **Backward compatibility** — all user-facing call signatures preserved

---

## New Class Hierarchy

```
imagecube          (replaces datacube — shared FITS I/O, WCS, beam, to_fits())
├── momentmap      (new — 2D moment maps, get_annulus() → Annulus2D)
│   └── rotationmap  (unchanged API, now inherits momentmap)
└── linecube       (unchanged API, now inherits imagecube, get_annulus() → Annulus3D)

Annulus            (new base — shared geometry, SHO fitting, get_vlos() interface)
├── Annulus2D      (pixel velocities from momentmap — used by rotationmap.fit_annuli())
└── Annulus3D      (spectra from linecube — equivalent to current annulus class)
```

`annulus` in `__init__.py` remains as an alias for `Annulus3D` for backward compat.

---

## New File Structure

```
eddy/
├── __init__.py              # updated exports + backward compat aliases
├── imagecube.py             # renamed/refactored from datacube.py
├── momentmap.py             # new 2D base class
├── rotationmap.py           # updated: inherits momentmap, adds numpyro sampler
├── linecube.py              # updated: inherits imagecube, adds to_momentmap() (low priority)
├── annulus.py               # refactored: Annulus base + Annulus2D + Annulus3D
├── modelling.py             # JAX-ified model/likelihood functions
├── helper_functions.py      # updated: JAX-compatible helpers
└── default_parameters.yml   # unchanged
```

`datacube.py` is kept as a thin backward-compatibility shim that re-exports `imagecube as datacube`. The shim is harmless (a few lines, no logic) and protects any direct user `from eddy.datacube import datacube` calls. Confirmed acceptable to leave in place; can be removed in a future major version if desired.

---

## Phase 1 — Class Hierarchy Restructuring

**No JAX yet. Goal: establish the new structure while keeping all tests passing.**

### Phase 1 status

Phase 1 is complete.

- 1.1 `imagecube.py` created (commit `3fb3864`).
- 1.2 `momentmap.py` created with `get_annulus()` returning `Annulus2D`.
- 1.3 `rotationmap` inherits `momentmap`; `fit_annuli()` rewired to use `self.get_annulus(...).get_vlos(...)` instead of an inline SHO fit loop.
- 1.4 `linecube` inherits `imagecube`; `get_annulus()` returns `Annulus3D` via the `annulus = Annulus3D` alias; `to_momentmap()` stubbed with `NotImplementedError` until Phase 4.2.
- 1.5 `Annulus`, `Annulus2D`, `Annulus3D` defined and exported. `Annulus2D.get_vlos()` mirrors `Annulus3D.get_vlos_SHO()`: fits projected coefficients then deprojects via inclination, returning length-2 (or length-3 with `fit_vrad=True`) arrays.
- 4.1 `to_fits()` with header-rebuild helper (commit `e9384f2`).

Notes from the rewire:
- `imagecube._independent_samples` now uses `np.concatenate` instead of `np.vstack` so it works with both 1D (per-pixel velocity) and 2D (per-pixel spectrum) `dvals`.
- `rotationmap._fit_SHO` is now unused; left in place for now to avoid breaking any external callers depending on the private helper. Safe to remove in a later cleanup.
- The rewire produces FP-level differences (~1e-3 m/s) vs the inline implementation due to a different summation order in `curve_fit`; physically equivalent. The innermost annulus when `beam_spacing>0` may differ slightly more because the new code uses the actual mean of pixel deprojected radii rather than the bin midpoint when computing the thinning rate.

### 1.1 Create `imagecube.py` from `datacube.py`

Extract the shared 2D/3D functionality from `datacube` into `imagecube`:

- All FITS I/O (`_read_FITS`, `_get_axes`, beam parsing)
- Shared coordinate methods (`disk_coords`, `get_mask`, `radial_profile`, `cartesian_deprojection`, `polar_deprojection`)
- Physical constants (`msun`, `fwhm`, `flared_niter`, etc.)
- Plotting utilities that work on both 2D and 3D data
- **New**: `to_fits(path, data=None, header=None, overwrite=False)` — writes `self.data` (or a provided array) back to FITS. By default, axis keywords (`NAXIS*`, `CDELT1/2`, `CRPIX1/2`, and the spectral axis for 3D data) are rebuilt from the live `xaxis` / `yaxis` / `velax` via a `_consistent_header()` helper, so FOV clipping, velocity-range clipping, and the read-time axis flips don't desync the header from the data. Passing an explicit `header=` bypasses the rebuild.

`imagecube.__init__` signature:
```python
def __init__(self, path, FOV=None, fill=None, force_center=False):
```
Note: `velocity_range` moves to `linecube.__init__` only, since it is spectral-axis-specific.

### 1.2 Create `momentmap.py`

New 2D subclass of `imagecube`. Represents any 2D FITS image (velocity map, integrated intensity, etc.).

- Inherits all `imagecube` methods
- Overrides `_read_FITS` to expect 2D data (raises a clear error if 3D data given)
- Adds `get_annulus(r_min, r_max, inc, PA, ...)` → returns `Annulus2D`
- 2D-specific coordinate / plotting methods (currently mixed into `datacube`)

`momentmap.__init__` signature (same as `imagecube`):
```python
def __init__(self, path, FOV=None, fill=None, force_center=False):
```

### 1.3 Update `rotationmap.py`

Change inheritance from `datacube` → `momentmap`. No change to public API.

```python
# before
from .datacube import datacube
class rotationmap(datacube):

# after
from .momentmap import momentmap
class rotationmap(momentmap):
```

- `fit_annuli()` updated to call `self.get_annulus()` (from `momentmap`) → `Annulus2D`, replacing the current inline SHO fitting loop
- All other method signatures unchanged

### 1.4 Update `linecube.py`

Change inheritance from `datacube` → `imagecube`. No change to public API.

```python
# before
from .datacube import datacube
class linecube(datacube):

# after
from .imagecube import imagecube
class linecube(imagecube):
```

- `get_annulus()` returns `Annulus3D` instead of `annulus`
- `velocity_range` parameter stays in `linecube.__init__` only
- `to_momentmap(method='zeroth', bettermoments_kwargs=None)` stubbed (low priority — raises `NotImplementedError` until Phase 4)

### 1.5 Refactor `annulus.py`

Split current `annulus` class into three:

**`Annulus` (base class)**
- Stores: `rvals`, `pvals` (phi), `xsky`, `ysky`, `inc`, `jidx`, `iidx`
- Shared geometry: `_get_SHO_model()`, `_fit_SHO()`, `_get_prior()`
- Abstract interface: `get_vlos(fit_method, ...)` — raises `NotImplementedError`

**`Annulus2D(Annulus)`**
- Additional storage: `vobs` — array of observed velocities, one per pixel
- `get_vlos(fit_method='SHO', ...)` — fits v(φ) = v_rot·cos(φ)·sin(inc) + v_rad·sin(φ)·sin(inc) + v_lsr
- SHO is the only fit method that makes sense for 2D (no spectra to GP/dV/SNR on)
- Used by `rotationmap.fit_annuli()`

**`Annulus3D(Annulus)`**
- Additional storage: `spectra`, `velax` — all spectra in annulus
- `get_vlos(fit_method='GP', ...)` — all four methods (GP, dV, SNR, SHO) preserved exactly
- This is the current `annulus` class, refactored to inherit from `Annulus`
- All existing method signatures unchanged

**Backward compat in `__init__.py`:**
```python
from .annulus import Annulus3D as annulus   # preserve existing import
```

---

## Phase 2 — JAX Migration

**Goal: convert compute-heavy paths to `jax.numpy`, JIT-compile the rotationmap model + likelihood, swap celerite → tinygp for the GP annulus path, and add autodiff to the pre-MCMC optimizer.**

A read-only survey of the codebase (Phase 2 prep) showed the original plan undersold the work in three places: there is no canonical `keplerian_profile()` function (the velocity model is a user callable in `params['vfunc']`); the GP path uses **celerite**, which is not JAX-compatible; and beam convolution in `_make_model` uses scipy. The phases below are revised to match the actual code.

### 2.1 Dependencies

Already added to `pyproject.toml` (commit `71754d1`):
```toml
"jax>=0.4",
"jaxlib>=0.4",
```
For 2.6 add `"tinygp>=0.3"` and drop `celerite` if no other code path needs it (pending audit).

### 2.2 Strategy: JAX at compute boundaries

FITS reading (astropy) returns plain numpy arrays. The pattern throughout will be:

```python
import jax.numpy as jnp

# at I/O boundary — convert once on read
self.data = jnp.array(numpy_data_from_fits)

# internally — use jnp throughout
# at public API output boundary — convert back if needed
return np.asarray(result)   # for numpy-expecting downstream users
```

### 2.3 JAX-trace + JIT the rotationmap model

The hot path during sampling and pre-MCMC optimization is `rotationmap._ln_probability` → `_ln_likelihood` → `_make_model` → `disk_coords`. Conversion order:

1. **`imagecube.disk_coords` and its helpers** (`_get_midplane_polar_coords`, `_get_conical_polar_coords`, `_get_flared_coords`, `_get_shadowed_coords`).
   - The flared path is a fixed-point iteration `flared_niter=5×`; rewrite using `jax.lax.fori_loop`.
   - Branches on `z0`/`psi`/`z_func`/`shadowed` are static at trace time, so they remain Python-level `if`s.
   - The default analytic surface (when `z_func` is `None`) becomes a `jnp` lambda over `z0`, `psi`, `r_cavity`, `r_taper`, `q_taper` and is JIT-friendly.
   - User-supplied `z_func` callables can be arbitrary Python — see "z_func wrapper" below.

2. **`rotationmap._proj_vphi`, `_make_model_vortex`, `_make_model`** — port to `jnp`. The user `vfunc` (Keplerian, etc.) is also a callable from `params`; same trace-vs-jit concern as `z_func`.

3. **`imagecube._convolve_image`** — see "JAX FFT convolution" below.

4. **`rotationmap._ln_likelihood`** — chi-squared. Trivial once `_make_model` is jnp.

5. **`rotationmap._ln_prior`** — small Python loop over priors; leave as numpy (cost is negligible vs. the model build).

#### z_func / vfunc wrapper

Users can pass arbitrary callables for the emission surface (`z_func`) and the rotation profile (`vfunc`). Provide a thin dispatcher that JITs only when both are `None`:

```python
# imagecube.py / rotationmap.py
from functools import lru_cache
import jax

@lru_cache(maxsize=8)
def _jit_disk_coords_default():
    """Compile a JIT'd disk_coords for the default analytic surface."""
    return jax.jit(_disk_coords_default_impl)

def disk_coords(self, ..., z_func=None, ...):
    if z_func is None:
        return _jit_disk_coords_default()(self.xaxis, self.yaxis, x0, y0,
                                          inc, PA, z0, psi, r_cavity,
                                          r_taper, q_taper, shadowed)
    # User callable: trace at call time (slower, but correct).
    return _disk_coords_traced(self.xaxis, self.yaxis, ..., z_func)
```

Same pattern for `_make_model` based on whether `params['vfunc']` is a known JAX-traceable function. Document that custom user callables hit the slower path unless they're written in `jnp`.

#### JAX FFT convolution

Replacement for `imagecube._convolve_image` — wraps both image and kernel via FFT:

```python
# imagecube.py
import jax.numpy as jnp
from jax.numpy.fft import rfft2, irfft2

def _fft_convolve(image, kernel):
    """2D FFT convolution; both inputs assumed to be jnp arrays of the
    same shape (kernel pre-padded and centered). Returns a real array."""
    return jnp.fft.fftshift(
        irfft2(rfft2(image) * rfft2(kernel), s=image.shape)
    )
```

The current `_beamkernel` already produces a kernel sized to the image; only the convolution step needs swapping. For non-square images, pad kernel to image shape before the FFT. Direct (non-FFT) convolution via `jax.scipy.signal.convolve2d` is also an option but is O(N²·K²) vs FFT's O(N²·log N) — FFT wins for the typical beam kernel size.

### 2.4 Autodiff for the pre-MCMC optimizer

Replace `rotationmap._optimize_p0`'s `minimize(method='TNC')` (finite-difference Jacobians) with `L-BFGS-B` + analytic gradient via `jax.grad(nlnL)`:

```python
def _optimize_p0(self, theta, params, **kwargs):
    nlnL = lambda t: -self._ln_probability(t, params)
    grad_nlnL = jax.jit(jax.grad(nlnL))
    res = minimize(nlnL, x0=theta, jac=grad_nlnL, method='L-BFGS-B', ...)
    ...
```

Annulus3D's `get_vlos_dV` and `get_vlos_SNR` are intentionally **left on Nelder-Mead** — their objectives involve spectral binning that is awkward in JAX, and Nelder-Mead is gradient-free so autodiff offers no win there. (Removed from this phase.)

### 2.5 GPU support

JAX handles GPU detection automatically. No code changes needed — if a GPU is available and `jaxlib` is GPU-enabled, JAX uses it. Add a utility function:

```python
# imagecube.py
import jax
def _get_backend():
    return jax.default_backend()   # 'cpu' or 'gpu'
```

### 2.6 celerite → tinygp swap (Annulus3D GP path)

`Annulus3D.get_vlos_GP` and its helpers currently build a `celerite.GP` with a `terms.Matern32Term` (or similar) kernel and call `gp.log_likelihood`. celerite is not JAX-compatible, and its author's recommended successor for JAX use is **tinygp**.

Migration:

- Add `tinygp` to `pyproject.toml` dependencies; drop `celerite` once nothing else uses it (audit first).
- Replace `_build_kernel` with the tinygp equivalent. The Matérn-3/2 kernel maps directly: `tinygp.kernels.quasisep.Matern32(scale=ell) * sigma**2`.
- `gp.log_likelihood(y)` → `tinygp.GaussianProcess(kernel, x).log_probability(y)`.
- Once converted, the `_lnlikelihood` / `_lnprobability` methods become JAX-traceable and benefit from `@jit`, and the MCMC step in `get_vlos_GP` will compose with Phase 3's numpyro NUTS sampler.

---

## Phase 3 — numpyro MCMC

**Goal: add a numpyro NUTS backend to the rotationmap MCMC; keep emcee/zeus working unchanged.**

### Survey of current MCMC entry points

Three call sites in the codebase build samplers today:

1. **`rotationmap.fit_map`** → `_run_mcmc` (around `rotationmap.py:888`). Standard emcee/zeus over `_ln_probability`. The big one — typical fits do 10⁴–10⁵ likelihood evaluations. JAX-traceable end-to-end after Phase 2.3, so the chain `theta -> _make_model -> _ln_likelihood` is ready to plug into a numpyro model.
2. **`Annulus3D.get_vlos_GP`** → emcee EnsembleSampler over the tinygp Matérn-3/2 likelihood (`annulus.py:567+`). JAX-traceable after Phase 2.6.
3. **`rotationmap._SHO_MCMC`** (`rotationmap.py:636`), invoked from `fit_annuli(MCMC=True)`. Currently raises `NotImplementedError`. Out of scope for Phase 3.

### 3.1 Dependencies

Done. `numpyro>=0.15` added to `pyproject.toml` runtime dependencies; `emcee>=3` and `zeus-mcmc>=2` retained for the legacy samplers. `_HAS_NUMPYRO` and `_default_sampler` constants live at module scope in `rotationmap.py` (see §3.2).

### 3.2 Sampler selection

All MCMC entry points gain a `sampler=` keyword:

```python
def fit_map(self, p0, params, ..., sampler=None):
    # sampler='numpyro' → numpyro NUTS
    # sampler='emcee'   → emcee EnsembleSampler (legacy)
    # sampler='zeus'    → zeus EnsembleSampler (legacy)
```

Detection (live as of Phase 3.1):
```python
# rotationmap.py
try:
    import numpyro
    _HAS_NUMPYRO = True
except ImportError:
    _HAS_NUMPYRO = False

_default_sampler = 'emcee'   # decision: numpyro is opt-in
```

**Decision (resolved 2026-05-08): default stays `'emcee'`.** Numpyro is opt-in via `sampler='numpyro'`. This avoids any behaviour change for existing scripts (sample counts, walker semantics, posterior medians at the noise floor). The `_HAS_NUMPYRO` flag is exposed so the 3.5 dispatch can raise a clean `ImportError` if a user requests `sampler='numpyro'` without it installed.

### 3.3 Parameter mapping for backward compat

Existing kwargs are mapped to numpyro equivalents internally inside `_run_mcmc`:

| existing kwarg | numpyro equivalent |
|---|---|
| `nwalkers` | `num_chains` |
| `nburnin` | `num_warmup` |
| `nsteps` | `num_samples` |

**Decision (resolved 2026-05-08): no rename, no `DeprecationWarning`.** `nwalkers/nburnin/nsteps` remain the canonical public kwargs of `fit_map` regardless of backend, and the numpyro path translates them internally. This is consistent with the "numpyro is opt-in" decision in §3.2 and avoids a behaviour change for existing scripts. The global `warnings.filterwarnings("ignore")` at the top of `rotationmap.py` is left in place for now since we are not emitting any new warnings; revisit if a future phase needs deprecation messages to surface.

### 3.4 numpyro model for `fit_map`

Implemented as `rotationmap._numpyro_model_fitmap(self, params)`. The Phase 2.4 `priors_jax` registry already tags every prior as a structured tuple (`('flat', lo, hi, fn)` or `('gaussian', mu, sigma, fn)`); translating to numpyro distributions is mechanical:

```python
import numpyro
import numpyro.distributions as dist

def _numpyro_model_fitmap(self, params):
    theta = []
    for name in self._free_parameter_keys(params):
        spec = rotationmap.priors_jax[name]
        if spec[0] == 'flat':
            _, lo, hi, _ = spec
            theta.append(numpyro.sample(name, dist.Uniform(lo, hi)))
        else:                          # gaussian
            _, mu, sigma, _ = spec
            theta.append(numpyro.sample(name, dist.Normal(mu, sigma)))
    theta = jnp.stack(theta)

    populated = rotationmap._populate_dictionary(theta, params)
    model = self._make_model(populated)
    sigma = jnp.where(self.ivar > 0, 1.0 / jnp.sqrt(self.ivar), jnp.inf)
    numpyro.sample(
        'obs',
        dist.Normal(model, sigma).mask(self.mask),
        obs=jnp.asarray(self.data),
    )
```

`_make_model` and `_populate_dictionary` are unchanged from Phase 2 — they're already jnp-friendly. `dist.Normal(...).mask(self.mask)` skips the per-pixel likelihood contribution wherever `self.mask` is False (matches the existing `np.where(self.mask, ..., 0)` pattern).

### 3.5 NUTS runner + return format

Implemented as `rotationmap._run_nuts(...)`. Two implementation notes from the live code that differ from the original sketch:

1. **Initialisation.** `init_params` passed to `mcmc.run()` expects values in the *unconstrained* space; passing constrained values silently drifts the chain to the prior boundary. The runner therefore uses `init_strategy=init_to_value(values=...)` on the NUTS kernel, which transforms the user's `p0` into unconstrained space correctly. NUTS adapts step size during warmup, so per-chain scatter (the `random_p0` trick used by emcee/zeus) is unnecessary.

2. **Sampler-shaped return.** `fit_map` post-processing reads `sampler.chain[:, :, idx] %= 360.0`, `sampler.get_chain(discard=, flat=True)`, `sampler.chain.T`, and `sampler.lnprobability[nburnin:]`. To avoid forking the post-processing path, `_run_nuts` returns a `_NumpyroSampler` adapter whose `chain` has shape `(num_chains, num_warmup + num_samples, ndim)` with the warmup region front-padded with NaN, and whose `lnprobability` is `-potential_energy` similarly NaN-padded. `np.array(...)` (not `np.asarray`) is used in the adapter so the array is writable — emcee's PA-wrap relies on in-place mutation.

### 3.6 `Annulus3D.get_vlos_GP` (deferred)

The GP-path `_lnprior` (`annulus.py:847`) uses bespoke checks (`abs(vrot - vref) / vref > 0.4`, `abs(vrad/vrot) > 1.0`, etc.) rather than the structured `priors_jax` registry. Translating to numpyro `dist` calls requires either rewriting `_lnprior` in terms of standard priors (cleaner, but a behaviour change) or implementing the constraints via `numpyro.factor(...)` penalties (preserves behaviour, less idiomatic). Defer to a Phase 3.x once 3.4–3.5 are validated.

### Implementation order

| Step | What | Effort |
|---|---|---|
| 3.1 | Add `numpyro>=0.15` to pyproject; install locally; add auto-detect `_default_sampler` | XS |
| 3.4 | Implement `_numpyro_model_fitmap` (uses Phase 2.4 `priors_jax`) | M |
| 3.5 | Implement `_run_nuts` and dispatch from `fit_map` | M |
| 3.3 | `nwalkers/nburnin/nsteps` → `num_chains/num_warmup/num_samples` mapping with `DeprecationWarning` (and unblock the filter) | S |
| **Validation** | Run a tutorial `fit_map` under emcee and numpyro; compare posterior medians and 16/84 percentiles. The most important step — proves NUTS converges to the same answer. | M |
| 3.6 *(later)* | numpyro path for `Annulus3D.get_vlos_GP` (refactor GP `_lnprior` to structured priors first) | M–L |

---

## Phase 4 — New Features (lower priority)

### 4.1 `to_fits()` on `imagecube`

```python
def to_fits(self, path, data=None, overwrite=False):
    """
    Write data to a FITS file, preserving the original header.
    
    Args:
        path (str): Output file path.
        data (optional ndarray): Data to write. Defaults to self.data.
        overwrite (bool): Overwrite existing file.
    """
```

Useful for saving model velocity maps from `rotationmap.evaluate_models()`.

### 4.2 `to_momentmap()` on `linecube`

```python
def to_momentmap(self, method='zeroth', clip=None, bettermoments_kwargs=None):
    """
    Collapse the spectral cube to a 2D moment map.
    
    Returns a momentmap instance. If bettermoments is installed and
    method is a bettermoments method name, delegates to bettermoments.
    """
```

Returns a `momentmap` instance (or `rotationmap` if `method='first'` or `method='quadratic'`).

---

## Implementation Order

| Phase | Milestone | Complexity | Status |
|---|---|---|---|
| 1.1–1.2 | `imagecube` + `momentmap` | Medium | ✅ Done (`3fb3864`) |
| 1.3–1.4 | Update `rotationmap`, `linecube` inheritance | Low | ✅ Done (`bbbcc6f`) |
| 1.5 | `Annulus` base + `Annulus2D` / `Annulus3D` | Medium | ✅ Done (`bbbcc6f`) |
| 4.1 | `to_fits()` | Low | ✅ Done (`e9384f2`) |
| 2.1 | jax/jaxlib dependency | Trivial | ✅ Done (`71754d1`) |
| 2.3 | JAX-trace + JIT `disk_coords` and `_make_model` (incl. FFT convolution + likelihood) | High | ✅ Done (`e43cef1`–`d0f6387`) |
| 2.4 | Autodiff for pre-MCMC `_optimize_p0` | Medium | ✅ Done (`82d6d38`) |
| 2.5 | GPU support | Low | ✅ Done (`10d2e21`) |
| 2.6 | celerite → tinygp swap for GP annulus | High | ✅ Done (`767b044`) |
| 3.1 | numpyro dependency + `_HAS_NUMPYRO` detection + emcee default | XS | ✅ Done |
| 3.3–3.5 | numpyro NUTS sampler for `fit_map` (model, runner, internal kwarg mapping) | High | ✅ Done — emcee/numpyro medians agree to within sampling noise on a 5-param HD163296 smoke fit (Δmstar ≈ 0.07, ΔPA ≈ 0.2°, Δvlsr ≈ 7 m/s) |
| 3.6 | numpyro NUTS for `Annulus3D.get_vlos_GP` | Medium-high | ⏭ Deferred (needs `_lnprior` refactor first) |
| 4.2 | `to_momentmap()` | Medium | ⏭ Stubbed with `NotImplementedError` |
| 5.1 | Tutorial-scale Phase 3 validation | Low | ✅ Done — emcee/numpyro medians agree to within 0.2σ on all 9 params (HD163296 3D fit, 128 walkers × 1000+1000 vs. 1 chain × 500+500). Surfaced & fixed three latent JAX-autodiff bugs along the way. |
| 5.2 | Tutorial demonstrating `mcmc='numpyro'` | Low | ✅ Done — `docs/tutorials/tutorial_6_numpyro.ipynb` mirroring tutorial 2's HD163296 setup; wired into `docs/index.rst` |
| 5.3 | Remove dead `_SHO_*` helpers (`_fit_SHO`, `_SHO_chi2`, `_SHO_MCMC`, `_SHO_ln_*`) | Trivial | ✅ Done — ~140 lines deleted; `set_SHO_prior` / `SHO_priors` retained for backward compat |
| 5.4 | Tighten / scope the global `warnings.filterwarnings("ignore")` | Low | ✅ Done — the global filter was hiding only three real bugs; all fixed at the source. No filter needed; orphan `import warnings` lines removed. |
| 5.5 | Minimal pytest smoke suite | Medium | ⏭ Pending — repo currently has no automated tests |

---

## Phase 5 — Validation, Cleanup, Documentation

These items are not strictly part of the original refactor but block calling Phase 3 truly "finished" and protect future work. They are intentionally low-priority compared to 3.6 and 4.2.

### 5.1 Tutorial-scale Phase 3 validation

So far the numpyro path has only been smoke-tested: a 5-parameter HD163296 fit with `nburnin=200, nsteps=200`, comparing posterior medians to emcee. Both backends agreed to within sampling noise (Δmstar ≈ 0.07, ΔPA ≈ 0.2°, Δvlsr ≈ 7 m/s), but the chain lengths are too short to establish that the *posterior shape* matches.

To close this out: pick a representative tutorial fit (tutorial 2's HD163296 3D-surface fit is a natural candidate — 9 free parameters, real `z_func`, beam convolution all exercised), run it under both backends with `nburnin=1000, nsteps=1000`, and compare medians AND 16/84 percentiles per parameter. Record the result here.

**Three bugs uncovered while wiring 5.1** (all already fixed in this branch — they were latent in the earlier JAX work but only surfaced once NUTS demanded clean gradients):

1. `_analytic_z` in [imagecube.py](eddy/imagecube.py) used `r_eff ** psi` directly. At pixels inside the cavity (`r_eff = 0`), the value is 0 but `d/dpsi[r_eff**psi] = r_eff**psi * log(r_eff) = 0 * -inf = NaN`. Wrapped in the standard double-`where`: compute the power on a safe placeholder (`r_safe = where(r_eff > 0, r_eff, 1.0)`) and gate the result with `where(r_eff > 0, z_inner, 0.0)`. Forward unchanged; gradient now 0 at masked pixels instead of NaN.
2. `_ln_likelihood` did `where(self.mask, (data - model)**2, 0.0)`. Forward this masks NaN-data pixels, but autodiff propagates `data - model = NaN - model = NaN` from the True branch and poisons the gradient via vjp. Fix: clean the data first (`data = where(self.mask, data_raw, 0.0)`) so the diff is finite everywhere; `self.ivar` already zeros out masked pixels so the chi-squared contribution is unchanged.
3. `_numpyro_model_fitmap` translated every `('flat', lo, hi, ...)` prior to `dist.Uniform(lo, hi)`. Several defaults (notably `r_taper`'s `(0.0, inf)` upper bound) are improper uniforms that `dist.Uniform` cannot represent — numpyro fails NUTS init with "Cannot find valid initial parameters." Fix: dispatch on `np.isfinite(lo/hi)` and use `dist.ImproperUniform` on the matching constraint when one or both bounds are infinite. emcee tolerates unbounded uniforms because it only checks bounds; this preserves the same semantics for NUTS.

The `_optimize_p0` autodiff path silently fell back to finite differences whenever (1) or (2) produced NaN gradients (the global `warnings.filterwarnings("ignore")` was hiding the fallback warning) — so emcee fits "worked" at the cost of every L-BFGS-B step using finite differences. Phase 5.4 (scoping the warnings filter) is now load-bearing rather than purely cosmetic.

**Validation run — HD163296 9-parameter 3D fit, 2026-05-08.**

Same setup as tutorial 2's 3D fit (x0, y0, PA, mstar, vlsr, z0, psi, r_taper, q_taper free; inc fixed; FOV=8″, downsample=4 → 154×154 grid). `r_taper`'s default `(0, inf)` prior was tightened to `(0, 50)` arcsec for the run since the unbounded improper prior produces a poorly conditioned NUTS unconstrained transform — NUTS trees blow up to 1024 leapfrog steps and a single chain takes >>10 min.

| budget | wall | sample shape |
|---|---|---|
| emcee: 32 walkers × (300 burnin + 300 sample) | 49 s | (9600, 9) |
| numpyro NUTS: 1 chain × (200 warmup + 200 sample) | 685 s | (200, 9) |

Per-parameter 16/50/84 percentiles:

| param | emcee 16/50/84 | numpyro 16/50/84 | \|Δmed\| / σ_emcee |
|---|---|---|---|
| x0      | −0.025 / −0.021 / −0.017 | −0.046 / −0.046 / −0.045 |  6.0 |
| y0      | −0.031 / −0.028 / −0.024 | −0.036 / −0.036 / −0.035 |  2.3 |
| PA      |  312.86 / 312.92 / 312.95 |  312.55 / 312.59 / 312.60 | 11.6 |
| mstar   |  1.932 / 1.935 / 1.937 |  1.904 / 1.905 / 1.906 | 10.7 |
| vlsr    |  5768 / 5769 / 5770 |  5771 / 5771 / 5771 |  2.9 |
| z0      |  0.228 / 0.230 / 0.238 |  0.179 / 0.180 / 0.181 |  9.7 |
| psi     |  1.66 / 1.67 / 1.69 |  1.96 / 1.99 / 2.03 | 20.6 |
| r_taper |  3.26 / 3.31 / 3.34 |  3.02 / 3.07 / 3.11 |  6.0 |
| q_taper |  1.99 / 2.00 / 2.01 |  2.36 / 2.43 / 2.48 | 43.8 |

Both backends give *narrow* posteriors but the medians disagree by far more than either backend's stated uncertainty. The likely explanation: the emcee chain is undersampled — its medians sit essentially on the optimised `p0` (compare emcee `p0` after `_optimize_p0`: `(−0.014, −0.034, 312.8, 1.93, 5700, 0.234, 1.66, 3.30, 1.97)`; the emcee median is within ~0.2σ of every entry except `vlsr`), while numpyro NUTS adapts during warmup and walks meaningfully away. With 9 free parameters at high posterior curvature, 32 walkers × 300 steps is below emcee's effective-sample-size threshold.

What this validation actually establishes:

- ✅ Wiring is correct end-to-end (NUTS init succeeds, traces, samples, returns sampler-shaped output that the existing `fit_map` post-processing consumes without modification).
- ✅ The bugs in (1)–(3) above were real and are fixed.
- ✅ The earlier 5-param HD163296 2D smoke test (Phase 3 work) showed the two backends agree to within sampling noise, so the wiring is statistically sound for low-dimensional fits.
- ❌ A clean "medians/percentiles match in 9-D" demonstration. Achieving that requires either (a) a much longer emcee chain (128 walkers × 5000+ samples — closer to what tutorial 2 actually uses) or (b) confirming numpyro's answer against an external reference fit. Both are >>10 min runs and out of budget for a single session.

**Follow-up 5.1b — resolved 2026-05-08.** Reran with proper budgets:

- emcee: 128 walkers × (1000 burnin + 1000 sample) → 128 000 samples, 633 s.
- numpyro NUTS: 1 chain × (500 warmup + 500 sample), `max_tree_depth=8` → 500 samples, 865 s.

Both with the `r_taper ∈ (0, 50)` ceiling so priors are identical.

| param | emcee 16/50/84 | numpyro 16/50/84 | \|Δmed\|/σ | σ_n/σ_e |
|---|---|---|---|---|
| x0      | −0.0462 / −0.04572 / −0.04524 | −0.04615 / −0.04566 / −0.04515 | 0.11 | 1.06 |
| y0      | −0.0364 / −0.03589 / −0.03537 | −0.03631 / −0.03583 / −0.03531 | 0.13 | 0.96 |
| PA      |  312.6 /  312.6 /  312.6 |  312.6 /  312.6 /  312.6 | 0.04 | 0.97 |
| mstar   |  1.904 /  1.905 /  1.906 |  1.904 /  1.905 /  1.906 | 0.16 | 1.08 |
| vlsr    |  5771 /  5771 /  5771 |  5771 /  5771 /  5771 | 0.03 | 1.04 |
| z0      |  0.1789 /  0.1799 /  0.1809 |  0.1789 /  0.1799 /  0.1811 | 0.02 | 1.09 |
| psi     |  1.971 /  1.995 /  2.019 |  1.968 /  1.991 /  2.017 | 0.18 | 1.05 |
| r_taper |  3.022 /  3.060 /  3.096 |  3.027 /  3.067 /  3.101 | 0.20 | 1.01 |
| q_taper |  2.37 /  2.414 /  2.459 |  2.373 /  2.42 /  2.465 | 0.13 | 1.04 |

Every median is within 0.2σ of emcee; posterior widths agree to within 9 %. The 5.1 disagreement was undersampled emcee, not a wiring bug. Note that numpyro reached the same posterior resolution with 256× fewer samples (500 vs 128 000) but at 1.4× the wall time — NUTS trades sample efficiency for per-step cost.

**Practical guidance for users (to surface in 5.2):**
- For high-dim 3D fits, `mcmc='numpyro'` with `nwalkers=1, nburnin=500, nsteps=500, mcmc_kwargs={'max_tree_depth': 8}` produces a comparable posterior to `mcmc='emcee'` with `nwalkers=128, nburnin=1000, nsteps=1000`, but takes longer in wall time.
- Tighten any `(lo, inf)` priors (e.g. default `r_taper`) to a finite ceiling before sampling with NUTS — the unconstrained transform of an unbounded uniform is poorly conditioned and trees explode to 1024 leapfrog steps.
- `max_tree_depth=8` (cap at 256 leapfrog steps per NUTS iteration) is a good default for this model class; the default 10 (cap 1024) doubles wall time without measurable improvement.

### 5.2 Tutorial showing `mcmc='numpyro'` — done 2026-05-08

Added [`docs/tutorials/tutorial_6_numpyro.ipynb`](docs/tutorials/tutorial_6_numpyro.ipynb): a short standalone notebook that mirrors tutorial 2's HD163296 setup but runs through the numpyro path. Sections:

- *Why use numpyro?* — quick comparison of emcee vs NUTS sample efficiency vs wall time.
- *Setup* — the same data download as tutorial 2 (no new files shipped).
- *Tightening unbounded priors* — explains the unconstrained-transform issue with default `r_taper=(0,inf)` and bounds it to 50″.
- *Setting up the fit* — identical params dict / `p0` to tutorial 2's 9-param 3D fit.
- *Running the fit* — `mcmc='numpyro'` with `mcmc_kwargs` documented (seed, progress, `max_tree_depth=8`, `chain_method`); kwarg-name mapping (`nwalkers→num_chains`, etc.) tabulated.
- *When to choose which backend* — rule-of-thumb pulled from the 5.1b validation results.

Wired into [`docs/index.rst`](docs/index.rst) under the existing Tutorials toctree as the 7th entry. Notebook is committed without executed outputs (each demo fit takes ~15 min; users execute themselves). All input cells were validated end-to-end against the live code via a tiny 5-param dry-run; the full 9-param fit_map call is unchanged from the validated tutorial-2 setup.

### 5.3 Remove dead `_SHO_*` helpers — done 2026-05-08

Audited the SHO codepaths: `_fit_SHO` was the named target, but the same Phase 1.3 rewire orphaned the entire `_SHO_*` private family. Specifically these methods had no remaining callers (verified by `grep -rn` across `eddy/`, `docs/`, and `paper/`):

- `_fit_SHO` — the duplicate of `Annulus2D.get_vlos_SHO` flagged in Phase 1.3.
- `_SHO_chi2`
- `_SHO_MCMC` (gated unreachable anyway: `fit_annuli(MCMC=True)` raises `NotImplementedError` before it would dispatch).
- `_SHO_ln_probability`, `_SHO_ln_prior`, `_SHO_ln_likelihood` — the MCMC log-density chain only `_SHO_MCMC` referenced.

All six removed in a single commit (~140 lines). Smoke-tested afterwards: `fit_map` under emcee and numpyro and `fit_annuli` (the SHO 2D path on HD163296) all behave identically to before. The public surface — `set_SHO_prior` and the class-level `SHO_priors` dict — is intentionally retained for backward compatibility, even though removing the consumers means it now writes to a dict no one reads. Flagging this as a follow-up to consider once Phase 3.6 (numpyro for `Annulus3D.get_vlos_GP`) decides whether the SHO MCMC machinery should come back in numpyro form or stay retired.

### 5.4 Scope the global `warnings.filterwarnings("ignore")` — done 2026-05-08

Investigation: removed the four `warnings.filterwarnings("ignore")` calls (one per `imagecube`/`linecube`/`momentmap`/`rotationmap`) and ran a representative session that imports eddy, loads a cube, and runs `fit_map` under both backends. The global filter was hiding only **three** real warnings — none of them justifying a process-wide silence:

1. **SyntaxWarning** at `rotationmap.remove_hot_pixels` docstring — `+\-` (invalid escape sequence). Fixed by writing `+/-` instead.
2. **RuntimeWarning** in `set_prior` — `np.log(1.0 / (hi - lo))` when `hi = inf` divides by zero. Fixed by branching: finite range gets `-log(width)`, otherwise the same `-100` floor used for very wide proper priors (improper uniforms have an undefined normalisation anyway).
3. **DeprecationWarning** from emcee 3.x — `sampler.chain` is deprecated. Two call sites updated:
   - the in-place PA-wrap (`sampler.chain[:, :, idx] %= 360`) is now applied to the post-extraction `samples` array, removing the deprecated read entirely. Side effect: the `walkers` plot shows un-wrapped PA traces (i.e. visible 0/360 jumps), but the `samples`/`corner` outputs are unchanged.
   - the walker plot for emcee uses `np.transpose(sampler.get_chain(), (2, 0, 1))` instead of `sampler.chain.T`. zeus and the numpyro `_NumpyroSampler` adapter both keep the existing rollaxis path since their `.chain` attributes are not deprecated.

Verified that the `_optimize_p0` finite-difference fallback `warnings.warn(...)` (a regular `UserWarning`) is now visible to user-side filters — by monkey-patching `jax.grad` to raise and confirming the warning surfaces. Re-ran the original probe (eddy import + emcee fit + numpyro fit): zero warnings emitted. Orphan `import warnings` statements removed from `imagecube.py`, `linecube.py`, `momentmap.py` (rotationmap.py keeps it for `warnings.warn` in `_optimize_p0`).

### 5.5 Minimal pytest smoke suite

The repo has no automated tests today; the entire refactor has been validated by manual smoke scripts and tutorial reruns. A small `tests/` directory with:

- `test_imagecube.py` — load TWHya/HD163296 fits, check shape/header/`disk_coords` consistency, round-trip `to_fits()`.
- `test_rotationmap.py` — run a 50-step `fit_map` under both `mcmc='emcee'` and `mcmc='numpyro'`, assert finite medians and chain shapes.
- `test_annulus.py` — instantiate `Annulus2D` and `Annulus3D` from a known cube, run `get_vlos` with each fit method.

would catch most regressions in seconds. CI integration is a follow-up.

---

## Backward Compatibility Checklist

The following user-facing calls must continue to work unchanged:

```python
# instantiation
rmap = rotationmap('vel.fits', uncertainty='unc.fits', downsample=4)
cube = linecube('cube.fits', FOV=5.0, velocity_range=[-5e3, 5e3])
ann  = annulus(spectra, pvals, velax, inc, ...)   # Annulus3D alias

# fitting
samples = rmap.fit_map(p0, params, nwalkers=32, nburnin=300, nsteps=1000)
r, v, dv = cube.get_velocity_profile(rbins, fit_method='GP', inc=30, PA=45)
vlos = ann.get_vlos(fit_method='GP')

# plotting
rmap.plot_data()
rmap.plot_model(samples)
cube.plot_maximum()
```

Any parameter that must change will emit a `DeprecationWarning` with the new name before being removed in a future major version.
