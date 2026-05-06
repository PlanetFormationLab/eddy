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

**Goal: replace emcee/zeus with numpyro NUTS as default; keep emcee as fallback.**

### 3.1 Dependencies

Add to `pyproject.toml`:
```toml
"numpyro>=0.15",
```
Keep `emcee>=3` and `zeus-mcmc>=2` as optional fallbacks.

### 3.2 Sampler selection

All MCMC entry points (`fit_map`, `get_vlos`) gain a `sampler` keyword:

```python
def fit_map(self, p0, params, ..., sampler='numpyro'):
    # sampler='numpyro'  → numpyro NUTS (default if installed)
    # sampler='emcee'    → emcee EnsembleSampler (existing code path)
    # sampler='zeus'     → zeus EnsembleSampler (existing code path)
```

Auto-detection fallback:
```python
try:
    import numpyro
    _default_sampler = 'numpyro'
except ImportError:
    _default_sampler = 'emcee'
```

### 3.3 Parameter mapping for backward compat

Existing kwargs are mapped to numpyro equivalents internally:

| existing kwarg | numpyro equivalent |
|---|---|
| `nwalkers` | `num_chains` |
| `nburnin` | `num_warmup` |
| `nsteps` | `num_samples` |

Users who pass `nwalkers=32, nburnin=300, nsteps=1000` get the same behavior. A `DeprecationWarning` is raised pointing to the new names, but the old names keep working.

### 3.4 numpyro model definition

The current `params` dict (which maps parameter names to either a free index in `p0` or a fixed value) translates naturally to numpyro:

```python
import numpyro
import numpyro.distributions as dist

def _numpyro_model(self, params, data, uncertainty):
    p = {}
    for name, val in params.items():
        if isinstance(val, int):   # free parameter — sample from prior
            prior = self._get_prior(name)
            p[name] = numpyro.sample(name, prior)
        else:                      # fixed — deterministic
            p[name] = numpyro.deterministic(name, val)
    
    model = self._get_velocity_model(**p)   # JAX-jitted
    numpyro.sample('obs', dist.Normal(model, uncertainty), obs=data)
```

Priors are read from `default_parameters.yml` exactly as now.

### 3.5 Return format

The return format from `fit_map` and `get_vlos` is unchanged: a numpy array of samples with shape `(nsteps * nwalkers, ndim)`. With numpyro, samples from all chains are concatenated to produce the same shape.

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

| Phase | Milestone | Complexity | Impact |
|---|---|---|---|
| 1.1–1.2 | `imagecube` + `momentmap` | Medium | Foundation for everything |
| 1.3–1.4 | Update `rotationmap`, `linecube` inheritance | Low | Clean up hierarchy |
| 1.5 | `Annulus` base + `Annulus2D` / `Annulus3D` | Medium | Enables 2D annulus fitting |
| 4.1 | `to_fits()` | Low | Immediate user value |
| 2.1 | jax/jaxlib dependency | Trivial | Enables 2.3+ |
| 2.3 | JAX-trace + JIT `disk_coords` and `_make_model` (incl. FFT convolution) | High | Foundation for autodiff and GPU |
| 2.4 | Autodiff for pre-MCMC `_optimize_p0` | Medium | Faster `fit_map` init |
| 2.6 | celerite → tinygp swap for GP annulus | High | Unblocks JIT for GP path; numpyro-friendly |
| 3.1–3.5 | numpyro NUTS sampler | High | Core sampling improvement |
| 2.5 | GPU support | Low | Free via JAX |
| 4.2 | `to_momentmap()` | Medium | Convenience |

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
