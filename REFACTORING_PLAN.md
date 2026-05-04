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

Done:
- 1.1 `imagecube.py` created (commit `3fb3864`).
- 1.2 `momentmap.py` created and inherits `imagecube` (commit `3fb3864`).
- 1.3 `rotationmap` now inherits `momentmap` (commit `3fb3864`).
- 1.4 `linecube` inherits `imagecube`; `get_annulus()` returns `Annulus3D` via the `annulus = Annulus3D` alias (commit `3fb3864`).
- 1.5 `Annulus` base + `Annulus2D` + `Annulus3D` defined and exported (commit `3fb3864`).
- 4.1 `to_fits()` with header-rebuild helper (commit `e9384f2`).

Still open:
- **1.2 `momentmap.get_annulus()`** — the method itself isn't implemented yet. Should return an `Annulus2D` instance built from the 2D map.
- **1.3 `rotationmap.fit_annuli()` rewiring** — currently still calls the inline `self._fit_SHO(...)` loop at `rotationmap.py:191`. Plan calls for replacing this with `self.get_annulus(...).get_vlos(...)` once `momentmap.get_annulus()` lands.
- **1.4 `linecube.to_momentmap()` stub** — add a method that raises `NotImplementedError` until Phase 4.2 is implemented, so the public API surface is in place.

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

**Goal: replace numpy with jax.numpy in compute-heavy paths; JIT-compile hot functions.**

### 2.1 Dependencies

Add to `pyproject.toml`:
```toml
"jax>=0.4",
"jaxlib>=0.4",
```
Keep `numpy` as a dependency (needed for FITS I/O boundary via astropy).

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

### 2.3 JIT-compile hot functions

In `modelling.py`, wrap all model evaluation with `@jax.jit`:
- `keplerian_profile()` — disk rotation model
- `_get_velocity_model()` in `rotationmap`
- Likelihood functions in `rotationmap.fit_map()` and `Annulus3D.get_vlos_GP()`

In `imagecube.py`, JIT coordinate transforms:
- `disk_coords()`
- `_get_mask()`

### 2.4 Autodiff for optimization

Replace `scipy.optimize.minimize` (which uses finite-difference Jacobians) with JAX-native optimization using `jax.grad`:

```python
# before — finite-difference gradients
from scipy.optimize import minimize
result = minimize(nll, p0, method='Nelder-Mead')

# after — exact analytic gradients via autodiff
from jax import grad, jit
grad_nll = jit(grad(nll))
# use with scipy L-BFGS-B (accepts analytic gradients) or optax
result = minimize(nll, p0, jac=grad_nll, method='L-BFGS-B')
```

This replaces the pre-MCMC `optimize=True` step in `fit_map` and the `scipy.optimize.minimize` calls in `get_vlos_dV` and `get_vlos_SNR`.

### 2.5 GPU support

JAX handles GPU detection automatically. No code changes needed — if a GPU is available and `jaxlib` is GPU-enabled, JAX uses it. Add a utility function:

```python
# imagecube.py
import jax
def _get_backend():
    return jax.default_backend()   # 'cpu' or 'gpu'
```

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
| 2.1–2.3 | JAX backend in `modelling.py` | High | Perf + autodiff |
| 2.4 | Autodiff optimization | Medium | Faster `fit_map` init |
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
