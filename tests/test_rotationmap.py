"""Smoke tests for :class:`eddy.rotationmap`.

Runs a tiny ``fit_map`` under both ``mcmc='emcee'`` and
``mcmc='numpyro'``. We do not check posterior agreement here (Phase 5.1b
already validated that to within sampling noise); these tests guard
against regressions in:

- The free-parameter wiring (``_free_parameter_keys`` / labels / p0 length).
- The ``_optimize_p0`` autodiff path (NaN gradients silently fell back
  to finite differences before Phase 5.1's bug fixes).
- The numpyro adapter return shape consumed by ``fit_map``'s
  post-processing.

To keep wall time low we use a 5-parameter (no flared surface) fit on
the FOV=8″ ÷ 4 downsampled HD163296 cube and a small step budget.
"""

import numpy as np
import pytest


def _five_param_setup():
    params = {
        'x0': 0,
        'y0': 1,
        'PA': 2,
        'mstar': 3,
        'vlsr': 4,
        'inc': 46.7,
        'dist': 101.0,
    }
    p0 = [0.0, 0.0, 312.0, 2.0, 5.7e3]
    return p0, params


def test_fit_map_emcee_smoke(hd163296_rotationmap):
    p0, params = _five_param_setup()
    samples = hd163296_rotationmap.fit_map(
        p0=p0, params=dict(params), optimize=True,
        nwalkers=16, nburnin=20, nsteps=20,
        mcmc='emcee',
        plots=['none'],
        returns=['samples'],
        mcmc_kwargs={'progress': False},
    )
    assert samples.ndim == 2
    assert samples.shape[1] == len(p0)
    assert np.all(np.isfinite(samples))


def test_fit_map_numpyro_smoke(hd163296_rotationmap):
    pytest.importorskip("numpyro")
    p0, params = _five_param_setup()
    samples = hd163296_rotationmap.fit_map(
        p0=p0, params=dict(params), optimize=True,
        nwalkers=1,         # num_chains
        nburnin=20,         # num_warmup
        nsteps=20,          # num_samples
        mcmc='numpyro',
        plots=['none'],
        returns=['samples'],
        mcmc_kwargs={'progress': False, 'seed': 0, 'max_tree_depth': 6},
    )
    assert samples.ndim == 2
    assert samples.shape[1] == len(p0)
    assert np.all(np.isfinite(samples))


def test_fit_map_returns_percentiles(hd163296_rotationmap):
    p0, params = _five_param_setup()
    pcts = hd163296_rotationmap.fit_map(
        p0=p0, params=dict(params), optimize=False,
        nwalkers=16, nburnin=10, nsteps=10,
        mcmc='emcee',
        plots=['none'],
        returns=['percentiles'],
        mcmc_kwargs={'progress': False},
    )
    # 16/50/84 percentiles for 5 free params.
    assert pcts.shape == (3, len(p0))
    assert np.all(np.isfinite(pcts))


def test_make_model_includes_radial_velocity(hd163296_rotationmap):
    """``vr_100``/``vr_q`` must perturb ``_make_model``'s output. Guards
    against a repeat of the regression where the radial velocity term was
    silently dropped from the sky-projected model."""
    base_params = {
        'x0': 0.0, 'y0': 0.0, 'PA': 312.0, 'mstar': 2.0, 'vlsr': 5.7e3,
        'inc': 46.7, 'dist': 101.0,
    }

    no_rad = hd163296_rotationmap.verify_params_dictionary(dict(base_params))
    assert no_rad['vradial'] is False
    model_no_rad = hd163296_rotationmap._make_model(no_rad)

    with_rad = hd163296_rotationmap.verify_params_dictionary(
        dict(base_params, vr_100=500.0, vr_q=0.0))
    assert with_rad['vradial'] is True
    model_with_rad = hd163296_rotationmap._make_model(with_rad)

    assert not np.array_equal(np.asarray(model_no_rad),
                               np.asarray(model_with_rad))
    assert np.all(np.isfinite(model_with_rad))


