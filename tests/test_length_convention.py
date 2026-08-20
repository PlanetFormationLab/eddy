"""The two correlation-length estimators must report the same quantity.

``fit_GRF`` returns the kernel ``ell`` of ``C = sigma^2 exp(-d^2 / 2 ell^2)``.
``calculate_heuristics`` measures a half-power lag, which is
``sqrt(2 ln 2) = 1.1774`` times larger, and converts it. Nothing enforced that
agreement before: the factor lived as a bare ``1.177`` literal in six places
and the heuristics returned raw lags, so downstream code applied its own
conversion -- and different callers applied different ones. These tests pin
the factor, the cross-estimator agreement, and the invariance of everything
that is not a length.
"""
import numpy as np
import pytest

from eddy.structurefunction import (StructureFunction, StructureFunctionStack,
                                    HALF_POWER_FACTOR, HALF_POWER_TO_KERNEL)


def test_factor_value_and_roundtrip():
    assert HALF_POWER_FACTOR == pytest.approx(np.sqrt(2.0 * np.log(2.0)))
    assert HALF_POWER_FACTOR == pytest.approx(1.177410, abs=1e-6)
    assert HALF_POWER_FACTOR * HALF_POWER_TO_KERNEL == pytest.approx(1.0)


def test_half_power_lag_of_gaussian_is_1p177_ell():
    """S_2 of a Gaussian-kernel field crosses half plateau at sqrt(2 ln 2) ell."""
    ell, sigma = 0.10, 1.0
    lags = np.linspace(0.0, 1.2, 2001)
    s2 = 2.0 * sigma**2 * (1.0 - np.exp(-lags**2 / (2.0 * ell**2)))
    hp = np.interp(0.5 * (2.0 * sigma**2), s2, lags)
    assert hp / ell == pytest.approx(HALF_POWER_FACTOR, rel=1e-4)


def _gaussian_stack(ell_r=0.10, ell_phi=0.10, sigma=1.0,
                    ref_rs=np.arange(0.4, 1.21, 0.05)):
    """Analytic stack: S_2 built directly from the kernel, no random draw.

    Isolates the units question from estimator noise -- if the conventions
    line up, recovery here is exact to interpolation error.
    """
    lags_x = np.linspace(0.0, 0.8, 161)
    lags_y = np.linspace(0.0, 90.0, 181)
    results = []
    for r in ref_rs:
        arc = np.radians(lags_y) * r
        sf = StructureFunction.__new__(StructureFunction)
        sf.lags_x, sf.lags_y = lags_x, lags_y
        sf.S2_x = 2 * sigma**2 * (1 - np.exp(-lags_x**2 / (2 * ell_r**2)))
        sf.S2_y = 2 * sigma**2 * (1 - np.exp(-arc**2 / (2 * ell_phi**2)))
        results.append(sf)
    return results, ref_rs


def test_heuristic_ell_matches_kernel_ell(monkeypatch):
    """T1b/T1c come back as the kernel ell that built the S_2, not the lag."""
    ell_r, ell_phi, sigma = 0.10, 0.30, 1.0
    results, ref_rs = _gaussian_stack(ell_r, ell_phi, sigma)

    stack = StructureFunctionStack.__new__(StructureFunctionStack)
    stack.results, stack.ref_rs = results, np.asarray(ref_rs)

    def fake_hp(axis="x", plateau=None):
        base = ell_r if axis == "x" else ell_phi
        n = len(ref_rs)
        if axis == "x":
            return np.full(n, base * HALF_POWER_FACTOR)
        return np.degrees(base * HALF_POWER_FACTOR / np.asarray(ref_rs))

    monkeypatch.setattr(stack, "half_power_lags", fake_hp, raising=False)
    monkeypatch.setattr(stack, "plateaus",
                        lambda: np.full(len(ref_rs), 2 * sigma**2),
                        raising=False)
    monkeypatch.setattr(
        stack, "collapse",
        lambda: type("C", (), {"plateau": staticmethod(lambda: 2 * sigma**2)})(),
        raising=False)
    for r in stack.results:
        r.lags_x = results[0].lags_x

    T1a, T1b, T1c, T2, T3, T4 = stack.calculate_heuristics()
    assert T1b == pytest.approx(ell_r, rel=1e-6)          # kernel ell, not 1.1774x
    assert T1c == pytest.approx(ell_phi, rel=1e-6)
    assert T2 == pytest.approx(ell_phi / ell_r, rel=1e-6)  # anisotropy invariant

    # halfpower mode returns the raw lags, exactly 1.1774x larger
    _, hb, hc, h2, h3, h4 = stack.calculate_heuristics(length_scale="halfpower")
    assert hb / T1b == pytest.approx(HALF_POWER_FACTOR, rel=1e-9)
    assert hc / T1c == pytest.approx(HALF_POWER_FACTOR, rel=1e-9)

    # everything that is not a length is invariant under the convention
    assert h2 == pytest.approx(T2, rel=1e-12)
    assert h3 == pytest.approx(T3, abs=1e-12) or (np.isnan(h3) and np.isnan(T3))
    assert h4 == pytest.approx(T4, abs=1e-12) or (np.isnan(h4) and np.isnan(T4))

    with pytest.raises(ValueError, match="length_scale"):
        stack.calculate_heuristics(length_scale="fwhm")


def _analytic_stack(ell_r, ell_phi, sigma=1.0,
                    ref_rs=np.arange(0.40, 1.21, 0.05)):
    """A real StructureFunctionStack whose S_2 slices are the analytic GRF.

    No monkeypatching: S_2 is built from the kernel, so every code path
    (plateau, half_power_lags, the neff weights, fit_GRF) runs for real.
    """
    lags_x = np.linspace(0.0, 0.9, 181)
    dphi = np.linspace(0.0, 120.0, 241)
    plateau = 2.0 * sigma**2
    results = []
    for r in ref_rs:
        arc = np.radians(dphi) * r
        S2_x = plateau * (1.0 - np.exp(-lags_x**2 / (2.0 * ell_r**2)))
        S2_y = plateau * (1.0 - np.exp(-arc**2 / (2.0 * ell_phi**2)))
        nx, ny = lags_x.size - 1, dphi.size - 1
        S2 = np.zeros((2 * nx + 1, 2 * ny + 1))
        results.append(StructureFunction(
            S2=S2, counts=np.ones_like(S2, dtype=int),
            dx=float(lags_x[1] - lags_x[0]), dy=float(dphi[1] - dphi[0]),
            lags_x=lags_x, lags_y=dphi, lags_i=dphi[:10],
            S2_x=S2_x, S2_y=S2_y, S2_i=np.zeros(10),
            azimuthal_axis="y"))
    return StructureFunctionStack(ref_rs=list(ref_rs), ref_band=0.02,
                                  results=results)


@pytest.mark.parametrize("ell_r,ell_phi", [(0.10, 0.10), (0.08, 0.32)])
def test_heuristics_and_fit_GRF_agree_end_to_end(ell_r, ell_phi):
    """The headline guarantee: both estimators return the SAME ell.

    This is the check that was missing. Before the kernel convention,
    calculate_heuristics returned a half-power lag and fit_GRF returned a
    kernel ell, so these two numbers differed by 1.1774 on identical data and
    nothing in the test suite noticed.
    """
    stack = _analytic_stack(ell_r, ell_phi)

    _, T1b, T1c, T2, _, _ = stack.calculate_heuristics()
    assert T1b == pytest.approx(ell_r, rel=0.02)
    assert T1c == pytest.approx(ell_phi, rel=0.05)

    params = stack.fit_GRF(r0=1.0, method="lsq", fit_alphaphi=False)[0]
    assert params["ell0r"] == pytest.approx(ell_r, rel=0.05)
    assert params["ell0phi"] == pytest.approx(ell_phi, rel=0.10)

    # ... and they agree with EACH OTHER, which is the actual invariant.
    assert T1b == pytest.approx(params["ell0r"], rel=0.06)
    assert T1c == pytest.approx(params["ell0phi"], rel=0.12)

    # The old convention would have failed the above by exactly this factor.
    hp = stack.calculate_heuristics(length_scale="halfpower")
    assert hp[1] / params["ell0r"] == pytest.approx(HALF_POWER_FACTOR, rel=0.06)
