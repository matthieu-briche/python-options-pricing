"""Monte Carlo Greeks (pathwise, LR, control), CRN finite differences, CRR tree."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for "from conftest import"

import math
from dataclasses import replace

import numpy as np
import pytest

from conftest import assert_mc_close
from greeks import (GREEKS, _asian_greeks_kernel, bs_greeks, crr_american,
                    fd_greeks_crn, geometric_asian_greeks, mc_asian_greeks,
                    mc_european_greeks)
from pricer import _asian_kernel, bs_price, lsm_american_put, mc_asian_arithmetic


# --------------------------------------------------------------------------- #
# European: each MC Greek against the closed form
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module", params=[("call", 100.0, 1.0), ("put", 100.0, 1.0),
                                        ("call", 120.0, 0.5), ("put", 85.0, 2.0)],
                ids=lambda p: f"{p[0]}-K{p[1]:g}-T{p[2]:g}")
def euro_case(request):
    from pricer import BlackScholes
    m = BlackScholes(s0=100.0, r=0.03, sigma=0.3, q=0.02)
    kind, K, T = request.param
    return m, K, T, kind, mc_european_greeks(m, K, T, kind, n_paths=1_000_000)


@pytest.mark.parametrize("greek", GREEKS)
def test_mc_european_greek(euro_case, greek):
    m, K, T, kind, est = euro_case
    assert_mc_close(est[greek], bs_greeks(m, K, T, kind)[greek])


def test_mc_european_greeks_reproducible(model):
    a = mc_european_greeks(model, 100, 1, n_paths=50_000, seed=3)
    b = mc_european_greeks(model, 100, 1, n_paths=50_000, seed=3)
    assert all(a[g] == b[g] for g in GREEKS)


def test_mc_european_gamma_identical_call_put(model):
    """Same seed -> same Z: the LR Gamma of a call and a put must differ
    only by noise, and stay within the CI of the common exact value."""
    c = mc_european_greeks(model, 100, 1, "call", n_paths=500_000)
    p = mc_european_greeks(model, 100, 1, "put", n_paths=500_000)
    ref = bs_greeks(model, 100, 1)["gamma"]
    assert_mc_close(c["gamma"], ref)
    assert_mc_close(p["gamma"], ref)


# --------------------------------------------------------------------------- #
# Asian: Numba kernel
# --------------------------------------------------------------------------- #
def _kernel_args(m, K, T, n, sign=1.0):
    dt = T / n
    return (m.s0, m.r, m.r - m.q - 0.5 * m.sigma**2, m.sigma, dt, K, sign, math.exp(-m.r * T))


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_asian_greeks_kernel_prices_match_price_kernel(model_div, sign):
    m, n, T = model_div, 12, 1.0
    z = np.random.default_rng(0).standard_normal((3000, n))
    out = _asian_greeks_kernel(z, *_kernel_args(m, 100, T, n, sign))
    dt = T / n
    a, g = _asian_kernel(z, math.log(m.s0), (m.r - m.q - 0.5 * m.sigma**2) * dt,
                         m.sigma * math.sqrt(dt), 100.0, sign)
    df = math.exp(-m.r * T)
    np.testing.assert_allclose(out[0], df * a, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(out[5], df * g, rtol=1e-10, atol=1e-12)


def test_asian_greeks_kernel_pathwise_delta_matches_bump(model):
    """On each path where the payoff is differentiable, the pathwise delta
    must be the exact derivative of the discounted payoff with respect to S0."""
    n, T, h = 12, 1.0, 1e-5
    z = np.random.default_rng(2).standard_normal((2000, n))
    base = _asian_greeks_kernel(z, *_kernel_args(model, 100, T, n))
    up = _asian_greeks_kernel(z, *_kernel_args(replace(model, s0=100 + h), 100, T, n))
    dn = _asian_greeks_kernel(z, *_kernel_args(replace(model, s0=100 - h), 100, T, n))
    bump = (up[0] - dn[0]) / (2 * h)
    smooth = np.abs(up[0] - dn[0]) > 0          # excludes out-of-the-money paths
    np.testing.assert_allclose(base[1][smooth], bump[smooth], rtol=1e-4, atol=1e-6)


def test_asian_greeks_kernel_pathwise_vega_matches_bump(model):
    n, T, h = 12, 1.0, 1e-6
    z = np.random.default_rng(3).standard_normal((2000, n))
    base = _asian_greeks_kernel(z, *_kernel_args(model, 100, T, n))
    up = _asian_greeks_kernel(z, *_kernel_args(replace(model, sigma=0.2 + h), 100, T, n))
    dn = _asian_greeks_kernel(z, *_kernel_args(replace(model, sigma=0.2 - h), 100, T, n))
    bump = (up[0] - dn[0]) / (2 * h)
    smooth = (up[0] > 0) & (dn[0] > 0)
    np.testing.assert_allclose(base[3][smooth], bump[smooth], rtol=1e-4, atol=1e-5)


def test_asian_geometric_estimators_unbiased(model_div):
    """Geometric rows of the kernel: MC mean ≈ exact geometric Greeks."""
    m, n, T, K = model_div, 12, 1.0, 100.0
    z = np.random.default_rng(4).standard_normal((400_000, n))
    out = _asian_greeks_kernel(z, *_kernel_args(m, K, T, n))
    exact = geometric_asian_greeks(m, K, T, n)
    for k in range(5):
        x = out[5 + k]
        se = x.std(ddof=1) / math.sqrt(x.size)
        assert abs(x.mean() - exact[k]) < 4 * se, f"greek #{k}"


# --------------------------------------------------------------------------- #
# Asian: full estimator
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["call", "put"])
def test_mc_asian_greeks_one_date_equals_bs(model, kind):
    est = mc_asian_greeks(model, 100, 1, n_steps=1, kind=kind, n_paths=200_000)
    ref = bs_greeks(model, 100, 1, kind)
    for g in est:
        assert_mc_close(est[g], ref[g], atol=1e-6)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_mc_asian_greeks_cv_vs_plain(model, kind):
    cv = mc_asian_greeks(model, 100, 1, 12, kind, n_paths=200_000, seed=1)
    plain = mc_asian_greeks(model, 100, 1, 12, kind, n_paths=200_000, control_variate=False, seed=2)
    for g in cv:
        assert abs(cv[g].price - plain[g].price) < 4 * math.hypot(cv[g].std_error, plain[g].std_error), g
        assert cv[g].std_error < plain[g].std_error, g


@pytest.mark.slow
def test_mc_asian_greeks_vs_crn_finite_differences(model):
    est = mc_asian_greeks(model, 100, 1, 12, n_paths=300_000)
    pricer = lambda mm, TT: mc_asian_arithmetic(mm, 100, TT, 12, n_paths=300_000, seed=9).price
    ref = fd_greeks_crn(pricer, model, 1.0)
    # The FD reference is itself noisy and biased (finite bump) -> relative tolerance
    for g in est:
        assert est[g].price == pytest.approx(ref[g], rel=0.02, abs=2e-3), g


# --------------------------------------------------------------------------- #
# CRN finite differences
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["call", "put"])
def test_fd_crn_on_closed_form(model_div, kind):
    """On a deterministic pricer, central FDs must recover the Greeks."""
    pricer = lambda mm, T: float(bs_price(mm, 100, T, kind))
    fd = fd_greeks_crn(pricer, model_div, 1.0, rel_ds=1e-3, d_sigma=1e-4, d_r=1e-5, d_t=1e-4)
    ref = bs_greeks(model_div, 100, 1.0, kind)
    for g in GREEKS:
        assert fd[g] == pytest.approx(float(ref[g]), rel=1e-4, abs=1e-5), g


def test_crn_is_essential(model):
    """With different seeds, the FD Gamma blows up: check that CRN
    reduces the error by at least an order of magnitude."""
    ref = bs_greeks(model, 100, 1)["gamma"]
    from pricer import mc_european
    crn = lambda mm, T: mc_european(mm, 100, T, n_paths=200_000, seed=1).price
    counter = iter(range(100, 200))
    no_crn = lambda mm, T: mc_european(mm, 100, T, n_paths=200_000, seed=next(counter)).price
    err_crn = abs(fd_greeks_crn(crn, model, 1.0)["gamma"] - ref)
    err_no = abs(fd_greeks_crn(no_crn, model, 1.0)["gamma"] - ref)
    assert err_crn * 10 < err_no


# --------------------------------------------------------------------------- #
# American
# --------------------------------------------------------------------------- #
def test_crr_american_call_no_dividend_equals_bs_greeks(model):
    tree = crr_american(model, 100, 1, "call", n=1000)
    ref = bs_greeks(model, 100, 1, "call")
    tol = dict(price=5e-3, delta=1e-3, gamma=1e-4, vega=0.05, rho=0.05, theta=0.01)
    for g in GREEKS:
        assert tree[g] == pytest.approx(float(ref[g]), abs=tol[g]), g


def test_crr_american_put_properties(model):
    am = crr_american(model, 100, 1, "put", n=1000)
    eu = bs_greeks(model, 100, 1, "put")
    assert am["price"] > eu["price"]                    # early exercise premium
    assert -1 < am["delta"] < 0 and am["gamma"] > 0 and am["vega"] > 0 and am["rho"] < 0


def test_crr_converges(model):
    p = [crr_american(model, 100, 1, "put", n=n)["price"] for n in (250, 500, 1000, 2000)]
    diffs = np.abs(np.diff(p))
    assert diffs[-1] < diffs[0] and diffs[-1] < 2e-3


@pytest.mark.slow
def test_lsm_greeks_vs_tree(model):
    lsm = lambda mm, T: lsm_american_put(mm, 100, T, n_paths=100_000, seed=3).price
    est = fd_greeks_crn(lsm, model, 1.0, rel_ds=0.02)
    ref = crr_american(model, 100, 1, "put", n=1000)
    tol = dict(price=0.04, delta=0.01, gamma=0.004, vega=0.6, rho=0.6, theta=0.15)
    for g in GREEKS:
        assert est[g] == pytest.approx(ref[g], abs=tol[g]), g


if __name__ == "__main__":
    # Direct launch (python file.py / VS Code ▶ button) -> go through pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
