"""Monte Carlo price estimators: accuracy, variance reduction, reproducibility."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for "from conftest import"

import math

import numpy as np
import pytest

from conftest import assert_mc_close
from greeks import crr_american
from pricer import (MCResult, _asian_kernel, _RunningStats, bs_price,
                    geometric_asian_price, lsm_american_put, mc_asian_arithmetic,
                    mc_european, mc_greeks_pathwise, rqmc_european)


# --------------------------------------------------------------------------- #
# Infrastructure
# --------------------------------------------------------------------------- #
def test_mcresult_ci_and_str():
    r = MCResult(10.0, 0.5, 1000)
    lo, hi = r.ci95
    assert lo < 10.0 < hi and hi - lo == pytest.approx(2 * 1.959963984540054 * 0.5)
    assert "N=1,000" in str(r)


@pytest.mark.parametrize("sizes", [[1000], [10, 990], [333, 333, 334], [1, 1, 998]])
def test_running_stats_equals_numpy(sizes):
    x = np.random.default_rng(0).lognormal(size=sum(sizes))
    rs = _RunningStats()
    start = 0
    for s in sizes:
        rs.update(x[start:start + s]); start += s
    res = rs.result()
    assert res.price == pytest.approx(x.mean(), rel=1e-12)
    assert res.std_error == pytest.approx(x.std(ddof=1) / math.sqrt(x.size), rel=1e-10)
    assert rs.result(scale=2.0).price == pytest.approx(2 * x.mean())


# --------------------------------------------------------------------------- #
# European
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("anti,cv", [(False, False), (True, False), (False, True), (True, True)])
def test_mc_european_unbiased(model_div, kind, anti, cv):
    res = mc_european(model_div, 110, 1.5, kind, n_paths=400_000, antithetic=anti, control_variate=cv)
    assert_mc_close(res, bs_price(model_div, 110, 1.5, kind))
    assert res.n_paths == 400_000


def test_variance_reduction_hierarchy(model):
    kw = dict(n_paths=400_000, seed=1)
    raw = mc_european(model, 100, 1, antithetic=False, control_variate=False, **kw)
    anti = mc_european(model, 100, 1, antithetic=True, control_variate=False, **kw)
    both = mc_european(model, 100, 1, **kw)
    assert both.std_error < anti.std_error < raw.std_error
    assert raw.std_error / both.std_error > 3          # significant gain


def test_mc_reproducible_and_seed_dependent(model):
    a = mc_european(model, 100, 1, n_paths=100_000, seed=123)
    b = mc_european(model, 100, 1, n_paths=100_000, seed=123)
    c = mc_european(model, 100, 1, n_paths=100_000, seed=124)
    assert a == b and a.price != c.price


def test_mc_standard_error_scales_as_inverse_sqrt_n(model):
    s1 = mc_european(model, 100, 1, n_paths=100_000, antithetic=False, control_variate=False).std_error
    s4 = mc_european(model, 100, 1, n_paths=400_000, antithetic=False, control_variate=False).std_error
    assert s1 / s4 == pytest.approx(2.0, rel=0.05)


def test_mc_coverage_of_confidence_interval(model):
    """Over 100 seeds, the CI95 must contain the true price ~95 times (binomial: > 88)."""
    ref = float(bs_price(model, 100, 1))
    hits = sum(lo <= ref <= hi for lo, hi in
               (mc_european(model, 100, 1, n_paths=20_000, seed=s).ci95 for s in range(100)))
    assert hits >= 88


@pytest.mark.parametrize("kind", ["call", "put"])
def test_rqmc_unbiased_and_more_precise(model_div, kind):
    q = rqmc_european(model_div, 95, 1, kind, log2_n=14, n_replicas=16)
    mc = mc_european(model_div, 95, 1, kind, n_paths=q.n_paths, antithetic=False, control_variate=False)
    assert_mc_close(q, bs_price(model_div, 95, 1, kind))
    assert q.std_error < mc.std_error / 10


def test_pathwise_greeks_legacy(model):
    g = mc_greeks_pathwise(model, 100, 1, n_paths=500_000)
    from pricer import bs_delta_vega
    d, v = bs_delta_vega(model, 100, 1)
    assert_mc_close(g["delta"], d)
    assert_mc_close(g["vega"], v)


# --------------------------------------------------------------------------- #
# Asian
# --------------------------------------------------------------------------- #
def _asian_numpy(z, s0, drift, vol, K, sign):
    """Slow, readable reference implementation."""
    log_s = math.log(s0) + np.cumsum(drift + vol * z, axis=1)
    a = np.exp(log_s).mean(axis=1)
    g = np.exp(log_s.mean(axis=1))
    return np.maximum(sign * (a - K), 0), np.maximum(sign * (g - K), 0)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_asian_kernel_matches_numpy(sign):
    z = np.random.default_rng(0).standard_normal((2000, 12))
    a, g = _asian_kernel(z, math.log(100.0), 0.001, 0.05, 100.0, sign)
    ra, rg = _asian_numpy(z, 100.0, 0.001, 0.05, 100.0, sign)
    np.testing.assert_allclose(a, ra, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(g, rg, rtol=1e-10, atol=1e-10)


def test_asian_kernel_am_gm_inequality():
    """AM-GM inequality: arithmetic call payoff ≥ geometric."""
    z = np.random.default_rng(1).standard_normal((5000, 52))
    a, g = _asian_kernel(z, math.log(100.0), 0.0, 0.03, 100.0, 1.0)
    assert np.all(a >= g - 1e-12)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_asian_cv_consistent_with_plain_mc(model, kind):
    cv = mc_asian_arithmetic(model, 100, 1, 12, kind, n_paths=200_000, seed=2)
    plain = mc_asian_arithmetic(model, 100, 1, 12, kind, n_paths=200_000, control_variate=False, seed=3)
    tol = 4 * math.hypot(cv.std_error, plain.std_error)
    assert abs(cv.price - plain.price) < tol
    assert cv.std_error < plain.std_error / 10


def test_asian_degenerates_to_european_with_one_date(model):
    # Without control: ordinary MC estimator
    plain = mc_asian_arithmetic(model, 100, 1, n_steps=1, n_paths=200_000, control_variate=False)
    assert_mc_close(plain, bs_price(model, 100, 1))
    # With control: n=1 -> geometric mean = arithmetic mean, PERFECT control
    cv = mc_asian_arithmetic(model, 100, 1, n_steps=1, n_paths=200_000)
    assert cv.price == pytest.approx(float(bs_price(model, 100, 1)), abs=1e-10)
    assert cv.std_error < 1e-10


def test_asian_bounds(model):
    a = mc_asian_arithmetic(model, 100, 1, 52, n_paths=100_000).price
    assert geometric_asian_price(model, 100, 1, 52) < a < float(bs_price(model, 100, 1))


# --------------------------------------------------------------------------- #
# American (Longstaff–Schwartz)
# --------------------------------------------------------------------------- #
def test_lsm_close_to_binomial_tree(model):
    lsm = lsm_american_put(model, 100, 1, n_paths=100_000)
    ref = crr_american(model, 100, 1, "put", n=1000)["price"]
    assert_mc_close(lsm, ref, atol=0.03)               # atol: LSM low bias


def test_lsm_respects_no_arbitrage(model):
    p = lsm_american_put(model, 100, 1, n_paths=50_000).price
    assert float(bs_price(model, 100, 1, "put")) < p < 100.0


def test_lsm_deep_in_the_money_is_intrinsic(model):
    assert lsm_american_put(model, 200, 1, n_paths=20_000).price == pytest.approx(100.0, abs=1e-6)


def test_crr_converges_to_bs_for_american_call_without_dividend(model):
    """Without dividends, exercising an American call early is never optimal."""
    tree = crr_american(model, 100, 1, "call", n=1000)
    assert tree["price"] == pytest.approx(float(bs_price(model, 100, 1)), abs=5e-3)


if __name__ == "__main__":
    # Direct launch (python file.py / VS Code ▶ button) -> go through pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
