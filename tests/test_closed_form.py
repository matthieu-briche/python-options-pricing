"""Closed forms: known values, parities, arbitrage bounds, PDE, properties."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for "from conftest import"

import math
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from conftest import fd, fd2
from greeks import bs_greeks, geometric_asian_greeks
from pricer import BlackScholes, bs_delta_vega, bs_price, geometric_asian_price

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kwargs", [dict(s0=0, r=0.0, sigma=0.2), dict(s0=-1, r=0.0, sigma=0.2),
                                    dict(s0=100, r=0.0, sigma=0.0), dict(s0=100, r=0.0, sigma=-0.1)])
def test_model_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        BlackScholes(**kwargs)


def test_model_is_immutable(model):
    with pytest.raises(FrozenInstanceError):
        model.s0 = 50.0


# --------------------------------------------------------------------------- #
# Black–Scholes price
# --------------------------------------------------------------------------- #
def test_bs_reference_values(model):
    # Classic reference values (Hull): S=K=100, r=5%, sigma=20%, T=1
    assert bs_price(model, 100, 1.0, "call") == pytest.approx(10.450583572185565, abs=1e-10)
    assert bs_price(model, 100, 1.0, "put") == pytest.approx(5.573526022256971, abs=1e-10)


@pytest.mark.parametrize("K", [60.0, 100.0, 140.0])
@pytest.mark.parametrize("T", [0.1, 1.0, 5.0])
def test_put_call_parity(model_div, K, T):
    m = model_div
    c, p = bs_price(m, K, T, "call"), bs_price(m, K, T, "put")
    assert c - p == pytest.approx(m.s0 * math.exp(-m.q * T) - K * math.exp(-m.r * T), abs=1e-10)


def test_vectorized_matches_scalar(model):
    Ks = np.linspace(50, 150, 101)
    Ts = np.linspace(0.1, 3, 101)
    vec = bs_price(model, Ks[:, None], Ts[None, :])
    assert vec.shape == (101, 101)
    for i, j in [(0, 0), (50, 50), (100, 7), (13, 100)]:
        assert vec[i, j] == pytest.approx(float(bs_price(model, Ks[i], Ts[j])), rel=1e-14)


def test_limits(model):
    # Very low strike: call ≈ discounted forward - K e^{-rT}; put ≈ 0
    assert bs_price(model, 1e-6, 1.0, "call") == pytest.approx(100 - 1e-6 * math.exp(-0.05), rel=1e-9)
    assert bs_price(model, 1e-6, 1.0, "put") == pytest.approx(0.0, abs=1e-12)
    # Maturity ≈ 0: intrinsic value
    assert bs_price(model, 90, 1e-10, "call") == pytest.approx(10.0, abs=1e-6)
    assert bs_price(model, 110, 1e-10, "put") == pytest.approx(10.0, abs=1e-6)


@settings(max_examples=200, deadline=None)
@given(s0=st.floats(10, 500), K=st.floats(10, 500), T=st.floats(0.01, 10),
       r=st.floats(-0.02, 0.15), q=st.floats(0, 0.1), sigma=st.floats(0.05, 1.0))
def test_no_arbitrage_bounds(s0, K, T, r, q, sigma):
    m = BlackScholes(s0, r, sigma, q)
    c, p = float(bs_price(m, K, T, "call")), float(bs_price(m, K, T, "put"))
    fwd_s, fwd_k = s0 * math.exp(-q * T), K * math.exp(-r * T)
    tol = 1e-9 * max(s0, K)
    assert max(fwd_s - fwd_k, 0) - tol <= c <= fwd_s + tol
    assert max(fwd_k - fwd_s, 0) - tol <= p <= fwd_k + tol


@settings(max_examples=100, deadline=None)
@given(K=st.floats(50, 150), sigma=st.floats(0.05, 0.8))
def test_monotonicity(K, sigma):
    m = BlackScholes(100.0, 0.03, sigma)
    eps = 1e-12 * 100.0      # deep in the money, Vega ≈ 0: equal up to rounding
    # Call decreasing in K, price increasing in sigma
    assert bs_price(m, K, 1.0) >= bs_price(m, K + 1, 1.0) - eps
    assert bs_price(replace(m, sigma=sigma + 0.01), K, 1.0) >= bs_price(m, K, 1.0) - eps


# --------------------------------------------------------------------------- #
# Closed-form Greeks
# --------------------------------------------------------------------------- #
CASES = [(100.0, 1.0), (80.0, 0.5), (130.0, 2.0), (100.0, 0.05)]


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("K,T", CASES)
def test_bs_greeks_match_finite_differences(model_div, kind, K, T):
    m = model_div
    g = bs_greeks(m, K, T, kind)
    p = lambda mm, TT=T: float(bs_price(mm, K, TT, kind))
    assert g["price"] == pytest.approx(p(m), rel=1e-14)
    assert g["delta"] == pytest.approx(fd(lambda s: p(replace(m, s0=s)), m.s0, 1e-4), abs=1e-7)
    assert g["gamma"] == pytest.approx(fd2(lambda s: p(replace(m, s0=s)), m.s0, 1e-2), rel=1e-4, abs=1e-7)
    assert g["vega"] == pytest.approx(fd(lambda v: p(replace(m, sigma=v)), m.sigma, 1e-6), rel=1e-6)
    assert g["rho"] == pytest.approx(fd(lambda r: p(replace(m, r=r)), m.r, 1e-6), rel=1e-6, abs=1e-6)
    assert g["theta"] == pytest.approx(-fd(lambda t: p(m, t), T, 1e-6), rel=1e-5, abs=1e-5)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_delta_vega_helper_consistent(model_div, kind):
    d, v = bs_delta_vega(model_div, 95.0, 1.3, kind)
    g = bs_greeks(model_div, 95.0, 1.3, kind)
    assert d == pytest.approx(g["delta"]) and v == pytest.approx(g["vega"])


@pytest.mark.parametrize("K,T", CASES)
def test_greeks_parity_relations(model_div, K, T):
    m = model_div
    c, p = bs_greeks(m, K, T, "call"), bs_greeks(m, K, T, "put")
    assert c["delta"] - p["delta"] == pytest.approx(math.exp(-m.q * T))
    assert c["gamma"] == pytest.approx(p["gamma"])
    assert c["vega"] == pytest.approx(p["vega"])
    assert c["rho"] - p["rho"] == pytest.approx(K * T * math.exp(-m.r * T))
    assert c["theta"] - p["theta"] == pytest.approx(
        m.q * m.s0 * math.exp(-m.q * T) - m.r * K * math.exp(-m.r * T))


@settings(max_examples=200, deadline=None)
@given(K=st.floats(40, 200), T=st.floats(0.02, 5), r=st.floats(0, 0.1),
       q=st.floats(0, 0.05), sigma=st.floats(0.05, 0.9), kind=st.sampled_from(["call", "put"]))
def test_black_scholes_pde(K, T, r, q, sigma, kind):
    """Θ + ½σ²S²Γ + (r-q)SΔ - rV = 0 (Θ = -∂V/∂T = ∂V/∂t)."""
    m = BlackScholes(100.0, r, sigma, q)
    g = bs_greeks(m, K, T, kind)
    lhs = g["theta"] + 0.5 * sigma**2 * m.s0**2 * g["gamma"] + (r - q) * m.s0 * g["delta"] - r * g["price"]
    assert abs(float(lhs)) < 1e-8 * max(1.0, float(g["price"]))


@settings(max_examples=100, deadline=None)
@given(K=st.floats(40, 200), T=st.floats(0.02, 5), kind=st.sampled_from(["call", "put"]))
def test_greek_signs(K, T, kind):
    g = bs_greeks(BlackScholes(100.0, 0.03, 0.25), K, T, kind)
    assert g["gamma"] >= 0 and g["vega"] >= 0
    assert (0 <= g["delta"] <= 1) if kind == "call" else (-1 <= g["delta"] <= 0)
    assert g["rho"] >= 0 if kind == "call" else g["rho"] <= 0


# --------------------------------------------------------------------------- #
# Geometric Asian
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["call", "put"])
def test_geometric_asian_one_date_is_european(model_div, kind):
    """n = 1: the average reduces to S_T -> price AND Greeks = Black–Scholes."""
    m = model_div
    assert geometric_asian_price(m, 105, 1.5, 1, kind) == pytest.approx(float(bs_price(m, 105, 1.5, kind)), rel=1e-12)
    geo = geometric_asian_greeks(m, 105, 1.5, 1, kind)
    bs = bs_greeks(m, 105, 1.5, kind)
    for k, name in enumerate(["price", "delta", "gamma", "vega", "rho"]):
        assert geo[k] == pytest.approx(float(bs[name]), rel=1e-5, abs=1e-7), name


@pytest.mark.parametrize("n", [1, 12, 52, 252])
def test_geometric_asian_parity(model_div, n):
    m, K, T = model_div, 100.0, 2.0
    mu = math.log(m.s0) + (m.r - m.q - 0.5 * m.sigma**2) * T * (n + 1) / (2 * n)
    s2 = m.sigma**2 * T * (n + 1) * (2 * n + 1) / (6 * n * n)
    eg = math.exp(mu + 0.5 * s2)
    c = geometric_asian_price(m, K, T, n, "call")
    p = geometric_asian_price(m, K, T, n, "put")
    assert c - p == pytest.approx(math.exp(-m.r * T) * (eg - K), abs=1e-10)


def test_geometric_asian_cheaper_than_european(model):
    # Averaging reduces variance -> cheaper ATM option
    assert geometric_asian_price(model, 100, 1, 52) < float(bs_price(model, 100, 1))


if __name__ == "__main__":
    # Direct launch (python file.py / VS Code ▶ button) -> go through pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
