"""
greeks.py — CPU-optimised sensitivities (Greeks).

Methods (G. Pagès, *Numerical Probability*, Ch. 2.2 and Ch. 10):
  - Black–Scholes closed forms (reference, vectorised).
  - PATHWISE method (tangent process): the payoff is differentiated along the
    path. Low variance, valid for Delta, Vega, Rho, Theta
    (Lipschitz payoff).
  - LIKELIHOOD RATIO method (log-likelihood): the density is differentiated,
    not the payoff -> works for discontinuous payoffs. Used here in a
    MIXED pathwise-LR estimator for Gamma (the pathwise Delta of a call
    is discontinuous, so it cannot be differentiated again path by path).
  - Finite differences with COMMON RANDOM NUMBERS (CRN): universal
    method (American options / LSM), same seed for base and bumped prices.
  - Vectorised CRR binomial tree: deterministic reference for the American.

CPU optimisation:
  - ONE simulation gives the price AND all the Greeks (no re-pricing).
  - Parallel Numba kernel for the Asian: 10 estimators per path in
    one pass, in registers.
  - Control variates on the Greeks too (geometric Asian, whose
    exact Greeks come from differentiating the closed form).
"""
from __future__ import annotations

if __name__ == "__main__":
    # File run directly (python greeks.py / VS Code ▶ button): the demo runs through
    # the imported "greeks" module and not "__main__", because the Numba cache
    # is tied to the module name.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    import greeks as _m
    _m._demo()
    raise SystemExit(0)

import math
from dataclasses import replace
from typing import Callable

import numpy as np
from numba import njit, prange
from scipy.special import ndtr

from pricer import (BlackScholes, MCResult, OptionType, _RunningStats,
                    check_proba_crr, geometric_asian_price)

GREEKS = ("price", "delta", "gamma", "vega", "rho", "theta")
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


# --------------------------------------------------------------------------- #
# 1. Black–Scholes closed forms (vectorised on K, T)
# --------------------------------------------------------------------------- #
def bs_greeks(m: BlackScholes, K, T, kind: OptionType = "call") -> dict[str, np.ndarray]:
    K, T = np.asarray(K, float), np.asarray(T, float)
    sqT = np.sqrt(T)
    d1 = (np.log(m.s0 / K) + (m.r - m.q + 0.5 * m.sigma**2) * T) / (m.sigma * sqT)
    d2 = d1 - m.sigma * sqT
    dfr, dfq = np.exp(-m.r * T), np.exp(-m.q * T)
    pdf1 = _INV_SQRT_2PI * np.exp(-0.5 * d1 * d1)
    gamma = dfq * pdf1 / (m.s0 * m.sigma * sqT)
    vega = m.s0 * dfq * pdf1 * sqT
    decay = -m.s0 * dfq * pdf1 * m.sigma / (2.0 * sqT)
    if kind == "call":
        price = m.s0 * dfq * ndtr(d1) - K * dfr * ndtr(d2)
        delta = dfq * ndtr(d1)
        rho = K * T * dfr * ndtr(d2)
        theta = decay - m.r * K * dfr * ndtr(d2) + m.q * m.s0 * dfq * ndtr(d1)
    else:
        price = K * dfr * ndtr(-d2) - m.s0 * dfq * ndtr(-d1)
        delta = dfq * (ndtr(d1) - 1.0)
        rho = -K * T * dfr * ndtr(-d2)
        theta = decay + m.r * K * dfr * ndtr(-d2) - m.q * m.s0 * dfq * ndtr(-d1)
    return dict(price=price, delta=delta, gamma=gamma, vega=vega, rho=rho, theta=theta)


# --------------------------------------------------------------------------- #
# 2. European Monte Carlo: all the Greeks in one simulation
# --------------------------------------------------------------------------- #
def mc_european_greeks(m: BlackScholes, K: float, T: float, kind: OptionType = "call",
                       n_paths: int = 2_000_000, chunk: int = 2**18,
                       seed: int = 11) -> dict[str, MCResult]:
    """
    Per-path estimators (f = payoff, ind = 1{in the money}, s = ±1):
      delta : e^{-rT} s ind S_T / S0                         (pathwise)
      gamma : e^{-rT} s ind S_T / S0² (Z/(σ√T) - 1)          (mixed pathwise-LR)
      vega  : e^{-rT} s ind S_T (√T Z - σT)                  (pathwise)
      rho   : -T e^{-rT} f + e^{-rT} s ind S_T T             (pathwise)
      theta : r e^{-rT} f - e^{-rT} s ind S_T (r-q-σ²/2 + σZ/(2√T))
    Antithetic variates (Z, -Z) on all the estimators.
    """
    rng = np.random.default_rng(seed)
    sqT = math.sqrt(T)
    mu = m.r - m.q - 0.5 * m.sigma**2
    df = math.exp(-m.r * T)
    sgn = 1.0 if kind == "call" else -1.0
    stats = {g: _RunningStats() for g in GREEKS}

    def estimators(z: np.ndarray) -> dict[str, np.ndarray]:
        st = m.s0 * np.exp(mu * T + m.sigma * sqT * z)
        f = np.maximum(sgn * (st - K), 0.0)
        h = df * sgn * (f > 0) * st                    # e^{-rT} f'(S_T) S_T
        return dict(
            price=df * f,
            delta=h / m.s0,
            gamma=h / m.s0**2 * (z / (m.sigma * sqT) - 1.0),
            vega=h * (sqT * z - m.sigma * T),
            rho=-T * df * f + h * T,
            theta=m.r * df * f - h * (mu + m.sigma * z / (2.0 * sqT)),
        )

    done, n_draws = 0, n_paths // 2
    while done < n_draws:
        b = min(chunk, n_draws - done)
        z = rng.standard_normal(b)
        e1, e2 = estimators(z), estimators(-z)
        for g in GREEKS:
            stats[g].update(0.5 * (e1[g] + e2[g]))
        done += b
    return {g: _with_n(stats[g].result(), n_paths) for g in GREEKS}


def _with_n(r: MCResult, n: int) -> MCResult:
    return MCResult(r.price, r.std_error, n)


# --------------------------------------------------------------------------- #
# 3. Arithmetic Asian: Numba kernel, 5 Greeks + geometric control
# --------------------------------------------------------------------------- #
@njit(parallel=True, fastmath=True, cache=True)
def _asian_greeks_kernel(z, s0, r, mu, sigma, dt, K, sgn, df):
    """
    Returns out[10, n]: rows 0-4 = arithmetic (price, delta, gamma, vega, rho),
    rows 5-9 = same for the geometric Asian (control variate).
    Mixed Gamma: the LR weight applies to the first increment Z_0 (the only link to S0).
    """
    n_paths, n = z.shape
    T = n * dt
    sqdt = math.sqrt(dt)
    out = np.empty((10, n_paths))
    for i in prange(n_paths):
        log_s = math.log(s0)
        w = 0.0
        a = 0.0      # sum S_tj
        a_sig = 0.0  # sum dS_tj/dsigma = S_tj (W_tj - sigma t_j)
        a_r = 0.0    # sum dS_tj/dr     = S_tj t_j
        lg = 0.0     # sum log S_tj
        g_sig = 0.0  # sum (W_tj - sigma t_j)
        g_r = 0.0    # sum t_j
        for j in range(n):
            t = (j + 1) * dt
            w += sqdt * z[i, j]
            log_s += mu * dt + sigma * sqdt * z[i, j]
            s = math.exp(log_s)
            a += s
            a_sig += s * (w - sigma * t)
            a_r += s * t
            lg += log_s
            g_sig += w - sigma * t
            g_r += t
        a /= n; a_sig /= n; a_r /= n
        g = math.exp(lg / n); gs = g * g_sig / n; gr = g * g_r / n
        lr = z[i, 0] / (sigma * sqdt) - 1.0
        # arithmetic
        fa = max(sgn * (a - K), 0.0)
        ia = df * sgn if fa > 0.0 else 0.0
        out[0, i] = df * fa
        out[1, i] = ia * a / s0
        out[2, i] = ia * a / (s0 * s0) * lr
        out[3, i] = ia * a_sig
        out[4, i] = -T * df * fa + ia * a_r
        # geometric
        fg = max(sgn * (g - K), 0.0)
        ig = df * sgn if fg > 0.0 else 0.0
        out[5, i] = df * fg
        out[6, i] = ig * g / s0
        out[7, i] = ig * g / (s0 * s0) * lr
        out[8, i] = ig * gs
        out[9, i] = -T * df * fg + ig * gr
    return out


def geometric_asian_greeks(m: BlackScholes, K: float, T: float, n: int,
                           kind: OptionType = "call") -> np.ndarray:
    """EXACT Greeks of the geometric Asian (derived from the closed form)."""
    p = lambda mm: geometric_asian_price(mm, K, T, n, kind)
    hs, hv, hr = 1e-3 * m.s0, 1e-5, 1e-6
    up, dn = replace(m, s0=m.s0 + hs), replace(m, s0=m.s0 - hs)
    return np.array([
        p(m),
        (p(up) - p(dn)) / (2 * hs),
        (p(up) - 2 * p(m) + p(dn)) / hs**2,
        (p(replace(m, sigma=m.sigma + hv)) - p(replace(m, sigma=m.sigma - hv))) / (2 * hv),
        (p(replace(m, r=m.r + hr)) - p(replace(m, r=m.r - hr))) / (2 * hr),
    ])


def mc_asian_greeks(m: BlackScholes, K: float, T: float, n_steps: int = 52,
                    kind: OptionType = "call", n_paths: int = 500_000,
                    chunk: int = 2**16, control_variate: bool = True,
                    seed: int = 5) -> dict[str, MCResult]:
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    args = (m.s0, m.r, m.r - m.q - 0.5 * m.sigma**2, m.sigma, dt, K,
            1.0 if kind == "call" else -1.0, math.exp(-m.r * T))
    names = GREEKS[:5]
    exact_geo = geometric_asian_greeks(m, K, T, n_steps, kind)

    beta = np.zeros(5)
    if control_variate:                                # independent pilot block
        o = _asian_greeks_kernel(rng.standard_normal((20_000, n_steps)), *args)
        for k in range(5):
            c = np.cov(o[k], o[k + 5])
            beta[k] = c[0, 1] / c[1, 1]

    stats = [_RunningStats() for _ in names]
    done = 0
    while done < n_paths:
        b = min(chunk, n_paths - done)
        o = _asian_greeks_kernel(rng.standard_normal((b, n_steps)), *args)
        for k in range(5):
            stats[k].update(o[k] - beta[k] * (o[k + 5] - exact_geo[k]))
        done += b
    return {g: s.result() for g, s in zip(names, stats)}


# --------------------------------------------------------------------------- #
# 4. Universal method: finite differences with common random numbers
# --------------------------------------------------------------------------- #
PriceFn = Callable[[BlackScholes, float], float]


def fd_greeks_crn(price_fn: PriceFn, m: BlackScholes, T: float,
                  rel_ds: float = 0.01, d_sigma: float = 0.01,
                  d_r: float = 1e-3, d_t: float = 1 / 52) -> dict[str, float]:
    """
    price_fn(model, T) MUST use a fixed seed (CRN): the Monte Carlo noise then
    largely cancels out in the differences. Central differences.
    Cost: 9 pricer evaluations -> keep it for products without a simple
    pathwise/LR estimator (American, barrier, ...).
    """
    ds = rel_ds * m.s0
    p0 = price_fn(m, T)
    pu, pd = price_fn(replace(m, s0=m.s0 + ds), T), price_fn(replace(m, s0=m.s0 - ds), T)
    vega = (price_fn(replace(m, sigma=m.sigma + d_sigma), T)
            - price_fn(replace(m, sigma=m.sigma - d_sigma), T)) / (2 * d_sigma)
    rho = (price_fn(replace(m, r=m.r + d_r), T) - price_fn(replace(m, r=m.r - d_r), T)) / (2 * d_r)
    theta = -(price_fn(m, T + d_t) - price_fn(m, T - d_t)) / (2 * d_t)
    return dict(price=p0, delta=(pu - pd) / (2 * ds),
                gamma=(pu - 2 * p0 + pd) / ds**2, vega=vega, rho=rho, theta=theta)


# --------------------------------------------------------------------------- #
# 5. American reference: vectorised CRR binomial tree
# --------------------------------------------------------------------------- #
def crr_american(m: BlackScholes, K: float, T: float, kind: OptionType = "put",
                 n: int = 2000) -> dict[str, float]:
    """Delta, Gamma, Theta read from the first nodes; Vega, Rho by re-pricing."""
    def tree(mm: BlackScholes):
        dt = T / n
        u = math.exp(mm.sigma * math.sqrt(dt)); d = 1.0 / u
        p = (math.exp((mm.r - mm.q) * dt) - d) / (u - d)
        check_proba_crr(p, mm, dt)
        disc = math.exp(-mm.r * dt)
        sgn = 1.0 if kind == "call" else -1.0
        j = np.arange(n + 1)
        v = np.maximum(sgn * (mm.s0 * u**j * d**(n - j) - K), 0.0)
        keep = {}
        for step in range(n - 1, -1, -1):
            j = np.arange(step + 1)
            s = mm.s0 * u**j * d**(step - j)
            v = np.maximum(disc * (p * v[1:] + (1 - p) * v[:-1]), sgn * (s - K))
            if step <= 2:
                keep[step] = (s, v.copy())
        return keep, dt

    k, dt = tree(m)
    (s1, v1), (s2, v2), (_, v0) = k[1], k[2], k[0]
    delta = (v1[1] - v1[0]) / (s1[1] - s1[0])
    gamma = ((v2[2] - v2[1]) / (s2[2] - s2[1]) - (v2[1] - v2[0]) / (s2[1] - s2[0])) \
        / (0.5 * (s2[2] - s2[0]))
    theta = (v2[1] - v0[0]) / (2 * dt)
    price = lambda mm: tree(mm)[0][0][1][0]
    # Differentiation steps are deliberately wide: the tree grid moves
    # with sigma and r, so a step that is too small (1e-3 / 1e-4) amplifies the
    # discretisation noise (differences > 1% on vega/rho vs QuantLib, long-dated options).
    h_sig, h_r = min(1e-2, 0.5 * m.sigma), 1e-3
    vega = (price(replace(m, sigma=m.sigma + h_sig))
            - price(replace(m, sigma=m.sigma - h_sig))) / (2 * h_sig)
    rho = (price(replace(m, r=m.r + h_r)) - price(replace(m, r=m.r - h_r))) / (2 * h_r)
    return dict(price=v0[0], delta=delta, gamma=gamma, vega=vega, rho=rho, theta=theta)


# --------------------------------------------------------------------------- #
# Quick demo
# --------------------------------------------------------------------------- #
def _demo() -> None:
    m = BlackScholes(s0=100.0, r=0.05, sigma=0.2)
    exact = bs_greeks(m, 100.0, 1.0)
    mc = mc_european_greeks(m, 100.0, 1.0, n_paths=1_000_000)
    print("European call S=K=100, r=5%, sigma=20%, T=1 (MC 1M paths)\n")
    print(f"{'':<7}{'Monte Carlo':>13}{'± CI95':>10}{'exact':>12}")
    for g_ in GREEKS:
        print(f"{g_:<7}{mc[g_].price:13.5f}{1.96 * mc[g_].std_error:10.5f}{float(exact[g_]):12.5f}")
    print("\nFor the full demo: python bench_greeks.py")

