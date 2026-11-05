"""
dividends.py — American options with discrete dividends (CRR tree).

Two dividend models, the same as QuantLib (FdBlackScholesVanillaEngine):

  model="spot"      The share price drops by D on the ex-date; volatility
                    applies to the whole price. The most realistic model.
                    Implementation: CRR tree on a fixed grid; on each
                    ex-date, V(t-, S) = V(t+, S - D) by interpolation on the
                    nodes (Vellekoop & Nieuwenhuis method, 2006).

  model="escrowed"  Volatility applies to S* = S - PV(future dividends).
                    The tree carries S*; exercise uses S = S* + PV(remaining
                    dividends). Simple, but underestimates the price variance.

Dividends: list of (t_i, D_i), t_i in years from today (ex-date),
D_i in price units. Only those with 0 < t_i <= T count.
The model's continuous rate q can still be used (repo / borrow cost).
"""
from __future__ import annotations

import math
import warnings
from dataclasses import replace
from typing import Literal, Sequence

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

from pricer import BlackScholes, check_proba_crr

OptionType = Literal["call", "put"]
DivModel = Literal["spot", "escrowed"]
Dividends = Sequence[tuple[float, float]]


def _tree_value(m: BlackScholes, K: float, T: float, kind: OptionType,
                dividends: Dividends, n: int, model: DivModel):
    """Returns {step: (node S, values)} for steps 0, 1, 2, and dt."""
    dt = T / n
    u = math.exp(m.sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp((m.r - m.q) * dt) - d) / (u - d)
    check_proba_crr(p, m, dt)
    disc = math.exp(-m.r * dt)
    sgn = 1.0 if kind == "call" else -1.0

    # Dividends moved to the nearest time step (step >= 1)
    divs = [(min(max(round(t / dt), 1), n), D) for t, D in dividends if 0 < t <= T and D > 0]
    div_at = {}
    for k, D in divs:
        div_at[k] = div_at.get(k, 0.0) + D

    if model == "escrowed":
        # PV, at the date of step k, of the dividends still to be received (step >= k: cum-dividend)
        pv = np.zeros(n + 1)            # pv[k]: PV at t_k of the dividends at steps >= k
        for kd, D in div_at.items():
            ks = np.arange(kd + 1)
            pv[ks] += D * np.exp(-m.r * (kd - ks) * dt)

        def pv_remaining(k):
            return pv[k]
        base = m.s0 - pv_remaining(0)
        if base <= 0:
            raise ValueError("Dividends exceed the share price: escrowed model not possible")
    else:
        base = m.s0

    def nodes(k):
        j = np.arange(k + 1)
        return base * u ** (2.0 * j - k)

    def spot(k, s_nodes):
        return s_nodes + pv_remaining(k) if model == "escrowed" else s_nodes

    s_n = nodes(n)
    v = np.maximum(sgn * (spot(n, s_n) - K), 0.0)
    keep = {}
    for k in range(n - 1, -1, -1):
        s_k = nodes(k)
        v = np.maximum(disc * (p * v[1:] + (1 - p) * v[:-1]), sgn * (spot(k, s_k) - K))
        if k <= 2:
            keep[k] = (spot(k, s_k), v.copy())
    return keep, dt


def _tree_spot(m, K, T, kind, dividends, n):
    """Spot model: fixed grid + dividend jump by interpolation."""
    dt = T / n
    u = math.exp(m.sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp((m.r - m.q) * dt) - d) / (u - d)
    check_proba_crr(p, m, dt)
    disc = math.exp(-m.r * dt)
    sgn = 1.0 if kind == "call" else -1.0
    div_at = {}
    for t, D in dividends:
        if 0 < t <= T and D > 0:
            k = min(max(round(t / dt), 1), n)
            div_at[k] = div_at.get(k, 0.0) + D

    def nodes(k):
        return m.s0 * u ** (2.0 * np.arange(k + 1) - k)

    def jump(k, s_k, v_ex):
        """V(t_k^-, S) = V(t_k^+, S - D); exercise possible just before the ex-date."""
        D = div_at[k]
        x = s_k - D
        # Cubic interpolation (linear interpolation overestimates a convex
        # function: bias ~ gamma·h²/8 per dividend). Below the smallest node,
        # linear extension towards the value at S=0 (0 for a call, K for a put).
        v_cum = np.empty_like(x)
        inside = x >= s_k[0]
        if inside.any():
            v_cum[inside] = CubicSpline(s_k, v_ex)(x[inside])
        if (~inside).any():
            v_zero = 0.0 if kind == "call" else K
            xo = np.maximum(x[~inside], 0.0)
            v_cum[~inside] = v_zero + (v_ex[0] - v_zero) * xo / s_k[0]
        return np.maximum(v_cum, sgn * (s_k - K))

    s_n = nodes(n)
    v = np.maximum(sgn * (s_n - K), 0.0)
    if n in div_at:
        v = jump(n, s_n, v)
    keep = {}
    for k in range(n - 1, -1, -1):
        s_k = nodes(k)
        v = np.maximum(disc * (p * v[1:] + (1 - p) * v[:-1]), sgn * (s_k - K))
        if k in div_at:
            v = jump(k, s_k, v)
        if k <= 2:
            keep[k] = (s_k, v.copy())
    return keep, dt


def _tree(m, K, T, kind, dividends, n, model):
    if model == "spot":
        return _tree_spot(m, K, T, kind, dividends, n)
    if model == "escrowed":
        return _tree_value(m, K, T, kind, dividends, n, model)
    raise ValueError("model must be 'spot' or 'escrowed'")


def crr_american_div(m: BlackScholes, K: float, T: float, kind: OptionType = "put",
                     dividends: Dividends = (), n: int = 2000,
                     model: DivModel = "spot") -> dict[str, float]:
    """
    Price and Greeks of an American option with discrete dividends.
    Same conventions as greeks.crr_american (vega per +1.00 of vol, rho per
    +1.00 of rate, theta per year). Delta/gamma/theta read from the first nodes;
    vega/rho by re-pricing (dividends as fixed, unchanged amounts).
    Assumption: no ex-date within the first two tree steps.

    Accuracy: price within ~0.001 of QuantLib (n=2000). With a large near-term dividend
    (e.g. 5% of the price within a month), rho keeps a discretisation bias in
    O(1/n) of about 2%; increase n, or extrapolate: rho ≈ 2·rho(2n) - rho(n).
    """
    dt0 = T / n
    precoces = [t for t, D in dividends if 0 < t <= T and D > 0 and round(t / dt0) <= 2]
    if precoces:
        warnings.warn(
            f"Ex-date within the first two tree steps (t={min(precoces):.2e} yr, "
            f"dt={dt0:.2e}): delta, gamma and theta read from the first nodes are "
            "distorted. Increase n or handle the dividend separately.",
            UserWarning, stacklevel=2)
    price = lambda mm: _tree(mm, K, T, kind, dividends, n, model)[0][0][1][0]
    k, dt = _tree(m, K, T, kind, dividends, n, model)
    (s1, v1), (s2, v2), (_, v0) = k[1], k[2], k[0]
    delta = (v1[1] - v1[0]) / (s1[1] - s1[0])
    gamma = ((v2[2] - v2[1]) / (s2[2] - s2[1]) - (v2[1] - v2[0]) / (s2[1] - s2[0])) \
        / (0.5 * (s2[2] - s2[0]))
    # Theta: change between t=0 and t=2dt at the central node. In the escrowed model, this
    # node does not match S0 exactly (the PV of dividends grows with
    # time): the delta effect of the price shift is removed.
    s0_node = k[0][0][0]
    theta = (v2[1] - v0[0] - delta * (s2[1] - s0_node)) / (2 * dt)
    h_sig, h_r = min(1e-2, 0.5 * m.sigma), 1e-3
    vega = (price(replace(m, sigma=m.sigma + h_sig))
            - price(replace(m, sigma=m.sigma - h_sig))) / (2 * h_sig)
    rho = (price(replace(m, r=m.r + h_r)) - price(replace(m, r=m.r - h_r))) / (2 * h_r)
    return dict(price=float(v0[0]), delta=float(delta), gamma=float(gamma),
                vega=float(vega), rho=float(rho), theta=float(theta))


def american_price_div(S: float, K: float, T: float, r: float, q: float, sigma: float,
                       kind: OptionType, dividends: Dividends = (), n: int = 400,
                       model: DivModel = "spot") -> float:
    """Price only (no Greeks), flat interface: useful for inverting implied vol."""
    m = BlackScholes(s0=S, r=r, sigma=sigma, q=q)
    return float(_tree(m, K, T, kind, dividends, n, model)[0][0][1][0])
