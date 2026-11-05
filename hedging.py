"""
hedging.py — Hedging backtest: P&L of a delta-hedging strategy.

The price of an option is the cost of hedging it (Bouchard & Chassagneux,
*Fundamentals and Advanced Techniques in Derivatives Hedging*, Springer 2016).
This module no longer checks that a price is right: it checks that it is HEDGEABLE,
and measures what an imperfect hedge costs.

The seller collects the premium p(0, S0) computed with the hedging vol σ̃
(m.sigma), then rebalances a stock position N times (and, optionally, a
second option to neutralise gamma). The price follows its true
dynamics: realised vol σ (constant, or different per path) and
real drift μ (not necessarily the risk-neutral drift).

Two sources of error (B&C, § 7.3):

  1. Misspecified vol (§ 7.3.1), in continuous time:
         P&L = ½ ∫_0^T e^{-rt} (σ̃² − σ_t²) S_t² Γ_t dt          (discounted to 0)
     Gamma > 0 and σ̃ > σ: gain; σ̃ < σ: loss. Zero gamma: insensitive to σ.
     If σ̃ = σ_max and the payoff is convex, P&L >= 0 on every path
     (Black-Scholes-Barenblatt equation, § 5.2.1).

  2. Discrete rebalancing (§ 7.3.2): standard deviation of the error in O(1/√N).
     To first order, each step contributes −½ Γ S² σ² (Z² − 1) Δt, hence
         Var(P&L) ≈ Σ_i e^{-2r t_i} ½ σ⁴ Δt² E[Γ(t_i, S_{t_i})² S_{t_i}⁴]
     computed by quadrature in theoretical_std().

Conventions: continuous dividend q reinvested in stock; cash earns r;
P&L = (portfolio value − payoff) at expiry, discounted to 0.
Positive = seller's gain.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.special import ndtr

from pricer import BlackScholes, OptionType

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


# --------------------------------------------------------------------------- #
# Black-Scholes vectorised on the spot (the reference pricer is vectorised on K, T)
# --------------------------------------------------------------------------- #
def _bs(S: np.ndarray, K: float, tau: float, r: float, q: float, sig: float,
        kind: OptionType):
    """(price, delta, gamma) for a vector of spots, at remaining maturity tau."""
    sgn = 1.0 if kind == "call" else -1.0
    if tau <= 0.0:
        dans = (sgn * (S - K) > 0).astype(float)
        return np.maximum(sgn * (S - K), 0.0), sgn * dans, np.zeros_like(S)
    sq = sig * math.sqrt(tau)
    d1 = (np.log(S / K) + (r - q + 0.5 * sig * sig) * tau) / sq
    d2 = d1 - sq
    dfq, dfr = math.exp(-q * tau), math.exp(-r * tau)
    if kind == "call":
        prix = S * dfq * ndtr(d1) - K * dfr * ndtr(d2)
        delta = dfq * ndtr(d1)
    else:
        prix = K * dfr * ndtr(-d2) - S * dfq * ndtr(-d1)
        delta = dfq * (ndtr(d1) - 1.0)
    gamma = dfq * _INV_SQRT_2PI * np.exp(-0.5 * d1 * d1) / (S * sq)
    return prix, delta, gamma


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HedgedOption:
    """Second option, traded at its model price (vol σ̃), to neutralise gamma."""
    K: float
    T: float                     # must be >= the expiry of the hedged option
    kind: OptionType = "call"


@dataclass
class HedgingResult:
    pnl: np.ndarray              # P&L discounted to 0, per path
    erreur_vol: np.ndarray       # ½ ∫ e^{-rt} (σ̃² − σ_t²) S² Γ_net dt (Riemann sum)
    prime: float                 # price collected at t = 0 (vol σ̃)
    n_reb: int
    infos: dict = field(default_factory=dict)

    @property
    def moyenne(self) -> float:
        return float(self.pnl.mean())

    @property
    def ecart_type(self) -> float:
        return float(self.pnl.std(ddof=1))

    @property
    def erreur_type(self) -> float:
        return self.ecart_type / math.sqrt(self.pnl.size)

    def cvar(self, niveau: float = 0.05) -> float:
        """Average loss in the worst `niveau` cases (expected shortfall), in P&L."""
        seuil = np.quantile(self.pnl, niveau)
        return float(self.pnl[self.pnl <= seuil].mean())

    def resume(self) -> dict:
        q = np.quantile(self.pnl, [0.01, 0.05, 0.5, 0.95])
        return dict(prime=self.prime, n_reb=self.n_reb, moyenne=self.moyenne,
                    erreur_type=self.erreur_type, ecart_type=self.ecart_type,
                    ecart_type_rel=self.ecart_type / self.prime,
                    q01=q[0], q05=q[1], mediane=q[2], q95=q[3], cvar05=self.cvar(0.05))

    def __str__(self) -> str:
        d = self.resume()
        return (f"premium {d['prime']:.4f} | N={d['n_reb']} | mean P&L {d['moyenne']:+.4f} "
                f"± {1.96 * d['erreur_type']:.4f} | std dev {d['ecart_type']:.4f} "
                f"({d['ecart_type_rel']:.1%} of premium) | CVaR 5% {d['cvar05']:+.4f}")


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
def simulate_hedging(m: BlackScholes, K: float, T: float, kind: OptionType = "call",
                       n_reb: int = 52, n_paths: int = 100_000,
                       sigma_reelle: float | np.ndarray | None = None,
                       mu: float | None = None,
                       couverture_gamma: HedgedOption | None = None,
                       chunk: int = 2**15, seed: int = 2026) -> HedgingResult:
    """
    Sells the option (K, T, kind) at the Black-Scholes price with vol m.sigma (= σ̃) and
    delta-hedges it (and gamma-hedges it if `couverture_gamma`) on n_reb equally spaced dates.

    sigma_reelle : realised vol. None -> σ̃ (correct model); float -> constant;
                   array (n_paths,) -> one constant vol per path.
    mu           : real drift of the price (excluding dividend). None -> r − q (risk-neutral).

    Exact simulation of the price between two dates (log-normal): the only error
    is the hedging error.
    """
    sig_c = m.sigma
    mu = m.r - m.q if mu is None else mu
    if sigma_reelle is None:
        sigma_reelle = sig_c
    sig_arr = np.broadcast_to(np.asarray(sigma_reelle, float), (n_paths,))
    if np.any(sig_arr <= 0):
        raise ValueError("sigma_reelle must be > 0")
    G = couverture_gamma
    if G is not None and G.T < T - 1e-12:
        raise ValueError("The hedging option must live at least until the hedged expiry")

    dt = T / n_reb
    sqdt = math.sqrt(dt)
    capi, reinv = math.exp(m.r * dt), math.exp(m.q * dt)
    sgn = 1.0 if kind == "call" else -1.0
    rng = np.random.default_rng(seed)
    prime = float(_bs(np.array([m.s0]), K, T, m.r, m.q, sig_c, kind)[0][0])

    pnl = np.empty(n_paths)
    err = np.empty(n_paths)
    for d in range(0, n_paths, chunk):
        b = min(chunk, n_paths - d)
        sig = sig_arr[d:d + b]
        S = np.full(b, m.s0)
        e_vol = np.zeros(b)

        def positions(S, t):
            """Number of shares, number of hedging options, net gamma."""
            _, delta, gamma = _bs(S, K, T - t, m.r, m.q, sig_c, kind)
            if G is None:
                return delta, 0.0, gamma, 0.0
            H, dH, gH = _bs(S, G.K, G.T - t, m.r, m.q, sig_c, G.kind)
            n2 = gamma / np.maximum(gH, 1e-300)
            return delta - n2 * dH, n2, gamma - n2 * gH, H

        a, n2, g_net, H = positions(S, 0.0)
        cash = prime - a * S - n2 * H
        for i in range(n_reb):
            t = i * dt
            e_vol += 0.5 * math.exp(-m.r * t) * (sig_c**2 - sig**2) * S * S * g_net * dt
            z = rng.standard_normal(b)
            S = S * np.exp((mu - 0.5 * sig * sig) * dt + sig * sqdt * z)
            a = a * reinv                       # dividend reinvested in stock
            cash = cash * capi
            t1 = (i + 1) * dt
            if G is not None:
                H = _bs(S, G.K, G.T - t1, m.r, m.q, sig_c, G.kind)[0]
            if i + 1 < n_reb:                   # self-financing rebalancing
                X = a * S + n2 * H + cash
                a, n2, g_net, H_new = positions(S, t1)
                if G is not None:
                    H = H_new
                cash = X - a * S - n2 * H
        X_T = a * S + n2 * H + cash
        pnl[d:d + b] = math.exp(-m.r * T) * (X_T - np.maximum(sgn * (S - K), 0.0))
        err[d:d + b] = e_vol
    return HedgingResult(pnl, err, prime, n_reb,
                              dict(sigma_couverture=sig_c, mu=mu, gamma_neutre=G is not None))


# --------------------------------------------------------------------------- #
# Reference: theoretical standard deviation of the discrete rebalancing error
# --------------------------------------------------------------------------- #
def theoretical_std(m: BlackScholes, K: float, T: float, kind: OptionType = "call",
                         n_reb: int = 52, mu: float | None = None, n_quad: int = 4001) -> float:
    """
    First order of the discrete hedging error, correct model (realised σ = σ̃):
        Var ≈ Σ_{i<N} e^{-2r t_i} ½ σ⁴ Δt² E[Γ(t_i, S_{t_i})² S_{t_i}⁴]
    The expectation is computed under the real law of S_{t_i} (drift mu) with a
    trapezoidal rule on a fine Gaussian grid: near expiry, Γ is
    a narrow spike (width σ√Δt) that Gauss-Hermite quadrature misses (under-
    estimation of 3 to 4% from N = 50). Independent of call/put type (same gamma).
    """
    del kind
    mu = m.r - m.q if mu is None else mu
    x = np.linspace(-9.0, 9.0, n_quad)
    w = np.exp(-0.5 * x * x)
    w = w / w.sum()
    dt = T / n_reb
    var = 0.0
    for i in range(n_reb):
        t = i * dt
        if t == 0.0:
            S = np.array([m.s0]); poids = np.array([1.0])
        else:
            S = m.s0 * np.exp((mu - 0.5 * m.sigma**2) * t + m.sigma * math.sqrt(t) * x)
            poids = w
        g = _bs(S, K, T - t, m.r, m.q, m.sigma, "call")[2]
        var += math.exp(-2 * m.r * t) * 0.5 * m.sigma**4 * dt**2 * float(poids @ (g * g * S**4))
    return math.sqrt(var)


# --------------------------------------------------------------------------- #
def _demo() -> None:
    m = BlackScholes(s0=100.0, r=0.03, sigma=0.20)
    K, T = 100.0, 0.5
    print("Sale of a 6-month ATM call, delta-hedged (100,000 paths)\n")
    for n in (12, 52, 252):
        res = simulate_hedging(m, K, T, n_reb=n)
        print(f"N={n:4d}  {res}")
        print(f"        theoretical std dev (1st order): {theoretical_std(m, K, T, n_reb=n):.4f}")
    print("\nHedging vol 20%, realised vol 25% (underestimated):")
    print("  ", simulate_hedging(m, K, T, n_reb=252, sigma_reelle=0.25))
    print("\nFor the full validation: python validate_hedging.py")


if __name__ == "__main__":
    _demo()
