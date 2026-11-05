"""
svi.py — SVI parametrisation of the volatility smile (Gatheral, 2004).

Total implied variance w(k) = σ_impl(k)² · T, in log-moneyness k = ln(K/F):

    w(k) = a + b · ( ρ (k − m) + sqrt((k − m)² + s²) )

    a : level          b : wing slope (≥ 0)           ρ : asymmetry (skew), |ρ| < 1
    m : translation    s : curvature at the minimum (> 0)

Advantages over a polynomial: wings linear in variance (consistent with Roger
Lee's theorem), sound extrapolation beyond quoted strikes, interpretable
parameters, and checkable no-arbitrage conditions.

Arbitrage checks provided:
  - butterfly: positive risk-neutral density ⇔ g(k) ≥ 0 (Durrleman)
  - calendar: w(k, T) increasing in T for fixed k
  - Roger Lee bound on the wings: b (1 + |ρ|) ≤ 4 / T  (enforced in the fit)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class SVI:
    a: float
    b: float
    rho: float
    m: float
    s: float
    T: float

    def w(self, k):
        """Total implied variance."""
        k = np.asarray(k, float)
        d = k - self.m
        return self.a + self.b * (self.rho * d + np.sqrt(d * d + self.s * self.s))

    def vol(self, k):
        """Implied volatility (annualised)."""
        return np.sqrt(np.maximum(self.w(k), 1e-12) / self.T)

    def __call__(self, k):
        return self.vol(k)

    # Analytical derivatives of w with respect to k
    def _dw(self, k):
        d = np.asarray(k, float) - self.m
        r = np.sqrt(d * d + self.s * self.s)
        return self.b * (self.rho + d / r), self.b * self.s * self.s / r ** 3

    def g(self, k):
        """Durrleman function: the density is positive iff g(k) >= 0."""
        k = np.asarray(k, float)
        w = np.maximum(self.w(k), 1e-12)
        w1, w2 = self._dw(k)
        return (1 - k * w1 / (2 * w)) ** 2 - w1 ** 2 / 4 * (1 / w + 0.25) + w2 / 2

    def min_variance(self):
        return self.a + self.b * self.s * np.sqrt(1 - self.rho ** 2)


def calibrate_svi(k, iv, T, poids=None, n_essais=8, seed=0) -> SVI:
    """
    Weighted least-squares fit on implied vol, with constraints:
    b >= 0, |ρ| < 1, s > 0, minimum variance >= 0, b(1+|ρ|) <= 4/T (Roger Lee).
    No butterfly arbitrage, enforced by a penalty on g(k) < 0, on a grid
    k ∈ [-1.5, 1.5] covering the extrapolated wings (where the data constrain
    nothing and a free fit easily degenerates).
    Several starting points: the problem is not convex.
    """
    k, iv = np.asarray(k, float), np.asarray(iv, float)
    poids = np.ones_like(k) if poids is None else np.asarray(poids, float)
    poids = poids / poids.mean()
    w_mkt = iv ** 2 * T
    b_max = 4.0 / T / 1.0  # tightened afterwards through the Roger Lee penalty
    kmin, kmax = k.min(), k.max()
    rng = np.random.default_rng(seed)
    k_grille = np.linspace(-1.5, 1.5, 61)

    def residus(p):
        a, b, rho, m, s = p
        modele = SVI(a, b, rho, m, s, T)
        res = poids * (modele.vol(k) - iv)
        # soft penalties: minimum variance >= 0 and Roger Lee bound
        pen_min = 10.0 * min(modele.min_variance(), 0.0)
        pen_lee = 10.0 * max(b * (1 + abs(rho)) - 4.0 / T, 0.0)
        # 1e-3 margin: the constraint must hold strictly, not "roughly"
        pen_papillon = 100.0 * np.minimum(modele.g(k_grille) - 1e-3, 0.0)
        return np.concatenate([res, [pen_min, pen_lee], pen_papillon])

    bornes = ([-1.0, 0.0, -0.999, kmin - 1.0, 1e-4],
              [max(w_mkt.max(), 1e-4), b_max, 0.999, kmax + 1.0, 2.0])
    meilleur = None
    for i in range(n_essais):
        if i == 0:
            x0 = [w_mkt.min() * 0.9, 0.1, -0.5, 0.0, 0.1]
        else:
            x0 = [rng.uniform(0, w_mkt.min()), rng.uniform(0.01, 1.0), rng.uniform(-0.95, 0.5),
                  rng.uniform(kmin, kmax), rng.uniform(0.01, 0.5)]
        x0 = np.clip(x0, np.array(bornes[0]) + 1e-6, np.array(bornes[1]) - 1e-6)
        try:
            sol = least_squares(residus, x0, bounds=bornes, method="trf", x_scale="jac")
        except ValueError:
            continue
        if meilleur is None or sol.cost < meilleur.cost:
            meilleur = sol
    return SVI(*meilleur.x, T=T)


def butterfly_arbitrage(svi: SVI, k_min=-1.5, k_max=1.5, n=601):
    """Returns (ok, g_min, k where g is minimal)."""
    k = np.linspace(k_min, k_max, n)
    g = svi.g(k)
    i = int(np.argmin(g))
    return bool(g[i] >= -1e-8), float(g[i]), float(k[i])


def calendar_arbitrage(svis: list[SVI], k_min=-0.5, k_max=0.5, n=201, plages=None):
    """
    Total variance increasing in maturity for fixed k. Returns the list of
    expiry pairs in violation (T1, T2, k, maximal w1 - w2).
    plages: optional list of quoted (k_min, k_max) per expiry (same order as
    svis); if provided, each pair is only tested on the intersection of its
    two quoted ranges.
    """
    ordre = sorted(range(len(svis)), key=lambda i: svis[i].T)
    violations = []
    for i1, i2 in zip(ordre, ordre[1:]):
        s1, s2 = svis[i1], svis[i2]
        lo, hi = k_min, k_max
        if plages is not None:
            lo = max(plages[i1][0], plages[i2][0])
            hi = min(plages[i1][1], plages[i2][1])
            if lo >= hi:
                continue
        k = np.linspace(lo, hi, n)
        diff = s1.w(k) - s2.w(k)
        i = int(np.argmax(diff))
        if diff[i] > 1e-6:
            violations.append((s1.T, s2.T, float(k[i]), float(diff[i])))
    return violations
