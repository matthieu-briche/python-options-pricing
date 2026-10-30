"""
svi.py — Paramétrisation SVI du smile de volatilité (Gatheral, 2004).

Variance totale implicite w(k) = σ_impl(k)² · T, en log-moneyness k = ln(K/F) :

    w(k) = a + b · ( ρ (k − m) + sqrt((k − m)² + s²) )

    a : niveau          b : pente des ailes (≥ 0)      ρ : asymétrie (skew), |ρ| < 1
    m : translation     s : courbure au minimum (> 0)

Avantages sur un polynôme : ailes linéaires en variance (conformes au théorème
de Roger Lee), extrapolation saine hors des strikes cotés, paramètres
interprétables, et conditions d'absence d'arbitrage vérifiables.

Contrôles d'arbitrage fournis :
  - papillon (butterfly) : densité risque-neutre positive ⇔ g(k) ≥ 0 (Durrleman)
  - calendaire : w(k, T) croissante en T à k fixé
  - borne de Roger Lee sur les ailes : b (1 + |ρ|) ≤ 4 / T  (imposée au calage)
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
        """Variance totale implicite."""
        k = np.asarray(k, float)
        d = k - self.m
        return self.a + self.b * (self.rho * d + np.sqrt(d * d + self.s * self.s))

    def vol(self, k):
        """Volatilité implicite (annualisée)."""
        return np.sqrt(np.maximum(self.w(k), 1e-12) / self.T)

    def __call__(self, k):
        return self.vol(k)

    # Dérivées analytiques de w en k
    def _dw(self, k):
        d = np.asarray(k, float) - self.m
        r = np.sqrt(d * d + self.s * self.s)
        return self.b * (self.rho + d / r), self.b * self.s * self.s / r ** 3

    def g(self, k):
        """Fonction de Durrleman : la densité est positive ssi g(k) >= 0."""
        k = np.asarray(k, float)
        w = np.maximum(self.w(k), 1e-12)
        w1, w2 = self._dw(k)
        return (1 - k * w1 / (2 * w)) ** 2 - w1 ** 2 / 4 * (1 / w + 0.25) + w2 / 2

    def min_variance(self):
        return self.a + self.b * self.s * np.sqrt(1 - self.rho ** 2)


def calibrate_svi(k, iv, T, poids=None, n_essais=8, seed=0) -> SVI:
    """
    Calage par moindres carrés pondérés sur la vol implicite, avec contraintes :
    b >= 0, |ρ| < 1, s > 0, variance minimale >= 0, b(1+|ρ|) <= 4/T (Roger Lee).
    Absence d'arbitrage papillon imposée par pénalité sur g(k) < 0, sur une grille
    k ∈ [-1.5, 1.5] qui couvre les ailes extrapolées (là où les données ne
    contraignent rien et où un calage libre dégénère facilement).
    Plusieurs points de départ : le problème n'est pas convexe.
    """
    k, iv = np.asarray(k, float), np.asarray(iv, float)
    poids = np.ones_like(k) if poids is None else np.asarray(poids, float)
    poids = poids / poids.mean()
    w_mkt = iv ** 2 * T
    b_max = 4.0 / T / 1.0  # resserré ensuite via la pénalité Roger Lee
    kmin, kmax = k.min(), k.max()
    rng = np.random.default_rng(seed)
    k_grille = np.linspace(-1.5, 1.5, 61)

    def residus(p):
        a, b, rho, m, s = p
        modele = SVI(a, b, rho, m, s, T)
        res = poids * (modele.vol(k) - iv)
        # pénalités douces : variance minimale >= 0 et borne de Roger Lee
        pen_min = 10.0 * min(modele.min_variance(), 0.0)
        pen_lee = 10.0 * max(b * (1 + abs(rho)) - 4.0 / T, 0.0)
        # marge de 1e-3 : la contrainte doit tenir strictement, pas « à peu près »
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
    """Renvoie (ok, g_min, k où g est minimal)."""
    k = np.linspace(k_min, k_max, n)
    g = svi.g(k)
    i = int(np.argmin(g))
    return bool(g[i] >= -1e-8), float(g[i]), float(k[i])


def calendar_arbitrage(svis: list[SVI], k_min=-0.5, k_max=0.5, n=201, plages=None):
    """
    Variance totale croissante en maturité à k fixé. Renvoie la liste des
    couples d'échéances en violation (T1, T2, k, w1 - w2 maximal).
    plages : liste optionnelle de (k_min, k_max) cotés par échéance (même ordre que
    svis) ; si fournie, chaque paire n'est testée que sur l'intersection de ses
    deux plages cotées.
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
