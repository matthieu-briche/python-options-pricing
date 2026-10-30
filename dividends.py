"""
dividends.py — Options américaines avec dividendes discrets (arbre CRR).

Deux modèles de dividendes, les mêmes que QuantLib (FdBlackScholesVanillaEngine) :

  model="spot"      Le cours chute de D à la date de détachement ; la volatilité
                    s'applique au cours entier. Le plus réaliste.
                    Implémentation : arbre CRR à grille fixe ; à chaque date de
                    détachement, V(t-, S) = V(t+, S - D) par interpolation sur les
                    nœuds (méthode de Vellekoop & Nieuwenhuis, 2006).

  model="escrowed"  La volatilité s'applique à S* = S - VA(dividendes futurs).
                    L'arbre porte S* ; l'exercice utilise S = S* + VA(dividendes
                    restants). Simple, mais sous-estime la variance du cours.

Dividendes : liste de (t_i, D_i), t_i en années depuis aujourd'hui (date de
détachement), D_i en unités de cours. Seuls ceux tels que 0 < t_i <= T comptent.
Le taux continu q du modèle reste utilisable (repo / coût d'emprunt).
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
    """Renvoie {pas: (S des nœuds, valeurs)} pour les pas 0, 1, 2, et dt."""
    dt = T / n
    u = math.exp(m.sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp((m.r - m.q) * dt) - d) / (u - d)
    check_proba_crr(p, m, dt)
    disc = math.exp(-m.r * dt)
    sgn = 1.0 if kind == "call" else -1.0

    # Dividendes ramenés au pas de temps le plus proche (pas >= 1)
    divs = [(min(max(round(t / dt), 1), n), D) for t, D in dividends if 0 < t <= T and D > 0]
    div_at = {}
    for k, D in divs:
        div_at[k] = div_at.get(k, 0.0) + D

    if model == "escrowed":
        # VA, à la date du pas k, des dividendes encore à recevoir (pas >= k : cum-dividende)
        pv = np.zeros(n + 1)            # pv[k] : VA en t_k des dividendes aux pas >= k
        for kd, D in div_at.items():
            ks = np.arange(kd + 1)
            pv[ks] += D * np.exp(-m.r * (kd - ks) * dt)

        def pv_remaining(k):
            return pv[k]
        base = m.s0 - pv_remaining(0)
        if base <= 0:
            raise ValueError("Dividendes supérieurs au cours : modèle escrowed impossible")
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
    """Modèle spot : grille fixe + saut de dividende par interpolation."""
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
        """V(t_k^-, S) = V(t_k^+, S - D) ; exercice possible juste avant détachement."""
        D = div_at[k]
        x = s_k - D
        # Interpolation cubique (une interpolation linéaire surestime une fonction
        # convexe : biais ~ gamma·h²/8 par dividende). Sous le plus petit nœud,
        # prolongement linéaire vers la valeur en S=0 (0 pour un call, K pour un put).
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
    raise ValueError("model doit valoir 'spot' ou 'escrowed'")


def crr_american_div(m: BlackScholes, K: float, T: float, kind: OptionType = "put",
                     dividends: Dividends = (), n: int = 2000,
                     model: DivModel = "spot") -> dict[str, float]:
    """
    Prix et grecques d'une option américaine avec dividendes discrets.
    Mêmes conventions que greeks.crr_american (vega pour +1.00 de vol, rho pour
    +1.00 de taux, theta par an). Delta/gamma/theta lus sur les premiers nœuds ;
    vega/rho par re-pricing (dividendes en montants fixes, inchangés).
    Hypothèse : pas de détachement dans les deux premiers pas de l'arbre.

    Précision : prix à ~0,001 de QuantLib (n=2000). Avec un gros dividende proche
    (ex. 5 % du cours sous un mois), le rho garde un biais de discrétisation en
    O(1/n) d'environ 2 % ; augmentez n, ou extrapolez : rho ≈ 2·rho(2n) - rho(n).
    """
    dt0 = T / n
    precoces = [t for t, D in dividends if 0 < t <= T and D > 0 and round(t / dt0) <= 2]
    if precoces:
        warnings.warn(
            f"Détachement dans les deux premiers pas de l'arbre (t={min(precoces):.2e} an, "
            f"dt={dt0:.2e}) : delta, gamma et theta lus sur les premiers nœuds sont "
            "faussés. Augmentez n ou traitez le dividende séparément.",
            UserWarning, stacklevel=2)
    price = lambda mm: _tree(mm, K, T, kind, dividends, n, model)[0][0][1][0]
    k, dt = _tree(m, K, T, kind, dividends, n, model)
    (s1, v1), (s2, v2), (_, v0) = k[1], k[2], k[0]
    delta = (v1[1] - v1[0]) / (s1[1] - s1[0])
    gamma = ((v2[2] - v2[1]) / (s2[2] - s2[1]) - (v2[1] - v2[0]) / (s2[1] - s2[0])) \
        / (0.5 * (s2[2] - s2[0]))
    # Theta : variation entre t=0 et t=2dt au nœud central. En modèle escrowed, ce
    # nœud ne correspond pas exactement à S0 (la VA des dividendes croît avec le
    # temps) : on retire l'effet delta du décalage de cours.
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
    """Prix seul (sans grecques), interface à plat : utile pour inverser la vol implicite."""
    m = BlackScholes(s0=S, r=r, sigma=sigma, q=q)
    return float(_tree(m, K, T, kind, dividends, n, model)[0][0][1][0])
