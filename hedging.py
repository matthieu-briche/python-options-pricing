"""
hedging.py — Backtest de couverture : P&L d'une stratégie de delta-hedging.

Le prix d'une option est le coût de sa couverture (Bouchard & Chassagneux,
*Fundamentals and Advanced Techniques in Derivatives Hedging*, Springer 2016).
Ce module ne vérifie plus qu'un prix est juste : il vérifie qu'il est COUVRABLE,
et mesure ce que coûte une couverture imparfaite.

Le vendeur encaisse la prime p(0, S0) calculée avec la vol de couverture σ̃
(m.sigma), puis rebalance N fois une position en actions (et, en option, en
une seconde option pour neutraliser le gamma). Le cours suit sa vraie
dynamique : vol réalisée σ (constante, ou différente par trajectoire) et
dérive réelle μ (pas forcément la dérive risque-neutre).

Deux sources d'erreur (B&C, § 7.3) :

  1. Vol mal spécifiée (§ 7.3.1), en temps continu :
         P&L = ½ ∫_0^T e^{-rt} (σ̃² − σ_t²) S_t² Γ_t dt          (actualisé en 0)
     Gamma > 0 et σ̃ > σ : gain ; σ̃ < σ : perte. Gamma nul : insensible à σ.
     Si σ̃ = σ_max et le payoff est convexe, P&L >= 0 sur toute trajectoire
     (équation de Black-Scholes-Barenblatt, § 5.2.1).

  2. Rebalancement discret (§ 7.3.2) : erreur d'écart-type en O(1/√N).
     Au premier ordre, chaque pas contribue −½ Γ S² σ² (Z² − 1) Δt, d'où
         Var(P&L) ≈ Σ_i e^{-2r t_i} ½ σ⁴ Δt² E[Γ(t_i, S_{t_i})² S_{t_i}⁴]
     calculé par quadrature dans theoretical_std().

Conventions : dividende continu q réinvesti en actions ; cash rémunéré à r ;
P&L = (valeur du portefeuille − payoff) à l'échéance, actualisé en 0.
Positif = gain du vendeur.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.special import ndtr

from pricer import BlackScholes, OptionType

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


# --------------------------------------------------------------------------- #
# Black-Scholes vectorisé sur le spot (le pricer de référence l'est sur K, T)
# --------------------------------------------------------------------------- #
def _bs(S: np.ndarray, K: float, tau: float, r: float, q: float, sig: float,
        kind: OptionType):
    """(prix, delta, gamma) pour un vecteur de spots, à maturité résiduelle tau."""
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
# Résultats
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HedgedOption:
    """Seconde option, traitée à son prix modèle (vol σ̃), pour neutraliser le gamma."""
    K: float
    T: float                     # doit être >= l'échéance de l'option couverte
    kind: OptionType = "call"


@dataclass
class HedgingResult:
    pnl: np.ndarray              # P&L actualisé en 0, par trajectoire
    erreur_vol: np.ndarray       # ½ ∫ e^{-rt} (σ̃² − σ_t²) S² Γ_net dt (somme de Riemann)
    prime: float                 # prix encaissé à t = 0 (vol σ̃)
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
        """Perte moyenne dans les `niveau` pires cas (expected shortfall), en P&L."""
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
        return (f"prime {d['prime']:.4f} | N={d['n_reb']} | P&L moyen {d['moyenne']:+.4f} "
                f"± {1.96 * d['erreur_type']:.4f} | écart-type {d['ecart_type']:.4f} "
                f"({d['ecart_type_rel']:.1%} de la prime) | CVaR 5 % {d['cvar05']:+.4f}")


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
    Vend l'option (K, T, kind) au prix Black-Scholes de vol m.sigma (= σ̃) et la
    couvre en delta (et en gamma si `couverture_gamma`) à n_reb dates équidistantes.

    sigma_reelle : vol réalisée. None -> σ̃ (modèle juste) ; float -> constante ;
                   tableau (n_paths,) -> une vol constante par trajectoire.
    mu           : dérive réelle du cours (hors dividende). None -> r − q (risque-neutre).

    Simulation exacte du cours entre deux dates (log-normal) : la seule erreur
    est celle de la couverture.
    """
    sig_c = m.sigma
    mu = m.r - m.q if mu is None else mu
    if sigma_reelle is None:
        sigma_reelle = sig_c
    sig_arr = np.broadcast_to(np.asarray(sigma_reelle, float), (n_paths,))
    if np.any(sig_arr <= 0):
        raise ValueError("sigma_reelle doit être > 0")
    G = couverture_gamma
    if G is not None and G.T < T - 1e-12:
        raise ValueError("L'option de couverture doit vivre au moins jusqu'à l'échéance couverte")

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
            """Nombre d'actions, nombre d'options de couverture, gamma net."""
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
            a = a * reinv                       # dividende réinvesti en actions
            cash = cash * capi
            t1 = (i + 1) * dt
            if G is not None:
                H = _bs(S, G.K, G.T - t1, m.r, m.q, sig_c, G.kind)[0]
            if i + 1 < n_reb:                   # rebalancement autofinancé
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
# Référence : écart-type théorique de l'erreur de rebalancement discret
# --------------------------------------------------------------------------- #
def theoretical_std(m: BlackScholes, K: float, T: float, kind: OptionType = "call",
                         n_reb: int = 52, mu: float | None = None, n_quad: int = 4001) -> float:
    """
    Premier ordre de l'erreur de couverture discrète, modèle juste (σ réalisée = σ̃) :
        Var ≈ Σ_{i<N} e^{-2r t_i} ½ σ⁴ Δt² E[Γ(t_i, S_{t_i})² S_{t_i}⁴]
    L'espérance est calculée sous la loi réelle de S_{t_i} (dérive mu) par une
    règle des trapèzes sur une grille gaussienne fine : près de l'échéance, Γ est
    un pic étroit (largeur σ√Δt) qu'une quadrature de Gauss-Hermite rate (sous-
    estimation de 3 à 4 % dès N = 50). Indépendant du type call/put (même gamma).
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
    print("Vente d'un call ATM 6 mois, couvert en delta (100 000 trajectoires)\n")
    for n in (12, 52, 252):
        res = simulate_hedging(m, K, T, n_reb=n)
        print(f"N={n:4d}  {res}")
        print(f"        écart-type théorique (1er ordre) : {theoretical_std(m, K, T, n_reb=n):.4f}")
    print("\nVol de couverture 20 %, vol réalisée 25 % (sous-estimation) :")
    print("  ", simulate_hedging(m, K, T, n_reb=252, sigma_reelle=0.25))
    print("\nPour la validation complète : python validate_hedging.py")


if __name__ == "__main__":
    _demo()
