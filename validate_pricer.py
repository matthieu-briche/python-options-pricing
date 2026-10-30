"""
Validation d'un pricer d'options américaines (prix + grecques).

Usage :
    pip install QuantLib numpy scipy pandas
    python validate_pricer.py              # grille rapide
    python validate_pricer.py --complet    # grille étendue
    python validate_pricer.py --module autre_fichier   # tester un autre module

Le module testé doit exposer :
    price(S, K, T, r, q, sigma, option_type) -> float
    greeks(S, K, T, r, q, sigma, option_type) -> dict(delta, gamma, vega, theta, rho)
(voir mon_pricer.py pour les conventions)

Tests effectués :
    1. Comparaison prix et grecques avec QuantLib (différences finies, grille fine)
    2. Propriétés théoriques (bornes, parité, monotonie, convexité)
    3. Cohérence interne des grecques (bump-and-reprice de VOTRE pricer)
    4. Convergence numérique
    5. Aller-retour volatilité implicite

Sorties : résumé dans la console + rapport_validation.csv (détail de la grille).
"""
import argparse
import importlib
import itertools
import math
import sys
import time

import numpy as np
import pandas as pd
import QuantLib as ql
from scipy.optimize import brentq
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------------------
TOL_PRIX_ABS = 0.01        # écart absolu toléré sur le prix (S = 100)
TOL_PRIX_REL = 1e-3        # ou écart relatif toléré (le plus large des deux)
TOL_GREC = {"delta": 2e-3, "gamma": 5e-4, "vega": 0.05, "theta": 0.05, "rho": 0.05}
TOL_GREC_REL = 1e-2        # tolérance relative alternative sur les grecques

# Facteurs à appliquer à VOS grecques pour revenir aux conventions du script
# (ex. si votre vega est pour +1 % de vol, mettez "vega": 100)
ECHELLES = {"delta": 1, "gamma": 1, "vega": 1, "theta": 1, "rho": 1}

S0 = 100.0
TODAY = ql.Date(6, 10, 2026)
ql.Settings.instance().evaluationDate = TODAY
DC = ql.Actual365Fixed()


# ---------------------------------------------------------------------------
# Référence QuantLib
# ---------------------------------------------------------------------------
def _ql_option(S, K, days, r, q, sigma, option_type, t_grid=400, x_grid=400,
               dividends=(), div_model="spot"):
    """dividends : liste de (jours jusqu'au détachement, montant)."""
    spot = ql.SimpleQuote(S)
    vol = ql.SimpleQuote(sigma)
    rate = ql.SimpleQuote(r)
    div = ql.SimpleQuote(q)
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(spot),
        ql.YieldTermStructureHandle(ql.FlatForward(TODAY, ql.QuoteHandle(div), DC)),
        ql.YieldTermStructureHandle(ql.FlatForward(TODAY, ql.QuoteHandle(rate), DC)),
        ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(TODAY, ql.NullCalendar(), ql.QuoteHandle(vol), DC)),
    )
    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if option_type == "call" else ql.Option.Put, K)
    exercise = ql.AmericanExercise(TODAY, TODAY + int(days))
    opt = ql.VanillaOption(payoff, exercise)
    if dividends:
        dv = ql.DividendVector([TODAY + int(d) for d, _ in dividends],
                               [float(D) for _, D in dividends])
        model = (ql.FdBlackScholesVanillaEngine.Spot if div_model == "spot"
                 else ql.FdBlackScholesVanillaEngine.Escrowed)
        engine = ql.FdBlackScholesVanillaEngine(process, dv, t_grid, x_grid, 0,
                                                ql.FdmSchemeDesc.Douglas(), False,
                                                -ql.nullDouble(), model)
    else:
        engine = ql.FdBlackScholesVanillaEngine(process, t_grid, x_grid)
    opt.setPricingEngine(engine)
    return opt, vol, rate


def ql_reference(S, K, days, r, q, sigma, option_type, dividends=(), div_model="spot"):
    kw = dict(dividends=dividends, div_model=div_model)
    opt, vol, rate = _ql_option(S, K, days, r, q, sigma, option_type, **kw)
    res = {"prix": opt.NPV(), "delta": opt.delta(), "gamma": opt.gamma()}
    # grille plus fine pour vega/rho : les différences amplifient l'erreur de grille
    opt, vol, rate = _ql_option(S, K, days, r, q, sigma, option_type, 800, 800, **kw)
    # theta : le theta natif du moteur FD est imprécis aux courtes maturités,
    # on le calcule par différence centrée (±1 jour). Le temps qui passe rapproche
    # l'échéance ET les détachements : on décale donc les deux ensemble.
    def shifted(h):
        divs = [(d + h, D) for d, D in dividends if d + h > 0]
        return _ql_option(S, K, days + h, r, q, sigma, option_type,
                          dividends=divs, div_model=div_model)[0].NPV()
    h_dn = 1 if days > 1 else 0
    res["theta"] = -(shifted(1) - shifted(-h_dn)) / ((1 + h_dn) / 365.0)
    # vega et rho par bump-and-reprice (non fournis par le moteur FD).
    # Pas de 0,01 (vol) et 0,001 (taux) : un pas plus petit fait ressortir le bruit
    # de la grille de différences finies (erreurs > 1 % à faible volatilité).
    hv, hr = min(1e-2, 0.5 * sigma), 1e-3
    vol.setValue(sigma + hv); up = opt.NPV()
    vol.setValue(sigma - hv); dn = opt.NPV()
    vol.setValue(sigma)
    res["vega"] = (up - dn) / (2 * hv)
    rate.setValue(r + hr); up = opt.NPV()
    rate.setValue(r - hr); dn = opt.NPV()
    rate.setValue(r)
    res["rho"] = (up - dn) / (2 * hr)
    return res


# ---------------------------------------------------------------------------
# Black-Scholes européen (pour les tests de propriétés)
# ---------------------------------------------------------------------------
def bs_european(S, K, T, r, q, sigma, option_type):
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------
class Report:
    def __init__(self):
        self.lignes = []

    def check(self, section, nom, ok, detail=""):
        self.lignes.append((section, nom, bool(ok), detail))

    def afficher(self):
        sections = dict.fromkeys(l[0] for l in self.lignes)
        total_ko = 0
        for s in sections:
            items = [l for l in self.lignes if l[0] == s]
            ko = [l for l in items if not l[2]]
            total_ko += len(ko)
            print(f"\n[{'OK' if not ko else 'ÉCHEC'}] {s} : {len(items) - len(ko)}/{len(items)}")
            for _, nom, _, detail in ko[:8]:
                print(f"     ✗ {nom}  {detail}")
            if len(ko) > 8:
                print(f"     … et {len(ko) - 8} autres (voir le CSV)")
        print("\n" + ("=" * 60))
        print("RÉSULTAT GLOBAL :", "TOUT EST COHÉRENT" if total_ko == 0
              else f"{total_ko} contrôle(s) en échec")
        print("=" * 60)
        return total_ko


def is_close(a, b, tol_abs, tol_rel):
    return abs(a - b) <= max(tol_abs, tol_rel * abs(b))


def param_grid(complet):
    if complet:
        Ks = [70, 80, 90, 100, 110, 120, 130]
        days = [7, 30, 91, 182, 365, 730]
        vols = [0.10, 0.20, 0.35, 0.60]
        rs = [0.0, 0.03, 0.06]
        qs = [0.0, 0.02, 0.05]
    else:
        Ks = [80, 100, 120]
        days = [30, 182, 730]
        vols = [0.15, 0.40]
        rs = [0.01, 0.05]
        qs = [0.0, 0.03]
    for K, d, v, r, q, t in itertools.product(Ks, days, vols, rs, qs, ["call", "put"]):
        yield dict(S=S0, K=float(K), days=d, T=d / 365.0, r=r, q=q, sigma=v, type=t)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_vs_quantlib(pm, bilan, complet):
    lignes = []
    for c in param_grid(complet):
        args = (c["S"], c["K"], c["T"], c["r"], c["q"], c["sigma"], c["type"])
        ref = ql_reference(c["S"], c["K"], c["days"], c["r"], c["q"], c["sigma"], c["type"])
        p = pm.price(*args)
        g = pm.greeks(*args)
        ligne = dict(c, prix=p, prix_ql=ref["prix"], ecart_prix=p - ref["prix"])
        ok = is_close(p, ref["prix"], TOL_PRIX_ABS, TOL_PRIX_REL)
        nom = f"{c['type']} K={c['K']:.0f} T={c['days']}j vol={c['sigma']} r={c['r']} q={c['q']}"
        bilan.check("1. Prix vs QuantLib", nom, ok,
                    f"vous={p:.4f} QL={ref['prix']:.4f}")
        for gk in ["delta", "gamma", "vega", "theta", "rho"]:
            mine = g[gk] * ECHELLES[gk]
            ligne[gk], ligne[gk + "_ql"] = mine, ref[gk]
            okg = is_close(mine, ref[gk], TOL_GREC[gk], TOL_GREC_REL)
            bilan.check(f"1. {gk.capitalize()} vs QuantLib", nom, okg,
                        f"vous={mine:.5f} QL={ref[gk]:.5f}")
        lignes.append(ligne)
    return pd.DataFrame(lignes)


def test_properties(pm, bilan):
    S, T, sig = S0, 0.5, 0.25
    sec = "2. Propriétés théoriques"

    # Call américain sans dividende = call européen
    for K in [80, 100, 120]:
        a = pm.price(S, K, T, 0.05, 0.0, sig, "call")
        e = bs_european(S, K, T, 0.05, 0.0, sig, "call")
        bilan.check(sec, f"Call sans dividende = européen (K={K})",
                    is_close(a, e, TOL_PRIX_ABS, TOL_PRIX_REL), f"am={a:.4f} eu={e:.4f}")

    # Put américain >= européen et >= intrinsèque ; <= K
    for K in [80, 100, 120, 150]:
        a = pm.price(S, K, T, 0.05, 0.0, sig, "put")
        e = bs_european(S, K, T, 0.05, 0.0, sig, "put")
        bilan.check(sec, f"Put am >= européen (K={K})", a >= e - 1e-6, f"am={a:.4f} eu={e:.4f}")
        bilan.check(sec, f"Put am >= intrinsèque (K={K})", a >= max(K - S, 0) - 1e-6)
        bilan.check(sec, f"Put am <= K (K={K})", a <= K + 1e-9)

    # Put très dans la monnaie avec taux élevé : exercice immédiat
    a = pm.price(S, 200, T, 0.08, 0.0, sig, "put")
    bilan.check(sec, "Put très ITM = valeur intrinsèque", abs(a - 100) < 1e-3, f"prix={a:.5f}")

    # Parité call-put américaine : S e^{-qT} - K <= C - P <= S - K e^{-rT}
    for K, r, q in [(90, 0.05, 0.0), (100, 0.03, 0.02), (110, 0.05, 0.04)]:
        C = pm.price(S, K, T, r, q, sig, "call")
        P = pm.price(S, K, T, r, q, sig, "put")
        bas, haut = S * math.exp(-q * T) - K, S - K * math.exp(-r * T)
        bilan.check(sec, f"Parité américaine (K={K}, r={r}, q={q})",
                    bas - 1e-3 <= C - P <= haut + 1e-3,
                    f"{bas:.3f} <= {C - P:.3f} <= {haut:.3f}")

    # Monotonie en vol et en maturité
    for t in ["call", "put"]:
        pv = [pm.price(S, 100, T, 0.03, 0.02, v, t) for v in [0.1, 0.2, 0.3, 0.5]]
        bilan.check(sec, f"{t} croissant en volatilité", all(np.diff(pv) > 0))
        pt = [pm.price(S, 100, x, 0.03, 0.02, sig, t) for x in [0.1, 0.25, 0.5, 1, 2]]
        bilan.check(sec, f"{t} croissant en maturité", all(np.diff(pt) >= -1e-6))

    # Convexité en strike (butterfly positif) et monotonie en strike
    Ks = np.arange(70, 131, 5.0)
    for t in ["call", "put"]:
        p = np.array([pm.price(S, k, T, 0.03, 0.02, sig, t) for k in Ks])
        fly = p[:-2] - 2 * p[1:-1] + p[2:]
        bilan.check(sec, f"{t} convexe en strike", fly.min() > -1e-4, f"min butterfly={fly.min():.2e}")
        sens = np.diff(p) <= 1e-6 if t == "call" else np.diff(p) >= -1e-6
        bilan.check(sec, f"{t} monotone en strike", all(sens))


def test_internal_greeks(pm, bilan):
    sec = "3. Grecques vs bump-and-reprice de votre pricer"
    # Pas de 5 % sur le spot : un pas plus petit rend le gamma "bump" instable
    # sur les pricers en arbre (le prix oscille en fonction de S entre les nœuds)
    hS, hv, hr, hT = 0.05 * S0, 1e-2, 1e-3, 1 / 365
    tol_rel = {"gamma": 0.05}
    cas = [(100, 0.5, 0.25, 0.03, 0.0, "put"), (90, 1.0, 0.30, 0.05, 0.02, "put"),
           (110, 0.25, 0.20, 0.02, 0.03, "call"), (100, 1.0, 0.40, 0.04, 0.05, "call")]
    for K, T, sig, r, q, t in cas:
        P = lambda S=S0, T=T, sig=sig, r=r: pm.price(S, K, T, r, q, sig, t)
        hD = 0.01 * S0   # pas fin pour le delta (dérivée première)
        fd = {
            "delta": (P(S=S0 + hD) - P(S=S0 - hD)) / (2 * hD),
            "gamma": (P(S=S0 + hS) - 2 * P() + P(S=S0 - hS)) / hS ** 2,
            "vega": (P(sig=sig + hv) - P(sig=sig - hv)) / (2 * hv),
            "rho": (P(r=r + hr) - P(r=r - hr)) / (2 * hr),
            "theta": -(P(T=T + hT) - P(T=T - hT)) / (2 * hT),
        }
        g = pm.greeks(S0, K, T, r, q, sig, t)
        for gk, v in fd.items():
            mine = g[gk] * ECHELLES[gk]
            bilan.check(sec, f"{gk} {t} K={K} T={T}",
                        is_close(mine, v, TOL_GREC[gk], tol_rel.get(gk, 2 * TOL_GREC_REL)),
                        f"analytique={mine:.5f} bump={v:.5f}")
        d = g["delta"] * ECHELLES["delta"]
        borne = (0 <= d <= 1) if t == "call" else (-1 <= d <= 0)
        bilan.check(sec, f"delta dans ses bornes {t} K={K}", borne, f"delta={d:.4f}")
        bilan.check(sec, f"gamma >= 0 {t} K={K}", g["gamma"] >= -1e-8)
        bilan.check(sec, f"vega >= 0 {t} K={K}", g["vega"] >= -1e-8)


def test_convergence(pm, bilan):
    """Le prix doit se stabiliser quand la finesse augmente (si votre pricer accepte n=)."""
    sec = "4. Convergence numérique"
    try:
        ns = [100, 200, 400, 800, 1600]
        p = [pm.price(100, 100, 1.0, 0.05, 0.0, 0.3, "put", n=n) for n in ns]
    except TypeError:
        bilan.check(sec, "Ignoré : price() n'accepte pas d'argument n=", True)
        return
    ecarts = np.abs(np.diff(p))
    ref = ql_reference(100, 100, 365, 0.05, 0.0, 0.3, "put")["prix"]
    print("\n   Convergence (put ATM 1 an) :")
    for n, x in zip(ns, p):
        print(f"     n={n:5d}  prix={x:.5f}  écart QL={x - ref:+.5f}")
    bilan.check(sec, "Écarts successifs décroissants", ecarts[-1] < ecarts[0],
                f"{ecarts.round(5).tolist()}")
    bilan.check(sec, "Erreur finale < tolérance", abs(p[-1] - ref) < TOL_PRIX_ABS,
                f"écart={p[-1] - ref:+.5f}")


def test_implied_vol(pm, bilan):
    sec = "5. Aller-retour volatilité implicite"
    for K, T, sig, t in [(80, 0.25, 0.35, "put"), (100, 0.5, 0.20, "put"),
                         (120, 1.0, 0.25, "call"), (100, 2.0, 0.45, "call")]:
        cible = pm.price(S0, K, T, 0.04, 0.02, sig, t)
        try:
            iv = brentq(lambda v: pm.price(S0, K, T, 0.04, 0.02, v, t) - cible,
                        0.01, 3.0, xtol=1e-8)
            bilan.check(sec, f"{t} K={K} T={T}", abs(iv - sig) < 1e-4,
                        f"vol initiale={sig} retrouvée={iv:.6f}")
        except ValueError as e:
            bilan.check(sec, f"{t} K={K} T={T}", False, f"inversion impossible : {e}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="mon_pricer", help="module contenant price/greeks")
    ap.add_argument("--complet", action="store_true", help="grille étendue (plus long)")
    ap.add_argument("--csv", default="rapport_validation.csv")
    a = ap.parse_args()

    pm = importlib.import_module(a.module)
    bilan = Report()
    t0 = time.time()
    print(f"Validation de '{a.module}' (QuantLib {ql.__version__})")

    df = test_vs_quantlib(pm, bilan, a.complet)
    test_properties(pm, bilan)
    test_internal_greeks(pm, bilan)
    test_convergence(pm, bilan)
    test_implied_vol(pm, bilan)

    df.to_csv(a.csv, index=False)
    e = df["ecart_prix"].abs()
    print(f"\n   Grille : {len(df)} options | écart prix moyen {e.mean():.5f} "
          f"| max {e.max():.5f} | durée {time.time() - t0:.0f}s")
    pire = df.loc[e.idxmax()]
    print(f"   Pire cas : {pire['type']} K={pire['K']:.0f} T={pire['days']}j "
          f"vol={pire['sigma']} r={pire['r']} q={pire['q']}")
    ko = bilan.afficher()
    print(f"Détail par option : {a.csv}")
    sys.exit(1 if ko else 0)


if __name__ == "__main__":
    main()
