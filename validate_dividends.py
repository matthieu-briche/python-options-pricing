"""
Validation de dividends.crr_american_div (options américaines, dividendes discrets).

  1. Prix et grecques vs QuantLib, modèles "spot" et "escrowed"
  2. Propriétés : sans dividende = crr_american ; monotonie en montant de dividende ;
     call américain >= call européen (exercice anticipé avant détachement) ;
     call profondément ITM exercé juste avant le détachement
  3. Écart entre les deux modèles (information, pas un test)

Usage : python validate_dividends.py
"""
import itertools
import math
import time

import pandas as pd
import QuantLib as ql

from dividends import crr_american_div
from greeks import crr_american
from pricer import BlackScholes
from validate_pricer import (Report, ql_reference, is_close, TOL_PRIX_ABS, TOL_PRIX_REL,
                            TOL_GREC, TOL_GREC_REL, TODAY, DC)

S0 = 100.0
# Échéanciers de dividendes, en jours jusqu'au détachement
ECHEANCIERS = {
    "trimestriel 0,75": [(40, 0.75), (131, 0.75), (222, 0.75), (313, 0.75), (404, 0.75)],
    "semestriel 2,5": [(75, 2.5), (257, 2.5)],
    "gros dividende 5": [(20, 5.0)],
}


def ql_european_div(S, K, days, r, sig, kind, divs):
    """Européen avec dividendes discrets (formule analytique, modèle escrowed)."""
    proc = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(S)),
        ql.YieldTermStructureHandle(ql.FlatForward(TODAY, 0.0, DC)),
        ql.YieldTermStructureHandle(ql.FlatForward(TODAY, r, DC)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(TODAY, ql.NullCalendar(), sig, DC)))
    opt = ql.VanillaOption(ql.PlainVanillaPayoff(
        ql.Option.Call if kind == "call" else ql.Option.Put, K),
        ql.EuropeanExercise(TODAY + days))
    dv = ql.DividendVector([TODAY + d for d, _ in divs], [D for _, D in divs])
    opt.setPricingEngine(ql.AnalyticDividendEuropeanEngine(proc, dv))
    return opt.NPV()


def main():
    bilan, lignes, t0 = Report(), [], time.time()

    # 1. Comparaison QuantLib
    for (nom_ech, ech), K, days, sig, kind, model in itertools.product(
            ECHEANCIERS.items(), [90, 100, 110], [91, 365], [0.20, 0.40],
            ["call", "put"], ["spot", "escrowed"]):
        divs = [(d, D) for d, D in ech if d < days]
        T, r = days / 365, 0.04
        g = crr_american_div(BlackScholes(S0, r, sig), K, T, kind,
                             [(d / 365, D) for d, D in divs], model=model)
        ref = ql_reference(S0, K, days, r, 0.0, sig, kind, dividends=divs, div_model=model)
        nom = f"{model} {kind} K={K} T={days}j vol={sig} [{nom_ech}]"
        ligne = dict(modele=model, type=kind, K=K, days=days, sigma=sig, dividendes=nom_ech)
        for gk_ql, gk in [("prix", "price"), ("delta", "delta"), ("gamma", "gamma"),
                          ("vega", "vega"), ("theta", "theta"), ("rho", "rho")]:
            mine, theirs = g[gk], ref[gk_ql]
            ligne[gk_ql], ligne[gk_ql + "_ql"] = mine, theirs
            if gk == "price":
                ok = is_close(mine, theirs, TOL_PRIX_ABS, TOL_PRIX_REL)
            elif gk == "rho" and nom_ech.startswith("gros"):
                # biais de discrétisation O(1/n) documenté dans crr_american_div
                ok = is_close(mine, theirs, TOL_GREC[gk], 0.03)
            else:
                ok = is_close(mine, theirs, TOL_GREC[gk], TOL_GREC_REL)
            bilan.check(f"1. {gk_ql.capitalize()} vs QuantLib (dividendes)", nom, ok,
                        f"vous={mine:.5f} QL={theirs:.5f}")
        lignes.append(ligne)

    # 2. Propriétés
    sec = "2. Propriétés avec dividendes"
    m = BlackScholes(S0, 0.04, 0.25)
    for kind in ["call", "put"]:
        a = crr_american_div(m, 100, 1.0, kind, [])["price"]
        b = crr_american(m, 100, 1.0, kind)["price"]
        bilan.check(sec, f"Sans dividende = crr_american ({kind})", abs(a - b) < 1e-9)
        prix = [crr_american_div(m, 100, 1.0, kind, [(0.4, D)])["price"] for D in [0, 1, 2, 4]]
        mono = all(x > y for x, y in zip(prix, prix[1:])) if kind == "call" \
            else all(x < y for x, y in zip(prix, prix[1:]))
        bilan.check(sec, f"{kind} {'décroissant' if kind == 'call' else 'croissant'} "
                    f"avec le dividende", mono, f"{[round(x, 3) for x in prix]}")
    for K in [80, 100]:
        divs = [(60, 3.0)]
        am = crr_american_div(m, K, 0.5, "call", [(60 / 365, 3.0)], model="escrowed")["price"]
        eu = ql_european_div(S0, K, 182, 0.04, 0.25, "call", divs)
        bilan.check(sec, f"Call américain >= européen avec dividende (K={K})", am >= eu - 1e-3,
                    f"am={am:.4f} eu={eu:.4f} prime d'exercice anticipé={am - eu:.4f}")
    # Call très ITM, gros dividende imminent : exercer juste avant le détachement
    # vaut S - K actualisé sur quelques jours, l'option vaut au moins S - K
    a = crr_american_div(BlackScholes(S0, 0.04, 0.15), 50, 0.5, "call", [(5 / 365, 8.0)])["price"]
    bilan.check(sec, "Call très ITM + gros dividende ≈ S - K (exercice anticipé)",
                abs(a - 50) < 0.05, f"prix={a:.4f}")

    # 3. Écart entre modèles
    df = pd.DataFrame(lignes)
    piv = df.pivot_table(index=["type", "K", "days", "sigma", "dividendes"],
                         columns="modele", values="prix")
    piv["spot - escrowed"] = piv["spot"] - piv["escrowed"]
    df.to_csv("rapport_dividendes.csv", index=False)
    print("\n   Écart de prix entre modèles (spot - escrowed) :")
    print("   moyenne {:+.4f} | min {:+.4f} | max {:+.4f}".format(
        piv["spot - escrowed"].mean(), piv["spot - escrowed"].min(), piv["spot - escrowed"].max()))
    e = (df.prix - df.prix_ql).abs()
    print(f"   Écart prix vs QL : moyen {e.mean():.5f} | max {e.max():.5f} | "
          f"{len(df)} options | durée {time.time() - t0:.0f}s")
    bilan.afficher()
    print("Note : rho testé à 3 % (au lieu de 1 %) pour le cas « gros dividende » :")
    print("       biais O(1/n) de l'arbre, documenté dans crr_american_div.")
    print("Détail : rapport_dividendes.csv")


if __name__ == "__main__":
    main()
