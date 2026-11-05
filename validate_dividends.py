"""
Validation of dividends.crr_american_div (American options, discrete dividends).

  1. Prices and Greeks vs QuantLib, "spot" and "escrowed" models
  2. Properties: no dividend = crr_american; monotonic in dividend amount;
     American call >= European call (early exercise before the ex-date);
     deep ITM call exercised just before the ex-date
  3. Difference between the two models (information, not a test)

Usage: python validate_dividends.py
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
# Dividend schedules, in days until the ex-date
ECHEANCIERS = {
    "quarterly 0.75": [(40, 0.75), (131, 0.75), (222, 0.75), (313, 0.75), (404, 0.75)],
    "semi-annual 2.5": [(75, 2.5), (257, 2.5)],
    "large dividend 5": [(20, 5.0)],
}


def ql_european_div(S, K, days, r, sig, kind, divs):
    """European with discrete dividends (analytical formula, escrowed model)."""
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

    # 1. QuantLib comparison
    for (nom_ech, ech), K, days, sig, kind, model in itertools.product(
            ECHEANCIERS.items(), [90, 100, 110], [91, 365], [0.20, 0.40],
            ["call", "put"], ["spot", "escrowed"]):
        divs = [(d, D) for d, D in ech if d < days]
        T, r = days / 365, 0.04
        g = crr_american_div(BlackScholes(S0, r, sig), K, T, kind,
                             [(d / 365, D) for d, D in divs], model=model)
        ref = ql_reference(S0, K, days, r, 0.0, sig, kind, dividends=divs, div_model=model)
        nom = f"{model} {kind} K={K} T={days}d vol={sig} [{nom_ech}]"
        ligne = dict(modele=model, type=kind, K=K, days=days, sigma=sig, dividendes=nom_ech)
        for gk_ql, gk in [("prix", "price"), ("delta", "delta"), ("gamma", "gamma"),
                          ("vega", "vega"), ("theta", "theta"), ("rho", "rho")]:
            mine, theirs = g[gk], ref[gk_ql]
            ligne[gk_ql], ligne[gk_ql + "_ql"] = mine, theirs
            if gk == "price":
                ok = is_close(mine, theirs, TOL_PRIX_ABS, TOL_PRIX_REL)
            elif gk == "rho" and nom_ech.startswith("large"):
                # O(1/n) discretisation bias documented in crr_american_div
                ok = is_close(mine, theirs, TOL_GREC[gk], 0.03)
            else:
                ok = is_close(mine, theirs, TOL_GREC[gk], TOL_GREC_REL)
            bilan.check(f"1. {gk.capitalize()} vs QuantLib (dividends)", nom, ok,
                        f"yours={mine:.5f} QL={theirs:.5f}")
        lignes.append(ligne)

    # 2. Properties
    sec = "2. Properties with dividends"
    m = BlackScholes(S0, 0.04, 0.25)
    for kind in ["call", "put"]:
        a = crr_american_div(m, 100, 1.0, kind, [])["price"]
        b = crr_american(m, 100, 1.0, kind)["price"]
        bilan.check(sec, f"No dividend = crr_american ({kind})", abs(a - b) < 1e-9)
        prix = [crr_american_div(m, 100, 1.0, kind, [(0.4, D)])["price"] for D in [0, 1, 2, 4]]
        mono = all(x > y for x, y in zip(prix, prix[1:])) if kind == "call" \
            else all(x < y for x, y in zip(prix, prix[1:]))
        bilan.check(sec, f"{kind} {'decreasing' if kind == 'call' else 'increasing'} "
                    f"in the dividend", mono, f"{[round(x, 3) for x in prix]}")
    for K in [80, 100]:
        divs = [(60, 3.0)]
        am = crr_american_div(m, K, 0.5, "call", [(60 / 365, 3.0)], model="escrowed")["price"]
        eu = ql_european_div(S0, K, 182, 0.04, 0.25, "call", divs)
        bilan.check(sec, f"American call >= European with dividend (K={K})", am >= eu - 1e-3,
                    f"am={am:.4f} eu={eu:.4f} early exercise premium={am - eu:.4f}")
    # Deep ITM call, large imminent dividend: exercising just before the ex-date
    # is worth S - K discounted over a few days, the option is worth at least S - K
    a = crr_american_div(BlackScholes(S0, 0.04, 0.15), 50, 0.5, "call", [(5 / 365, 8.0)])["price"]
    bilan.check(sec, "Deep ITM call + large dividend ≈ S - K (early exercise)",
                abs(a - 50) < 0.05, f"price={a:.4f}")

    # 3. Difference between models
    df = pd.DataFrame(lignes)
    piv = df.pivot_table(index=["type", "K", "days", "sigma", "dividendes"],
                         columns="modele", values="prix")
    piv["spot - escrowed"] = piv["spot"] - piv["escrowed"]
    df.to_csv("rapport_dividendes.csv", index=False)
    print("\n   Price difference between models (spot - escrowed):")
    print("   mean {:+.4f} | min {:+.4f} | max {:+.4f}".format(
        piv["spot - escrowed"].mean(), piv["spot - escrowed"].min(), piv["spot - escrowed"].max()))
    e = (df.prix - df.prix_ql).abs()
    print(f"   Price difference vs QL: mean {e.mean():.5f} | max {e.max():.5f} | "
          f"{len(df)} options | duration {time.time() - t0:.0f}s")
    bilan.afficher()
    print("Note: rho tested at 3% (instead of 1%) for the 'large dividend' case:")
    print("       O(1/n) tree bias, documented in crr_american_div.")
    print("Details: rapport_dividendes.csv")


if __name__ == "__main__":
    main()
