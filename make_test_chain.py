"""
Génère une chaîne d'options synthétique réaliste, pricée par QuantLib (moteur
indépendant de votre pricer), pour tester validate_market.py sans connexion.

Paramètres « vrais » connus : taux 4 %, borrow 0,30 %, dividende trimestriel 0,26,
smile avec skew. Le script de validation doit retrouver le borrow et repricer
(quasiment) toute la chaîne dans la fourchette.

Usage : python make_test_chain.py  ->  chaine_test.csv, dividendes_test.csv
"""
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import QuantLib as ql

S, R, BORROW = 227.50, 0.04, 0.003
TODAY = date(2026, 10, 6)
DIVS = [(TODAY + timedelta(days=d), 0.26) for d in (35, 126, 217, 308, 399)]
JOURS = [17, 45, 73, 108, 199, 290]
rng = np.random.default_rng(7)


def vol_smile(K, T):
    F = S * math.exp((R - BORROW) * T)
    x = math.log(K / F) / math.sqrt(T)
    return 0.24 + 0.02 * math.sqrt(T) - 0.09 * x + 0.12 * x * x


def ql_price(K, jours, kind, sig):
    d0 = ql.Date(TODAY.day, TODAY.month, TODAY.year)
    ql.Settings.instance().evaluationDate = d0
    dc = ql.Actual365Fixed()
    proc = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(S)),
        ql.YieldTermStructureHandle(ql.FlatForward(d0, BORROW, dc)),
        ql.YieldTermStructureHandle(ql.FlatForward(d0, R, dc)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(d0, ql.NullCalendar(), sig, dc)))
    dv = [(d0 + (dd - TODAY).days, D) for dd, D in DIVS if (dd - TODAY).days < jours]
    opt = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call if kind == "call" else ql.Option.Put, K),
                           ql.AmericanExercise(d0, d0 + jours))
    eng = ql.FdBlackScholesVanillaEngine(
        proc, ql.DividendVector([d for d, _ in dv], [D for _, D in dv]), 400, 400, 0,
        ql.FdmSchemeDesc.Douglas(), False, -ql.nullDouble(), ql.FdBlackScholesVanillaEngine.Spot)
    opt.setPricingEngine(eng)
    return opt.NPV()


lignes = []
for j in JOURS:
    T = j / 365
    pas = 2.5 if j < 60 else 5.0
    for K in np.arange(round(S * 0.75 / pas) * pas, S * 1.28, pas):
        for kind in ("call", "put"):
            sig = vol_smile(K, T)
            mid = ql_price(K, j, kind, sig)
            if mid < 0.03:
                continue
            spread = round(max(0.02, 0.012 * mid + 0.03 + 0.04 * abs(math.log(K / S))), 2)
            m = mid + rng.uniform(-0.3, 0.3) * spread / 2       # bruit de cotation
            bid, ask = max(round(m - spread / 2, 2), 0.0), round(m + spread / 2, 2)
            lignes.append(dict(expiration=(TODAY + timedelta(days=j)).isoformat(), type=kind,
                               strike=K, bid=bid, ask=ask, volume=int(rng.integers(0, 3000)),
                               open_interest=int(rng.integers(10, 20000)), iv_source=sig))

pd.DataFrame(lignes).to_csv("chaine_test.csv", index=False)
pd.DataFrame([dict(ex_date=d.isoformat(), montant=D) for d, D in DIVS]).to_csv("dividendes_test.csv", index=False)
print(f"{len(lignes)} cotations écrites dans chaine_test.csv (spot {S}, date {TODAY})")
