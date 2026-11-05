"""
Validation of an American option pricer (prices + Greeks).

Usage:
    pip install QuantLib numpy scipy pandas
    python validate_pricer.py              # quick grid
    python validate_pricer.py --complet    # extended grid
    python validate_pricer.py --module other_file   # test another module

The module under test must expose:
    price(S, K, T, r, q, sigma, option_type) -> float
    greeks(S, K, T, r, q, sigma, option_type) -> dict(delta, gamma, vega, theta, rho)
(see crr_adapter.py for the conventions)

Tests performed:
    1. Comparison of prices and Greeks with QuantLib (finite differences, fine grid)
    2. Theoretical properties (bounds, parity, monotonicity, convexity)
    3. Internal consistency of the Greeks (bump-and-reprice of YOUR pricer)
    4. Numerical convergence
    5. Implied volatility round trip

Outputs: summary in the console + rapport_validation.csv (grid details).
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
# Parameters
# ---------------------------------------------------------------------------
TOL_PRIX_ABS = 0.01        # absolute price difference tolerated (S = 100)
TOL_PRIX_REL = 1e-3        # or relative difference tolerated (the wider of the two)
TOL_GREC = {"delta": 2e-3, "gamma": 5e-4, "vega": 0.05, "theta": 0.05, "rho": 0.05}
TOL_GREC_REL = 1e-2        # alternative relative tolerance on the Greeks

# Factors to apply to YOUR Greeks to get back to the script's conventions
# (e.g. if your vega is per +1% of vol, set "vega": 100)
ECHELLES = {"delta": 1, "gamma": 1, "vega": 1, "theta": 1, "rho": 1}

S0 = 100.0
TODAY = ql.Date(6, 10, 2026)
ql.Settings.instance().evaluationDate = TODAY
DC = ql.Actual365Fixed()


# ---------------------------------------------------------------------------
# QuantLib reference
# ---------------------------------------------------------------------------
def _ql_option(S, K, days, r, q, sigma, option_type, t_grid=400, x_grid=400,
               dividends=(), div_model="spot"):
    """dividends: list of (days until the ex-date, amount)."""
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
    # finer grid for vega/rho: differences amplify the grid error
    opt, vol, rate = _ql_option(S, K, days, r, q, sigma, option_type, 800, 800, **kw)
    # theta: the FD engine's native theta is inaccurate at short maturities,
    # so it is computed by central difference (±1 day). The passage of time brings
    # expiry AND ex-dates closer, so both are shifted together.
    def shifted(h):
        divs = [(d + h, D) for d, D in dividends if d + h > 0]
        return _ql_option(S, K, days + h, r, q, sigma, option_type,
                          dividends=divs, div_model=div_model)[0].NPV()
    h_dn = 1 if days > 1 else 0
    res["theta"] = -(shifted(1) - shifted(-h_dn)) / ((1 + h_dn) / 365.0)
    # vega and rho by bump-and-reprice (not provided by the FD engine).
    # Steps of 0.01 (vol) and 0.001 (rate): a smaller step brings out the noise
    # of the finite-difference grid (errors > 1% at low volatility).
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
# European Black-Scholes (for the property tests)
# ---------------------------------------------------------------------------
def bs_european(S, K, T, r, q, sigma, option_type):
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)


# ---------------------------------------------------------------------------
# Utilities
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
            print(f"\n[{'OK' if not ko else 'FAIL'}] {s}: {len(items) - len(ko)}/{len(items)}")
            for _, nom, _, detail in ko[:8]:
                print(f"     ✗ {nom}  {detail}")
            if len(ko) > 8:
                print(f"     … and {len(ko) - 8} more (see the CSV)")
        print("\n" + ("=" * 60))
        print("OVERALL RESULT:", "ALL CONSISTENT" if total_ko == 0
              else f"{total_ko} check(s) failed")
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
        nom = f"{c['type']} K={c['K']:.0f} T={c['days']}d vol={c['sigma']} r={c['r']} q={c['q']}"
        bilan.check("1. Price vs QuantLib", nom, ok,
                    f"yours={p:.4f} QL={ref['prix']:.4f}")
        for gk in ["delta", "gamma", "vega", "theta", "rho"]:
            mine = g[gk] * ECHELLES[gk]
            ligne[gk], ligne[gk + "_ql"] = mine, ref[gk]
            okg = is_close(mine, ref[gk], TOL_GREC[gk], TOL_GREC_REL)
            bilan.check(f"1. {gk.capitalize()} vs QuantLib", nom, okg,
                        f"yours={mine:.5f} QL={ref[gk]:.5f}")
        lignes.append(ligne)
    return pd.DataFrame(lignes)


def test_properties(pm, bilan):
    S, T, sig = S0, 0.5, 0.25
    sec = "2. Theoretical properties"

    # American call without dividend = European call
    for K in [80, 100, 120]:
        a = pm.price(S, K, T, 0.05, 0.0, sig, "call")
        e = bs_european(S, K, T, 0.05, 0.0, sig, "call")
        bilan.check(sec, f"Call without dividend = European (K={K})",
                    is_close(a, e, TOL_PRIX_ABS, TOL_PRIX_REL), f"am={a:.4f} eu={e:.4f}")

    # American put >= European and >= intrinsic; <= K
    for K in [80, 100, 120, 150]:
        a = pm.price(S, K, T, 0.05, 0.0, sig, "put")
        e = bs_european(S, K, T, 0.05, 0.0, sig, "put")
        bilan.check(sec, f"American put >= European (K={K})", a >= e - 1e-6, f"am={a:.4f} eu={e:.4f}")
        bilan.check(sec, f"American put >= intrinsic (K={K})", a >= max(K - S, 0) - 1e-6)
        bilan.check(sec, f"American put <= K (K={K})", a <= K + 1e-9)

    # Deep in-the-money put with a high rate: immediate exercise
    a = pm.price(S, 200, T, 0.08, 0.0, sig, "put")
    bilan.check(sec, "Deep ITM put = intrinsic value", abs(a - 100) < 1e-3, f"price={a:.5f}")

    # American put-call parity: S e^{-qT} - K <= C - P <= S - K e^{-rT}
    for K, r, q in [(90, 0.05, 0.0), (100, 0.03, 0.02), (110, 0.05, 0.04)]:
        C = pm.price(S, K, T, r, q, sig, "call")
        P = pm.price(S, K, T, r, q, sig, "put")
        bas, haut = S * math.exp(-q * T) - K, S - K * math.exp(-r * T)
        bilan.check(sec, f"American parity (K={K}, r={r}, q={q})",
                    bas - 1e-3 <= C - P <= haut + 1e-3,
                    f"{bas:.3f} <= {C - P:.3f} <= {haut:.3f}")

    # Monotonicity in vol and maturity
    for t in ["call", "put"]:
        pv = [pm.price(S, 100, T, 0.03, 0.02, v, t) for v in [0.1, 0.2, 0.3, 0.5]]
        bilan.check(sec, f"{t} increasing in volatility", all(np.diff(pv) > 0))
        pt = [pm.price(S, 100, x, 0.03, 0.02, sig, t) for x in [0.1, 0.25, 0.5, 1, 2]]
        bilan.check(sec, f"{t} increasing in maturity", all(np.diff(pt) >= -1e-6))

    # Convexity in strike (positive butterfly) and monotonicity in strike
    Ks = np.arange(70, 131, 5.0)
    for t in ["call", "put"]:
        p = np.array([pm.price(S, k, T, 0.03, 0.02, sig, t) for k in Ks])
        fly = p[:-2] - 2 * p[1:-1] + p[2:]
        bilan.check(sec, f"{t} convex in strike", fly.min() > -1e-4, f"min butterfly={fly.min():.2e}")
        sens = np.diff(p) <= 1e-6 if t == "call" else np.diff(p) >= -1e-6
        bilan.check(sec, f"{t} monotonic in strike", all(sens))


def test_internal_greeks(pm, bilan):
    sec = "3. Greeks vs bump-and-reprice of your pricer"
    # 5% step on the spot: a smaller step makes the "bump" gamma unstable
    # on tree pricers (the price oscillates with S between the nodes)
    hS, hv, hr, hT = 0.05 * S0, 1e-2, 1e-3, 1 / 365
    tol_rel = {"gamma": 0.05}
    cas = [(100, 0.5, 0.25, 0.03, 0.0, "put"), (90, 1.0, 0.30, 0.05, 0.02, "put"),
           (110, 0.25, 0.20, 0.02, 0.03, "call"), (100, 1.0, 0.40, 0.04, 0.05, "call")]
    for K, T, sig, r, q, t in cas:
        P = lambda S=S0, T=T, sig=sig, r=r: pm.price(S, K, T, r, q, sig, t)
        hD = 0.01 * S0   # fine step for delta (first derivative)
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
                        f"analytical={mine:.5f} bump={v:.5f}")
        d = g["delta"] * ECHELLES["delta"]
        borne = (0 <= d <= 1) if t == "call" else (-1 <= d <= 0)
        bilan.check(sec, f"delta within its bounds {t} K={K}", borne, f"delta={d:.4f}")
        bilan.check(sec, f"gamma >= 0 {t} K={K}", g["gamma"] >= -1e-8)
        bilan.check(sec, f"vega >= 0 {t} K={K}", g["vega"] >= -1e-8)


def test_convergence(pm, bilan):
    """The price must stabilise as resolution increases (if your pricer accepts n=)."""
    sec = "4. Numerical convergence"
    try:
        ns = [100, 200, 400, 800, 1600]
        p = [pm.price(100, 100, 1.0, 0.05, 0.0, 0.3, "put", n=n) for n in ns]
    except TypeError:
        bilan.check(sec, "Skipped: price() does not accept an n= argument", True)
        return
    ecarts = np.abs(np.diff(p))
    ref = ql_reference(100, 100, 365, 0.05, 0.0, 0.3, "put")["prix"]
    print("\n   Convergence (1-year ATM put):")
    for n, x in zip(ns, p):
        print(f"     n={n:5d}  price={x:.5f}  QL diff={x - ref:+.5f}")
    bilan.check(sec, "Successive differences decreasing", ecarts[-1] < ecarts[0],
                f"{ecarts.round(5).tolist()}")
    bilan.check(sec, "Final error < tolerance", abs(p[-1] - ref) < TOL_PRIX_ABS,
                f"diff={p[-1] - ref:+.5f}")


def test_implied_vol(pm, bilan):
    sec = "5. Implied volatility round trip"
    for K, T, sig, t in [(80, 0.25, 0.35, "put"), (100, 0.5, 0.20, "put"),
                         (120, 1.0, 0.25, "call"), (100, 2.0, 0.45, "call")]:
        cible = pm.price(S0, K, T, 0.04, 0.02, sig, t)
        try:
            iv = brentq(lambda v: pm.price(S0, K, T, 0.04, 0.02, v, t) - cible,
                        0.01, 3.0, xtol=1e-8)
            bilan.check(sec, f"{t} K={K} T={T}", abs(iv - sig) < 1e-4,
                        f"initial vol={sig} recovered={iv:.6f}")
        except ValueError as e:
            bilan.check(sec, f"{t} K={K} T={T}", False, f"inversion failed: {e}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="mon_pricer", help="module containing price/greeks")
    ap.add_argument("--complet", action="store_true", help="extended grid (slower)")
    ap.add_argument("--csv", default="rapport_validation.csv")
    a = ap.parse_args()

    pm = importlib.import_module(a.module)
    bilan = Report()
    t0 = time.time()
    print(f"Validation of '{a.module}' (QuantLib {ql.__version__})")

    df = test_vs_quantlib(pm, bilan, a.complet)
    test_properties(pm, bilan)
    test_internal_greeks(pm, bilan)
    test_convergence(pm, bilan)
    test_implied_vol(pm, bilan)

    df.to_csv(a.csv, index=False)
    e = df["ecart_prix"].abs()
    print(f"\n   Grid: {len(df)} options | mean price diff {e.mean():.5f} "
          f"| max {e.max():.5f} | duration {time.time() - t0:.0f}s")
    pire = df.loc[e.idxmax()]
    print(f"   Worst case: {pire['type']} K={pire['K']:.0f} T={pire['days']}d "
          f"vol={pire['sigma']} r={pire['r']} q={pire['q']}")
    ko = bilan.afficher()
    print(f"Details per option: {a.csv}")
    sys.exit(1 if ko else 0)


if __name__ == "__main__":
    main()
