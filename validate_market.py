"""
validate_market.py — Test the American pricer (with discrete dividends) on a
real quoted option chain.

Two ways to use it:

  1) Automatic download (Yahoo Finance, delayed data):
        pip install yfinance matplotlib
        python validate_market.py --ticker AAPL --taux 0.04

  2) From CSV files (your own source: IBKR, CBOE, Bloomberg…):
        python validate_market.py --chaine chaine.csv --dividendes dividends.csv \
                                 --spot 227.5 --date 2026-10-06 --heure 16:00 --taux 0.04

     --heure : New York time at which the spot and quotes were
               taken (16:00 by default, i.e. closing data).

     chaine.csv     : expiration (YYYY-MM-DD), type (call/put), strike, bid, ask,
                      [volume], [open_interest], [iv_source]
     dividends.csv : ex_date (YYYY-MM-DD), montant      (optional file)

What the script does:
  1. Filters out unusable quotes (spread too wide, zero bid…).
  2. Fits, per expiry, an implied borrow cost such that call and put implied
     vols match near the money. It absorbs errors
     in rates and dividends, as well as the stock's real borrow cost.
  3. Inverts your pricer to get the bid, mid and ask implied vol of each option.
  4. Fits a smooth smile per expiry (on out-of-the-money options),
     reprices each option with this smile and measures the share repriced within the
     bid-ask spread: this is the main criterion of consistency with the market.
  5. Diagnostics: call/put vol gap at the same strike, comparison with the source's
     implied vol if provided.

Time measurement (New York time, ACT/365 to the nearest second):
  - valuation : time of the spot snapshot (Yahoo) or --date/--heure (CSV);
  - expiration: 16:00 on the expiry date (US equity options, PM settlement);
  - ex-date   : 09:30 on the ex-dividend date (the price drops at the open).
  A T rounded to the day does not change the repricing rate (implied vol and
  repricing use the same T), but it biases vol LEVELS: about
  -0.7 pt at 17 days for a one-day gap, more at short expiries.
  Out of scope: trading time (weekends, public holidays), index options
  settled in the morning (AM settlement).

Outputs: resultats_marche_<nom>.csv and smiles_<nom>.png
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import date, datetime, time as heure_du_jour, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from dividends import american_price_div
from pricer import sigma_min_crr
from svi import calendar_arbitrage, butterfly_arbitrage, calibrate_svi

# Quote quality filters
SPREAD_MAX = 0.25        # maximum (ask - bid) / mid
MID_MIN = 0.05           # minimum mid price
MONEYNESS = (0.70, 1.30) # bounds on K / forward
JOURS_MIN, JOURS_MAX = 7, 450
N_PAS = 400              # tree steps for the inversion (accuracy ~0.01 vol pt)
VOL_MIN, VOL_MAX = 0.005, 4.0

# Time measurement
FUSEAU = "America/New_York"
HEURE_EXPIRATION = heure_du_jour(16, 0)    # end of session, US equity options
HEURE_DETACHEMENT = heure_du_jour(9, 30)   # the price drops at the open on the ex-dividend date
SECONDES_PAR_AN = 365.0 * 86400.0          # ACT/365, like the script's rates and vols


# --------------------------------------------------------------------------- #
# Time
# --------------------------------------------------------------------------- #
def market_tz() -> ZoneInfo:
    try:
        return ZoneInfo(FUSEAU)
    except Exception as e:                                  # Windows without an IANA database
        raise RuntimeError("Time zone not found: install it with "
                           "'python -m pip install tzdata'") from e


def market_time(jour, heure: heure_du_jour) -> datetime:
    """Date (date, or YYYY-MM-DD string optionally followed by a time) at the given time, in New York."""
    if not isinstance(jour, date):
        jour = date.fromisoformat(str(jour)[:10])
    elif isinstance(jour, datetime):
        jour = jour.date()
    return datetime.combine(jour, heure, tzinfo=market_tz())


def year_fraction(debut: datetime, fin: datetime) -> float:
    """Duration in ACT/365 years, to the nearest second (timezone-aware datetimes)."""
    # Converting to UTC is essential: Python subtracts two datetimes in the SAME zone
    # in wall-clock time, and would ignore the hour gained or lost at a clock change.
    utc = ZoneInfo("UTC")
    return (fin.astimezone(utc) - debut.astimezone(utc)).total_seconds() / SECONDES_PAR_AN


def maturity(expiration, valo: datetime) -> float:
    """Time to expiration (16:00 New York on the expiry date), in years."""
    return year_fraction(valo, market_time(expiration, HEURE_EXPIRATION))


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def download_yahoo(ticker: str, nb_echeances: int):
    import yfinance as yf
    tk = yf.Ticker(ticker)
    hist = tk.history(period="1d", interval="1m")
    if len(hist):
        spot = float(hist["Close"].iloc[-1])
        # A one-minute bar is timestamped at its start: the bar's closing price
        # corresponds to the following minute.
        valo = hist.index[-1].to_pydatetime().astimezone(market_tz()) + timedelta(minutes=1)
    else:
        spot = float(tk.fast_info["last_price"])
        valo = datetime.now(market_tz())
    today = valo.date()
    lignes = []
    exps = [e for e in tk.options
            if JOURS_MIN <= maturity(e, valo) * 365 <= JOURS_MAX]
    # expiries spread across the curve rather than the first N
    if len(exps) > nb_echeances:
        idx = np.unique(np.linspace(0, len(exps) - 1, nb_echeances).round().astype(int))
        exps = [exps[i] for i in idx]
    for e in exps:
        oc = tk.option_chain(e)
        for kind, df in (("call", oc.calls), ("put", oc.puts)):
            for _, row in df.iterrows():
                lignes.append(dict(expiration=e, type=kind, strike=row["strike"],
                                   bid=row["bid"], ask=row["ask"], volume=row.get("volume"),
                                   open_interest=row.get("openInterest"),
                                   iv_source=row.get("impliedVolatility")))
    chaine = pd.DataFrame(lignes)

    # Dividends: projects future ex-dates from the history
    hist_div = tk.dividends
    divs = pd.DataFrame(columns=["ex_date", "montant"])
    if len(hist_div) >= 2:
        hd = hist_div[hist_div.index >= hist_div.index[-1] - pd.Timedelta(days=800)]
        ecarts = np.diff(hd.index.values).astype("timedelta64[D]").astype(int)
        periode = int(np.median(ecarts)) if len(ecarts) else 91
        montant = float(hd.iloc[-1])
        d = hd.index[-1].date()
        futurs = []
        while True:
            d = d + pd.Timedelta(days=periode).to_pytimedelta()
            if (d - today).days > JOURS_MAX + 30:
                break
            if d > today:
                futurs.append(dict(ex_date=d.isoformat(), montant=montant))
        divs = pd.DataFrame(futurs)
        print(f"   Projected dividends: {montant:.4f} every ~{periode} days "
              f"({len(futurs)} ex-dates). Check them: this is an extrapolation.")
    return chaine, divs, spot, valo


def load_csv(chemin_chaine, chemin_div):
    chaine = pd.read_csv(chemin_chaine)
    chaine["type"] = chaine["type"].str.lower().str.strip()
    divs = pd.read_csv(chemin_div) if chemin_div else pd.DataFrame(columns=["ex_date", "montant"])
    return chaine, divs


# --------------------------------------------------------------------------- #
# Calculations
# --------------------------------------------------------------------------- #
def divs_for(divs: pd.DataFrame, valo: datetime, T: float):
    """
    Dividends (t in years, amount) still to go ex before expiry.
    The ex-date falls at the open of the ex-dividend day: a dividend whose ex-date
    is today is no longer to come when valuing after 09:30.
    """
    out = []
    for _, d in divs.iterrows():
        t = year_fraction(valo, market_time(d["ex_date"], HEURE_DETACHEMENT))
        if 0 < t <= T:
            out.append((t, float(d["montant"])))
    return out


def forward(S, T, r, q, dv):
    return S * math.exp((r - q) * T) - sum(D * math.exp((r - q) * (T - t)) for t, D in dv)


def implied_vol(prix, S, K, T, r, q, kind, dv):
    """None if the price is outside the model bounds (arbitrage or bad data)."""
    f = lambda v: american_price_div(S, K, T, r, q, v, kind, dv, n=N_PAS) - prix
    # Lower bound: below |r-q|·√dt, the tree probability leaves (0, 1)
    v_min = max(VOL_MIN, 1.001 * sigma_min_crr(r, q, T, N_PAS))
    if v_min >= VOL_MAX:
        return None
    lo, hi = f(v_min), f(VOL_MAX)
    if lo > 0 or hi < 0:
        return None
    return brentq(f, v_min, VOL_MAX, xtol=1e-5)


def calibrate_borrow(df_e, S, T, r, dv):
    """Borrow cost q such that vol(call) = vol(put) on near-the-money strikes."""
    F0 = forward(S, T, r, 0.0, dv)
    paires = df_e.pivot_table(index="strike", columns="type", values="mid").dropna()
    if paires.empty:
        return 0.0
    ks = paires.index.values[np.argsort(np.abs(paires.index.values - F0))][:3]
    estimations = []
    for K in ks:
        def ecart(q):
            vc = implied_vol(paires.loc[K, "call"], S, K, T, r, q, "call", dv)
            vp = implied_vol(paires.loc[K, "put"], S, K, T, r, q, "put", dv)
            if vc is None or vp is None:
                return np.nan
            return vc - vp
        # Sweep, then brentq on the first interval where the gap changes sign
        # (at the extreme bounds, one of the two vols may no longer be invertible)
        # At short expiries, a small forward shift (a price off by a few
        # cents) translates into a high annualised borrow: wide grid.
        param_grid = [0.0, -0.01, 0.01, -0.03, 0.03, -0.06, 0.06, -0.10, 0.10,
                  -0.15, 0.15, -0.25, 0.25, -0.40, 0.40]
        param_grid = sorted(param_grid)
        vals = [ecart(x) for x in param_grid]
        for (x0, f0), (x1, f1) in zip(zip(param_grid, vals), zip(param_grid[1:], vals[1:])):
            if np.isfinite(f0) and np.isfinite(f1) and f0 * f1 <= 0:
                try:
                    estimations.append(brentq(ecart, x0, x1, xtol=1e-4))
                except (ValueError, RuntimeError):
                    pass
                break
    return float(np.median(estimations)) if estimations else 0.0


def fit_smile(x, iv, poids):
    """Smooth smile: polynomial in log-moneyness (degree 2, or 3 with enough points)."""
    deg = 3 if len(x) >= 10 else 2 if len(x) >= 4 else 1
    return np.polynomial.polynomial.Polynomial.fit(x, iv, deg, w=poids)


# --------------------------------------------------------------------------- #
# Plot: smiles per expiry (small multiples)
# --------------------------------------------------------------------------- #
def plot_results(res: pd.DataFrame, nom: str, libelle_smile: str = "Fitted smile"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    CALL, PUT, BAND = "#2a78d6", "#eb6834", "#c3c2b7"
    exps = sorted(res["expiration"].unique())
    nc = min(3, len(exps)); nr = math.ceil(len(exps) / nc)
    fig, axes = plt.subplots(nr, nc, figsize=(4.6 * nc, 3.4 * nr), squeeze=False,
                             facecolor=SURF)
    for ax in axes.flat[len(exps):]:
        ax.set_visible(False)
    for ax, e in zip(axes.flat, exps):
        d_all = res[(res.expiration == e) & res.iv_mid.notna()].sort_values("strike")
        # Only out-of-the-money options are plotted (market convention): the implied vol
        # of a deep in-the-money option is poorly determined (price ≈ intrinsic)
        d = d_all[((d_all.type == "put") & (d_all.strike <= d_all.forward))
                  | ((d_all.type == "call") & (d_all.strike > d_all.forward))]
        ax.set_facecolor(SURF)
        for kind, col, lab in (("put", PUT, "Puts (mid)"), ("call", CALL, "Calls (mid)")):
            dk = d[d.type == kind]
            ax.vlines(dk.strike, 100 * dk.iv_bid, 100 * dk.iv_ask, color=BAND, lw=2,
                      zorder=1, label="Fourchette bid-ask" if kind == "put" else None)
            ax.scatter(dk.strike, 100 * dk.iv_mid, s=14, color=col, zorder=3,
                       edgecolors=SURF, linewidths=0.8, label=lab)
        fit = d.dropna(subset=["iv_smile"]).sort_values("strike")
        ax.plot(fit.strike, 100 * fit.iv_smile, color=INK, lw=1.6, zorder=2, label=libelle_smile)
        jours = float(d_all.jours.iloc[0]) if len(d_all) else 0.0
        dans = d_all.dans_fourchette.mean() if d_all.dans_fourchette.notna().any() else float("nan")
        lib_j = f"{jours:.1f} d" if jours < 30 else f"{jours:.0f} d"
        ax.set_title(f"{e}  ({lib_j})  ·  {dans:.0%} within spread",
                     fontsize=9.5, color=INK, loc="left")
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8, length=0)
        ax.set_xlabel("Strike", fontsize=8, color=INK2)
        ax.set_ylabel("Implied vol (%)", fontsize=8, color=INK2)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=9,
               labelcolor=INK, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"{nom}: market implied vols and fitted smile", y=1.05,
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    chemin = f"smiles_{nom}.png"
    fig.savefig(chemin, dpi=150, bbox_inches="tight", facecolor=SURF)
    return chemin


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticker")
    ap.add_argument("--chaine"); ap.add_argument("--dividendes")
    ap.add_argument("--spot", type=float); ap.add_argument("--date")
    ap.add_argument("--heure", default="16:00",
                    help="New York time of the snapshot (HH:MM, default 16:00 = close)")
    ap.add_argument("--taux", type=float, default=None,
                    help="continuous risk-free rate (e.g. 0.04); use the day's SOFR/OIS")
    ap.add_argument("--nb-echeances", type=int, default=6)
    ap.add_argument("--sans-borrow", action="store_true", help="do not fit a borrow cost")
    ap.add_argument("--nom", default=None)
    ap.add_argument("--smile", choices=["svi", "poly"], default="svi",
                    help="smile parametrisation: SVI (default) or polynomial")
    a = ap.parse_args()

    if a.taux is None:
        print("WARNING: --taux not provided, 4% used by default. Provide the day's rate.")
        a.taux = 0.04
    t0 = time.time()
    if a.ticker:
        print(f"Downloading the {a.ticker} chain (Yahoo Finance, delayed data)…")
        chaine, divs, spot, valo = download_yahoo(a.ticker, a.nb_echeances)
        today = valo.date()
        nom = a.nom or a.ticker
        chaine.to_csv(f"chaine_{nom}_{today}.csv", index=False)
        divs.to_csv(f"dividendes_{nom}_{today}.csv", index=False)
    elif a.chaine and a.spot:
        chaine, divs = load_csv(a.chaine, a.dividendes)
        spot = a.spot
        if a.date:
            valo = market_time(a.date, heure_du_jour.fromisoformat(a.heure))
        else:
            valo = datetime.now(market_tz())
        today = valo.date()
        nom = a.nom or "chaine"
    else:
        ap.error("provide --ticker, or --chaine and --spot")

    r = a.taux
    chaine["T"] = chaine["expiration"].map(lambda e: maturity(e, valo))
    chaine["jours"] = 365.0 * chaine["T"]          # calendar days, fractional
    chaine["mid"] = (chaine["bid"] + chaine["ask"]) / 2
    n0 = len(chaine)
    ok = ((chaine.bid > 0) & (chaine.ask > chaine.bid) & (chaine.mid >= MID_MIN)
          & ((chaine.ask - chaine.bid) / chaine.mid <= SPREAD_MAX)
          & chaine.jours.between(JOURS_MIN, JOURS_MAX))
    chaine = chaine[ok].copy()
    print(f"Spot {spot:.2f} | rate {r:.2%} | valuation {valo:%Y-%m-%d %H:%M} (New York) "
          f"| {n0} quotes, {len(chaine)} kept after filters")

    resultats, resume, svis = [], [], []
    for e, df_e in chaine.groupby("expiration"):
        T = float(df_e["T"].iloc[0])
        dv = divs_for(divs, valo, T)
        q = 0.0 if a.sans_borrow else calibrate_borrow(df_e, spot, T, r, dv)
        F = forward(spot, T, r, q, dv)
        df_e = df_e[(df_e.strike / F).between(*MONEYNESS)].copy()
        for col, px in (("iv_bid", "bid"), ("iv_mid", "mid"), ("iv_ask", "ask")):
            df_e[col] = [implied_vol(p, spot, K, T, r, q, k, dv)
                         for p, K, k in zip(df_e[px], df_e.strike, df_e.type)]
        df_e[["iv_bid", "iv_mid", "iv_ask"]] = df_e[["iv_bid", "iv_mid", "iv_ask"]].astype(float)
        df_e["x"] = np.log(df_e.strike / F)
        # Out-of-the-money options: the most liquid and least sensitive to early exercise
        otm = df_e[((df_e.type == "put") & (df_e.strike <= F)) |
                   ((df_e.type == "call") & (df_e.strike > F))].dropna(subset=["iv_mid"])
        df_e["iv_smile"] = np.nan
        df_e["dans_fourchette"] = np.nan
        if len(otm) >= 3:
            largeur = (otm.iv_ask.fillna(otm.iv_mid + 0.02) - otm.iv_bid.fillna(otm.iv_mid - 0.02)).clip(lower=1e-3)
            if a.smile == "svi":
                smile = calibrate_svi(otm.x.values, otm.iv_mid.values, T, 1 / largeur.values)
                svis.append((e, smile, float(otm.x.min()), float(otm.x.max())))
            else:
                smile = fit_smile(otm.x.values, otm.iv_mid.values, 1 / largeur.values)
            lo, hi = otm.x.min(), otm.x.max()
            dedans = df_e.x.between(lo, hi)
            df_e.loc[dedans, "iv_smile"] = smile(df_e.loc[dedans, "x"].values)
            prix_smile = [american_price_div(spot, K, T, r, q, v, k, dv, n=N_PAS) if np.isfinite(v) else np.nan
                          for K, v, k in zip(df_e.strike, df_e.iv_smile, df_e.type)]
            df_e["prix_smile"] = prix_smile
            valide = df_e.prix_smile.notna()
            df_e.loc[valide, "dans_fourchette"] = ((df_e.prix_smile >= df_e.bid - 1e-9)
                                                   & (df_e.prix_smile <= df_e.ask + 1e-9))[valide].astype(float)
        # Call / put vol gap at the same strike (input diagnostic)
        p = df_e.pivot_table(index="strike", columns="type", values="iv_mid").dropna()
        ecart_cp = (p["call"] - p["put"]).abs().median() if {"call", "put"} <= set(p.columns) else np.nan
        hors_bornes = df_e.iv_mid.isna().sum()
        df_e["borrow"] = q
        df_e["forward"] = F
        df_e["valorisation"] = valo.isoformat(timespec="minutes")
        resultats.append(df_e)
        resume.append(dict(echeance=e, jours=float(df_e.jours.iloc[0]), options=len(df_e),
                           nb_div=len(dv), borrow=q, forward=F,
                           dans_fourchette=df_e.dans_fourchette.mean(),
                           erreur_vol_pts=100 * (df_e.iv_smile - df_e.iv_mid).abs().median(),
                           ecart_call_put_pts=100 * ecart_cp, hors_bornes=int(hors_bornes)))
        print(f"   {e} ({resume[-1]['jours']:5.1f} d): {len(df_e):3d} options | borrow {q:+.2%} "
              f"| within spread {resume[-1]['dans_fourchette']:.0%} "
              f"| call/put vol gap {resume[-1]['ecart_call_put_pts']:.2f} pt "
              f"| out of bounds {hors_bornes}"
              + (f" | butterfly {'OK' if butterfly_arbitrage(svis[-1][1])[0] else 'ARBITRAGE'}"
                 if a.smile == "svi" and svis and svis[-1][0] == e else ""))

    res = pd.concat(resultats)
    res.to_csv(f"resultats_marche_{nom}.csv", index=False)
    rs = pd.DataFrame(resume)
    tot = res.dans_fourchette.mean()
    print("\n" + "=" * 70)
    print(f"Options repriced within the bid-ask spread: {tot:.1%}  (out of {res.dans_fourchette.notna().sum()})")
    print(f"Median smile - market gap: {100 * (res.iv_smile - res.iv_mid).abs().median():.2f} vol pt")
    print(f"Median call/put vol gap at the same strike: {rs.ecart_call_put_pts.median():.2f} vol pt")
    if "iv_source" in res and res.iv_source.notna().any():
        diff = 100 * (res.iv_mid - res.iv_source).abs().median()
        print(f"Median gap to the source's implied vol: {diff:.2f} pt "
              "(indicative: the source uses its own rates and dividends)")
    if a.smile == "svi" and svis:
        params = pd.DataFrame([dict(echeance=e, T=sv.T, a=sv.a, b=sv.b, rho=sv.rho, m=sv.m, s=sv.s,
                                    vol_atm=float(sv.vol(0.0)),
                                    papillon_ok=butterfly_arbitrage(sv)[0],
                                    g_min=butterfly_arbitrage(sv)[1]) for e, sv, _, _ in svis])
        params.to_csv(f"svi_params_{nom}.csv", index=False)
        liste = [sv for _, sv, _, _ in svis]
        cal_cote = calendar_arbitrage(liste, plages=[(lo, hi) for _, _, lo, hi in svis])
        cal_ailes = calendar_arbitrage(liste, -0.5, 0.5)
        fmt = lambda c: "none" if not c else "; ".join(
            f"{t1 * 365:.0f}d→{t2 * 365:.0f}d at k={k:+.2f}" for t1, t2, k, _ in c)
        print(f"Butterfly arbitrage (density < 0, k ∈ [-1.5, 1.5]): "
              f"{(~params.papillon_ok).sum()} expiries out of {len(params)}")
        print(f"Calendar arbitrage, quoted zone: {fmt(cal_cote)}")
        print(f"Calendar arbitrage, extrapolated wings (|k| ≤ 0.5): {fmt(cal_ailes)}")
        print(f"SVI parameters: svi_params_{nom}.csv")
    print("=" * 70)
    print("How to read this:")
    print("  > 90% within spread: pricer and inputs consistent with the market.")
    print("  High call/put gap (> 1 pt) despite the borrow: review dividends or rates.")
    print("  'Out of bounds' options: price below the model's minimum value (stale data,")
    print("  or poorly captured early exercise premium).")
    png = plot_results(res, nom, "Smile SVI" if a.smile == "svi" else "Polynomial smile")
    print(f"\nFiles: resultats_marche_{nom}.csv, {png}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
