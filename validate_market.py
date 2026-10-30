"""
validate_market.py — Tester le pricer américain (avec dividendes discrets) sur une
vraie chaîne d'options cotées.

Deux façons de l'utiliser :

  1) Téléchargement automatique (Yahoo Finance, données différées) :
        pip install yfinance matplotlib
        python validate_market.py --ticker AAPL --taux 0.04

  2) À partir de fichiers CSV (votre propre source : IBKR, CBOE, Bloomberg…) :
        python validate_market.py --chaine chaine.csv --dividendes dividends.csv \
                                 --spot 227.5 --date 2026-10-06 --heure 16:00 --taux 0.04

     --heure : heure de New York à laquelle le spot et les cotations ont été
               relevés (16:00 par défaut, soit des données de clôture).

     chaine.csv     : expiration (AAAA-MM-JJ), type (call/put), strike, bid, ask,
                      [volume], [open_interest], [iv_source]
     dividends.csv : ex_date (AAAA-MM-JJ), montant      (fichier facultatif)

Ce que fait le script :
  1. Filtre les cotations inexploitables (fourchette trop large, bid nul…).
  2. Cale, par échéance, un coût d'emprunt implicite (borrow) tel que les vols
     implicites call et put coïncident près de la monnaie. Il absorbe les erreurs
     de taux, de dividendes et le coût réel d'emprunt du titre.
  3. Inverse votre pricer pour obtenir la vol implicite bid, mid et ask de chaque option.
  4. Ajuste un smile lisse par échéance (sur les options hors de la monnaie),
     reprice chaque option avec ce smile et mesure la part repricée dans la
     fourchette bid-ask : c'est le critère principal de cohérence avec le marché.
  5. Diagnostics : écart de vol call/put au même strike, comparaison avec la vol
     implicite de la source si fournie.

Mesure du temps (heure de New York, ACT/365 à la seconde près) :
  - valorisation : instant du relevé du spot (Yahoo) ou --date/--heure (CSV) ;
  - expiration   : 16:00 le jour d'échéance (options sur actions US, règlement PM) ;
  - détachement  : 09:30 le jour ex-dividende (le cours chute à l'ouverture).
  Un T arrondi au jour ne change pas le taux de repricing (vol implicite et
  repricing utilisent le même T), mais biaise les NIVEAUX de vol : environ
  -0,7 pt à 17 jours pour un jour d'écart, davantage sur les échéances courtes.
  Hors du périmètre : temps de bourse (week-ends, jours fériés), options sur
  indices réglées le matin (AM settlement).

Sorties : resultats_marche_<nom>.csv et smiles_<nom>.png
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

# Filtres de qualité des cotations
SPREAD_MAX = 0.25        # (ask - bid) / mid maximal
MID_MIN = 0.05           # prix mid minimal
MONEYNESS = (0.70, 1.30) # bornes de K / forward
JOURS_MIN, JOURS_MAX = 7, 450
N_PAS = 400              # pas de l'arbre pour l'inversion (précision ~0,01 vol pt)
VOL_MIN, VOL_MAX = 0.005, 4.0

# Mesure du temps
FUSEAU = "America/New_York"
HEURE_EXPIRATION = heure_du_jour(16, 0)    # fin de séance, options sur actions US
HEURE_DETACHEMENT = heure_du_jour(9, 30)   # le cours chute à l'ouverture du jour ex-dividende
SECONDES_PAR_AN = 365.0 * 86400.0          # ACT/365, comme les taux et les vols du script


# --------------------------------------------------------------------------- #
# Temps
# --------------------------------------------------------------------------- #
def market_tz() -> ZoneInfo:
    try:
        return ZoneInfo(FUSEAU)
    except Exception as e:                                  # Windows sans base IANA
        raise RuntimeError("Fuseau horaire introuvable : installez-le avec "
                           "'python -m pip install tzdata'") from e


def market_time(jour, heure: heure_du_jour) -> datetime:
    """Date (date, ou chaîne AAAA-MM-JJ éventuellement suivie d'une heure) à l'heure donnée, à New York."""
    if not isinstance(jour, date):
        jour = date.fromisoformat(str(jour)[:10])
    elif isinstance(jour, datetime):
        jour = jour.date()
    return datetime.combine(jour, heure, tzinfo=market_tz())


def year_fraction(debut: datetime, fin: datetime) -> float:
    """Durée en années ACT/365, à la seconde près (datetimes avec fuseau)."""
    # Passage en UTC indispensable : Python soustrait deux datetimes de MÊME fuseau
    # en heure murale, et ignorerait l'heure gagnée ou perdue au changement d'heure.
    utc = ZoneInfo("UTC")
    return (fin.astimezone(utc) - debut.astimezone(utc)).total_seconds() / SECONDES_PAR_AN


def maturity(expiration, valo: datetime) -> float:
    """Temps jusqu'à l'expiration (16:00 New York le jour d'échéance), en années."""
    return year_fraction(valo, market_time(expiration, HEURE_EXPIRATION))


# --------------------------------------------------------------------------- #
# Chargement des données
# --------------------------------------------------------------------------- #
def download_yahoo(ticker: str, nb_echeances: int):
    import yfinance as yf
    tk = yf.Ticker(ticker)
    hist = tk.history(period="1d", interval="1m")
    if len(hist):
        spot = float(hist["Close"].iloc[-1])
        # Une barre d'une minute est horodatée à son début : le cours de clôture
        # de la barre correspond à la minute suivante.
        valo = hist.index[-1].to_pydatetime().astimezone(market_tz()) + timedelta(minutes=1)
    else:
        spot = float(tk.fast_info["last_price"])
        valo = datetime.now(market_tz())
    today = valo.date()
    lignes = []
    exps = [e for e in tk.options
            if JOURS_MIN <= maturity(e, valo) * 365 <= JOURS_MAX]
    # échéances réparties sur la courbe plutôt que les N premières
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

    # Dividendes : projection des futurs détachements à partir de l'historique
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
        print(f"   Dividendes projetés : {montant:.4f} tous les ~{periode} jours "
              f"({len(futurs)} détachements). Vérifiez-les : c'est une extrapolation.")
    return chaine, divs, spot, valo


def load_csv(chemin_chaine, chemin_div):
    chaine = pd.read_csv(chemin_chaine)
    chaine["type"] = chaine["type"].str.lower().str.strip()
    divs = pd.read_csv(chemin_div) if chemin_div else pd.DataFrame(columns=["ex_date", "montant"])
    return chaine, divs


# --------------------------------------------------------------------------- #
# Calculs
# --------------------------------------------------------------------------- #
def divs_for(divs: pd.DataFrame, valo: datetime, T: float):
    """
    Dividendes (t en années, montant) encore à détacher avant l'échéance.
    Détachement à l'ouverture du jour ex-dividende : un dividende dont l'ex-date
    est aujourd'hui n'est plus à venir si l'on valorise après 09:30.
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
    """None si le prix est hors des bornes du modèle (arbitrage ou donnée sale)."""
    f = lambda v: american_price_div(S, K, T, r, q, v, kind, dv, n=N_PAS) - prix
    # Borne basse : au-dessous de |r-q|·√dt, la probabilité de l'arbre sort de ]0, 1[
    v_min = max(VOL_MIN, 1.001 * sigma_min_crr(r, q, T, N_PAS))
    if v_min >= VOL_MAX:
        return None
    lo, hi = f(v_min), f(VOL_MAX)
    if lo > 0 or hi < 0:
        return None
    return brentq(f, v_min, VOL_MAX, xtol=1e-5)


def calibrate_borrow(df_e, S, T, r, dv):
    """Coût d'emprunt q tel que vol(call) = vol(put) sur les strikes proches de la monnaie."""
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
        # Balayage puis brentq sur le premier intervalle où l'écart change de signe
        # (aux bornes extrêmes, l'une des deux vols peut ne plus être inversible)
        # Aux échéances courtes, un petit décalage de forward (cours décalé de
        # quelques centimes) se traduit par un borrow annualisé élevé : grille large.
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
    """Smile lisse : polynôme en log-moneyness (degré 2, ou 3 si assez de points)."""
    deg = 3 if len(x) >= 10 else 2 if len(x) >= 4 else 1
    return np.polynomial.polynomial.Polynomial.fit(x, iv, deg, w=poids)


# --------------------------------------------------------------------------- #
# Graphique : smiles par échéance (petits multiples)
# --------------------------------------------------------------------------- #
def plot_results(res: pd.DataFrame, nom: str, libelle_smile: str = "Smile ajusté"):
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
        # On ne trace que les options hors de la monnaie (convention de marché) : la vol
        # implicite d'une option très dans la monnaie est mal déterminée (prix ≈ intrinsèque)
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
        lib_j = f"{jours:.1f} j" if jours < 30 else f"{jours:.0f} j"
        ax.set_title(f"{e}  ({lib_j})  ·  {dans:.0%} dans la fourchette",
                     fontsize=9.5, color=INK, loc="left")
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8, length=0)
        ax.set_xlabel("Strike", fontsize=8, color=INK2)
        ax.set_ylabel("Vol implicite (%)", fontsize=8, color=INK2)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=9,
               labelcolor=INK, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"{nom} : vols implicites du marché et smile ajusté", y=1.05,
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
                    help="heure de New York du relevé (HH:MM, défaut 16:00 = clôture)")
    ap.add_argument("--taux", type=float, default=None,
                    help="taux sans risque continu (ex. 0.04) ; prenez le SOFR/OIS du jour")
    ap.add_argument("--nb-echeances", type=int, default=6)
    ap.add_argument("--sans-borrow", action="store_true", help="ne pas caler de coût d'emprunt")
    ap.add_argument("--nom", default=None)
    ap.add_argument("--smile", choices=["svi", "poly"], default="svi",
                    help="paramétrisation du smile : SVI (défaut) ou polynôme")
    a = ap.parse_args()

    if a.taux is None:
        print("ATTENTION : --taux non fourni, 4 % utilisé par défaut. Renseignez le taux du jour.")
        a.taux = 0.04
    t0 = time.time()
    if a.ticker:
        print(f"Téléchargement de la chaîne {a.ticker} (Yahoo Finance, données différées)…")
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
        ap.error("donnez --ticker, ou --chaine et --spot")

    r = a.taux
    chaine["T"] = chaine["expiration"].map(lambda e: maturity(e, valo))
    chaine["jours"] = 365.0 * chaine["T"]          # jours calendaires, fractionnaires
    chaine["mid"] = (chaine["bid"] + chaine["ask"]) / 2
    n0 = len(chaine)
    ok = ((chaine.bid > 0) & (chaine.ask > chaine.bid) & (chaine.mid >= MID_MIN)
          & ((chaine.ask - chaine.bid) / chaine.mid <= SPREAD_MAX)
          & chaine.jours.between(JOURS_MIN, JOURS_MAX))
    chaine = chaine[ok].copy()
    print(f"Spot {spot:.2f} | taux {r:.2%} | valorisation {valo:%Y-%m-%d %H:%M} (New York) "
          f"| {n0} cotations, {len(chaine)} retenues après filtres")

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
        # Options hors de la monnaie : les plus liquides et les moins sensibles à l'exercice anticipé
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
        # Écart de vol call / put au même strike (diagnostic des entrées)
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
        print(f"   {e} ({resume[-1]['jours']:5.1f} j) : {len(df_e):3d} options | borrow {q:+.2%} "
              f"| dans la fourchette {resume[-1]['dans_fourchette']:.0%} "
              f"| écart vol call/put {resume[-1]['ecart_call_put_pts']:.2f} pt "
              f"| hors bornes {hors_bornes}"
              + (f" | papillon {'OK' if butterfly_arbitrage(svis[-1][1])[0] else 'ARBITRAGE'}"
                 if a.smile == "svi" and svis and svis[-1][0] == e else ""))

    res = pd.concat(resultats)
    res.to_csv(f"resultats_marche_{nom}.csv", index=False)
    rs = pd.DataFrame(resume)
    tot = res.dans_fourchette.mean()
    print("\n" + "=" * 70)
    print(f"Options repricées dans la fourchette bid-ask : {tot:.1%}  (sur {res.dans_fourchette.notna().sum()})")
    print(f"Écart médian smile - marché : {100 * (res.iv_smile - res.iv_mid).abs().median():.2f} pt de vol")
    print(f"Écart médian vol call/put au même strike : {rs.ecart_call_put_pts.median():.2f} pt de vol")
    if "iv_source" in res and res.iv_source.notna().any():
        diff = 100 * (res.iv_mid - res.iv_source).abs().median()
        print(f"Écart médian avec la vol implicite de la source : {diff:.2f} pt "
              "(indicatif : la source utilise ses propres taux et dividendes)")
    if a.smile == "svi" and svis:
        params = pd.DataFrame([dict(echeance=e, T=sv.T, a=sv.a, b=sv.b, rho=sv.rho, m=sv.m, s=sv.s,
                                    vol_atm=float(sv.vol(0.0)),
                                    papillon_ok=butterfly_arbitrage(sv)[0],
                                    g_min=butterfly_arbitrage(sv)[1]) for e, sv, _, _ in svis])
        params.to_csv(f"svi_params_{nom}.csv", index=False)
        liste = [sv for _, sv, _, _ in svis]
        cal_cote = calendar_arbitrage(liste, plages=[(lo, hi) for _, _, lo, hi in svis])
        cal_ailes = calendar_arbitrage(liste, -0.5, 0.5)
        fmt = lambda c: "aucun" if not c else "; ".join(
            f"{t1 * 365:.0f}j→{t2 * 365:.0f}j à k={k:+.2f}" for t1, t2, k, _ in c)
        print(f"Arbitrage papillon (densité < 0, k ∈ [-1,5 ; 1,5]) : "
              f"{(~params.papillon_ok).sum()} échéance(s) sur {len(params)}")
        print(f"Arbitrage calendaire, zone cotée : {fmt(cal_cote)}")
        print(f"Arbitrage calendaire, ailes extrapolées (|k| ≤ 0,5) : {fmt(cal_ailes)}")
        print(f"Paramètres SVI : svi_params_{nom}.csv")
    print("=" * 70)
    print("Lecture :")
    print("  > 90 % dans la fourchette : pricer et entrées cohérents avec le marché.")
    print("  Écart call/put élevé (> 1 pt) malgré le borrow : dividendes ou taux à revoir.")
    print("  Options 'hors bornes' : prix sous la valeur minimale du modèle (donnée périmée,")
    print("  ou prime d'exercice anticipé mal captée).")
    png = plot_results(res, nom, "Smile SVI" if a.smile == "svi" else "Smile polynomial")
    print(f"\nFichiers : resultats_marche_{nom}.csv, {png}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
