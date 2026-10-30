"""
Validation du backtest de couverture (hedging.simulate_hedging), d'après
Bouchard & Chassagneux, ch. 7.3 et 5.2.1.

  1. Modèle juste : P&L moyen nul. Sous la mesure risque-neutre, c'est exact même
     en temps discret (portefeuille autofinancé actualisé = martingale).
  2. Rebalancement discret : écart-type en 1/√N (pente log-log ≈ −0,5) et égal
     au premier ordre théorique ½ σ⁴ Δt² Σ E[Γ² S⁴].
  3. Vol mal spécifiée : P&L ≈ ½ ∫ e^{-rt}(σ̃² − σ²) S² Γ dt trajectoire par
     trajectoire, et P&L moyen = BS(σ̃) − BS(σ) exactement.
  4. Couverture robuste (Black-Scholes-Barenblatt) : vol réalisée inconnue dans
     [σ_min, σ_max], payoff convexe -> couvrir à σ_max ne perd jamais (à l'erreur
     de discrétisation près), couvrir à σ_min ne gagne jamais.
  5. Gamma-neutre : avec une seconde option, le P&L devient insensible à la vol
     réalisée et l'erreur de discrétisation s'effondre.
  6. La dérive réelle μ ne change pas la dispersion du P&L.

Usage : python validate_hedging.py        (≈ 1 à 2 min)
Sorties : console + hedging.png
"""
import time

import numpy as np

from hedging import HedgedOption, theoretical_std, simulate_hedging
from pricer import BlackScholes, bs_price
from validate_pricer import Report

S0, R = 100.0, 0.03
NP = 50_000


def main():
    bilan, t0 = Report(), time.time()
    graph = {}

    # 1. Modèle juste : moyenne nulle
    sec = "1. Modèle juste : P&L moyen nul (risque-neutre)"
    for kind in ("call", "put"):
        for K in (90, 100, 110):
            for q in (0.0, 0.02):
                m = BlackScholes(S0, R, 0.25, q)
                res = simulate_hedging(m, K, 0.5, kind, n_reb=52, n_paths=NP, seed=K)
                bilan.check(sec, f"{kind} K={K} q={q}", abs(res.moyenne) <= 3.5 * res.erreur_type,
                            f"moyenne={res.moyenne:+.4f} ± {res.erreur_type:.4f}")

    # 2. Convergence en 1/√N
    sec = "2. Rebalancement discret : écart-type en 1/√N"
    m, K, T = BlackScholes(S0, R, 0.20), 100.0, 0.5
    Ns = [13, 26, 52, 104, 208, 416]
    et, th = [], []
    print("\n   Call ATM 6 mois, vol 20 % — écart-type du P&L :")
    for n in Ns:
        res = simulate_hedging(m, K, T, n_reb=n, n_paths=NP, seed=n)
        et.append(res.ecart_type)
        th.append(theoretical_std(m, K, T, n_reb=n))
        print(f"     N={n:4d}  simulé {et[-1]:.4f}   théorique {th[-1]:.4f}   "
              f"ratio {et[-1] / th[-1]:.3f}   ({et[-1] / res.prime:.1%} de la prime)")
        bilan.check(sec, f"Simulé ≈ théorique (N={n})", 0.95 <= et[-1] / th[-1] <= 1.08,
                    f"ratio={et[-1] / th[-1]:.3f}")
    pente = np.polyfit(np.log(Ns), np.log(et), 1)[0]
    bilan.check(sec, "Pente log-log ≈ −0,5", -0.55 <= pente <= -0.45, f"pente={pente:.3f}")
    print(f"     pente log-log : {pente:.3f}")
    graph["N"], graph["et"], graph["th"] = Ns, et, th

    # 3. Vol mal spécifiée
    sec = "3. Vol mal spécifiée : P&L = ½∫(σ̃² − σ²)S²Γ dt"
    sig_vraie = 0.20
    for sig_c in (0.15, 0.25):
        for kind in ("call", "put"):
            m = BlackScholes(S0, R, sig_c)
            res = simulate_hedging(m, 100, 0.5, kind, n_reb=500, n_paths=20_000,
                                     sigma_reelle=sig_vraie, seed=7)
            nom = f"{kind} σ̃={sig_c} σ={sig_vraie}"
            exact = float(bs_price(m, 100, 0.5, kind)
                          - bs_price(BlackScholes(S0, R, sig_vraie), 100, 0.5, kind))
            bilan.check(sec, f"Moyenne = BS(σ̃) − BS(σ) : {nom}",
                        abs(res.moyenne - exact) <= 3.5 * res.erreur_type,
                        f"moyenne={res.moyenne:+.4f} exact={exact:+.4f}")
            signe_ok = res.moyenne > 0 if sig_c > sig_vraie else res.moyenne < 0
            bilan.check(sec, f"Signe : {'gain' if sig_c > sig_vraie else 'perte'} ({nom})", signe_ok)
            # L'écart P&L − formule est l'erreur de discrétisation : il doit
            # décroître en 1/√N (×4 rebalancements -> écart-type ÷2).
            res4 = simulate_hedging(m, 100, 0.5, kind, n_reb=125, n_paths=20_000,
                                      sigma_reelle=sig_vraie, seed=7)
            e500 = np.std(res.pnl - res.erreur_vol)
            e125 = np.std(res4.pnl - res4.erreur_vol)
            r2 = 1 - np.var(res.pnl - res.erreur_vol) / np.var(res.pnl)
            bilan.check(sec, f"Écart à la formule en 1/√N : {nom}", 1.7 <= e125 / e500 <= 2.3,
                        f"N=125 : {e125:.4f}  N=500 : {e500:.4f}  ratio={e125 / e500:.2f} (R²={r2:.3f})")
            print(f"   {nom} : P&L moyen {res.moyenne:+.4f} (exact {exact:+.4f}) | "
                  f"R² formule {r2:.3f} | écart résiduel N=125 {e125:.4f} -> N=500 {e500:.4f}")
            if kind == "call" and sig_c == 0.25:
                graph["mis_pnl"], graph["mis_err"] = res.pnl, res.erreur_vol

    # 4. Couverture robuste (Barenblatt)
    sec = "4. Vol incertaine : couverture à σ_max / σ_min"
    rng = np.random.default_rng(1)
    sig_r = rng.uniform(0.10, 0.30, 20_000)
    for kind in ("call", "put"):
        hi = simulate_hedging(BlackScholes(S0, R, 0.30), 100, 0.5, kind, n_reb=500,
                                n_paths=20_000, sigma_reelle=sig_r, seed=3)
        lo = simulate_hedging(BlackScholes(S0, R, 0.10), 100, 0.5, kind, n_reb=500,
                                n_paths=20_000, sigma_reelle=sig_r, seed=3)
        # bruit de discrétisation à N=500 : ~1er ordre à la vol la plus forte
        bruit = theoretical_std(BlackScholes(S0, R, 0.30), 100, 0.5, n_reb=500)
        bilan.check(sec, f"σ̃=σ_max : terme de vol >= 0 partout ({kind})", hi.erreur_vol.min() >= 0)
        bilan.check(sec, f"σ̃=σ_max : P&L >= −3 × bruit de discrétisation ({kind})",
                    np.quantile(hi.pnl, 0.001) >= -3 * bruit,
                    f"quantile 0,1 %={np.quantile(hi.pnl, 0.001):+.4f} bruit={bruit:.4f}")
        bilan.check(sec, f"σ̃=σ_min : terme de vol <= 0 partout ({kind})", lo.erreur_vol.max() <= 0)
        print(f"   {kind} : prime σ_max {hi.prime:.3f} -> P&L moyen {hi.moyenne:+.3f} | "
              f"prime σ_min {lo.prime:.3f} -> P&L moyen {lo.moyenne:+.3f}")

    # 5. Gamma-neutre
    sec = "5. Couverture gamma-neutre"
    m = BlackScholes(S0, R, 0.20)
    sig_r = np.random.default_rng(2).uniform(0.15, 0.25, NP)
    G = HedgedOption(K=100, T=1.0)
    d_only = simulate_hedging(m, 100, 0.5, n_reb=52, n_paths=NP, sigma_reelle=sig_r, seed=4)
    g_neut = simulate_hedging(m, 100, 0.5, n_reb=52, n_paths=NP, sigma_reelle=sig_r, seed=4,
                                couverture_gamma=G)
    ratio = g_neut.ecart_type / d_only.ecart_type
    print(f"\n   Vol réalisée ∈ [15 %, 25 %], couverte à 20 %, N=52 :")
    print(f"     delta seul    : {d_only}")
    print(f"     gamma-neutre  : {g_neut}")
    bilan.check(sec, "Vol incertaine : écart-type divisé par > 3", ratio < 1 / 3, f"ratio={ratio:.3f}")
    bilan.check(sec, "Terme de vol quasi nul", np.abs(g_neut.erreur_vol).max() < 1e-8,
                f"max={np.abs(g_neut.erreur_vol).max():.2e}")
    a = simulate_hedging(m, 100, 0.5, n_reb=52, n_paths=NP, seed=5)
    b = simulate_hedging(m, 100, 0.5, n_reb=52, n_paths=NP, seed=5, couverture_gamma=G)
    bilan.check(sec, "Modèle juste : erreur de discrétisation divisée par > 3",
                b.ecart_type < a.ecart_type / 3, f"{a.ecart_type:.4f} -> {b.ecart_type:.4f}")
    bilan.check(sec, "P&L moyen nul (gamma-neutre)", abs(b.moyenne) <= 3.5 * b.erreur_type,
                f"moyenne={b.moyenne:+.5f} ± {b.erreur_type:.5f}")
    graph["d_only"], graph["g_neut"] = d_only.pnl, g_neut.pnl

    # 6. Dérive réelle
    sec = "6. Indépendance à la dérive réelle μ"
    m = BlackScholes(S0, R, 0.20)
    base = simulate_hedging(m, 100, 0.5, n_reb=52, n_paths=NP, seed=6)
    for mu in (-0.10, 0.15, 0.30):
        res = simulate_hedging(m, 100, 0.5, n_reb=52, n_paths=NP, seed=6, mu=mu)
        th_mu = theoretical_std(m, 100, 0.5, n_reb=52, mu=mu)
        bilan.check(sec, f"μ={mu:+.2f} : écart-type ≈ 1er ordre théorique",
                    0.95 <= res.ecart_type / th_mu <= 1.08,
                    f"simulé={res.ecart_type:.4f} théorique={th_mu:.4f} (μ=r : {base.ecart_type:.4f})")
        bilan.check(sec, f"μ={mu:+.2f} : P&L moyen ≈ 0 (< 2 % de la prime)",
                    abs(res.moyenne) < 0.02 * res.prime, f"moyenne={res.moyenne:+.4f}")

    print(f"\n   Durée : {time.time() - t0:.0f}s")
    ko = bilan.afficher()
    print(f"Graphique : {plot_results(graph)}")
    return ko


def plot_results(g):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    BLEU, ORANGE = "#2a78d6", "#eb6834"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=SURF)
    for ax in axes:
        ax.set_facecolor(SURF)
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8, length=0)

    # (a) 1/√N
    ax = axes[0]
    ax.loglog(g["N"], g["th"], color=INK, lw=1.6, label="1er ordre théorique")
    ax.scatter(g["N"], g["et"], s=40, color=BLEU, edgecolors=SURF, linewidths=1.2, zorder=3,
               label="Simulé")
    ax.set_xticks(g["N"]); ax.set_xticklabels(g["N"]); ax.minorticks_off()
    yt = [0.25, 0.5, 1.0]
    ax.set_yticks(yt); ax.set_yticklabels([f"{v:g}" for v in yt])
    ax.set_xlabel("Nombre de rebalancements N", fontsize=8, color=INK2)
    ax.set_ylabel("Écart-type du P&L", fontsize=8, color=INK2)
    ax.set_title("Rebalancement discret : erreur en 1/√N", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)

    # (b) vol mal spécifiée
    ax = axes[1]
    n = min(4000, len(g["mis_pnl"]))
    ax.scatter(g["mis_err"][:n], g["mis_pnl"][:n], s=8, color=BLEU, alpha=0.35, lw=0)
    lim = [min(g["mis_err"][:n].min(), g["mis_pnl"][:n].min()),
           max(g["mis_err"][:n].max(), g["mis_pnl"][:n].max())]
    ax.plot(lim, lim, color=INK, lw=1.2, label="P&L = formule")
    ax.set_xlabel(r"Formule $\frac{1}{2}\int e^{-rt}(\tilde\sigma^2-\sigma^2)\,S^2\,\Gamma\,dt$",
                  fontsize=8, color=INK2)
    ax.set_ylabel("P&L simulé", fontsize=8, color=INK2)
    ax.set_title("Vol couverte 25 %, réalisée 20 % (N=500)", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)

    # (c) gamma-neutre
    ax = axes[2]
    lo, hi = np.quantile(g["d_only"], [0.002, 0.998])
    bins = np.linspace(lo, hi, 70)
    ax.hist(g["d_only"], bins=bins, color=ORANGE, alpha=0.75, label="Delta seul")
    ax.hist(g["g_neut"], bins=bins, color=BLEU, alpha=0.75, label="Delta + gamma")
    ax.set_xlabel("P&L (vol réalisée ∈ [15 %, 25 %], couverte à 20 %)", fontsize=8, color=INK2)
    ax.set_ylabel("Trajectoires", fontsize=8, color=INK2)
    ax.set_title("Neutraliser le gamma neutralise la vol", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)

    fig.suptitle("Backtest de couverture : vente d'un call ATM 6 mois", x=0.01, ha="left",
                 fontsize=12, color=INK, y=1.02)
    fig.tight_layout()
    chemin = "hedging.png"
    fig.savefig(chemin, dpi=150, bbox_inches="tight", facecolor=SURF)
    return chemin


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
