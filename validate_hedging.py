"""
Validation of the hedging backtest (hedging.simulate_hedging), after
Bouchard & Chassagneux, ch. 7.3 and 5.2.1.

  1. Correct model: zero mean P&L. Under the risk-neutral measure this is exact even
     in discrete time (discounted self-financing portfolio = martingale).
  2. Discrete rebalancing: standard deviation in 1/√N (log-log slope ≈ −0.5), equal
     to the theoretical first order ½ σ⁴ Δt² Σ E[Γ² S⁴].
  3. Misspecified vol: P&L ≈ ½ ∫ e^{-rt}(σ̃² − σ²) S² Γ dt path by
     path, and mean P&L = BS(σ̃) − BS(σ) exactly.
  4. Robust hedging (Black-Scholes-Barenblatt): unknown realised vol in
     [σ_min, σ_max], convex payoff -> hedging at σ_max never loses (up to the
     discretisation error), hedging at σ_min never gains.
  5. Gamma-neutral: with a second option, the P&L becomes insensitive to realised
     vol and the discretisation error collapses.
  6. The real drift μ does not change the P&L dispersion.

Usage: python validate_hedging.py        (≈ 1 to 2 min)
Outputs: console + hedging.png
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

    # 1. Correct model: zero mean
    sec = "1. Correct model: zero mean P&L (risk-neutral)"
    for kind in ("call", "put"):
        for K in (90, 100, 110):
            for q in (0.0, 0.02):
                m = BlackScholes(S0, R, 0.25, q)
                res = simulate_hedging(m, K, 0.5, kind, n_reb=52, n_paths=NP, seed=K)
                bilan.check(sec, f"{kind} K={K} q={q}", abs(res.moyenne) <= 3.5 * res.erreur_type,
                            f"mean={res.moyenne:+.4f} ± {res.erreur_type:.4f}")

    # 2. Convergence in 1/√N
    sec = "2. Discrete rebalancing: standard deviation in 1/√N"
    m, K, T = BlackScholes(S0, R, 0.20), 100.0, 0.5
    Ns = [13, 26, 52, 104, 208, 416]
    et, th = [], []
    print("\n   6-month ATM call, vol 20% — P&L standard deviation:")
    for n in Ns:
        res = simulate_hedging(m, K, T, n_reb=n, n_paths=NP, seed=n)
        et.append(res.ecart_type)
        th.append(theoretical_std(m, K, T, n_reb=n))
        print(f"     N={n:4d}  simulated {et[-1]:.4f}   theoretical {th[-1]:.4f}   "
              f"ratio {et[-1] / th[-1]:.3f}   ({et[-1] / res.prime:.1%} of premium)")
        bilan.check(sec, f"Simulated ≈ theoretical (N={n})", 0.95 <= et[-1] / th[-1] <= 1.08,
                    f"ratio={et[-1] / th[-1]:.3f}")
    pente = np.polyfit(np.log(Ns), np.log(et), 1)[0]
    bilan.check(sec, "Log-log slope ≈ −0.5", -0.55 <= pente <= -0.45, f"slope={pente:.3f}")
    print(f"     log-log slope: {pente:.3f}")
    graph["N"], graph["et"], graph["th"] = Ns, et, th

    # 3. Misspecified vol
    sec = "3. Misspecified vol: P&L = ½∫(σ̃² − σ²)S²Γ dt"
    sig_vraie = 0.20
    for sig_c in (0.15, 0.25):
        for kind in ("call", "put"):
            m = BlackScholes(S0, R, sig_c)
            res = simulate_hedging(m, 100, 0.5, kind, n_reb=500, n_paths=20_000,
                                     sigma_reelle=sig_vraie, seed=7)
            nom = f"{kind} σ̃={sig_c} σ={sig_vraie}"
            exact = float(bs_price(m, 100, 0.5, kind)
                          - bs_price(BlackScholes(S0, R, sig_vraie), 100, 0.5, kind))
            bilan.check(sec, f"Mean = BS(σ̃) − BS(σ): {nom}",
                        abs(res.moyenne - exact) <= 3.5 * res.erreur_type,
                        f"mean={res.moyenne:+.4f} exact={exact:+.4f}")
            signe_ok = res.moyenne > 0 if sig_c > sig_vraie else res.moyenne < 0
            bilan.check(sec, f"Sign: {'gain' if sig_c > sig_vraie else 'loss'} ({nom})", signe_ok)
            # The P&L − formula gap is the discretisation error: it must
            # decrease in 1/√N (×4 rebalancings -> standard deviation ÷2).
            res4 = simulate_hedging(m, 100, 0.5, kind, n_reb=125, n_paths=20_000,
                                      sigma_reelle=sig_vraie, seed=7)
            e500 = np.std(res.pnl - res.erreur_vol)
            e125 = np.std(res4.pnl - res4.erreur_vol)
            r2 = 1 - np.var(res.pnl - res.erreur_vol) / np.var(res.pnl)
            bilan.check(sec, f"Gap to the formula in 1/√N: {nom}", 1.7 <= e125 / e500 <= 2.3,
                        f"N=125: {e125:.4f}  N=500: {e500:.4f}  ratio={e125 / e500:.2f} (R²={r2:.3f})")
            print(f"   {nom}: mean P&L {res.moyenne:+.4f} (exact {exact:+.4f}) | "
                  f"formula R² {r2:.3f} | residual gap N=125 {e125:.4f} -> N=500 {e500:.4f}")
            if kind == "call" and sig_c == 0.25:
                graph["mis_pnl"], graph["mis_err"] = res.pnl, res.erreur_vol

    # 4. Robust hedging (Barenblatt)
    sec = "4. Uncertain vol: hedging at σ_max / σ_min"
    rng = np.random.default_rng(1)
    sig_r = rng.uniform(0.10, 0.30, 20_000)
    for kind in ("call", "put"):
        hi = simulate_hedging(BlackScholes(S0, R, 0.30), 100, 0.5, kind, n_reb=500,
                                n_paths=20_000, sigma_reelle=sig_r, seed=3)
        lo = simulate_hedging(BlackScholes(S0, R, 0.10), 100, 0.5, kind, n_reb=500,
                                n_paths=20_000, sigma_reelle=sig_r, seed=3)
        # discretisation noise at N=500: ~first order at the highest vol
        bruit = theoretical_std(BlackScholes(S0, R, 0.30), 100, 0.5, n_reb=500)
        bilan.check(sec, f"σ̃=σ_max: vol term >= 0 everywhere ({kind})", hi.erreur_vol.min() >= 0)
        bilan.check(sec, f"σ̃=σ_max: P&L >= −3 × discretisation noise ({kind})",
                    np.quantile(hi.pnl, 0.001) >= -3 * bruit,
                    f"quantile 0.1%={np.quantile(hi.pnl, 0.001):+.4f} noise={bruit:.4f}")
        bilan.check(sec, f"σ̃=σ_min: vol term <= 0 everywhere ({kind})", lo.erreur_vol.max() <= 0)
        print(f"   {kind}: premium σ_max {hi.prime:.3f} -> mean P&L {hi.moyenne:+.3f} | "
              f"premium σ_min {lo.prime:.3f} -> mean P&L {lo.moyenne:+.3f}")

    # 5. Gamma-neutral
    sec = "5. Gamma-neutral hedging"
    m = BlackScholes(S0, R, 0.20)
    sig_r = np.random.default_rng(2).uniform(0.15, 0.25, NP)
    G = HedgedOption(K=100, T=1.0)
    d_only = simulate_hedging(m, 100, 0.5, n_reb=52, n_paths=NP, sigma_reelle=sig_r, seed=4)
    g_neut = simulate_hedging(m, 100, 0.5, n_reb=52, n_paths=NP, sigma_reelle=sig_r, seed=4,
                                couverture_gamma=G)
    ratio = g_neut.ecart_type / d_only.ecart_type
    print(f"\n   Realised vol ∈ [15%, 25%], hedged at 20%, N=52:")
    print(f"     delta only    : {d_only}")
    print(f"     gamma-neutral : {g_neut}")
    bilan.check(sec, "Uncertain vol: standard deviation divided by > 3", ratio < 1 / 3, f"ratio={ratio:.3f}")
    bilan.check(sec, "Vol term almost zero", np.abs(g_neut.erreur_vol).max() < 1e-8,
                f"max={np.abs(g_neut.erreur_vol).max():.2e}")
    a = simulate_hedging(m, 100, 0.5, n_reb=52, n_paths=NP, seed=5)
    b = simulate_hedging(m, 100, 0.5, n_reb=52, n_paths=NP, seed=5, couverture_gamma=G)
    bilan.check(sec, "Correct model: discretisation error divided by > 3",
                b.ecart_type < a.ecart_type / 3, f"{a.ecart_type:.4f} -> {b.ecart_type:.4f}")
    bilan.check(sec, "Zero mean P&L (gamma-neutral)", abs(b.moyenne) <= 3.5 * b.erreur_type,
                f"mean={b.moyenne:+.5f} ± {b.erreur_type:.5f}")
    graph["d_only"], graph["g_neut"] = d_only.pnl, g_neut.pnl

    # 6. Real drift
    sec = "6. Independence from the real drift μ"
    m = BlackScholes(S0, R, 0.20)
    base = simulate_hedging(m, 100, 0.5, n_reb=52, n_paths=NP, seed=6)
    for mu in (-0.10, 0.15, 0.30):
        res = simulate_hedging(m, 100, 0.5, n_reb=52, n_paths=NP, seed=6, mu=mu)
        th_mu = theoretical_std(m, 100, 0.5, n_reb=52, mu=mu)
        bilan.check(sec, f"μ={mu:+.2f}: standard deviation ≈ theoretical first order",
                    0.95 <= res.ecart_type / th_mu <= 1.08,
                    f"simulated={res.ecart_type:.4f} theoretical={th_mu:.4f} (μ=r: {base.ecart_type:.4f})")
        bilan.check(sec, f"μ={mu:+.2f}: mean P&L ≈ 0 (< 2% of premium)",
                    abs(res.moyenne) < 0.02 * res.prime, f"mean={res.moyenne:+.4f}")

    print(f"\n   Duration: {time.time() - t0:.0f}s")
    ko = bilan.afficher()
    print(f"Plot: {plot_results(graph)}")
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
    ax.loglog(g["N"], g["th"], color=INK, lw=1.6, label="Theoretical first order")
    ax.scatter(g["N"], g["et"], s=40, color=BLEU, edgecolors=SURF, linewidths=1.2, zorder=3,
               label="Simulated")
    ax.set_xticks(g["N"]); ax.set_xticklabels(g["N"]); ax.minorticks_off()
    yt = [0.25, 0.5, 1.0]
    ax.set_yticks(yt); ax.set_yticklabels([f"{v:g}" for v in yt])
    ax.set_xlabel("Number of rebalancings N", fontsize=8, color=INK2)
    ax.set_ylabel("P&L standard deviation", fontsize=8, color=INK2)
    ax.set_title("Discrete rebalancing: error in 1/√N", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)

    # (b) misspecified vol
    ax = axes[1]
    n = min(4000, len(g["mis_pnl"]))
    ax.scatter(g["mis_err"][:n], g["mis_pnl"][:n], s=8, color=BLEU, alpha=0.35, lw=0)
    lim = [min(g["mis_err"][:n].min(), g["mis_pnl"][:n].min()),
           max(g["mis_err"][:n].max(), g["mis_pnl"][:n].max())]
    ax.plot(lim, lim, color=INK, lw=1.2, label="P&L = formula")
    ax.set_xlabel(r"Formula $\frac{1}{2}\int e^{-rt}(\tilde\sigma^2-\sigma^2)\,S^2\,\Gamma\,dt$",
                  fontsize=8, color=INK2)
    ax.set_ylabel("Simulated P&L", fontsize=8, color=INK2)
    ax.set_title("Hedged vol 25%, realised 20% (N=500)", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)

    # (c) gamma-neutral
    ax = axes[2]
    lo, hi = np.quantile(g["d_only"], [0.002, 0.998])
    bins = np.linspace(lo, hi, 70)
    ax.hist(g["d_only"], bins=bins, color=ORANGE, alpha=0.75, label="Delta only")
    ax.hist(g["g_neut"], bins=bins, color=BLEU, alpha=0.75, label="Delta + gamma")
    ax.set_xlabel("P&L (realised vol ∈ [15%, 25%], hedged at 20%)", fontsize=8, color=INK2)
    ax.set_ylabel("Paths", fontsize=8, color=INK2)
    ax.set_title("Neutralising gamma neutralises vol", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)

    fig.suptitle("Hedging backtest: sale of a 6-month ATM call", x=0.01, ha="left",
                 fontsize=12, color=INK, y=1.02)
    fig.tight_layout()
    chemin = "hedging.png"
    fig.savefig(chemin, dpi=150, bbox_inches="tight", facecolor=SURF)
    return chemin


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
