"""
Validation du put américain Longstaff-Schwartz (pricer.lsm_american_put) contre QuantLib.

Le LSM est un Monte-Carlo : on ne peut pas exiger l'égalité exacte. On vérifie :
  1. Biais : LSM est une borne inférieure (sous-optimalité de la règle d'exercice
     + nombre fini de dates d'exercice). On attend  -TOL_BIAIS <= LSM - QL <= 3 x erreur-type.
  2. Convergence en nombre de dates d'exercice (n_steps) : l'écart doit diminuer.
  3. Stabilité vs seed : la dispersion entre seeds doit être cohérente avec l'erreur-type.

Usage : python validate_lsm.py
"""
import itertools
import time

import numpy as np
import pandas as pd

from pricer import BlackScholes, lsm_american_put
from validate_pricer import ql_reference, Report

TOL_BIAIS = 0.05   # biais bas toléré (en unités de prix, S = 100)
S0 = 100.0


def main():
    bilan, lignes, t0 = Report(), [], time.time()
    lsm_american_put(BlackScholes(100, 0.05, 0.2), 100, 1.0, n_paths=1000)  # compilation Numba

    sec = "1. LSM vs QuantLib (biais attendu vers le bas)"
    for K, days, sig, r, q in itertools.product([80, 100, 120], [91, 365, 730],
                                                [0.20, 0.40], [0.05], [0.0, 0.03]):
        T = days / 365
        res = lsm_american_put(BlackScholes(S0, r, sig, q), K, T)
        ref = ql_reference(S0, K, days, r, q, sig, "put")["prix"]
        ecart = res.price - ref
        ok = -TOL_BIAIS <= ecart <= 3 * res.std_error
        bilan.check(sec, f"put K={K} T={days}j vol={sig} q={q}", ok,
                    f"LSM={res.price:.4f}±{res.std_error:.4f} QL={ref:.4f} écart={ecart:+.4f}")
        lignes.append(dict(K=K, days=days, sigma=sig, r=r, q=q, lsm=res.price,
                           erreur_type=res.std_error, ql=ref, ecart=ecart,
                           ecart_en_se=ecart / res.std_error))

    sec = "2. Convergence en nombre de dates d'exercice"
    m, K, T = BlackScholes(S0, 0.05, 0.3), 100.0, 1.0
    ref = ql_reference(S0, K, 365, 0.05, 0.0, 0.3, "put")["prix"]
    print("\n   Put ATM 1 an, vol 30 % :  QL =", f"{ref:.4f}")
    ecarts = []
    for n_steps in [10, 25, 50, 100, 250]:
        res = lsm_american_put(m, K, T, n_steps=n_steps, n_paths=400_000)
        ecarts.append(res.price - ref)
        print(f"     n_steps={n_steps:4d}  LSM={res.price:.4f} ± {res.std_error:.4f}  écart={ecarts[-1]:+.4f}")
    bilan.check(sec, "Le biais diminue avec plus de dates", abs(ecarts[-1]) < abs(ecarts[0]),
                f"{[round(e, 4) for e in ecarts]}")

    sec = "3. Stabilité entre seeds"
    prix = [lsm_american_put(m, K, T, seed=s) for s in range(10)]
    disp = np.std([p.price for p in prix], ddof=1)
    se = np.mean([p.std_error for p in prix])
    bilan.check(sec, "Dispersion ≈ erreur-type annoncée", 0.5 * se < disp < 2 * se,
                f"dispersion={disp:.5f} erreur-type={se:.5f}")

    df = pd.DataFrame(lignes)
    df.to_csv("rapport_lsm.csv", index=False)
    print(f"\n   Écart moyen LSM - QL : {df.ecart.mean():+.4f} | min {df.ecart.min():+.4f} "
          f"| max {df.ecart.max():+.4f} | durée {time.time() - t0:.0f}s")
    bilan.afficher()
    print("Détail : rapport_lsm.csv")


if __name__ == "__main__":
    main()
