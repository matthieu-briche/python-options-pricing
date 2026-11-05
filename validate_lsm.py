"""
Validation of the Longstaff-Schwartz American put (pricer.lsm_american_put) against QuantLib.

LSM is a Monte Carlo method, so exact equality cannot be required. The checks are:
  1. Bias: LSM is a lower bound (suboptimal exercise rule
     + finite number of exercise dates). Expected: -TOL_BIAIS <= LSM - QL <= 3 x standard error.
  2. Convergence in the number of exercise dates (n_steps): the gap must shrink.
  3. Stability vs seed: the dispersion across seeds must match the standard error.

Usage: python validate_lsm.py
"""
import itertools
import time

import numpy as np
import pandas as pd

from pricer import BlackScholes, lsm_american_put
from validate_pricer import ql_reference, Report

TOL_BIAIS = 0.05   # tolerated low bias (in price units, S = 100)
S0 = 100.0


def main():
    bilan, lignes, t0 = Report(), [], time.time()
    lsm_american_put(BlackScholes(100, 0.05, 0.2), 100, 1.0, n_paths=1000)  # Numba compilation

    sec = "1. LSM vs QuantLib (downward bias expected)"
    for K, days, sig, r, q in itertools.product([80, 100, 120], [91, 365, 730],
                                                [0.20, 0.40], [0.05], [0.0, 0.03]):
        T = days / 365
        res = lsm_american_put(BlackScholes(S0, r, sig, q), K, T)
        ref = ql_reference(S0, K, days, r, q, sig, "put")["prix"]
        ecart = res.price - ref
        ok = -TOL_BIAIS <= ecart <= 3 * res.std_error
        bilan.check(sec, f"put K={K} T={days}d vol={sig} q={q}", ok,
                    f"LSM={res.price:.4f}±{res.std_error:.4f} QL={ref:.4f} diff={ecart:+.4f}")
        lignes.append(dict(K=K, days=days, sigma=sig, r=r, q=q, lsm=res.price,
                           erreur_type=res.std_error, ql=ref, ecart=ecart,
                           ecart_en_se=ecart / res.std_error))

    sec = "2. Convergence in the number of exercise dates"
    m, K, T = BlackScholes(S0, 0.05, 0.3), 100.0, 1.0
    ref = ql_reference(S0, K, 365, 0.05, 0.0, 0.3, "put")["prix"]
    print("\n   1-year ATM put, vol 30%:  QL =", f"{ref:.4f}")
    ecarts = []
    for n_steps in [10, 25, 50, 100, 250]:
        res = lsm_american_put(m, K, T, n_steps=n_steps, n_paths=400_000)
        ecarts.append(res.price - ref)
        print(f"     n_steps={n_steps:4d}  LSM={res.price:.4f} ± {res.std_error:.4f}  diff={ecarts[-1]:+.4f}")
    bilan.check(sec, "Bias shrinks with more dates", abs(ecarts[-1]) < abs(ecarts[0]),
                f"{[round(e, 4) for e in ecarts]}")

    sec = "3. Stability across seeds"
    prix = [lsm_american_put(m, K, T, seed=s) for s in range(10)]
    disp = np.std([p.price for p in prix], ddof=1)
    se = np.mean([p.std_error for p in prix])
    bilan.check(sec, "Dispersion ≈ reported standard error", 0.5 * se < disp < 2 * se,
                f"dispersion={disp:.5f} std_error={se:.5f}")

    df = pd.DataFrame(lignes)
    df.to_csv("rapport_lsm.csv", index=False)
    print(f"\n   Mean gap LSM - QL: {df.ecart.mean():+.4f} | min {df.ecart.min():+.4f} "
          f"| max {df.ecart.max():+.4f} | duration {time.time() - t0:.0f}s")
    bilan.afficher()
    print("Details: rapport_lsm.csv")


if __name__ == "__main__":
    main()
