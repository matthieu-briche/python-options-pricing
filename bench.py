"""Demo + benchmark: python bench.py"""
import sys
from pathlib import Path
# pricer.py and greeks.py are in the same folder as this script
sys.path.insert(0, str(Path(__file__).resolve().parent))

import math
import time

import numpy as np

from pricer import (BlackScholes, bs_delta_vega, bs_price, geometric_asian_price,
                    lsm_american_put, mc_asian_arithmetic, mc_european,
                    mc_greeks_pathwise, rqmc_european)


def timed(label, fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    print(f"{label:<42} {time.perf_counter() - t0:7.3f}s  {out}")
    return out


def naive_python_call(m, K, T, n, seed=42):
    """What NOT to do: a pure Python loop."""
    import random
    random.seed(seed)
    s = 0.0
    for _ in range(n):
        st = m.s0 * math.exp((m.r - 0.5 * m.sigma**2) * T + m.sigma * math.sqrt(T) * random.gauss(0, 1))
        s += max(st - K, 0.0)
    return math.exp(-m.r * T) * s / n


if __name__ == "__main__":
    m = BlackScholes(s0=100.0, r=0.05, sigma=0.2)
    K, T = 100.0, 1.0
    print(f"Black–Scholes exact call          : {float(bs_price(m, K, T)):.6f}")
    d, v = bs_delta_vega(m, K, T)
    print(f"Exact Delta / Vega                : {float(d):.6f} / {float(v):.6f}\n")

    print("== European (2M paths) ==")
    timed("Pure Python (200k only)", naive_python_call, m, K, T, 200_000)
    timed("NumPy MC plain", mc_european, m, K, T, antithetic=False, control_variate=False)
    timed("NumPy MC antithetic", mc_european, m, K, T, control_variate=False)
    timed("NumPy MC antithetic + control", mc_european, m, K, T)
    timed("RQMC Sobol 32 x 2^16", rqmc_european, m, K, T)

    print("\n== Greeks pathwise ==")
    g = timed("Delta/Vega pathwise (1M)", mc_greeks_pathwise, m, K, T)

    print("\n== Arithmetic Asian (52 dates) ==")
    print(f"Exact geometric (control)         : {geometric_asian_price(m, K, T, 52):.6f}")
    mc_asian_arithmetic(m, K, T, n_paths=10_000)          # JIT compilation (cached)
    timed("Numba, no control", mc_asian_arithmetic, m, K, T, control_variate=False)
    timed("Numba, geometric control", mc_asian_arithmetic, m, K, T)

    print("\n== American put (Longstaff–Schwartz) ==")
    print(f"Exact European put (lower bound)  : {float(bs_price(m, K, T, 'put')):.6f}")
    timed("LSM 200k x 50 dates", lsm_american_put, m, K, T)

    print("\n== Vectorisation: 10,000 strikes at once ==")
    Ks = np.linspace(50, 150, 10_000)
    t0 = time.perf_counter(); bs_price(m, Ks, T); print(f"bs_price vectorised: {time.perf_counter()-t0:.4f}s")
