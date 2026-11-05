"""
pricer.py — CPU-optimised option pricing.

Methods drawn from G. Pagès, *Numerical Probability* (Springer, 2018):
  - Ch. 2  : Monte Carlo + confidence interval, Greeks (pathwise / tangent method)
  - Ch. 3  : variance reduction (antithetic, control variate with optimal beta)
  - Ch. 4  : randomised quasi-Monte Carlo (scrambled Sobol)
  - Ch. 7-8: path-dependent (Asian) options with the exact Black–Scholes scheme
  - Ch. 12 : American options by regression (Longstaff–Schwartz)

CPU optimisation principles applied:
  1. NumPy vectorisation (no Python loop over the paths).
  2. Numba kernels @njit(parallel=True, fastmath=True) for time loops
     (path-dependent) -> machine code, multi-core, no (n_paths, n_steps) matrix.
  3. Chunked processing: bounded memory, data stays in the CPU cache.
  4. Contiguous float64, in-place operations, no needless allocations.
  5. Modern RNG (PCG64) with SeedSequence -> reproducible and parallelisable.
  6. Variance reduction = the best "CPU optimiser": dividing the variance
     by 10 is the same as dividing the number of simulations (hence the time) by 10.
"""
from __future__ import annotations

if __name__ == "__main__":
    # File run directly (python pricer.py / VS Code ▶ button): the demo runs through
    # the imported "pricer" module and not "__main__", because the Numba cache
    # is tied to the module name.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    import pricer as _m
    _m._demo()
    raise SystemExit(0)

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numba import njit, prange
from scipy.special import ndtr, ndtri
from scipy.stats import qmc

OptionType = Literal["call", "put"]
Z95 = 1.959963984540054


# --------------------------------------------------------------------------- #
# Model and result
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class BlackScholes:
    """Black–Scholes model: dS = S((r - q)dt + sigma dW)."""
    s0: float
    r: float
    sigma: float
    q: float = 0.0

    def __post_init__(self) -> None:
        if self.s0 <= 0 or self.sigma <= 0:
            raise ValueError("s0 and sigma must be > 0")


def check_proba_crr(p: float, m: BlackScholes, dt: float) -> None:
    """
    CRR tree: the risk-neutral probability must lie in (0, 1), otherwise the tree
    produces meaningless prices (arbitrage in the tree). With u = e^{σ√dt},
    this is exactly equivalent to  σ > |r - q| · √dt.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(
            f"CRR tree: probability p={p:.4f} outside (0, 1) "
            f"(sigma={m.sigma}, r-q={m.r - m.q:+.4f}, dt={dt:.2e}). "
            f"sigma > |r-q|·√dt = {abs(m.r - m.q) * math.sqrt(dt):.4g} is required: "
            "increase n or check the parameters.")


def sigma_min_crr(r: float, q: float, T: float, n: int) -> float:
    """Smallest volatility accepted by an n-step CRR tree (see check_proba_crr)."""
    return abs(r - q) * math.sqrt(T / n)


@dataclass(frozen=True, slots=True)
class MCResult:
    price: float
    std_error: float
    n_paths: int

    @property
    def ci95(self) -> tuple[float, float]:
        h = Z95 * self.std_error
        return self.price - h, self.price + h

    def __str__(self) -> str:
        lo, hi = self.ci95
        return f"{self.price:.6f} ± {Z95*self.std_error:.6f}  (CI95 [{lo:.6f}, {hi:.6f}], N={self.n_paths:,})"


class _RunningStats:
    """Online mean/variance by blocks (Chan et al.): O(1) memory."""
    __slots__ = ("n", "mean", "m2")

    def __init__(self) -> None:
        self.n, self.mean, self.m2 = 0, 0.0, 0.0

    def update(self, x: np.ndarray) -> None:
        nb = x.size
        mb = float(x.mean())
        m2b = float(((x - mb) ** 2).sum())
        n = self.n + nb
        d = mb - self.mean
        self.mean += d * nb / n
        self.m2 += m2b + d * d * self.n * nb / n
        self.n = n

    def result(self, scale: float = 1.0) -> MCResult:
        var = self.m2 / (self.n - 1)
        return MCResult(scale * self.mean, scale * math.sqrt(var / self.n), self.n)


# --------------------------------------------------------------------------- #
# 1. Closed forms (reference + control variates) — vectorised
# --------------------------------------------------------------------------- #
def bs_price(m: BlackScholes, K, T, kind: OptionType = "call"):
    """Black–Scholes price, vectorised on K and T (NumPy broadcast)."""
    K, T = np.asarray(K, float), np.asarray(T, float)
    sq = m.sigma * np.sqrt(T)
    d1 = (np.log(m.s0 / K) + (m.r - m.q + 0.5 * m.sigma**2) * T) / sq
    d2 = d1 - sq
    df_r, df_q = np.exp(-m.r * T), np.exp(-m.q * T)
    if kind == "call":
        return m.s0 * df_q * ndtr(d1) - K * df_r * ndtr(d2)
    return K * df_r * ndtr(-d2) - m.s0 * df_q * ndtr(-d1)


def bs_delta_vega(m: BlackScholes, K, T, kind: OptionType = "call"):
    K, T = np.asarray(K, float), np.asarray(T, float)
    sq = m.sigma * np.sqrt(T)
    d1 = (np.log(m.s0 / K) + (m.r - m.q + 0.5 * m.sigma**2) * T) / sq
    df_q = np.exp(-m.q * T)
    delta = df_q * (ndtr(d1) if kind == "call" else ndtr(d1) - 1.0)
    vega = m.s0 * df_q * np.sqrt(T) * np.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
    return delta, vega


def geometric_asian_price(m: BlackScholes, K: float, T: float, n: int,
                          kind: OptionType = "call") -> float:
    """Discrete geometric Asian (dates iT/n, i=1..n): Gaussian log G."""
    mu = math.log(m.s0) + (m.r - m.q - 0.5 * m.sigma**2) * T * (n + 1) / (2 * n)
    s = m.sigma * math.sqrt(T * (n + 1) * (2 * n + 1) / (6 * n * n))
    d2 = (mu - math.log(K)) / s
    d1 = d2 + s
    fwd = math.exp(mu + 0.5 * s * s)
    if kind == "call":
        return math.exp(-m.r * T) * (fwd * ndtr(d1) - K * ndtr(d2))
    return math.exp(-m.r * T) * (K * ndtr(-d2) - fwd * ndtr(-d1))


# --------------------------------------------------------------------------- #
# 2. European Monte Carlo: antithetic + control variate (Ch. 2-3)
# --------------------------------------------------------------------------- #
def mc_european(m: BlackScholes, K: float, T: float, kind: OptionType = "call",
                n_paths: int = 2_000_000, chunk: int = 2**18,
                antithetic: bool = True, control_variate: bool = True,
                seed: int = 42) -> MCResult:
    """
    Exact simulation of S_T (no discretisation).
    Control variate: S_T, with known expectation S0 e^{(r-q)T}.
    Optimal beta estimated on an independent pilot block (avoids bias).
    """
    rng = np.random.default_rng(seed)
    drift = (m.r - m.q - 0.5 * m.sigma**2) * T
    vol = m.sigma * math.sqrt(T)
    fwd = m.s0 * math.exp((m.r - m.q) * T)
    sign = 1.0 if kind == "call" else -1.0

    def payoff_and_cv(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        st = m.s0 * np.exp(drift + vol * z)
        pay = np.maximum(sign * (st - K), 0.0)
        if antithetic:
            st2 = m.s0 * np.exp(drift - vol * z)
            pay = 0.5 * (pay + np.maximum(sign * (st2 - K), 0.0))
            st = 0.5 * (st + st2)
        return pay, st

    beta = 0.0
    if control_variate:
        p, c = payoff_and_cv(rng.standard_normal(min(chunk, 50_000)))
        cov = np.cov(p, c)
        beta = cov[0, 1] / cov[1, 1]

    stats = _RunningStats()
    n_draws = n_paths // 2 if antithetic else n_paths
    done = 0
    while done < n_draws:
        b = min(chunk, n_draws - done)
        p, c = payoff_and_cv(rng.standard_normal(b))
        if control_variate:
            p -= beta * (c - fwd)
        stats.update(p)
        done += b
    res = stats.result(math.exp(-m.r * T))
    return MCResult(res.price, res.std_error, n_paths)


# --------------------------------------------------------------------------- #
# 3. Randomised quasi-Monte Carlo (Ch. 4): scrambled Sobol + replications
# --------------------------------------------------------------------------- #
def rqmc_european(m: BlackScholes, K: float, T: float, kind: OptionType = "call",
                  log2_n: int = 16, n_replicas: int = 32, seed: int = 0) -> MCResult:
    """
    Each replication = an independent scrambled Sobol -> unbiased estimator.
    The error is measured from the dispersion between replications (valid CI).
    """
    drift = (m.r - m.q - 0.5 * m.sigma**2) * T
    vol = m.sigma * math.sqrt(T)
    sign = 1.0 if kind == "call" else -1.0
    ss = np.random.SeedSequence(seed)
    est = np.empty(n_replicas)
    for i, child in enumerate(ss.spawn(n_replicas)):
        u = qmc.Sobol(d=1, scramble=True, seed=np.random.default_rng(child)).random_base2(log2_n).ravel()
        z = ndtri(u)
        st = m.s0 * np.exp(drift + vol * z)
        est[i] = np.maximum(sign * (st - K), 0.0).mean()
    df = math.exp(-m.r * T)
    return MCResult(df * est.mean(), df * est.std(ddof=1) / math.sqrt(n_replicas),
                    n_replicas * 2**log2_n)


# --------------------------------------------------------------------------- #
# 4. Monte Carlo Greeks: pathwise method / tangent process (Ch. 2.2.4)
# --------------------------------------------------------------------------- #
def mc_greeks_pathwise(m: BlackScholes, K: float, T: float, n_paths: int = 1_000_000,
                       seed: int = 7) -> dict[str, MCResult]:
    """Delta and Vega of a call: derivative of the payoff along the path."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    sq = math.sqrt(T)
    st = m.s0 * np.exp((m.r - m.q - 0.5 * m.sigma**2) * T + m.sigma * sq * z)
    itm = st > K
    df = math.exp(-m.r * T)
    d_delta = df * itm * st / m.s0                              # dS_T/dS0 = S_T/S0
    d_vega = df * itm * st * (sq * z - m.sigma * T)             # dS_T/dsigma
    out = {}
    for name, x in (("delta", d_delta), ("vega", d_vega)):
        out[name] = MCResult(float(x.mean()), float(x.std(ddof=1) / math.sqrt(n_paths)), n_paths)
    return out


# --------------------------------------------------------------------------- #
# 5. Arithmetic Asian: parallel Numba kernel + geometric control
# --------------------------------------------------------------------------- #
@njit(parallel=True, fastmath=True, cache=True)
def _asian_kernel(z, log_s0, drift_dt, vol_dt, K, sign):
    """
    z: (n_paths, n_steps) Gaussians. One path per prange iteration:
    time loop in registers, no price matrix stored.
    Returns the arithmetic and geometric payoffs (control variate).
    """
    n_paths, n_steps = z.shape
    arith = np.empty(n_paths)
    geo = np.empty(n_paths)
    for i in prange(n_paths):
        log_s = log_s0
        acc = 0.0
        acc_log = 0.0
        for j in range(n_steps):
            log_s += drift_dt + vol_dt * z[i, j]
            acc += math.exp(log_s)
            acc_log += log_s
        a = acc / n_steps
        g = math.exp(acc_log / n_steps)
        arith[i] = max(sign * (a - K), 0.0)
        geo[i] = max(sign * (g - K), 0.0)
    return arith, geo


def mc_asian_arithmetic(m: BlackScholes, K: float, T: float, n_steps: int = 52,
                        kind: OptionType = "call", n_paths: int = 500_000,
                        chunk: int = 2**16, control_variate: bool = True,
                        seed: int = 1) -> MCResult:
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift_dt = (m.r - m.q - 0.5 * m.sigma**2) * dt
    vol_dt = m.sigma * math.sqrt(dt)
    sign = 1.0 if kind == "call" else -1.0
    geo_exact = geometric_asian_price(m, K, T, n_steps, kind) * math.exp(m.r * T)

    def block(b):
        z = rng.standard_normal((b, n_steps))
        return _asian_kernel(z, math.log(m.s0), drift_dt, vol_dt, K, sign)

    beta = 0.0
    if control_variate:
        a, g = block(20_000)
        cov = np.cov(a, g)
        beta = cov[0, 1] / cov[1, 1]

    stats = _RunningStats()
    done = 0
    while done < n_paths:
        b = min(chunk, n_paths - done)
        a, g = block(b)
        if control_variate:
            a -= beta * (g - geo_exact)
        stats.update(a)
        done += b
    return stats.result(math.exp(-m.r * T))


# --------------------------------------------------------------------------- #
# 6. American: Longstaff–Schwartz (Ch. 12)
# --------------------------------------------------------------------------- #
@njit(parallel=True, cache=True)
def _lsm_kernel(z, log_s0, drift_dt, vol_dt, K, df, degree, n_blocks):
    """
    z: (n_steps, n_paths) Gaussians, transformed IN PLACE into prices S (no
    copy: memory divided by 2). Row t = date (t+1)·dt, contiguous access.

    Regression through the normal equations: G c = b with G_kl = Σ x^(k+l),
    b_k = Σ x^k · cash over the in-the-money paths. The sums are
    computed in parallel by blocks (reductions with no write conflicts),
    then a (degree+1)×(degree+1) system is solved: O(n) cost per date,
    instead of an SVD of the n × (degree+1) matrix with lstsq.
    """
    n_steps, n = z.shape
    p = degree + 1
    # 1. Paths: cumulative log-prices then exponential, in place
    for t in range(n_steps):
        for i in prange(n):
            prev = log_s0 if t == 0 else z[t - 1, i]
            z[t, i] = prev + drift_dt + vol_dt * z[t, i]
    for t in range(n_steps):
        for i in prange(n):
            z[t, i] = math.exp(z[t, i])

    cash = np.empty(n)
    for i in prange(n):
        cash[i] = max(K - z[n_steps - 1, i], 0.0)

    block = (n + n_blocks - 1) // n_blocks
    mom = np.empty((n_blocks, 2 * p - 1))
    rhs = np.empty((n_blocks, p))
    cnt = np.empty(n_blocks, np.int64)
    G = np.empty((p, p))
    coef = np.zeros(p)

    # 2. Backward dynamic programming
    for t in range(n_steps - 2, -1, -1):
        for i in prange(n):
            cash[i] *= df
        for b in prange(n_blocks):                       # partial sums per block
            mom_b = np.zeros(2 * p - 1)
            rhs_b = np.zeros(p)
            c = 0
            for i in range(b * block, min(n, (b + 1) * block)):
                s = z[t, i]
                if s < K:
                    x = s / K                            # normalisation: good conditioning
                    xk = 1.0
                    for k in range(2 * p - 1):
                        mom_b[k] += xk
                        if k < p:
                            rhs_b[k] += xk * cash[i]
                        xk *= x
                    c += 1
            mom[b] = mom_b
            rhs[b] = rhs_b
            cnt[b] = c
        if cnt.sum() <= p:
            continue
        m_tot = mom.sum(axis=0)
        r_tot = rhs.sum(axis=0)
        for k in range(p):
            for l in range(p):
                G[k, l] = m_tot[k + l]
        coef[:] = np.linalg.solve(G, r_tot)
        for i in prange(n):                              # exercise decision
            s = z[t, i]
            if s < K:
                x = s / K
                cont = 0.0
                for k in range(p - 1, -1, -1):           # Horner scheme
                    cont = cont * x + coef[k]
                if K - s > cont:
                    cash[i] = K - s
    for i in prange(n):
        cash[i] *= df
    return cash


def lsm_american_put(m: BlackScholes, K: float, T: float, n_steps: int = 50,
                     n_paths: int = 200_000, degree: int = 3, seed: int = 3) -> MCResult:
    """
    Backward dynamic programming; conditional expectation approximated by
    polynomial regression on the in-the-money paths.
    (Slightly low-biased estimator — a lower bound on the true price.)
    Full loop in a parallel Numba kernel (see _lsm_kernel).
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_steps, n_paths))
    cash = _lsm_kernel(z, math.log(m.s0), (m.r - m.q - 0.5 * m.sigma**2) * dt,
                       m.sigma * math.sqrt(dt), float(K), math.exp(-m.r * dt),
                       degree, 64)
    price = max(float(cash.mean()), K - m.s0)           # immediate exercise possible
    return MCResult(price, float(cash.std(ddof=1) / math.sqrt(n_paths)), n_paths)


# --------------------------------------------------------------------------- #
# Quick demo
# --------------------------------------------------------------------------- #
def _demo() -> None:
    m = BlackScholes(s0=100.0, r=0.05, sigma=0.2)
    print("S=K=100, r=5%, sigma=20%, T=1\n")
    print(f"Exact European call      : {float(bs_price(m, 100, 1)):.6f}")
    print(f"European call MC         : {mc_european(m, 100, 1)}")
    print(f"European call QMC        : {rqmc_european(m, 100, 1)}")
    print(f"Arithmetic Asian (52 d.) : {mc_asian_arithmetic(m, 100, 1)}")
    print(f"American put (LSM)       : {lsm_american_put(m, 100, 1)}")
    print("\nFor the full demo: python bench.py")

