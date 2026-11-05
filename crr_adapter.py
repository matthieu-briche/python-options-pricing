"""
Adapter: exposes greeks.crr_american with the interface expected by validate_pricer.py.

    price(S, K, T, r, q, sigma, option_type, n=...)
    greeks(S, K, T, r, q, sigma, option_type) -> dict(delta, gamma, vega, theta, rho)

crr_american conventions (same as the script's):
vega per +1.00 of vol, rho per +1.00 of rate, theta per year.
"""
from greeks import crr_american
from pricer import BlackScholes

N = 2000  # crr_american default value


def price(S, K, T, r, q, sigma, option_type, n=N):
    m = BlackScholes(s0=S, r=r, sigma=sigma, q=q)
    return float(crr_american(m, K, T, kind=option_type, n=n)["price"])


def greeks(S, K, T, r, q, sigma, option_type, n=N):
    m = BlackScholes(s0=S, r=r, sigma=sigma, q=q)
    res = crr_american(m, K, T, kind=option_type, n=n)
    return {g: float(res[g]) for g in ("delta", "gamma", "vega", "theta", "rho")}
