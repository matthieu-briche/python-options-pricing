"""
Adaptateur : expose greeks.crr_american avec l'interface attendue par validate_pricer.py.

    price(S, K, T, r, q, sigma, option_type, n=...)
    greeks(S, K, T, r, q, sigma, option_type) -> dict(delta, gamma, vega, theta, rho)

Conventions de crr_american (identiques à celles du script) :
vega pour +1.00 de vol, rho pour +1.00 de taux, theta par an.
"""
from greeks import crr_american
from pricer import BlackScholes

N = 2000  # valeur par défaut de crr_american


def price(S, K, T, r, q, sigma, option_type, n=N):
    m = BlackScholes(s0=S, r=r, sigma=sigma, q=q)
    return float(crr_american(m, K, T, kind=option_type, n=n)["price"])


def greeks(S, K, T, r, q, sigma, option_type, n=N):
    m = BlackScholes(s0=S, r=r, sigma=sigma, q=q)
    res = crr_american(m, K, T, kind=option_type, n=n)
    return {g: float(res[g]) for g in ("delta", "gamma", "vega", "theta", "rho")}
