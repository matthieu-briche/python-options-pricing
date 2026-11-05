import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""Tests of dividends.crr_american_div (QuantLib 1.43 reference values, FD 1600x1600)."""
import pytest

from dividends import crr_american_div, american_price_div
from greeks import crr_american
from pricer import BlackScholes

M = BlackScholes(s0=100.0, r=0.04, sigma=0.25)
DIVS = [(45 / 365, 1.5), (136 / 365, 1.5), (227 / 365, 1.5), (318 / 365, 1.5)]


@pytest.mark.parametrize("kind", ["call", "put"])
def test_sans_dividende_identique_a_crr(kind):
    a = crr_american_div(M, 100, 1.0, kind, [])
    b = crr_american(M, 100, 1.0, kind)
    for g in ("price", "delta", "gamma", "theta"):
        assert a[g] == pytest.approx(b[g], abs=1e-9)


@pytest.mark.parametrize("kind, K, model, ref", [
    ("call", 90, "spot", 14.1752), ("call", 100, "spot", 9.0816),
    ("put", 100, "spot", 10.9631), ("put", 110, "spot", 17.3338),
    ("call", 90, "escrowed", 13.9009), ("call", 100, "escrowed", 8.7823),
    ("put", 100, "escrowed", 10.6696), ("put", 110, "escrowed", 17.0663),
])
def test_prix_vs_quantlib(kind, K, model, ref):
    assert crr_american_div(M, K, 1.0, kind, DIVS, model=model)["price"] == pytest.approx(ref, abs=0.005)


def test_call_decroissant_put_croissant_avec_dividende():
    calls = [american_price_div(100, 100, 1.0, 0.04, 0, 0.25, "call", [(0.4, D)]) for D in (0, 1, 3)]
    puts = [american_price_div(100, 100, 1.0, 0.04, 0, 0.25, "put", [(0.4, D)]) for D in (0, 1, 3)]
    assert calls[0] > calls[1] > calls[2]
    assert puts[0] < puts[1] < puts[2]


def test_exercice_anticipe_avant_gros_dividende():
    # Deep ITM call, dividend of 8 in 5 days: exercise before the ex-date
    p = crr_american_div(BlackScholes(100, 0.04, 0.15), 50, 0.5, "call", [(5 / 365, 8.0)])["price"]
    assert p == pytest.approx(50.0, abs=0.05)


def test_modele_inconnu():
    with pytest.raises(ValueError):
        crr_american_div(M, 100, 1.0, "put", DIVS, model="other")
