"""
Tests of the pure functions of validate_market.py (no network, no plots).

download_yahoo and plotting are not tested: network dependency and graphical
output, low value for a unit test.
"""
import sys
from pathlib import Path
# Direct launch (VS Code ▶ button): makes the project root modules importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import math
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pandas")
import validate_market as vm                                     # noqa: E402
from dividends import american_price_div                       # noqa: E402

NY = vm.market_tz()
CLOTURE = datetime(2026, 10, 6, 16, 0, tzinfo=NY)       # valuation at the close
UN_JOUR = 1 / 365

S, R = 227.5, 0.04
DIVS = pd.DataFrame({"ex_date": ["2026-09-01", "2026-11-10", "2027-02-09", "2027-12-01"],
                     "montant": [0.26, 0.26, 0.26, 0.26]})


# --------------------------------------------------------------------------- #
# Time measurement
# --------------------------------------------------------------------------- #
def test_maturite_de_cloture_a_cloture_en_jours_entiers():
    assert vm.maturity("2026-10-14", CLOTURE) == pytest.approx(8 * UN_JOUR, abs=1e-12)


def test_maturite_en_cours_de_seance():
    """Snapshot at 11:00: 8 days and 5 hours remain until the 16:00 expiry."""
    valo = datetime(2026, 10, 6, 11, 0, tzinfo=NY)
    assert vm.maturity("2026-10-14", valo) == pytest.approx((8 + 5 / 24) * UN_JOUR, abs=1e-12)


def test_maturite_independante_du_fuseau_de_l_ordinateur():
    """The same instant expressed in Paris gives the same maturity."""
    valo_paris = datetime(2026, 10, 6, 22, 0, tzinfo=vm.ZoneInfo("Europe/Paris"))   # = 16:00 in New York
    assert vm.maturity("2026-10-14", valo_paris) == pytest.approx(8 * UN_JOUR, abs=1e-12)


def test_maturite_traverse_le_changement_d_heure():
    """End of daylight saving time in New York on 1 November 2026: the week lasts one hour more."""
    valo = datetime(2026, 10, 30, 16, 0, tzinfo=NY)
    assert vm.maturity("2026-11-06", valo) == pytest.approx((7 + 1 / 24) * UN_JOUR, abs=1e-12)


def test_instant_accepte_date_chaine_et_datetime():
    attendu = datetime(2026, 11, 10, 9, 30, tzinfo=NY)
    for jour in (date(2026, 11, 10), "2026-11-10", "2026-11-10 00:00:00",
                 pd.Timestamp("2026-11-10")):
        assert vm.market_time(jour, time(9, 30)) == attendu


# --------------------------------------------------------------------------- #
# divs_for and forward
# --------------------------------------------------------------------------- #
def test_divs_pour_garde_seulement_ceux_avant_echeance():
    dv = vm.divs_for(DIVS, CLOTURE, 0.5)
    # 2026-09-01 is in the past, 2027-12-01 is after expiry
    assert [round(t * 365) for t, _ in dv] == [35, 126]
    assert all(D == 0.26 for _, D in dv)


def test_divs_pour_detachement_a_l_ouverture():
    """
    Valuation at the close, ex-date 35 days later at 09:30: 35 d - 6 h 30,
    plus 1 h for crossing the end of daylight saving time (1 November).
    """
    divs = pd.DataFrame({"ex_date": ["2026-11-10 00:00:00"], "montant": [0.5]})
    attendu = (35 - 6.5 / 24 + 1 / 24) * UN_JOUR
    assert vm.divs_for(divs, CLOTURE, 1.0) == [(pytest.approx(attendu, abs=1e-12), 0.5)]


@pytest.mark.parametrize("heure, attendu", [(time(9, 0), 1), (time(10, 0), 0)])
def test_dividende_du_jour_avant_ou_apres_l_ouverture(heure, attendu):
    """Ex-date today: still to come before 09:30, already gone after."""
    divs = pd.DataFrame({"ex_date": ["2026-10-06"], "montant": [0.5]})
    valo = datetime.combine(date(2026, 10, 6), heure, tzinfo=NY)
    assert len(vm.divs_for(divs, valo, 1.0)) == attendu


def test_divs_pour_table_vide():
    assert vm.divs_for(pd.DataFrame(columns=["ex_date", "montant"]), CLOTURE, 1.0) == []


def test_forward_sans_dividende():
    assert vm.forward(S, 0.75, R, 0.01, []) == pytest.approx(S * math.exp(0.03 * 0.75))


def test_forward_retranche_les_dividendes_capitalises():
    dv = [(0.1, 1.0), (0.6, 2.0)]
    attendu = S * math.exp(R) - 1.0 * math.exp(R * 0.9) - 2.0 * math.exp(R * 0.4)
    assert vm.forward(S, 1.0, R, 0.0, dv) == pytest.approx(attendu)


# --------------------------------------------------------------------------- #
# implied_vol
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("K, T, sig, q, kind", [
    (220.0, 0.25, 0.28, 0.003, "put"),
    (240.0, 0.50, 0.24, 0.000, "call"),
    (200.0, 1.10, 0.31, 0.010, "put"),
    (227.5, 0.05, 0.22, 0.003, "call"),
])
def test_vol_implicite_aller_retour(K, T, sig, q, kind):
    dv = vm.divs_for(DIVS, CLOTURE, T)
    prix = american_price_div(S, K, T, R, q, sig, kind, dv, n=vm.N_PAS)
    assert vm.implied_vol(prix, S, K, T, R, q, kind, dv) == pytest.approx(sig, abs=1e-4)


def test_vol_implicite_none_sous_la_valeur_intrinseque():
    """Price below an American put's intrinsic value: no vol reproduces it."""
    assert vm.implied_vol(25.0, S, 260.0, 0.5, R, 0.0, "put", []) is None


def test_vol_implicite_none_au_dessus_du_maximum():
    """A call cannot be worth more than the underlying."""
    assert vm.implied_vol(S + 1.0, S, 220.0, 0.5, R, 0.0, "call", []) is None


def test_vol_implicite_avec_borrow_extreme():
    """
    Regression: with q = -40%, the tree evaluated at VOL_MIN had p > 1. Since the
    check on p, the lower search bound is raised to |r-q|·√dt.
    """
    T, q, sig = 1.2, -0.40, 0.30
    prix = american_price_div(S, 230.0, T, R, q, sig, "put", [], n=vm.N_PAS)
    assert vm.implied_vol(prix, S, 230.0, T, R, q, "put", []) == pytest.approx(sig, abs=1e-4)


# --------------------------------------------------------------------------- #
# calibrate_borrow
# --------------------------------------------------------------------------- #
def _paires(q_vrai, T, sig=0.26, ecart_strike=2.5):
    dv = vm.divs_for(DIVS, CLOTURE, T)
    F = vm.forward(S, T, R, q_vrai, dv)
    ks = [round(F / ecart_strike) * ecart_strike + i * ecart_strike for i in (-2, -1, 0, 1, 2)]
    lignes = [dict(strike=K, type=kind,
                   mid=american_price_div(S, K, T, R, q_vrai, sig, kind, dv, n=vm.N_PAS))
              for K in ks for kind in ("call", "put")]
    return pd.DataFrame(lignes), dv


@pytest.mark.slow
@pytest.mark.parametrize("q_vrai, T", [(0.003, 0.3), (0.02, 0.8), (-0.01, 0.5)])
def test_caler_borrow_retrouve_un_borrow_connu(q_vrai, T):
    df, dv = _paires(q_vrai, T)
    assert vm.calibrate_borrow(df, S, T, R, dv) == pytest.approx(q_vrai, abs=1e-3)


def test_caler_borrow_sans_paire_renvoie_zero():
    df = pd.DataFrame({"strike": [220.0, 230.0], "type": ["call", "put"], "mid": [12.0, 8.0]})
    assert vm.calibrate_borrow(df, S, 0.5, R, []) == 0.0


# --------------------------------------------------------------------------- #
# fit_smile (polynomial variant)
# --------------------------------------------------------------------------- #
def test_ajuster_smile_reproduit_un_polynome():
    x = np.linspace(-0.3, 0.3, 12)
    iv = 0.25 - 0.1 * x + 0.4 * x**2
    fit = vm.fit_smile(x, iv, np.ones_like(x))
    np.testing.assert_allclose(fit(x), iv, atol=1e-10)


@pytest.mark.parametrize("n, deg", [(3, 1), (6, 2), (12, 3)])
def test_ajuster_smile_degre_selon_nombre_de_points(n, deg):
    x = np.linspace(-0.2, 0.2, n)
    fit = vm.fit_smile(x, 0.2 + x**2, np.ones(n))
    assert fit.degree() == deg


if __name__ == "__main__":
    # Direct launch (python file.py / VS Code ▶ button) -> go through pytest
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
