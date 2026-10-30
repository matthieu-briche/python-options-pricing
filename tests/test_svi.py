"""Tests de svi.py : formule, contrôles d'arbitrage, calage."""
import sys
from pathlib import Path
# Lancement direct (bouton ▶ de VS Code) : rend importables les modules de la racine du projet
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from svi import SVI, calendar_arbitrage, butterfly_arbitrage, calibrate_svi

# Tranche « saine » : skew négatif typique d'un indice actions
SAIN = SVI(a=0.04, b=0.4, rho=-0.4, m=0.0, s=0.1, T=1.0)
# Exemple d'Axel Vogt (Gatheral & Jacquier, 2014) : arbitrage papillon connu
VOGT = SVI(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, s=0.4153, T=1.0)


# --------------------------------------------------------------------------- #
# Formule et dérivées
# --------------------------------------------------------------------------- #
def test_variance_au_minimum_egale_formule_fermee():
    k = np.linspace(-3, 3, 200_001)
    assert SAIN.w(k).min() == pytest.approx(SAIN.min_variance(), abs=1e-8)


def test_vol_est_racine_de_w_sur_T():
    sv = SVI(0.02, 0.3, -0.5, 0.05, 0.2, T=0.5)
    k = np.linspace(-1, 1, 11)
    np.testing.assert_allclose(sv.vol(k), np.sqrt(sv.w(k) / 0.5))
    np.testing.assert_allclose(sv(k), sv.vol(k))         # __call__


def test_ailes_lineaires_pentes_b_1_plus_ou_moins_rho():
    """Loin du minimum, w est linéaire de pente b(1+ρ) à droite et -b(1-ρ) à gauche."""
    h = 1e-3
    pente_d = (SAIN.w(50 + h) - SAIN.w(50 - h)) / (2 * h)
    pente_g = (SAIN.w(-50 + h) - SAIN.w(-50 - h)) / (2 * h)
    assert pente_d == pytest.approx(SAIN.b * (1 + SAIN.rho), rel=1e-4)
    assert pente_g == pytest.approx(-SAIN.b * (1 - SAIN.rho), rel=1e-4)


def test_derivees_analytiques_vs_differences_finies():
    k = np.linspace(-1.2, 1.2, 25)
    h = 1e-5
    w1, w2 = SAIN._dw(k)
    np.testing.assert_allclose(w1, (SAIN.w(k + h) - SAIN.w(k - h)) / (2 * h), atol=1e-7)
    np.testing.assert_allclose(w2, (SAIN.w(k + h) - 2 * SAIN.w(k) + SAIN.w(k - h)) / h**2,
                               atol=1e-4)


# --------------------------------------------------------------------------- #
# Arbitrage papillon (fonction g de Durrleman)
# --------------------------------------------------------------------------- #
def test_papillon_ok_sur_tranche_saine():
    ok, g_min, _ = butterfly_arbitrage(SAIN)
    assert ok and g_min > 0


def test_papillon_detecte_exemple_de_vogt():
    ok, g_min, k_min = butterfly_arbitrage(VOGT)
    assert not ok
    assert g_min < 0
    assert 0.5 < k_min < 1.2          # violation connue dans l'aile droite


def test_g_negatif_ssi_densite_negative():
    """Cohérence de g avec la densité obtenue par différences finies des prix (Breeden-Litzenberger)."""
    from scipy.special import ndtr
    k = np.linspace(-1.0, 1.4, 481)
    F = 1.0

    def call(kk):
        w = VOGT.w(kk)
        d1 = -kk / np.sqrt(w) + np.sqrt(w) / 2
        return F * ndtr(d1) - F * np.exp(kk) * ndtr(d1 - np.sqrt(w))

    K = F * np.exp(k)
    h = 1e-3
    dens = (call(np.log(K + h)) - 2 * call(k) + call(np.log(K - h))) / h**2
    g = VOGT.g(k)
    # On écarte les points où g ou la densité sont trop proches de 0 pour trancher
    zone_nette = (np.abs(g) > 1e-3) & (np.abs(dens) > 1e-8)
    assert (dens[zone_nette] < 0).sum() > 20          # la zone d'arbitrage est bien testée
    assert ((g < 0) == (dens < 0))[zone_nette].all()


# --------------------------------------------------------------------------- #
# Arbitrage calendaire
# --------------------------------------------------------------------------- #
def test_calendaire_aucune_violation_si_variance_croissante():
    s1 = SVI(0.02, 0.2, -0.4, 0.0, 0.1, T=0.25)
    s2 = SVI(0.05, 0.2, -0.4, 0.0, 0.1, T=1.0)
    assert calendar_arbitrage([s1, s2]) == []


def test_calendaire_detecte_croisement():
    s1 = SVI(0.06, 0.2, -0.4, 0.0, 0.1, T=0.25)   # plus de variance à échéance courte
    s2 = SVI(0.03, 0.2, -0.4, 0.0, 0.1, T=1.0)
    v = calendar_arbitrage([s1, s2])
    assert len(v) == 1
    t1, t2, _, ecart = v[0]
    assert (t1, t2) == (0.25, 1.0) and ecart == pytest.approx(0.03, abs=1e-9)


def test_calendaire_independant_de_l_ordre_de_la_liste():
    s1 = SVI(0.06, 0.2, -0.4, 0.0, 0.1, T=0.25)
    s2 = SVI(0.03, 0.2, -0.4, 0.0, 0.1, T=1.0)
    assert calendar_arbitrage([s2, s1]) == calendar_arbitrage([s1, s2])


def test_calendaire_plages_disjointes_ignorees():
    s1 = SVI(0.06, 0.2, -0.4, 0.0, 0.1, T=0.25)
    s2 = SVI(0.03, 0.2, -0.4, 0.0, 0.1, T=1.0)
    assert calendar_arbitrage([s1, s2], plages=[(-0.5, -0.2), (0.1, 0.4)]) == []


# --------------------------------------------------------------------------- #
# Calage
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("vrai", [
    SVI(0.010, 0.10, -0.60, 0.02, 0.15, T=0.25),
    SVI(0.030, 0.15, -0.40, 0.00, 0.20, T=1.00),
    SVI(0.002, 0.05, -0.20, 0.01, 0.08, T=0.05),
])
def test_calage_retrouve_les_vols_d_un_svi_connu(vrai):
    k = np.linspace(-0.4, 0.3, 25)
    cale = calibrate_svi(k, vrai.vol(k), vrai.T)
    np.testing.assert_allclose(cale.vol(k), vrai.vol(k), atol=5e-4)   # < 0,05 pt de vol
    assert butterfly_arbitrage(cale)[0]


def test_calage_bruite_reste_sans_arbitrage_et_respecte_lee():
    rng = np.random.default_rng(1)
    vrai = SVI(0.03, 0.15, -0.4, 0.0, 0.2, T=0.5)
    k = np.linspace(-0.5, 0.35, 30)
    iv = vrai.vol(k) + rng.normal(0, 0.003, k.size)
    cale = calibrate_svi(k, iv, vrai.T)
    assert butterfly_arbitrage(cale)[0]
    assert cale.b * (1 + abs(cale.rho)) <= 4 / cale.T + 1e-6
    assert cale.min_variance() >= -1e-6
    assert np.sqrt(np.mean((cale.vol(k) - vrai.vol(k)) ** 2)) < 0.003


def test_calage_poids_privilegient_les_points_lourds():
    vrai = SVI(0.03, 0.15, -0.4, 0.0, 0.2, T=0.5)
    k = np.linspace(-0.4, 0.3, 15)
    iv = vrai.vol(k).copy()
    iv[0] += 0.02                                    # point aberrant dans l'aile gauche
    poids = np.ones_like(k); poids[0] = 1e-3
    cale = calibrate_svi(k, iv, vrai.T, poids=poids)
    assert abs(cale.vol(k[0]) - vrai.vol(k[0])) < 0.005


def test_calage_reproductible():
    k = np.linspace(-0.4, 0.3, 20)
    iv = SAIN.vol(k)
    assert calibrate_svi(k, iv, 1.0, seed=3) == calibrate_svi(k, iv, 1.0, seed=3)


# --------------------------------------------------------------------------- #
# Propriété : toute tranche SVI respectant la condition suffisante de
# Gatheral-Jacquier (b(1+|ρ|) ≤ 4/T et a ≥ 0 « modéré ») n'a pas de variance < 0
# --------------------------------------------------------------------------- #
@settings(max_examples=200, deadline=None)
@given(a=st.floats(0.0, 0.1), b=st.floats(0.0, 1.0), rho=st.floats(-0.99, 0.99),
       m=st.floats(-0.5, 0.5), s=st.floats(1e-3, 1.0))
def test_variance_positive_si_a_positif(a, b, rho, m, s):
    sv = SVI(a, b, rho, m, s, T=1.0)
    assert sv.min_variance() >= -1e-12
    assert (sv.w(np.linspace(-3, 3, 61)) >= -1e-12).all()


if __name__ == "__main__":
    # Lancement direct (python fichier.py / bouton ▶ de VS Code) -> on passe par pytest
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
