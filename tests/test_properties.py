"""
Theoretical properties and internal consistency of pricer.py, greeks.py and dividends.py.

No external reference here (QuantLib stays in the validate_*.py scripts):
this tests what must hold for ANY reasonable set of parameters
(arbitrage bounds, parities, monotonicity, consistency between methods), with
hypothesis to sweep the parameter space.

Run without the heavy Monte Carlo tests: python -m pytest -m "not slow"
"""
import sys
from pathlib import Path
# Direct launch (VS Code ▶ button): makes the project root modules importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import math
import warnings

import numpy as np
import pytest
from hypothesis import assume, given, settings, strategies as st

from dividends import crr_american_div, american_price_div
from greeks import (GREEKS, bs_greeks, crr_american, fd_greeks_crn,
                    mc_asian_greeks, mc_european_greeks)
from pricer import (BlackScholes, _RunningStats, bs_delta_vega, bs_price,
                    check_proba_crr, geometric_asian_price, lsm_american_put,
                    mc_asian_arithmetic, mc_european, mc_greeks_pathwise,
                    rqmc_european, sigma_min_crr)

N_ARBRE = 300          # tree steps in the property tests (fast, accuracy ~1e-2)
TOL_ARBRE = 0.03       # discretisation error tolerated on a price (S ≈ 100)

# Reasonable "equity market" parameter space
s_K = st.floats(70.0, 140.0)
s_T = st.floats(0.05, 2.0)
s_sig = st.floats(0.08, 0.70)
s_r = st.floats(0.0, 0.08)
s_q = st.floats(0.0, 0.05)
s_kind = st.sampled_from(["call", "put"])
PROPS = settings(max_examples=40, deadline=None)


# =========================================================================== #
# 1. Model and utilities
# =========================================================================== #
@pytest.mark.parametrize("kw", [dict(s0=0.0, r=0.0, sigma=0.2),
                                dict(s0=100.0, r=0.0, sigma=0.0),
                                dict(s0=-1.0, r=0.0, sigma=0.2)])
def test_modele_refuse_parametres_invalides(kw):
    with pytest.raises(ValueError):
        BlackScholes(**kw)


def test_running_stats_par_blocs_egal_numpy():
    x = np.random.default_rng(0).normal(3.0, 2.0, 10_007)
    rs = _RunningStats()
    for bloc in np.array_split(x, 13):          # blocks of unequal sizes
        rs.update(bloc)
    res = rs.result()
    assert res.price == pytest.approx(x.mean(), rel=1e-12)
    assert res.std_error == pytest.approx(x.std(ddof=1) / math.sqrt(x.size), rel=1e-10)
    assert res.n_paths == x.size


def test_mcresult_intervalle_de_confiance():
    res = _RunningStats(); res.update(np.array([1.0, 2.0, 3.0, 4.0]))
    r = res.result()
    lo, hi = r.ci95
    assert lo < r.price < hi
    assert hi - r.price == pytest.approx(1.959963984540054 * r.std_error)


# =========================================================================== #
# 2. Closed-form Black-Scholes
# =========================================================================== #
@PROPS
@given(K=s_K, T=s_T, sig=s_sig, r=s_r, q=s_q)
def test_parite_call_put_europeenne(K, T, sig, r, q):
    m = BlackScholes(100.0, r, sig, q)
    c, p = bs_price(m, K, T, "call"), bs_price(m, K, T, "put")
    assert c - p == pytest.approx(100 * math.exp(-q * T) - K * math.exp(-r * T), abs=1e-9)


@PROPS
@given(K=s_K, T=s_T, sig=s_sig, r=s_r, q=s_q, kind=s_kind)
def test_bornes_europeennes(K, T, sig, r, q, kind):
    m = BlackScholes(100.0, r, sig, q)
    v = float(bs_price(m, K, T, kind))
    fs, fk = 100 * math.exp(-q * T), K * math.exp(-r * T)
    if kind == "call":
        assert max(fs - fk, 0) - 1e-9 <= v <= fs + 1e-9
    else:
        assert max(fk - fs, 0) - 1e-9 <= v <= fk + 1e-9


@PROPS
@given(K=s_K, T=s_T, sig=s_sig, r=s_r, q=s_q, kind=s_kind)
def test_bs_greeks_egales_differences_finies(K, T, sig, r, q, kind):
    from dataclasses import replace
    m = BlackScholes(100.0, r, sig, q)
    g = {k: float(v) for k, v in bs_greeks(m, K, T, kind).items()}
    P = lambda mm=m, TT=T: float(bs_price(mm, K, TT, kind))
    h = 1e-4
    assert g["price"] == pytest.approx(P(), abs=1e-12)
    fd = {
        "delta": (P(replace(m, s0=100 + h)) - P(replace(m, s0=100 - h))) / (2 * h),
        "gamma": (P(replace(m, s0=100 + 1e-2)) - 2 * P() + P(replace(m, s0=100 - 1e-2))) / 1e-4,
        "vega": (P(replace(m, sigma=sig + h)) - P(replace(m, sigma=sig - h))) / (2 * h),
        "rho": (P(replace(m, r=r + h)) - P(replace(m, r=r - h))) / (2 * h),
        "theta": -(P(TT=T + h) - P(TT=T - h)) / (2 * h),
    }
    for k, v in fd.items():
        assert g[k] == pytest.approx(v, rel=1e-4, abs=1e-5), k


def test_bs_delta_vega_coherent_avec_bs_greeks():
    m = BlackScholes(100.0, 0.03, 0.25, 0.01)
    for kind in ("call", "put"):
        d, v = bs_delta_vega(m, 105.0, 0.7, kind)
        g = bs_greeks(m, 105.0, 0.7, kind)
        assert float(d) == pytest.approx(float(g["delta"]))
        assert float(v) == pytest.approx(float(g["vega"]))


def test_bs_price_vectorise():
    m = BlackScholes(100.0, 0.05, 0.2)
    Ks = np.array([80.0, 100.0, 120.0])
    vec = bs_price(m, Ks, 1.0)
    assert vec.shape == (3,)
    np.testing.assert_allclose(vec, [float(bs_price(m, k, 1.0)) for k in Ks])


@pytest.mark.parametrize("kind", ["call", "put"])
def test_asiatique_geometrique_une_date_egale_black_scholes(kind):
    """With n = 1, the geometric mean is S_T: this recovers Black-Scholes."""
    m = BlackScholes(100.0, 0.04, 0.3, 0.01)
    assert geometric_asian_price(m, 95.0, 0.8, 1, kind) == pytest.approx(
        float(bs_price(m, 95.0, 0.8, kind)), rel=1e-12)


def test_asiatique_geometrique_moins_chere_que_europeenne():
    """Averaging reduces variance: geometric Asian call < ATM European call."""
    m = BlackScholes(100.0, 0.05, 0.3)
    assert geometric_asian_price(m, 100.0, 1.0, 52) < float(bs_price(m, 100.0, 1.0))


# =========================================================================== #
# 3. CRR tree (greeks.crr_american) — fix: probability in (0, 1)
# =========================================================================== #
def test_crr_refuse_probabilite_hors_bornes():
    """Regression: very low vol + large r - q gave p ≈ 2.9 and a wrong price."""
    m = BlackScholes(s0=100.0, r=0.04, sigma=0.005, q=-0.40)
    with pytest.raises(ValueError, match="outside"):
        crr_american(m, 100.0, 1.2, "put", n=400)


def test_crr_refuse_probabilite_negative():
    m = BlackScholes(s0=100.0, r=0.0, sigma=0.005, q=0.40)
    with pytest.raises(ValueError):
        crr_american(m, 100.0, 1.2, "call", n=400)


@pytest.mark.parametrize("r, q", [(0.04, -0.40), (0.0, 0.40), (0.05, 0.0)])
def test_seuil_sigma_min_crr_exact(r, q):
    """Just above the threshold: accepted; just below: rejected."""
    T, n = 1.2, 400
    s_min = sigma_min_crr(r, q, T, n)
    dt = T / n

    def p_de(sig):
        u = math.exp(sig * math.sqrt(dt)); d = 1 / u
        return (math.exp((r - q) * dt) - d) / (u - d)

    check_proba_crr(p_de(s_min * 1.01), BlackScholes(100, r, s_min * 1.01, q), dt)
    with pytest.raises(ValueError):
        check_proba_crr(p_de(s_min * 0.99), BlackScholes(100, r, s_min * 0.99, q), dt)


@PROPS
@given(K=s_K, T=s_T, sig=s_sig, r=s_r, q=s_q)
def test_put_americain_entre_europeen_et_K(K, T, sig, r, q):
    m = BlackScholes(100.0, r, sig, q)
    am = crr_american(m, K, T, "put", n=N_ARBRE)["price"]
    eu = float(bs_price(m, K, T, "put"))
    assert am >= eu - TOL_ARBRE
    assert am >= max(K - 100.0, 0.0) - 1e-9          # exact: the tree enforces intrinsic value
    assert am <= K + 1e-9


@PROPS
@given(K=s_K, T=s_T, sig=s_sig, r=s_r)
def test_call_americain_sans_dividende_egal_europeen(K, T, sig, r):
    """Merton: without dividends (q = 0, r ≥ 0), early exercise of a call is suboptimal."""
    m = BlackScholes(100.0, r, sig)
    am = crr_american(m, K, T, "call", n=N_ARBRE)["price"]
    assert am == pytest.approx(float(bs_price(m, K, T, "call")), abs=TOL_ARBRE)


@PROPS
@given(K=s_K, T=s_T, sig=s_sig, r=s_r, q=s_q)
def test_parite_americaine(K, T, sig, r, q):
    """S e^{-qT} - K ≤ C - P ≤ S - K e^{-rT} (American put-call parity bounds)."""
    m = BlackScholes(100.0, r, sig, q)
    c = crr_american(m, K, T, "call", n=N_ARBRE)["price"]
    p = crr_american(m, K, T, "put", n=N_ARBRE)["price"]
    assert 100 * math.exp(-q * T) - K - TOL_ARBRE <= c - p <= 100 - K * math.exp(-r * T) + TOL_ARBRE


@PROPS
@given(K=s_K, T=s_T, sig=s_sig, r=s_r, q=s_q, kind=s_kind)
def test_crr_grecques_signes_et_bornes(K, T, sig, r, q, kind):
    g = crr_american(BlackScholes(100.0, r, sig, q), K, T, kind, n=N_ARBRE)
    if kind == "call":
        assert -1e-9 <= g["delta"] <= 1 + 1e-9
    else:
        assert -1 - 1e-9 <= g["delta"] <= 1e-9
    assert g["gamma"] >= -1e-6
    assert g["vega"] >= -1e-6


@settings(max_examples=25, deadline=None)
@given(K=s_K, T=s_T, r=s_r, q=s_q, kind=s_kind)
def test_crr_croissant_en_volatilite(K, T, r, q, kind):
    prix = [crr_american(BlackScholes(100.0, r, s, q), K, T, kind, n=N_ARBRE)["price"]
            for s in (0.15, 0.30, 0.50)]
    assert prix[0] <= prix[1] + 1e-6 <= prix[2] + 2e-6


@settings(max_examples=20, deadline=None)
@given(T=s_T, sig=s_sig, r=s_r, q=s_q, kind=s_kind)
def test_crr_convexe_et_monotone_en_strike(T, sig, r, q, kind):
    m = BlackScholes(100.0, r, sig, q)
    Ks = np.arange(80.0, 125.0, 5.0)
    p = np.array([crr_american(m, k, T, kind, n=N_ARBRE)["price"] for k in Ks])
    assert (p[:-2] - 2 * p[1:-1] + p[2:]).min() > -TOL_ARBRE
    d = np.diff(p)
    assert (d <= TOL_ARBRE).all() if kind == "call" else (d >= -TOL_ARBRE).all()


def test_crr_converge_quand_n_augmente():
    m = BlackScholes(100.0, 0.05, 0.3)
    ecarts = [abs(crr_american(m, 100.0, 1.0, "call", n=n)["price"]
                  - float(bs_price(m, 100.0, 1.0, "call"))) for n in (50, 200, 800)]
    assert ecarts[2] < ecarts[0]
    assert ecarts[2] < 0.01


def test_crr_grecques_coherentes_avec_reprice():
    """Delta and gamma read from the tree ≈ finite differences of the tree price."""
    from dataclasses import replace
    m, K, T = BlackScholes(100.0, 0.05, 0.25, 0.01), 100.0, 0.5
    g = crr_american(m, K, T, "put", n=1000)
    P = lambda s: crr_american(replace(m, s0=s), K, T, "put", n=1000)["price"]
    h = 2.0
    assert g["delta"] == pytest.approx((P(100 + h) - P(100 - h)) / (2 * h), abs=5e-3)
    assert g["gamma"] == pytest.approx((P(100 + 5) - 2 * g["price"] + P(100 - 5)) / 25, rel=0.05)


# =========================================================================== #
# 4. Discrete dividends (dividends.py) — fix: early ex-date
# =========================================================================== #
M_DIV = BlackScholes(100.0, 0.04, 0.25)


def test_dividende_dans_les_deux_premiers_pas_signale():
    with pytest.warns(UserWarning, match="first two tree steps"):
        crr_american_div(M_DIV, 100, 1.0, "put", [(0.5 / 2000, 1.0)])


def test_pas_d_avertissement_pour_un_dividende_normal():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        crr_american_div(M_DIV, 100, 1.0, "put", [(0.3, 1.0)], n=N_ARBRE)


def test_dividendes_hors_echeance_ou_passes_ignores():
    ref = american_price_div(100, 100, 1.0, 0.04, 0, 0.25, "put", [], n=N_ARBRE)
    for divs in ([(-0.1, 2.0)], [(1.5, 2.0)], [(0.3, 0.0)]):
        assert american_price_div(100, 100, 1.0, 0.04, 0, 0.25, "put", divs, n=N_ARBRE) \
            == pytest.approx(ref, abs=1e-12)


def test_escrowed_refuse_dividendes_superieurs_au_cours():
    with pytest.raises(ValueError):
        crr_american_div(M_DIV, 100, 1.0, "put", [(0.5, 150.0)], n=N_ARBRE, model="escrowed")


def test_arbre_div_refuse_probabilite_hors_bornes():
    with pytest.raises(ValueError):
        american_price_div(100, 100, 1.2, 0.04, -0.40, 0.005, "put", [(0.5, 1.0)], n=400)


@settings(max_examples=25, deadline=None)
@given(K=s_K, t_div=st.floats(0.1, 0.9), D1=st.floats(0.0, 3.0), dD=st.floats(0.5, 3.0),
       sig=s_sig, model=st.sampled_from(["spot", "escrowed"]))
def test_call_baisse_put_monte_avec_le_dividende(K, t_div, D1, dD, sig, model):
    p = lambda kind, D: american_price_div(100, K, 1.0, 0.04, 0.0, sig, kind,
                                           [(t_div, D)], n=N_ARBRE, model=model)
    assert p("call", D1 + dD) <= p("call", D1) + 1e-6
    assert p("put", D1 + dD) >= p("put", D1) - 1e-6


@settings(max_examples=25, deadline=None)
@given(K=s_K, sig=s_sig, kind=s_kind, model=st.sampled_from(["spot", "escrowed"]),
       D=st.floats(0.5, 5.0), t_div=st.floats(0.05, 0.95))
def test_dividende_au_dessus_de_l_intrinseque(K, sig, kind, model, D, t_div):
    v = american_price_div(100, K, 1.0, 0.04, 0.0, sig, kind, [(t_div, D)],
                           n=N_ARBRE, model=model)
    assert v >= max((100 - K) if kind == "call" else (K - 100), 0.0) - 1e-9


@pytest.mark.parametrize("model", ["spot", "escrowed"])
def test_prix_seul_coherent_avec_crr_american_div(model):
    divs = [(0.3, 1.5), (0.8, 1.5)]
    a = american_price_div(100, 105, 1.0, 0.04, 0.01, 0.3, "put", divs, n=500, model=model)
    b = crr_american_div(BlackScholes(100, 0.04, 0.3, 0.01), 105, 1.0, "put", divs,
                         n=500, model=model)["price"]
    assert a == pytest.approx(b, abs=1e-12)


def test_spot_et_escrowed_proches_pour_petits_dividendes():
    divs = [(0.25, 0.2), (0.75, 0.2)]
    s = american_price_div(100, 100, 1.0, 0.04, 0, 0.25, "put", divs, n=800, model="spot")
    e = american_price_div(100, 100, 1.0, 0.04, 0, 0.25, "put", divs, n=800, model="escrowed")
    assert s == pytest.approx(e, abs=0.02)


# =========================================================================== #
# 5. CRN finite differences (universal method)
# =========================================================================== #
@pytest.mark.parametrize("kind", ["call", "put"])
def test_fd_greeks_crn_sur_pricer_exact(kind):
    """Plugged into the closed form, fd_greeks_crn must recover bs_greeks."""
    m, K, T = BlackScholes(100.0, 0.03, 0.25, 0.01), 100.0, 1.0
    fd = fd_greeks_crn(lambda mm, TT: float(bs_price(mm, K, TT, kind)), m, T,
                       d_sigma=1e-3, d_r=1e-4, d_t=1e-3)
    ex = bs_greeks(m, K, T, kind)
    for g in GREEKS:
        assert fd[g] == pytest.approx(float(ex[g]), rel=2e-3, abs=1e-4), g


# =========================================================================== #
# 6. Monte Carlo (fixed seeds: deterministic tests)
# =========================================================================== #
CAS_MC = [(100.0, 1.0, 0.2, 0.05, 0.0), (90.0, 0.5, 0.35, 0.02, 0.03),
          (120.0, 2.0, 0.25, 0.04, 0.01)]


@pytest.mark.parametrize("K, T, sig, r, q", CAS_MC)
@pytest.mark.parametrize("kind", ["call", "put"])
def test_mc_european_dans_l_intervalle_de_confiance(K, T, sig, r, q, kind):
    m = BlackScholes(100.0, r, sig, q)
    res = mc_european(m, K, T, kind, n_paths=400_000)
    assert abs(res.price - float(bs_price(m, K, T, kind))) < 4 * res.std_error


@pytest.mark.parametrize("K, T, sig, r, q", CAS_MC)
def test_rqmc_european_dans_l_intervalle_de_confiance(K, T, sig, r, q):
    m = BlackScholes(100.0, r, sig, q)
    res = rqmc_european(m, K, T, log2_n=14, n_replicas=16)
    assert abs(res.price - float(bs_price(m, K, T))) < 4 * res.std_error


def test_reduction_de_variance_effective():
    m = BlackScholes(100.0, 0.05, 0.2)
    brut = mc_european(m, 100, 1, n_paths=200_000, antithetic=False, control_variate=False)
    anti = mc_european(m, 100, 1, n_paths=200_000, control_variate=False)
    cv = mc_european(m, 100, 1, n_paths=200_000)
    assert cv.std_error < anti.std_error < brut.std_error
    assert cv.std_error < 0.5 * brut.std_error


def test_mc_reproductible_avec_meme_seed():
    m = BlackScholes(100.0, 0.05, 0.2)
    assert mc_european(m, 100, 1, n_paths=50_000, seed=5) == mc_european(m, 100, 1, n_paths=50_000, seed=5)


def test_mc_greeks_pathwise_vs_formule_fermee():
    m = BlackScholes(100.0, 0.05, 0.2, 0.01)
    est = mc_greeks_pathwise(m, 100.0, 1.0, n_paths=400_000)
    ex = bs_greeks(m, 100.0, 1.0)
    for g in ("delta", "vega"):
        assert abs(est[g].price - float(ex[g])) < 4 * est[g].std_error, g


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["call", "put"])
def test_mc_european_greeks_vs_formule_fermee(kind):
    m = BlackScholes(100.0, 0.05, 0.2, 0.01)
    est = mc_european_greeks(m, 100.0, 1.0, kind, n_paths=1_000_000)
    ex = bs_greeks(m, 100.0, 1.0, kind)
    for g in GREEKS:
        assert abs(est[g].price - float(ex[g])) < 4 * est[g].std_error + 1e-9, g


@pytest.mark.slow
def test_asiatique_arithmetique_au_dessus_de_la_geometrique():
    """AM-GM inequality: A ≥ G path by path, hence call(A) ≥ call(G)."""
    m = BlackScholes(100.0, 0.05, 0.25)
    res = mc_asian_arithmetic(m, 100.0, 1.0, n_paths=200_000)
    assert res.price >= geometric_asian_price(m, 100.0, 1.0, 52) - 3 * res.std_error


@pytest.mark.slow
def test_asiatique_controle_sans_biais_et_plus_precis():
    m = BlackScholes(100.0, 0.05, 0.25)
    sans = mc_asian_arithmetic(m, 100.0, 1.0, n_paths=200_000, control_variate=False, seed=2)
    avec = mc_asian_arithmetic(m, 100.0, 1.0, n_paths=200_000, seed=2)
    assert avec.std_error < 0.2 * sans.std_error
    assert abs(avec.price - sans.price) < 4 * math.hypot(avec.std_error, sans.std_error)


@pytest.mark.slow
def test_asiatique_grecques_controle_coherentes():
    m = BlackScholes(100.0, 0.05, 0.25)
    sans = mc_asian_greeks(m, 100.0, 1.0, n_paths=200_000, control_variate=False)
    avec = mc_asian_greeks(m, 100.0, 1.0, n_paths=200_000)
    for g in ("price", "delta", "vega", "rho"):
        assert abs(avec[g].price - sans[g].price) < 4 * math.hypot(avec[g].std_error, sans[g].std_error), g
    assert avec["delta"].price == pytest.approx(0.6, abs=0.1)        # order of magnitude


@pytest.mark.slow
@pytest.mark.parametrize("K, sig, q", [(100.0, 0.2, 0.0), (110.0, 0.35, 0.02), (90.0, 0.3, 0.0)])
def test_lsm_encadre_par_europeen_et_arbre(K, sig, q):
    m = BlackScholes(100.0, 0.05, sig, q)
    res = lsm_american_put(m, K, 1.0, n_paths=200_000)
    eu = float(bs_price(m, K, 1.0, "put"))
    arbre = crr_american(m, K, 1.0, "put", n=1000)["price"]
    assert res.price >= eu - 3 * res.std_error
    assert res.price <= arbre + 3 * res.std_error
    assert res.price == pytest.approx(arbre, abs=0.05)


def test_lsm_put_tres_dans_la_monnaie_vaut_intrinseque():
    res = lsm_american_put(BlackScholes(100.0, 0.08, 0.2), 200.0, 0.5, n_paths=20_000)
    assert res.price == pytest.approx(100.0, abs=1e-9)


if __name__ == "__main__":
    # Direct launch (python file.py / VS Code ▶ button) -> go through pytest
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
