"""Shared fixtures and helpers.

Monte Carlo testing philosophy:
  - FIXED seeds -> deterministic tests (never a "flaky" test in CI);
  - an estimator is judged correct if the exact reference is within
    N_SIGMA standard deviations (4σ: false failure ~ 6e-5 if the seed changed);
  - variance reduction (std_error) is tested too, not only the price.
"""
import math
import sys
from pathlib import Path

# pricer.py / greeks.py importable however the tests are launched
_SRC = str(Path(__file__).resolve().parents[1])  # project root folder
if _SRC in sys.path:
    sys.path.remove(_SRC)
sys.path.insert(0, _SRC)  # first position: takes priority over any other "pricer" module

import pytest

from pricer import BlackScholes

N_SIGMA = 4.0


@pytest.fixture
def model() -> BlackScholes:
    return BlackScholes(s0=100.0, r=0.05, sigma=0.2)


@pytest.fixture
def model_div() -> BlackScholes:
    return BlackScholes(s0=100.0, r=0.03, sigma=0.3, q=0.02)


def assert_mc_close(res, ref: float, n_sigma: float = N_SIGMA, atol: float = 0.0) -> None:
    """|estimate - reference| <= n_sigma * std_error + atol."""
    err = abs(res.price - float(ref))
    tol = n_sigma * res.std_error + atol
    assert err <= tol, f"estimate={res.price:.6f} ref={float(ref):.6f} " \
                       f"diff={err:.2e} > tolerance={tol:.2e} (se={res.std_error:.2e})"


def fd(f, x: float, h: float) -> float:
    """Central derivative."""
    return (f(x + h) - f(x - h)) / (2 * h)


def fd2(f, x: float, h: float) -> float:
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h * h)


def isclose(a, b, rel=1e-6, abs_=1e-8) -> bool:
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)
