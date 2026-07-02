import numpy as np
import pandas as pd
import pytest
from ffhrp.factor_model import fit_factor_model
from ffhrp.covariance import build_covariance


def _make_fm_and_factors(n_obs=400, seed=7):
    rng = np.random.default_rng(seed)
    f1 = rng.normal(0, 0.01, n_obs)
    f2 = rng.normal(0, 0.008, n_obs)
    idx = pd.date_range("2022-01-01", periods=n_obs, freq="B")
    factors = pd.DataFrame({"F1": f1, "F2": f2}, index=idx)

    returns = {}
    for name, b in [("X", [1.0, 0.5]), ("Y", [0.3, -0.2]), ("Z", [0.8, 0.1])]:
        noise = rng.normal(0, 0.005, n_obs)
        returns[name] = b[0] * f1 + b[1] * f2 + noise

    excess = pd.DataFrame(returns, index=idx)
    fm = fit_factor_model(excess, factors, min_obs_ratio=5)
    return fm, factors


def test_sigma_symmetric():
    fm, factors = _make_fm_and_factors()
    cov = build_covariance(fm, factors, annualize=252)
    np.testing.assert_allclose(
        cov.sigma.values, cov.sigma.values.T, atol=1e-10
    )


def test_sigma_positive_semidefinite():
    fm, factors = _make_fm_and_factors()
    cov = build_covariance(fm, factors, annualize=252)
    eigvals = np.linalg.eigvalsh(cov.sigma.values)
    assert (eigvals >= -1e-9).all(), f"Sigma has negative eigenvalue: {eigvals.min()}"


def test_correlation_diagonal_ones():
    fm, factors = _make_fm_and_factors()
    cov = build_covariance(fm, factors, annualize=252)
    np.testing.assert_allclose(np.diag(cov.correlation.values), 1.0, atol=1e-10)


def test_correlation_in_range():
    fm, factors = _make_fm_and_factors()
    cov = build_covariance(fm, factors, annualize=252)
    assert cov.correlation.values.min() >= -1.0 - 1e-9
    assert cov.correlation.values.max() <= 1.0 + 1e-9


def test_sigma_decomposes_correctly():
    """Sigma = B F B^T + D — verify the arithmetic is correct."""
    fm, factors = _make_fm_and_factors()
    ann = 252
    cov = build_covariance(fm, factors, annualize=ann)

    B = fm.betas[fm.factor_cols].values
    F_ann = factors[fm.factor_cols].cov().values * ann
    D = fm.residual_variances.values * ann

    expected = B @ F_ann @ B.T + np.diag(D)
    expected = (expected + expected.T) / 2  # symmetrize
    np.testing.assert_allclose(cov.sigma.values, expected, atol=1e-12)


def test_factor_cov_symmetric():
    fm, factors = _make_fm_and_factors()
    cov = build_covariance(fm, factors, annualize=252)
    np.testing.assert_allclose(
        cov.factor_cov.values, cov.factor_cov.values.T, atol=1e-12
    )
