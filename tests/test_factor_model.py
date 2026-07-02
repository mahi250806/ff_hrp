import numpy as np
import pandas as pd
import pytest
from ffhrp.factor_model import fit_factor_model


def _synthetic_data(n_obs=500, seed=42):
    """
    Generate synthetic returns with known betas.
    Returns excess_returns (T x 2) and factors (T x 2) so
    we can verify beta recovery.
    """
    rng = np.random.default_rng(seed)
    TRUE_BETAS = {
        "A": np.array([1.2, 0.3]),
        "B": np.array([0.5, -0.4]),
    }
    TRUE_ALPHA = {"A": 0.0001, "B": -0.0002}

    f1 = rng.normal(0, 0.01, n_obs)
    f2 = rng.normal(0, 0.008, n_obs)
    factors = pd.DataFrame({"F1": f1, "F2": f2})
    factors.index = pd.date_range("2022-01-01", periods=n_obs, freq="B")

    returns = {}
    for name, betas in TRUE_BETAS.items():
        noise = rng.normal(0, 0.005, n_obs)
        ret = TRUE_ALPHA[name] + betas[0] * f1 + betas[1] * f2 + noise
        returns[name] = ret

    excess = pd.DataFrame(returns, index=factors.index)
    return excess, factors, TRUE_BETAS, TRUE_ALPHA


def test_beta_recovery():
    excess, factors, true_betas, true_alpha = _synthetic_data()
    result = fit_factor_model(excess, factors, min_obs_ratio=5)

    for asset, expected in true_betas.items():
        estimated = result.betas.loc[asset].values
        np.testing.assert_allclose(estimated, expected, atol=0.05,
                                   err_msg=f"Beta mismatch for {asset}")


def test_alpha_recovery():
    excess, factors, true_betas, true_alpha = _synthetic_data()
    result = fit_factor_model(excess, factors, min_obs_ratio=5)

    for asset, expected in true_alpha.items():
        estimated = float(result.alphas.loc[asset])
        assert abs(estimated - expected) < 1e-3, (
            f"Alpha mismatch for {asset}: got {estimated:.6f}, expected {expected}"
        )


def test_r2_in_range():
    excess, factors, _, _ = _synthetic_data()
    result = fit_factor_model(excess, factors, min_obs_ratio=5)
    assert (result.r2 >= 0).all() and (result.r2 <= 1).all()


def test_residuals_shape():
    excess, factors, _, _ = _synthetic_data()
    result = fit_factor_model(excess, factors, min_obs_ratio=5)
    assert result.residuals.shape[0] == len(excess)
    assert set(result.residuals.columns) >= set(excess.columns)


def test_min_obs_guard():
    """Assets with too few observations must be flagged, not crash."""
    excess, factors, _, _ = _synthetic_data(n_obs=10)
    # min_obs_ratio=20, K=2 → needs 40 obs; we only have 10
    result = fit_factor_model(excess, factors, min_obs_ratio=20)
    assert "A" in result.flagged or "B" in result.flagged


def test_residual_variance_positive():
    excess, factors, _, _ = _synthetic_data()
    result = fit_factor_model(excess, factors, min_obs_ratio=5)
    assert (result.residual_variances > 0).all()
