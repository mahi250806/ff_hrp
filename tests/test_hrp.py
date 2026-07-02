import numpy as np
import pandas as pd
import pytest
from ffhrp.hrp import hrp


def _random_corr(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    cov = A @ A.T + np.eye(n) * 0.5
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    assets = [f"A{i}" for i in range(n)]
    return pd.DataFrame(corr, index=assets, columns=assets), pd.DataFrame(cov, index=assets, columns=assets)


def test_weights_sum_to_one():
    corr, cov = _random_corr(8)
    result = hrp(corr, cov)
    assert abs(result.weights.sum() - 1.0) < 1e-10


def test_weights_nonnegative():
    corr, cov = _random_corr(8)
    result = hrp(corr, cov)
    assert (result.weights >= 0).all()


def test_all_assets_allocated():
    corr, cov = _random_corr(8)
    result = hrp(corr, cov)
    assert set(result.weights.index) == set(corr.index)


def test_degenerate_single_asset():
    corr = pd.DataFrame([[1.0]], index=["X"], columns=["X"])
    cov = pd.DataFrame([[0.04]], index=["X"], columns=["X"])
    result = hrp(corr, cov)
    assert abs(result.weights["X"] - 1.0) < 1e-10
    assert result.weights.sum() == pytest.approx(1.0)


def test_equal_weight_is_1_over_n():
    n = 6
    corr, cov = _random_corr(n)
    result = hrp(corr, cov)
    np.testing.assert_allclose(result.equal_weights.values, 1.0 / n)


def test_linkage_matrix_shape():
    n = 7
    corr, cov = _random_corr(n)
    result = hrp(corr, cov)
    assert result.linkage_matrix.shape == (n - 1, 4)


def test_quasi_diag_order_is_permutation():
    n = 5
    corr, cov = _random_corr(n)
    result = hrp(corr, cov)
    assert sorted(result.quasi_diag_order) == list(range(n))


def test_corr_reordered_shape():
    n = 6
    corr, cov = _random_corr(n)
    result = hrp(corr, cov)
    assert result.corr_reordered.shape == (n, n)


def test_large_basket():
    """Verify no crash and valid weights for a larger basket."""
    corr, cov = _random_corr(20, seed=99)
    result = hrp(corr, cov)
    assert abs(result.weights.sum() - 1.0) < 1e-9
    assert (result.weights >= 0).all()
