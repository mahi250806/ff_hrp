from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


@dataclass
class HRPResult:
    weights: pd.Series           # final HRP weights, sums to 1
    equal_weights: pd.Series     # 1/N benchmark
    linkage_matrix: np.ndarray   # scipy linkage output (N-1 x 4)
    quasi_diag_order: list       # integer indices into assets
    corr_reordered: pd.DataFrame


def _cluster_var(cov: np.ndarray, items: list) -> float:
    """Variance of the minimum-variance portfolio within a cluster."""
    sub = cov[np.ix_(items, items)]
    inv = np.linalg.pinv(sub)
    w = inv.sum(axis=1)
    w /= w.sum()
    return float(w @ sub @ w)


def _recursive_bisect(cov: np.ndarray, sorted_items: list) -> pd.Series:
    """
    Recursive bisection step of HRP.
    Returns a Series keyed by integer asset indices with allocation weights.
    """
    weights = pd.Series(1.0, index=sorted_items, dtype=float)
    clusters = [sorted_items]

    while clusters:
        next_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            mid = len(cluster) // 2
            left, right = cluster[:mid], cluster[mid:]

            v_left = _cluster_var(cov, left)
            v_right = _cluster_var(cov, right)

            # alpha = weight assigned to left cluster
            alpha = 1.0 - v_left / (v_left + v_right)
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha

            next_clusters.extend([left, right])
        clusters = next_clusters

    return weights


def hrp(correlation: pd.DataFrame, covariance: pd.DataFrame) -> HRPResult:
    """
    Hierarchical Risk Parity.

    1. Convert correlation -> distance matrix.
    2. Single-linkage clustering -> quasi-diagonal ordering.
    3. Recursive bisection on the covariance matrix -> weights.

    Parameters
    ----------
    correlation : N x N correlation matrix (from factor model)
    covariance  : N x N covariance matrix (annualized, from factor model)
    """
    assets = correlation.index.tolist()
    n = len(assets)

    if n == 1:
        return HRPResult(
            weights=pd.Series([1.0], index=assets),
            equal_weights=pd.Series([1.0], index=assets),
            linkage_matrix=np.zeros((0, 4)),
            quasi_diag_order=[0],
            corr_reordered=correlation.copy(),
        )

    corr_arr = correlation.values
    # Distance: d_ij = sqrt(0.5 * (1 - rho_ij)), lies in [0, 1]
    dist_arr = np.sqrt(np.clip((1.0 - corr_arr) / 2.0, 0.0, 1.0))
    np.fill_diagonal(dist_arr, 0.0)

    condensed = squareform(dist_arr, checks=False)
    link = linkage(condensed, method="single")

    order = list(leaves_list(link))  # quasi-diagonal ordering (integer indices)

    cov_arr = covariance.values
    raw_weights = _recursive_bisect(cov_arr, order)

    # Map integer indices -> asset names
    weights = pd.Series(
        {assets[i]: raw_weights.loc[i] for i in range(n)},
        dtype=float,
    )
    weights /= weights.sum()  # ensure exact sum-to-1

    eq_weights = pd.Series(1.0 / n, index=assets)

    asset_order = [assets[i] for i in order]
    corr_reordered = correlation.loc[asset_order, asset_order]

    return HRPResult(
        weights=weights,
        equal_weights=eq_weights,
        linkage_matrix=link,
        quasi_diag_order=order,
        corr_reordered=corr_reordered,
    )
