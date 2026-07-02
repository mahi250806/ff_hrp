from dataclasses import dataclass

import numpy as np
import pandas as pd

from .factor_model import FactorModelResult


@dataclass
class CovarianceResult:
    factor_cov: pd.DataFrame   # F: K x K, annualized
    sigma: pd.DataFrame        # Sigma: N x N, annualized
    correlation: pd.DataFrame  # derived from Sigma
    assets: list
    factor_cols: list


def build_covariance(
    fm: FactorModelResult,
    factors: pd.DataFrame,
    annualize: int = 252,
) -> CovarianceResult:
    """
    Reconstruct Sigma = B @ F @ B.T + diag(D)

    Annualization convention: multiply daily variances by 252, monthly by 12.
    F is estimated from the realized factor returns in the sample.
    D is the per-asset unbiased residual variance from OLS.
    """
    factor_cols = fm.factor_cols
    F_per_period = factors[factor_cols].cov().values  # K x K
    F_ann = F_per_period * annualize

    B = fm.betas[factor_cols].values           # N x K
    D = fm.residual_variances.values * annualize  # N

    systematic = B @ F_ann @ B.T              # N x N
    sigma_arr = systematic + np.diag(D)       # N x N

    # Enforce exact symmetry (floating-point drift)
    sigma_arr = (sigma_arr + sigma_arr.T) / 2.0

    assets = fm.betas.index.tolist()
    sigma_df = pd.DataFrame(sigma_arr, index=assets, columns=assets)
    F_df = pd.DataFrame(F_ann, index=factor_cols, columns=factor_cols)

    # Correlation matrix
    std = np.sqrt(np.maximum(np.diag(sigma_arr), 1e-12))
    corr_arr = sigma_arr / np.outer(std, std)
    corr_arr = np.clip(corr_arr, -1.0, 1.0)
    np.fill_diagonal(corr_arr, 1.0)
    corr_df = pd.DataFrame(corr_arr, index=assets, columns=assets)

    return CovarianceResult(
        factor_cov=F_df,
        sigma=sigma_df,
        correlation=corr_df,
        assets=assets,
        factor_cols=factor_cols,
    )
