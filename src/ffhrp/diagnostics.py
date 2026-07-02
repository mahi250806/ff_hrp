from dataclasses import dataclass

import numpy as np
import pandas as pd

from .covariance import CovarianceResult
from .factor_model import FactorModelResult


@dataclass
class DiagnosticsResult:
    r2: pd.Series
    factor_var_share: pd.Series   # fraction of annualized variance from factors
    idio_var_share: pd.Series     # fraction idiosyncratic
    residual_corr: pd.DataFrame   # residual correlation matrix
    total_var: pd.Series          # annualized total variance per asset


def compute_diagnostics(
    fm: FactorModelResult,
    cov: CovarianceResult,
    annualize: int = 252,
) -> DiagnosticsResult:
    """
    Decompose variance into factor-explained vs idiosyncratic shares,
    and compute the residual correlation matrix to expose hidden sector
    concentration that factors didn't absorb.
    """
    assets = fm.betas.index.tolist()

    sigma = cov.sigma
    total_var = pd.Series(np.diag(sigma.values), index=assets, name="total_var")

    # Factor-explained variance: diag(B F B^T)
    B = fm.betas[fm.factor_cols].values   # N x K
    F_ann = cov.factor_cov.values         # K x K
    systematic_var = np.diag(B @ F_ann @ B.T)
    factor_var_share = pd.Series(
        systematic_var / np.maximum(total_var.values, 1e-12),
        index=assets,
        name="factor_var_share",
    )
    idio_var_share = (1.0 - factor_var_share).rename("idio_var_share")

    # Residual correlation (exposes uncaptured sector co-movement)
    resid = fm.residuals[assets].dropna()
    if resid.shape[0] > 2:
        resid_cov_arr = resid.cov().values
        std = np.sqrt(np.maximum(np.diag(resid_cov_arr), 1e-12))
        resid_corr_arr = resid_cov_arr / np.outer(std, std)
        resid_corr_arr = np.clip(resid_corr_arr, -1.0, 1.0)
        np.fill_diagonal(resid_corr_arr, 1.0)
    else:
        resid_corr_arr = np.eye(len(assets))

    residual_corr = pd.DataFrame(resid_corr_arr, index=assets, columns=assets)

    return DiagnosticsResult(
        r2=fm.r2,
        factor_var_share=factor_var_share,
        idio_var_share=idio_var_share,
        residual_corr=residual_corr,
        total_var=total_var,
    )
