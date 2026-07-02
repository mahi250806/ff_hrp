from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FactorModelResult:
    betas: pd.DataFrame          # N x K  (assets x factors)
    alphas: pd.Series            # N
    residuals: pd.DataFrame      # T x N  (NaN where return was NaN)
    residual_variances: pd.Series  # N  (sigma^2 of residuals)
    r2: pd.Series                # N
    factor_cols: list
    flagged: list                # assets that didn't meet min_obs guard


def fit_factor_model(
    excess_returns: pd.DataFrame,
    factors: pd.DataFrame,
    min_obs_ratio: int = 20,
) -> FactorModelResult:
    """
    Per-asset OLS: excess_return_i = alpha_i + B_i @ factors + epsilon_i

    Parameters
    ----------
    excess_returns : T x N, already net of RF
    factors        : T x K, must NOT include the RF column
    min_obs_ratio  : flag asset if obs < min_obs_ratio * K
    """
    factor_cols = [c for c in factors.columns if c != "RF"]
    K = len(factor_cols)
    F = factors[factor_cols].values  # T x K

    # Design matrix with intercept: T x (K+1)
    X_full = np.column_stack([np.ones(len(F)), F])

    betas_dict, alphas_dict, resid_dict, resid_var_dict, r2_dict = {}, {}, {}, {}, {}
    flagged = []

    for asset in excess_returns.columns:
        y = excess_returns[asset].values
        mask = ~np.isnan(y)
        y_clean, X_clean = y[mask], X_full[mask]

        if y_clean.shape[0] < min_obs_ratio * K:
            flagged.append(asset)
            continue

        try:
            coeffs, _, _, _ = np.linalg.lstsq(X_clean, y_clean, rcond=None)
        except np.linalg.LinAlgError:
            flagged.append(asset)
            continue

        alpha = coeffs[0]
        betas = coeffs[1:]

        y_hat = X_clean @ coeffs
        resid = y_clean - y_hat
        ss_res = float(resid @ resid)
        demeaned = y_clean - y_clean.mean()
        ss_tot = float(demeaned @ demeaned)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        # unbiased residual variance: divide by (n - K - 1)
        dof = max(y_clean.shape[0] - K - 1, 1)
        resid_var = ss_res / dof

        # Align residuals back to full time index (NaN for missing obs)
        full_resid = np.full(len(y), np.nan)
        full_resid[mask] = resid

        betas_dict[asset] = betas
        alphas_dict[asset] = alpha
        resid_dict[asset] = full_resid
        resid_var_dict[asset] = resid_var
        r2_dict[asset] = r2

    betas_df = pd.DataFrame(betas_dict, index=factor_cols).T        # N x K
    alphas_s = pd.Series(alphas_dict, name="alpha")
    residuals_df = pd.DataFrame(resid_dict, index=excess_returns.index)
    resid_var_s = pd.Series(resid_var_dict, name="resid_var")
    r2_s = pd.Series(r2_dict, name="R2")

    return FactorModelResult(
        betas=betas_df,
        alphas=alphas_s,
        residuals=residuals_df,
        residual_variances=resid_var_s,
        r2=r2_s,
        factor_cols=factor_cols,
        flagged=flagged,
    )
