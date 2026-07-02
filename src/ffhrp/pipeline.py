"""
Pipeline orchestrator: wire all stages, return a stable Result object.

Usage (CLI):
    python -m ffhrp.pipeline --config config/portfolio.yaml
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PortfolioConfig, load_config
from .covariance import CovarianceResult, build_covariance
from .data_loader import ANNUALIZATION, CACHE_DIR, load_data
from .diagnostics import DiagnosticsResult, compute_diagnostics
from .factor_model import FactorModelResult, fit_factor_model
from .hrp import HRPResult, hrp

OUTPUTS_DIR = Path(__file__).parent.parent.parent / "outputs"


@dataclass
class Result:
    # --- Stage 1: Data ---
    excess_returns: pd.DataFrame   # T x N, decimal, excess of RF
    factors: pd.DataFrame          # T x (K+1), includes RF column
    rf: pd.Series                  # T, risk-free rate
    date_range: tuple              # (start_date, end_date)
    dropped_assets: list           # [(ticker, reason), ...]

    # --- Stage 2: Factor model ---
    betas: pd.DataFrame            # N x K
    alphas: pd.Series              # N
    r2: pd.Series                  # N
    residuals: pd.DataFrame        # T x N
    residual_variances: pd.Series  # N
    factor_cols: list              # factor column names

    # --- Stage 3: Covariance ---
    factor_cov: pd.DataFrame       # K x K, annualized
    sigma: pd.DataFrame            # N x N, annualized
    correlation: pd.DataFrame      # N x N, pre-reorder

    # --- Stage 4: HRP ---
    weights: pd.Series             # N, sums to 1
    equal_weights: pd.Series       # N, 1/N
    linkage_matrix: np.ndarray     # scipy linkage (N-1 x 4)
    quasi_diag_order: list         # integer indices
    corr_reordered: pd.DataFrame   # correlation in HRP order

    # --- Stage 5: Diagnostics ---
    diagnostics: DiagnosticsResult

    # --- Meta ---
    config: PortfolioConfig


def run(
    config: PortfolioConfig,
    cache_dir: Path = CACHE_DIR,
    outputs_dir: Path = OUTPUTS_DIR,
) -> Result:
    """
    Execute the full pipeline and return a Result carrying every intermediate.
    The Result is the UI's single source of truth — nothing is recomputed in app/.
    """
    ann = ANNUALIZATION[config.run.frequency]

    # ── Stage 1: Data ────────────────────────────────────────────────────────
    excess_returns, factors, rf, date_range, dropped_assets = load_data(
        config, cache_dir
    )
    factor_cols = [c for c in factors.columns if c != "RF"]

    # ── Stage 2: Factor model ────────────────────────────────────────────────
    fm = fit_factor_model(excess_returns, factors[factor_cols], config.run.min_obs_ratio)

    for asset in fm.flagged:
        dropped_assets.append((asset, "insufficient observations for factor regression"))
    valid_assets = [a for a in excess_returns.columns if a not in fm.flagged]
    excess_returns_valid = excess_returns[valid_assets]

    # ── Stage 3: Covariance ──────────────────────────────────────────────────
    cov = build_covariance(fm, factors, annualize=ann)

    # ── Stage 4: HRP ─────────────────────────────────────────────────────────
    hrp_result = hrp(cov.correlation, cov.sigma)

    # ── Stage 5: Diagnostics ─────────────────────────────────────────────────
    diag = compute_diagnostics(fm, cov, annualize=ann)

    # ── Write outputs ─────────────────────────────────────────────────────────
    _write_outputs(fm, cov, hrp_result, diag, outputs_dir)

    return Result(
        excess_returns=excess_returns_valid,
        factors=factors,
        rf=rf,
        date_range=date_range,
        dropped_assets=dropped_assets,
        betas=fm.betas,
        alphas=fm.alphas,
        r2=fm.r2,
        residuals=fm.residuals,
        residual_variances=fm.residual_variances,
        factor_cols=factor_cols,
        factor_cov=cov.factor_cov,
        sigma=cov.sigma,
        correlation=cov.correlation,
        weights=hrp_result.weights,
        equal_weights=hrp_result.equal_weights,
        linkage_matrix=hrp_result.linkage_matrix,
        quasi_diag_order=hrp_result.quasi_diag_order,
        corr_reordered=hrp_result.corr_reordered,
        diagnostics=diag,
        config=config,
    )


def _write_outputs(
    fm: FactorModelResult,
    cov: CovarianceResult,
    hrp_result: HRPResult,
    diag: DiagnosticsResult,
    outputs_dir: Path,
) -> None:
    outputs_dir.mkdir(parents=True, exist_ok=True)

    with open(outputs_dir / "weights.json", "w") as f:
        json.dump(hrp_result.weights.to_dict(), f, indent=2)

    fm.betas.to_csv(outputs_dir / "betas.csv")

    np.save(str(outputs_dir / "cov.npy"), cov.sigma.values)

    diag_dict = {
        "r2": diag.r2.to_dict(),
        "factor_var_share": diag.factor_var_share.to_dict(),
        "idio_var_share": diag.idio_var_share.to_dict(),
        "total_var": diag.total_var.to_dict(),
    }
    with open(outputs_dir / "diagnostics.json", "w") as f:
        json.dump(diag_dict, f, indent=2)


def _cli():
    parser = argparse.ArgumentParser(description="Run the FF+HRP pipeline")
    parser.add_argument(
        "--config", default="config/portfolio.yaml", help="Path to portfolio.yaml"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = run(cfg)

    print(f"\nDate range : {result.date_range[0].date()} → {result.date_range[1].date()}")
    print(f"Assets     : {list(result.weights.index)}")
    if result.dropped_assets:
        print(f"Dropped    : {result.dropped_assets}")
    print(f"\nHRP weights:\n{result.weights.round(4).to_string()}")
    print(f"\nOutputs written to: {OUTPUTS_DIR}")


if __name__ == "__main__":
    _cli()
