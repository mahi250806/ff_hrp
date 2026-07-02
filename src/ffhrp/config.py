from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import yaml

VALID_FACTOR_MODELS = {"ff3", "carhart4", "ff5", "ff5_mom"}
VALID_FREQUENCIES = {"daily", "monthly"}


@dataclass
class AssetConfig:
    sheet_symbol: str
    ticker: str
    asset_class: str
    proxy_note: Optional[str] = None


@dataclass
class RunConfig:
    frequency: str = "daily"
    lookback_years: int = 2
    factor_model: str = "carhart4"
    min_obs_ratio: int = 20
    risk_free_from_factors: bool = True


@dataclass
class PortfolioConfig:
    run: RunConfig
    assets: List[AssetConfig]


def load_config(path: str | Path) -> PortfolioConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    run_raw = raw.get("run", {})
    run = RunConfig(
        frequency=run_raw.get("frequency", "daily"),
        lookback_years=run_raw.get("lookback_years", 2),
        factor_model=run_raw.get("factor_model", "carhart4"),
        min_obs_ratio=run_raw.get("min_obs_ratio", 20),
        risk_free_from_factors=run_raw.get("risk_free_from_factors", True),
    )

    if run.frequency not in VALID_FREQUENCIES:
        raise ValueError(
            f"Invalid frequency '{run.frequency}'. Must be one of {VALID_FREQUENCIES}"
        )
    if run.factor_model not in VALID_FACTOR_MODELS:
        raise ValueError(
            f"Invalid factor_model '{run.factor_model}'. Must be one of {VALID_FACTOR_MODELS}"
        )

    assets = []
    for a in raw.get("assets", []):
        assets.append(
            AssetConfig(
                sheet_symbol=a["sheet_symbol"],
                ticker=a["ticker"],
                asset_class=a["asset_class"],
                proxy_note=a.get("proxy_note"),
            )
        )

    return PortfolioConfig(run=run, assets=assets)
