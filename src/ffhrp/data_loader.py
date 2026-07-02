import io
import urllib.request
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from .config import PortfolioConfig

CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
ANNUALIZATION = {"daily": 252, "monthly": 12}

# Ken French data URLs
_FF3_DAILY = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
_MOM_DAILY = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"
_FF5_DAILY = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
_FF3_MONTHLY = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
_MOM_MONTHLY = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"
_FF5_MONTHLY = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"


def _fetch_zip_csv(url: str) -> str:
    """Download a zip from Ken French's site and return the inner CSV text."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    zf = zipfile.ZipFile(io.BytesIO(data))
    csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
    return zf.read(csv_name).decode("latin-1")


def _parse_french_table(raw: str, is_daily: bool) -> pd.DataFrame:
    """
    Extract the main data table from a Ken French CSV file.
    Data rows start with a YYYYMMDD (daily, 8-digit) or YYYYMM (monthly, 6-digit) date.
    Handles both whitespace-delimited and comma-delimited variants.
    """
    date_len = 8 if is_daily else 6
    rows = []
    for line in raw.splitlines():
        # Normalise: strip and collapse any delimiter to spaces
        stripped = line.strip().replace(",", " ")
        if not stripped:
            continue
        parts = stripped.split()
        if not parts:
            continue
        first = parts[0]
        if first.isdigit() and len(first) == date_len:
            rows.append(parts)
        elif rows:
            # First non-numeric line after data block = end of first table
            break
    if not rows:
        raise ValueError(
            f"No data rows found in French factor file "
            f"(expected {date_len}-digit dates). "
            f"First 5 lines: {raw.splitlines()[:5]}"
        )
    return pd.DataFrame(rows)


def _load_french_raw(factor_model: str, is_daily: bool) -> pd.DataFrame:
    """Download and parse the base 3- or 5-factor table."""
    if factor_model in ("ff3", "carhart4"):
        url = _FF3_DAILY if is_daily else _FF3_MONTHLY
        raw = _fetch_zip_csv(url)
        df = _parse_french_table(raw, is_daily)
        df.columns = ["date", "Mkt-RF", "SMB", "HML", "RF"]
    else:  # ff5, ff5_mom
        url = _FF5_DAILY if is_daily else _FF5_MONTHLY
        raw = _fetch_zip_csv(url)
        df = _parse_french_table(raw, is_daily)
        df.columns = ["date", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    return df


def _load_mom_raw(is_daily: bool) -> pd.DataFrame:
    url = _MOM_DAILY if is_daily else _MOM_MONTHLY
    raw = _fetch_zip_csv(url)
    df = _parse_french_table(raw, is_daily)
    df.columns = ["date", "Mom"]
    return df


def _build_french_df(factor_model: str, frequency: str) -> pd.DataFrame:
    is_daily = frequency == "daily"
    date_fmt = "%Y%m%d" if is_daily else "%Y%m"

    base = _load_french_raw(factor_model, is_daily)
    base["date"] = pd.to_datetime(base["date"].str.strip(), format=date_fmt)
    if not is_daily:
        base["date"] = base["date"] + pd.offsets.MonthEnd(0)
    base = base.set_index("date").sort_index()
    base = base.apply(pd.to_numeric, errors="coerce").dropna() / 100.0

    if factor_model in ("carhart4", "ff5_mom"):
        mom = _load_mom_raw(is_daily)
        mom["date"] = pd.to_datetime(mom["date"].str.strip(), format=date_fmt)
        if not is_daily:
            mom["date"] = mom["date"] + pd.offsets.MonthEnd(0)
        mom = mom.set_index("date").sort_index()
        mom["Mom"] = pd.to_numeric(mom["Mom"], errors="coerce") / 100.0
        base = base.join(mom["Mom"], how="inner")

    return base


def _get_french_factors(
    factor_model: str,
    frequency: str,
    start: datetime,
    end: datetime,
    cache_dir: Path,
) -> pd.DataFrame:
    freq_tag = frequency[0]
    cache_path = cache_dir / f"french_{factor_model}_{freq_tag}.parquet"

    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if cached.index.min() <= start and cached.index.max() >= end:
            return cached.loc[start:end]

    df = _build_french_df(factor_model, frequency)
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    return df.loc[start:end]


def _get_prices(
    tickers: list,
    start: datetime,
    end: datetime,
    cache_dir: Path,
) -> pd.DataFrame:
    cache_path = cache_dir / "prices.parquet"

    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        missing = [t for t in tickers if t not in cached.columns]
        if (
            not missing
            and cached.index.min() <= start
            and cached.index.max() >= end
        ):
            return cached[tickers].loc[start:end]

    downloaded = yf.download(
        tickers, start=start, end=end, auto_adjust=True, progress=False
    )
    if "Close" in downloaded.columns.get_level_values(0) if hasattr(downloaded.columns, "get_level_values") else False:
        prices = downloaded["Close"]
    else:
        prices = downloaded

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])

    prices = prices.sort_index()

    # Merge with existing cache
    if cache_path.exists():
        old = pd.read_parquet(cache_path)
        prices = pd.concat([old, prices]).sort_index()
        prices = prices[~prices.index.duplicated(keep="last")]

    cache_dir.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(cache_path)

    available = [t for t in tickers if t in prices.columns]
    return prices[available].loc[start:end]


def load_data(config: PortfolioConfig, cache_dir: Path = CACHE_DIR):
    """
    Pull prices and French factors, compute excess returns, align on trading calendar.

    Returns
    -------
    excess_returns : DataFrame (T x N), decimal, excess of RF
    factors        : DataFrame with factor columns + RF
    rf             : Series of daily RF
    date_range     : (start_date, end_date)
    dropped_assets : list of (ticker, reason) tuples
    """
    end = datetime.today()
    start = end - timedelta(days=int(config.run.lookback_years * 365.25) + 30)

    tickers = [a.ticker for a in config.assets]

    prices = _get_prices(tickers, start, end, cache_dir)

    if config.run.frequency == "daily":
        returns = prices.pct_change().dropna(how="all")
    else:
        returns = prices.resample("ME").last().pct_change().dropna(how="all")

    factors_full = _get_french_factors(
        config.run.factor_model, config.run.frequency, start, end, cache_dir
    )

    common_idx = returns.index.intersection(factors_full.index)
    returns = returns.loc[common_idx]
    factors = factors_full.loc[common_idx]

    rf = factors["RF"]
    excess_returns = returns.sub(rf, axis=0)

    factor_cols = [c for c in factors.columns if c != "RF"]
    n_factors = len(factor_cols)
    min_obs = config.run.min_obs_ratio * n_factors

    dropped_assets = []
    valid_tickers = []
    for ticker in tickers:
        if ticker not in excess_returns.columns:
            dropped_assets.append((ticker, "ticker not found in price download"))
            continue
        n_valid = excess_returns[ticker].dropna().shape[0]
        if n_valid < min_obs:
            dropped_assets.append(
                (ticker, f"only {n_valid} obs, need {min_obs} (min_obs_ratio × n_factors)")
            )
        else:
            valid_tickers.append(ticker)

    excess_returns = excess_returns[valid_tickers].dropna()
    factors = factors.loc[excess_returns.index]
    rf = rf.loc[excess_returns.index]

    date_range = (excess_returns.index.min(), excess_returns.index.max())
    return excess_returns, factors, rf, date_range, dropped_assets
