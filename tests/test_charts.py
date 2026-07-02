"""
Smoke tests: every chart builder must return the correct figure type
without raising, given a valid (synthetic) Result object.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pytest

from ffhrp.factor_model import FactorModelResult
from ffhrp.covariance import CovarianceResult
from ffhrp.hrp import HRPResult
from ffhrp.diagnostics import DiagnosticsResult

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import app.charts as C


# ── Fixtures ──────────────────────────────────────────────────────────────────

ASSETS = ["LLY", "MSFT", "NVDA", "MU", "COCO"]
FACTORS = ["Mkt-RF", "SMB", "HML", "Mom"]
N = len(ASSETS)
K = len(FACTORS)
T = 120


@pytest.fixture
def betas():
    rng = np.random.default_rng(1)
    return pd.DataFrame(rng.normal(0.5, 0.4, (N, K)), index=ASSETS, columns=FACTORS)


@pytest.fixture
def r2():
    return pd.Series([0.72, 0.68, 0.55, 0.41, 0.12], index=ASSETS, name="R2")


@pytest.fixture
def excess_returns():
    rng = np.random.default_rng(2)
    idx = pd.date_range("2023-01-01", periods=T, freq="B")
    return pd.DataFrame(rng.normal(0, 0.01, (T, N)), index=idx, columns=ASSETS)


@pytest.fixture
def factors_df():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2023-01-01", periods=T, freq="B")
    return pd.DataFrame(rng.normal(0, 0.008, (T, K)), index=idx, columns=FACTORS)


@pytest.fixture
def factor_cov():
    rng = np.random.default_rng(4)
    A = rng.standard_normal((K, K))
    cov = A @ A.T * 1e-4
    return pd.DataFrame(cov, index=FACTORS, columns=FACTORS)


@pytest.fixture
def sigma():
    rng = np.random.default_rng(5)
    A = rng.standard_normal((N, N))
    cov = A @ A.T * 1e-2 + np.eye(N) * 0.05
    return pd.DataFrame(cov, index=ASSETS, columns=ASSETS)


@pytest.fixture
def corr(sigma):
    std = np.sqrt(np.diag(sigma.values))
    c = sigma.values / np.outer(std, std)
    np.fill_diagonal(c, 1.0)
    return pd.DataFrame(c, index=ASSETS, columns=ASSETS)


@pytest.fixture
def hrp_result(corr, sigma):
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    dist = np.sqrt(np.clip((1 - corr.values) / 2, 0, 1))
    np.fill_diagonal(dist, 0)
    link = linkage(squareform(dist, checks=False), method="single")
    weights = pd.Series(np.array([0.25, 0.20, 0.22, 0.18, 0.15]), index=ASSETS)
    eq = pd.Series(0.2, index=ASSETS)
    return HRPResult(
        weights=weights,
        equal_weights=eq,
        linkage_matrix=link,
        quasi_diag_order=list(range(N)),
        corr_reordered=corr,
    )


@pytest.fixture
def diagnostics(r2):
    fvs = pd.Series([0.6, 0.55, 0.45, 0.35, 0.1], index=ASSETS, name="factor_var_share")
    ivs = 1 - fvs
    rc = pd.DataFrame(np.eye(N), index=ASSETS, columns=ASSETS)
    tv = pd.Series([0.08, 0.07, 0.12, 0.09, 0.06], index=ASSETS)
    return DiagnosticsResult(r2=r2, factor_var_share=fvs, idio_var_share=ivs,
                             residual_corr=rc, total_var=tv)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_betas_heatmap_returns_figure(betas):
    fig = C.betas_heatmap(betas)
    assert isinstance(fig, go.Figure)


def test_r2_bar_returns_figure(r2):
    fig = C.r2_bar(r2)
    assert isinstance(fig, go.Figure)


def test_regression_scatter_returns_figure(excess_returns, factors_df):
    fig = C.regression_scatter(
        excess_returns["LLY"],
        factors_df["Mkt-RF"],
        "LLY",
        beta=1.2,
        alpha=0.0001,
    )
    assert isinstance(fig, go.Figure)


def test_factor_cov_heatmap_returns_figure(factor_cov):
    fig = C.factor_cov_heatmap(factor_cov)
    assert isinstance(fig, go.Figure)


def test_correlation_heatmap_returns_figure(corr):
    fig = C.correlation_heatmap(corr)
    assert isinstance(fig, go.Figure)


def test_variance_decomp_bar_returns_figure(diagnostics):
    fig = C.variance_decomp_bar(diagnostics.factor_var_share, diagnostics.idio_var_share)
    assert isinstance(fig, go.Figure)


def test_hrp_weights_bar_returns_figure(hrp_result):
    fig = C.hrp_weights_bar(hrp_result.weights, hrp_result.equal_weights)
    assert isinstance(fig, go.Figure)


def test_hrp_weights_pie_returns_figure(hrp_result):
    fig = C.hrp_weights_pie(hrp_result.weights)
    assert isinstance(fig, go.Figure)


def test_dendrogram_returns_matplotlib_figure(hrp_result):
    fig = C.dendrogram_fig(hrp_result.linkage_matrix, ASSETS)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
