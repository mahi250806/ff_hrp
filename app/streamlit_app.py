"""
Factor-Model + HRP Portfolio Allocator — Streamlit UI
Run: streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

# Resolve src/ (ffhrp package) and root (app package) onto sys.path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

import streamlit as st

from ffhrp.config import AssetConfig, PortfolioConfig, RunConfig, load_config
from ffhrp.pipeline import run as pipeline_run

import app.charts as C

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Factor + HRP Allocator",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS injection ──────────────────────────────────────────────────────
st.html("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=IBM+Plex+Mono:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:       #0D0F14;
    --surface:  #161A23;
    --border:   #252A36;
    --grid:     #1E2330;
    --text-hi:  #E8EBF0;
    --text-lo:  #6B7385;
    --accent:   #00C2FF;
    --warm:     #FF6B35;
    --neg:      #E05260;
    --font-ui:  'Inter', sans-serif;
    --font-num: 'IBM Plex Mono', monospace;
  }

  html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text-hi) !important;
    font-family: var(--font-ui) !important;
  }
  [data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
  }
  [data-testid="stSidebar"] * { color: var(--text-hi) !important; }

  /* Remove default Streamlit padding */
  .block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }

  /* Section card */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.2rem 1.4rem 1rem;
    margin-bottom: 1.2rem;
  }
  .stage-label {
    font-family: var(--font-num);
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.3rem;
  }
  .stage-title {
    font-family: var(--font-ui);
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text-hi);
    margin-bottom: 0.8rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
  }
  .caption {
    font-family: var(--font-num);
    font-size: 0.78rem;
    color: var(--text-lo);
    margin-top: 0.6rem;
    line-height: 1.5;
  }
  .stat-chip {
    display: inline-block;
    font-family: var(--font-num);
    font-size: 0.72rem;
    color: var(--accent);
    background: rgba(0,194,255,0.08);
    border: 1px solid rgba(0,194,255,0.2);
    border-radius: 4px;
    padding: 2px 8px;
    margin: 2px 4px 2px 0;
  }
  .warn-chip {
    display: inline-block;
    font-family: var(--font-num);
    font-size: 0.72rem;
    color: var(--warm);
    background: rgba(255,107,53,0.08);
    border: 1px solid rgba(255,107,53,0.2);
    border-radius: 4px;
    padding: 2px 8px;
    margin: 2px 4px 2px 0;
  }

  /* Streamlit widget overrides */
  .stSlider > div > div > div > div { background: var(--accent) !important; }
  .stCheckbox label { font-size: 0.82rem !important; font-family: var(--font-num) !important; }
  .stSelectbox label, .stSlider label {
    font-family: var(--font-num) !important;
    font-size: 0.75rem !important;
    color: var(--text-lo) !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  div[data-testid="stDataFrame"] { background: var(--surface) !important; }
  .stButton > button {
    background: var(--accent) !important;
    color: var(--bg) !important;
    border: none !important;
    font-family: var(--font-num) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.06em;
    border-radius: 4px !important;
    padding: 0.45rem 1.4rem !important;
    width: 100%;
  }
  .stButton > button:hover { opacity: 0.85 !important; }

  /* Hide Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
</style>
""")

# ── Sidebar controls ──────────────────────────────────────────────────────────
_DEFAULT_CONFIG_PATH = _ROOT / "config" / "portfolio.yaml"
_base_cfg = load_config(_DEFAULT_CONFIG_PATH)
_ALL_ASSETS = _base_cfg.assets

with st.sidebar:
    st.markdown(
        "<div style='font-family:var(--font-num);font-size:0.65rem;"
        "letter-spacing:0.12em;text-transform:uppercase;color:var(--accent);"
        "margin-bottom:0.2rem'>◈ FF + HRP</div>"
        "<div style='font-size:1rem;font-weight:600;margin-bottom:1.4rem'>"
        "Portfolio Allocator</div>",
        unsafe_allow_html=True,
    )

    frequency = st.selectbox(
        "Frequency", ["daily", "monthly"], index=0, key="freq"
    )
    lookback = st.slider(
        "Lookback (years)", min_value=1, max_value=5, value=2, step=1, key="lookback"
    )
    factor_model = st.selectbox(
        "Factor model",
        ["carhart4", "ff3", "ff5", "ff5_mom"],
        index=0,
        key="fm",
    )

    st.markdown(
        "<div style='font-family:var(--font-num);font-size:0.68rem;"
        "text-transform:uppercase;letter-spacing:0.08em;color:var(--text-lo);"
        "margin:1.2rem 0 0.5rem'>Assets</div>",
        unsafe_allow_html=True,
    )
    selected_tickers = []
    for asset in _ALL_ASSETS:
        label = f"{asset.ticker}  –  {asset.asset_class}"
        if st.checkbox(label, value=True, key=f"asset_{asset.ticker}"):
            selected_tickers.append(asset.ticker)

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("RUN PIPELINE", key="run")


# ── Pipeline execution (cached) ───────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _cached_run(tickers: tuple, frequency: str, lookback: int, factor_model: str):
    """
    Cache is keyed on the FULL input set.
    Toggling an asset re-runs the pipeline — dropping a name changes the
    covariance and therefore the clustering, so partial re-slicing is wrong.
    """
    assets = [a for a in _ALL_ASSETS if a.ticker in set(tickers)]
    cfg = PortfolioConfig(
        run=RunConfig(
            frequency=frequency,
            lookback_years=lookback,
            factor_model=factor_model,
        ),
        assets=assets,
    )
    return pipeline_run(cfg)


# Trigger on first load AND on button press
_cache_key = (tuple(sorted(selected_tickers)), frequency, lookback, factor_model)

with st.spinner("Running pipeline…"):
    try:
        result = _cached_run(*_cache_key)
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        st.stop()

# ── Helper: card wrapper ──────────────────────────────────────────────────────
def _card(stage_num: str, title: str):
    st.markdown(
        f"<div class='stage-label'>Stage {stage_num}</div>"
        f"<div class='stage-title'>{title}</div>",
        unsafe_allow_html=True,
    )


def _caption(text: str):
    st.markdown(f"<div class='caption'>{text}</div>", unsafe_allow_html=True)


def _chip(text: str, warn: bool = False):
    cls = "warn-chip" if warn else "stat-chip"
    st.markdown(f"<span class='{cls}'>{text}</span>", unsafe_allow_html=True)


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='font-family:var(--font-ui);font-size:1.5rem;font-weight:600;"
    "color:var(--text-hi);margin-bottom:0.2rem'>"
    "Factor-Model + HRP Portfolio Allocator</h1>"
    "<p style='font-family:var(--font-num);font-size:0.78rem;color:var(--text-lo);"
    "margin-bottom:1.5rem'>"
    "Carhart 4-factor OLS  ·  Σ = BFBᵀ + D  ·  Hierarchical Risk Parity</p>",
    unsafe_allow_html=True,
)

# ── Stage 1 — Data ────────────────────────────────────────────────────────────
with st.container():
    _card("1", "Data — Returns & Factors")

    d_start = result.date_range[0].strftime("%d %b %Y")
    d_end = result.date_range[1].strftime("%d %b %Y")
    n_obs = len(result.excess_returns)
    n_assets = result.excess_returns.shape[1]

    col_chips = st.columns([1, 1, 1, 1])
    with col_chips[0]:
        _chip(f"{n_obs} observations")
    with col_chips[1]:
        _chip(f"{n_assets} assets")
    with col_chips[2]:
        _chip(f"{d_start} → {d_end}")
    with col_chips[3]:
        _chip(f"{result.config.run.factor_model.upper()}")

    if result.dropped_assets:
        for ticker, reason in result.dropped_assets:
            _chip(f"Dropped {ticker}: {reason}", warn=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            "<div style='font-family:var(--font-num);font-size:0.7rem;"
            "color:var(--text-lo);margin-bottom:0.3rem'>EXCESS RETURNS (head)</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            (result.excess_returns * 100).head(6).round(3).style.format("{:.3f}%"),
            use_container_width=True,
            height=220,
        )
    with c2:
        factor_display = [c for c in result.factors.columns]
        st.markdown(
            "<div style='font-family:var(--font-num);font-size:0.7rem;"
            "color:var(--text-lo);margin-bottom:0.3rem'>FRENCH FACTORS (head)</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            (result.factors * 100).head(6).round(4).style.format("{:.4f}%"),
            use_container_width=True,
            height=220,
        )

    _caption(
        f"Daily prices downloaded from yfinance and converted to simple returns. "
        f"Ken French {result.config.run.factor_model.upper()} factors aligned to the same "
        f"trading calendar. All returns shown are excess of the risk-free rate (RF column). "
        f"Window: {d_start} → {d_end} ({n_obs} trading days)."
    )

st.markdown("<hr style='border-color:var(--border);margin:0.5rem 0 1rem'>", unsafe_allow_html=True)

# ── Stage 2 — Factor regression ───────────────────────────────────────────────
with st.container():
    _card("2", "Factor Regression — OLS Betas & R²")

    # Summary table
    summary = result.betas.copy()
    summary.insert(0, "alpha (ann bp)", result.alphas * 252 * 1e4)
    summary["R²"] = result.r2
    summary["resid σ (ann %)"] = (result.residual_variances ** 0.5) * (252 ** 0.5) * 100

    st.dataframe(summary.round(4), use_container_width=True, height=260)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(
            "<div style='font-family:var(--font-num);font-size:0.7rem;"
            "color:var(--text-lo);margin-bottom:0.3rem'>BETAS HEATMAP</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            C.betas_heatmap(result.betas),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with c2:
        st.markdown(
            "<div style='font-family:var(--font-num);font-size:0.7rem;"
            "color:var(--text-lo);margin-bottom:0.3rem'>R² BY ASSET</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            C.r2_bar(result.r2),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    # Regression scatter for the asset with highest R²
    best_asset = result.r2.idxmax()
    mkt_col = "Mkt-RF"
    mkt_series = result.factors[mkt_col].loc[result.excess_returns.index]
    er_series = result.excess_returns[best_asset]
    beta_val = float(result.betas.loc[best_asset, mkt_col])
    alpha_val = float(result.alphas.loc[best_asset])

    st.markdown(
        "<div style='font-family:var(--font-num);font-size:0.7rem;"
        "color:var(--text-lo);margin:0.6rem 0 0.3rem'>"
        f"EXAMPLE REGRESSION SCATTER — {best_asset} vs Mkt-RF</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        C.regression_scatter(er_series, mkt_series, best_asset, beta_val, alpha_val),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    _caption(
        "OLS fit of each stock's daily excess return on the factor set. "
        "Betas are the exposures — a market beta of 1.4 means the stock amplifies "
        "index moves by 40%. Alpha is the unexplained return per year (in basis points). "
        "R² shows how well the four factors explain the stock's daily variance; "
        "low R² for COCO or ARKG is expected and is a sign they diversify the basket."
    )

st.markdown("<hr style='border-color:var(--border);margin:0.5rem 0 1rem'>", unsafe_allow_html=True)

# ── Stage 3 — Covariance reconstruction ──────────────────────────────────────
with st.container():
    _card("3", "Covariance — Σ = BFBᵀ + D")

    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.markdown(
            "<div style='font-family:var(--font-num);font-size:0.7rem;"
            "color:var(--text-lo);margin-bottom:0.3rem'>FACTOR COVARIANCE F (K×K, annualized × 10⁴)</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            C.factor_cov_heatmap(result.factor_cov),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with c2:
        st.markdown(
            "<div style='font-family:var(--font-num);font-size:0.7rem;"
            "color:var(--text-lo);margin-bottom:0.3rem'>RECONSTRUCTED CORRELATION Σ-DERIVED</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            C.correlation_heatmap(result.correlation),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    st.markdown(
        "<div style='font-family:var(--font-num);font-size:0.7rem;"
        "color:var(--text-lo);margin:0.8rem 0 0.3rem'>VARIANCE DECOMPOSITION — FACTOR vs IDIOSYNCRATIC</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        C.variance_decomp_bar(
            result.diagnostics.factor_var_share,
            result.diagnostics.idio_var_share,
        ),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    _caption(
        "Instead of estimating Σ directly from 2 years of returns (noisy with ~500 obs), "
        "we reconstruct it as Σ = BFBᵀ + D: factor exposures B, factor covariance F "
        "estimated from the factor series, plus diagonal D of residual variances. "
        "This shrinks the estimate toward a factor structure and avoids inverting a "
        "noisy sample matrix. The stacked bars show what fraction of each stock's "
        "annualized variance comes from factors vs its own idiosyncratic moves."
    )

st.markdown("<hr style='border-color:var(--border);margin:0.5rem 0 1rem'>", unsafe_allow_html=True)

# ── Stage 4 — Clustering ──────────────────────────────────────────────────────
with st.container():
    _card("4", "Clustering — Dendrogram & Quasi-Diagonal Reordering")

    asset_names = result.correlation.index.tolist()

    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown(
            "<div style='font-family:var(--font-num);font-size:0.7rem;"
            "color:var(--text-lo);margin-bottom:0.3rem'>DENDROGRAM (single linkage)</div>",
            unsafe_allow_html=True,
        )
        dend_fig = C.dendrogram_fig(result.linkage_matrix, asset_names)
        st.pyplot(dend_fig, use_container_width=True)
    with c2:
        st.markdown(
            "<div style='font-family:var(--font-num);font-size:0.7rem;"
            "color:var(--text-lo);margin-bottom:0.3rem'>CORRELATION — HRP ORDERING</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            C.correlation_heatmap(result.corr_reordered, "HRP order"),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    st.markdown(
        "<div style='font-family:var(--font-num);font-size:0.7rem;"
        "color:var(--text-lo);margin:0.8rem 0 0.3rem'>RESIDUAL CORRELATION (after removing factors)</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        C.correlation_heatmap(result.diagnostics.residual_corr),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    _caption(
        "HRP uses single-linkage hierarchical clustering to find which assets move together, "
        "then reorders the covariance matrix so similar assets sit next to each other "
        "(quasi-diagonalization) before splitting risk down the tree. "
        "The residual correlation heatmap shows co-movement the four factors didn't capture — "
        "watch for the semiconductor cluster (NVDA, AVGO, MU) still showing up here."
    )

st.markdown("<hr style='border-color:var(--border);margin:0.5rem 0 1rem'>", unsafe_allow_html=True)

# ── Stage 5 — Allocation ──────────────────────────────────────────────────────
with st.container():
    _card("5", "Allocation — HRP Weights")

    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.markdown(
            "<div style='font-family:var(--font-num);font-size:0.7rem;"
            "color:var(--text-lo);margin-bottom:0.3rem'>HRP vs EQUAL WEIGHT</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            C.hrp_weights_bar(result.weights, result.equal_weights),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with c2:
        st.markdown(
            "<div style='font-family:var(--font-num);font-size:0.7rem;"
            "color:var(--text-lo);margin-bottom:0.3rem'>WEIGHT DISTRIBUTION</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            C.hrp_weights_pie(result.weights),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    # Weight table
    wt = result.weights.sort_values(ascending=False).rename("HRP")
    eq = result.equal_weights.rename("1/N")
    diff = (wt - eq).rename("Δ vs 1/N")
    wt_df = (
        wt.to_frame()
        .join(eq)
        .join(diff)
        .style.format("{:.2%}")
        .background_gradient(subset=["HRP"], cmap="Blues")
    )
    st.dataframe(wt_df, use_container_width=True, height=310)

    _caption(
        "HRP splits the risk budget recursively down the dendrogram tree: at each branch "
        "it allocates proportionally to the inverse variance of each sub-cluster. "
        "Assets in tightly correlated clusters (e.g., the semis) share a single branch's "
        "budget, so the allocator naturally diversifies within that group. "
        "Where HRP leans in vs equal-weight signals a lower-risk or more isolated asset."
    )
