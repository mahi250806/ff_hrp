# ─────────────────────────────────────────────────────────────────────────────
# Design tokens — ALL colors/fonts derived from this block; nothing hardcoded
# below this comment.
#
# PALETTE
#   BG        = #0D0F14   near-black ground
#   SURFACE   = #161A23   card / sidebar surface
#   BORDER    = #252A36   hairline borders
#   GRID      = #1E2330   subtle gridlines
#   TEXT_HI   = #E8EBF0   primary text
#   TEXT_LO   = #6B7385   muted / caption text
#   ACCENT    = #00C2FF   cyan — reserved for HRP weights (the signal)
#   WARM      = #FF6B35   orange — secondary accent
#   NEG       = #E05260   red for negative values
#   FACTOR_PALETTE = [#4A90E2, #7ED321, #F5A623, #BD10E0]  (4 factors)
#
# FONTS
#   UI_FONT   = 'Inter'          headings / labels
#   NUM_FONT  = 'IBM Plex Mono'  all numbers (tabular figures)
#
# ACCENT RULE
#   Only one saturated color per chart; everything else is TEXT_LO or muted.
#   HRP weight bar/pie uses ACCENT. All other charts use muted or gradient.
# ─────────────────────────────────────────────────────────────────────────────

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import dendrogram

# ── Token constants ───────────────────────────────────────────────────────────
BG = "#0D0F14"
SURFACE = "#161A23"
BORDER = "#252A36"
GRID = "#1E2330"
TEXT_HI = "#E8EBF0"
TEXT_LO = "#6B7385"
ACCENT = "#00C2FF"
WARM = "#FF6B35"
NEG = "#E05260"
FACTOR_PALETTE = ["#4A90E2", "#7ED321", "#F5A623", "#BD10E0", "#50E3C2", "#E86EFF"]

UI_FONT = "Inter, sans-serif"
NUM_FONT = "'IBM Plex Mono', monospace"

# Plotly layout defaults shared across all figures
_BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family=UI_FONT, color=TEXT_HI, size=12),
    margin=dict(l=8, r=8, t=8, b=8),
    showlegend=False,
)

_AXIS_STYLE = dict(
    gridcolor=GRID,
    gridwidth=1,
    zeroline=False,
    tickfont=dict(family=NUM_FONT, color=TEXT_LO, size=11),
    linecolor=BORDER,
    tickcolor=BORDER,
)

# Diverging correlation color scale: blue → white → red
_CORR_COLORSCALE = [
    [0.0, "#2563EB"],
    [0.25, "#93C5FD"],
    [0.5, "#F8FAFC"],
    [0.75, "#FCA5A5"],
    [1.0, "#DC2626"],
]


def _fig(**kwargs) -> go.Figure:
    layout = {**_BASE_LAYOUT, **kwargs}
    return go.Figure(layout=go.Layout(**layout))


# ── Chart builders ────────────────────────────────────────────────────────────


def betas_heatmap(betas: pd.DataFrame) -> go.Figure:
    """Heatmap of factor betas: assets (rows) x factors (cols)."""
    z = betas.values
    text = np.round(z, 2).astype(str)

    fig = _fig()
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=betas.columns.tolist(),
            y=betas.index.tolist(),
            colorscale=_CORR_COLORSCALE,
            zmid=0,
            text=text,
            texttemplate="%{text}",
            textfont=dict(family=NUM_FONT, size=10, color=TEXT_HI),
            colorbar=dict(
                thickness=10,
                tickfont=dict(family=NUM_FONT, color=TEXT_LO, size=10),
                tickcolor=BORDER,
                outlinewidth=0,
            ),
        )
    )
    fig.update_layout(
        xaxis=dict(**_AXIS_STYLE, showgrid=False),
        yaxis=dict(**_AXIS_STYLE, showgrid=False, autorange="reversed"),
        height=300,
    )
    return fig


def r2_bar(r2: pd.Series) -> go.Figure:
    """Horizontal bar chart of R² per asset, sorted descending."""
    r2_sorted = r2.sort_values(ascending=True)
    colors = [ACCENT if v >= 0.5 else TEXT_LO for v in r2_sorted.values]

    fig = _fig()
    fig.add_trace(
        go.Bar(
            x=r2_sorted.values,
            y=r2_sorted.index.tolist(),
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.2f}" for v in r2_sorted.values],
            textposition="outside",
            textfont=dict(family=NUM_FONT, size=11, color=TEXT_LO),
        )
    )
    fig.update_layout(
        xaxis=dict(**_AXIS_STYLE, range=[0, 1.05], tickformat=".0%"),
        yaxis=dict(**_AXIS_STYLE, showgrid=False),
        height=320,
    )
    return fig


def regression_scatter(
    excess_return: pd.Series,
    market_factor: pd.Series,
    asset_name: str,
    beta: float,
    alpha: float,
) -> go.Figure:
    """Scatter of excess returns vs market factor with OLS fit line."""
    x = market_factor.values
    y = excess_return.values
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]

    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = alpha + beta * x_line

    fig = _fig()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(color=TEXT_LO, size=3, opacity=0.6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            line=dict(color=ACCENT, width=1.5),
        )
    )
    fig.update_layout(
        xaxis=dict(**_AXIS_STYLE, title=dict(text="Mkt-RF", font=dict(color=TEXT_LO))),
        yaxis=dict(**_AXIS_STYLE, title=dict(text=f"{asset_name} excess ret", font=dict(color=TEXT_LO))),
        height=280,
        annotations=[
            dict(
                text=f"β = {beta:.2f}  α = {alpha*252*100:.1f} bp/yr",
                xref="paper", yref="paper",
                x=0.02, y=0.97, showarrow=False,
                font=dict(family=NUM_FONT, size=11, color=TEXT_LO),
                align="left",
            )
        ],
    )
    return fig


def factor_cov_heatmap(factor_cov: pd.DataFrame) -> go.Figure:
    """Small heatmap of the K×K factor covariance matrix (annualized)."""
    z = factor_cov.values * 1e4  # in basis points² for readability
    text = np.round(z, 1).astype(str)

    fig = _fig()
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=factor_cov.columns.tolist(),
            y=factor_cov.index.tolist(),
            colorscale="Blues",
            text=text,
            texttemplate="%{text}",
            textfont=dict(family=NUM_FONT, size=11, color=TEXT_HI),
            colorbar=dict(
                thickness=10,
                title=dict(text="× 10⁻⁴", font=dict(color=TEXT_LO, size=10)),
                tickfont=dict(family=NUM_FONT, color=TEXT_LO, size=10),
                outlinewidth=0,
            ),
        )
    )
    fig.update_layout(
        xaxis=dict(**_AXIS_STYLE, showgrid=False),
        yaxis=dict(**_AXIS_STYLE, showgrid=False, autorange="reversed"),
        height=260,
    )
    return fig


def correlation_heatmap(corr: pd.DataFrame, title_note: str = "") -> go.Figure:
    """Diverging correlation heatmap with cell annotations."""
    z = corr.values
    labels = corr.columns.tolist()
    n = len(labels)
    text = np.round(z, 2).astype(str)
    show_text = n <= 12  # annotate cells only when small enough to read

    fig = _fig()
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale=_CORR_COLORSCALE,
            text=text if show_text else None,
            texttemplate="%{text}" if show_text else None,
            textfont=dict(family=NUM_FONT, size=9, color="#1E293B"),
            colorbar=dict(
                thickness=10,
                tickvals=[-1, -0.5, 0, 0.5, 1],
                tickformat=".1f",
                tickfont=dict(family=NUM_FONT, color=TEXT_LO, size=10),
                outlinewidth=0,
            ),
        )
    )
    fig.update_layout(
        xaxis=dict(**_AXIS_STYLE, showgrid=False, tickangle=-45),
        yaxis=dict(**_AXIS_STYLE, showgrid=False, autorange="reversed"),
        height=340,
    )
    return fig


def variance_decomp_bar(
    factor_var_share: pd.Series,
    idio_var_share: pd.Series,
) -> go.Figure:
    """Stacked bar: factor-explained vs idiosyncratic share of variance per asset."""
    assets = factor_var_share.index.tolist()
    order = factor_var_share.sort_values(ascending=False).index.tolist()

    fig = _fig(showlegend=True)
    fig.add_trace(
        go.Bar(
            name="Factor",
            x=order,
            y=[factor_var_share[a] for a in order],
            marker=dict(color=ACCENT, line=dict(width=0)),
        )
    )
    fig.add_trace(
        go.Bar(
            name="Idiosyncratic",
            x=order,
            y=[idio_var_share[a] for a in order],
            marker=dict(color=BORDER, line=dict(width=0)),
        )
    )
    fig.update_layout(
        barmode="stack",
        xaxis=dict(**_AXIS_STYLE, showgrid=False),
        yaxis=dict(**_AXIS_STYLE, tickformat=".0%", range=[0, 1.05]),
        legend=dict(
            orientation="h",
            x=0, y=1.08,
            font=dict(color=TEXT_LO, size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=280,
    )
    return fig


def hrp_weights_bar(weights: pd.Series, equal_weights: pd.Series) -> go.Figure:
    """
    The signature visual — HRP weights vs equal-weight, grouped horizontal bars.
    HRP in ACCENT; equal-weight in muted.
    """
    order = weights.sort_values(ascending=True).index.tolist()

    fig = _fig(showlegend=True)
    fig.add_trace(
        go.Bar(
            name="Equal Weight",
            x=[equal_weights[a] for a in order],
            y=order,
            orientation="h",
            marker=dict(color=BORDER, line=dict(width=0)),
            width=0.35,
            offset=-0.4,
        )
    )
    fig.add_trace(
        go.Bar(
            name="HRP",
            x=[weights[a] for a in order],
            y=order,
            orientation="h",
            marker=dict(color=ACCENT, line=dict(width=0)),
            width=0.35,
            offset=0.05,
            text=[f"{weights[a]:.1%}" for a in order],
            textposition="outside",
            textfont=dict(family=NUM_FONT, size=11, color=ACCENT),
        )
    )
    max_w = max(weights.max(), equal_weights.max())
    fig.update_layout(
        barmode="overlay",
        xaxis=dict(**_AXIS_STYLE, tickformat=".0%", range=[0, max_w * 1.25]),
        yaxis=dict(**_AXIS_STYLE, showgrid=False),
        legend=dict(
            orientation="h",
            x=0, y=1.08,
            font=dict(color=TEXT_LO, size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=360,
    )
    return fig


def hrp_weights_pie(weights: pd.Series) -> go.Figure:
    """Donut chart of HRP weights — accent gradient, no legend chrome."""
    n = len(weights)
    # Gradient from ACCENT to WARM across assets sorted by weight
    order = weights.sort_values(ascending=False)

    # Build a gradient palette
    import matplotlib.colors as mcolors
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "hrp", [ACCENT, "#006B8F", WARM], N=n
    )
    colors = [mcolors.to_hex(cmap(i / max(n - 1, 1))) for i in range(n)]

    fig = _fig(showlegend=True)
    fig.add_trace(
        go.Pie(
            labels=order.index.tolist(),
            values=order.values.tolist(),
            hole=0.55,
            marker=dict(colors=colors, line=dict(color=BG, width=2)),
            textfont=dict(family=NUM_FONT, size=11, color=TEXT_HI),
            textinfo="label+percent",
            hovertemplate="%{label}: %{percent}<extra></extra>",
        )
    )
    fig.update_layout(
        showlegend=False,
        height=320,
        margin=dict(l=0, r=0, t=8, b=8),
    )
    return fig


def dendrogram_fig(
    linkage_matrix: np.ndarray,
    asset_names: list,
) -> plt.Figure:
    """
    Matplotlib dendrogram styled to match the dark palette.
    Returns a matplotlib Figure for st.pyplot().
    """
    matplotlib.rcParams.update(
        {
            "font.family": "monospace",
            "text.color": TEXT_HI,
            "axes.labelcolor": TEXT_LO,
            "xtick.color": TEXT_LO,
            "ytick.color": TEXT_LO,
        }
    )

    fig, ax = plt.subplots(figsize=(9, 3.5))
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")

    dendrogram(
        linkage_matrix,
        labels=asset_names,
        ax=ax,
        color_threshold=0,
        above_threshold_color=ACCENT,
        leaf_rotation=45,
        leaf_font_size=10,
    )

    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.tick_params(axis="both", which="both", color=BORDER)
    ax.set_ylabel("Distance", color=TEXT_LO, fontsize=10)

    fig.tight_layout(pad=0.5)
    return fig
