# Factor-Model + HRP Portfolio Allocator

Estimates a Carhart 4-factor model on a 10-asset equity basket, reconstructs a
factor-based covariance matrix (Σ = BFBᵀ + D), and allocates weights via
Hierarchical Risk Parity. A Streamlit UI walks through each stage of the pipeline.

---

## Setup

```bash
cd ff_hrp
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -e ".[dev]"
```

---

## Run the CLI pipeline

```bash
python -m ffhrp.pipeline --config config/portfolio.yaml
```

Writes to `outputs/`:
- `weights.json` — final HRP weights
- `betas.csv` — factor betas per asset
- `cov.npy` — full covariance matrix Σ
- `diagnostics.json` — R², factor/idio variance shares

---

## Run the Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

Opens a 5-stage walk-through:

| Stage | What you see |
|-------|-------------|
| 1 — Data | Aligned excess-return matrix, French factors, date range |
| 2 — Regression | Betas table, heatmap, R² bars, example scatter |
| 3 — Covariance | Factor covariance F, correlation heatmap, variance decomposition |
| 4 — Clustering | Dendrogram, quasi-diagonalized correlation, residual correlation |
| 5 — Allocation | HRP weights bar/pie, comparison vs 1/N |

Live controls in the sidebar let you change frequency, lookback, factor model,
and toggle individual assets. Every change re-runs the full pipeline (asset
inclusion affects the covariance and therefore the clustering).

---

## Run tests

```bash
pytest -v
```

---

## Covariance rationale — Σ = BFBᵀ + D

With only ~500 daily observations and 10 assets, a raw sample covariance matrix
is noisy and poorly conditioned. The factor-model reconstruction sidesteps this:

1. **B** (N × K) — factor betas from per-asset OLS regressions.
2. **F** (K × K) — factor covariance, estimated from the Ken French factor time
   series directly. K = 4 (Carhart) so this is a well-determined 4×4 matrix.
3. **D** (N × N diagonal) — residual variances from each OLS regression,
   capturing the idiosyncratic (stock-specific) risk the factors don't explain.

The resulting Σ is guaranteed symmetric positive semi-definite, has a clear
economic interpretation (factor risk + idio risk), and needs only K²/2 + N
parameters instead of N(N+1)/2.

---

## Proxy caveats

| Sheet symbol | Ticker | Note |
|---|---|---|
| DRAM | MU | Micron is used as a memory-cycle / DRAM proxy. Its factor loadings reflect the semi cycle more than broad IT. |
| USCOCOA_CASH | COCO | iPath Bloomberg Cocoa Subindex ETN. This is a commodity ETN, not an equity. Expect R² < 0.15 — the Fama-French factors explain almost none of its variance, which is precisely why it diversifies the basket. |
| ARKG | ARKG | ARK Genomic Innovation ETF — thematic, high idiosyncratic risk. Low factor R² expected. |

Excluded by design: UUP (FX/macro, different return process) and RNMBY
(Rheinmetall ADR: thin market, FX-contaminated returns).

---

## Annualization

All variances and covariances are annualized by multiplying by **252** (daily
frequency) or **12** (monthly). This convention is applied consistently in
`build_covariance()` and `compute_diagnostics()`.
