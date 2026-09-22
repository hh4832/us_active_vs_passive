# US Active vs Passive

Reproducible research pipeline for comparing a real Firstrade active portfolio with cash-flow-matched passive portfolios. It reconstructs fills rather than pretending that trades begin earning returns on the next day.

## Current status

The repository contains the complete analysis framework and synthetic unit tests. A real research run requires:

- a Firstrade CSV or Excel export;
- `FINLAB_API_TOKEN` in the environment;
- optional preconfigured `rclone` access for Google Drive upload.

The pipeline never silently fixes transaction anomalies. Phase A audit files are written before FinLab retrieval or performance analysis. Missing fields, unsupported activity types, negative positions, unavailable tickers, missing adjustment factors, and unresolved reconciliation items are preserved in the warnings output.

### Audited external price fallback

FinLab remains the primary price source. If a real holding is absent from both FinLab US stock and fund datasets, an optional `data/fallback_prices.csv` may provide an externally reviewed series:

```csv
date,ticker,close,adj_close,source,no_corporate_action_confirmed
2026-01-05,EXAMPLE,100.00,99.50,external_manual,
```

The first five columns are required. `adj_close` may be blank only when `no_corporate_action_confirmed` is explicitly set to `true` after confirming that no split or corporate action occurred; the pipeline then uses a factor of 1.0 and records a warning. The repository does not download, invent, or silently forward-fill missing execution-date inputs. External data should not be committed unless its provenance and redistribution status are appropriate.

Every run writes `trades/price_resolution_audit.csv`, `trades/negative_holdings_diagnostic.csv`, and transaction-type diagnostic tables before performance reconstruction. Unresolved execution prices remain fatal.

## Methodology

### Actual-fill reconstruction

Raw broker fills are nominal prices, while FinLab adjusted closes use an adjusted scale. Every buy and sell uses:

```text
adjustment_factor(t)       = adj_close(t) / close(t)
adjusted_execution_price  = actual_execution_price * adjustment_factor(t)
```

Daily investment P&L is decomposed into:

```text
existing-position market P&L
+ buy execution-to-close P&L
+ sell previous-close-to-execution P&L
+ dividend income
- fees
```

Without intraday timestamps, external cash flows are assumed available before that day's close. The main analysis applies actual recorded fills and marks the remaining position at close. A next-trading-day sensitivity series is generated separately.

> Important: adjusted prices and cash dividends can overlap economically depending on the vendor's adjustment convention. The pipeline follows the requested total-return-consistent adjusted-price methodology while retaining explicit broker dividends for account reconciliation. Review this treatment against the exact FinLab adjustment definition before interpreting economic attribution.

### NAV and returns

```text
NAV(t) = cash(t) + sum(shares(i,t) * adjusted_market_price(i,t))
TWR(t) = (NAV(t) - external_cash_flow(t)) / NAV(t-1) - 1
```

NAV is for account reconciliation; TWR removes external deposits and withdrawals. Money-weighted XIRR support is included in `src/performance.py` when the available cash-flow history supports a valid root.

### Portfolio definitions

| Portfolio | Holdings |
|---|---|
| Active Risk Assets | Active stocks + GLD; excludes VOO, VGT, SGOV, SYSB |
| Active Equity Sleeve | Compatibility alias for Active Risk Assets; SYSB is excluded |
| Active + SGOV | Active stocks + GLD + SGOV; excludes VOO, VGT, SYSB |
| Full Actual Account | All actual securities and cash |
| Passive variants | VOO, VGT, 80/20 VOO/SGOV, 80/20 VGT/SGOV |
| Matched variants | Beta-matched and volatility-matched VOO/SGOV |

All passive benchmarks receive the same external contributions and withdrawals on the same dates as the actual account and permit fractional shares.

### Metrics and formulas

- Cumulative return: `product(1 + r_t) - 1`
- Annualized return: `(1 + cumulative_return)^(252 / n) - 1`
- Annualized volatility: `std(r_t) * sqrt(252)`
- Sharpe: `mean(r_t - rf_daily) / std(r_t) * sqrt(252)`
- Sortino: `mean(r_t - rf_daily) / downside_deviation * sqrt(252)`
- Maximum drawdown: `min(wealth / running_peak - 1)`
- Calmar: `annualized_return / abs(maximum_drawdown)`
- Beta: `cov(portfolio, benchmark) / var(benchmark)`
- Annualized alpha: `252 * (mean(portfolio) - beta * mean(benchmark))`
- Tracking error: `std(portfolio - benchmark) * sqrt(252)`
- Information ratio: `annualized active return / tracking error`
- Arithmetic capture: mean active return divided by mean benchmark return within up/down days
- Geometric capture: annualized compounded active return divided by annualized compounded benchmark return within the same regime

Sharpe and Sortino are produced both with `rf = 0` and the configurable annual risk-free rate (default 4%).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export FINLAB_API_TOKEN='...'
pytest -q
```

Place the private broker export in `data/raw/`; this directory is ignored by Git. Do not commit tokens or Google credentials.

## Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hh4832/us_active_vs_passive/blob/main/notebooks/us_active_vs_passive_colab.ipynb)

The production notebook is `notebooks/us_active_vs_passive_colab.ipynb`. Before running it, prepare:

- a Colab Secret named `FINLAB_API_TOKEN`;
- one Firstrade CSV or Excel export;
- permission to mount Google Drive; and
- the actual mounted `MyDrive` output path.

The configured Drive folder ID identifies the intended folder, but it is not a Linux filesystem path. After mounting Drive in Colab, set `DRIVE_OUTPUT_ROOT` to that folder's real path under `/content/drive/MyDrive/`. If the folder name or location differs, only this parameter needs to change.

Run the notebook from top to bottom. It clones the latest `main`, installs dependencies, authenticates FinLab, accepts the Firstrade export, requires the complete unit-test suite to pass, runs the production pipeline, displays the main results, and copies the timestamped run folder to Drive.

The notebook is an execution and review interface, not the source of the analysis logic. Production accounting, benchmarking, attribution, and reporting remain in `src/` and `scripts/run_analysis.py`.

## Run

```bash
python scripts/run_analysis.py data/raw/firstrade.csv
```

Optional upload using an already authenticated rclone remote:

```bash
python scripts/run_analysis.py data/raw/firstrade.csv --upload --rclone-remote gdrive
```

The target Drive folder ID is configured in `config/config.yaml`. Credentials are never stored in this repository.

## Outputs

Each run creates `outputs/run_YYYYMMDD_HHMMSS/` with:

```text
summary/     performance tables, reconciliation, conclusion template
daily/       NAV, TWR, holdings, exposure, drawdown
trades/      audit, normalized fills, FIFO realized P&L, ticker summary
risk/        episodes, regimes, capture, rolling and matched results
benchmarks/  cash-flow-matched benchmark series
figures/     equity, drawdown and rolling-risk charts
metadata/    run info, data sources and warnings
```

`run_info.json` records the input hash, dates, sources, classifications, assumptions, portfolio definitions, benchmark weights, software versions, git SHA, and all warnings.

## Repository structure

```text
config/       versioned assumptions and aliases
data/         ignored raw/processed inputs
notebooks/    thin exploratory entry point
scripts/      production CLI
src/          tested research and accounting logic
templates/    evidence-based conclusion structure
tests/        synthetic regression tests
outputs/      ignored generated artifacts
```

## Reconciliation and limitations

The pipeline checks its internal daily accounting identity. True broker reconciliation additionally needs the latest Firstrade holdings and ending account value; absent inputs are explicitly marked `NOT_PROVIDED`, never treated as matched. Differences above `$1` or `0.01%` must be investigated.

Interpretation must acknowledge the short 2026 YTD period, limited regime coverage, non-random portfolio formation, contribution timing, missing intraday timestamps, price-scale conversion, incomplete fee/tax fields, differing universes, and the inability of short-period results to establish durable alpha.
