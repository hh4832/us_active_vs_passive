# US Active vs Passive

Reproducible research pipeline for comparing a real Firstrade active portfolio with cash-flow-matched passive portfolios. It reconstructs fills rather than pretending that trades begin earning returns on the next day.

## Current status

The repository contains the complete analysis framework and synthetic unit tests. A real research run requires:

- a Firstrade CSV or Excel export;
- `TIINGO_API_TOKEN` in the environment;
- optional preconfigured `rclone` access for Google Drive upload.

The production price backend is Tiingo EOD. The pipeline never silently fixes transaction or market-data anomalies. Audit files are written before performance analysis; unsupported activity types, negative positions, unavailable tickers, missing trade-date prices, missing held-position prices, and unresolved reconciliation items remain visible or fatal as appropriate.

### Tiingo production prices

For every actual-account and benchmark ticker, production retrieves at least:

- `close` — raw end-of-day close;
- `adjClose` — split- and dividend-adjusted total-return close;
- `divCash` — cash distribution on the ex-date; and
- `splitFactor` — split/distribution adjustment factor.

The pipeline audits ticker coverage, every buy/sell date, every actual holding session, and all returned corporate-action rows. Any missing `close` or `adjClose` on a trade date or actual holding date is fatal. Genuine market-data gaps are never silently forward-filled.

Every run writes `trades/ticker_coverage_audit.csv`, `trades/trade_date_coverage_audit.csv`, `trades/holding_period_coverage.csv`, `trades/corporate_actions.csv`, `trades/price_resolution_audit.csv`, and transaction diagnostics before performance reconstruction. The old FinLab and manual-fallback loaders remain only as legacy modules and are not imported by the production runner.

## Methodology

### Actual-fill reconstruction

Raw broker fills are nominal prices. An adjusted execution price is retained for audit and adjusted-scale research:

```text
adjustment_factor(t)       = Tiingo adjClose(t) / Tiingo close(t)
adjusted_execution_price  = actual Firstrade fill * adjustment_factor(t)
```

Daily investment P&L is decomposed into:

```text
existing-position market P&L
+ buy execution-to-close P&L
+ sell previous-close-to-execution P&L
+ dividend income
- fees
```

Without intraday timestamps, external cash flows are assumed available before that day's close. Actual portfolios use the raw Firstrade fill, raw Tiingo close, and actual Firstrade cash dividends. Passive benchmarks and relative-risk series use Tiingo `adjClose`. A next-trading-day sensitivity series is generated separately.

This raw-price actual-account path prevents the broker's cash dividend from being counted again through a dividend-adjusted market-price series.

### NAV and returns

```text
NAV(t) = cash(t) + sum(shares(i,t) * raw_market_close(i,t))
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

## Strategy reset analysis

The full Firstrade transaction history is always reconstructed first. The main active-versus-passive research comparison is then reset at the `analysis_start_date` in `config/config.yaml`, currently `2026-04-01`.

At the 2026-04-01 close, the pipeline preserves every reconstructed security quantity and the real account cash balance. The full-account strategy-start NAV is:

```text
opening cash + sum(opening shares × Tiingo raw close on 2026-04-01)
```

The reset index is exactly `1.0` on that date. Historical holdings are preserved, but all pre-start profit and loss is excluded. Post-start deposits and withdrawals remain external flows, so they do not inflate or depress TWR.

The active sleeve is defined from the configured portfolio exclusions rather than a hard-coded ticker list. Its opening inventory is marked at the 2026-04-01 raw close. Subsequent active buys are recorded as sleeve contributions; active sells and broker dividends transferred back to account cash are recorded as withdrawals. Cash-flow-matched VOO, VGT, and 80/20 equity/SGOV portfolios receive those same dated amounts. Decision-matched VOO and VGT additionally test the contribution-only counterfactual of directing each active buy's actual dollar amount to the benchmark on the same date.

This is why the research cannot use Firstrade `Total Gain` directly: that field can contain pre-reset cost basis and historical P&L, does not isolate active-sleeve capital transfers, and cannot enforce identical benchmark funding dates. Each run therefore writes opening-state audits, dated sleeve cash flows, reset performance, attribution, regime analysis, and reset-specific figures while retaining every full-history output.

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
export TIINGO_API_TOKEN='...'
pytest -q
```

Place the private broker export in `data/raw/`; this directory is ignored by Git. Do not commit tokens or Google credentials.

## Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hh4832/us_active_vs_passive/blob/main/notebooks/us_active_vs_passive_colab.ipynb)

The production notebook is `notebooks/us_active_vs_passive_colab.ipynb`. Before running it, prepare:

- a Colab Secret named `TIINGO_API_TOKEN`;
- one Firstrade CSV or Excel export;
- permission to mount Google Drive; and
- the actual mounted `MyDrive` output path.

The configured Drive folder ID identifies the intended folder, but it is not a Linux filesystem path. After mounting Drive in Colab, the default is `DRIVE_OUTPUT_ROOT = "/content/drive/MyDrive/Quant_Research/us_active_vs_passive"`. If the folder name or location differs, only this parameter needs to change.

Run the notebook from top to bottom. It clones the latest `main`, installs dependencies, loads the Tiingo token, accepts the Firstrade export, requires the complete unit-test suite to pass, runs the production pipeline, displays the main results, and copies the timestamped run folder to Drive.

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
trades/      coverage/corporate-action audits, normalized fills, FIFO P&L, ticker summary
risk/        episodes, regimes, capture, rolling and matched results
benchmarks/  cash-flow-matched benchmark series
figures/     equity, drawdown and rolling-risk charts
metadata/    run info, data sources and warnings
```

Reset-specific outputs include `summary/opening_state_20260401.csv`, `summary/opening_cash_20260401.csv`, `summary/performance_since_start.csv`, `benchmarks/active_sleeve_cashflows.csv`, `benchmarks/decision_matched_transactions.csv`, `risk/active_sleeve_contribution_since_start.csv`, and `risk/regime_analysis_since_start.csv`.

`run_info.json` records `price_source = Tiingo`, Tiingo's API latest date, coverage status, input hash, analysis dates, assumptions, portfolio definitions, benchmark weights, software versions, git SHA, and all warnings.

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

Interpretation must acknowledge the short 2026 YTD period, limited regime coverage, non-random portfolio formation, contribution timing, missing intraday timestamps, incomplete fee/tax fields, differing universes, vendor adjustment conventions, and the inability of short-period results to establish durable alpha.
