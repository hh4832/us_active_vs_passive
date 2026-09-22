#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.attribution import fifo_realized_pnl, ticker_summary
from src.benchmarks import beta_matched_weights, cashflow_matched_benchmark, volatility_matched_weights
from src.drawdown import drawdown_episodes, drawdown_series
from src.export import base_run_info, create_run_directory, write_json
from src.load_transactions import load_transactions
from src.normalize_transactions import audit_transactions, normalize_transactions
from src.performance import performance_metrics, relative_metrics
from src.plotting import create_account_figures, create_since_start_figures, create_standard_figures
from src.portfolio_accounting import reconstruct_next_day_sensitivity, reconstruct_portfolio
from src.price_adjustment import attach_adjusted_execution_prices
from src.regime import capture_ratios, regime_analysis
from src.strategy_reset import (
    active_sleeve_attribution,
    build_active_sleeve_since_start,
    build_reset_portfolio,
    decision_matched_cashflows,
    opening_state_audits,
)
from src.tiingo_loader import audit_tiingo_coverage, load_tiingo_prices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct Firstrade performance and fair passive benchmarks")
    parser.add_argument("transactions", type=Path, help="Firstrade CSV/XLSX")
    parser.add_argument("--config", type=Path, default=ROOT / "config/config.yaml")
    parser.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    parser.add_argument("--upload", action="store_true", help="Upload run folder using an existing rclone remote")
    parser.add_argument("--rclone-remote", default="gdrive")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    raw = load_transactions(args.transactions)
    tx, detected = normalize_transactions(raw, cfg["column_aliases"])
    audit = audit_transactions(tx)
    # Phase A is persisted before market-data loading or performance calculations.
    run = create_run_directory(args.outputs)
    audit.summary.to_csv(run / "trades/transaction_audit.csv", index=False)
    audit.ticker_summary.to_csv(run / "trades/transaction_ticker_audit.csv", index=False)
    audit.negative_holdings.to_csv(run / "trades/negative_holdings_diagnostic.csv", index=False)
    audit.transaction_type_diagnostic.to_csv(run / "trades/transaction_type_diagnostic.csv", index=False)
    audit.unrecognized_types.to_csv(run / "trades/unrecognized_transaction_types.csv", index=False)
    tx.to_csv(run / "trades/normalized_transactions_pre_price.csv", index=False)
    write_json(run / "metadata/warnings.json", audit.warnings)

    strategy_start = pd.Timestamp(cfg["analysis_start_date"]).normalize()
    if not audit.negative_holdings.empty:
        negative_dates = pd.to_datetime(audit.negative_holdings["date"])
        pre_start_negative = audit.negative_holdings.loc[negative_dates.le(strategy_start)]
        if not pre_start_negative.empty:
            raise RuntimeError(
                "Negative holdings exist on or before the strategy reset date; "
                "opening inventory cannot be inferred. See trades/negative_holdings_diagnostic.csv"
            )

    universe = sorted(tx.loc[tx.type.isin(["buy", "sell"]), "ticker"].dropna().unique())
    required = sorted(set(universe) | {"VOO", "VGT", "SGOV"})
    bundle = load_tiingo_prices(required, start_date=tx["date"].dropna().min())
    coverage = audit_tiingo_coverage(bundle, tx, actual_tickers=set(universe))
    coverage.ticker_coverage.to_csv(run / "trades/ticker_coverage_audit.csv", index=False)
    coverage.trade_date_coverage.to_csv(run / "trades/trade_date_coverage_audit.csv", index=False)
    coverage.holding_period_coverage.to_csv(run / "trades/holding_period_coverage.csv", index=False)
    coverage.corporate_actions.to_csv(run / "trades/corporate_actions.csv", index=False)
    if not coverage.passed:
        raise RuntimeError(
            "Tiingo price coverage failed; see ticker_coverage_audit.csv, "
            "trade_date_coverage_audit.csv, and holding_period_coverage.csv"
        )
    tx, adjustment_warnings = attach_adjusted_execution_prices(
        tx, bundle.close, bundle.adj_close, price_sources={ticker: "tiingo" for ticker in required},
    )
    price_audit_columns = [
        "source_row", "date", "ticker", "type", "raw_execution_price", "close", "adj_close",
        "adjustment_factor", "adjusted_execution_price", "adjustment_source",
        "adjustment_reference_date", "adjustment_date_shift_days", "warning",
    ]
    tx.loc[tx.type.isin(["buy", "sell"]), price_audit_columns].to_csv(
        run / "trades/price_resolution_audit.csv", index=False,
    )
    fatal_trade_adjustments = tx.type.isin(["buy", "sell"]) & tx.adjusted_execution_price.isna()
    warnings = audit.warnings + adjustment_warnings
    if fatal_trade_adjustments.any():
        write_json(run / "metadata/warnings.json", warnings)
        unresolved = tx.loc[fatal_trade_adjustments, ["source_row", "date", "ticker", "type", "warning"]]
        raise RuntimeError(
            f"Cannot reconstruct {fatal_trade_adjustments.sum()} trade(s) without adjusted execution prices; "
            f"audit saved to {run / 'trades/price_resolution_audit.csv'}; unresolved={unresolved.to_dict('records')}"
        )
    tx.to_csv(run / "trades/normalized_transactions.csv", index=False)

    definitions = cfg["portfolio_definitions"]
    active_included = set(universe) - set(definitions["active_equity_sleeve"].get("exclude", []))
    results = {}
    for name, definition in definitions.items():
        included = set(universe) - set(definition.get("exclude", []))
        results[name] = reconstruct_portfolio(
            tx, bundle.close, included_tickers=included, execution_price_column="raw_execution_price",
        )
        results[name].daily.to_csv(run / f"daily/{name}_daily_nav.csv")
        results[name].holdings.to_csv(run / f"daily/{name}_daily_holdings.csv", index=False)
        warnings.extend(results[name].warnings)
    actual = results["full_actual_account"]
    actual.daily.to_csv(run / "daily/daily_nav.csv")
    actual.holdings.to_csv(run / "daily/daily_holdings.csv", index=False)
    actual.daily[["twr"]].to_csv(run / "daily/daily_returns.csv")
    actual.daily[["cash", "market_value"]].to_csv(run / "daily/daily_exposure.csv")
    drawdown_series(actual.daily.twr).rename("drawdown").to_csv(run / "daily/daily_drawdown.csv")

    # Strategy-reset analysis uses the complete history to establish the 2026-04-01
    # account state, then discards all earlier P&L from the comparison period.
    full_reset = build_reset_portfolio(actual, strategy_start)
    reset_definition = definitions.get("active_risk_assets", definitions["active_equity_sleeve"])
    reset_active_tickers = set(universe) - set(reset_definition.get("exclude", []))
    active_reset = build_active_sleeve_since_start(
        tx, bundle.close, actual, strategy_start, reset_active_tickers,
        execution_price_column="raw_execution_price",
    )
    if float(full_reset.daily.iloc[0]["nav"]) <= 0:
        raise RuntimeError("Strategy-start full-account NAV must be positive")
    if float(active_reset.daily.iloc[0]["nav"]) <= 0:
        raise RuntimeError("Strategy-start active-sleeve opening capital must be positive")
    full_reset.daily.to_csv(run / "daily/full_account_since_start_daily_nav.csv")
    full_reset.holdings.to_csv(run / "daily/full_account_since_start_daily_holdings.csv", index=False)
    active_reset.daily.to_csv(run / "daily/active_sleeve_since_start_daily_nav.csv")
    active_reset.holdings.to_csv(run / "daily/active_sleeve_since_start_daily_holdings.csv", index=False)
    active_reset.cashflows.to_csv(run / "benchmarks/active_sleeve_cashflows.csv", index=False)

    opening_state, opening_cash = opening_state_audits(
        actual, bundle.close, strategy_start, reset_active_tickers, set(cfg.get("known_funds", [])),
    )
    opening_state.to_csv(run / "summary/opening_state_20260401.csv", index=False)
    opening_cash.to_csv(run / "summary/opening_cash_20260401.csv", index=False)

    since_index = full_reset.daily.index
    since_prices = bundle.adj_close.reindex(since_index)
    full_reset_flows = full_reset.daily["external_cash_flow"].copy()
    full_reset_flows.iloc[0] = float(full_reset.daily.iloc[0]["nav"])
    active_reset_flows = active_reset.daily["external_cash_flow"].copy()
    since_benchmarks = {
        "full_account_matched_voo": cashflow_matched_benchmark(full_reset_flows, since_prices, {"VOO": 1.0}),
        "full_account_matched_vgt": cashflow_matched_benchmark(full_reset_flows, since_prices, {"VGT": 1.0}),
        "cf_matched_voo": cashflow_matched_benchmark(active_reset_flows, since_prices, {"VOO": 1.0}),
        "cf_matched_vgt": cashflow_matched_benchmark(active_reset_flows, since_prices, {"VGT": 1.0}),
        "cf_matched_voo_sgov_80_20": cashflow_matched_benchmark(
            active_reset_flows, since_prices, {"VOO": 0.8, "SGOV": 0.2},
        ),
        "cf_matched_vgt_sgov_80_20": cashflow_matched_benchmark(
            active_reset_flows, since_prices, {"VGT": 0.8, "SGOV": 0.2},
        ),
    }
    # Explicit aliases match the research-output terminology in the protocol.
    since_benchmarks["voo_sgov_80_20_since_start"] = since_benchmarks["cf_matched_voo_sgov_80_20"]
    since_benchmarks["vgt_sgov_80_20_since_start"] = since_benchmarks["cf_matched_vgt_sgov_80_20"]

    voo_since = since_prices["VOO"].pct_change(fill_method=None)
    vgt_since = since_prices["VGT"].pct_change(fill_method=None)
    sgov_since = since_prices["SGOV"].pct_change(fill_method=None)
    active_since = active_reset.daily["twr"]
    since_beta = relative_metrics(active_since, voo_since)["beta"]
    since_dynamic = {
        "beta_matched_voo_sgov": beta_matched_weights(since_beta),
        "volatility_matched_voo_sgov": volatility_matched_weights(active_since, voo_since, sgov_since),
    }
    for name, weights in since_dynamic.items():
        since_benchmarks[name] = cashflow_matched_benchmark(active_reset_flows, since_prices, weights)

    opening_active_capital = float(active_reset.daily.iloc[0]["nav"])
    decision_flows, decision_base = decision_matched_cashflows(
        tx, since_index, strategy_start, reset_active_tickers, opening_active_capital,
        execution_price_column="raw_execution_price",
    )
    decision_rows = []
    for benchmark in ["VOO", "VGT"]:
        key = f"decision_matched_{benchmark.lower()}"
        since_benchmarks[key] = cashflow_matched_benchmark(decision_flows, since_prices, {benchmark: 1.0})
        for row in decision_base.itertuples(index=False):
            price = float(since_prices.loc[row.date, benchmark])
            decision_rows.append({
                "date": row.date, "active_ticker": row.active_ticker,
                "active_amount": row.active_amount, "benchmark": benchmark,
                "benchmark_price": price, "benchmark_shares": row.active_amount / price,
                "source_row": row.source_row,
            })
    pd.DataFrame(decision_rows).to_csv(run / "benchmarks/decision_matched_transactions.csv", index=False)

    pd.concat({name: frame["nav"] for name, frame in since_benchmarks.items()}, axis=1).to_csv(
        run / "benchmarks/cashflow_matched_since_start.csv"
    )
    since_returns = pd.concat(
        {
            "full_account_since_start": full_reset.daily["twr"],
            "active_sleeve_since_start": active_since,
            **{name: frame["return"] for name, frame in since_benchmarks.items()},
            "voo_total_return_since_start": voo_since,
            "vgt_total_return_since_start": vgt_since,
        },
        axis=1,
    )
    since_returns.to_csv(run / "benchmarks/benchmark_returns_since_start.csv")

    since_summary_rows = []
    for name in since_returns.columns:
        metrics = performance_metrics(
            since_returns[name], risk_free_rate=float(cfg["risk_free_rate"]),
            annualization=int(cfg["annualization_factor"]),
        )
        relative = relative_metrics(since_returns[name], voo_since)
        captures = capture_ratios(since_returns[name], voo_since).set_index("regime")
        since_summary_rows.append({
            "portfolio": name, **metrics,
            "beta_vs_voo": relative["beta"], "alpha_vs_voo": relative["alpha"],
            "upside_capture": captures.loc["up", "arithmetic_daily_capture"],
            "downside_capture": captures.loc["down", "arithmetic_daily_capture"],
            "tracking_error": relative["tracking_error"],
            "information_ratio": relative["information_ratio"],
        })
    since_summary = pd.DataFrame(since_summary_rows)
    since_summary.to_csv(run / "summary/performance_since_start.csv", index=False)
    (run / "summary/performance_since_start.md").write_text(
        since_summary.to_markdown(index=False), encoding="utf-8"
    )
    regime_analysis(active_since, voo_since).to_csv(
        run / "risk/regime_analysis_since_start.csv", index=False,
    )
    active_sleeve_attribution(
        tx, active_reset, strategy_start, reset_active_tickers,
        execution_price_column="raw_execution_price",
    ).to_csv(run / "risk/active_sleeve_contribution_since_start.csv", index=False)
    create_since_start_figures(
        since_returns, run / "figures", start_date=strategy_start.date().isoformat(),
    )

    flows = actual.daily.external_cash_flow
    benchmark_frames = {}
    benchmark_prices = bundle.adj_close.reindex(actual.daily.index)
    for name, weights in cfg["benchmarks"].items():
        benchmark_frames[name] = cashflow_matched_benchmark(flows, benchmark_prices, weights)
    active = results["active_equity_sleeve"].daily.twr
    voo_ret = bundle.adj_close.VOO.pct_change(fill_method=None).reindex(active.index)
    vgt_ret = bundle.adj_close.VGT.pct_change(fill_method=None).reindex(active.index)
    sgov_ret = bundle.adj_close.SGOV.pct_change(fill_method=None).reindex(active.index)
    beta = relative_metrics(active, voo_ret)["beta"]
    dynamic = {"beta_matched_voo_sgov": beta_matched_weights(beta),
               "volatility_matched_voo_sgov": volatility_matched_weights(active, voo_ret, sgov_ret)}
    for name, weights in dynamic.items():
        benchmark_frames[name] = cashflow_matched_benchmark(flows, benchmark_prices, weights)
    pd.concat({k: v.nav for k, v in benchmark_frames.items()}, axis=1).to_csv(run / "benchmarks/cashflow_matched_benchmarks.csv")

    returns = pd.concat({k: v.daily.twr for k, v in results.items()} | {k: v["return"] for k, v in benchmark_frames.items()}, axis=1)
    returns.to_csv(run / "benchmarks/benchmark_nav.csv")
    summary_rows = []
    for name in returns:
        for rf_label, rf in [("rf_0", 0.0), ("rf_config", float(cfg["risk_free_rate"]))]:
            summary_rows.append({"portfolio": name, "rf_version": rf_label, **performance_metrics(returns[name], risk_free_rate=rf, annualization=cfg["annualization_factor"]),
                                 **{f"vs_voo_{k}": v for k, v in relative_metrics(returns[name], voo_ret).items()},
                                 **{f"vs_vgt_{k}": v for k, v in relative_metrics(returns[name], vgt_ret).items()}})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(run / "summary/performance_summary.csv", index=False)
    (run / "summary/performance_summary.md").write_text(summary.to_markdown(index=False), encoding="utf-8")

    drawdown_episodes(active, voo_ret).to_csv(run / "risk/drawdown_episodes.csv", index=False)
    regime_analysis(active, voo_ret).to_csv(run / "risk/regime_analysis.csv", index=False)
    capture_ratios(active, voo_ret).to_csv(run / "risk/capture_ratios.csv", index=False)
    pd.DataFrame(dynamic).T.to_csv(run / "risk/beta_matched_results.csv")
    rolling = pd.DataFrame({"active_vol_20": active.rolling(20).std() * 252 ** .5,
                            "active_vol_60": active.rolling(60).std() * 252 ** .5,
                            "beta_60_vs_voo": active.rolling(60).cov(voo_ret) / voo_ret.rolling(60).var(),
                            "corr_60_vs_voo": active.rolling(60).corr(voo_ret)})
    rolling.to_csv(run / "risk/rolling_metrics.csv")
    fifo_realized_pnl(tx, price_column="raw_execution_price").to_csv(run / "trades/realized_pnl_fifo.csv", index=False)
    tickers = ticker_summary(
        tx, bundle.close.reindex(actual.daily.index).iloc[-1], price_column="raw_execution_price",
    )
    tickers.to_csv(run / "trades/ticker_summary.csv", index=False)

    sensitivity = reconstruct_next_day_sensitivity(
        tx, bundle.close, included_tickers=active_included, execution_price_column="raw_execution_price",
    )
    pd.DataFrame([{"convention": "actual_fill", **performance_metrics(active)},
                  {"convention": "next_day", **performance_metrics(sensitivity.daily.twr)}]).to_csv(run / "risk/timing_sensitivity.csv", index=False)
    create_standard_figures(returns[["active_equity_sleeve", "voo", "vgt", "voo_sgov_80_20", "beta_matched_voo_sgov"]], run / "figures")
    create_account_figures(actual.daily, actual.holdings, tickers, run / "figures")

    # A zero daily difference demonstrates the accounting identity. Broker ending balance/holdings remain external inputs.
    reconciliation = pd.DataFrame([{"item": "daily_accounting_identity_max_abs_difference",
                                    "calculated": actual.daily.reconciliation_difference.abs().max(), "status": "PASS" if actual.daily.reconciliation_difference.abs().max() <= 0.01 else "WARNING"},
                                   {"item": "broker_ending_nav", "calculated": None, "status": "NOT_PROVIDED"},
                                   {"item": "broker_latest_holdings", "calculated": None, "status": "NOT_PROVIDED"}])
    reconciliation.to_csv(run / "summary/account_reconciliation.csv", index=False)
    if (reconciliation.status == "WARNING").any(): warnings.append("Accounting reconciliation exceeded tolerance")

    coverage_status = {
        "ticker_coverage": "PASS" if coverage.ticker_coverage["status"].eq("PASS").all() else "FAIL",
        "trade_date_coverage": "PASS" if coverage.trade_date_coverage["status"].eq("PASS").all() else "FAIL",
        "holding_period_coverage": "PASS" if coverage.holding_period_coverage["status"].eq("PASS").all() else "FAIL",
    }
    info = base_run_info(args.transactions, analysis_start=actual.daily.index.min(), analysis_end=actual.daily.index.max(),
                         price_source="Tiingo", api_latest_date=bundle.latest_date,
                         strategy_start_date=strategy_start,
                         strategy_start_nav=float(full_reset.daily.iloc[0]["nav"]),
                         active_sleeve_opening_capital=opening_active_capital,
                         strategy_reset_policy={
                             "valuation_point": "2026-04-01 end-of-day raw close after recorded same-day transactions",
                             "pre_start_pnl_included": False,
                             "opening_inventory_preserved": True,
                             "active_trade_cash_legs_are_external_flows": True,
                             "decision_matched_primary": "contribution-only",
                         },
                         coverage_status=coverage_status, tickers=required,
                         detected_columns=detected, risk_free_rate=cfg["risk_free_rate"], annualization_factor=cfg["annualization_factor"],
                         transaction_timing_convention=cfg["transaction_timing"], cash_flow_timing=cfg["cash_flow_timing"],
                         realized_pnl_method=cfg["realized_pnl_method"], portfolio_definitions=definitions,
                         benchmark_definitions=cfg["benchmarks"] | dynamic | since_dynamic,
                         actual_account_price_methodology="raw Tiingo close plus actual Firstrade cash dividends",
                         benchmark_price_methodology="Tiingo adjClose total-return series",
                         adjustment_methodology="adjusted_execution_price = actual Firstrade fill * Tiingo adjClose/close",
                         price_resolution_policy={
                             "primary": "exact Tiingo EOD close and adjClose",
                             "missing_trade_or_held_position_price": "fatal",
                             "silent_forward_fill": False,
                         }, warnings=warnings)
    write_json(run / "metadata/run_info.json", info)
    write_json(run / "metadata/data_sources.json", {
        "price_source": "Tiingo", "endpoint": "tiingo/daily/{ticker}/prices",
        "fields": ["close", "adjClose", "divCash", "splitFactor"],
    })
    write_json(run / "metadata/warnings.json", warnings)
    conclusion = ROOT / "templates/research_conclusion_template.md"
    shutil.copy2(conclusion, run / "summary/research_conclusion.md")
    if args.upload:
        subprocess.run(["rclone", "copy", str(run), f"{args.rclone_remote}:{cfg['google_drive_folder_id']}/{run.name}"], check=True)
    print(json.dumps({"run_directory": str(run), "warnings": len(warnings)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
