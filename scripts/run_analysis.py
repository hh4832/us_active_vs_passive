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
from src.finlab_loader import load_finlab_prices
from src.load_transactions import load_transactions
from src.normalize_transactions import audit_transactions, normalize_transactions
from src.performance import performance_metrics, relative_metrics
from src.plotting import create_account_figures, create_standard_figures
from src.portfolio_accounting import reconstruct_next_day_sensitivity, reconstruct_portfolio
from src.price_adjustment import attach_adjusted_execution_prices
from src.regime import capture_ratios, regime_analysis


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
    tx.to_csv(run / "trades/normalized_transactions_pre_price.csv", index=False)
    write_json(run / "metadata/warnings.json", audit.warnings)

    universe = sorted(tx.loc[tx.type.isin(["buy", "sell"]), "ticker"].dropna().unique())
    required = sorted(set(universe) | {"VOO", "VGT", "SGOV"})
    bundle = load_finlab_prices(required, cfg.get("known_funds", []))
    tx, adjustment_warnings = attach_adjusted_execution_prices(tx, bundle.close, bundle.adj_close)
    fatal_trade_adjustments = tx.type.isin(["buy", "sell"]) & tx.adjusted_execution_price.isna()
    warnings = audit.warnings + bundle.warnings + adjustment_warnings
    if fatal_trade_adjustments.any():
        write_json(run / "metadata/warnings.json", warnings)
        raise RuntimeError(f"Cannot reconstruct {fatal_trade_adjustments.sum()} trade(s) without adjusted execution prices; audit saved to {run}")
    tx.to_csv(run / "trades/normalized_transactions.csv", index=False)

    definitions = cfg["portfolio_definitions"]
    results = {}
    for name, definition in definitions.items():
        included = set(universe) - set(definition.get("exclude", []))
        results[name] = reconstruct_portfolio(tx, bundle.adj_close, included_tickers=included)
        results[name].daily.to_csv(run / f"daily/{name}_daily_nav.csv")
        results[name].holdings.to_csv(run / f"daily/{name}_daily_holdings.csv", index=False)
        warnings.extend(results[name].warnings)
    actual = results["full_actual_account"]
    actual.daily.to_csv(run / "daily/daily_nav.csv")
    actual.holdings.to_csv(run / "daily/daily_holdings.csv", index=False)
    actual.daily[["twr"]].to_csv(run / "daily/daily_returns.csv")
    actual.daily[["cash", "market_value"]].to_csv(run / "daily/daily_exposure.csv")
    drawdown_series(actual.daily.twr).rename("drawdown").to_csv(run / "daily/daily_drawdown.csv")

    flows = actual.daily.external_cash_flow
    benchmark_frames = {}
    for name, weights in cfg["benchmarks"].items():
        benchmark_frames[name] = cashflow_matched_benchmark(flows, bundle.adj_close.reindex(actual.daily.index).ffill(), weights)
    active = results["active_equity_sleeve"].daily.twr
    voo_ret = bundle.adj_close.VOO.pct_change().reindex(active.index)
    vgt_ret = bundle.adj_close.VGT.pct_change().reindex(active.index)
    sgov_ret = bundle.adj_close.SGOV.pct_change().reindex(active.index)
    beta = relative_metrics(active, voo_ret)["beta"]
    dynamic = {"beta_matched_voo_sgov": beta_matched_weights(beta),
               "volatility_matched_voo_sgov": volatility_matched_weights(active, voo_ret, sgov_ret)}
    for name, weights in dynamic.items(): benchmark_frames[name] = cashflow_matched_benchmark(flows, bundle.adj_close.reindex(actual.daily.index).ffill(), weights)
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
    fifo_realized_pnl(tx).to_csv(run / "trades/realized_pnl_fifo.csv", index=False)
    tickers = ticker_summary(tx, bundle.adj_close.reindex(actual.daily.index).ffill().iloc[-1])
    tickers.to_csv(run / "trades/ticker_summary.csv", index=False)

    sensitivity = reconstruct_next_day_sensitivity(tx, bundle.adj_close, included_tickers=set(universe) - {"VOO", "VGT", "SGOV"})
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

    info = base_run_info(args.transactions, analysis_start=actual.daily.index.min(), analysis_end=actual.daily.index.max(),
                         finlab_data_sources=["us_price:close", "us_price:adj_close", "us_fund_price:close", "us_fund_price:adj_close"],
                         finlab_latest_date=bundle.adj_close.index.max(), tickers=required, stock_fund_classification=bundle.classification,
                         detected_columns=detected, risk_free_rate=cfg["risk_free_rate"], annualization_factor=cfg["annualization_factor"],
                         transaction_timing_convention=cfg["transaction_timing"], cash_flow_timing=cfg["cash_flow_timing"],
                         realized_pnl_method=cfg["realized_pnl_method"], portfolio_definitions=definitions,
                         benchmark_definitions=cfg["benchmarks"] | dynamic,
                         adjustment_methodology="adjusted_execution_price = raw execution price * adj_close / close", warnings=warnings)
    write_json(run / "metadata/run_info.json", info)
    write_json(run / "metadata/data_sources.json", info["finlab_data_sources"])
    write_json(run / "metadata/warnings.json", warnings)
    conclusion = ROOT / "templates/research_conclusion_template.md"
    shutil.copy2(conclusion, run / "summary/research_conclusion.md")
    if args.upload:
        subprocess.run(["rclone", "copy", str(run), f"{args.rclone_remote}:{cfg['google_drive_folder_id']}/{run.name}"], check=True)
    print(json.dumps({"run_directory": str(run), "warnings": len(warnings)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
