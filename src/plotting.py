from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .drawdown import drawdown_series


def _save(fig, path: Path, note: str) -> None:
    fig.text(0.01, 0.01, note, fontsize=7, color="dimgray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def create_standard_figures(returns: pd.DataFrame, output_dir: str | Path, note: str = "Source: Firstrade + Tiingo EOD") -> None:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    date_range = f"{returns.index.min().date()} to {returns.index.max().date()}"
    wealth = (1 + returns.fillna(0)).cumprod()
    fig, ax = plt.subplots(figsize=(10, 6)); wealth.plot(ax=ax); ax.set(title=f"Normalized NAV ({date_range})", ylabel="Growth of $1", xlabel="Date"); ax.legend(title="Portfolio"); _save(fig, output / "equity_curve.png", note)
    fig, ax = plt.subplots(figsize=(10, 5)); returns.apply(drawdown_series).plot(ax=ax); ax.set(title=f"Drawdown ({date_range})", ylabel="Drawdown", xlabel="Date"); ax.legend(title="Portfolio"); _save(fig, output / "drawdown_curve.png", note)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    (returns.rolling(20).std() * 252 ** 0.5).plot(ax=axes[0]); axes[0].set(title="Rolling 20-day annualized volatility", ylabel="Volatility")
    (returns.rolling(60).std() * 252 ** 0.5).plot(ax=axes[1]); axes[1].set(title="Rolling 60-day annualized volatility", ylabel="Volatility", xlabel="Date")
    _save(fig, output / "rolling_volatility.png", note)
    if "active_equity_sleeve" in returns and "voo" in returns:
        active, voo = returns.active_equity_sleeve, returns.voo
        rolling_beta = active.rolling(60).cov(voo) / voo.rolling(60).var()
        fig, ax = plt.subplots(figsize=(10, 5)); rolling_beta.plot(ax=ax, label="Active beta vs VOO"); ax.axhline(1, color="gray", linestyle="--"); ax.set(title=f"Rolling 60-day beta ({date_range})", ylabel="Beta", xlabel="Date"); ax.legend(); _save(fig, output / "rolling_beta.png", note)
        fig, ax = plt.subplots(figsize=(10, 5)); active.rolling(60).corr(voo).plot(ax=ax, label="Active correlation vs VOO"); ax.set(title=f"Rolling 60-day correlation ({date_range})", ylabel="Correlation", xlabel="Date"); ax.legend(); _save(fig, output / "rolling_correlation.png", note)
        for benchmark in [b for b in ["voo", "vgt"] if b in returns]:
            excess = ((1 + active.fillna(0)).cumprod() / (1 + returns[benchmark].fillna(0)).cumprod()) - 1
            fig, ax = plt.subplots(figsize=(10, 5)); excess.plot(ax=ax, label=f"Active minus {benchmark.upper()}"); ax.axhline(0, color="gray", linewidth=.8); ax.set(title=f"Cumulative excess return vs {benchmark.upper()} ({date_range})", ylabel="Relative wealth - 1", xlabel="Date"); ax.legend(); _save(fig, output / f"excess_vs_{benchmark}.png", note)


def create_account_figures(daily: pd.DataFrame, holdings: pd.DataFrame, ticker_summary: pd.DataFrame, output_dir: str | Path, note: str = "Source: Firstrade actual fills/dividends + Tiingo raw close") -> None:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    # A stacked area chart cannot represent an account whose cash balance
    # crosses zero (for example, temporary margin use).  Lines preserve both
    # signs and keep chart generation from becoming a production failure.
    fig, ax = plt.subplots(figsize=(10, 5)); ax.plot(daily.index, daily["cash"], label="cash"); ax.plot(daily.index, daily["market_value"], label="market_value"); ax.axhline(0.0, color="gray", linewidth=.8); ax.set(title="Portfolio exposure over time", ylabel="USD-equivalent value", xlabel="Date"); ax.legend(); _save(fig, output / "allocation_over_time.png", note)
    if not holdings.empty:
        pivot = holdings.pivot_table(index="date", columns="ticker", values="market_value", aggfunc="sum", fill_value=0)
        total = pivot.sum(axis=1).replace(0, pd.NA)
        sgov = pivot.get("SGOV", pd.Series(0, index=pivot.index)) / total
        equity = 1 - sgov.fillna(0)
        fig, ax = plt.subplots(figsize=(10, 5)); pd.DataFrame({"SGOV": sgov, "Other invested assets": equity}).plot.area(ax=ax); ax.set(title="SGOV / invested asset allocation", ylabel="Weight", xlabel="Date"); _save(fig, output / "sgov_equity_allocation.png", note)
    if not ticker_summary.empty:
        ordered = ticker_summary.sort_values("total_pnl")
        fig, ax = plt.subplots(figsize=(10, 6)); ax.barh(ordered.ticker, ordered.total_pnl); ax.axvline(0, color="gray", linewidth=.8); ax.set(title="Ticker contribution to total P&L", xlabel="P&L", ylabel="Ticker"); _save(fig, output / "contribution.png", note)


def create_since_start_figures(
    returns: pd.DataFrame,
    output_dir: str | Path,
    *,
    start_date: str,
    note: str = "Strategy reset; actual sleeves use Tiingo raw close and broker dividends; benchmarks use Tiingo adjClose",
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    def wealth(columns: list[str]) -> pd.DataFrame:
        available = [column for column in columns if column in returns]
        return (1.0 + returns[available].fillna(0.0)).cumprod()

    core = [
        "full_account_since_start", "active_sleeve_since_start",
        "full_account_matched_voo", "full_account_matched_vgt",
        "cf_matched_voo", "cf_matched_vgt",
    ]
    fig, ax = plt.subplots(figsize=(11, 6))
    wealth(core).plot(ax=ax)
    ax.set(title=f"Strategy-reset equity curves since {start_date}", ylabel="Growth of $1", xlabel="Date")
    _save(fig, output / "equity_curve_since_20260401.png", note)

    fig, ax = plt.subplots(figsize=(11, 5))
    returns[[c for c in core if c in returns]].apply(drawdown_series).plot(ax=ax)
    ax.set(title=f"Strategy-reset drawdowns since {start_date}", ylabel="Drawdown", xlabel="Date")
    _save(fig, output / "drawdown_since_20260401.png", note)

    for benchmark, filename, label in [
        ("cf_matched_voo", "active_vs_cf_matched_voo.png", "VOO"),
        ("cf_matched_vgt", "active_vs_cf_matched_vgt.png", "VGT"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        wealth(["active_sleeve_since_start", benchmark]).plot(ax=ax)
        ax.set(title=f"Active sleeve vs cash-flow-matched {label}", ylabel="Growth of $1", xlabel="Date")
        _save(fig, output / filename, note)

    fig, ax = plt.subplots(figsize=(10, 5))
    wealth(["active_sleeve_since_start", "decision_matched_voo", "decision_matched_vgt"]).plot(ax=ax)
    ax.set(title="Active decisions vs contribution-only passive counterfactuals", ylabel="Growth of $1", xlabel="Date")
    _save(fig, output / "decision_matched_comparison.png", note)

    rolling_columns = [
        "active_sleeve_since_start", "cf_matched_voo", "cf_matched_vgt",
        "beta_matched_voo_sgov", "volatility_matched_voo_sgov",
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    (returns[[c for c in rolling_columns if c in returns]].rolling(20).std() * 252 ** 0.5).plot(ax=ax)
    ax.set(title="Rolling 20-day annualized volatility since strategy reset", ylabel="Volatility", xlabel="Date")
    _save(fig, output / "rolling_volatility_since_start.png", note)

    if "active_sleeve_since_start" in returns and "voo_total_return_since_start" in returns:
        active = returns["active_sleeve_since_start"]
        voo = returns["voo_total_return_since_start"]
        rolling_beta = active.rolling(60).cov(voo) / voo.rolling(60).var()
        fig, ax = plt.subplots(figsize=(10, 5))
        rolling_beta.plot(ax=ax, label="Active beta vs VOO")
        ax.axhline(1.0, color="gray", linestyle="--")
        ax.set(title="Rolling 60-day beta since strategy reset", ylabel="Beta", xlabel="Date")
        ax.legend()
        _save(fig, output / "rolling_beta_since_start.png", note)
