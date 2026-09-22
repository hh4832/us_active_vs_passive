from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .performance import performance_metrics, relative_metrics, xirr
from .portfolio_accounting import time_weighted_returns
from .regime import capture_ratios


@dataclass
class NewMoneyCohortResult:
    name: str
    daily: pd.DataFrame
    holdings: pd.DataFrame
    cashflows: pd.DataFrame
    trade_audit: pd.DataFrame
    dividend_audit: pd.DataFrame
    warnings: list[str]


def _parse_eligible_shares(description: object) -> float | None:
    match = re.search(r"\bON\s+([0-9]+(?:\.[0-9]+)?)\s+SHS\b", str(description), flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def _parse_record_date(description: object, payment_date: pd.Timestamp) -> pd.Timestamp | None:
    match = re.search(r"\bREC\s+(\d{1,2}/\d{1,2}/\d{2,4})\b", str(description), flags=re.IGNORECASE)
    if not match:
        return None
    parsed = pd.to_datetime(match.group(1), errors="coerce")
    if pd.isna(parsed):
        return None
    parsed = pd.Timestamp(parsed).normalize()
    if parsed.year > payment_date.year + 1:
        parsed = parsed.replace(year=parsed.year - 100)
    return parsed


def build_new_money_cohort(
    name: str,
    transactions: pd.DataFrame,
    prices: pd.DataFrame,
    start_date: str | pd.Timestamp,
    included_tickers: set[str],
    *,
    execution_price_column: str = "raw_execution_price",
) -> NewMoneyCohortResult:
    """Track only active-security lots purchased on or after the research start.

    Every cohort buy is an external contribution.  Cohort-attributable sell
    proceeds and broker net dividends are withdrawals, so terminal-equivalent
    wealth can include them without treating cash transfers as investment P&L.
    Pre-start inventory is never introduced into this ledger.
    """
    prices = prices.sort_index().copy()
    start = pd.Timestamp(start_date).normalize()
    if start not in prices.index:
        raise ValueError(f"Research start date {start.date()} is not a trading date")
    dates = prices.loc[start:].index
    tx = transactions.loc[
        transactions["date"].ge(start)
        & transactions["ticker"].isin(included_tickers)
        & transactions["type"].isin(["buy", "sell", "split", "dividend"])
    ].sort_values(["date", "source_row"])
    positions = {ticker: 0.0 for ticker in prices.columns}
    snapshots: dict[pd.Timestamp, dict[str, float]] = {}
    daily_rows: list[dict] = []
    holding_rows: list[dict] = []
    cashflow_rows: list[dict] = []
    trade_rows: list[dict] = []
    dividend_rows: list[dict] = []
    warnings: list[str] = []
    previous_prices: pd.Series | None = None

    for date in dates:
        today_prices = prices.loc[date]
        day_tx = tx.loc[tx["date"].eq(date)]
        comparable_previous = previous_prices.copy() if previous_prices is not None else None

        for idx, row in day_tx.loc[day_tx["type"].eq("split")].iterrows():
            ticker = str(row["ticker"])
            before = positions.get(ticker, 0.0)
            if before <= 0:
                continue
            broker_before = _parse_eligible_shares(row.get("description", ""))
            issued = float(row["quantity"])
            if not broker_before or broker_before <= 0 or issued <= 0:
                raise ValueError(f"Cannot allocate cohort split for {ticker} at source row {row.get('source_row', idx)}")
            ratio = (broker_before + issued) / broker_before
            positions[ticker] = before * ratio
            if comparable_previous is not None:
                comparable_previous.loc[ticker] = float(previous_prices[ticker]) / ratio

        required = {
            ticker for ticker, qty in positions.items() if abs(qty) > 1e-12
        } | set(day_tx.loc[day_tx["type"].isin(["buy", "sell"]), "ticker"].dropna().astype(str))
        missing = sorted(
            ticker for ticker in required
            if ticker not in prices or pd.isna(today_prices.get(ticker)) or float(today_prices[ticker]) <= 0
        )
        if missing:
            raise ValueError(f"Missing new-money cohort market price on {date.date()}: {missing}")

        existing_pnl = 0.0 if comparable_previous is None else sum(
            qty * (float(today_prices[ticker]) - float(comparable_previous[ticker]))
            for ticker, qty in positions.items() if abs(qty) > 1e-12
        )
        trade_pnl = dividend_income = external_flow = 0.0
        for idx, row in day_tx.loc[day_tx["type"].isin(["buy", "sell"])].iterrows():
            ticker = str(row["ticker"])
            actual_type = str(row["type"])
            actual_quantity = float(row["quantity"])
            before = positions.get(ticker, 0.0)
            execution = float(row[execution_price_column])
            if not np.isfinite(execution) or execution <= 0:
                raise ValueError(f"Cannot calculate cohort contribution at source row {row.get('source_row', idx)}")
            fee = abs(float(row.get("fee", 0.0) or 0.0))
            close = float(today_prices[ticker])
            previous = close if comparable_previous is None else float(comparable_previous[ticker])
            cohort_buy = cohort_sell = ignored_sell = 0.0
            if actual_type == "buy":
                cohort_buy = actual_quantity
                positions[ticker] = before + cohort_buy
                cash_amount = cohort_buy * execution + fee
                external_flow += cash_amount
                trade_pnl += cohort_buy * (close - execution) - fee
                flow_type = "contribution"
            else:
                cohort_sell = min(actual_quantity, max(before, 0.0))
                ignored_sell = actual_quantity - cohort_sell
                positions[ticker] = before - cohort_sell
                allocated_fee = fee * cohort_sell / actual_quantity if actual_quantity else 0.0
                proceeds = cohort_sell * execution - allocated_fee
                cash_amount = -proceeds
                external_flow += cash_amount
                existing_pnl -= cohort_sell * (close - previous)
                trade_pnl += cohort_sell * (execution - previous) - allocated_fee
                flow_type = "withdrawal"
            if positions[ticker] < -1e-9:
                raise ValueError(f"Negative cohort holding for {ticker} on {date.date()}")
            trade_rows.append({
                "cohort": name, "date": date, "source_row": row.get("source_row", idx),
                "ticker": ticker, "actual_type": actual_type,
                "actual_quantity": actual_quantity, "cohort_quantity_before": before,
                "cohort_buy_quantity": cohort_buy, "cohort_sell_quantity": cohort_sell,
                "ignored_pre_start_sell_quantity": ignored_sell,
                "cohort_quantity_after": positions[ticker],
                "execution_price": execution, "cash_amount": cash_amount,
            })
            if abs(cash_amount) > 1e-12:
                cashflow_rows.append({
                    "date": date, "flow_type": flow_type, "amount": cash_amount,
                    "related_ticker": ticker, "source_row": row.get("source_row", idx),
                })

        # Dividend allocation uses broker-reported eligible total shares and the
        # cohort position on the stated record date.  The CSV amount is already
        # net of any withholding disclosed in the description.
        for idx, row in day_tx.loc[day_tx["type"].eq("dividend")].iterrows():
            ticker = str(row["ticker"])
            broker_cash = float(row.get("dividend", 0.0) or 0.0)
            eligible = _parse_eligible_shares(row.get("description", ""))
            record_date = _parse_record_date(row.get("description", ""), date)
            cohort_shares = 0.0
            method = "unallocated_missing_eligible_shares"
            allocated = 0.0
            if eligible and eligible > 0 and record_date is not None:
                eligible_dates = [snapshot_date for snapshot_date in snapshots if snapshot_date <= record_date]
                if eligible_dates:
                    cohort_shares = snapshots[max(eligible_dates)].get(ticker, 0.0)
                cohort_shares = min(max(cohort_shares, 0.0), eligible)
                allocated = broker_cash * cohort_shares / eligible
                method = "broker_net_cash_prorated_by_record_date_shares"
            else:
                warnings.append(
                    f"Dividend source row {row.get('source_row', idx)} could not be allocated to {name}"
                )
            if allocated:
                dividend_income += allocated
                external_flow -= allocated
                cashflow_rows.append({
                    "date": date, "flow_type": "withdrawal", "amount": -allocated,
                    "related_ticker": ticker, "source_row": row.get("source_row", idx),
                })
            dividend_rows.append({
                "cohort": name, "date": date, "ticker": ticker,
                "broker_dividend_cash": broker_cash,
                "broker_eligible_shares": eligible,
                "cohort_shares": cohort_shares,
                "cohort_dividend_allocated": allocated,
                "allocation_method": method,
                "source_row": row.get("source_row", idx),
            })

        market_value = sum(
            qty * float(today_prices[ticker])
            for ticker, qty in positions.items() if abs(qty) > 1e-12
        )
        daily_rows.append({
            "date": date, "cash": 0.0, "market_value": market_value,
            "nav": market_value, "external_cash_flow": external_flow,
            "existing_position_market_pnl": existing_pnl,
            "trade_execution_pnl": trade_pnl,
            "dividend_income": dividend_income,
            "investment_pnl": existing_pnl + trade_pnl + dividend_income,
        })
        holding_rows.extend({
            "date": date, "ticker": ticker, "shares": qty,
            "price": float(today_prices[ticker]),
            "market_value": qty * float(today_prices[ticker]),
        } for ticker, qty in positions.items() if abs(qty) > 1e-12)
        snapshots[date] = dict(positions)
        previous_prices = today_prices

    daily = pd.DataFrame(daily_rows).set_index("date")
    daily["twr"] = time_weighted_returns(daily["nav"], daily["external_cash_flow"])
    daily["portfolio_index"] = (1.0 + daily["twr"].fillna(0.0)).cumprod()
    daily["reconciliation_difference"] = (
        daily["nav"] - daily["nav"].shift(1)
        - daily["external_cash_flow"] - daily["investment_pnl"]
    )
    return NewMoneyCohortResult(
        name=name, daily=daily, holdings=pd.DataFrame(holding_rows),
        cashflows=pd.DataFrame(cashflow_rows, columns=[
            "date", "flow_type", "amount", "related_ticker", "source_row",
        ]),
        trade_audit=pd.DataFrame(trade_rows),
        dividend_audit=pd.DataFrame(dividend_rows), warnings=warnings,
    )


def summarize_new_money_result(
    name: str,
    daily: pd.DataFrame,
    voo_returns: pd.Series,
    *,
    risk_free_rate: float,
    annualization: int,
    transaction_cashflows: pd.DataFrame | None = None,
) -> dict[str, float | str]:
    flows = daily["external_cash_flow"].astype(float)
    if transaction_cashflows is None:
        contributed = float(flows.clip(lower=0).sum())
        withdrawn = float(-flows.clip(upper=0).sum())
    else:
        contributed = float(transaction_cashflows.loc[
            transaction_cashflows["flow_type"].eq("contribution"), "amount"
        ].sum())
        withdrawn = float(-transaction_cashflows.loc[
            transaction_cashflows["flow_type"].eq("withdrawal"), "amount"
        ].sum())
    ending_market = float(daily.iloc[-1].get("market_value", daily.iloc[-1]["nav"]))
    ending_cash = float(daily.iloc[-1].get("cash", 0.0))
    ending_wealth = ending_market + ending_cash + withdrawn
    dollar_pnl = ending_wealth - contributed
    net_contributed = contributed - withdrawn
    return_column = "twr" if "twr" in daily.columns else "return"
    returns = daily[return_column]
    twr_metrics = performance_metrics(
        returns, risk_free_rate=risk_free_rate, annualization=annualization,
    )
    relative = relative_metrics(returns, voo_returns, annualization=annualization)
    captures = capture_ratios(returns, voo_returns).set_index("regime")
    return {
        "portfolio": name,
        "total_contributed_capital": contributed,
        "total_withdrawn_capital": withdrawn,
        "net_contributed_capital": net_contributed,
        "ending_market_value": ending_market,
        "ending_cash": ending_cash,
        "ending_wealth": ending_wealth,
        "dollar_pnl": dollar_pnl,
        "return_on_net_contributed_capital": dollar_pnl / net_contributed if net_contributed else np.nan,
        "return_on_capital": dollar_pnl / contributed if contributed else np.nan,
        "xirr_mwr": xirr(flows, ending_market + ending_cash, daily.index.max()),
        "twr": twr_metrics["cumulative_return"],
        "annualized_volatility": twr_metrics["annualized_volatility"],
        "max_drawdown": twr_metrics["max_drawdown"],
        "sharpe": twr_metrics["sharpe"], "sortino": twr_metrics["sortino"],
        "beta_vs_voo": relative["beta"],
        "upside_capture": captures.loc["up", "arithmetic_daily_capture"],
        "downside_capture": captures.loc["down", "arithmetic_daily_capture"],
    }
