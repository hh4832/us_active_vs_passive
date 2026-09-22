from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .portfolio_accounting import AccountingResult, time_weighted_returns


@dataclass
class ResetPortfolio:
    daily: pd.DataFrame
    holdings: pd.DataFrame
    cashflows: pd.DataFrame
    opening_holdings: pd.DataFrame
    opening_cash: float


def _require_start_date(index: pd.Index, start_date: str | pd.Timestamp) -> pd.Timestamp:
    start = pd.Timestamp(start_date).normalize()
    if start not in index:
        raise ValueError(f"Strategy start date {start.date()} is not a Tiingo trading date")
    return start


def build_reset_portfolio(
    full_daily_accounting: AccountingResult,
    start_date: str | pd.Timestamp,
) -> ResetPortfolio:
    """Reset a reconstructed account at the start-date close without losing inventory.

    The full history remains the source of cash and holdings.  The first reset
    observation is a base value only, so all pre-start profit and loss is
    excluded while post-start broker deposits/withdrawals retain their original
    TWR treatment.
    """
    start = _require_start_date(full_daily_accounting.daily.index, start_date)
    daily = full_daily_accounting.daily.loc[start:].copy()
    daily.iloc[0, daily.columns.get_loc("external_cash_flow")] = 0.0
    daily.iloc[0, daily.columns.get_loc("twr")] = np.nan
    daily["portfolio_index"] = (1.0 + daily["twr"].fillna(0.0)).cumprod()
    daily.iloc[0, daily.columns.get_loc("portfolio_index")] = 1.0
    holdings = full_daily_accounting.holdings.loc[
        full_daily_accounting.holdings["date"].ge(start)
    ].copy()
    opening = holdings.loc[holdings["date"].eq(start)].copy()
    return ResetPortfolio(
        daily=daily,
        holdings=holdings,
        cashflows=pd.DataFrame(columns=[
            "date", "flow_type", "amount", "source", "destination",
            "related_ticker", "source_row",
        ]),
        opening_holdings=opening,
        opening_cash=float(daily.iloc[0]["cash"]),
    )


def build_active_sleeve_since_start(
    transactions: pd.DataFrame,
    prices: pd.DataFrame,
    full_accounting: AccountingResult,
    start_date: str | pd.Timestamp,
    included_tickers: set[str],
    *,
    execution_price_column: str = "raw_execution_price",
) -> ResetPortfolio:
    """Build an active sleeve with auditable trade-level external cash flows.

    The start-date state is the full-account end-of-day inventory marked at raw
    close.  Thereafter every active buy is funded by a contribution and every
    active sell is withdrawn to account cash.  Broker dividends are investment
    income and are withdrawn to account cash on receipt.  This keeps the sleeve
    free of an arbitrary allocation of the account-level cash balance.
    """
    prices = prices.sort_index().copy()
    start = _require_start_date(prices.index, start_date)
    if start not in full_accounting.daily.index:
        raise ValueError(f"Full account has no state on {start.date()}")
    if execution_price_column not in transactions:
        raise ValueError(f"Missing execution price column: {execution_price_column}")

    opening = full_accounting.holdings.loc[
        full_accounting.holdings["date"].eq(start)
        & full_accounting.holdings["ticker"].isin(included_tickers)
    ].copy()
    positions = {ticker: 0.0 for ticker in prices.columns}
    for row in opening.itertuples(index=False):
        positions[str(row.ticker)] = float(row.shares)

    tx = transactions.loc[
        transactions["ticker"].isin(included_tickers)
        & transactions["type"].isin(["buy", "sell", "dividend", "fee"])
        & transactions["date"].gt(start)
    ].sort_values(["date", "source_row"])
    dates = prices.loc[start:].index
    opening_capital = sum(
        qty * float(prices.loc[start, ticker])
        for ticker, qty in positions.items() if abs(qty) > 1e-12
    )
    cashflow_rows: list[dict] = [{
        "date": start, "flow_type": "opening_capital", "amount": opening_capital,
        "source": "opening_inventory", "destination": "active_sleeve",
        "related_ticker": "", "source_row": pd.NA,
    }]
    daily_rows: list[dict] = []
    holding_rows: list[dict] = []
    ledger_rows: list[dict] = []
    previous_prices: pd.Series | None = None

    for date in dates:
        today_prices = prices.loc[date]
        day_tx = tx.loc[tx["date"].eq(date)]
        required = {
            ticker for ticker, qty in positions.items() if abs(qty) > 1e-12
        } | set(day_tx["ticker"].dropna().astype(str))
        missing = sorted(
            ticker for ticker in required
            if ticker not in prices or pd.isna(today_prices.get(ticker))
            or float(today_prices[ticker]) <= 0
        )
        if missing:
            raise ValueError(f"Missing active-sleeve market price on {date.date()}: {missing}")

        existing_pnl = 0.0 if previous_prices is None else sum(
            qty * (float(today_prices[t]) - float(previous_prices[t]))
            for t, qty in positions.items() if abs(qty) > 1e-12
        )
        trade_pnl = dividends = fees = external_flow = 0.0
        for idx, row in day_tx.iterrows():
            kind = str(row["type"])
            ticker = str(row["ticker"])
            source_row = row.get("source_row", idx)
            if kind == "dividend":
                value = float(row.get("dividend", 0.0) or 0.0)
                dividends += value
                external_flow -= value
                cashflow_rows.append({
                    "date": date, "flow_type": "withdrawal", "amount": -value,
                    "source": "active_sleeve", "destination": "cash",
                    "related_ticker": ticker, "source_row": source_row,
                })
                ledger_rows.append({
                    "row": idx, "date": date, "type": kind, "ticker": ticker,
                    "cash_delta": value, "external_cash_flow": -value,
                })
                continue
            if kind == "fee":
                value = abs(float(row.get("fee", 0.0) or row.get("amount", 0.0) or 0.0))
                fees += value
                external_flow += value
                cashflow_rows.append({
                    "date": date, "flow_type": "contribution", "amount": value,
                    "source": "cash", "destination": "active_sleeve",
                    "related_ticker": ticker, "source_row": source_row,
                })
                continue

            quantity = float(row["quantity"])
            execution = float(row[execution_price_column])
            fee = abs(float(row.get("fee", 0.0) or 0.0))
            close = float(today_prices[ticker])
            previous = close if previous_prices is None else float(previous_prices[ticker])
            before = positions.get(ticker, 0.0)
            if kind == "buy":
                positions[ticker] = before + quantity
                amount = quantity * execution + fee
                external_flow += amount
                trade_pnl += quantity * (close - execution) - fee
                flow_type, source, destination = "contribution", "cash_or_excluded_assets", "active_sleeve"
            else:
                after = before - quantity
                if after < -1e-9:
                    raise ValueError(f"Negative active holding for {ticker} on {date.date()}: {after}")
                positions[ticker] = after
                amount = -(quantity * execution - fee)
                external_flow += amount
                existing_pnl -= quantity * (close - previous)
                trade_pnl += quantity * (execution - previous) - fee
                flow_type, source, destination = "withdrawal", "active_sleeve", "cash"
            fees += fee
            cashflow_rows.append({
                "date": date, "flow_type": flow_type, "amount": amount,
                "source": source, "destination": destination,
                "related_ticker": ticker, "source_row": source_row,
            })
            ledger_rows.append({
                "row": idx, "date": date, "type": kind, "ticker": ticker,
                "quantity": quantity, "execution_price": execution,
                "close": close, "fee": fee, "external_cash_flow": amount,
                "shares_after": positions[ticker],
            })

        market_value = sum(
            qty * float(today_prices[ticker])
            for ticker, qty in positions.items() if abs(qty) > 1e-12
        )
        daily_rows.append({
            "date": date, "cash": 0.0, "market_value": market_value,
            "nav": market_value,
            "external_cash_flow": opening_capital if date == start else external_flow,
            "existing_position_market_pnl": existing_pnl,
            "trade_execution_pnl": trade_pnl,
            "dividend_income": dividends, "fees": fees,
            "investment_pnl": existing_pnl + trade_pnl + dividends,
        })
        holding_rows.extend({
            "date": date, "ticker": ticker, "shares": qty,
            "price": float(today_prices[ticker]),
            "market_value": qty * float(today_prices[ticker]),
        } for ticker, qty in positions.items() if abs(qty) > 1e-12)
        previous_prices = today_prices

    daily = pd.DataFrame(daily_rows).set_index("date")
    daily["twr"] = time_weighted_returns(daily["nav"], daily["external_cash_flow"])
    daily["portfolio_index"] = (1.0 + daily["twr"].fillna(0.0)).cumprod()
    daily.iloc[0, daily.columns.get_loc("portfolio_index")] = 1.0
    previous_nav = daily["nav"].shift(1)
    daily["reconciliation_difference"] = (
        daily["nav"] - previous_nav - daily["external_cash_flow"] - daily["investment_pnl"]
    )
    return ResetPortfolio(
        daily=daily,
        holdings=pd.DataFrame(holding_rows),
        cashflows=pd.DataFrame(cashflow_rows),
        opening_holdings=opening,
        opening_cash=0.0,
    )


def decision_matched_cashflows(
    transactions: pd.DataFrame,
    dates: pd.Index,
    start_date: str | pd.Timestamp,
    included_tickers: set[str],
    opening_capital: float,
    *,
    execution_price_column: str = "raw_execution_price",
) -> tuple[pd.Series, pd.DataFrame]:
    """Return contribution-only counterfactual flows for active buy decisions."""
    start = _require_start_date(dates, start_date)
    flows = pd.Series(0.0, index=dates, name="external_cash_flow")
    flows.loc[start] = float(opening_capital)
    rows = [{
        "date": start, "active_ticker": "OPENING_ACTIVE_INVENTORY",
        "active_amount": float(opening_capital), "source_row": pd.NA,
    }]
    buys = transactions.loc[
        transactions["date"].gt(start)
        & transactions["type"].eq("buy")
        & transactions["ticker"].isin(included_tickers)
    ].sort_values(["date", "source_row"])
    for _, row in buys.iterrows():
        date = pd.Timestamp(row["date"])
        if date not in flows.index:
            raise ValueError(f"Active buy date {date.date()} is not in benchmark prices")
        amount = float(row["quantity"]) * float(row[execution_price_column]) + abs(float(row.get("fee", 0.0) or 0.0))
        flows.loc[date] += amount
        rows.append({
            "date": date, "active_ticker": str(row["ticker"]),
            "active_amount": amount, "source_row": row.get("source_row", pd.NA),
        })
    return flows, pd.DataFrame(rows)


def opening_state_audits(
    full_accounting: AccountingResult,
    prices: pd.DataFrame,
    start_date: str | pd.Timestamp,
    active_tickers: set[str],
    known_funds: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = _require_start_date(full_accounting.daily.index, start_date)
    opening = full_accounting.holdings.loc[
        full_accounting.holdings["date"].eq(start)
    ].copy()
    rows = []
    for row in opening.itertuples(index=False):
        ticker = str(row.ticker)
        if ticker in active_tickers:
            classification = "active_risk_asset"
        elif ticker == "SGOV":
            classification = "cash_equivalent"
        elif ticker in {"VOO", "VGT"}:
            classification = "passive_equity"
        elif ticker in known_funds:
            classification = "other_fund"
        else:
            classification = "excluded_security"
        raw_close = float(prices.loc[start, ticker])
        rows.append({
            "ticker": ticker, "shares": float(row.shares), "raw_close": raw_close,
            "market_value": float(row.shares) * raw_close,
            "classification": classification,
            "included_in_active_sleeve": ticker in active_tickers,
            "included_in_full_account": True,
        })
    state = pd.DataFrame(rows, columns=[
        "ticker", "shares", "raw_close", "market_value", "classification",
        "included_in_active_sleeve", "included_in_full_account",
    ])
    cash = float(full_accounting.daily.loc[start, "cash"])
    total_securities = float(state["market_value"].sum()) if not state.empty else 0.0
    active_value = float(state.loc[state["included_in_active_sleeve"], "market_value"].sum()) if not state.empty else 0.0
    sgov_value = float(state.loc[state["ticker"].eq("SGOV"), "market_value"].sum()) if not state.empty else 0.0
    passive_value = total_securities - active_value - sgov_value
    cash_summary = pd.DataFrame([{
        "date": start, "reconstructed_cash": cash,
        "total_securities_market_value": total_securities,
        "total_account_nav": cash + total_securities,
        "active_sleeve_market_value": active_value,
        "passive_holdings_market_value": passive_value,
        "sgov_market_value": sgov_value,
    }])
    return state, cash_summary


def active_sleeve_attribution(
    transactions: pd.DataFrame,
    active: ResetPortfolio,
    start_date: str | pd.Timestamp,
    included_tickers: set[str],
    *,
    execution_price_column: str = "raw_execution_price",
) -> pd.DataFrame:
    """FIFO attribution with opening inventory rebased to the strategy-start close."""
    start = pd.Timestamp(start_date).normalize()
    final_date = active.daily.index.max()
    opening = active.opening_holdings.set_index("ticker") if not active.opening_holdings.empty else pd.DataFrame()
    ending = active.holdings.loc[active.holdings["date"].eq(final_date)]
    ending = ending.set_index("ticker") if not ending.empty else pd.DataFrame()
    post = transactions.loc[
        transactions["date"].gt(start)
        & transactions["ticker"].isin(included_tickers)
        & transactions["type"].isin(["buy", "sell", "dividend"])
    ].sort_values(["date", "source_row"])
    tickers = sorted(set(post["ticker"].dropna().astype(str)) | set(active.opening_holdings.get("ticker", [])))

    wealth = active.daily["portfolio_index"]
    drawdown = wealth / wealth.cummax() - 1.0
    trough = drawdown.idxmin()
    peak = wealth.loc[:trough].idxmax()
    holding_values = active.holdings.pivot_table(
        index="date", columns="ticker", values="market_value", aggfunc="sum", fill_value=0.0,
    ).reindex(active.daily.index, fill_value=0.0)
    rows = []
    opening_total = float(active.daily.loc[start, "nav"])
    peak_nav = float(active.daily.loc[peak, "nav"])
    for ticker in tickers:
        lots: deque[list[float]] = deque()
        starting_shares = float(opening.loc[ticker, "shares"]) if ticker in opening.index else 0.0
        starting_price = float(opening.loc[ticker, "price"]) if ticker in opening.index else np.nan
        if starting_shares:
            lots.append([starting_shares, starting_price * starting_shares])
        realized = net_contribution = dividends = 0.0
        group = post.loc[post["ticker"].eq(ticker)]
        for _, tx in group.iterrows():
            kind = str(tx["type"])
            if kind == "dividend":
                dividends += float(tx.get("dividend", 0.0) or 0.0)
                continue
            qty = float(tx["quantity"])
            price = float(tx[execution_price_column])
            fee = abs(float(tx.get("fee", 0.0) or 0.0))
            if kind == "buy":
                cost = qty * price + fee
                lots.append([qty, cost])
                net_contribution += cost
            else:
                proceeds = qty * price - fee
                net_contribution -= proceeds
                remaining = qty
                sold_cost = 0.0
                while remaining > 1e-12:
                    if not lots:
                        raise ValueError(f"Attribution sell exceeds reset inventory for {ticker}")
                    lot_qty, lot_cost = lots[0]
                    matched = min(remaining, lot_qty)
                    allocated = lot_cost * matched / lot_qty
                    sold_cost += allocated
                    remaining -= matched
                    if matched == lot_qty:
                        lots.popleft()
                    else:
                        lots[0] = [lot_qty - matched, lot_cost - allocated]
                realized += proceeds - sold_cost
        remaining_cost = sum(cost for _, cost in lots)
        ending_value = float(ending.loc[ticker, "market_value"]) if ticker in ending.index else 0.0
        unrealized = ending_value - remaining_cost
        total_pnl = realized + unrealized + dividends
        ticker_flows = active.cashflows.loc[active.cashflows["related_ticker"].eq(ticker)].copy()
        window_flow = float(ticker_flows.loc[
            ticker_flows["date"].gt(peak) & ticker_flows["date"].le(trough), "amount"
        ].sum())
        peak_value = float(holding_values.loc[peak].get(ticker, 0.0))
        trough_value = float(holding_values.loc[trough].get(ticker, 0.0))
        rows.append({
            "ticker": ticker,
            "starting_market_value": starting_shares * starting_price if starting_shares else 0.0,
            "net_contribution": net_contribution,
            "realized_pnl": realized, "unrealized_pnl": unrealized,
            "dividend": dividends, "total_pnl": total_pnl,
            "return_contribution": total_pnl / opening_total if opening_total else np.nan,
            "drawdown_contribution": (trough_value - peak_value - window_flow) / peak_nav if peak_nav else np.nan,
        })
    return pd.DataFrame(rows)
