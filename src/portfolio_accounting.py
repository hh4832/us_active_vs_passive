from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AccountingResult:
    daily: pd.DataFrame
    holdings: pd.DataFrame
    transaction_ledger: pd.DataFrame
    warnings: list[str]


def time_weighted_returns(nav: pd.Series, external_cash_flow: pd.Series) -> pd.Series:
    """Daily TWR using the configured end-of-day-available cash-flow convention."""
    nav = nav.astype(float)
    flow = external_cash_flow.reindex(nav.index, fill_value=0).astype(float)
    previous = nav.shift(1)
    result = (nav - flow) / previous - 1
    result.iloc[0] = np.nan if abs(nav.iloc[0] - flow.iloc[0]) < 1e-12 else (nav.iloc[0] - flow.iloc[0]) / abs(flow.iloc[0])
    return result.replace([np.inf, -np.inf], np.nan)


def reconstruct_portfolio(
    transactions: pd.DataFrame,
    adj_close: pd.DataFrame,
    *,
    included_tickers: set[str] | None = None,
    allow_negative: bool = False,
) -> AccountingResult:
    """Reconstruct cash, holdings, NAV, TWR and same-day P&L decomposition.

    Execution prices must already be expressed on the adjusted-price scale.
    Transactions without intraday timestamps are applied at their recorded fill,
    and remaining holdings are marked to that day's close.
    """
    tx = transactions.copy()
    if included_tickers is not None:
        security = tx["ticker"].isin(included_tickers)
        tx = tx[security | tx["type"].isin(["deposit", "withdrawal"])].copy()
    if tx.empty:
        raise ValueError("No transactions remain for portfolio reconstruction")
    prices = adj_close.sort_index().copy().ffill()
    start = min(tx["date"].dropna().min(), prices.index.min())
    end = prices.index.max()
    dates = prices.loc[start:end].index
    tx = tx[tx["date"].isin(dates)].sort_values(["date", "source_row"])
    positions = {ticker: 0.0 for ticker in prices.columns}
    cash = 0.0
    warnings: list[str] = []
    daily_rows: list[dict] = []
    holding_rows: list[dict] = []
    ledger_rows: list[dict] = []
    previous_prices: pd.Series | None = None

    for date in dates:
        today_prices = prices.loc[date]
        day_tx = tx[tx["date"].eq(date)]
        existing_pnl = 0.0 if previous_prices is None else sum(
            qty * (float(today_prices[t]) - float(previous_prices[t]))
            for t, qty in positions.items() if qty and pd.notna(today_prices.get(t)) and pd.notna(previous_prices.get(t))
        )
        buy_pnl = sell_pnl = dividends = fees = external_flow = 0.0
        for idx, row in day_tx.iterrows():
            kind = row["type"]
            if kind in {"deposit", "withdrawal"}:
                value = float(row["cash_flow"])
                cash += value; external_flow += value
                ledger_rows.append({"row": idx, "date": date, "type": kind, "cash_delta": value})
                continue
            if kind == "dividend":
                value = float(row["dividend"])
                cash += value; dividends += value
                ledger_rows.append({"row": idx, "date": date, "type": kind, "ticker": row["ticker"], "cash_delta": value})
                continue
            if kind == "fee":
                value = float(row["fee"] or row["amount"] or 0)
                cash -= abs(value); fees += abs(value)
                continue
            if kind not in {"buy", "sell"}:
                warnings.append(f"Skipped unsupported transaction row {idx}: {kind}")
                continue
            ticker = str(row["ticker"])
            if ticker not in prices.columns:
                warnings.append(f"Skipped row {idx}: no adjusted price series for {ticker}")
                continue
            execution = float(row["adjusted_execution_price"])
            quantity = float(row["quantity"])
            fee = float(row.get("fee", 0) or 0)
            close = float(today_prices[ticker])
            previous = close if previous_prices is None or pd.isna(previous_prices.get(ticker)) else float(previous_prices[ticker])
            before = positions.get(ticker, 0.0)
            if kind == "buy":
                positions[ticker] = before + quantity
                cash -= quantity * execution + fee
                buy_pnl += quantity * (close - execution)
                # Remove the new shares from existing-position P&L: they were not held overnight.
            else:
                after = before - quantity
                if after < -1e-9 and not allow_negative:
                    raise ValueError(f"Negative holding for {ticker} on {date.date()}: {after}")
                positions[ticker] = after
                cash += quantity * execution - fee
                sell_pnl += quantity * (execution - previous)
                # Existing P&L assumed all opening shares reached close; replace sold shares' close leg.
                existing_pnl -= quantity * (close - previous)
            fees += fee
            ledger_rows.append({"row": idx, "date": date, "type": kind, "ticker": ticker, "quantity": quantity,
                                "execution_price": execution, "close": close, "fee": fee,
                                "cash_delta": (-1 if kind == "buy" else 1) * quantity * execution - fee,
                                "shares_after": positions[ticker]})
        market_value = sum(float(qty) * float(today_prices[t]) for t, qty in positions.items() if qty and pd.notna(today_prices.get(t)))
        nav = cash + market_value
        daily_rows.append({"date": date, "cash": cash, "market_value": market_value, "nav": nav,
                           "external_cash_flow": external_flow, "existing_position_market_pnl": existing_pnl,
                           "buy_execution_to_close_pnl": buy_pnl, "sell_previous_close_to_execution_pnl": sell_pnl,
                           "dividend_income": dividends, "fees": fees,
                           "investment_pnl": existing_pnl + buy_pnl + sell_pnl + dividends - fees})
        holding_rows.extend({"date": date, "ticker": ticker, "shares": qty, "price": float(today_prices[ticker]),
                             "market_value": qty * float(today_prices[ticker])}
                            for ticker, qty in positions.items() if abs(qty) > 1e-12)
        previous_prices = today_prices
    daily = pd.DataFrame(daily_rows).set_index("date")
    daily["twr"] = time_weighted_returns(daily["nav"], daily["external_cash_flow"])
    previous_nav = daily["nav"].shift(1)
    daily["reconciliation_difference"] = daily["nav"] - previous_nav - daily["external_cash_flow"] - daily["investment_pnl"]
    return AccountingResult(daily, pd.DataFrame(holding_rows), pd.DataFrame(ledger_rows), warnings)


def reconstruct_next_day_sensitivity(transactions: pd.DataFrame, adj_close: pd.DataFrame, **kwargs) -> AccountingResult:
    shifted = transactions.copy()
    trading_dates = pd.Index(adj_close.index)
    def next_date(value: pd.Timestamp) -> pd.Timestamp:
        pos = trading_dates.searchsorted(value, side="right")
        return trading_dates[min(pos, len(trading_dates) - 1)]
    trade_mask = shifted["type"].isin(["buy", "sell"])
    shifted.loc[trade_mask, "date"] = shifted.loc[trade_mask, "date"].map(next_date)
    shifted.loc[trade_mask, "adjusted_execution_price"] = [adj_close.loc[d, t] for d, t in zip(shifted.loc[trade_mask, "date"], shifted.loc[trade_mask, "ticker"])]
    return reconstruct_portfolio(shifted, adj_close, **kwargs)

