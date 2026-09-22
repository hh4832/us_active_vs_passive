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
    seed = previous.fillna(0.0).abs().lt(1e-12) & flow.gt(0)
    result.loc[seed] = (nav.loc[seed] - flow.loc[seed]) / flow.loc[seed]
    idle = previous.fillna(0.0).abs().lt(1e-12) & flow.abs().lt(1e-12)
    result.loc[idle] = np.nan
    return result.replace([np.inf, -np.inf], np.nan)


def reconstruct_portfolio(
    transactions: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    included_tickers: set[str] | None = None,
    allow_negative: bool = False,
    execution_price_column: str = "adjusted_execution_price",
) -> AccountingResult:
    """Reconstruct cash, holdings, NAV, TWR and same-day P&L decomposition.

    Execution prices and valuation prices must use the same scale. Production
    actual-account reconstruction passes raw Tiingo close prices together with
    the broker's raw execution prices and actual cash dividends.
    Transactions without intraday timestamps are applied at their recorded fill,
    and remaining holdings are marked to that day's close.
    """
    tx = transactions.copy()
    if included_tickers is not None:
        security = tx["ticker"].isin(included_tickers)
        cash_only = tx["type"].isin([
            "deposit", "withdrawal", "interest", "withholding_tax", "fee",
            "margin_interest_expense", "internal_transfer", "non_investment_credit",
        ])
        tx = tx[security | cash_only].copy()
    if tx.empty:
        raise ValueError("No transactions remain for portfolio reconstruction")
    if execution_price_column not in tx.columns:
        raise ValueError(f"Missing execution price column: {execution_price_column}")
    prices = prices.sort_index().copy()
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
        comparable_previous = previous_prices.copy() if previous_prices is not None else None
        split_value_adjustment = 0.0
        for idx, row in day_tx.loc[day_tx["type"].eq("split")].iterrows():
            ticker = str(row["ticker"])
            issued = float(row["quantity"])
            if abs(float(row.get("price", 0.0) or 0.0)) > 1e-12 or abs(float(row.get("amount", 0.0) or 0.0)) > 1e-12:
                raise ValueError(f"Split row {idx} has non-zero price or amount")
            before = positions.get(ticker, 0.0)
            if before <= 0 or issued <= 0:
                raise ValueError(f"Cannot apply split for {ticker} on {date.date()}: before={before}, issued={issued}")
            ratio = (before + issued) / before
            positions[ticker] = before + issued
            if comparable_previous is not None:
                comparable_previous.loc[ticker] = float(previous_prices[ticker]) / ratio
            ledger_rows.append({
                "row": idx, "date": date, "type": "split", "ticker": ticker,
                "quantity": issued, "cash_delta": 0.0, "shares_after": positions[ticker],
                "split_ratio": ratio,
            })
        required_today = {
            str(t) for t, qty in positions.items() if abs(qty) > 1e-12
        } | set(day_tx.loc[day_tx["type"].isin(["buy", "sell", "split"]), "ticker"].dropna().astype(str))
        missing_today = sorted(
            ticker for ticker in required_today
            if ticker not in prices.columns or pd.isna(today_prices.get(ticker)) or float(today_prices[ticker]) <= 0
        )
        if missing_today:
            raise ValueError(f"Missing held/traded market price on {date.date()}: {missing_today}")
        existing_pnl = 0.0 if comparable_previous is None else sum(
            qty * (float(today_prices[t]) - float(comparable_previous[t]))
            for t, qty in positions.items() if qty and pd.notna(today_prices.get(t)) and pd.notna(comparable_previous.get(t))
        )
        buy_pnl = sell_pnl = dividends = interest_income = taxes = fees = external_flow = 0.0
        margin_interest = non_investment_credit = 0.0
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
            if kind == "interest":
                if pd.isna(row["cash_flow"]):
                    raise ValueError(f"Interest row {idx} has no auditable cash amount")
                value = abs(float(row["cash_flow"]))
                cash += value; interest_income += value
                ledger_rows.append({"row": idx, "date": date, "type": kind, "cash_delta": value})
                continue
            if kind == "margin_interest_expense":
                value = abs(float(row["cash_flow"] or row.get("amount", 0.0) or 0.0))
                cash -= value; margin_interest += value
                ledger_rows.append({"row": idx, "date": date, "type": kind, "cash_delta": -value})
                continue
            if kind == "non_investment_credit":
                value = abs(float(row["cash_flow"] or row.get("amount", 0.0) or 0.0))
                cash += value; external_flow += value; non_investment_credit += value
                ledger_rows.append({"row": idx, "date": date, "type": kind, "cash_delta": value, "external_cash_flow": value})
                continue
            if kind in {"internal_transfer", "corporate_action", "split"}:
                if kind != "split":
                    ledger_rows.append({"row": idx, "date": date, "type": kind, "cash_delta": 0.0})
                continue
            if kind == "withholding_tax":
                if pd.isna(row["cash_flow"]):
                    raise ValueError(f"Withholding-tax row {idx} has no auditable cash amount")
                value = abs(float(row["cash_flow"]))
                cash -= value; taxes += value
                ledger_rows.append({"row": idx, "date": date, "type": kind, "cash_delta": -value})
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
                warnings.append(f"Skipped row {idx}: no market price series for {ticker}")
                continue
            execution = float(row[execution_price_column])
            quantity = float(row["quantity"])
            fee = float(row.get("fee", 0) or 0)
            close = float(today_prices[ticker])
            previous = close if comparable_previous is None or pd.isna(comparable_previous.get(ticker)) else float(comparable_previous[ticker])
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
                           "dividend_income": dividends, "interest_income": interest_income,
                           "margin_interest_expense": margin_interest,
                           "non_investment_credit": non_investment_credit,
                           "split_value_adjustment": split_value_adjustment,
                           "withholding_tax": taxes, "fees": fees,
                           "investment_pnl": existing_pnl + buy_pnl + sell_pnl + dividends + interest_income - taxes - fees - margin_interest})
        holding_rows.extend({"date": date, "ticker": ticker, "shares": qty, "price": float(today_prices[ticker]),
                             "market_value": qty * float(today_prices[ticker])}
                            for ticker, qty in positions.items() if abs(qty) > 1e-12)
        previous_prices = today_prices
    daily = pd.DataFrame(daily_rows).set_index("date")
    daily["twr"] = time_weighted_returns(daily["nav"], daily["external_cash_flow"])
    previous_nav = daily["nav"].shift(1)
    daily["reconciliation_difference"] = daily["nav"] - previous_nav - daily["external_cash_flow"] - daily["investment_pnl"]
    return AccountingResult(daily, pd.DataFrame(holding_rows), pd.DataFrame(ledger_rows), warnings)


def reconstruct_next_day_sensitivity(
    transactions: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    execution_price_column: str = "adjusted_execution_price",
    **kwargs,
) -> AccountingResult:
    shifted = transactions.copy()
    trading_dates = pd.Index(prices.index)
    def next_date(value: pd.Timestamp) -> pd.Timestamp:
        pos = trading_dates.searchsorted(value, side="right")
        return trading_dates[min(pos, len(trading_dates) - 1)]
    trade_mask = shifted["type"].isin(["buy", "sell"])
    shifted.loc[trade_mask, "date"] = shifted.loc[trade_mask, "date"].map(next_date)
    shifted.loc[trade_mask, execution_price_column] = [
        prices.loc[d, t]
        for d, t in zip(shifted.loc[trade_mask, "date"], shifted.loc[trade_mask, "ticker"])
    ]
    return reconstruct_portfolio(
        shifted, prices, execution_price_column=execution_price_column, **kwargs,
    )
