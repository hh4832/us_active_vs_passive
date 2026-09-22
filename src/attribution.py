from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd


def fifo_realized_pnl(
    transactions: pd.DataFrame,
    *,
    price_column: str = "adjusted_execution_price",
) -> pd.DataFrame:
    lots: dict[str, deque] = defaultdict(deque)
    rows: list[dict] = []
    for idx, row in transactions.sort_values(["date", "source_row"]).iterrows():
        if row["type"] == "split":
            ticker = str(row["ticker"])
            current = sum(lot[0] for lot in lots[ticker])
            issued = float(row["quantity"])
            if current <= 0 or issued <= 0:
                raise ValueError(f"Cannot apply FIFO split for {ticker} at row {idx}")
            ratio = (current + issued) / current
            for lot in lots[ticker]:
                lot[0] *= ratio
            continue
        if row["type"] not in {"buy", "sell"}:
            continue
        ticker, qty, price = str(row["ticker"]), float(row["quantity"]), float(row[price_column])
        if row["type"] == "buy":
            lots[ticker].append([qty, price, row["date"], float(row.get("fee", 0) or 0)])
            continue
        remaining = qty
        sell_fee_per_share = float(row.get("fee", 0) or 0) / qty if qty else 0
        while remaining > 1e-12:
            if not lots[ticker]:
                raise ValueError(f"FIFO sell exceeds available {ticker} shares at row {idx}")
            lot_qty, cost, buy_date, buy_fee = lots[ticker][0]
            matched = min(remaining, lot_qty)
            allocated_buy_fee = buy_fee * matched / lot_qty if lot_qty else 0
            pnl = matched * (price - cost) - allocated_buy_fee - matched * sell_fee_per_share
            rows.append({"ticker": ticker, "buy_date": buy_date, "sell_date": row["date"], "quantity": matched,
                         "buy_price": cost, "sell_price": price, "price_basis": price_column,
                         "realized_pnl": pnl, "holding_days": (row["date"] - buy_date).days})
            remaining -= matched
            if matched == lot_qty: lots[ticker].popleft()
            else:
                lots[ticker][0][0] -= matched
                lots[ticker][0][3] -= allocated_buy_fee
    return pd.DataFrame(rows)


def ticker_summary(
    transactions: pd.DataFrame,
    final_prices: pd.Series,
    *,
    price_column: str = "adjusted_execution_price",
) -> pd.DataFrame:
    fifo = fifo_realized_pnl(transactions, price_column=price_column)
    rows = []
    for ticker, group in transactions.dropna(subset=["ticker"]).groupby("ticker"):
        buys, sells = group[group.type.eq("buy")], group[group.type.eq("sell")]
        buy_qty, sell_qty = buys.quantity.sum(), sells.quantity.sum()
        split_qty = group.loc[group.type.eq("split"), "quantity"].sum()
        ending_qty = buy_qty + split_qty - sell_qty
        invested = (buys.quantity * buys[price_column]).sum()
        realized = fifo.loc[fifo.ticker.eq(ticker), "realized_pnl"].sum() if not fifo.empty else 0.0
        avg_buy = invested / buy_qty if buy_qty else np.nan
        avg_sell = (sells.quantity * sells[price_column]).sum() / sell_qty if sell_qty else np.nan
        # Remaining cost is computed by replaying FIFO indirectly: total buy cost minus cost basis of sold lots.
        sold_cost = (fifo.loc[fifo.ticker.eq(ticker), "quantity"] * fifo.loc[fifo.ticker.eq(ticker), "buy_price"]).sum() if not fifo.empty else 0
        remaining_cost = invested - sold_cost
        market_value = ending_qty * float(final_prices.get(ticker, np.nan))
        unrealized = market_value - remaining_cost if np.isfinite(market_value) else np.nan
        dividend = group.dividend.sum()
        rows.append({"ticker": ticker, "total_buys": len(buys), "total_sells": len(sells), "total_invested_capital": invested,
                     "realized_pnl": realized, "unrealized_pnl": unrealized, "dividend": dividend,
                     "total_pnl": realized + (unrealized if np.isfinite(unrealized) else 0) + dividend,
                     "weighted_average_buy_price": avg_buy, "weighted_average_sell_price": avg_sell,
                     "execution_price_basis": price_column,
                     "ending_shares": ending_qty})
    return pd.DataFrame(rows)


def covariance_volatility_contribution(asset_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    common = asset_returns.dropna().columns.intersection(weights.index)
    cov = asset_returns[common].dropna().cov().to_numpy()
    w = weights[common].to_numpy(dtype=float)
    variance = float(w @ cov @ w)
    if variance <= 0:
        return pd.Series(np.nan, index=common)
    return pd.Series(w * (cov @ w) / variance, index=common, name="variance_contribution")
