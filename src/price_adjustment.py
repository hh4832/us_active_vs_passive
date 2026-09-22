from __future__ import annotations

import numpy as np
import pandas as pd


def adjustment_factor(close: float, adj_close: float) -> float:
    if not np.isfinite(close) or close <= 0 or not np.isfinite(adj_close) or adj_close <= 0:
        raise ValueError(f"Invalid close/adj_close: {close}/{adj_close}")
    return float(adj_close / close)


def attach_adjusted_execution_prices(transactions: pd.DataFrame, close: pd.DataFrame, adj_close: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = transactions.copy()
    warnings: list[str] = []
    factors, adjusted, raw_close, raw_adj = [], [], [], []
    for idx, row in out.iterrows():
        if row["type"] not in {"buy", "sell"} or pd.isna(row["ticker"]) or pd.isna(row["date"]):
            factors.append(np.nan); adjusted.append(np.nan); raw_close.append(np.nan); raw_adj.append(np.nan); continue
        ticker, date = str(row["ticker"]), pd.Timestamp(row["date"])
        try:
            c, ac = float(close.loc[date, ticker]), float(adj_close.loc[date, ticker])
            factor = adjustment_factor(c, ac)
            if factor < 0.01 or factor > 100:
                warnings.append(f"Extreme adjustment factor {factor:.6g} for {ticker} on {date.date()}")
            factors.append(factor); adjusted.append(float(row["price"]) * factor); raw_close.append(c); raw_adj.append(ac)
        except (KeyError, TypeError, ValueError) as exc:
            warnings.append(f"Missing/invalid adjustment inputs for row {idx} ({ticker}, {date.date()}): {exc}")
            factors.append(np.nan); adjusted.append(np.nan); raw_close.append(np.nan); raw_adj.append(np.nan)
    out["raw_execution_price"] = out["price"]
    out["close"] = raw_close
    out["adj_close"] = raw_adj
    out["adjustment_factor"] = factors
    out["adjusted_execution_price"] = adjusted
    return out, warnings

