from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

CANONICAL_COLUMNS = ["date", "type", "ticker", "quantity", "price", "amount", "dividend", "cash_flow", "fee"]
TYPE_PATTERNS = {
    "buy": r"\b(buy|bought)\b|買進|買入",
    "sell": r"\b(sell|sold)\b|賣出",
    "dividend": r"dividend|qualified dividend|股息|配息",
    "deposit": r"deposit|wire in|ach in|入金|存入",
    "withdrawal": r"withdraw|wire out|ach out|出金|提款",
    "split": r"split|stock split|拆股",
    "fee": r"fee|commission|手續費",
}


@dataclass
class TransactionAudit:
    summary: pd.DataFrame
    ticker_summary: pd.DataFrame
    warnings: list[str]


def detect_columns(columns: Sequence[str], aliases: Mapping[str, Sequence[str]]) -> dict[str, str]:
    normalized = {str(c).strip().lower(): str(c) for c in columns}
    found: dict[str, str] = {}
    for canonical, candidates in aliases.items():
        for candidate in [canonical, *candidates]:
            if str(candidate).strip().lower() in normalized:
                found[canonical] = normalized[str(candidate).strip().lower()]
                break
    return found


def _number(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.replace(r"[$,()]", lambda m: "-" if m.group() == "(" else "", regex=True)
    return pd.to_numeric(text.str.replace(")", "", regex=False), errors="coerce")


def classify_type(value: object) -> str:
    text = str(value).strip().lower()
    for kind, pattern in TYPE_PATTERNS.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            return kind
    return "unknown"


def normalize_transactions(raw: pd.DataFrame, aliases: Mapping[str, Sequence[str]]) -> tuple[pd.DataFrame, dict[str, str]]:
    mapping = detect_columns(raw.columns, aliases)
    required = {"date", "type"}
    missing = required - mapping.keys()
    if missing:
        raise ValueError(f"Missing required transaction columns: {sorted(missing)}; detected={mapping}")
    out = pd.DataFrame(index=raw.index)
    for column in CANONICAL_COLUMNS:
        out[column] = raw[mapping[column]] if column in mapping else np.nan
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["raw_type"] = out["type"].astype("string")
    out["type"] = out["type"].map(classify_type)
    out["ticker"] = out["ticker"].astype("string").str.strip().str.upper().replace({"": pd.NA, "NAN": pd.NA})
    for column in ["quantity", "price", "amount", "dividend", "cash_flow", "fee"]:
        out[column] = _number(out[column])
    out["quantity"] = out["quantity"].abs()
    out["fee"] = out["fee"].fillna(0).abs()
    out["dividend"] = out["dividend"].fillna(0)
    out["cash_flow"] = out["cash_flow"].fillna(0)
    out.loc[out["type"] == "deposit", "cash_flow"] = out.loc[out["type"] == "deposit", ["cash_flow", "amount"]].bfill(axis=1).iloc[:, 0].abs()
    out.loc[out["type"] == "withdrawal", "cash_flow"] = -out.loc[out["type"] == "withdrawal", ["cash_flow", "amount"]].bfill(axis=1).iloc[:, 0].abs()
    out.loc[out["type"] == "dividend", "dividend"] = out.loc[out["type"] == "dividend", ["dividend", "amount"]].bfill(axis=1).iloc[:, 0].abs()
    out["source_row"] = raw.index
    return out.sort_values(["date", "source_row"]).reset_index(drop=True), mapping


def audit_transactions(tx: pd.DataFrame) -> TransactionAudit:
    warnings: list[str] = []
    if tx["date"].isna().any():
        warnings.append(f"{tx['date'].isna().sum()} rows have invalid/missing dates")
    unknown = tx["type"].eq("unknown")
    if unknown.any():
        warnings.append(f"{unknown.sum()} rows have unrecognized transaction types")
    trades = tx["type"].isin(["buy", "sell"])
    missing_price = trades & (tx["price"].isna() | tx["quantity"].isna())
    if missing_price.any():
        warnings.append(f"{missing_price.sum()} trades have missing price or quantity")
    positions: dict[str, float] = {}
    negative_rows: list[int] = []
    for idx, row in tx.loc[trades].iterrows():
        ticker = row["ticker"]
        if pd.isna(ticker):
            continue
        delta = row["quantity"] if row["type"] == "buy" else -row["quantity"]
        positions[str(ticker)] = positions.get(str(ticker), 0.0) + float(delta)
        if positions[str(ticker)] < -1e-9:
            negative_rows.append(int(idx))
    if negative_rows:
        warnings.append(f"Negative holdings detected at normalized rows: {negative_rows}")
    if tx["type"].eq("split").any():
        warnings.append("Split transaction(s) require explicit ratio validation")
    tickers = tx.dropna(subset=["ticker"]).groupby("ticker", sort=True)
    ticker_summary = tickers.agg(first_trade=("date", "min"), last_trade=("date", "max"))
    ticker_summary["buy_shares"] = tickers.apply(lambda g: g.loc[g.type.eq("buy"), "quantity"].sum(), include_groups=False)
    ticker_summary["sell_shares"] = tickers.apply(lambda g: g.loc[g.type.eq("sell"), "quantity"].sum(), include_groups=False)
    ticker_summary["ending_shares"] = ticker_summary["buy_shares"] - ticker_summary["sell_shares"]
    summary = pd.DataFrame([{
        "start_date": tx["date"].min(), "end_date": tx["date"].max(), "rows": len(tx),
        "buys": int(tx.type.eq("buy").sum()), "sells": int(tx.type.eq("sell").sum()),
        "dividends": int(tx.type.eq("dividend").sum()),
        "deposits_withdrawals": int(tx.type.isin(["deposit", "withdrawal"]).sum()),
        "unknown": int(unknown.sum()), "missing_trade_price_or_quantity": int(missing_price.sum()),
        "negative_holding_events": len(negative_rows),
    }])
    return TransactionAudit(summary, ticker_summary.reset_index(), warnings)

