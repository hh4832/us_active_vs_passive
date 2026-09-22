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
    "interest": r"\b(credit interest|interest income|cash interest)\b|利息收入",
    "withholding_tax": r"\b(withholding tax|tax withheld|foreign tax paid)\b|預扣稅|扣繳稅",
    "journal": r"\bjournal\b|轉帳分錄",
    "transfer": r"\b(stock transfer|transfer in|transfer out|account transfer)\b|轉入|轉出",
    "corporate_action": r"\b(corporate action|merger|spin[ -]?off|reorganization)\b|公司行動|合併|分拆",
}

CORE_TRANSACTION_TYPES = {"buy", "sell", "dividend", "deposit", "withdrawal", "split", "fee"}


@dataclass
class TransactionAudit:
    summary: pd.DataFrame
    ticker_summary: pd.DataFrame
    negative_holdings: pd.DataFrame
    transaction_type_diagnostic: pd.DataFrame
    unrecognized_types: pd.DataFrame
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
    interest_mask = out["type"].eq("interest")
    interest_values = out["cash_flow"].where(out["cash_flow"].ne(0), out["amount"])
    out.loc[interest_mask, "cash_flow"] = interest_values.loc[interest_mask].abs()
    tax_mask = out["type"].eq("withholding_tax")
    tax_values = out["cash_flow"].where(out["cash_flow"].ne(0), out["amount"])
    out.loc[tax_mask, "cash_flow"] = -tax_values.loc[tax_mask].abs()
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
    cash_values = tx["cash_flow"] if "cash_flow" in tx else pd.Series(np.nan, index=tx.index)
    missing_classified_cash = tx["type"].isin(["interest", "withholding_tax"]) & cash_values.isna()
    if missing_classified_cash.any():
        warnings.append(
            f"{missing_classified_cash.sum()} classified interest/withholding-tax rows have no auditable cash amount"
        )
    positions: dict[str, float] = {}
    negative_records: list[dict] = []
    for idx, row in tx.loc[trades].iterrows():
        ticker = row["ticker"]
        if pd.isna(ticker):
            continue
        before = positions.get(str(ticker), 0.0)
        delta = row["quantity"] if row["type"] == "buy" else -row["quantity"]
        after = before + float(delta)
        positions[str(ticker)] = after
        if after < -1e-9:
            negative_records.append({
                "normalized_row": int(idx),
                "source_row": row.get("source_row", idx),
                "ticker": str(ticker),
                "date": row["date"],
                "transaction_type": row["type"],
                "raw_type": row.get("raw_type", pd.NA),
                "quantity": row["quantity"],
                "cumulative_position_before": before,
                "cumulative_position_after": after,
                "status": "unresolved",
                "possible_causes": (
                    "opening position before export; transfer/journal; corporate action; "
                    "missing acquisition row; or normalization issue"
                ),
            })
    negative_holdings = pd.DataFrame(negative_records, columns=[
        "normalized_row", "source_row", "ticker", "date", "transaction_type", "raw_type", "quantity",
        "cumulative_position_before", "cumulative_position_after", "status", "possible_causes",
    ])
    if not negative_holdings.empty:
        warnings.append(f"Negative holdings detected at normalized rows: {negative_holdings['normalized_row'].tolist()}")
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
        "negative_holding_events": len(negative_holdings),
    }])

    review = tx.loc[~tx["type"].isin(CORE_TRANSACTION_TYPES)].copy()
    diagnostic_rows: list[dict] = []
    if not review.empty:
        for (raw_type, normalized_type), group in review.groupby(["raw_type", "type"], dropna=False, sort=True):
            source_rows = group["source_row"].tolist()
            examples = group[["source_row", "date", "ticker", "quantity", "amount"]].head(3).copy()
            examples["date"] = examples["date"].astype("string")
            diagnostic_rows.append({
                "raw_type": raw_type,
                "normalized_type": normalized_type,
                "count": len(group),
                "example_source_rows": ",".join(map(str, source_rows[:5])),
                "example_rows": examples.to_dict("records"),
                "review_status": "unresolved" if normalized_type == "unknown" else "classified_requires_semantic_review",
            })
    transaction_type_diagnostic = pd.DataFrame(diagnostic_rows, columns=[
        "raw_type", "normalized_type", "count", "example_source_rows", "example_rows", "review_status",
    ])
    unrecognized_types = transaction_type_diagnostic.loc[
        transaction_type_diagnostic["normalized_type"].eq("unknown")
    ].reset_index(drop=True)
    manual_types = transaction_type_diagnostic.loc[
        ~transaction_type_diagnostic["normalized_type"].eq("unknown"), "normalized_type"
    ].value_counts().to_dict()
    if manual_types:
        warnings.append(f"Transactions requiring semantic review after classification: {manual_types}")
    return TransactionAudit(
        summary=summary,
        ticker_summary=ticker_summary.reset_index(),
        negative_holdings=negative_holdings,
        transaction_type_diagnostic=transaction_type_diagnostic,
        unrecognized_types=unrecognized_types,
        warnings=warnings,
    )
