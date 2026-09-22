from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

CANONICAL_COLUMNS = [
    "date", "type", "description", "ticker", "quantity", "price",
    "amount", "dividend", "cash_flow", "fee",
]
TYPE_PATTERNS = {
    "buy": r"\b(buy|bought)\b|買進|買入",
    "sell": r"\b(sell|sold)\b|賣出",
    "dividend": r"dividend|qualified dividend|股息|配息",
    "deposit": r"deposit|wire in|wire funds received|ach in|存款|入金|存入",
    "withdrawal": r"withdraw|wire out|ach out|出金|提款",
    "split": r"split|stock split|拆股",
    "fee": r"fee|commission|手續費",
    "interest": r"\b(credit interest|interest income|cash interest)\b|利息收入",
    "withholding_tax": r"\b(withholding tax|tax withheld|foreign tax paid)\b|預扣稅|扣繳稅",
    "margin_interest_expense": r"margin interest expense|margin interest|融資利息費用",
    "internal_transfer": r"\bxfer\s+(cash\s+to\s+margin|margin\s+to\s+cash|ffs\s+to\s+cash|cash\s+from\s+ffs)\b",
    "non_investment_credit": r"\brebate for wire\b",
    "journal": r"\bjournal\b|轉帳分錄",
    "transfer": r"\b(stock transfer|transfer in|transfer out|account transfer)\b|轉入|轉出",
    "corporate_action": r"\b(corporate action|merger|spin[ -]?off|reorganization)\b|公司行動|合併|分拆",
}

CORE_TRANSACTION_TYPES = {
    "buy", "sell", "dividend", "deposit", "withdrawal", "split", "fee",
    "interest", "withholding_tax", "margin_interest_expense",
    "internal_transfer", "non_investment_credit", "corporate_action",
}


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


def classify_type(value: object, description: object = "") -> str:
    raw_text = str(value).strip().lower()
    description_text = str(description).strip().lower()
    # Description-level semantics override a generic Firstrade raw type such as
    # 買進 or 其他.  Keep these rules narrow and auditable.
    if re.search(r"\b(stk|stock)\s+split\b", description_text, flags=re.IGNORECASE):
        return "split"
    if re.search(TYPE_PATTERNS["internal_transfer"], description_text, flags=re.IGNORECASE):
        return "internal_transfer"
    if re.search(TYPE_PATTERNS["non_investment_credit"], description_text, flags=re.IGNORECASE):
        return "non_investment_credit"
    if re.search(r"\bname change\b", description_text, flags=re.IGNORECASE):
        return "corporate_action"
    text = f"{raw_text} {description_text}"
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
    out["description"] = out["description"].astype("string").fillna("")
    out["raw_description"] = out["description"]
    out["type"] = [
        classify_type(raw_type, description)
        for raw_type, description in zip(out["raw_type"], out["description"])
    ]
    out["ticker"] = out["ticker"].astype("string").str.strip().str.upper().replace({"": pd.NA, "NAN": pd.NA})
    for column in ["quantity", "price", "amount", "dividend", "cash_flow", "fee"]:
        out[column] = _number(out[column])
    out["quantity"] = out["quantity"].abs()
    out["fee"] = out["fee"].fillna(0).abs()
    out["dividend"] = out["dividend"].fillna(0)
    out["cash_flow"] = out["cash_flow"].fillna(0)
    deposit_mask = out["type"].eq("deposit")
    deposit_values = out["cash_flow"].where(out["cash_flow"].ne(0), out["amount"])
    out.loc[deposit_mask, "cash_flow"] = deposit_values.loc[deposit_mask].abs()
    withdrawal_mask = out["type"].eq("withdrawal")
    withdrawal_values = out["cash_flow"].where(out["cash_flow"].ne(0), out["amount"])
    out.loc[withdrawal_mask, "cash_flow"] = -withdrawal_values.loc[withdrawal_mask].abs()
    dividend_mask = out["type"].eq("dividend")
    dividend_values = out["dividend"].where(out["dividend"].ne(0), out["amount"])
    out.loc[dividend_mask, "dividend"] = dividend_values.loc[dividend_mask].abs()
    interest_mask = out["type"].eq("interest")
    interest_values = out["cash_flow"].where(out["cash_flow"].ne(0), out["amount"])
    out.loc[interest_mask, "cash_flow"] = interest_values.loc[interest_mask].abs()
    tax_mask = out["type"].eq("withholding_tax")
    tax_values = out["cash_flow"].where(out["cash_flow"].ne(0), out["amount"])
    out.loc[tax_mask, "cash_flow"] = -tax_values.loc[tax_mask].abs()
    margin_mask = out["type"].eq("margin_interest_expense")
    margin_values = out["cash_flow"].where(out["cash_flow"].ne(0), out["amount"])
    out.loc[margin_mask, "cash_flow"] = -margin_values.loc[margin_mask].abs()
    credit_mask = out["type"].eq("non_investment_credit")
    credit_values = out["cash_flow"].where(out["cash_flow"].ne(0), out["amount"])
    out.loc[credit_mask, "cash_flow"] = credit_values.loc[credit_mask].abs()
    out.loc[out["type"].isin(["internal_transfer", "corporate_action", "split"]), "cash_flow"] = 0.0
    out["source_row"] = raw.index
    return out.sort_values(["date", "source_row"]).reset_index(drop=True), mapping


def unresolved_cash_affecting_rows(tx: pd.DataFrame) -> pd.DataFrame:
    """Unknown rows with non-zero amount cannot be safely ignored in production."""
    amount = tx.get("amount", pd.Series(0.0, index=tx.index)).fillna(0.0)
    return tx.loc[tx["type"].eq("unknown") & amount.abs().gt(1e-9)].copy()


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
    missing_classified_cash = tx["type"].isin([
        "interest", "withholding_tax", "margin_interest_expense", "non_investment_credit",
    ]) & cash_values.isna()
    if missing_classified_cash.any():
        warnings.append(
            f"{missing_classified_cash.sum()} classified cash rows have no auditable amount"
        )
    positions: dict[str, float] = {}
    negative_records: list[dict] = []
    position_events = tx["type"].isin(["buy", "sell", "split"])
    for idx, row in tx.loc[position_events].iterrows():
        ticker = row["ticker"]
        if pd.isna(ticker):
            continue
        before = positions.get(str(ticker), 0.0)
        delta = row["quantity"] if row["type"] in {"buy", "split"} else -row["quantity"]
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
    ticker_summary["split_shares"] = tickers.apply(lambda g: g.loc[g.type.eq("split"), "quantity"].sum(), include_groups=False)
    ticker_summary["ending_shares"] = ticker_summary["buy_shares"] + ticker_summary["split_shares"] - ticker_summary["sell_shares"]
    summary = pd.DataFrame([{
        "start_date": tx["date"].min(), "end_date": tx["date"].max(), "rows": len(tx),
        "buys": int(tx.type.eq("buy").sum()), "sells": int(tx.type.eq("sell").sum()),
        "dividends": int(tx.type.eq("dividend").sum()),
        "deposits_withdrawals": int(tx.type.isin(["deposit", "withdrawal"]).sum()),
        "margin_interest_expense": int(tx.type.eq("margin_interest_expense").sum()),
        "internal_transfers": int(tx.type.eq("internal_transfer").sum()),
        "non_investment_credits": int(tx.type.eq("non_investment_credit").sum()),
        "splits": int(tx.type.eq("split").sum()),
        "unknown": int(unknown.sum()), "missing_trade_price_or_quantity": int(missing_price.sum()),
        "negative_holding_events": len(negative_holdings),
    }])

    review = tx.copy()
    diagnostic_defaults = {
        "raw_type": review.get("type", pd.Series(pd.NA, index=review.index)),
        "description": "", "amount": np.nan, "quantity": np.nan,
        "price": np.nan, "ticker": pd.NA, "source_row": review.index,
    }
    for column, default in diagnostic_defaults.items():
        if column not in review:
            review[column] = default
    diagnostic_rows: list[dict] = []
    if not review.empty:
        for (raw_type, normalized_type), group in review.groupby(["raw_type", "type"], dropna=False, sort=True):
            source_rows = group["source_row"].tolist()
            examples = group[["source_row", "date", "ticker", "description", "quantity", "amount"]].head(3).copy()
            examples["date"] = examples["date"].astype("string")
            diagnostic_rows.append({
                "raw_type": raw_type,
                "normalized_type": normalized_type,
                "count": len(group),
                "example_source_rows": ",".join(map(str, source_rows[:5])),
                "example_rows": examples.to_dict("records"),
                "review_status": "unresolved" if normalized_type == "unknown" else "classified",
            })
    transaction_type_diagnostic = pd.DataFrame(diagnostic_rows, columns=[
        "raw_type", "normalized_type", "count", "example_source_rows", "example_rows", "review_status",
    ])
    unrecognized_types = review.loc[unknown, [
        "source_row", "date", "raw_type", "description", "amount", "ticker", "quantity", "price",
    ]].reset_index(drop=True)
    return TransactionAudit(
        summary=summary,
        ticker_summary=ticker_summary.reset_index(),
        negative_holdings=negative_holdings,
        transaction_type_diagnostic=transaction_type_diagnostic,
        unrecognized_types=unrecognized_types,
        warnings=warnings,
    )
