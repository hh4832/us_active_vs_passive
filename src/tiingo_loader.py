from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import requests


TIINGO_EOD_URL = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"
REQUIRED_FIELDS = {"date", "close", "adjClose", "divCash", "splitFactor"}


@dataclass
class TiingoPriceBundle:
    close: pd.DataFrame
    adj_close: pd.DataFrame
    div_cash: pd.DataFrame
    split_factor: pd.DataFrame
    ticker_coverage: pd.DataFrame
    corporate_actions: pd.DataFrame
    latest_date: pd.Timestamp
    source: str = "Tiingo"


@dataclass
class TiingoCoverageAudit:
    ticker_coverage: pd.DataFrame
    trade_date_coverage: pd.DataFrame
    holding_period_coverage: pd.DataFrame
    corporate_actions: pd.DataFrame

    @property
    def passed(self) -> bool:
        frames = [self.ticker_coverage, self.trade_date_coverage, self.holding_period_coverage]
        return all(frame.empty or frame["status"].eq("PASS").all() for frame in frames)


def _response_json(response: Any, ticker: str) -> list[dict]:
    try:
        response.raise_for_status()
    except Exception as exc:
        status = getattr(response, "status_code", "unknown")
        raise RuntimeError(f"Tiingo request failed for {ticker}: HTTP {status}") from exc
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError(f"Tiingo returned an unexpected response for {ticker}")
    return payload


def load_tiingo_prices(
    tickers: list[str],
    *,
    start_date: str | date | pd.Timestamp,
    end_date: str | date | pd.Timestamp | None = None,
    token: str | None = None,
    session: Any | None = None,
) -> TiingoPriceBundle:
    """Load raw and total-return EOD prices from Tiingo only.

    The token is read from ``TIINGO_API_TOKEN`` when not passed explicitly.
    No alternate provider or local fallback is used in this production loader.
    """
    token = token or os.getenv("TIINGO_API_TOKEN")
    if not token:
        raise RuntimeError("TIINGO_API_TOKEN is not set")
    requested = sorted({str(t).strip().upper() for t in tickers if str(t).strip()})
    if not requested:
        raise ValueError("No tickers requested from Tiingo")

    client = session or requests.Session()
    headers = {"Authorization": f"Token {token}", "Content-Type": "application/json"}
    params: dict[str, str] = {"startDate": str(pd.Timestamp(start_date).date()), "resampleFreq": "daily"}
    if end_date is not None:
        params["endDate"] = str(pd.Timestamp(end_date).date())

    frames: dict[str, pd.DataFrame] = {}
    coverage_rows: list[dict] = []
    action_frames: list[pd.DataFrame] = []
    for ticker in requested:
        response = client.get(
            TIINGO_EOD_URL.format(ticker=ticker), params=params, headers=headers, timeout=60,
        )
        payload = _response_json(response, ticker)
        frame = pd.DataFrame(payload)
        missing_fields = sorted(REQUIRED_FIELDS - set(frame.columns))
        if frame.empty or missing_fields:
            coverage_rows.append({
                "ticker": ticker, "row_count": len(frame), "first_date": pd.NaT, "last_date": pd.NaT,
                "missing_fields": ",".join(missing_fields), "status": "FAIL",
            })
            continue
        frame = frame[["date", "close", "adjClose", "divCash", "splitFactor"]].copy()
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce").dt.tz_convert(None).dt.normalize()
        for column in ["close", "adjClose", "divCash", "splitFactor"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        invalid_date = frame["date"].isna()
        duplicates = frame["date"].duplicated(keep=False)
        if invalid_date.any() or duplicates.any():
            raise RuntimeError(f"Tiingo returned invalid or duplicate dates for {ticker}")
        frame = frame.set_index("date").sort_index()
        frames[ticker] = frame
        valid = frame["close"].gt(0) & frame["adjClose"].gt(0)
        coverage_rows.append({
            "ticker": ticker,
            "row_count": len(frame),
            "first_date": frame.index.min(),
            "last_date": frame.index.max(),
            "missing_fields": "",
            "invalid_price_rows": int((~valid).sum()),
            "status": "PASS" if valid.all() else "FAIL",
        })
        actions = frame.loc[frame["divCash"].fillna(0).ne(0) | frame["splitFactor"].fillna(1).ne(1)].reset_index()
        if not actions.empty:
            actions.insert(1, "ticker", ticker)
            action_frames.append(actions)

    ticker_coverage = pd.DataFrame(coverage_rows).sort_values("ticker").reset_index(drop=True)
    def wide(column: str) -> pd.DataFrame:
        if not frames:
            return pd.DataFrame()
        return pd.concat({ticker: frame[column] for ticker, frame in frames.items()}, axis=1).sort_index()

    corporate_actions = (
        pd.concat(action_frames, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
        if action_frames else
        pd.DataFrame(columns=["date", "ticker", "close", "adjClose", "divCash", "splitFactor"])
    )
    close = wide("close")
    return TiingoPriceBundle(
        close=close,
        adj_close=wide("adjClose"),
        div_cash=wide("divCash"),
        split_factor=wide("splitFactor"),
        ticker_coverage=ticker_coverage,
        corporate_actions=corporate_actions,
        latest_date=pd.Timestamp(close.index.max()) if not close.empty else pd.NaT,
    )


def audit_tiingo_coverage(
    bundle: TiingoPriceBundle,
    transactions: pd.DataFrame,
    *,
    actual_tickers: set[str],
) -> TiingoCoverageAudit:
    trades = transactions.loc[transactions["type"].isin(["buy", "sell"])].copy()
    trade_rows: list[dict] = []
    for _, row in trades.iterrows():
        ticker = str(row["ticker"])
        trade_date = pd.Timestamp(row["date"]).normalize()
        close = bundle.close.at[trade_date, ticker] if ticker in bundle.close and trade_date in bundle.close.index else np.nan
        adj_close = bundle.adj_close.at[trade_date, ticker] if ticker in bundle.adj_close and trade_date in bundle.adj_close.index else np.nan
        valid = pd.notna(close) and pd.notna(adj_close) and close > 0 and adj_close > 0
        trade_rows.append({
            "source_row": row.get("source_row"), "date": trade_date, "ticker": ticker,
            "type": row["type"], "close": close, "adjClose": adj_close,
            "status": "PASS" if valid else "FAIL",
        })
    trade_audit = pd.DataFrame(trade_rows, columns=[
        "source_row", "date", "ticker", "type", "close", "adjClose", "status",
    ])

    sessions = pd.DatetimeIndex(bundle.close.index).sort_values()
    holding_rows: list[dict] = []
    for ticker in sorted(actual_tickers):
        ticker_trades = trades.loc[trades["ticker"].eq(ticker)].sort_values(["date", "source_row"])
        if ticker_trades.empty:
            continue
        if sessions.empty:
            missing_dates = sorted(pd.DatetimeIndex(ticker_trades["date"].dropna().unique()))
            holding_rows.append({
                "ticker": ticker, "holding_start": pd.NaT, "holding_end": pd.NaT,
                "held_sessions": 0, "covered_sessions": 0,
                "missing_sessions": len(missing_dates),
                "missing_dates": ",".join(str(d.date()) for d in missing_dates),
                "status": "FAIL",
            })
            continue
        position = 0.0
        held_dates: list[pd.Timestamp] = []
        for session_date in sessions[sessions >= ticker_trades["date"].min()]:
            day = ticker_trades.loc[ticker_trades["date"].eq(session_date)]
            for _, row in day.iterrows():
                position += float(row["quantity"]) if row["type"] == "buy" else -float(row["quantity"])
            if abs(position) > 1e-9:
                held_dates.append(pd.Timestamp(session_date))
        missing_dates = [
            d for d in held_dates
            if ticker not in bundle.close or ticker not in bundle.adj_close
            or pd.isna(bundle.close.at[d, ticker]) or pd.isna(bundle.adj_close.at[d, ticker])
            or bundle.close.at[d, ticker] <= 0 or bundle.adj_close.at[d, ticker] <= 0
        ]
        holding_rows.append({
            "ticker": ticker,
            "holding_start": min(held_dates) if held_dates else pd.NaT,
            "holding_end": max(held_dates) if held_dates else pd.NaT,
            "held_sessions": len(held_dates),
            "covered_sessions": len(held_dates) - len(missing_dates),
            "missing_sessions": len(missing_dates),
            "missing_dates": ",".join(str(d.date()) for d in missing_dates),
            "status": "PASS" if not missing_dates else "FAIL",
        })
    holding_audit = pd.DataFrame(holding_rows, columns=[
        "ticker", "holding_start", "holding_end", "held_sessions", "covered_sessions",
        "missing_sessions", "missing_dates", "status",
    ])
    return TiingoCoverageAudit(
        ticker_coverage=bundle.ticker_coverage.copy(),
        trade_date_coverage=trade_audit,
        holding_period_coverage=holding_audit,
        corporate_actions=bundle.corporate_actions.copy(),
    )
