from __future__ import annotations

import numpy as np
import pandas as pd


FACTOR_TOLERANCE = 1e-6
MAX_NEIGHBOR_CALENDAR_DAYS = 3
MAX_SINGLE_SIDE_SESSIONS = 1


def adjustment_factor(close: float, adj_close: float) -> float:
    if not np.isfinite(close) or close <= 0 or not np.isfinite(adj_close) or adj_close <= 0:
        raise ValueError(f"Invalid close/adj_close: {close}/{adj_close}")
    return float(adj_close / close)


def _value(frame: pd.DataFrame, date: pd.Timestamp, ticker: str) -> float:
    if ticker not in frame.columns or date not in frame.index:
        return np.nan
    value = frame.at[date, ticker]
    return float(value) if pd.notna(value) else np.nan


def _factor_series(close: pd.DataFrame, adj_close: pd.DataFrame, ticker: str) -> pd.Series:
    if ticker not in close.columns or ticker not in adj_close.columns:
        return pd.Series(index=pd.DatetimeIndex([]), dtype=float)
    pair = pd.concat([close[ticker].rename("close"), adj_close[ticker].rename("adj_close")], axis=1).sort_index()
    valid = (
        pair["close"].notna()
        & pair["adj_close"].notna()
        & np.isfinite(pair["close"])
        & np.isfinite(pair["adj_close"])
        & pair["close"].gt(0)
        & pair["adj_close"].gt(0)
    )
    result = (pair.loc[valid, "adj_close"] / pair.loc[valid, "close"]).astype(float)
    result.index = pd.DatetimeIndex(result.index).normalize()
    return result


def _session_distance(index: pd.Index, date: pd.Timestamp, reference: pd.Timestamp) -> int:
    sessions = pd.DatetimeIndex(index).drop_duplicates().sort_values()
    if date in sessions and reference in sessions:
        return abs(int(sessions.get_loc(reference)) - int(sessions.get_loc(date)))
    low, high = sorted([date, reference])
    return int(((sessions > low) & (sessions <= high)).sum())


def _source_name(provider: str, resolution: str) -> str:
    if provider in {"finlab", "stock", "fund"}:
        return resolution
    if resolution == "exact_finlab":
        return f"exact_{provider}"
    return f"{provider}_{resolution}"


def attach_adjusted_execution_prices(
    transactions: pd.DataFrame,
    close: pd.DataFrame,
    adj_close: pd.DataFrame,
    *,
    price_sources: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    out = transactions.copy()
    warnings: list[str] = []
    factors, adjusted, raw_close, raw_adj = [], [], [], []
    resolution_sources, reference_dates, shift_days, row_warnings = [], [], [], []
    market_sessions = close.index.union(adj_close.index).sort_values()
    price_sources = price_sources or {}
    for idx, row in out.iterrows():
        if row["type"] not in {"buy", "sell"} or pd.isna(row["ticker"]) or pd.isna(row["date"]):
            factors.append(np.nan); adjusted.append(np.nan); raw_close.append(np.nan); raw_adj.append(np.nan)
            resolution_sources.append(pd.NA); reference_dates.append(pd.NaT); shift_days.append(pd.NA); row_warnings.append(pd.NA)
            continue
        ticker, date = str(row["ticker"]), pd.Timestamp(row["date"]).normalize()
        provider = price_sources.get(ticker, "finlab")
        c, ac = _value(close, date, ticker), _value(adj_close, date, ticker)
        raw_close.append(c); raw_adj.append(ac)
        factor = np.nan
        source = "unresolved"
        reference_date = pd.NaT
        date_shift: object = pd.NA
        warning: str | None = None
        try:
            factor = adjustment_factor(c, ac)
            source = _source_name(provider, "exact_finlab")
            reference_date = date
            date_shift = 0
            if provider not in {"finlab", "stock", "fund"}:
                warning = f"Row {idx} {ticker} {date.date()}: used audited external price source {provider}"
                if "no_corporate_action_factor_1" in provider:
                    warning += "; adjustment factor 1.0 depends on explicit no-corporate-action confirmation"
        except (TypeError, ValueError):
            series = _factor_series(close, adj_close, ticker)
            index_values = series.index.to_numpy(dtype="datetime64[ns]")
            exact_value = np.datetime64(date.to_datetime64(), "ns")
            lower = exact_value - np.timedelta64(MAX_NEIGHBOR_CALENDAR_DAYS, "D")
            upper = exact_value + np.timedelta64(MAX_NEIGHBOR_CALENDAR_DAYS, "D")
            nearby = series[(index_values >= lower) & (index_values <= upper) & (index_values != exact_value)]
            previous = nearby[nearby.index < date]
            following = nearby[nearby.index > date]
            prev_item = (previous.index[-1], float(previous.iloc[-1])) if not previous.empty else None
            next_item = (following.index[0], float(following.iloc[0])) if not following.empty else None
            if prev_item and next_item:
                relative_difference = abs(prev_item[1] / next_item[1] - 1)
                if relative_difference <= FACTOR_TOLERANCE:
                    factor = prev_item[1]
                    reference_date = min(
                        [prev_item[0], next_item[0]],
                        key=lambda candidate: (abs((candidate - date).days), candidate > date),
                    )
                    date_shift = int((reference_date - date).days)
                    source = _source_name(provider, "neighbor_factor_verified")
                    warning = (
                        f"Row {idx} {ticker} {date.date()}: exact factor missing; verified identical neighboring factors "
                        f"at {prev_item[0].date()} and {next_item[0].date()}"
                    )
                else:
                    warning = (
                        f"FATAL row {idx} {ticker} {date.date()}: neighboring adjustment factors disagree "
                        f"({prev_item[0].date()}={prev_item[1]:.12g}, {next_item[0].date()}={next_item[1]:.12g})"
                    )
            elif prev_item or next_item:
                reference_date, factor_candidate = prev_item or next_item  # type: ignore[misc]
                session_distance = _session_distance(market_sessions, date, reference_date)
                if session_distance <= MAX_SINGLE_SIDE_SESSIONS:
                    factor = factor_candidate
                    date_shift = int((reference_date - date).days)
                    source = _source_name(provider, "neighbor_factor_single_side")
                    warning = (
                        f"Row {idx} {ticker} {date.date()}: exact factor missing; used one-sided factor from "
                        f"{reference_date.date()} ({session_distance} trading session away)"
                    )
                else:
                    factor = np.nan
                    reference_date = pd.NaT
                    warning = (
                        f"FATAL row {idx} {ticker} {date.date()}: only one neighboring factor exists and is "
                        f"{session_distance} trading sessions away"
                    )
            else:
                warning = f"FATAL row {idx} {ticker} {date.date()}: no valid exact or ±3-day adjustment factor"

        if np.isfinite(factor) and (factor < 0.01 or factor > 100):
            extreme = f"Extreme adjustment factor {factor:.6g} for {ticker} on {date.date()}"
            warning = f"{warning}; {extreme}" if warning else extreme
        if warning:
            warnings.append(warning)
        factors.append(factor)
        adjusted.append(float(row["price"]) * factor if np.isfinite(factor) and pd.notna(row["price"]) else np.nan)
        resolution_sources.append(source)
        reference_dates.append(reference_date)
        shift_days.append(date_shift)
        row_warnings.append(warning if warning else pd.NA)
    out["raw_execution_price"] = out["price"]
    out["close"] = raw_close
    out["adj_close"] = raw_adj
    out["adjustment_factor"] = factors
    out["adjusted_execution_price"] = adjusted
    out["adjustment_source"] = resolution_sources
    out["adjustment_reference_date"] = pd.to_datetime(reference_dates)
    out["adjustment_date_shift_days"] = pd.array(shift_days, dtype="Int64")
    out["warning"] = row_warnings
    return out, warnings
