"""Legacy manual fallback loader; not used by the Tiingo-only production path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"date", "ticker", "close", "adj_close", "source"}
CONFIRMATION_COLUMN = "no_corporate_action_confirmed"


@dataclass
class ExternalPriceBundle:
    close: pd.DataFrame
    adj_close: pd.DataFrame
    sources: dict[str, str]
    warnings: list[str]


def _confirmed(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "confirmed"}


def empty_external_price_bundle(warning: str | None = None) -> ExternalPriceBundle:
    return ExternalPriceBundle(
        close=pd.DataFrame(),
        adj_close=pd.DataFrame(),
        sources={},
        warnings=[warning] if warning else [],
    )


def load_external_prices(
    path: str | Path,
    *,
    required_tickers: set[str] | None = None,
) -> ExternalPriceBundle:
    """Load an explicitly supplied, auditable external price series.

    `adj_close` may be blank only when the same row explicitly confirms that no
    split or corporate action occurred. In that case the factor is set to 1.0
    and a warning is emitted. Nothing is downloaded or inferred here.
    """
    path = Path(path)
    if not path.is_file():
        return empty_external_price_bundle(f"External fallback price file not found: {path}")

    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Fallback price file missing required columns: {sorted(missing)}")
    if frame.empty:
        return empty_external_price_bundle(f"External fallback price file is empty: {path}")

    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["ticker"] = out["ticker"].astype("string").str.strip().str.upper()
    out["source"] = out["source"].astype("string").str.strip()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["adj_close"] = pd.to_numeric(out["adj_close"], errors="coerce")
    if out["date"].isna().any():
        raise ValueError("Fallback price file contains invalid dates")
    if out["ticker"].isna().any() or out["ticker"].eq("").any():
        raise ValueError("Fallback price file contains blank tickers")
    if out["source"].isna().any() or out["source"].eq("").any():
        raise ValueError("Fallback price file contains blank sources")
    if out.duplicated(["date", "ticker"]).any():
        duplicates = out.loc[out.duplicated(["date", "ticker"], keep=False), ["date", "ticker"]]
        raise ValueError(f"Fallback price file has duplicate date/ticker rows: {duplicates.to_dict('records')}")
    invalid_close = ~np.isfinite(out["close"]) | out["close"].le(0)
    if invalid_close.any():
        rows = out.index[invalid_close].tolist()
        raise ValueError(f"Fallback price file has invalid close values at rows: {rows}")
    invalid_adj = out["adj_close"].notna() & (~np.isfinite(out["adj_close"]) | out["adj_close"].le(0))
    if invalid_adj.any():
        rows = out.index[invalid_adj].tolist()
        raise ValueError(f"Fallback price file has invalid adj_close values at rows: {rows}")

    if required_tickers is not None:
        out = out[out["ticker"].isin({str(t).upper() for t in required_tickers})].copy()
    if out.empty:
        return empty_external_price_bundle("External fallback file has no rows for the required missing tickers")

    confirmations = (
        out[CONFIRMATION_COLUMN].map(_confirmed)
        if CONFIRMATION_COLUMN in out
        else pd.Series(False, index=out.index)
    )
    warnings: list[str] = []
    assumed = out["adj_close"].isna() & confirmations
    for idx, row in out.loc[assumed].iterrows():
        out.loc[idx, "adj_close"] = row["close"]
        warnings.append(
            f"External factor 1.0 used for {row['ticker']} on {row['date'].date()} from {row['source']} "
            f"because no split/corporate action was explicitly confirmed"
        )
    unresolved = out["adj_close"].isna() & ~confirmations
    for _, row in out.loc[unresolved].iterrows():
        warnings.append(
            f"External adj_close missing for {row['ticker']} on {row['date'].date()} from {row['source']}; "
            f"set {CONFIRMATION_COLUMN}=true only after corporate-action review"
        )

    source_counts = out.groupby("ticker")["source"].nunique()
    ambiguous = source_counts[source_counts > 1]
    if not ambiguous.empty:
        raise ValueError(f"Each fallback ticker must use one source; multiple sources found for {ambiguous.index.tolist()}")
    sources = out.groupby("ticker")["source"].first().astype(str).to_dict()
    for ticker in out.loc[assumed, "ticker"].unique():
        sources[str(ticker)] = f"{sources[str(ticker)]}_no_corporate_action_factor_1"

    close = out.pivot(index="date", columns="ticker", values="close").sort_index()
    adj_close = out.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    close.columns.name = None
    adj_close.columns.name = None
    return ExternalPriceBundle(close=close, adj_close=adj_close, sources=sources, warnings=warnings)
