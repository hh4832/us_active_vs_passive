"""Legacy FinLab price loader.

Production analysis uses :mod:`src.tiingo_loader`. This module is retained only
for reproducibility of historical experiments and is not imported by the
production runner.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd


@dataclass
class PriceBundle:
    close: pd.DataFrame
    adj_close: pd.DataFrame
    classification: dict[str, str]
    warnings: list[str]


def load_finlab_prices(tickers: list[str], known_funds: list[str] | None = None) -> PriceBundle:
    """Load requested tickers, checking the alternate FinLab source before failing."""
    token = os.getenv("FINLAB_API_TOKEN")
    if not token:
        raise RuntimeError("FINLAB_API_TOKEN is not set")
    import finlab
    from finlab import data

    finlab.login(token)
    datasets = {
        "stock": (data.get("us_price:close"), data.get("us_price:adj_close")),
        "fund": (data.get("us_fund_price:close"), data.get("us_fund_price:adj_close")),
    }
    known_funds = set(known_funds or [])
    classification: dict[str, str] = {}
    warnings: list[str] = []
    close_parts, adj_parts = [], []
    for ticker in sorted(set(tickers)):
        preferred = "fund" if ticker in known_funds else "stock"
        alternate = "stock" if preferred == "fund" else "fund"
        source = preferred if ticker in datasets[preferred][1].columns else alternate if ticker in datasets[alternate][1].columns else None
        if source is None:
            warnings.append(f"{ticker} absent from both FinLab stock and fund price sources")
            continue
        if source != preferred:
            warnings.append(f"{ticker} found in alternate {source} source, expected {preferred}")
        classification[ticker] = source
        close_parts.append(datasets[source][0][[ticker]])
        adj_parts.append(datasets[source][1][[ticker]])
    if not adj_parts:
        raise ValueError("No requested tickers were available from FinLab")
    return PriceBundle(pd.concat(close_parts, axis=1).sort_index(), pd.concat(adj_parts, axis=1).sort_index(), classification, warnings)
