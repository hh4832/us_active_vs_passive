from __future__ import annotations

import numpy as np
import pandas as pd


def cashflow_matched_benchmark(external_flows: pd.Series, prices: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Invest every contribution on the same date, allowing fractional shares."""
    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError("Benchmark weights must sum to 1")
    missing = set(weights) - set(prices.columns)
    if missing:
        raise ValueError(f"Missing benchmark prices: {sorted(missing)}")
    invalid = prices[list(weights)].isna() | prices[list(weights)].le(0)
    if invalid.any().any():
        raise ValueError("Missing/invalid benchmark market prices; silent ffill is prohibited")
    idx = prices.index
    flows = external_flows.reindex(idx, fill_value=0.0)
    shares = {t: 0.0 for t in weights}
    rows = []
    for date in idx:
        flow = float(flows.loc[date])
        if flow >= 0:
            for ticker, weight in weights.items():
                shares[ticker] += flow * weight / float(prices.loc[date, ticker])
        else:
            nav_before = sum(shares[t] * float(prices.loc[date, t]) for t in weights)
            if -flow > nav_before + 1e-9:
                raise ValueError(f"Withdrawal exceeds benchmark NAV on {date.date()}")
            fraction = -flow / nav_before if nav_before else 0
            for ticker in weights:
                shares[ticker] *= 1 - fraction
        nav = sum(shares[t] * float(prices.loc[date, t]) for t in weights)
        rows.append({"date": date, "nav": nav, "external_cash_flow": flow, **{f"shares_{t}": q for t, q in shares.items()}})
    result = pd.DataFrame(rows).set_index("date")
    previous = result["nav"].shift(1)
    result["return"] = (result["nav"] - result["external_cash_flow"]) / previous - 1
    return result


def beta_matched_weights(beta: float) -> dict[str, float]:
    equity = float(np.clip(beta, 0, 1))
    return {"VOO": equity, "SGOV": 1 - equity}


def volatility_matched_weights(active_returns: pd.Series, voo_returns: pd.Series, sgov_returns: pd.Series) -> dict[str, float]:
    aligned = pd.concat([active_returns, voo_returns, sgov_returns], axis=1).dropna()
    if aligned.empty:
        raise ValueError("No overlapping observations for volatility matching")
    target = aligned.iloc[:, 0].std()
    grid = np.linspace(0, 1, 1001)
    vols = [(w * aligned.iloc[:, 1] + (1 - w) * aligned.iloc[:, 2]).std() for w in grid]
    weight = float(grid[int(np.argmin(np.abs(np.asarray(vols) - target)))])
    return {"VOO": weight, "SGOV": 1 - weight}
