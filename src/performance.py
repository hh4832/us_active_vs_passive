from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1 + returns.dropna()).cumprod()
    if wealth.empty:
        return np.nan
    return float((wealth / wealth.cummax() - 1).min())


def performance_metrics(returns: pd.Series, *, risk_free_rate: float = 0.04, annualization: int = 252) -> dict[str, float]:
    r = returns.dropna().astype(float)
    if r.empty:
        return {k: np.nan for k in ["cumulative_return", "annualized_return", "annualized_volatility", "sharpe", "sortino", "max_drawdown", "calmar"]}
    cumulative = float((1 + r).prod() - 1)
    annual_return = float((1 + cumulative) ** (annualization / len(r)) - 1)
    volatility = float(r.std(ddof=1) * np.sqrt(annualization))
    daily_rf = (1 + risk_free_rate) ** (1 / annualization) - 1
    excess = r - daily_rf
    sharpe = float(excess.mean() / r.std(ddof=1) * np.sqrt(annualization)) if r.std(ddof=1) else np.nan
    downside = np.sqrt(np.mean(np.minimum(excess, 0) ** 2))
    sortino = float(excess.mean() / downside * np.sqrt(annualization)) if downside else np.nan
    mdd = max_drawdown(r)
    calmar = annual_return / abs(mdd) if mdd else np.nan
    return {"cumulative_return": cumulative, "annualized_return": annual_return, "annualized_volatility": volatility,
            "sharpe": sharpe, "sortino": sortino, "max_drawdown": mdd, "calmar": calmar}


def relative_metrics(portfolio: pd.Series, benchmark: pd.Series, *, annualization: int = 252) -> dict[str, float]:
    frame = pd.concat([portfolio.rename("p"), benchmark.rename("b")], axis=1).dropna()
    if len(frame) < 2 or frame.b.var() == 0:
        return {k: np.nan for k in ["beta", "alpha", "correlation", "tracking_error", "information_ratio"]}
    beta = frame.p.cov(frame.b) / frame.b.var()
    alpha = (frame.p.mean() - beta * frame.b.mean()) * annualization
    active = frame.p - frame.b
    tracking_error = active.std(ddof=1) * np.sqrt(annualization)
    information_ratio = active.mean() * annualization / tracking_error if tracking_error else np.nan
    return {"beta": float(beta), "alpha": float(alpha), "correlation": float(frame.p.corr(frame.b)),
            "tracking_error": float(tracking_error), "information_ratio": float(information_ratio)}


def xirr(cash_flows: pd.Series, ending_value: float, ending_date: pd.Timestamp) -> float:
    dated = [(pd.Timestamp(d), -float(v)) for d, v in cash_flows.items() if v != 0]
    dated.append((pd.Timestamp(ending_date), float(ending_value)))
    if not any(v < 0 for _, v in dated) or not any(v > 0 for _, v in dated):
        return np.nan
    origin = min(d for d, _ in dated)
    def npv(rate: float) -> float:
        return sum(v / (1 + rate) ** ((d - origin).days / 365.25) for d, v in dated)
    low, high = -0.9999, 10.0
    if npv(low) * npv(high) > 0:
        return np.nan
    for _ in range(200):
        mid = (low + high) / 2
        if npv(low) * npv(mid) <= 0: high = mid
        else: low = mid
    return float((low + high) / 2)

