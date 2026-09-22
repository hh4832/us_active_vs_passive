from __future__ import annotations

import pandas as pd


def drawdown_series(returns: pd.Series) -> pd.Series:
    wealth = (1 + returns.fillna(0)).cumprod()
    return wealth / wealth.cummax() - 1


def drawdown_episodes(returns: pd.Series, benchmark_returns: pd.Series | None = None, top_n: int = 10) -> pd.DataFrame:
    dd = drawdown_series(returns)
    episodes: list[dict] = []
    in_episode = False
    peak = None
    for date, value in dd.items():
        if value < 0 and not in_episode:
            in_episode = True
            loc = dd.index.get_loc(date)
            peak = dd.index[max(0, loc - 1)]
        if in_episode and value == 0:
            segment = dd.loc[peak:date]
            trough = segment.idxmin()
            row = {"peak_date": peak, "trough_date": trough, "recovery_date": date,
                   "maximum_drawdown": float(segment.min()), "days_peak_to_trough": (trough - peak).days,
                   "days_underwater": (date - peak).days}
            if benchmark_returns is not None:
                b = drawdown_series(benchmark_returns).reindex(segment.index)
                row["benchmark_drawdown_same_period"] = float(b.min())
                row["active_excess_drawdown"] = row["maximum_drawdown"] - row["benchmark_drawdown_same_period"]
            episodes.append(row); in_episode = False
    if in_episode and peak is not None:
        segment = dd.loc[peak:]
        trough = segment.idxmin()
        episodes.append({"peak_date": peak, "trough_date": trough, "recovery_date": pd.NaT,
                         "maximum_drawdown": float(segment.min()), "days_peak_to_trough": (trough - peak).days,
                         "days_underwater": (segment.index[-1] - peak).days})
    return pd.DataFrame(episodes).sort_values("maximum_drawdown").head(top_n) if episodes else pd.DataFrame()

