from __future__ import annotations

import numpy as np
import pandas as pd

from .drawdown import drawdown_series


def capture_ratios(active: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    frame = pd.concat([active.rename("active"), benchmark.rename("benchmark")], axis=1).dropna()
    rows = []
    for label, mask in {"up": frame.benchmark > 0, "down": frame.benchmark < 0}.items():
        subset = frame.loc[mask]
        arithmetic = subset.active.mean() / subset.benchmark.mean() if len(subset) and subset.benchmark.mean() else np.nan
        a_geo = (1 + subset.active).prod() ** (252 / len(subset)) - 1 if len(subset) else np.nan
        b_geo = (1 + subset.benchmark).prod() ** (252 / len(subset)) - 1 if len(subset) else np.nan
        geometric = a_geo / b_geo if b_geo else np.nan
        rows.append({"regime": label, "n": len(subset), "arithmetic_daily_capture": arithmetic, "geometric_period_capture": geometric})
    return pd.DataFrame(rows)


def regime_analysis(active: pd.Series, voo: pd.Series) -> pd.DataFrame:
    frame = pd.concat([active.rename("active"), voo.rename("voo")], axis=1).dropna()
    voo_dd = drawdown_series(frame.voo)
    regimes = {
        "VOO > 0": frame.voo > 0,
        "VOO < 0": frame.voo < 0,
        "VOO <= -1%": frame.voo <= -0.01,
        "VOO <= -2%": frame.voo <= -0.02,
        "VOO drawdown > 5%": voo_dd <= -0.05,
        "VOO drawdown > 10%": voo_dd <= -0.10,
    }
    rows = []
    for name, mask in regimes.items():
        sub = frame.loc[mask]
        beta = sub.active.cov(sub.voo) / sub.voo.var() if len(sub) > 1 and sub.voo.var() else np.nan
        rows.append({"regime": name, "n": len(sub), "mean_return": sub.active.mean(), "median_return": sub.active.median(),
                     "win_rate": sub.active.gt(0).mean() if len(sub) else np.nan, "volatility": sub.active.std(ddof=1),
                     "beta": beta, "relative_return": (sub.active - sub.voo).mean() if len(sub) else np.nan,
                     "arithmetic_capture": sub.active.mean() / sub.voo.mean() if len(sub) and sub.voo.mean() else np.nan})
    return pd.DataFrame(rows)

