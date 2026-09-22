import numpy as np
import pandas as pd
import pytest

from src.performance import max_drawdown, performance_metrics


def test_max_drawdown():
    returns = pd.Series([0.10, -0.20, 0.05])
    assert max_drawdown(returns) == pytest.approx(-0.20)


def test_sharpe():
    returns = pd.Series([0.01, 0.02, -0.01, 0.00])
    result = performance_metrics(returns, risk_free_rate=0, annualization=252)
    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    assert result["sharpe"] == pytest.approx(expected)

