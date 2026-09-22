import pandas as pd
import pytest

from src.price_adjustment import adjustment_factor, attach_adjusted_execution_prices
from tests.conftest import make_tx


def test_adjusted_execution_price_conversion():
    date = pd.Timestamp("2026-01-02")
    tx = make_tx([{"date": date, "type": "buy", "ticker": "ABC", "quantity": 2, "price": 100}])
    close = pd.DataFrame({"ABC": [200]}, index=[date])
    adj = pd.DataFrame({"ABC": [100]}, index=[date])
    result, warnings = attach_adjusted_execution_prices(tx, close, adj)
    assert not warnings
    assert result.loc[0, "adjustment_factor"] == pytest.approx(0.5)
    assert result.loc[0, "adjusted_execution_price"] == pytest.approx(50)


def test_invalid_adjustment_inputs_fail():
    with pytest.raises(ValueError): adjustment_factor(0, 10)

