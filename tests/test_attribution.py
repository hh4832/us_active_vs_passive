import pandas as pd
import pytest

from src.attribution import fifo_realized_pnl
from tests.conftest import make_tx


def test_fifo_realized_pnl():
    tx = make_tx([
        {"date": pd.Timestamp("2026-01-01"), "type": "buy", "ticker": "ABC", "quantity": 2, "adjusted_execution_price": 10},
        {"date": pd.Timestamp("2026-01-02"), "type": "buy", "ticker": "ABC", "quantity": 2, "adjusted_execution_price": 20},
        {"date": pd.Timestamp("2026-01-03"), "type": "sell", "ticker": "ABC", "quantity": 3, "adjusted_execution_price": 30},
    ])
    result = fifo_realized_pnl(tx)
    assert result.quantity.sum() == 3
    assert result.realized_pnl.sum() == pytest.approx(50)

