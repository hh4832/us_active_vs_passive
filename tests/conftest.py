import pandas as pd
import pytest


@pytest.fixture
def dates():
    return pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])


def make_tx(rows):
    defaults = {"ticker": pd.NA, "quantity": 0.0, "price": float("nan"), "amount": float("nan"),
                "dividend": 0.0, "cash_flow": 0.0, "fee": 0.0, "adjustment_factor": 1.0,
                "adjusted_execution_price": float("nan"), "raw_execution_price": float("nan"),
                "description": "", "raw_type": ""}
    return pd.DataFrame([{**defaults, **row, "source_row": i} for i, row in enumerate(rows)])
