import pandas as pd

from src.plotting import create_account_figures


def test_account_allocation_chart_allows_cash_to_cross_zero(tmp_path):
    dates = pd.to_datetime(["2026-04-01", "2026-04-02", "2026-04-03"])
    daily = pd.DataFrame({
        "cash": [100.0, -25.0, 10.0],
        "market_value": [0.0, 125.0, 115.0],
    }, index=dates)
    create_account_figures(daily, pd.DataFrame(), pd.DataFrame(), tmp_path)
    assert (tmp_path / "allocation_over_time.png").is_file()
