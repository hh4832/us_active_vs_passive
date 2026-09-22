import pandas as pd
import pytest

from src.benchmarks import cashflow_matched_benchmark


def test_benchmark_cash_flow_dates_match(dates):
    flows = pd.Series([100, 0, 50], index=dates)
    prices = pd.DataFrame({"VOO": [10, 11, 10]}, index=dates)
    result = cashflow_matched_benchmark(flows, prices, {"VOO": 1.0})
    pd.testing.assert_series_equal(result.external_cash_flow, flows.astype(float), check_names=False)
    assert result.loc[dates[2], "shares_VOO"] == pytest.approx(15)

