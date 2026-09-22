import pandas as pd
import pytest

from src.portfolio_accounting import reconstruct_portfolio, time_weighted_returns
from tests.conftest import make_tx


def test_buy_increases_holding_and_execution_to_close_pnl(dates):
    tx = make_tx([{"date": dates[0], "type": "deposit", "cash_flow": 1000},
                  {"date": dates[0], "type": "buy", "ticker": "ABC", "quantity": 10, "adjusted_execution_price": 100}])
    prices = pd.DataFrame({"ABC": [103, 108, 109]}, index=dates)
    result = reconstruct_portfolio(tx, prices)
    assert result.holdings.query("date == @dates[0]").shares.iloc[0] == 10
    assert result.daily.loc[dates[0], "buy_execution_to_close_pnl"] == pytest.approx(30)
    assert result.daily.loc[dates[0], "nav"] == pytest.approx(1030)


def test_sell_decreases_holding_and_previous_close_to_execution_pnl(dates):
    tx = make_tx([{"date": dates[0], "type": "deposit", "cash_flow": 1000},
                  {"date": dates[0], "type": "buy", "ticker": "ABC", "quantity": 10, "adjusted_execution_price": 100},
                  {"date": dates[1], "type": "sell", "ticker": "ABC", "quantity": 10, "adjusted_execution_price": 110}])
    prices = pd.DataFrame({"ABC": [108, 112, 113]}, index=dates)
    result = reconstruct_portfolio(tx, prices)
    assert result.holdings.query("date == @dates[1]").empty
    assert result.daily.loc[dates[1], "sell_previous_close_to_execution_pnl"] == pytest.approx(20)
    assert result.daily.loc[dates[1], "nav"] == pytest.approx(1100)


def test_negative_holding_rejected(dates):
    tx = make_tx([{"date": dates[0], "type": "sell", "ticker": "ABC", "quantity": 1, "adjusted_execution_price": 100}])
    with pytest.raises(ValueError, match="Negative holding"):
        reconstruct_portfolio(tx, pd.DataFrame({"ABC": [100, 100, 100]}, index=dates))


def test_dividend_enters_nav(dates):
    tx = make_tx([{"date": dates[0], "type": "deposit", "cash_flow": 100},
                  {"date": dates[1], "type": "dividend", "ticker": "ABC", "dividend": 5}])
    result = reconstruct_portfolio(tx, pd.DataFrame({"ABC": [10, 10, 10]}, index=dates))
    assert result.daily.loc[dates[1], "dividend_income"] == 5
    assert result.daily.loc[dates[1], "nav"] == 105


def test_external_cash_flow_does_not_change_twr(dates):
    nav = pd.Series([100, 150, 150], index=dates)
    flows = pd.Series([100, 50, 0], index=dates)
    result = time_weighted_returns(nav, flows)
    assert result.iloc[1] == pytest.approx(0)
    assert result.iloc[2] == pytest.approx(0)

