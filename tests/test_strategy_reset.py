import pandas as pd
import pytest

from src.benchmarks import beta_matched_weights, cashflow_matched_benchmark
from src.portfolio_accounting import reconstruct_portfolio
from src.strategy_reset import (
    active_sleeve_attribution,
    build_active_sleeve_since_start,
    build_reset_portfolio,
    decision_matched_cashflows,
    opening_state_audits,
)
from tests.conftest import make_tx


@pytest.fixture
def reset_case():
    dates = pd.to_datetime([
        "2026-03-31", "2026-04-01", "2026-04-02", "2026-04-03", "2026-04-06",
    ])
    tx = make_tx([
        {"date": dates[0], "type": "deposit", "cash_flow": 1000},
        {"date": dates[0], "type": "buy", "ticker": "ABC", "quantity": 5,
         "raw_execution_price": 90, "adjusted_execution_price": 90},
        {"date": dates[2], "type": "buy", "ticker": "ABC", "quantity": 1,
         "raw_execution_price": 105, "adjusted_execution_price": 105, "fee": 1},
        {"date": dates[3], "type": "sell", "ticker": "ABC", "quantity": 1,
         "raw_execution_price": 106, "adjusted_execution_price": 106, "fee": 1},
        {"date": dates[4], "type": "dividend", "ticker": "ABC", "dividend": 2},
    ])
    prices = pd.DataFrame({
        "ABC": [90, 100, 110, 105, 120],
        "VOO": [100, 101, 102, 103, 104],
        "VGT": [200, 202, 204, 206, 208],
        "SGOV": [100, 100.01, 100.02, 100.03, 100.04],
    }, index=dates)
    full = reconstruct_portfolio(
        tx, prices, included_tickers={"ABC"}, execution_price_column="raw_execution_price",
    )
    reset = build_reset_portfolio(full, "2026-04-01")
    active = build_active_sleeve_since_start(
        tx, prices, full, "2026-04-01", {"ABC"},
        execution_price_column="raw_execution_price",
    )
    return dates, tx, prices, full, reset, active


def test_opening_holdings_correctly_reconstructed(reset_case):
    dates, _, _, _, reset, _ = reset_case
    opening = reset.opening_holdings.set_index("ticker")
    assert opening.loc["ABC", "shares"] == pytest.approx(5)
    assert opening.loc["ABC", "date"] == dates[1]


def test_reset_date_nav_equals_opening_market_value_plus_cash(reset_case):
    dates, _, _, _, reset, _ = reset_case
    row = reset.daily.loc[dates[1]]
    assert row.nav == pytest.approx(row.market_value + row.cash)
    assert row.nav == pytest.approx(1050)


def test_pre_start_pnl_is_excluded(reset_case):
    dates, _, _, full, reset, _ = reset_case
    assert full.daily.loc[dates[1], "investment_pnl"] == pytest.approx(50)
    assert pd.isna(reset.daily.loc[dates[1], "twr"])
    assert reset.daily.loc[dates[1], "portfolio_index"] == pytest.approx(1.0)


def test_post_start_pnl_is_included(reset_case):
    dates, _, _, _, reset, _ = reset_case
    assert reset.daily.loc[dates[2], "twr"] != 0
    assert reset.daily.loc[dates[2], "portfolio_index"] != pytest.approx(1.0)


def test_active_contribution_does_not_inflate_twr(reset_case):
    dates, _, _, _, _, active = reset_case
    assert active.daily.loc[dates[2], "external_cash_flow"] == pytest.approx(106)
    assert active.daily.loc[dates[2], "twr"] == pytest.approx(54 / 500)


def test_active_withdrawal_does_not_distort_twr(reset_case):
    dates, _, _, _, _, active = reset_case
    assert active.daily.loc[dates[3], "external_cash_flow"] == pytest.approx(-105)
    assert active.daily.loc[dates[3], "twr"] == pytest.approx(-30 / 660)


@pytest.mark.parametrize("ticker", ["VOO", "VGT"])
def test_cashflow_matched_benchmark_receives_same_flows(reset_case, ticker):
    _, _, prices, _, _, active = reset_case
    flows = active.daily["external_cash_flow"]
    result = cashflow_matched_benchmark(flows, prices.loc[flows.index], {ticker: 1.0})
    pd.testing.assert_series_equal(
        result["external_cash_flow"], flows.astype(float), check_names=False,
    )


def test_opening_active_inventory_is_included(reset_case):
    dates, _, _, _, _, active = reset_case
    assert active.daily.loc[dates[1], "nav"] == pytest.approx(500)
    assert active.cashflows.iloc[0]["flow_type"] == "opening_capital"
    assert active.cashflows.iloc[0]["amount"] == pytest.approx(500)


def test_decision_matched_buy_uses_same_dollar_amount(reset_case):
    dates, tx, _, _, _, active = reset_case
    flows, audit = decision_matched_cashflows(
        tx, active.daily.index, "2026-04-01", {"ABC"}, 500,
        execution_price_column="raw_execution_price",
    )
    buy = audit.loc[audit.active_ticker.eq("ABC")].iloc[0]
    assert buy.active_amount == pytest.approx(106)
    assert flows.loc[dates[2]] == pytest.approx(106)


def test_beta_matched_weights_sum_to_one():
    assert sum(beta_matched_weights(0.72).values()) == pytest.approx(1.0)


def test_active_dividend_is_not_double_counted(reset_case):
    dates, _, _, _, _, active = reset_case
    # Five shares rise from 105 to 120 ($75) and the broker dividend is $2.
    assert active.daily.loc[dates[4], "investment_pnl"] == pytest.approx(77)
    assert active.daily.loc[dates[4], "external_cash_flow"] == pytest.approx(-2)
    assert active.daily.loc[dates[4], "twr"] == pytest.approx(77 / 525)


def test_reset_output_starts_exactly_on_configured_date(reset_case):
    _, _, _, _, reset, active = reset_case
    expected = pd.Timestamp("2026-04-01")
    assert reset.daily.index.min() == expected
    assert active.daily.index.min() == expected


def test_pre_start_transactions_still_affect_opening_holdings(reset_case):
    _, tx, _, _, _, active = reset_case
    assert tx.loc[tx.date.lt("2026-04-01"), "type"].eq("buy").any()
    assert active.opening_holdings.set_index("ticker").loc["ABC", "shares"] == pytest.approx(5)


def test_opening_state_audit_reconciles_to_full_account_nav(reset_case):
    dates, _, prices, full, _, _ = reset_case
    state, cash = opening_state_audits(
        full, prices, "2026-04-01", {"ABC"}, set(),
    )
    assert bool(state.loc[0, "included_in_active_sleeve"])
    assert cash.loc[0, "total_account_nav"] == pytest.approx(full.daily.loc[dates[1], "nav"])


def test_active_attribution_contains_required_fields(reset_case):
    _, tx, _, _, _, active = reset_case
    result = active_sleeve_attribution(
        tx, active, "2026-04-01", {"ABC"},
        execution_price_column="raw_execution_price",
    )
    required = {
        "ticker", "starting_market_value", "net_contribution", "realized_pnl",
        "unrealized_pnl", "dividend", "total_pnl", "return_contribution",
        "drawdown_contribution",
    }
    assert required.issubset(result.columns)
