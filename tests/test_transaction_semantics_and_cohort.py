import pandas as pd
import pytest
import yaml

from src.attribution import ticker_summary
from src.benchmarks import cashflow_matched_benchmark
from src.new_money_cohort import build_new_money_cohort, summarize_new_money_result
from src.normalize_transactions import normalize_transactions, unresolved_cash_affecting_rows
from src.portfolio_accounting import reconstruct_portfolio
from tests.conftest import make_tx


ALIASES = {
    "date": ["日期"], "type": ["交易類別"], "description": ["說明"],
    "ticker": ["代號"], "quantity": ["數量"], "price": ["價格"],
    "amount": ["金額"],
}


def normalize_one(raw_type, description, amount=0, ticker=pd.NA, quantity=0, price=0):
    raw = pd.DataFrame([{
        "日期": "2026-04-02", "交易類別": raw_type, "說明": description,
        "代號": ticker, "數量": quantity, "價格": price, "金額": amount,
    }])
    return normalize_transactions(raw, ALIASES)[0].iloc[0]


def test_chinese_deposit_is_classified_and_uses_amount():
    row = normalize_one("存款", "Wire Funds Received", amount=1000)
    assert row["type"] == "deposit"
    assert row["cash_flow"] == pytest.approx(1000)


def test_deposit_is_external_flow_not_investment_pnl(dates):
    tx = make_tx([{"date": dates[0], "type": "deposit", "cash_flow": 1000}])
    result = reconstruct_portfolio(tx, pd.DataFrame({"ABC": [10, 10, 10]}, index=dates))
    assert result.daily.loc[dates[0], "cash"] == pytest.approx(1000)
    assert result.daily.loc[dates[0], "external_cash_flow"] == pytest.approx(1000)
    assert result.daily.loc[dates[0], "investment_pnl"] == pytest.approx(0)


def test_margin_interest_semantics_and_accounting(dates):
    normalized = normalize_one("融資利息費用", "FROM 06/16 THRU 07/15", amount=-5)
    assert normalized["type"] == "margin_interest_expense"
    tx = make_tx([
        {"date": dates[0], "type": "deposit", "cash_flow": 100},
        {"date": dates[1], "type": "margin_interest_expense", "cash_flow": -5, "amount": -5},
    ])
    result = reconstruct_portfolio(tx, pd.DataFrame({"ABC": [10, 10, 10]}, index=dates))
    assert result.daily.loc[dates[1], "cash"] == pytest.approx(95)
    assert result.daily.loc[dates[1], "investment_pnl"] == pytest.approx(-5)
    assert result.daily.loc[dates[1], "external_cash_flow"] == pytest.approx(0)
    assert result.daily.loc[dates[1], "margin_interest_expense"] == pytest.approx(5)


def test_internal_transfer_has_no_account_effect(dates):
    row = normalize_one("其他", "XFER CASH TO MARGIN", amount=-25)
    assert row["type"] == "internal_transfer"
    tx = make_tx([
        {"date": dates[0], "type": "deposit", "cash_flow": 100},
        {"date": dates[1], "type": "internal_transfer", "amount": -25},
    ])
    result = reconstruct_portfolio(tx, pd.DataFrame({"ABC": [10, 10, 10]}, index=dates))
    assert result.daily.loc[dates[1], "nav"] == pytest.approx(100)
    assert result.daily.loc[dates[1], "external_cash_flow"] == pytest.approx(0)
    assert result.daily.loc[dates[1], "investment_pnl"] == pytest.approx(0)
    assert result.daily.loc[dates[1], "twr"] == pytest.approx(0)


def test_wire_rebate_is_external_noninvestment_credit(dates):
    row = normalize_one("其他", "REBATE FOR WIRE 2024-06-27", amount=25)
    assert row["type"] == "non_investment_credit"
    tx = make_tx([
        {"date": dates[0], "type": "deposit", "cash_flow": 100},
        {"date": dates[1], "type": "non_investment_credit", "cash_flow": 25, "amount": 25},
    ])
    result = reconstruct_portfolio(tx, pd.DataFrame({"ABC": [10, 10, 10]}, index=dates))
    assert result.daily.loc[dates[1], "cash"] == pytest.approx(125)
    assert result.daily.loc[dates[1], "external_cash_flow"] == pytest.approx(25)
    assert result.daily.loc[dates[1], "investment_pnl"] == pytest.approx(0)
    assert result.daily.loc[dates[1], "twr"] == pytest.approx(0)


def test_vgt_split_is_not_buy_and_has_no_invested_capital():
    split = normalize_one(
        "買進", "VANGUARD ETF STK SPLIT ON 6.70400 SHS REC 04/17/26",
        amount=0, ticker="VGT", quantity=46.928, price=0,
    )
    assert split["type"] == "split"
    dates = pd.to_datetime(["2026-04-20", "2026-04-21", "2026-04-22"])
    tx = make_tx([
        {"date": dates[0], "type": "deposit", "cash_flow": 5363.2},
        {"date": dates[0], "type": "buy", "ticker": "VGT", "quantity": 6.704,
         "raw_execution_price": 800, "adjusted_execution_price": 800},
        {"date": dates[1], "type": "split", "ticker": "VGT", "quantity": 46.928,
         "price": 0, "amount": 0, "description": split["description"]},
    ])
    prices = pd.DataFrame({"VGT": [800, 100, 101]}, index=dates)
    result = reconstruct_portfolio(tx, prices, execution_price_column="raw_execution_price")
    assert result.holdings.query("date == @dates[1]").iloc[0].shares == pytest.approx(53.632)
    assert result.daily.loc[dates[1], "twr"] == pytest.approx(0)
    summary = ticker_summary(tx, prices.iloc[-1], price_column="raw_execution_price")
    assert summary.loc[0, "total_invested_capital"] == pytest.approx(5363.2)
    assert summary.loc[0, "ending_shares"] == pytest.approx(53.632)


def test_vgt_split_does_not_create_new_money_contribution():
    dates = pd.to_datetime(["2026-04-20", "2026-04-21", "2026-04-22"])
    tx = make_tx([{
        "date": dates[1], "type": "split", "ticker": "VGT", "quantity": 46.928,
        "price": 0, "amount": 0,
        "description": "VGT STK SPLIT ON 6.70400 SHS REC 04/17/26",
    }])
    cohort = build_new_money_cohort(
        "test", tx, pd.DataFrame({"VGT": [800, 100, 101]}, index=dates),
        dates[0], {"VGT"}, execution_price_column="raw_execution_price",
    )
    assert cohort.cashflows.empty
    assert cohort.daily["nav"].eq(0).all()


@pytest.fixture
def cohort_case():
    dates = pd.to_datetime([
        "2026-03-31", "2026-04-01", "2026-04-02", "2026-04-03", "2026-04-06",
    ])
    tx = make_tx([
        {"date": dates[0], "type": "buy", "ticker": "ABC", "quantity": 10,
         "raw_execution_price": 90},
        {"date": dates[1], "type": "sell", "ticker": "ABC", "quantity": 5,
         "raw_execution_price": 100},
        {"date": dates[2], "type": "buy", "ticker": "ABC", "quantity": 3,
         "raw_execution_price": 105},
        {"date": dates[3], "type": "sell", "ticker": "ABC", "quantity": 5,
         "raw_execution_price": 110},
    ])
    prices = pd.DataFrame({
        "ABC": [90, 100, 106, 110, 112], "GLD": [200, 201, 202, 203, 204],
        "VOO": [100, 101, 102, 103, 104], "VGT": [200, 202, 204, 206, 208],
        "SGOV": [100, 100, 100, 100, 100],
    }, index=dates)
    cohort = build_new_money_cohort(
        "new_money_active_equity", tx, prices, "2026-04-01", {"ABC"},
        execution_price_column="raw_execution_price",
    )
    return dates, tx, prices, cohort


def test_new_money_starts_with_zero_and_excludes_pre_start_inventory(cohort_case):
    dates, _, _, cohort = cohort_case
    assert cohort.daily.loc[dates[1], "nav"] == pytest.approx(0)
    assert cohort.holdings.loc[cohort.holdings.date.eq(dates[1])].empty


def test_pre_start_sell_is_ignored_and_recorded(cohort_case):
    dates, _, _, cohort = cohort_case
    row = cohort.trade_audit.loc[cohort.trade_audit.date.eq(dates[1])].iloc[0]
    assert row.cohort_sell_quantity == pytest.approx(0)
    assert row.ignored_pre_start_sell_quantity == pytest.approx(5)
    assert row.cohort_quantity_after == pytest.approx(0)


def test_post_start_buy_enters_cohort_and_sell_is_capped(cohort_case):
    dates, _, _, cohort = cohort_case
    buy = cohort.trade_audit.loc[cohort.trade_audit.date.eq(dates[2])].iloc[0]
    sell = cohort.trade_audit.loc[cohort.trade_audit.date.eq(dates[3])].iloc[0]
    assert buy.cohort_buy_quantity == pytest.approx(3)
    assert sell.cohort_sell_quantity == pytest.approx(3)
    assert sell.ignored_pre_start_sell_quantity == pytest.approx(2)
    assert sell.cohort_quantity_after == pytest.approx(0)


def test_config_separates_equity_from_discretionary_gld():
    config = yaml.safe_load(open("config/config.yaml", encoding="utf-8"))
    assert "GLD" in config["portfolio_definitions"]["active_equity_selection"]["exclude"]
    assert "GLD" not in config["portfolio_definitions"]["active_discretionary_risk"]["exclude"]


@pytest.mark.parametrize("ticker", ["VOO", "VGT"])
def test_decision_benchmark_uses_same_date_and_dollar(cohort_case, ticker):
    dates, _, prices, cohort = cohort_case
    contributions = cohort.cashflows.loc[cohort.cashflows.flow_type.eq("contribution")]
    flows = contributions.groupby("date").amount.sum().reindex(cohort.daily.index, fill_value=0.0)
    benchmark = cashflow_matched_benchmark(flows, prices.loc[cohort.daily.index], {ticker: 1.0})
    assert benchmark.loc[dates[2], "external_cash_flow"] == pytest.approx(315)
    assert benchmark.loc[dates[2], f"shares_{ticker}"] == pytest.approx(315 / prices.loc[dates[2], ticker])
    assert benchmark.external_cash_flow.sum() == pytest.approx(contributions.amount.sum())


def test_buy_timing_changes_ending_wealth_but_not_pure_voo_twr():
    dates = pd.to_datetime(["2026-04-01", "2026-04-02", "2026-04-03"])
    prices = pd.DataFrame({"VOO": [100, 110, 120]}, index=dates)
    early = cashflow_matched_benchmark(pd.Series([100, 0, 0], index=dates), prices, {"VOO": 1.0})
    late = cashflow_matched_benchmark(pd.Series([0, 100, 0], index=dates), prices, {"VOO": 1.0})
    assert early.iloc[-1].nav != pytest.approx(late.iloc[-1].nav)
    assert early["return"].dropna().iloc[-1] == pytest.approx(late["return"].dropna().iloc[-1])


def test_pure_voo_twr_equivalence_does_not_make_wealth_equal():
    dates = pd.to_datetime(["2026-04-01", "2026-04-02", "2026-04-03"])
    prices = pd.DataFrame({"VOO": [100, 110, 120]}, index=dates)
    small = cashflow_matched_benchmark(pd.Series([100, 0, 0], index=dates), prices, {"VOO": 1.0})
    large = cashflow_matched_benchmark(pd.Series([200, 0, 0], index=dates), prices, {"VOO": 1.0})
    pd.testing.assert_series_equal(small["return"], large["return"])
    assert small.iloc[-1].nav != pytest.approx(large.iloc[-1].nav)


def test_cohort_dividend_allocates_only_post_start_share_fraction():
    dates = pd.to_datetime(["2026-04-01", "2026-04-02", "2026-04-03", "2026-04-06"])
    tx = make_tx([
        {"date": dates[1], "type": "buy", "ticker": "ABC", "quantity": 2,
         "raw_execution_price": 100},
        {"date": dates[3], "type": "dividend", "ticker": "ABC", "dividend": 12,
         "description": "ABC CASH DIV ON 12.00000 SHS REC 04/03/26 PAY 04/06/26"},
    ])
    prices = pd.DataFrame({"ABC": [100, 100, 100, 100]}, index=dates)
    cohort = build_new_money_cohort(
        "new_money_active_equity", tx, prices, "2026-04-01", {"ABC"},
        execution_price_column="raw_execution_price",
    )
    audit = cohort.dividend_audit.iloc[0]
    assert audit.broker_eligible_shares == pytest.approx(12)
    assert audit.cohort_shares == pytest.approx(2)
    assert audit.cohort_dividend_allocated == pytest.approx(2)


def test_unknown_nonzero_amount_is_cash_affecting_fatal_candidate():
    tx = pd.DataFrame({"type": ["unknown", "unknown"], "amount": [25.0, 0.0]})
    unresolved = unresolved_cash_affecting_rows(tx)
    assert len(unresolved) == 1
    assert unresolved.iloc[0].amount == pytest.approx(25)


def test_summary_preserves_gross_same_day_contribution_and_withdrawal():
    dates = pd.to_datetime(["2026-04-01", "2026-04-02"])
    daily = pd.DataFrame({
        "nav": [0.0, 70.0], "market_value": [0.0, 70.0], "cash": [0.0, 0.0],
        "external_cash_flow": [0.0, 70.0], "twr": [0.0, 0.0],
    }, index=dates)
    cashflows = pd.DataFrame({
        "flow_type": ["contribution", "withdrawal"], "amount": [100.0, -30.0],
    })
    summary = summarize_new_money_result(
        "test", daily, pd.Series([0.0, 0.0], index=dates),
        risk_free_rate=0.0, annualization=252, transaction_cashflows=cashflows,
    )
    assert summary["total_contributed_capital"] == pytest.approx(100.0)
    assert summary["total_withdrawn_capital"] == pytest.approx(30.0)
    assert summary["ending_wealth"] == pytest.approx(100.0)
    assert summary["dollar_pnl"] == pytest.approx(0.0)
