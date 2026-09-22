import pandas as pd

from src.tiingo_loader import audit_tiingo_coverage, load_tiingo_prices
from tests.conftest import make_tx


class FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payloads):
        self.payloads = payloads

    def get(self, url, **kwargs):
        ticker = url.split("/")[-2]
        return FakeResponse(self.payloads.get(ticker, []))


def row(date, close, adj_close, *, dividend=0.0, split=1.0):
    return {
        "date": f"{date}T00:00:00.000Z", "close": close, "adjClose": adj_close,
        "divCash": dividend, "splitFactor": split,
    }


def test_tiingo_loader_and_all_coverage_audits_pass():
    payloads = {
        "ABC": [row("2026-01-02", 100, 99), row("2026-01-05", 101, 100, dividend=1)],
        "VOO": [row("2026-01-02", 500, 490), row("2026-01-05", 501, 491)],
    }
    bundle = load_tiingo_prices(
        ["ABC", "VOO"], start_date="2026-01-02", token="test",
        session=FakeSession(payloads),
    )
    tx = make_tx([
        {"date": pd.Timestamp("2026-01-02"), "type": "buy", "ticker": "ABC", "quantity": 1, "price": 100},
        {"date": pd.Timestamp("2026-01-05"), "type": "sell", "ticker": "ABC", "quantity": 1, "price": 101},
    ])
    audit = audit_tiingo_coverage(bundle, tx, actual_tickers={"ABC"})
    assert audit.passed
    assert audit.ticker_coverage["status"].eq("PASS").all()
    assert audit.trade_date_coverage["status"].eq("PASS").all()
    assert audit.holding_period_coverage.loc[0, "missing_sessions"] == 0
    assert len(audit.corporate_actions) == 1
    assert audit.corporate_actions.loc[0, "divCash"] == 1


def test_tiingo_holding_period_gap_is_fatal():
    payloads = {
        "ABC": [row("2026-01-02", 100, 99), row("2026-01-05", None, None)],
    }
    bundle = load_tiingo_prices(
        ["ABC"], start_date="2026-01-02", token="test", session=FakeSession(payloads),
    )
    tx = make_tx([
        {"date": pd.Timestamp("2026-01-02"), "type": "buy", "ticker": "ABC", "quantity": 1, "price": 100},
    ])
    audit = audit_tiingo_coverage(bundle, tx, actual_tickers={"ABC"})
    assert not audit.passed
    assert audit.holding_period_coverage.loc[0, "status"] == "FAIL"
    assert audit.holding_period_coverage.loc[0, "missing_dates"] == "2026-01-05"


def test_tiingo_missing_ticker_is_fatal():
    bundle = load_tiingo_prices(
        ["MISSING"], start_date="2026-01-02", token="test", session=FakeSession({}),
    )
    tx = make_tx([
        {"date": pd.Timestamp("2026-01-02"), "type": "buy", "ticker": "MISSING", "quantity": 1, "price": 1},
    ])
    audit = audit_tiingo_coverage(bundle, tx, actual_tickers={"MISSING"})
    assert not audit.passed
    assert audit.ticker_coverage.loc[0, "status"] == "FAIL"
    assert audit.trade_date_coverage.loc[0, "status"] == "FAIL"
