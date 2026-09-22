from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.external_price_loader import load_external_prices
from src.price_adjustment import attach_adjusted_execution_prices
from tests.conftest import make_tx


def _trade(date="2026-01-05", ticker="ABC", price=100.0):
    return make_tx([{"date": pd.Timestamp(date), "type": "buy", "ticker": ticker, "quantity": 1, "price": price}])


def test_exact_finlab_factor():
    date = pd.Timestamp("2026-01-05")
    close = pd.DataFrame({"ABC": [200.0]}, index=[date])
    adj = pd.DataFrame({"ABC": [100.0]}, index=[date])
    result, warnings = attach_adjusted_execution_prices(_trade(), close, adj, price_sources={"ABC": "stock"})
    assert not warnings
    assert result.loc[0, "adjustment_factor"] == pytest.approx(0.5)
    assert result.loc[0, "adjustment_source"] == "exact_finlab"
    assert result.loc[0, "adjustment_reference_date"] == date
    assert result.loc[0, "adjustment_date_shift_days"] == 0


def test_missing_exact_date_uses_verified_identical_neighboring_factors():
    dates = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
    close = pd.DataFrame({"ABC": [100.0, np.nan, 110.0]}, index=dates)
    adj = pd.DataFrame({"ABC": [50.0, np.nan, 55.0]}, index=dates)
    result, warnings = attach_adjusted_execution_prices(_trade(price=80), close, adj)
    assert result.loc[0, "adjustment_factor"] == pytest.approx(0.5)
    assert result.loc[0, "adjusted_execution_price"] == pytest.approx(40)
    assert result.loc[0, "adjustment_source"] == "neighbor_factor_verified"
    assert result.loc[0, "adjustment_reference_date"] == dates[2]
    assert result.loc[0, "adjustment_date_shift_days"] == 1
    assert any("verified identical neighboring factors" in warning for warning in warnings)


def test_neighboring_factors_differ_is_fatal():
    dates = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
    close = pd.DataFrame({"ABC": [100.0, np.nan, 100.0]}, index=dates)
    adj = pd.DataFrame({"ABC": [50.0, np.nan, 60.0]}, index=dates)
    result, warnings = attach_adjusted_execution_prices(_trade(), close, adj)
    assert pd.isna(result.loc[0, "adjusted_execution_price"])
    assert result.loc[0, "adjustment_source"] == "unresolved"
    assert any("FATAL" in warning and "disagree" in warning for warning in warnings)


def test_completely_missing_ticker_uses_fallback_csv(tmp_path: Path):
    fallback = tmp_path / "fallback_prices.csv"
    fallback.write_text(
        "date,ticker,close,adj_close,source\n2026-01-05,SYSB,50,49.5,external_manual\n",
        encoding="utf-8",
    )
    bundle = load_external_prices(fallback, required_tickers={"SYSB"})
    result, warnings = attach_adjusted_execution_prices(
        _trade(ticker="SYSB", price=50), bundle.close, bundle.adj_close, price_sources=bundle.sources,
    )
    assert any("audited external price source" in warning for warning in warnings)
    assert result.loc[0, "adjustment_factor"] == pytest.approx(0.99)
    assert result.loc[0, "adjustment_source"] == "exact_external_manual"


def test_completely_missing_ticker_without_fallback_is_fatal(tmp_path: Path):
    bundle = load_external_prices(tmp_path / "does_not_exist.csv", required_tickers={"SYSB"})
    result, warnings = attach_adjusted_execution_prices(
        _trade(ticker="SYSB"), bundle.close, bundle.adj_close, price_sources=bundle.sources,
    )
    assert pd.isna(result.loc[0, "adjusted_execution_price"])
    assert any("FATAL" in warning for warning in warnings)
    assert any("not found" in warning for warning in bundle.warnings)


def test_fallback_audit_fields_are_populated(tmp_path: Path):
    fallback = tmp_path / "fallback_prices.csv"
    fallback.write_text(
        "date,ticker,close,adj_close,source\n2026-01-05,SYSB,50,50,external_manual\n",
        encoding="utf-8",
    )
    bundle = load_external_prices(fallback)
    result, _ = attach_adjusted_execution_prices(
        _trade(ticker="SYSB"), bundle.close, bundle.adj_close, price_sources=bundle.sources,
    )
    required = {
        "raw_execution_price", "close", "adj_close", "adjustment_factor", "adjusted_execution_price",
        "adjustment_source", "adjustment_reference_date", "adjustment_date_shift_days", "warning",
    }
    assert required.issubset(result.columns)
    assert result.loc[0, "adjustment_source"] == "exact_external_manual"
    assert result.loc[0, "adjustment_date_shift_days"] == 0


def test_sysb_is_excluded_from_active_equity_sleeve():
    config = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
    assert "SYSB" in config["known_funds"]
    assert "SYSB" in config["portfolio_definitions"]["active_equity_sleeve"]["exclude"]
    assert "SYSB" in config["portfolio_definitions"]["active_plus_sgov"]["exclude"]
    assert "SYSB" not in config["portfolio_definitions"]["full_actual_account"]["exclude"]


def test_neighbor_resolution_does_not_fill_missing_market_prices():
    dates = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
    close = pd.DataFrame({"ABC": [100.0, np.nan, 110.0]}, index=dates)
    adj = pd.DataFrame({"ABC": [50.0, np.nan, 55.0]}, index=dates)
    result, _ = attach_adjusted_execution_prices(_trade(price=80), close, adj)
    assert pd.isna(result.loc[0, "close"])
    assert pd.isna(result.loc[0, "adj_close"])
    assert result.loc[0, "raw_execution_price"] == 80
    assert result.loc[0, "adjusted_execution_price"] == pytest.approx(40)


def test_single_side_factor_is_allowed_only_one_trading_session_away():
    dates = pd.to_datetime(["2026-01-05", "2026-01-06"])
    close = pd.DataFrame({"ABC": [np.nan, 100.0]}, index=dates)
    adj = pd.DataFrame({"ABC": [np.nan, 50.0]}, index=dates)
    result, warnings = attach_adjusted_execution_prices(_trade(), close, adj)
    assert result.loc[0, "adjustment_factor"] == pytest.approx(0.5)
    assert result.loc[0, "adjustment_source"] == "neighbor_factor_single_side"
    assert any("1 trading session away" in warning for warning in warnings)


def test_single_side_factor_more_than_one_trading_session_away_is_fatal():
    dates = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
    close = pd.DataFrame({"ABC": [np.nan, np.nan, 100.0]}, index=dates)
    adj = pd.DataFrame({"ABC": [np.nan, np.nan, 50.0]}, index=dates)
    result, warnings = attach_adjusted_execution_prices(_trade(), close, adj)
    assert pd.isna(result.loc[0, "adjusted_execution_price"])
    assert any("2 trading sessions away" in warning for warning in warnings)


def test_factor_one_requires_explicit_corporate_action_confirmation(tmp_path: Path):
    unconfirmed = tmp_path / "unconfirmed.csv"
    unconfirmed.write_text(
        "date,ticker,close,adj_close,source\n2026-01-05,SYSB,50,,external_manual\n",
        encoding="utf-8",
    )
    unresolved = load_external_prices(unconfirmed)
    result, warnings = attach_adjusted_execution_prices(
        _trade(ticker="SYSB"), unresolved.close, unresolved.adj_close, price_sources=unresolved.sources,
    )
    assert pd.isna(result.loc[0, "adjusted_execution_price"])
    assert any("FATAL" in warning for warning in warnings)

    confirmed = tmp_path / "confirmed.csv"
    confirmed.write_text(
        "date,ticker,close,adj_close,source,no_corporate_action_confirmed\n"
        "2026-01-05,SYSB,50,,external_manual,true\n",
        encoding="utf-8",
    )
    resolved = load_external_prices(confirmed)
    result, _ = attach_adjusted_execution_prices(
        _trade(ticker="SYSB"), resolved.close, resolved.adj_close, price_sources=resolved.sources,
    )
    assert result.loc[0, "adjustment_factor"] == pytest.approx(1.0)
    assert "no_corporate_action_factor_1" in result.loc[0, "adjustment_source"]
    assert "factor 1.0" in result.loc[0, "warning"]
    assert resolved.warnings
