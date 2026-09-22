import pandas as pd

from src.normalize_transactions import audit_transactions, normalize_transactions


ALIASES = {"date": ["日期"], "type": ["交易類別"], "ticker": ["股票代號"], "quantity": ["數量"], "price": ["成交價格"]}


def test_column_detection_and_normalization():
    raw = pd.DataFrame({"日期": ["2026-01-02"], "交易類別": ["買進"], "股票代號": ["abc"], "數量": ["1"], "成交價格": ["$10.00"]})
    result, mapping = normalize_transactions(raw, ALIASES)
    assert mapping["ticker"] == "股票代號"
    assert result.loc[0, "type"] == "buy"
    assert result.loc[0, "ticker"] == "ABC"
    assert result.loc[0, "price"] == 10


def test_audit_detects_negative_holding():
    tx = pd.DataFrame({"date": pd.to_datetime(["2026-01-02"]), "type": ["sell"], "ticker": ["ABC"],
                       "quantity": [2.0], "price": [10.0], "source_row": [0]})
    audit = audit_transactions(tx)
    assert audit.summary.loc[0, "negative_holding_events"] == 1
    assert any("Negative holdings" in warning for warning in audit.warnings)
    assert audit.negative_holdings.loc[0, "ticker"] == "ABC"
    assert audit.negative_holdings.loc[0, "cumulative_position_after"] == -2
    assert audit.negative_holdings.loc[0, "status"] == "unresolved"


def test_semantically_clear_types_are_classified_but_manual_types_remain_auditable():
    raw = pd.DataFrame({
        "日期": ["2026-01-02", "2026-01-03", "2026-01-04"],
        "交易類別": ["Credit Interest", "Withholding Tax", "Journal"],
        "股票代號": [pd.NA, pd.NA, "ABC"],
        "數量": [pd.NA, pd.NA, 1],
        "成交價格": [pd.NA, pd.NA, pd.NA],
        "amount": [5, -1, pd.NA],
    })
    aliases = {**ALIASES, "amount": ["amount"]}
    result, _ = normalize_transactions(raw, aliases)
    assert result["type"].tolist() == ["interest", "withholding_tax", "journal"]
    audit = audit_transactions(result)
    assert set(audit.transaction_type_diagnostic["normalized_type"]) == {"interest", "withholding_tax", "journal"}
    assert audit.unrecognized_types.empty
