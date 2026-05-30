from __future__ import annotations

import pytest
from pydantic import ValidationError

from scrc.contracts import (
    AnomalyResult,
    ConfidenceTier,
    MacroSignals,
    RegimeLabel,
    ShapFeature,
    ShapValue,
    StockoutRiskResult,
)


def test_probability_upper_bound_is_enforced() -> None:
    with pytest.raises(ValidationError):
        StockoutRiskResult(
            sku_id="A",
            store_id="CA_1",
            stockout_probability=1.5,
            calibrated=True,
            confidence_tier=ConfidenceTier.HIGH,
        )


def test_stockout_carries_shap_attribution() -> None:
    result = StockoutRiskResult(
        sku_id="A",
        store_id="CA_1",
        stockout_probability=0.81,
        calibrated=True,
        confidence_tier=ConfidenceTier.HIGH,
        shap_values=[ShapValue(feature="lead_time", value=12.0, contribution=0.22)],
        plain_language_brief="Elevated lead time is the dominant driver.",
    )
    assert result.shap_values[0].feature == "lead_time"


def test_anomaly_result_attribution() -> None:
    result = AnomalyResult(
        port_ids=["USLAX"],
        congestion_score=0.7,
        anomaly_flag=True,
        anomaly_score=0.93,
        top_features=[ShapFeature(feature="dwell_time", shap_value=0.4)],
    )
    assert result.anomaly_flag is True
    assert result.top_features[0].feature == "dwell_time"


def test_macro_regime_enum() -> None:
    signals = MacroSignals(
        series_values={"T10Y2Y": -0.3},
        regime_label=RegimeLabel.SHOCK,
        regime_confidence=0.88,
        relevant_tariff_flags=["section_301"],
    )
    assert signals.regime_label is RegimeLabel.SHOCK
