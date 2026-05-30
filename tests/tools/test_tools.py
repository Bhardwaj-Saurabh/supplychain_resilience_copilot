from __future__ import annotations

from collections.abc import Mapping, Sequence

from scrc.contracts import (
    AnomalyResult,
    CongestionMetrics,
    MacroSignals,
    QuantileForecastResult,
    RegimeLabel,
    StockoutRiskResult,
)
from scrc.tools import (
    ForecastTool,
    LogisticsTool,
    MacroTool,
    StaticFeatureProvider,
    StockoutTool,
    classify_regime,
)


class FakeForecaster:
    def forecast(
        self,
        sku_id: str,
        store_id: str,
        context: Sequence[float],
        horizon_days: int,
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> QuantileForecastResult:
        # Echo the context length so the test can assert it was read from the store.
        base = float(len(context))
        return QuantileForecastResult(
            sku_id=sku_id,
            store_id=store_id,
            horizon_days=horizon_days,
            p10=base,
            p50=base + 1,
            p90=base + 2,
        )


class FakeAnomalyModel:
    def predict(
        self, port_ids: Sequence[str], features: Mapping[str, float], congestion_score: float
    ) -> AnomalyResult:
        return AnomalyResult(
            port_ids=list(port_ids),
            congestion_score=congestion_score,
            anomaly_flag=congestion_score > 0.8,
            anomaly_score=congestion_score,
            top_features=[],
        )


class FakeStockoutModel:
    def predict(
        self, sku_id: str, store_id: str, features: Mapping[str, float]
    ) -> StockoutRiskResult:
        from scrc.contracts import ConfidenceTier

        return StockoutRiskResult(
            sku_id=sku_id,
            store_id=store_id,
            stockout_probability=min(features.get("lead_time", 0.0) / 30.0, 1.0),
            calibrated=True,
            confidence_tier=ConfidenceTier.MEDIUM,
        )


def test_forecast_tool_reads_context_and_delegates() -> None:
    provider = StaticFeatureProvider(demand_contexts={("A", "CA_1"): [1.0, 2.0, 3.0, 4.0]})
    tool = ForecastTool(FakeForecaster(), provider, lookback=3)
    result = tool.chronos_forecast("A", "CA_1", horizon_days=14)
    assert isinstance(result, QuantileForecastResult)
    # lookback=3 trims the 4-point context to 3.
    assert result.p10 == 3.0
    assert result.horizon_days == 14


def test_logistics_tool_congestion_average_and_anomaly() -> None:
    provider = StaticFeatureProvider(
        logistics={
            "USLAX": {"congestion_index": 0.9, "dwell_hours": 40.0},
            "USNYC": {"congestion_index": 0.5, "dwell_hours": 20.0},
        }
    )
    tool = LogisticsTool(FakeAnomalyModel(), provider)

    congestion = tool.get_port_congestion(["USLAX", "USNYC"])
    assert isinstance(congestion, CongestionMetrics)
    assert congestion.congestion_score == 0.7

    anomaly = tool.detect_freight_anomaly("USLAX")
    assert isinstance(anomaly, AnomalyResult)
    assert anomaly.congestion_score == 0.9
    assert anomaly.anomaly_flag is True


def test_stockout_tool_passthrough() -> None:
    tool = StockoutTool(FakeStockoutModel())
    result = tool.classify_stockout_risk("A", "CA_1", {"lead_time": 15.0})
    assert isinstance(result, StockoutRiskResult)
    assert result.stockout_probability == 0.5


def test_macro_tool_assess() -> None:
    provider = StaticFeatureProvider(macro={"t10y2y": -0.4, "ism_pmi": 48.0})
    tool = MacroTool(provider, series_ids=["t10y2y", "ism_pmi"])
    signals = tool.assess_macro()
    assert isinstance(signals, MacroSignals)
    assert signals.regime_label is RegimeLabel.TIGHTENING


def test_classify_regime_branches() -> None:
    assert classify_regime({"consumer_sentiment": 40.0})[0] is RegimeLabel.SHOCK
    assert classify_regime({"t10y2y": -0.5})[0] is RegimeLabel.TIGHTENING
    assert classify_regime({"ism_pmi": 58.0})[0] is RegimeLabel.EASING
    assert classify_regime({"ism_pmi": 50.0})[0] is RegimeLabel.NEUTRAL
