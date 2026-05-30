from __future__ import annotations

from collections.abc import Mapping

from scrc.agents import (
    DemandAgent,
    LogisticsAgent,
    MacroAgent,
    ProvenanceContext,
    StockoutAgent,
    SupervisorAgent,
    assemble_features,
)
from scrc.contracts import (
    AnomalyResult,
    ConfidenceTier,
    DecisionRequest,
    EscalationTier,
    MacroSignals,
    QuantileForecastResult,
    RegimeLabel,
    StockoutRiskResult,
)

# --- fakes (agents are decoupled from concrete tools at runtime) -----------


class FakeForecastTool:
    def chronos_forecast(
        self, sku_id: str, store_id: str, horizon_days: int, covariates: object = None
    ) -> QuantileForecastResult:
        return QuantileForecastResult(
            sku_id=sku_id, store_id=store_id, horizon_days=horizon_days, p10=90, p50=100, p90=110
        )


class FakeLogisticsTool:
    def __init__(self, anomaly: bool = False) -> None:
        self._anomaly = anomaly

    def detect_freight_anomaly(self, port_id: str) -> AnomalyResult:
        return AnomalyResult(
            port_ids=[port_id],
            congestion_score=0.9 if self._anomaly else 0.2,
            anomaly_flag=self._anomaly,
            anomaly_score=0.95 if self._anomaly else 0.1,
            top_features=[],
        )


class FakeMacroTool:
    def __init__(self, regime: RegimeLabel = RegimeLabel.NEUTRAL) -> None:
        self._regime = regime

    def assess_macro(self) -> MacroSignals:
        return MacroSignals(series_values={}, regime_label=self._regime, regime_confidence=0.6)


class FakeStockoutTool:
    def __init__(self, probability: float) -> None:
        self._p = probability
        self.last_features: Mapping[str, float] | None = None

    def classify_stockout_risk(
        self, sku_id: str, store_id: str, features: Mapping[str, float]
    ) -> StockoutRiskResult:
        self.last_features = features
        return StockoutRiskResult(
            sku_id=sku_id,
            store_id=store_id,
            stockout_probability=self._p,
            calibrated=True,
            confidence_tier=ConfidenceTier.MEDIUM,
        )


def _request() -> DecisionRequest:
    return DecisionRequest(sku_id="A", store_id="CA_1", horizon_days=14, port_ids=["USLAX"])


def _provenance() -> ProvenanceContext:
    return ProvenanceContext(
        feature_schema_version="1.0",
        policy_config_version="default",
        prompt_template_version="1.0",
        llm_model_id="gpt-4o",
        code_git_sha="deadbeef",
        model_versions={"chronos": "1"},
    )


def test_specialist_agents_return_typed_outputs() -> None:
    req = _request()
    assert DemandAgent(FakeForecastTool()).run(req).p50 == 100  # type: ignore[arg-type]
    assert LogisticsAgent(FakeLogisticsTool()).run(req).port_ids == ["USLAX"]  # type: ignore[arg-type]
    assert MacroAgent(FakeMacroTool()).run().regime_label is RegimeLabel.NEUTRAL  # type: ignore[arg-type]


def test_logistics_agent_requires_ports() -> None:
    import pytest

    with pytest.raises(ValueError):
        LogisticsAgent(FakeLogisticsTool()).run(  # type: ignore[arg-type]
            DecisionRequest(sku_id="A", store_id="CA_1")
        )


def test_stockout_agent_assembles_joint_features() -> None:
    forecast = QuantileForecastResult(
        sku_id="A", store_id="CA_1", horizon_days=14, p10=90, p50=100, p90=140
    )
    anomaly = AnomalyResult(
        port_ids=["USLAX"], congestion_score=0.8, anomaly_flag=True, anomaly_score=0.9
    )
    macro = MacroSignals(series_values={}, regime_label=RegimeLabel.SHOCK, regime_confidence=0.7)
    tool = FakeStockoutTool(0.6)
    StockoutAgent(tool).run(_request(), forecast, anomaly, macro)  # type: ignore[arg-type]
    assert tool.last_features is not None
    assert tool.last_features["macro_shock"] == 1.0
    assert tool.last_features["anomaly_flag"] == 1.0
    assert tool.last_features["demand_interval_width"] == 50.0


def test_assemble_features_merges_base() -> None:
    forecast = QuantileForecastResult(
        sku_id="A", store_id="CA_1", horizon_days=7, p10=1, p50=2, p90=3
    )
    anomaly = AnomalyResult(
        port_ids=[], congestion_score=0.1, anomaly_flag=False, anomaly_score=0.0
    )
    macro = MacroSignals(series_values={}, regime_label=RegimeLabel.NEUTRAL, regime_confidence=0.5)
    feats = assemble_features(forecast, anomaly, macro, base={"lead_time": 12.0})
    assert feats["lead_time"] == 12.0
    assert feats["macro_shock"] == 0.0


def _synthesise(probability: float, anomaly: bool, regime: RegimeLabel) -> object:
    req = _request()
    forecast = QuantileForecastResult(
        sku_id="A", store_id="CA_1", horizon_days=14, p10=95, p50=100, p90=110
    )
    anomaly_res = AnomalyResult(
        port_ids=["USLAX"],
        congestion_score=0.9 if anomaly else 0.2,
        anomaly_flag=anomaly,
        anomaly_score=0.9 if anomaly else 0.1,
    )
    macro = MacroSignals(series_values={}, regime_label=regime, regime_confidence=0.6)
    stockout = StockoutRiskResult(
        sku_id="A",
        store_id="CA_1",
        stockout_probability=probability,
        calibrated=True,
        confidence_tier=ConfidenceTier.MEDIUM,
    )
    return SupervisorAgent(_provenance()).synthesise(req, forecast, anomaly_res, macro, stockout)


def test_supervisor_routine_is_autonomous_no_actions() -> None:
    decision = _synthesise(0.1, anomaly=False, regime=RegimeLabel.NEUTRAL)
    assert decision.tier is EscalationTier.ROUTINE  # type: ignore[attr-defined]
    assert decision.autonomous is True  # type: ignore[attr-defined]
    assert decision.recommended_actions == []  # type: ignore[attr-defined]


def test_supervisor_critical_blocks_autonomy_with_actions() -> None:
    decision = _synthesise(0.9, anomaly=True, regime=RegimeLabel.SHOCK)
    assert decision.tier is EscalationTier.CRITICAL  # type: ignore[attr-defined]
    assert decision.autonomous is False  # type: ignore[attr-defined]
    assert len(decision.recommended_actions) >= 1  # type: ignore[attr-defined]
    assert decision.provenance.input_hash  # type: ignore[attr-defined]


def test_supervisor_is_reproducible() -> None:
    first = _synthesise(0.62, anomaly=False, regime=RegimeLabel.NEUTRAL)
    second = _synthesise(0.62, anomaly=False, regime=RegimeLabel.NEUTRAL)
    assert first.decision_id == second.decision_id  # type: ignore[attr-defined]
    assert first.tier is second.tier  # type: ignore[attr-defined]
