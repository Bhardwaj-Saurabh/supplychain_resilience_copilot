from __future__ import annotations

from collections.abc import Mapping

from scrc.agents import (
    DemandAgent,
    LogisticsAgent,
    MacroAgent,
    ProvenanceContext,
    StockoutAgent,
    SupervisorAgent,
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
from scrc.orchestration import AgentBundle, build_graph


class FakeForecastTool:
    def __init__(self, width: float = 6.0) -> None:
        self._w = width

    def chronos_forecast(
        self, sku_id: str, store_id: str, horizon_days: int, covariates: object = None
    ) -> QuantileForecastResult:
        return QuantileForecastResult(
            sku_id=sku_id,
            store_id=store_id,
            horizon_days=horizon_days,
            p10=100 - self._w / 2,
            p50=100,
            p90=100 + self._w / 2,
        )


class FakeLogisticsTool:
    def __init__(self, anomaly: bool) -> None:
        self._anomaly = anomaly

    def detect_freight_anomaly(self, port_id: str) -> AnomalyResult:
        return AnomalyResult(
            port_ids=[port_id],
            congestion_score=0.9 if self._anomaly else 0.2,
            anomaly_flag=self._anomaly,
            anomaly_score=0.95 if self._anomaly else 0.1,
        )


class FakeMacroTool:
    def __init__(self, regime: RegimeLabel) -> None:
        self._regime = regime

    def assess_macro(self) -> MacroSignals:
        return MacroSignals(series_values={}, regime_label=self._regime, regime_confidence=0.6)


class FakeStockoutTool:
    def __init__(self, probability: float) -> None:
        self._p = probability

    def classify_stockout_risk(
        self, sku_id: str, store_id: str, features: Mapping[str, float]
    ) -> StockoutRiskResult:
        return StockoutRiskResult(
            sku_id=sku_id,
            store_id=store_id,
            stockout_probability=self._p,
            calibrated=True,
            confidence_tier=ConfidenceTier.MEDIUM,
        )


def _bundle(prob: float, anomaly: bool, regime: RegimeLabel, width: float = 6.0) -> AgentBundle:
    provenance = ProvenanceContext(
        feature_schema_version="1.0",
        policy_config_version="default",
        prompt_template_version="1.0",
        llm_model_id="gpt-4o",
        code_git_sha="deadbeef",
    )
    return AgentBundle(
        demand=DemandAgent(FakeForecastTool(width)),  # type: ignore[arg-type]
        logistics=LogisticsAgent(FakeLogisticsTool(anomaly)),  # type: ignore[arg-type]
        macro=MacroAgent(FakeMacroTool(regime)),  # type: ignore[arg-type]
        stockout=StockoutAgent(FakeStockoutTool(prob)),  # type: ignore[arg-type]
        supervisor=SupervisorAgent(provenance),
    )


def _run(bundle: AgentBundle, request: DecisionRequest) -> dict[str, object]:
    app = build_graph(bundle)
    return app.invoke({"request": request, "errors": []})


def test_graph_routine_path() -> None:
    request = DecisionRequest(sku_id="A", store_id="CA_1", port_ids=["USLAX"])
    state = _run(_bundle(0.1, anomaly=False, regime=RegimeLabel.NEUTRAL), request)
    decision = state["decision"]
    assert decision.tier is EscalationTier.ROUTINE  # type: ignore[attr-defined]
    assert decision.autonomous is True  # type: ignore[attr-defined]


def test_graph_critical_path_multi_signal() -> None:
    request = DecisionRequest(sku_id="A", store_id="CA_1", port_ids=["USLAX"])
    state = _run(_bundle(0.9, anomaly=True, regime=RegimeLabel.SHOCK), request)
    decision = state["decision"]
    assert decision.tier is EscalationTier.CRITICAL  # type: ignore[attr-defined]
    assert decision.autonomous is False  # type: ignore[attr-defined]
    assert decision.recommended_actions  # type: ignore[attr-defined]


def test_graph_escalates_to_critical_on_missing_signal() -> None:
    # No port_ids -> the logistics agent raises -> node degrades to anomaly=None
    # -> Supervisor escalates to CRITICAL rather than acting on a partial picture.
    request = DecisionRequest(sku_id="A", store_id="CA_1")
    state = _run(_bundle(0.1, anomaly=False, regime=RegimeLabel.NEUTRAL), request)
    decision = state["decision"]
    assert decision.tier is EscalationTier.CRITICAL  # type: ignore[attr-defined]
    assert decision.autonomous is False  # type: ignore[attr-defined]
    assert any("logistics" in e for e in state.get("errors", []))  # type: ignore[union-attr]


def test_graph_decision_is_reproducible() -> None:
    request = DecisionRequest(sku_id="A", store_id="CA_1", port_ids=["USLAX"])
    first = _run(_bundle(0.62, anomaly=False, regime=RegimeLabel.NEUTRAL), request)["decision"]
    second = _run(_bundle(0.62, anomaly=False, regime=RegimeLabel.NEUTRAL), request)["decision"]
    assert first.decision_id == second.decision_id  # type: ignore[attr-defined]
    assert first.tier is second.tier  # type: ignore[attr-defined]
