from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langgraph.types import Command

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


def _invoke(bundle: AgentBundle, request: DecisionRequest, thread: str) -> tuple[Any, dict, dict]:
    app = build_graph(bundle)
    config = {"configurable": {"thread_id": thread}}
    result = app.invoke({"request": request, "errors": []}, config)
    return app, config, result


def _request(ports: bool = True) -> DecisionRequest:
    return DecisionRequest(sku_id="A", store_id="CA_1", port_ids=["USLAX"] if ports else [])


def test_routine_decision_is_autonomous_and_completes() -> None:
    _, _, result = _invoke(_bundle(0.1, anomaly=False, regime=RegimeLabel.NEUTRAL), _request(), "r")
    assert "__interrupt__" not in result
    assert result["decision"].tier is EscalationTier.ROUTINE
    assert result["decision"].autonomous is True


def test_escalated_decision_interrupts_for_hitl() -> None:
    _, _, result = _invoke(_bundle(0.9, anomaly=True, regime=RegimeLabel.SHOCK), _request(), "c")
    assert "__interrupt__" in result  # graph paused for a human
    assert result["decision"].tier is EscalationTier.CRITICAL
    payload = result["__interrupt__"][0].value
    assert payload["tier"] == "critical"
    assert payload["brief"]  # the planner receives a full brief, not a bare prompt


def test_resume_records_human_outcome() -> None:
    app, config, result = _invoke(
        _bundle(0.62, anomaly=False, regime=RegimeLabel.NEUTRAL), _request(), "resume"
    )
    assert "__interrupt__" in result
    final = app.invoke(Command(resume={"approved": True, "reviewer_id": "planner1"}), config)
    assert "__interrupt__" not in final
    assert final["human_outcome"]["approved"] is True
    assert final["human_outcome"]["reviewer_id"] == "planner1"
    assert final["decision"].tier is EscalationTier.REVIEW


def test_missing_signal_escalates_to_critical_and_interrupts() -> None:
    # No port_ids -> logistics agent raises -> anomaly None -> Supervisor escalates.
    _, _, result = _invoke(
        _bundle(0.1, anomaly=False, regime=RegimeLabel.NEUTRAL), _request(ports=False), "m"
    )
    assert result["decision"].tier is EscalationTier.CRITICAL
    assert result["decision"].autonomous is False
    assert any("logistics" in e for e in result.get("errors", []))
    assert "__interrupt__" in result


def test_decision_is_reproducible_across_threads() -> None:
    _, _, first = _invoke(_bundle(0.62, anomaly=False, regime=RegimeLabel.NEUTRAL), _request(), "a")
    _, _, second = _invoke(
        _bundle(0.62, anomaly=False, regime=RegimeLabel.NEUTRAL), _request(), "b"
    )
    assert first["decision"].decision_id == second["decision"].decision_id
    assert first["decision"].tier is second["decision"].tier
