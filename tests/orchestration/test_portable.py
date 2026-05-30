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
from scrc.orchestration import AgentBundle, run_pipeline


class _Forecast:
    def chronos_forecast(self, sku_id, store_id, horizon_days, covariates=None):  # type: ignore[no-untyped-def]
        return QuantileForecastResult(
            sku_id=sku_id, store_id=store_id, horizon_days=horizon_days, p10=97, p50=100, p90=103
        )


class _Logistics:
    def __init__(self, anomaly: bool) -> None:
        self._a = anomaly

    def detect_freight_anomaly(self, port_id: str) -> AnomalyResult:
        return AnomalyResult(
            port_ids=[port_id],
            congestion_score=0.9 if self._a else 0.2,
            anomaly_flag=self._a,
            anomaly_score=0.9 if self._a else 0.1,
        )


class _Macro:
    def __init__(self, regime: RegimeLabel) -> None:
        self._r = regime

    def assess_macro(self) -> MacroSignals:
        return MacroSignals(series_values={}, regime_label=self._r, regime_confidence=0.6)


class _Stockout:
    def __init__(self, p: float) -> None:
        self._p = p

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


def _bundle(prob: float, anomaly: bool, regime: RegimeLabel) -> AgentBundle:
    return AgentBundle(
        demand=DemandAgent(_Forecast()),  # type: ignore[arg-type]
        logistics=LogisticsAgent(_Logistics(anomaly)),  # type: ignore[arg-type]
        macro=MacroAgent(_Macro(regime)),  # type: ignore[arg-type]
        stockout=StockoutAgent(_Stockout(prob)),  # type: ignore[arg-type]
        supervisor=SupervisorAgent(
            ProvenanceContext(
                feature_schema_version="1.0",
                policy_config_version="default",
                prompt_template_version="1.0",
                llm_model_id="gpt-4o",
                code_git_sha="deadbeef",
            )
        ),
    )


def _request(ports: bool = True) -> DecisionRequest:
    return DecisionRequest(sku_id="A", store_id="CA_1", port_ids=["USLAX"] if ports else [])


def test_pipeline_produces_routine_decision() -> None:
    decision = run_pipeline(_bundle(0.1, anomaly=False, regime=RegimeLabel.NEUTRAL), _request())
    assert decision.tier is EscalationTier.ROUTINE
    assert decision.autonomous is True


def test_pipeline_produces_critical_decision() -> None:
    decision = run_pipeline(_bundle(0.9, anomaly=True, regime=RegimeLabel.SHOCK), _request())
    assert decision.tier is EscalationTier.CRITICAL
    assert decision.autonomous is False


def test_pipeline_escalates_on_specialist_failure() -> None:
    # No ports -> logistics agent raises -> escalate_missing -> CRITICAL.
    decision = run_pipeline(
        _bundle(0.1, anomaly=False, regime=RegimeLabel.NEUTRAL), _request(ports=False)
    )
    assert decision.tier is EscalationTier.CRITICAL


def test_pipeline_matches_no_orchestrator_contract() -> None:
    # Same agents, no orchestrator: still a fully typed SupervisorDecision (P5).
    decision = run_pipeline(_bundle(0.62, anomaly=False, regime=RegimeLabel.NEUTRAL), _request())
    assert decision.provenance.input_hash
    assert decision.decision_id == decision.provenance.input_hash[:16]
