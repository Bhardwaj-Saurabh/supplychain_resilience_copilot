"""Supervisor Agent (Layer 4) — synthesis, tiering, action proposal.

Synthesises the four agent outputs into a typed ``SupervisorDecision``. The tier
and autonomy come from the deterministic governance policy (ADR-0001); the
Supervisor proposes ranked actions and stamps provenance, but never lets an LLM
choose the tier. For escalated decisions it builds a ``ReviewRequest`` whose
brief is LLM-narrated when a ``BriefWriter`` is wired, or a deterministic SHAP
template otherwise (the "LLM down" path, architecture.md §17/§19).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from scrc.contracts import (
    ActionRecommendation,
    ActionType,
    AnomalyResult,
    DecisionProvenance,
    DecisionRequest,
    MacroSignals,
    QuantileForecastResult,
    RegimeLabel,
    ReviewRequest,
    StockoutRiskResult,
    SupervisorDecision,
)
from scrc.governance import EscalationSignals, evaluate_escalation
from scrc.governance.escalation import HIGH_UNCERTAINTY_RATIO, REVIEW_MIN_PROB
from scrc.llm import BriefWriter, compose_brief_facts


@dataclass(frozen=True)
class ProvenanceContext:
    """Static provenance fields, injected at the composition root. The per-run
    ``input_hash`` is computed from the request (architecture.md §20)."""

    feature_schema_version: str
    policy_config_version: str
    prompt_template_version: str
    llm_model_id: str
    code_git_sha: str
    model_versions: dict[str, str] = field(default_factory=dict)


def _uncertainty_ratio(forecast: QuantileForecastResult) -> float:
    return forecast.interval_width / max(abs(forecast.p50), 1e-9)


class SupervisorAgent:
    def __init__(
        self, provenance: ProvenanceContext, brief_writer: BriefWriter | None = None
    ) -> None:
        self._provenance = provenance
        self._brief_writer = brief_writer

    def build_review_request(
        self, decision: SupervisorDecision, stockout: StockoutRiskResult | None
    ) -> ReviewRequest:
        """Compose the HITL brief for an escalated decision (PRD §6.2).

        With a model output: LLM-narrated when a BriefWriter is wired, else the
        deterministic SHAP facts verbatim. Without one (missing/timed-out
        signal): a plain note — the planner is told the picture is incomplete
        rather than shown a fabricated brief.
        """
        if stockout is None:
            brief = (
                f"{decision.tier.value.upper()}: a required upstream signal was "
                "missing or timed out; no model output is available. Manual "
                "assessment required before any action."
            )
        elif self._brief_writer is not None:
            brief = self._brief_writer.write(decision, stockout)
        else:
            brief = compose_brief_facts(decision, stockout)
        return ReviewRequest(
            decision_id=decision.decision_id,
            tier=decision.tier,
            brief=brief,
            recommended_actions=decision.recommended_actions,
            counterfactual=None,
        )

    def synthesise(
        self,
        request: DecisionRequest,
        forecast: QuantileForecastResult,
        anomaly: AnomalyResult,
        macro: MacroSignals,
        stockout: StockoutRiskResult,
    ) -> SupervisorDecision:
        signals = self._signals(forecast, anomaly, macro, stockout)
        outcome = evaluate_escalation(signals)
        actions = self._propose_actions(anomaly, stockout)
        provenance = self._build_provenance(request)
        return SupervisorDecision(
            decision_id=provenance.input_hash[:16],
            sku_id=request.sku_id,
            store_id=request.store_id,
            tier=outcome.tier,
            stockout_probability=stockout.stockout_probability,
            autonomous=outcome.autonomous,
            recommended_actions=actions,
            provenance=provenance,
        )

    @staticmethod
    def _signals(
        forecast: QuantileForecastResult,
        anomaly: AnomalyResult,
        macro: MacroSignals,
        stockout: StockoutRiskResult,
    ) -> EscalationSignals:
        ratio = _uncertainty_ratio(forecast)
        macro_shock = macro.regime_label is RegimeLabel.SHOCK
        elevated = [
            anomaly.anomaly_flag,
            macro_shock,
            stockout.stockout_probability >= REVIEW_MIN_PROB,
            ratio > HIGH_UNCERTAINTY_RATIO,
        ]
        return EscalationSignals(
            stockout_probability=stockout.stockout_probability,
            uncertainty_ratio=ratio,
            anomaly_flag=anomaly.anomaly_flag,
            macro_shock=macro_shock,
            signal_count=sum(elevated),
        )

    @staticmethod
    def _propose_actions(
        anomaly: AnomalyResult, stockout: StockoutRiskResult
    ) -> list[ActionRecommendation]:
        proposals: list[tuple[ActionType, str, bool]] = []
        if anomaly.anomaly_flag:
            proposals.append((ActionType.REROUTE, "logistics anomaly detected", True))
            proposals.append((ActionType.EXPEDITE, "mitigate inbound delay risk", True))
        if stockout.stockout_probability >= REVIEW_MIN_PROB:
            proposals.append(
                (ActionType.SAFETY_STOCK_TRANSFER, "elevated stockout probability", True)
            )
            proposals.append((ActionType.SUBSTITUTE, "prepare substitute SKU", False))
        return [
            ActionRecommendation(action_type=a, rank=i + 1, rationale=r, reversible=rev)
            for i, (a, r, rev) in enumerate(proposals)
        ]

    def escalate_missing(self, request: DecisionRequest, reason: str) -> SupervisorDecision:
        """Build a CRITICAL, non-autonomous decision when a required upstream
        signal is missing/timed-out — never hallucinate past it (PRD §7.4)."""
        outcome = evaluate_escalation(
            EscalationSignals(stockout_probability=0.0, uncertainty_ratio=0.0, missing_signal=True)
        )
        provenance = self._build_provenance(request)
        return SupervisorDecision(
            decision_id=provenance.input_hash[:16],
            sku_id=request.sku_id,
            store_id=request.store_id,
            tier=outcome.tier,
            stockout_probability=0.0,
            autonomous=outcome.autonomous,
            recommended_actions=[],
            provenance=provenance,
        )

    def _build_provenance(self, request: DecisionRequest) -> DecisionProvenance:
        input_hash = hashlib.sha256(request.model_dump_json().encode()).hexdigest()
        p = self._provenance
        return DecisionProvenance(
            model_versions=p.model_versions,
            feature_schema_version=p.feature_schema_version,
            policy_config_version=p.policy_config_version,
            prompt_template_version=p.prompt_template_version,
            llm_model_id=p.llm_model_id,
            code_git_sha=p.code_git_sha,
            input_hash=input_hash,
        )
