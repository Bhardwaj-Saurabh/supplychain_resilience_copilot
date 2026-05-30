from __future__ import annotations

from scrc.agents import ProvenanceContext, SupervisorAgent
from scrc.contracts import (
    ActionRecommendation,
    ActionType,
    ConfidenceTier,
    DecisionProvenance,
    EscalationTier,
    ShapValue,
    StockoutRiskResult,
    SupervisorDecision,
)
from scrc.llm import BriefWriter, compose_brief_facts


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return "NARRATED BRIEF"


def _decision(tier: EscalationTier = EscalationTier.REVIEW) -> SupervisorDecision:
    provenance = DecisionProvenance(
        feature_schema_version="1.0",
        policy_config_version="default",
        prompt_template_version="1.0",
        llm_model_id="gpt-4o",
        code_git_sha="deadbeef",
        input_hash="abc123",
    )
    return SupervisorDecision(
        decision_id="d1",
        sku_id="A",
        store_id="CA_1",
        tier=tier,
        stockout_probability=0.62,
        autonomous=False,
        recommended_actions=[
            ActionRecommendation(
                action_type=ActionType.SAFETY_STOCK_TRANSFER, rank=1, rationale="x", reversible=True
            )
        ],
        provenance=provenance,
    )


def _stockout() -> StockoutRiskResult:
    return StockoutRiskResult(
        sku_id="A",
        store_id="CA_1",
        stockout_probability=0.62,
        calibrated=True,
        confidence_tier=ConfidenceTier.MEDIUM,
        shap_values=[ShapValue(feature="lead_time", value=18.0, contribution=0.31)],
    )


def test_compose_brief_facts_includes_numbers_and_drivers() -> None:
    facts = compose_brief_facts(_decision(), _stockout())
    assert "62%" in facts
    assert "lead_time" in facts
    assert "REVIEW" in facts
    assert "safety_stock_transfer" in facts


def test_brief_writer_narrates_via_llm() -> None:
    llm = FakeLLM()
    brief = BriefWriter(llm).write(_decision(), _stockout())
    assert brief == "NARRATED BRIEF"
    # the LLM received the factual block as the user prompt (it only narrates).
    system, user = llm.calls[0]
    assert "never make your own prediction" in system.lower() or "never invent" in system.lower()
    assert "lead_time" in user


def test_supervisor_review_request_uses_llm_when_wired() -> None:
    supervisor = SupervisorAgent(_provenance(), brief_writer=BriefWriter(FakeLLM()))
    review = supervisor.build_review_request(_decision(), _stockout())
    assert review.brief == "NARRATED BRIEF"
    assert review.tier is EscalationTier.REVIEW
    assert review.recommended_actions[0].action_type is ActionType.SAFETY_STOCK_TRANSFER


def test_supervisor_review_request_falls_back_without_llm() -> None:
    supervisor = SupervisorAgent(_provenance())  # no brief writer
    review = supervisor.build_review_request(_decision(), _stockout())
    # deterministic SHAP facts become the brief — planner still gets the numbers.
    assert "lead_time" in review.brief
    assert "62%" in review.brief


def _provenance() -> ProvenanceContext:
    return ProvenanceContext(
        feature_schema_version="1.0",
        policy_config_version="default",
        prompt_template_version="1.0",
        llm_model_id="gpt-4o",
        code_git_sha="deadbeef",
    )
