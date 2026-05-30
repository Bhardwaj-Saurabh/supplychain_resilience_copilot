from __future__ import annotations

import pytest
from pydantic import ValidationError

from scrc.contracts import (
    ActionRecommendation,
    ActionType,
    DecisionProvenance,
    EscalationTier,
    SupervisorDecision,
)


def _provenance() -> DecisionProvenance:
    return DecisionProvenance(
        model_versions={"chronos": "3", "xgboost": "7", "isoforest": "2"},
        feature_schema_version="1.0",
        policy_config_version="default",
        prompt_template_version="1.0",
        llm_model_id="gpt-4o",
        code_git_sha="deadbeef",
        input_hash="abc123",
    )


def test_supervisor_decision_json_roundtrip() -> None:
    decision = SupervisorDecision(
        decision_id="d1",
        sku_id="A",
        store_id="CA_1",
        tier=EscalationTier.REVIEW,
        stockout_probability=0.62,
        autonomous=False,
        recommended_actions=[
            ActionRecommendation(
                action_type=ActionType.EXPEDITE,
                rank=1,
                rationale="lead time spike",
                reversible=True,
            )
        ],
        provenance=_provenance(),
    )
    restored = SupervisorDecision.model_validate_json(decision.model_dump_json())
    assert restored.tier is EscalationTier.REVIEW
    assert restored.recommended_actions[0].action_type is ActionType.EXPEDITE
    assert restored.provenance.input_hash == "abc123"


def test_action_rank_must_be_at_least_one() -> None:
    with pytest.raises(ValidationError):
        ActionRecommendation(
            action_type=ActionType.REROUTE, rank=0, rationale="x", reversible=False
        )


def test_tier_serialises_to_canonical_value() -> None:
    assert EscalationTier.CRITICAL.value == "critical"
    assert ActionType.SAFETY_STOCK_TRANSFER.value == "safety_stock_transfer"
