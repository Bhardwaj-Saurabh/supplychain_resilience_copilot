from __future__ import annotations

from scrc.contracts import (
    ActionRecommendation,
    ActionType,
    DecisionProvenance,
    EscalationTier,
    SupervisorDecision,
)
from scrc.governance import InMemoryRollbackRegistry, register_reversible_actions


def _decision() -> SupervisorDecision:
    provenance = DecisionProvenance(
        feature_schema_version="1.0",
        policy_config_version="default",
        prompt_template_version="1.0",
        llm_model_id="gpt-4o",
        code_git_sha="deadbeef",
        input_hash="hash123",
    )
    return SupervisorDecision(
        decision_id="d1",
        sku_id="A",
        store_id="CA_1",
        tier=EscalationTier.MONITOR,
        stockout_probability=0.4,
        autonomous=True,
        recommended_actions=[
            ActionRecommendation(
                action_type=ActionType.REROUTE, rank=1, rationale="x", reversible=True
            ),
            ActionRecommendation(
                action_type=ActionType.SUBSTITUTE, rank=2, rationale="y", reversible=False
            ),
        ],
        provenance=provenance,
    )


def test_only_reversible_actions_are_registered() -> None:
    registry = InMemoryRollbackRegistry()
    entries = register_reversible_actions(_decision(), registry)
    assert len(entries) == 1
    assert entries[0].action_type is ActionType.REROUTE
    assert registry.pending()[0].decision_id == "d1"
    assert entries[0].window_expires > entries[0].registered_at


def test_mark_rolled_back_clears_from_pending() -> None:
    registry = InMemoryRollbackRegistry()
    entry = register_reversible_actions(_decision(), registry)[0]
    registry.mark_rolled_back(entry.entry_id)
    assert registry.pending() == []
