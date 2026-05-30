from __future__ import annotations

from scrc.contracts import (
    DecisionProvenance,
    EscalationTier,
    SupervisorDecision,
)
from scrc.governance import (
    AuditWriteError,
    InMemoryAuditLog,
    enforce_audit,
)


def _decision(
    autonomous: bool, tier: EscalationTier = EscalationTier.MONITOR
) -> SupervisorDecision:
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
        tier=tier,
        stockout_probability=0.4,
        autonomous=autonomous,
        provenance=provenance,
    )


class FailingAuditLog:
    def log_decision(self, decision: object, human_outcome: object = None) -> object:
        raise AuditWriteError("backend unavailable")


def test_in_memory_audit_records_decision() -> None:
    log = InMemoryAuditLog()
    record = log.log_decision(_decision(autonomous=True))
    assert record.decision_id == "d1"
    assert record.input_hash == "hash123"
    assert len(log.records) == 1


def test_successful_audit_keeps_autonomy() -> None:
    outcome = enforce_audit(_decision(autonomous=True), InMemoryAuditLog())
    assert outcome.audit_id is not None
    assert outcome.downgraded is False
    assert outcome.decision.autonomous is True


def test_failed_audit_downgrades_autonomous_decision() -> None:
    # No-audit-no-autonomy (ADR-0002): unauditable autonomous decision -> HITL.
    outcome = enforce_audit(_decision(autonomous=True), FailingAuditLog())
    assert outcome.downgraded is True
    assert outcome.decision.autonomous is False
    assert outcome.audit_id is None


def test_failed_audit_leaves_escalated_decision_untouched() -> None:
    outcome = enforce_audit(
        _decision(autonomous=False, tier=EscalationTier.REVIEW), FailingAuditLog()
    )
    assert outcome.downgraded is False
    assert outcome.decision.autonomous is False
    assert outcome.audit_id is None
