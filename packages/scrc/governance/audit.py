"""Decision audit trail and the no-audit-no-autonomy invariant (ADR-0002).

Every Supervisor decision is logged (architecture.md §7.3). The invariant: if a
decision cannot be durably audited, the system **must not act autonomously** —
it degrades to escalate-only. Autonomy is a privilege contingent on traceability.

The ``AuditLog`` port keeps governance free of any backend dependency; the
MLflow-backed adapter lives in ``scrc.observability``. ``InMemoryAuditLog`` is
the pure default used by tests and local runs.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Protocol

from pydantic import Field

from scrc.contracts import EscalationTier, SCRCModel, SupervisorDecision


class AuditWriteError(RuntimeError):
    """Raised by an ``AuditLog`` when a decision cannot be durably persisted."""


class AuditRecord(SCRCModel):
    """The persisted audit entry for one decision."""

    audit_id: str
    decision_id: str
    sku_id: str
    store_id: str
    tier: EscalationTier
    stockout_probability: float
    autonomous: bool
    input_hash: str
    logged_at: datetime
    human_outcome: dict[str, object] | None = None


class AuditLog(Protocol):
    """Port: durably record a decision. Raises ``AuditWriteError`` on failure."""

    def log_decision(
        self, decision: SupervisorDecision, human_outcome: dict[str, object] | None = None
    ) -> AuditRecord: ...


class InMemoryAuditLog:
    """In-process audit log (tests/local). Implements ``AuditLog``."""

    def __init__(self) -> None:
        self.records: list[AuditRecord] = []

    def log_decision(
        self, decision: SupervisorDecision, human_outcome: dict[str, object] | None = None
    ) -> AuditRecord:
        record = _to_record(decision, human_outcome)
        self.records.append(record)
        return record


def _to_record(
    decision: SupervisorDecision, human_outcome: dict[str, object] | None
) -> AuditRecord:
    return AuditRecord(
        audit_id=uuid.uuid4().hex,
        decision_id=decision.decision_id,
        sku_id=decision.sku_id,
        store_id=decision.store_id,
        tier=decision.tier,
        stockout_probability=decision.stockout_probability,
        autonomous=decision.autonomous,
        input_hash=decision.provenance.input_hash,
        logged_at=datetime.now(UTC),
        human_outcome=human_outcome,
    )


class AuditOutcome(SCRCModel):
    """Result of enforcing the audit gate on a decision."""

    decision: SupervisorDecision
    audit_id: str | None
    downgraded: bool = Field(default=False)


def enforce_audit(decision: SupervisorDecision, audit: AuditLog) -> AuditOutcome:
    """Apply the no-audit-no-autonomy invariant (ADR-0002).

    Logs the decision. If the write fails and the decision was autonomous, the
    decision is **downgraded to non-autonomous** so it is routed to a human
    rather than executed unlogged. Escalated decisions proceed to HITL whether
    or not the audit write succeeded (the review workflow logs them).
    """
    try:
        record = audit.log_decision(decision)
    except AuditWriteError:
        if decision.autonomous:
            return AuditOutcome(
                decision=decision.model_copy(update={"autonomous": False}),
                audit_id=None,
                downgraded=True,
            )
        return AuditOutcome(decision=decision, audit_id=None)
    return AuditOutcome(decision=decision, audit_id=record.audit_id)
