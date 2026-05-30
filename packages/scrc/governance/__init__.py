"""Governance layer (cross-cutting): escalation, audit, rollback, breakers.

Deterministic and contract-only by design (ADR-0001, ADR-0002). Depends on
``scrc.contracts`` and nothing else in ``scrc`` (enforced in .importlinter).
"""

from __future__ import annotations

from scrc.governance.audit import (
    AuditLog,
    AuditOutcome,
    AuditRecord,
    AuditWriteError,
    InMemoryAuditLog,
    enforce_audit,
)
from scrc.governance.breaker import CircuitBreaker, CircuitOpenError, call_with_timeout
from scrc.governance.escalation import (
    EscalationOutcome,
    EscalationSignals,
    evaluate_escalation,
)
from scrc.governance.rollback import (
    DEFAULT_WINDOW,
    InMemoryRollbackRegistry,
    RollbackEntry,
    RollbackRegistry,
    register_reversible_actions,
)

__all__ = [
    "DEFAULT_WINDOW",
    "AuditLog",
    "AuditOutcome",
    "AuditRecord",
    "AuditWriteError",
    "CircuitBreaker",
    "CircuitOpenError",
    "EscalationOutcome",
    "EscalationSignals",
    "InMemoryAuditLog",
    "InMemoryRollbackRegistry",
    "RollbackEntry",
    "RollbackRegistry",
    "call_with_timeout",
    "enforce_audit",
    "evaluate_escalation",
    "register_reversible_actions",
]
