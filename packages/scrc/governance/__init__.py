"""Governance layer (cross-cutting): escalation, HITL, audit, rollback.

Deterministic and contract-only by design (ADR-0001, ADR-0002). Depends on
``scrc.contracts`` and nothing else in ``scrc`` (enforced in .importlinter).
"""

from __future__ import annotations

from scrc.governance.escalation import (
    EscalationOutcome,
    EscalationSignals,
    evaluate_escalation,
)

__all__ = [
    "EscalationOutcome",
    "EscalationSignals",
    "evaluate_escalation",
]
