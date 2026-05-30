"""Observability (cross-cutting): OTEL tracing, Prometheus metrics, MLflow audit.

Emission/infra adapters that depend on ``scrc.contracts`` and ``scrc.governance``
only. Tracing and metrics degrade to no-ops without their optional deps, so the
core paths never require the observability stack to be installed.
"""

from __future__ import annotations

from scrc.observability.audit_mlflow import MlflowAuditLog
from scrc.observability.metrics import record_audit_downgrade, record_decision
from scrc.observability.tracing import decision_span

__all__ = [
    "MlflowAuditLog",
    "decision_span",
    "record_audit_downgrade",
    "record_decision",
]
