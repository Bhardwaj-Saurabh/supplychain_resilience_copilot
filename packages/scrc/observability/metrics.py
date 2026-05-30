"""Prometheus metrics for infra/system telemetry (ADR-0005).

Agent/LLM tracing and evaluation live in Opik; these are the system metrics
Opik does not cover (escalation rates, decision latency). Helpers **degrade to
no-ops** when prometheus_client isn't installed.
"""

from __future__ import annotations

from typing import Any

_metrics: dict[str, Any] | None = None
_unavailable = False


def _registry() -> dict[str, Any] | None:
    global _metrics, _unavailable
    if _metrics is not None:
        return _metrics
    if _unavailable:
        return None
    try:
        from prometheus_client import Counter, Histogram
    except ImportError:
        _unavailable = True
        return None
    _metrics = {
        "decisions": Counter(
            "scrc_decisions_total", "Supervisor decisions", ["tier", "autonomous"]
        ),
        "downgrades": Counter(
            "scrc_audit_downgrades_total", "Autonomy downgrades from audit failure"
        ),
        "latency": Histogram("scrc_decision_seconds", "End-to-end decision latency (s)"),
    }
    return _metrics


def record_decision(tier: str, autonomous: bool) -> None:
    metrics = _registry()
    if metrics is None:
        return
    metrics["decisions"].labels(tier=tier, autonomous=str(autonomous).lower()).inc()


def record_audit_downgrade() -> None:
    metrics = _registry()
    if metrics is None:
        return
    metrics["downgrades"].inc()
