from __future__ import annotations

import pytest

from scrc.contracts import DecisionProvenance, EscalationTier, SupervisorDecision
from scrc.governance import AuditWriteError
from scrc.observability import MlflowAuditLog, decision_span, record_decision


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
        provenance=provenance,
    )


def test_metrics_and_tracing_degrade_to_noops() -> None:
    # These must be safe to call whether or not the observability stack is present.
    record_decision("monitor", autonomous=True)
    with decision_span("decision", tier="monitor", autonomous=True):
        pass


def test_mlflow_audit_wraps_backend_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import mlflow

    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("tracking server unreachable")

    monkeypatch.setattr(mlflow, "set_experiment", boom)
    with pytest.raises(AuditWriteError):
        MlflowAuditLog().log_decision(_decision())
