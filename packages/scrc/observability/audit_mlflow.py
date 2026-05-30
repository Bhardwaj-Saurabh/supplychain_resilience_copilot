"""MLflow-backed audit log (implements the governance ``AuditLog`` port).

MLflow is the registry *and* the audit trail (ADR-0007). Each decision is logged
as a run with its provenance, tier, and outcome (architecture.md §7.3). Any
backend failure is surfaced as ``AuditWriteError`` so the no-audit-no-autonomy
invariant (ADR-0002) can downgrade the decision rather than act unlogged.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from scrc.contracts import SupervisorDecision
from scrc.governance import AuditRecord, AuditWriteError


class MlflowAuditLog:
    """AuditLog adapter writing to an MLflow experiment."""

    def __init__(self, experiment: str = "scrc-decisions") -> None:
        self._experiment = experiment

    def log_decision(
        self, decision: SupervisorDecision, human_outcome: dict[str, object] | None = None
    ) -> AuditRecord:
        audit_id = uuid.uuid4().hex
        try:
            import mlflow

            mlflow.set_experiment(self._experiment)
            with mlflow.start_run(run_name=decision.decision_id):
                mlflow.log_params(
                    {
                        "decision_id": decision.decision_id,
                        "sku_id": decision.sku_id,
                        "store_id": decision.store_id,
                        "tier": decision.tier.value,
                        "autonomous": decision.autonomous,
                        "input_hash": decision.provenance.input_hash,
                        "feature_schema_version": decision.provenance.feature_schema_version,
                        "policy_config_version": decision.provenance.policy_config_version,
                        "llm_model_id": decision.provenance.llm_model_id,
                        **{
                            f"model_version.{k}": v
                            for k, v in decision.provenance.model_versions.items()
                        },
                    }
                )
                mlflow.log_metric("stockout_probability", decision.stockout_probability)
                if human_outcome is not None:
                    mlflow.set_tags({f"hitl.{k}": str(v) for k, v in human_outcome.items()})
        except Exception as exc:  # any backend failure -> no-audit-no-autonomy
            raise AuditWriteError(f"MLflow audit write failed: {exc}") from exc

        return AuditRecord(
            audit_id=audit_id,
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
