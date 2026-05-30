"""MLflow model registry helpers (ML lifecycle).

MLflow is the registry **and** the audit log (architecture.md §8, ADR-0007). The
governance invariant: **only registry-promoted versions are callable by agent
tools** — serving resolves the production *alias* at call time, so promotion and
rollback are an alias flip, not a redeploy.

``mlflow`` is imported lazily inside each function so ``scrc.ml`` imports without
MLflow installed (it is in the ``ml`` extra); unit tests don't need a server.
"""

from __future__ import annotations

from typing import Any


def log_and_register(
    model: Any,
    name: str,
    params: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
) -> str:
    """Log a fitted sklearn-compatible model as a new registered version.

    Returns the MLflow run id. Promotion to the production alias is a separate,
    governed step (see :func:`promote_to_production`).
    """
    import mlflow
    import mlflow.sklearn

    with mlflow.start_run() as run:
        if params:
            mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name=name, registered_model_name=name)
        return run.info.run_id


def promote_to_production(name: str, version: str, alias: str = "production") -> None:
    """Point the production alias at a specific registered version."""
    import mlflow

    client = mlflow.MlflowClient()
    client.set_registered_model_alias(name=name, alias=alias, version=version)


def load_production_model(name: str, alias: str = "production") -> Any:
    """Load the alias-resolved production model. Tools call only this path."""
    import mlflow.sklearn

    return mlflow.sklearn.load_model(f"models:/{name}@{alias}")
