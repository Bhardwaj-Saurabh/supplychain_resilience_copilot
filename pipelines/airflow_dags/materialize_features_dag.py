"""Airflow DAG: apply Feast definitions and materialise to the online store.

Runs after the three ingest DAGs have refreshed the parquet sources. Pushes
features into the PostgreSQL online store (ADR-0006) for low-latency agent tool
calls. Requires the ``pipeline`` extra (Feast + Airflow).
"""

from __future__ import annotations

from datetime import datetime

from airflow.decorators import dag, task

FEAST_REPO = "pipelines/feast"


@dag(schedule="@daily", start_date=datetime(2024, 1, 1), catchup=False, tags=["feast"])
def materialize_features() -> None:
    @task.bash
    def feast_apply() -> str:
        return f"feast -c {FEAST_REPO} apply"

    @task.bash
    def feast_materialize() -> str:
        # Incremental materialisation up to the logical execution date.
        return f"feast -c {FEAST_REPO} materialize-incremental {{{{ ds }}}}T00:00:00"

    feast_apply() >> feast_materialize()


materialize_features()
