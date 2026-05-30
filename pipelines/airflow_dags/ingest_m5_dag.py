"""Airflow DAG: load M5 tables -> engineer demand features -> validate -> parquet.

Thin orchestration over ``scrc.data``. Requires the ``pipeline`` extra.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import pandas as pd
from airflow.decorators import dag, task

from scrc.data import (
    DemandFeatureRow,
    M5Loader,
    engineer_demand_features,
    validate_rows,
)

M5_DIR = os.environ.get("M5_DATA_DIR", "data/raw/m5")
OUTPUT = "data/feast/demand.parquet"


@dag(schedule="@daily", start_date=datetime(2024, 1, 1), catchup=False, tags=["ingest", "demand"])
def ingest_m5() -> None:
    @task
    def build() -> list[dict[str, Any]]:
        loader = M5Loader(M5_DIR)
        features = engineer_demand_features(
            loader.load_sales(), loader.load_calendar(), loader.load_prices()
        )
        return features.to_dict(orient="records")

    @task
    def validate_and_write(records: list[dict[str, Any]]) -> None:
        frame = pd.DataFrame(records)
        report = validate_rows(frame, DemandFeatureRow)
        if not report.ok:
            raise ValueError(f"demand feature validation failed: {report.errors[:5]}")
        frame.to_parquet(OUTPUT, index=False)

    validate_and_write(build())


ingest_m5()
