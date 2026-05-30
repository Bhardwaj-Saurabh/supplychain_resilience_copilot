"""Airflow DAG: ingest freight/port data -> engineer -> validate -> parquet.

BTS FAF is the primary logistics signal; MarineTraffic AIS is enrichment cached
with a 4h TTL (PRD §13). The concrete BTS/AIS clients are added later; this DAG
reads a staged raw file and applies the (unit-tested) feature engineering.
Requires the ``pipeline`` extra.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import pandas as pd
from airflow.decorators import dag, task

from scrc.data import (
    LogisticsFeatureRow,
    engineer_logistics_features,
    validate_rows,
)

RAW = os.environ.get("FREIGHT_RAW_PATH", "data/raw/freight.csv")
OUTPUT = "data/feast/logistics.parquet"


@dag(schedule="@daily", start_date=datetime(2024, 1, 1), catchup=False, tags=["ingest", "logistics"])
def ingest_logistics() -> None:
    @task
    def build() -> list[dict[str, Any]]:
        raw = pd.read_csv(RAW)
        return engineer_logistics_features(raw).to_dict(orient="records")

    @task
    def validate_and_write(records: list[dict[str, Any]]) -> None:
        frame = pd.DataFrame(records)
        report = validate_rows(frame, LogisticsFeatureRow)
        if not report.ok:
            raise ValueError(f"logistics feature validation failed: {report.errors[:5]}")
        frame.to_parquet(OUTPUT, index=False)

    validate_and_write(build())


ingest_logistics()
