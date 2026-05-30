"""Airflow DAG: ingest FRED macro series -> engineer -> validate -> parquet.

Thin orchestration over ``scrc.data`` (the logic lives there and is unit-tested
independently). Requires the ``pipeline`` extra (Airflow); not imported by tests.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import pandas as pd
from airflow.decorators import dag, task

from scrc.data import (
    FredClient,
    MacroFeatureRow,
    engineer_macro_features,
    validate_rows,
)
from scrc.data.transforms import FRED_SERIES_MAP

OUTPUT = "data/feast/macro.parquet"


@dag(schedule="@daily", start_date=datetime(2024, 1, 1), catchup=False, tags=["ingest", "macro"])
def ingest_fred() -> None:
    @task
    def fetch() -> list[dict[str, Any]]:
        client = FredClient(api_key=os.environ["FRED_API_KEY"])
        rows: list[dict[str, Any]] = []
        for series_id in FRED_SERIES_MAP:
            rows.extend(obs.model_dump() for obs in client.fetch_series(series_id))
        return rows

    @task
    def transform(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return engineer_macro_features(pd.DataFrame(raw)).to_dict(orient="records")

    @task
    def validate_and_write(records: list[dict[str, Any]]) -> None:
        frame = pd.DataFrame(records)
        report = validate_rows(frame, MacroFeatureRow)
        if not report.ok:
            raise ValueError(f"macro feature validation failed: {report.errors[:5]}")
        frame.to_parquet(OUTPUT, index=False)

    validate_and_write(transform(fetch()))


ingest_fred()
