# Pipelines (Layer 1 orchestration)

Thin orchestration wrappers around the **data layer** ([packages/scrc/data/](../packages/scrc/data/)).
All ingestion, feature-engineering, and validation logic lives in `scrc.data` and
is unit-tested there; these files only schedule and wire it. Requires the
`pipeline` extra (`uv pip install -e ".[pipeline]"` — Airflow + Feast).

## Airflow DAGs (`airflow_dags/`)

| DAG | Flow |
|---|---|
| `ingest_fred` | FRED series → `engineer_macro_features` → validate → `data/feast/macro.parquet` |
| `ingest_m5` | M5 tables → `engineer_demand_features` → validate → `data/feast/demand.parquet` |
| `ingest_logistics` | Freight/port raw → `engineer_logistics_features` → validate → `data/feast/logistics.parquet` |
| `materialize_features` | `feast apply` + `materialize-incremental` into the PostgreSQL online store |

Each ingest DAG runs the schema-validation gate (PRD §8.1) before writing; a
validation failure aborts the run.

## Feast (`feast/`)

- [`feature_store.yaml`](feast/feature_store.yaml) — PostgreSQL online store + SQL registry (ADR-0006); offline = parquet.
- [`features.py`](feast/features.py) — entities and feature views mirroring `scrc.data.schemas`, tagged with `FEATURE_SCHEMA_VERSION`.

Offline (training) reads from parquet; online (low-latency agent tool calls)
reads from PostgreSQL. AIS logistics features use a 4h TTL (PRD §13).
