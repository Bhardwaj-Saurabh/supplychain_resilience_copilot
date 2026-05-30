"""Feast feature definitions for the SCRC offline/online stores.

Thin declaration layer: the feature *columns* mirror the schemas in
``scrc.data.schemas`` and are versioned with ``FEATURE_SCHEMA_VERSION`` (tagged
below) so a schema bump is visible in the registry and in decision provenance
(architecture.md §20). Offline sources are parquet files materialised by the
Airflow pipeline; the online store is PostgreSQL (ADR-0006).

Requires the ``pipeline`` extra (`feast`); not imported by unit tests.
"""

from __future__ import annotations

from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Bool, Float32

from scrc.data.schemas import FEATURE_SCHEMA_VERSION

_TAGS = {"feature_schema_version": FEATURE_SCHEMA_VERSION}
_DATA = "data/feast"

# --- Entities --------------------------------------------------------------
sku = Entity(name="sku", join_keys=["sku_id"])
store = Entity(name="store", join_keys=["store_id"])
port = Entity(name="port", join_keys=["port_id"])
market = Entity(name="market", join_keys=["market_id"])  # singleton for global macro

# --- Sources ---------------------------------------------------------------
demand_source = FileSource(path=f"{_DATA}/demand.parquet", timestamp_field="event_timestamp")
macro_source = FileSource(path=f"{_DATA}/macro.parquet", timestamp_field="event_timestamp")
logistics_source = FileSource(path=f"{_DATA}/logistics.parquet", timestamp_field="event_timestamp")

# --- Feature views ---------------------------------------------------------
demand_features = FeatureView(
    name="demand_features",
    entities=[sku, store],
    ttl=timedelta(days=7),
    source=demand_source,
    tags=_TAGS,
    schema=[
        Field(name="sell_price", dtype=Float32),
        Field(name="lag_7", dtype=Float32),
        Field(name="lag_28", dtype=Float32),
        Field(name="rolling_mean_28", dtype=Float32),
        Field(name="on_promo", dtype=Bool),
        Field(name="snap_flag", dtype=Bool),
        Field(name="event_flag", dtype=Bool),
    ],
)

macro_features = FeatureView(
    name="macro_features",
    entities=[market],
    ttl=timedelta(days=40),
    source=macro_source,
    tags=_TAGS,
    schema=[
        Field(name="t10y2y", dtype=Float32),
        Field(name="cpi_yoy", dtype=Float32),
        Field(name="ism_pmi", dtype=Float32),
        Field(name="fuel_cpi", dtype=Float32),
        Field(name="jobless_claims", dtype=Float32),
        Field(name="consumer_sentiment", dtype=Float32),
    ],
)

logistics_features = FeatureView(
    name="logistics_features",
    entities=[port],
    # AIS enrichment is cached with a 4h TTL (PRD §13); BTS FAF is primary.
    ttl=timedelta(hours=4),
    source=logistics_source,
    tags=_TAGS,
    schema=[
        Field(name="congestion_index", dtype=Float32),
        Field(name="dwell_hours", dtype=Float32),
        Field(name="freight_tonnage", dtype=Float32),
        Field(name="rolling_zscore", dtype=Float32),
    ],
)
