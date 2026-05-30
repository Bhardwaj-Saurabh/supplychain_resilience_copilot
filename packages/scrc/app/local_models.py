"""Local, service-free models and seed data for the demo profile.

These let the full stack boot and serve genuine ML-driven decisions without an
MLflow registry, Chronos endpoint, or Feast store. The classifier and anomaly
detector are the *real* ``scrc.ml`` models trained on synthetic data at startup;
only the forecaster is a local stand-in for the Chronos endpoint.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import fmean, pstdev

import numpy as np
import pandas as pd

from scrc.contracts import QuantileForecastResult
from scrc.ml import FreightAnomalyDetector, StockoutClassifier
from scrc.tools import StaticFeatureProvider

DEMO_SKU = "DEMO_SKU"
DEMO_STORE = "CA_1"
DEMO_PORT = "USLAX"

# The joint feature vector the StockoutAgent assembles (must match the model).
STOCKOUT_FEATURES = [
    "demand_p50",
    "demand_interval_width",
    "congestion_score",
    "anomaly_score",
    "anomaly_flag",
    "macro_shock",
    "macro_confidence",
]
LOGISTICS_FEATURES = ["congestion_index", "dwell_hours", "rolling_zscore"]


class NaiveQuantileForecaster:
    """Local Forecaster stand-in: total-over-horizon quantiles from the context
    series mean/spread. Production uses ``scrc.ml.ChronosForecaster``."""

    def forecast(
        self,
        sku_id: str,
        store_id: str,
        context: Sequence[float],
        horizon_days: int,
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> QuantileForecastResult:
        base = fmean(context) if context else 100.0
        sd = pstdev(context) if len(context) > 1 else base * 0.1
        total = base * horizon_days
        half_width = 1.2816 * sd * math.sqrt(horizon_days)
        p10 = max(total - half_width, 0.0)
        p90 = total + half_width
        return QuantileForecastResult(
            sku_id=sku_id,
            store_id=store_id,
            horizon_days=horizon_days,
            p10=p10,
            p50=total,
            p90=p90,
            interval_width=p90 - p10,
        )


def train_demo_stockout(seed: int = 0) -> StockoutClassifier:
    rng = np.random.default_rng(seed)
    n = 600
    df = pd.DataFrame(
        {
            "demand_p50": rng.normal(100, 20, n),
            "demand_interval_width": np.abs(rng.normal(30, 15, n)),
            "congestion_score": rng.uniform(0, 1, n),
            "anomaly_score": rng.normal(0.3, 0.3, n),
            "anomaly_flag": rng.integers(0, 2, n).astype(float),
            "macro_shock": (rng.uniform(0, 1, n) < 0.15).astype(float),
            "macro_confidence": rng.uniform(0.4, 0.9, n),
        }
    )
    logit = (
        (df["demand_interval_width"] - 30) / 20
        + (df["congestion_score"] - 0.5) * 1.5
        + df["anomaly_flag"] * 0.8
        + df["macro_shock"] * 0.8
        + rng.normal(0, 1, n)
    )
    y = (logit > 0.5).astype(int)
    classifier = StockoutClassifier(STOCKOUT_FEATURES)
    classifier.train(df[STOCKOUT_FEATURES], pd.Series(y))
    return classifier


def train_demo_anomaly(seed: int = 0) -> FreightAnomalyDetector:
    rng = np.random.default_rng(seed)
    n = 300
    df = pd.DataFrame(
        {
            "congestion_index": rng.normal(0.3, 0.08, n).clip(0, 1),
            "dwell_hours": rng.normal(12, 2, n),
            "rolling_zscore": rng.normal(0, 0.6, n),
        }
    )
    detector = FreightAnomalyDetector(LOGISTICS_FEATURES, n_background=40, n_estimators=120)
    detector.fit(df)
    return detector


def seed_provider() -> StaticFeatureProvider:
    """Seed online features for the demo SKU / store / port."""
    return StaticFeatureProvider(
        demand_contexts={(DEMO_SKU, DEMO_STORE): [100.0 + (i % 7) for i in range(30)]},
        logistics={
            DEMO_PORT: {"congestion_index": 0.35, "dwell_hours": 12.0, "rolling_zscore": 0.4}
        },
        macro={"t10y2y": 0.4, "ism_pmi": 51.0, "consumer_sentiment": 78.0},
    )
