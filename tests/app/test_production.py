from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from scrc.app.feast_provider import FeastFeatureProvider
from scrc.app.local_models import STOCKOUT_FEATURES, train_demo_stockout
from scrc.app.model_store import STOCKOUT_FILENAME, load_stockout_model, save_model
from scrc.app.production import (
    ProductionConfig,
    ProductionConfigError,
    ProductionDependencies,
    assemble_bundle,
)
from scrc.contracts import (
    AnomalyResult,
    ConfidenceTier,
    DecisionRequest,
    QuantileForecastResult,
    StockoutRiskResult,
)
from scrc.governance import InMemoryAuditLog
from scrc.orchestration import build_graph
from scrc.tools import StaticFeatureProvider

# --- Feast provider --------------------------------------------------------


class _FakeResp:
    def __init__(self, data: dict[str, list[Any]]) -> None:
        self._data = data

    def to_dict(self) -> dict[str, list[Any]]:
        return self._data


class _FakeStore:
    def __init__(self, values: dict[str, Any]) -> None:
        self._values = values

    def get_online_features(
        self, features: list[str], entity_rows: list[dict[str, Any]]
    ) -> _FakeResp:
        out: dict[str, list[Any]] = {
            ref.split(":")[-1]: [self._values.get(ref.split(":")[-1])] for ref in features
        }
        for key, value in entity_rows[0].items():
            out[key] = [value]
        return _FakeResp(out)


def test_feast_provider_maps_online_features() -> None:
    store = _FakeStore(
        {
            "recent_unit_sales": [1.0, 2.0, 3.0, 4.0],
            "congestion_index": 0.5,
            "dwell_hours": 10.0,
            "rolling_zscore": 0.1,
            "t10y2y": 0.4,
        }
    )
    provider = FeastFeatureProvider(store)
    assert provider.demand_context("A", "CA_1", 2) == [3.0, 4.0]
    assert provider.logistics_features("USLAX") == {
        "congestion_index": 0.5,
        "dwell_hours": 10.0,
        "rolling_zscore": 0.1,
    }
    assert provider.macro_latest(["t10y2y"]) == {"t10y2y": 0.4}


# --- model store -----------------------------------------------------------


def test_model_store_round_trip(tmp_path: Path) -> None:
    save_model(train_demo_stockout(), tmp_path / STOCKOUT_FILENAME)
    loaded = load_stockout_model(tmp_path)
    result = loaded.predict("A", "CA_1", dict.fromkeys(STOCKOUT_FEATURES, 1.0))
    assert 0.0 <= result.stockout_probability <= 1.0


def test_load_model_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_stockout_model(tmp_path)


# --- production wiring (with injected fakes) -------------------------------


class _Forecaster:
    def forecast(
        self,
        sku_id: str,
        store_id: str,
        context: Sequence[float],
        horizon_days: int,
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> QuantileForecastResult:
        return QuantileForecastResult(
            sku_id=sku_id, store_id=store_id, horizon_days=horizon_days, p10=97, p50=100, p90=103
        )


class _AnomalyModel:
    def predict(
        self, port_ids: Sequence[str], features: Mapping[str, float], congestion_score: float
    ) -> AnomalyResult:
        return AnomalyResult(
            port_ids=list(port_ids),
            congestion_score=congestion_score,
            anomaly_flag=False,
            anomaly_score=0.1,
        )


class _StockoutModel:
    def predict(
        self, sku_id: str, store_id: str, features: Mapping[str, float]
    ) -> StockoutRiskResult:
        return StockoutRiskResult(
            sku_id=sku_id,
            store_id=store_id,
            stockout_probability=0.1,
            calibrated=True,
            confidence_tier=ConfidenceTier.LOW,
        )


def test_assemble_bundle_produces_a_working_graph() -> None:
    provider = StaticFeatureProvider(
        demand_contexts={("A", "CA_1"): [100.0, 101.0]},
        logistics={"USLAX": {"congestion_index": 0.2}},
        macro={"t10y2y": 0.4, "ism_pmi": 51.0},
    )
    deps = ProductionDependencies(
        provider=provider,
        forecaster=_Forecaster(),
        stockout_model=_StockoutModel(),
        anomaly_model=_AnomalyModel(),
        audit=InMemoryAuditLog(),
    )
    app = build_graph(assemble_bundle(deps), audit=deps.audit)
    result = app.invoke(
        {"request": DecisionRequest(sku_id="A", store_id="CA_1", port_ids=["USLAX"]), "errors": []},
        {"configurable": {"thread_id": "prod-wiring"}},
    )
    assert result["decision"].sku_id == "A"
    assert result["audit_id"] is not None


# --- config -----------------------------------------------------------------


def test_production_config_requires_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("FEAST_REPO_PATH", "SCRC_MODEL_DIR", "CHRONOS_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ProductionConfigError):
        ProductionConfig.from_env()
