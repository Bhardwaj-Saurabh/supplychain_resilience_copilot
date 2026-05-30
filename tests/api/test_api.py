from __future__ import annotations

from collections.abc import Mapping

from fastapi.testclient import TestClient

from scrc.agents import (
    DemandAgent,
    LogisticsAgent,
    MacroAgent,
    ProvenanceContext,
    StockoutAgent,
    SupervisorAgent,
)
from scrc.api import create_app
from scrc.contracts import (
    AnomalyResult,
    ConfidenceTier,
    MacroSignals,
    QuantileForecastResult,
    RegimeLabel,
    StockoutRiskResult,
)
from scrc.orchestration import AgentBundle, build_graph


class _Forecast:
    def chronos_forecast(self, sku_id, store_id, horizon_days, covariates=None):  # type: ignore[no-untyped-def]
        return QuantileForecastResult(
            sku_id=sku_id, store_id=store_id, horizon_days=horizon_days, p10=97, p50=100, p90=103
        )


class _Logistics:
    def __init__(self, anomaly: bool) -> None:
        self._a = anomaly

    def detect_freight_anomaly(self, port_id: str) -> AnomalyResult:
        return AnomalyResult(
            port_ids=[port_id],
            congestion_score=0.9 if self._a else 0.2,
            anomaly_flag=self._a,
            anomaly_score=0.9 if self._a else 0.1,
        )


class _Macro:
    def __init__(self, regime: RegimeLabel) -> None:
        self._r = regime

    def assess_macro(self) -> MacroSignals:
        return MacroSignals(series_values={}, regime_label=self._r, regime_confidence=0.6)


class _Stockout:
    def __init__(self, p: float) -> None:
        self._p = p

    def classify_stockout_risk(
        self, sku_id: str, store_id: str, features: Mapping[str, float]
    ) -> StockoutRiskResult:
        return StockoutRiskResult(
            sku_id=sku_id,
            store_id=store_id,
            stockout_probability=self._p,
            calibrated=True,
            confidence_tier=ConfidenceTier.MEDIUM,
        )


def _client(prob: float, anomaly: bool, regime: RegimeLabel) -> TestClient:
    bundle = AgentBundle(
        demand=DemandAgent(_Forecast()),  # type: ignore[arg-type]
        logistics=LogisticsAgent(_Logistics(anomaly)),  # type: ignore[arg-type]
        macro=MacroAgent(_Macro(regime)),  # type: ignore[arg-type]
        stockout=StockoutAgent(_Stockout(prob)),  # type: ignore[arg-type]
        supervisor=SupervisorAgent(
            ProvenanceContext(
                feature_schema_version="1.0",
                policy_config_version="default",
                prompt_template_version="1.0",
                llm_model_id="gpt-4o",
                code_git_sha="deadbeef",
            )
        ),
    )
    return TestClient(create_app(build_graph(bundle)))


_BODY = {"sku_id": "A", "store_id": "CA_1", "port_ids": ["USLAX"]}


def test_health() -> None:
    assert _client(0.1, False, RegimeLabel.NEUTRAL).get("/health").json() == {"status": "ok"}


def test_autonomous_decision_completes() -> None:
    resp = _client(0.1, anomaly=False, regime=RegimeLabel.NEUTRAL).post("/decisions", json=_BODY)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["decision"]["tier"] == "routine"


def test_escalated_decision_requires_review_then_resumes() -> None:
    client = _client(0.9, anomaly=True, regime=RegimeLabel.SHOCK)
    resp = client.post("/decisions", json=_BODY)
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "review_required"
    assert body["review"]["tier"] == "critical"
    assert body["review"]["brief"]

    thread_id = body["thread_id"]
    resume = client.post(
        f"/decisions/{thread_id}/resume",
        json={"approved": True, "reviewer_id": "planner1"},
    )
    assert resume.status_code == 200
    resolved = resume.json()
    assert resolved["status"] == "resolved"
    assert resolved["human_outcome"]["approved"] is True
    assert resolved["decision"]["autonomous"] is False


def test_invalid_request_is_rejected() -> None:
    resp = _client(0.1, False, RegimeLabel.NEUTRAL).post("/decisions", json={"sku_id": "A"})
    assert resp.status_code == 422  # missing store_id
