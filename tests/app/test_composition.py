"""End-to-end smoke test of the demo composition root.

Boots the full stack in-process (real XGBoost + Isolation Forest trained on
synthetic data, local forecaster, real regime classifier, LangGraph + audit
gate, FastAPI) and serves a genuine ML-driven decision.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from scrc.app import Settings, build_app
from scrc.app.local_models import DEMO_PORT, DEMO_SKU, DEMO_STORE


def test_demo_app_serves_a_real_decision() -> None:
    client = TestClient(build_app())
    assert client.get("/health").json() == {"status": "ok"}

    resp = client.post(
        "/decisions",
        json={"sku_id": DEMO_SKU, "store_id": DEMO_STORE, "port_ids": [DEMO_PORT]},
    )
    assert resp.status_code in (200, 202)
    body = resp.json()
    assert body["thread_id"]
    if resp.status_code == 200:
        assert body["status"] == "completed"
        assert body["decision"]["sku_id"] == DEMO_SKU
        # provenance proves the real pipeline ran end-to-end.
        assert body["decision"]["provenance"]["input_hash"]
    else:
        assert body["status"] == "review_required"
        assert body["review"]["brief"]


def test_production_profile_requires_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    from scrc.app.production import ProductionConfigError

    # Unconfigured environment -> production composition fails loudly, not silently.
    for name in ("FEAST_REPO_PATH", "SCRC_MODEL_DIR", "CHRONOS_ENDPOINT", "AZURE_OPENAI_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ProductionConfigError):
        build_app(Settings(profile="production"))
