"""Composition root — wires the whole system into a runnable FastAPI app.

This is the top of the stack: the one place concrete tools, models, providers,
and the orchestrator are assembled. ``build_app`` is a uvicorn factory:

    uvicorn scrc.app.composition:build_app --factory --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI

from scrc.agents import (
    DemandAgent,
    LogisticsAgent,
    MacroAgent,
    ProvenanceContext,
    StockoutAgent,
    SupervisorAgent,
)
from scrc.api import create_app
from scrc.app.local_models import (
    NaiveQuantileForecaster,
    seed_provider,
    train_demo_anomaly,
    train_demo_stockout,
)
from scrc.app.settings import Settings
from scrc.orchestration import AgentBundle, build_graph
from scrc.tools import ForecastTool, LogisticsTool, MacroTool, StockoutTool


def _provenance(profile: str) -> ProvenanceContext:
    return ProvenanceContext(
        feature_schema_version="1.0",
        policy_config_version="default",
        prompt_template_version="1.0",
        llm_model_id="none" if profile == "demo" else "gpt-4o",
        code_git_sha="local",
        model_versions={"profile": profile},
    )


def build_demo_bundle() -> AgentBundle:
    """Wire the demo bundle: real classifier/anomaly models (trained on synthetic
    data), a local forecaster, the real regime classifier, no LLM (deterministic
    brief), in-memory audit/rollback."""
    provider = seed_provider()
    return AgentBundle(
        demand=DemandAgent(ForecastTool(NaiveQuantileForecaster(), provider)),
        logistics=LogisticsAgent(LogisticsTool(train_demo_anomaly(), provider)),
        macro=MacroAgent(MacroTool(provider)),
        stockout=StockoutAgent(StockoutTool(train_demo_stockout())),
        supervisor=SupervisorAgent(_provenance("demo")),
    )


def build_production_bundle() -> AgentBundle:
    """Production wiring is deployment-specific: a ChronosForecaster against the
    Foundry endpoint, MLflow-registry-loaded XGBoost/IsolationForest behind the
    tool ports, a Feast-backed FeatureProvider, an AzureOpenAIClient BriefWriter,
    and the MlflowAuditLog passed to build_graph. The adapters all exist
    (scrc.ml, scrc.llm, scrc.observability); assembling them requires the live
    services and credentials, so it is left to the deployment entrypoint."""
    raise NotImplementedError(
        "production composition requires MLflow/Feast/Chronos/Azure services; "
        "wire the existing adapters at the deployment entrypoint"
    )


def build_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    bundle = build_production_bundle() if settings.profile == "production" else build_demo_bundle()
    return create_app(build_graph(bundle))
