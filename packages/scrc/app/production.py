"""Production composition: real adapters wired from environment config.

``assemble_bundle`` is pure wiring (testable with fakes). ``load_production_
dependencies`` constructs the real adapters — Feast online store, MLflow-
materialised models, the Chronos endpoint, Azure OpenAI brief writer, MLflow
audit — and so requires those services; it raises ``ProductionConfigError`` when
the environment is not configured.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from scrc.agents import (
    DemandAgent,
    LogisticsAgent,
    MacroAgent,
    ProvenanceContext,
    StockoutAgent,
    SupervisorAgent,
)
from scrc.app.feast_provider import FeastFeatureProvider
from scrc.app.model_store import load_anomaly_model, load_stockout_model
from scrc.governance import AuditLog
from scrc.llm import AzureDeployment, AzureOpenAIClient, BriefWriter
from scrc.ml import ChronosForecaster
from scrc.observability import MlflowAuditLog
from scrc.orchestration import AgentBundle, build_graph
from scrc.tools import (
    AnomalyModel,
    FeatureProvider,
    Forecaster,
    ForecastTool,
    LogisticsTool,
    MacroTool,
    StockoutModel,
    StockoutTool,
)

_REQUIRED_ENV = [
    "FEAST_REPO_PATH",
    "SCRC_MODEL_DIR",
    "CHRONOS_ENDPOINT",
    "CHRONOS_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_DEPLOYMENT_PRIMARY",
]


class ProductionConfigError(RuntimeError):
    """Raised when the production profile is selected but not fully configured."""


@dataclass(frozen=True)
class ProductionConfig:
    feast_repo: str
    model_dir: str
    chronos_endpoint: str
    chronos_api_key: str
    azure_endpoint: str
    azure_api_key: str
    azure_deployment_primary: str
    azure_deployment_secondary: str | None
    azure_api_version: str
    code_git_sha: str

    @classmethod
    def from_env(cls) -> ProductionConfig:
        missing = [name for name in _REQUIRED_ENV if not os.environ.get(name)]
        if missing:
            raise ProductionConfigError(f"missing required env: {', '.join(missing)}")
        return cls(
            feast_repo=os.environ["FEAST_REPO_PATH"],
            model_dir=os.environ["SCRC_MODEL_DIR"],
            chronos_endpoint=os.environ["CHRONOS_ENDPOINT"],
            chronos_api_key=os.environ["CHRONOS_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_deployment_primary=os.environ["AZURE_OPENAI_DEPLOYMENT_PRIMARY"],
            azure_deployment_secondary=os.environ.get("AZURE_OPENAI_DEPLOYMENT_SECONDARY"),
            azure_api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            code_git_sha=os.environ.get("SCRC_GIT_SHA", "production"),
        )


@dataclass(frozen=True)
class ProductionDependencies:
    provider: FeatureProvider
    forecaster: Forecaster
    stockout_model: StockoutModel
    anomaly_model: AnomalyModel
    audit: AuditLog
    brief_writer: BriefWriter | None = None
    code_git_sha: str = "production"


def assemble_bundle(deps: ProductionDependencies) -> AgentBundle:
    """Pure wiring of production dependencies into the agent bundle."""
    provenance = ProvenanceContext(
        feature_schema_version="1.0",
        policy_config_version="default",
        prompt_template_version="1.0",
        llm_model_id="gpt-4o",
        code_git_sha=deps.code_git_sha,
    )
    return AgentBundle(
        demand=DemandAgent(ForecastTool(deps.forecaster, deps.provider)),
        logistics=LogisticsAgent(LogisticsTool(deps.anomaly_model, deps.provider)),
        macro=MacroAgent(MacroTool(deps.provider)),
        stockout=StockoutAgent(StockoutTool(deps.stockout_model), features=deps.provider),
        supervisor=SupervisorAgent(provenance, brief_writer=deps.brief_writer),
    )


def load_production_dependencies(config: ProductionConfig) -> ProductionDependencies:
    secondary = (
        AzureDeployment(
            endpoint=config.azure_endpoint,
            api_key=config.azure_api_key,
            deployment=config.azure_deployment_secondary,
            api_version=config.azure_api_version,
        )
        if config.azure_deployment_secondary
        else None
    )
    brief_writer = BriefWriter(
        AzureOpenAIClient(
            AzureDeployment(
                endpoint=config.azure_endpoint,
                api_key=config.azure_api_key,
                deployment=config.azure_deployment_primary,
                api_version=config.azure_api_version,
            ),
            secondary,
        )
    )
    return ProductionDependencies(
        provider=FeastFeatureProvider.from_repo(config.feast_repo),
        forecaster=ChronosForecaster(config.chronos_endpoint, config.chronos_api_key),
        stockout_model=load_stockout_model(config.model_dir),
        anomaly_model=load_anomaly_model(config.model_dir),
        audit=MlflowAuditLog(),
        brief_writer=brief_writer,
        code_git_sha=config.code_git_sha,
    )


def build_production_graph(config: ProductionConfig, checkpointer: Any = None) -> Any:
    """Assemble and compile the production decision graph.

    Pass a durable ``PostgresSaver`` for HITL persistence (ADR-0006). Its
    connection is a context manager whose lifetime must match the running
    process, so the deployment entrypoint owns it, e.g.::

        with PostgresSaver.from_conn_string(dsn) as saver:
            saver.setup()
            app = create_app(build_production_graph(config, checkpointer=saver))
            uvicorn.run(app, ...)

    Without one, build_graph falls back to an in-process MemorySaver.
    """
    deps = load_production_dependencies(config)
    bundle = assemble_bundle(deps)
    return build_graph(bundle, checkpointer=checkpointer, audit=deps.audit)
