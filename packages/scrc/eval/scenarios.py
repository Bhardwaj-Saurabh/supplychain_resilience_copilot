"""Scenario bundles for disruption replay.

Each scenario configures *real* tools (with scripted models satisfying the tool
ports) and *real* agents, then runs them through ``run_pipeline``. The macro
regime comes from the real ``MacroTool`` classifier reading scripted FRED values,
so the replay exercises the genuine escalation path — only the model outputs are
scripted to reproduce the historical signal pattern.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from scrc.agents import (
    DemandAgent,
    LogisticsAgent,
    MacroAgent,
    ProvenanceContext,
    StockoutAgent,
    SupervisorAgent,
)
from scrc.contracts import (
    AnomalyResult,
    ConfidenceTier,
    QuantileForecastResult,
    StockoutRiskResult,
    SupervisorDecision,
)
from scrc.eval.datasets import DisruptionScenario, EvalCase
from scrc.orchestration import AgentBundle, run_pipeline
from scrc.tools import ForecastTool, LogisticsTool, MacroTool, StaticFeatureProvider, StockoutTool

_PORT = "USLAX"
_PROVENANCE = ProvenanceContext(
    feature_schema_version="1.0",
    policy_config_version="default",
    prompt_template_version="1.0",
    llm_model_id="gpt-4o",
    code_git_sha="eval",
)


@dataclass(frozen=True)
class _ScenarioConfig:
    quantiles: tuple[float, float, float]  # p10, p50, p90
    anomaly_flag: bool
    anomaly_score: float
    congestion: float
    macro_values: dict[str, float]
    stockout_probability: float


# Macro values chosen so the real regime classifier returns the intended regime:
# consumer_sentiment < 55 -> SHOCK; otherwise neutral here.
_SHOCK_MACRO = {"consumer_sentiment": 40.0}
_NEUTRAL_MACRO = {"t10y2y": 0.5, "ism_pmi": 50.0}

_CONFIG: dict[DisruptionScenario, _ScenarioConfig] = {
    DisruptionScenario.BASELINE: _ScenarioConfig(
        (98.0, 100.0, 104.0), False, 0.1, 0.2, _NEUTRAL_MACRO, 0.10
    ),
    DisruptionScenario.COVID: _ScenarioConfig(
        (80.0, 120.0, 180.0), True, 0.95, 0.9, _SHOCK_MACRO, 0.90
    ),
    DisruptionScenario.SUEZ_2021: _ScenarioConfig(
        (90.0, 100.0, 115.0), True, 0.85, 0.9, _NEUTRAL_MACRO, 0.60
    ),
    DisruptionScenario.RED_SEA_2024: _ScenarioConfig(
        (60.0, 100.0, 140.0), True, 0.80, 0.9, _SHOCK_MACRO, 0.50
    ),
}


@dataclass(frozen=True)
class _Forecaster:
    quantiles: tuple[float, float, float]

    def forecast(
        self,
        sku_id: str,
        store_id: str,
        context: Sequence[float],
        horizon_days: int,
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> QuantileForecastResult:
        p10, p50, p90 = self.quantiles
        return QuantileForecastResult(
            sku_id=sku_id,
            store_id=store_id,
            horizon_days=horizon_days,
            p10=p10,
            p50=p50,
            p90=p90,
            interval_width=p90 - p10,
        )


@dataclass(frozen=True)
class _AnomalyModel:
    flag: bool
    score: float

    def predict(
        self, port_ids: Sequence[str], features: Mapping[str, float], congestion_score: float
    ) -> AnomalyResult:
        return AnomalyResult(
            port_ids=list(port_ids),
            congestion_score=congestion_score,
            anomaly_flag=self.flag,
            anomaly_score=self.score,
        )


@dataclass(frozen=True)
class _StockoutModel:
    probability: float

    def predict(
        self, sku_id: str, store_id: str, features: Mapping[str, float]
    ) -> StockoutRiskResult:
        return StockoutRiskResult(
            sku_id=sku_id,
            store_id=store_id,
            stockout_probability=self.probability,
            calibrated=True,
            confidence_tier=ConfidenceTier.MEDIUM,
        )


def build_scenario_bundle(scenario: DisruptionScenario) -> AgentBundle:
    cfg = _CONFIG[scenario]
    features = StaticFeatureProvider(
        logistics={_PORT: {"congestion_index": cfg.congestion}},
        macro=cfg.macro_values,
    )
    return AgentBundle(
        demand=DemandAgent(ForecastTool(_Forecaster(cfg.quantiles), features)),
        logistics=LogisticsAgent(
            LogisticsTool(_AnomalyModel(cfg.anomaly_flag, cfg.anomaly_score), features)
        ),
        macro=MacroAgent(MacroTool(features)),
        stockout=StockoutAgent(StockoutTool(_StockoutModel(cfg.stockout_probability))),
        supervisor=SupervisorAgent(_PROVENANCE),
    )


def scenario_decision_fn(case: EvalCase) -> SupervisorDecision:
    """Run one eval case end-to-end through the framework-agnostic pipeline."""
    return run_pipeline(build_scenario_bundle(case.scenario), case.request)
