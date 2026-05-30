"""Evaluation datasets: routing reproducibility + disruption replay (PRD §11.1).

Each ``EvalCase`` names a disruption scenario, the request to run, and the tier
the governed system should reach. Scenarios are realised by configured agent
bundles in ``scenarios.py`` and run end-to-end through ``run_pipeline`` — the
disruption-replay evaluation the PRD calls for (COVID, Suez 2021, Red Sea 2024).
"""

from __future__ import annotations

from enum import StrEnum

from scrc.contracts import DecisionRequest, EscalationTier, SCRCModel


class DisruptionScenario(StrEnum):
    BASELINE = "baseline"
    COVID = "covid"
    SUEZ_2021 = "suez_2021"
    RED_SEA_2024 = "red_sea_2024"


class EvalCase(SCRCModel):
    name: str
    scenario: DisruptionScenario
    request: DecisionRequest
    expected_tier: EscalationTier


def _request(tag: str) -> DecisionRequest:
    return DecisionRequest(sku_id=tag, store_id="CA_1", port_ids=["USLAX"])


def disruption_cases() -> list[EvalCase]:
    """The golden disruption-replay set with expected escalation tiers."""
    return [
        EvalCase(
            name="Baseline calm market",
            scenario=DisruptionScenario.BASELINE,
            request=_request("baseline"),
            expected_tier=EscalationTier.ROUTINE,
        ),
        EvalCase(
            name="COVID-19 demand shock",
            scenario=DisruptionScenario.COVID,
            request=_request("covid"),
            expected_tier=EscalationTier.CRITICAL,
        ),
        EvalCase(
            name="Suez Canal 2021 blockage",
            scenario=DisruptionScenario.SUEZ_2021,
            request=_request("suez"),
            expected_tier=EscalationTier.REVIEW,
        ),
        EvalCase(
            name="Red Sea 2024 rerouting",
            scenario=DisruptionScenario.RED_SEA_2024,
            request=_request("red_sea"),
            expected_tier=EscalationTier.CRITICAL,
        ),
    ]
