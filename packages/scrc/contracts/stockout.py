"""Stockout-Risk Classifier Agent output contract (XGBoost + SHAP + MAPIE)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from scrc.contracts.common import SCHEMA_VERSION, Probability, SCRCModel


class ConfidenceTier(StrEnum):
    """MAPIE-derived confidence band on the calibrated probability."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ShapValue(SCRCModel):
    """A per-feature SHAP attribution for the stockout classifier.

    ``value`` is the feature's input value; ``contribution`` is its signed SHAP
    contribution to the prediction.
    """

    feature: str = Field(min_length=1)
    value: float
    contribution: float


class StockoutRiskResult(SCRCModel):
    """Calibrated stockout probability with attribution and a planner brief.

    This is the joint risk signal the Supervisor tiers on. The probability must
    be isotonic-calibrated (P2) — ``calibrated`` records whether it is.
    """

    schema_version: str = SCHEMA_VERSION
    sku_id: str = Field(min_length=1)
    store_id: str = Field(min_length=1)
    stockout_probability: Probability
    calibrated: bool
    confidence_tier: ConfidenceTier
    shap_values: list[ShapValue] = Field(default_factory=list)
    plain_language_brief: str = ""
