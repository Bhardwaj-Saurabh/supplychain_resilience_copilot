"""Macro Signal Agent output contract (FRED → regime classifier)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from scrc.contracts.common import SCHEMA_VERSION, Probability, SCRCModel


class RegimeLabel(StrEnum):
    """Macroeconomic regime (PRD §6.2)."""

    TIGHTENING = "tightening"
    NEUTRAL = "neutral"
    EASING = "easing"
    SHOCK = "shock"


class MacroSignals(SCRCModel):
    """Latest FRED observations plus the classified regime overlay."""

    schema_version: str = SCHEMA_VERSION
    series_values: dict[str, float] = Field(default_factory=dict)
    regime_label: RegimeLabel
    regime_confidence: Probability
    relevant_tariff_flags: list[str] = Field(default_factory=list)
