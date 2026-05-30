"""Macro signal tool (Layer 3).

Fetches the latest FRED-derived values via the feature provider and applies a
lightweight, deterministic regime classifier (PRD §6.2). The rule is a
documented heuristic over the yield-curve spread, ISM PMI, and consumer
sentiment — a trained classifier can replace ``classify_regime`` later without
changing the tool's contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from scrc.contracts import MacroSignals, RegimeLabel
from scrc.tools.ports import FeatureProvider

DEFAULT_SERIES = ("t10y2y", "ism_pmi", "consumer_sentiment", "fuel_cpi", "jobless_claims")


def classify_regime(signals: Mapping[str, float]) -> tuple[RegimeLabel, float]:
    """Classify the macro regime from latest signals. Deterministic heuristic."""
    sentiment = signals.get("consumer_sentiment")
    spread = signals.get("t10y2y")
    pmi = signals.get("ism_pmi")

    if sentiment is not None and sentiment < 55.0:
        return RegimeLabel.SHOCK, 0.7
    if spread is not None and spread < 0.0:
        # Deeper inversion -> higher confidence in a tightening read.
        return RegimeLabel.TIGHTENING, min(1.0, 0.5 + abs(spread))
    if pmi is not None and pmi >= 55.0:
        return RegimeLabel.EASING, 0.6
    return RegimeLabel.NEUTRAL, 0.5


class MacroTool:
    """The ``get_fred_series`` + ``classify_macro_regime`` tool."""

    def __init__(self, features: FeatureProvider, series_ids: Sequence[str] = DEFAULT_SERIES):
        self._features = features
        self._series_ids = list(series_ids)

    def assess_macro(self) -> MacroSignals:
        values = self._features.macro_latest(self._series_ids)
        label, confidence = classify_regime(values)
        return MacroSignals(
            series_values=values,
            regime_label=label,
            regime_confidence=confidence,
            relevant_tariff_flags=[],
        )
