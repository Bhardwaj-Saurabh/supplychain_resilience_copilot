"""Tool / capability layer (Layer 3): models exposed as typed agent tools.

Depends on ``scrc.contracts`` plus the ports defined here — never on
``scrc.ml``/``scrc.data`` (the concrete models satisfy the ports structurally
and are wired at a composition root). This is the ML-as-Tool boundary agents
call against.
"""

from __future__ import annotations

from scrc.tools.forecasting import ForecastTool
from scrc.tools.logistics import LogisticsTool
from scrc.tools.macro import MacroTool, classify_regime
from scrc.tools.ports import (
    AnomalyModel,
    FeatureProvider,
    Forecaster,
    StockoutModel,
)
from scrc.tools.providers import StaticFeatureProvider
from scrc.tools.stockout import StockoutTool

__all__ = [
    "AnomalyModel",
    "FeatureProvider",
    "ForecastTool",
    "Forecaster",
    "LogisticsTool",
    "MacroTool",
    "StaticFeatureProvider",
    "StockoutModel",
    "StockoutTool",
    "classify_regime",
]
