"""Runtime settings for the composition root."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Deployment configuration.

    ``demo`` boots with no external services (real XGBoost/IsolationForest trained
    on synthetic data at startup, a local forecaster, in-memory audit). ``production``
    wires Chronos/Azure/MLflow/Feast adapters and requires those services.
    """

    profile: str = "demo"

    @classmethod
    def from_env(cls) -> Settings:
        return cls(profile=os.environ.get("SCRC_PROFILE", "demo").lower())
