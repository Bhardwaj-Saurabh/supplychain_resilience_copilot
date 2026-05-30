"""MAF port build test — skipped unless the agent_framework SDK is installed.

The framework-agnostic behaviour is verified in test_portable.py; this asserts
the port assembles against the real SDK when the ``maf`` extra is present.
"""

from __future__ import annotations

import pytest

pytest.importorskip("agent_framework")

from scrc.agents import (
    DemandAgent,
    LogisticsAgent,
    MacroAgent,
    ProvenanceContext,
    StockoutAgent,
    SupervisorAgent,
)
from scrc.orchestration import AgentBundle
from scrc.orchestration.maf import build_maf_workflow

from .test_portable import _Forecast, _Logistics, _Macro, _Stockout


def test_maf_workflow_builds() -> None:
    bundle = AgentBundle(
        demand=DemandAgent(_Forecast()),  # type: ignore[arg-type]
        logistics=LogisticsAgent(_Logistics(False)),  # type: ignore[arg-type]
        macro=MacroAgent(_Macro_neutral()),  # type: ignore[arg-type]
        stockout=StockoutAgent(_Stockout(0.1)),  # type: ignore[arg-type]
        supervisor=SupervisorAgent(
            ProvenanceContext(
                feature_schema_version="1.0",
                policy_config_version="default",
                prompt_template_version="1.0",
                llm_model_id="gpt-4o",
                code_git_sha="deadbeef",
            )
        ),
    )
    assert build_maf_workflow(bundle) is not None


def _Macro_neutral() -> _Macro:
    from scrc.contracts import RegimeLabel

    return _Macro(RegimeLabel.NEUTRAL)
