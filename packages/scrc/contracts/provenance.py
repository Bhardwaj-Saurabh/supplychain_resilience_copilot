"""Decision provenance — what produced a decision (architecture.md §20).

Stamping every decision with the exact versions and input hash is what makes a
decision reconstructable and the ≥95% routing-reproducibility test meaningful:
identical ``input_hash`` + identical versions must yield the same tier.
"""

from __future__ import annotations

from pydantic import Field

from scrc.contracts.common import SCRCModel


class DecisionProvenance(SCRCModel):
    """The provenance tuple recorded on every SupervisorDecision."""

    #: Registry versions per model, e.g. {"chronos": "3", "xgboost": "7", "isoforest": "2"}.
    model_versions: dict[str, str] = Field(default_factory=dict)
    feature_schema_version: str
    policy_config_version: str
    prompt_template_version: str
    llm_model_id: str
    code_git_sha: str
    #: Hash of the canonical decision input — the reproducibility key.
    input_hash: str
