"""Shared primitives for SCRC typed contracts.

Contracts are the unit of separation between layers (ADR-0003): this package
depends on nothing else in ``scrc``. Every model is **frozen** — decision
artefacts are immutable once produced, which keeps the audit trail and
provenance trustworthy (architecture.md §20). Unknown fields are forbidden so
schema drift surfaces loudly rather than silently.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

#: Contract schema version. Evolution is additive within a major version
#: (architecture.md §20); tool results carry this so consumers can adapt.
SCHEMA_VERSION = "1.0"


class SCRCModel(BaseModel):
    """Base for all typed contracts: immutable and strict."""

    model_config = ConfigDict(frozen=True, extra="forbid")


#: A calibrated probability in [0, 1].
Probability = Annotated[float, Field(ge=0.0, le=1.0)]

#: A non-negative magnitude (e.g. an interval width).
NonNegFloat = Annotated[float, Field(ge=0.0)]
