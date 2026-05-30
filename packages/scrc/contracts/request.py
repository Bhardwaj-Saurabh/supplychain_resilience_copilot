"""Decision request — the typed input that kicks off a Supervisor run."""

from __future__ import annotations

from pydantic import Field

from scrc.contracts.common import SCRCModel


class DecisionRequest(SCRCModel):
    """A request to assess one SKU-store and recommend/route an action."""

    sku_id: str = Field(min_length=1)
    store_id: str = Field(min_length=1)
    horizon_days: int = Field(default=14, gt=0)
    port_ids: list[str] = Field(default_factory=list)
