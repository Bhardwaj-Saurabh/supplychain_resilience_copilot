from __future__ import annotations

import pytest
from pydantic import ValidationError

from scrc.contracts import QuantileForecastResult


def _fc(**over: object) -> QuantileForecastResult:
    base: dict[str, object] = {
        "sku_id": "A",
        "store_id": "CA_1",
        "horizon_days": 14,
        "p10": 5.0,
        "p50": 8.0,
        "p90": 12.0,
    }
    base.update(over)
    return QuantileForecastResult(**base)  # type: ignore[arg-type]


def test_interval_width_is_derived_when_omitted() -> None:
    assert _fc().interval_width == pytest.approx(7.0)


def test_explicit_interval_width_is_kept() -> None:
    assert _fc(interval_width=7.0).interval_width == 7.0


def test_quantile_order_is_enforced() -> None:
    with pytest.raises(ValidationError):
        _fc(p10=12.0, p50=8.0, p90=5.0)


def test_horizon_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        _fc(horizon_days=0)


def test_model_is_frozen() -> None:
    fc = _fc()
    with pytest.raises((ValidationError, TypeError)):
        fc.p50 = 9.0  # type: ignore[misc]


def test_unknown_fields_are_forbidden() -> None:
    with pytest.raises(ValidationError):
        _fc(bogus=1)
