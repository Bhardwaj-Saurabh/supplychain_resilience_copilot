from __future__ import annotations

import pytest

from scrc.ml.forecasting import parse_chronos_response


def test_parse_sums_quantiles_over_horizon() -> None:
    payload = {
        "predictions": {
            "0.1": [1.0, 2.0, 3.0],
            "0.5": [2.0, 3.0, 4.0],
            "0.9": [3.0, 4.0, 5.0],
        }
    }
    result = parse_chronos_response("A", "CA_1", payload, covariate_flags=["price"])
    assert result.horizon_days == 3
    assert result.p10 == 6.0 and result.p50 == 9.0 and result.p90 == 12.0
    # interval_width derived by the contract (P90 - P10).
    assert result.interval_width == 6.0
    assert result.covariate_flags_used == ["price"]


def test_parse_accepts_pXX_keys() -> None:
    payload = {"predictions": {"p10": [1.0], "p50": [2.0], "p90": [3.0]}}
    result = parse_chronos_response("A", "CA_1", payload)
    assert result.p50 == 2.0


def test_parse_rejects_unequal_arrays() -> None:
    payload = {"predictions": {"0.1": [1.0], "0.5": [2.0, 3.0], "0.9": [3.0]}}
    with pytest.raises(ValueError):
        parse_chronos_response("A", "CA_1", payload)
