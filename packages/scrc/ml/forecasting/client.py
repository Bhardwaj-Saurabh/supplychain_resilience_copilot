"""Chronos-2 forecasting client (ML serving, Layer 2).

Chronos-2 is served on an Azure AI Foundry managed endpoint (ADR-0004); this is
a thin HTTP client, not a local model run. The response parsing is a pure
function so it is unit-testable without the endpoint.

The endpoint returns per-quantile arrays over the future horizon. We reduce each
quantile to a single **total expected demand over the horizon** (the sum across
horizon steps) — the quantity the stockout classifier reasons about — and keep
``interval_width`` (P90 - P10) as the confidence signal (P2; architecture.md §8).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx

from scrc.contracts import QuantileForecastResult

_QUANTILE_KEYS = {"p10": ("0.1", "p10"), "p50": ("0.5", "p50"), "p90": ("0.9", "p90")}


def _extract_quantile(predictions: dict[str, Any], keys: tuple[str, ...]) -> list[float]:
    for key in keys:
        if key in predictions:
            return [float(v) for v in predictions[key]]
    raise KeyError(f"none of {keys} present in Chronos response predictions")


def parse_chronos_response(
    sku_id: str,
    store_id: str,
    payload: dict[str, Any],
    covariate_flags: Sequence[str] | None = None,
) -> QuantileForecastResult:
    """Parse a Chronos endpoint payload into a ``QuantileForecastResult``.

    Expects ``payload["predictions"]`` to map quantile keys (``"0.1"``/``"0.5"``/
    ``"0.9"`` or ``"p10"``/...) to per-step arrays of equal length. Pure: no I/O.
    """
    predictions = payload["predictions"]
    p10_steps = _extract_quantile(predictions, _QUANTILE_KEYS["p10"])
    p50_steps = _extract_quantile(predictions, _QUANTILE_KEYS["p50"])
    p90_steps = _extract_quantile(predictions, _QUANTILE_KEYS["p90"])
    horizon = len(p50_steps)
    if not (len(p10_steps) == len(p90_steps) == horizon) or horizon == 0:
        raise ValueError("Chronos quantile arrays must be non-empty and equal length")
    p10, p50, p90 = sum(p10_steps), sum(p50_steps), sum(p90_steps)
    return QuantileForecastResult(
        sku_id=sku_id,
        store_id=store_id,
        horizon_days=horizon,
        p10=p10,
        p50=p50,
        p90=p90,
        interval_width=p90 - p10,
        covariate_flags_used=list(covariate_flags or []),
    )


class ChronosForecaster:
    """Adapter over the Chronos-2 inference endpoint.

    The ``httpx.Client`` is injectable so tests can stub the transport.
    """

    def __init__(self, endpoint: str, api_key: str, client: httpx.Client | None = None) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._api_key = api_key
        self._client = client or httpx.Client(timeout=60.0)

    def forecast(
        self,
        sku_id: str,
        store_id: str,
        context: Sequence[float],
        horizon_days: int,
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> QuantileForecastResult:
        body: dict[str, Any] = {
            "context": list(context),
            "prediction_length": horizon_days,
            "quantile_levels": [0.1, 0.5, 0.9],
        }
        if covariates:
            body["covariates"] = {k: list(v) for k, v in covariates.items()}
        resp = self._client.post(
            f"{self._endpoint}/score",
            json=body,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        resp.raise_for_status()
        return parse_chronos_response(
            sku_id, store_id, resp.json(), covariate_flags=list(covariates or {})
        )
