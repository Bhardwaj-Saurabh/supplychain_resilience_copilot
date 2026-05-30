# ADR-0004: Retain Chronos-2 (Azure-hosted, open-source) over TimeGEN-1

- **Status:** Accepted
- **Date:** 2026-05-30
- **Deciders:** Lead AI Architect
- **Related:** PRD §4.2, §4.3 (forecasting rationale), §9.2 · architecture.md §8

## Context

Demand forecasting is the anchor signal. The PRD specifies Amazon Chronos-2 for zero-shot quantile forecasting with covariate support. A concern was raised that using a "Amazon" model alongside Azure infrastructure implies a two-cloud footprint, and whether an Azure-native forecasting foundation model (e.g. Nixtla TimeGEN-1, available in the Azure AI Foundry catalog) should replace it.

Key facts:
- Chronos-2 is **open-source (Apache 2.0)**, distributed via Hugging Face / AutoGluon. It is *not* an AWS-hosted service and creates no AWS dependency.
- It can be served on an **Azure AI Foundry managed endpoint** (or a self-hosted AutoGluon container locally), keeping the runtime single-cloud on Azure.
- It produces native P10/P50/P90 quantiles and supports past/future covariates — both load-bearing for the uncertainty-propagation design (PRD §2.2 G4, architecture.md §8).
- Portability of an open-source model serves the teaching goal of a cloud-agnostic architecture.

## Decision

We will **retain Chronos-2** as the demand-forecasting model, served on an Azure AI Foundry managed endpoint (cloud) or an AutoGluon container (local). The architecture and PRD are clarified to state explicitly that Chronos-2 is open-source and Azure-hosted, so there is **no second-cloud dependency**. TimeGEN-1 is documented as a viable Azure-native alternative but is not adopted.

## Consequences

### Positive
- Single-cloud (Azure) runtime with no AWS dependency, resolving the original concern.
- Native quantile + covariate support maps directly onto uncertainty propagation with no extra calibration layer.
- Open-source weights keep the stack portable and teachable; no vendor lock-in on the anchor model.

### Negative / costs
- Self-hosting/serving Chronos-2 carries more operational responsibility than consuming a first-party managed forecasting API.
- Zero-shot accuracy on intermittent M5 SKUs is a known risk (PRD §13) requiring covariate tuning and classical baselines.

### Neutral / follow-ups
- Benchmark Chronos-2 against classical baselines (ETS/ARIMA) and consider LightGBM fallback for sparse series (PRD §13).
- If serving overhead proves prohibitive, revisit TimeGEN-1 via a new ADR; the tool contract (`chronos_forecast → QuantileForecastResult`) is model-agnostic, so a swap is contained.

## Alternatives considered

- **Nixtla TimeGEN-1 (Azure AI Foundry)** — rejected for now: first-party Azure-managed and lower ops burden, but adopting it for a perceived (not real) cloud concern would trade away an open-source, portable anchor model central to the teaching goal.
- **Azure ML AutoML Forecasting** — rejected: native to Azure ML but trained-per-series rather than zero-shot, and a weaker fit for the foundation-model teaching narrative.
