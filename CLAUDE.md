# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

Implementation is **underway, phase by phase** (see the roadmap in [README.md](README.md)). Done so far:
- **Phase 0–1:** repo scaffolding, tooling, machine-enforced layer boundaries, Docker Compose skeleton, and all typed contracts in [packages/scrc/contracts/](packages/scrc/contracts/).
- **Phase 2 (Module 1):** the data & feature layer in [packages/scrc/data/](packages/scrc/data/) (ingestion, feature engineering, validation) plus thin Feast/Airflow wrappers in [pipelines/](pipelines/).
- **Phase 3 (Module 2):** ML serving in [packages/scrc/ml/](packages/scrc/ml/) (Chronos client, XGBoost+SHAP, Isolation Forest+KernelSHAP, MLflow registry) and the ML-as-Tool boundary in [packages/scrc/tools/](packages/scrc/tools/) — tools depend on **structural ports**, not on `scrc.ml`/`scrc.data` (enforced).
- **Phase 4 (Module 3):** deterministic four-tier escalation in [packages/scrc/governance/](packages/scrc/governance/), the five framework-agnostic agents in [packages/scrc/agents/](packages/scrc/agents/), and the LangGraph `StateGraph` in [packages/scrc/orchestration/](packages/scrc/orchestration/) (fan-out → join → Supervisor; missing signal → CRITICAL). Only `scrc.orchestration` imports LangGraph (P5).
- **Phase 5 (Module 4):** the LLM layer in [packages/scrc/llm/](packages/scrc/llm/) (Azure OpenAI via Foundry, primary→secondary failover; SHAP-to-brief that narrates a deterministic factual block, never predicts) and `interrupt()`-based HITL in the graph — REVIEW/CRITICAL decisions pause for a planner and resume from the checkpoint with their outcome. LLM-down falls back to the raw SHAP brief.
- **Phase 6 (Module 5):** governance internals in [packages/scrc/governance/](packages/scrc/governance/) (audit log + **no-audit-no-autonomy** gate per ADR-0002, rollback registry, circuit breaker) and observability adapters in [packages/scrc/observability/](packages/scrc/observability/) (MLflow audit, OTEL tracing, Prometheus — the latter two degrade to no-ops without their deps). The graph now has an `audit` node between Supervisor and routing: it logs every decision, downgrades unauditable autonomous decisions to HITL, and registers reversible actions for rollback before execution.
- **Phase 7 (Module 6):** the FastAPI surface in [packages/scrc/api/](packages/scrc/api/) (`POST /decisions`, the `POST /decisions/{id}/resume` HITL webhook) over orchestration runner helpers; the framework-agnostic [`run_pipeline`](packages/scrc/orchestration/portable.py); and the Microsoft Agent Framework port in [packages/scrc/orchestration/maf/](packages/scrc/orchestration/maf/) reusing the same agents (see [docs/maf_port.md](docs/maf_port.md)). `agent-framework` is not a locked extra (pre-release deps) — install manually to run the MAF test.

The authoritative spec is [docs/supply_chain_copilot_PRD.md](docs/supply_chain_copilot_PRD.md) (PRD v1.0); the design is in [docs/architecture.md](docs/architecture.md) and decisions in [docs/adr/](docs/adr/). Follow these rather than improvising — they are intentionally prescriptive because this is also a teaching reference architecture. **When adding code, respect the layer/dependency rules in [.importlinter](.importlinter)** (`contracts ← data/ml ← tools ← agents ← orchestration`; governance/observability cross-cutting) — they are enforced in CI.

## What this system is

A multi-agent supply chain decision-support system whose central thesis is the **ML-as-Agent-Tool pattern**: real trained models (Amazon Chronos-2 forecasting, XGBoost stockout classifier, Isolation Forest anomaly detector) are served as tools that LangGraph agents call — *not* LLMs role-playing as forecasters. Two properties drive nearly every design decision and must be preserved by any code added:

1. **Uncertainty and explainability are never discarded.** Chronos-2 quantile spread (P10/P50/P90 interval width), XGBoost isotonic-calibrated probabilities, and SHAP/KernelSHAP attributions flow *through* the agents as structured tool results and into escalation logic. Don't reduce a model output to a point estimate before the Supervisor sees it.
2. **Governance is first-class, not an afterthought.** Every Supervisor decision (model inputs, outputs, confidence, SHAP, routing outcome, human approval) is logged to MLflow as an audit trail. Human-in-the-loop is enforced in code via LangGraph `interrupt()`, not via prompts.

## Architecture (intended)

Five agents on a shared LangGraph `StateGraph` with checkpointed state in PostgreSQL:

- **Supervisor Agent** — fans out to the four specialists, synthesises their typed outputs into a risk tier, generates ranked actions (expedite / reroute / safety-stock transfer / substitute), executes low tiers autonomously, routes high tiers to HITL via `interrupt()`, and logs every decision to MLflow.
- **Demand Forecasting Agent** — Chronos-2 → `QuantileForecastResult` (P10/P50/P90 + interval width as confidence signal).
- **Logistics Risk Agent** — Isolation Forest on AIS/BTS series → anomaly score + per-feature SHAP.
- **Macro Signal Agent** — FRED series → regime classifier (`tightening`/`neutral`/`easing`/`shock`).
- **Stockout-Risk Classifier Agent** — assembles the joint feature vector from the other three, calls XGBoost, applies isotonic calibration → calibrated probability + SHAP + plain-language brief.

**Escalation is the core control flow.** A four-tier model gates autonomy, driven by stockout probability and signal conjunction:

| Tier | Condition (see PRD §6.3) | Action |
|---|---|---|
| `ROUTINE` | prob < 0.30, narrow interval, no anomaly | autonomous + log |
| `MONITOR` | prob 0.30–0.55, or single anomaly | autonomous + dashboard flag |
| `REVIEW` | prob 0.55–0.75, or multi-signal conjunction | `interrupt()` to planner with full brief |
| `CRITICAL` | prob > 0.75, or anomaly + macro shock + high uncertainty | block autonomy; escalate with SHAP + counterfactual |

Cross-cutting principle: a missing/timed-out agent result is treated as **maximum uncertainty** and routed conservatively — never hallucinated past. Reversible autonomous actions must be registered in the rollback registry before execution.

## Technology stack (as specified — do not substitute without reason)

- **Agent framework:** LangGraph is primary. Microsoft Agent Framework 1.0 is a *parallel port* of one sub-graph (teaching deliverable, module 6) — keep the core implementation framework-portable; isolate framework-specific code.
- **ML:** Chronos-2 (open-source/Apache 2.0, via HF/AutoGluon — **not an AWS service**; served on an Azure AI Foundry managed endpoint to keep the stack single-cloud), scikit-learn + XGBoost (classification), scikit-learn Isolation Forest (anomaly). SHAP for attribution; MAPIE + Chronos native quantiles for uncertainty. Azure AutoML is benchmarking-only — **never in the production inference path.**
- **MLOps:** MLflow (tracking, registry, *and* audit log — only registry-promoted versions are callable by agent tools). Drift-triggered retraining via Optuna, gated by the same HITL pattern as operational decisions.
- **Data:** Apache Airflow DAGs ingest FRED, BTS FAF, MarineTraffic AIS, M5 (Kaggle). Feast feature store on PostgreSQL (offline = training, online = low-latency agent tool calls). Cache rate-limited AIS responses in Feast with 4h TTL; use BTS FAF as primary logistics signal, AIS as enrichment.
- **API:** FastAPI exposes agent actions, model inference, and the HITL approval webhook.
- **Observability & eval:** OpenTelemetry → **Opik** (Comet, open-source) for agent/LLM trace visualisation *and* the evaluation suite — one trace spans Airflow → all agents → Supervisor → HITL outcome. Prometheus + Grafana are kept only for infra/system metrics (4 dashboards), which Opik doesn't cover. Opik replaces both Jaeger and LangSmith; the OTEL Collector stays as the emission layer.
- **LLM backbone:** Azure OpenAI (GPT-4o) via **Azure AI Foundry** (which also hosts the Chronos-2 endpoint — single platform). Failover is to a **secondary Azure OpenAI deployment** (alt region/model), not a second provider — single-provider/single-cloud; still swappable via the LangChain provider abstraction. The LLM does reasoning and SHAP-to-brief translation — never prediction.
- **Deployment:** Docker Compose (12 services), single `.env` for all secrets, `Makefile` for setup/teardown/data-init. Azure-optimised but cloud-agnostic.

## Build and run

Python 3.11+ with [`uv`](https://docs.astral.sh/uv/). Quality gates (run by `make check`):

- `make setup` — create the venv and install `.[dev]`. Add data libs with `uv pip install -e ".[dev,data]"`.
- `make lint` (ruff) · `make type` (mypy **strict**) · `make lint-imports` (layer boundaries) · `make test` (pytest).
- Single test: `uv run pytest tests/contracts/test_forecasting.py::test_model_is_frozen`.
- `make fmt` — ruff format + autofix.

Conventions that matter: contracts are **frozen** Pydantic models with `extra="forbid"`; the `scrc.data.*` modules have a mypy override relaxing only `warn_return_any` (pandas) — keep the rest strict.

Still to come (PRD-specified, not yet built): `docker-compose up` for the full service stack (currently a skeleton with infra services real and app/inference services as build stubs — see [deploy/docker-compose.yml](deploy/docker-compose.yml)); `make data-init`; and the **Opik** evaluation harness (routing reproducibility ≥ 95%, disruption replay: COVID, Suez 2021, Red Sea 2024).

## Hard constraints (PRD §12 — out of scope for v1)

No autonomous PO execution without approval; no pre-built ERP connectors (expose the FastAPI webhook instead); no streaming ingestion (Airflow batch only); no multi-tenancy; no conversational interface (all outputs are structured typed artefacts); supply-chain/logistics data only.
