# Supply Chain Resilience Co-Pilot
## Product Requirements Document

> **Version:** 1.0 (Draft) · **Date:** May 2026 · **Status:** Requirements complete — architecture design pending

---

| Field | Detail |
|---|---|
| **Document title** | Supply Chain Resilience Co-Pilot — Product Requirements Document |
| **Version** | 1.0 (Draft) |
| **Date** | May 2026 |
| **Author** | Lead AI Architect |
| **Project type** | Open-source portfolio project / enterprise AI teaching platform |
| **Target audience** | Enterprise AI engineers, architects, and supply chain technology leaders |
| **Frameworks** | LangGraph (primary) · Microsoft Agent Framework 1.0 (port) |
| **Deployment** | Fully containerised · Docker Compose · Cloud-agnostic (Azure-optimised) |
| **Status** | Requirements complete — architecture design pending |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Solution Overview](#3-solution-overview)
4. [Technology Stack](#4-technology-stack)
5. [Data Sources and APIs](#5-data-sources-and-apis)
6. [Agent Architecture](#6-agent-architecture)
7. [Observability and Governance](#7-observability-and-governance)
8. [MLOps and Model Lifecycle](#8-mlops-and-model-lifecycle)
9. [Deployment Architecture](#9-deployment-architecture)
10. [Teaching Platform Design](#10-teaching-platform-design)
11. [Success Criteria](#11-success-criteria)
12. [Out of Scope](#12-out-of-scope)
13. [Risks and Mitigations](#13-risks-and-mitigations)
- [Appendix — Technology Reference](#appendix--technology-reference)

---

## 1. Executive Summary

The Supply Chain Resilience Co-Pilot is an advanced, fully open-source multi-agent AI system that addresses the most significant unsolved problem in enterprise supply chain management: the inability to translate machine learning predictions into governed, explainable, timely action.

Organisations today have demand forecasts. They have risk scores. They have exception dashboards. What they do not have is an orchestration layer that combines those signals — across demand, logistics, and macroeconomic dimensions — into a ranked action plan that a supply chain planner can trust, interrogate, and act on within minutes rather than hours.

This project builds that orchestration layer using three real trained machine learning models — Amazon Chronos-2 for time-series demand forecasting, scikit-learn XGBoost for stockout-risk classification, and scikit-learn Isolation Forest for logistics anomaly detection — served as tools to a LangGraph multi-agent supervisor. The supervisor reasons across all three signal streams simultaneously, propagates calibrated uncertainty through its decision logic, and either acts autonomously within defined thresholds or escalates to a human reviewer with a fully explained, context-rich brief.

The project is simultaneously a production-grade reference architecture and a teaching platform — the first open-source system to demonstrate the ML-as-Agent-Tool pattern at enterprise scale, with full observability, audit logging, and a documented port from LangGraph to Microsoft Agent Framework 1.0.

---

## 2. Problem Statement

### 2.1 The real-world operational problem

Global supply chains operate under mounting structural volatility. Organisations face concurrent pressures from tariff uncertainty, geopolitical realignment, extreme weather events, a demographic retirement cliff that is stripping institutional planning expertise, and a $106 trillion global infrastructure investment gap through 2040. Industry surveys confirm that 78% of supply chain leaders anticipate disruptions to intensify over the next two years, yet only 25% feel genuinely prepared to respond.

Significant investment has gone into supply chain AI over the past decade. Demand forecasting models exist. Risk dashboards exist. Exception management tools exist. Incumbent platforms — Blue Yonder, SAP IBP, Kinaxis, and o9 Solutions — have layered machine learning onto their planning suites and, in 2025 and 2026, rebranded their capabilities as agentic AI. Yet the fundamental operational problem remains unsolved.

> *Supply chains don't break because of a lack of data. They break because the time between an anomaly appearing in the data and a corrective action being taken is measured in hours or days — by which point the production line may already have stopped.*

The issue is structural. Supply chain AI is built in silos. Demand planning generates forecasts without knowing what the logistics platform sees about port delays. Risk dashboards flag supplier problems that never reach procurement systems. Inventory algorithms optimise stock levels based on outdated inbound assumptions. Macroeconomic signals — interest rates, fuel costs, tariff changes, freight indices — are tracked in separate tools with no path into operational decision-making. Each system holds a fragment of reality. None holds the whole picture, and no mechanism exists to reason across all signals simultaneously and produce a governed, explainable recommended action.

| Metric | Value |
|---|---|
| Operations leaders who say AI investments have not fully delivered | **89%** |
| Supply chain leaders still lacking real-time coordination despite modern ERP | **72%** |
| Brands with AI running in live supply chain workflows today | **10%** |
| Agentic AI projects predicted to be cancelled by end of 2027 (Gartner) | **40%+** |

---

### 2.2 The five identified gaps

Research across enterprise deployments, industry analyst reports, and the open-source landscape identifies five compounding gaps that together define why supply chain AI has failed to move from prediction to action.

| # | Gap | Description |
|---|---|---|
| **G1** | **Insight-to-action chasm** | Predictions are delivered as dashboards. No orchestration layer generates, ranks, or routes a response. The bottleneck is not model accuracy — it is the absence of anything that acts on the output. |
| **G2** | **Siloed signal reasoning** | Demand, logistics, and macro signals exist in separate systems. No platform or project combines all three into a single reasoning loop that synthesises conjoint risk before recommending action. |
| **G3** | **LLM-only agents masquerading as ML systems** | Dominant open-source multi-agent projects use prompted LLMs as both the reasoning and prediction layers. No trained model produces actual forecasts. This is the primary source of hallucinated outputs and eroded planner trust. |
| **G4** | **Uncertainty and explainability discarded** | ML models produce calibrated quantile outputs and SHAP attributions. Existing agentic systems strip this information before reaching the agent, which then acts confidently on uncertain inputs without being able to explain its recommendations to planners. |
| **G5** | **No governance or audit architecture** | Every supply chain agent demo that exists is a prototype with no durable state, no structured HITL escalation gate, no audit trail, and no rollback mechanism — failing exactly the requirements that kill 40%+ of enterprise agentic AI deployments. |

---

### 2.3 The combined problem

The supply chain industry in 2026 finds itself with abundant prediction capability and almost no execution capability. The compounding effect of these five gaps forces organisations into one of two unsatisfactory positions: full manual operation, in which AI investment functions as an expensive reporting layer, or premature full automation, in which agents act without calibrated confidence, explainability, or oversight, producing the failures that are now eroding enterprise trust in agentic AI broadly.

What is needed is a multi-agent system in which trained ML models serve as first-class prediction tools — not as LLM role-play — where agent reasoning operates over calibrated uncertainty and SHAP-attributed explanations, where demand, logistics, and macro signals are synthesised in a single supervisor reasoning loop, and where every recommendation is either executed autonomously within defined confidence and reversibility thresholds or escalated to a human reviewer with the complete decision context needed to act in minutes rather than hours.

No such system exists in the open source. No course teaches the architectural patterns required to build it. No reference project demonstrates how to wire ML uncertainty into agentic escalation logic, how to implement governed HITL using framework-native interrupt primitives, or how the same architecture maps across LangGraph and Microsoft Agent Framework 1.0.

> *Supply chain organisations do not lack predictions — they lack a governed, explainable, cross-signal orchestration layer that turns calibrated ML outputs into trusted, auditable actions. The engineering community does not yet have a reference architecture showing how to build one.*

---

## 3. Solution Overview

### 3.1 What this system is

The Supply Chain Resilience Co-Pilot is an end-to-end, fully open-source multi-agent AI system deployed via Docker Compose. It consists of four specialised LangGraph agents — a Demand Forecasting Agent, a Logistics Risk Agent, a Macro Signal Agent, and a Stockout-Risk Classifier Agent — coordinated by a Supervisor Agent that synthesises all signals, applies uncertainty-aware escalation logic, and either acts autonomously or routes to a human reviewer with a complete, SHAP-explained decision brief.

The system ingests real data from public APIs, runs real trained ML models, produces real calibrated predictions, and generates governed, auditable recommendations. It is not a demonstration of what agentic AI could do. It is a working implementation of what it takes to do it correctly.

---

### 3.2 Core capabilities

- **Demand forecasting** using Amazon Chronos-2 (zero-shot, with covariate support) producing P10, P50, and P90 quantile forecasts across M5 Walmart SKU-store series
- **Stockout-risk classification** using scikit-learn XGBoost with isotonic-calibrated probabilities and full SHAP feature attribution
- **Logistics anomaly detection** using scikit-learn Isolation Forest on AIS port congestion and BTS freight time series, with per-feature anomaly contribution scores
- **Cross-signal synthesis** by a LangGraph Supervisor Agent that reasons over all three model outputs simultaneously — treating quantile width as a confidence proxy and stockout probability as the joint risk signal
- **Uncertainty-aware escalation**: actions routed through a four-tier model (`ROUTINE` / `MONITOR` / `REVIEW` / `CRITICAL`) driven by model confidence scores and business impact severity
- **SHAP-to-brief translation**: the Interpreter Agent converts raw SHAP values into plain-language explanations readable by supply chain planners — not data scientists
- **Human-in-the-loop gates** implemented via LangGraph `interrupt()` with full decision context passed to the reviewer, not a bare approval prompt
- **Full audit trail**: every agent decision, ML inference call, and human approval logged to MLflow with structured metadata and configurable WORM properties
- **Drift-triggered retraining**: a Monitoring Agent watches rolling MLflow metrics and triggers Optuna hyperparameter search and re-registration when forecast accuracy degrades
- **Real data pipelines** via Apache Airflow DAGs ingesting from FRED, BTS FAF, MarineTraffic AIS, and M5 on a configurable schedule

---

### 3.3 What this system is not

- **Not a Blue Yonder or SAP IBP competitor.** It is a composable, API-first orchestration layer that can sit alongside enterprise planning suites and reason over signals they cannot access.
- **Not a financial trading system** or general-purpose market predictor. Its scope is supply chain operational decision support.
- **Not a replacement for supply chain planners.** Every action above a configurable impact threshold requires human approval. The system amplifies planner capacity; it does not eliminate planner judgement.
- **Not a chatbot.** There is no conversational interface. Every output is a structured, typed decision artefact.

---

## 4. Technology Stack

### 4.1 Stack design principles

The technology choices are governed by four principles:

- **Open source first** — every core component is free to use, inspect, and modify
- **Container-native** — the entire system runs in Docker Compose with no cloud provider lock-in, though optimised for Azure
- **Production discipline** — the same observability, governance, and drift-monitoring patterns required in enterprise deployment are present from day one
- **Teachability** — each technology choice must be explainable and replaceable, making the stack a curriculum as much as an architecture

---

### 4.2 Full technology stack

| Layer | Tool / Technology | Role in the system |
|---|---|---|
| **Agent framework** | LangGraph (primary) · Microsoft Agent Framework 1.0 (port) | Multi-agent orchestration, StateGraph with checkpointing, `interrupt()`-based HITL, durable workflow support |
| **ML — forecasting** | Amazon Chronos-2 (open-source, Apache 2.0; via Hugging Face / AutoGluon) | Zero-shot quantile demand forecasting on M5 Walmart SKU-store series; supports past and future covariates (price, promo, holidays). **Open-source model, not an AWS service** — served on an Azure AI Foundry managed endpoint, so the stack stays single-cloud (Azure) |
| **ML — classification** | scikit-learn · XGBoost · Azure AutoML | Stockout-risk classification with isotonic probability calibration; Azure AutoML for ensemble benchmarking and model selection |
| **ML — anomaly detection** | scikit-learn · Isolation Forest | Unsupervised anomaly detection on AIS port congestion and BTS freight indices; KernelSHAP for feature attribution |
| **Explainability** | SHAP (SHapley Additive Explanations) | Per-prediction feature attribution for XGBoost and Isolation Forest; values passed as structured agent tool results |
| **Uncertainty quantification** | MAPIE (conformal prediction) · Chronos-2 native quantiles | Distribution-free prediction intervals on regression tasks; quantile spread as confidence proxy for agent escalation logic |
| **ML tracking & registry** | MLflow | Experiment tracking, model versioning, agent decision logging, WORM audit trail, drift metric monitoring, model serving via `mlflow.pyfunc` |
| **Data pipeline** | Apache Airflow | DAG-based pipeline orchestration for FRED, BTS FAF, MarineTraffic AIS, and M5 ingestion; incremental loads, schema validation, feature store writes |
| **Feature store** | Feast (open source) on PostgreSQL | Offline feature store for training; online store for low-latency agent tool calls at inference time |
| **Data sources — demand** | M5 Walmart (Kaggle) · Corporación Favorita | 42,840 hierarchical daily SKU-store sales series; calendar, price, and SNAP covariates |
| **Data sources — logistics** | BTS Freight Analysis Framework · MarineTraffic AIS API · Census CFS 2022 | Freight tonnage, mode splits, port dwell times, terminal congestion index, vessel ETAs |
| **Data sources — macro** | FRED API (Federal Reserve, St. Louis) | 800,000+ economic time series: fuel CPI, ISM PMI, T10Y2Y spread, UMCSENT, ICSA jobless claims |
| **Observability — agent tracing & eval** | Opik (Comet, open-source) · OpenTelemetry Collector (OTEL) | Agent/LLM trace capture and visualisation for every agent node, ML inference call, and tool invocation; OTEL spans (propagated across LangGraph and Airflow) are ingested by Opik. Opik also hosts the offline/online evaluation suite — see Evaluation row |
| **Observability — infra metrics** | Prometheus · Grafana | Agent throughput, escalation rates, HITL latency, ML model inference latency, forecast drift metrics; pre-built Grafana dashboards (system/infra metrics that Opik does not cover) |
| **LLM provider** | Azure OpenAI (GPT-4o) via **Azure AI Foundry** — primary + secondary deployment for failover | LLM backbone for agent reasoning, SHAP-to-brief translation, and Supervisor synthesis. Single-provider/single-cloud: failover is to a secondary Azure OpenAI deployment (alternate region/model), not a second provider. Still swappable via the LangChain provider abstraction |
| **Cloud platform** | Azure AI Foundry · Azure Container Apps | Azure AI Foundry hosts **both** the Chronos-2 managed inference endpoint **and** the Azure OpenAI LLM deployments; Container Apps for orchestration; Azure AutoML for model benchmarking |
| **Containerisation** | Docker · Docker Compose | Full local development stack in a single compose file: agents, Airflow, MLflow, Feast, Prometheus, Grafana, Opik, PostgreSQL |
| **Database** | PostgreSQL | Feature store backend (Feast), MLflow metadata, Airflow metadata, audit log persistence |
| **API layer** | FastAPI | REST endpoints exposing agent actions, HITL approval webhooks, and model inference for integration with external systems |
| **Evaluation** | Opik (Comet, open-source) · Custom evaluation harness | Agent behavioural reproducibility tests (identical inputs → consistent routing ≥ 95%), disruption replay evaluation (COVID, Suez 2021, Red Sea 2024); Opik datasets, experiments, and scoring metrics drive the harness — unifying tracing and evaluation in one open-source tool |

---

### 4.3 Architecture decisions — rationale

#### LangGraph as the primary agent framework

LangGraph is selected as the primary framework because it provides the most mature Python-native implementation of the patterns this system requires: typed `StateGraph` with checkpoint-based durable state, `interrupt()` for code-enforced HITL (not prompt-based), the `Command` primitive for dynamic conditional routing, and OpenTelemetry-based trace export consumed by Opik for agent observability and evaluation. It is the most-searched agent framework in 2026 and has the broadest community, maximising teaching reach.

Microsoft Agent Framework 1.0 is used as a parallel port of one workflow to demonstrate framework-agnostic architectural thinking and to serve the Azure-native enterprise audience. The MAF port documents differences in `WorkflowBuilder`, `Executors`, `DurableWorkflows`, and the Magentic-One orchestration pattern — a direct teaching differentiator.

#### scikit-learn + XGBoost + Azure AutoML for classification and anomaly detection

scikit-learn is the industry-standard Python ML library and is universally understood by the target audience. XGBoost within scikit-learn provides gradient boosted classification with native probability calibration (isotonic regression) and full SHAP support — both essential for the uncertainty propagation and explainability requirements of this system. Azure AutoML is used for ensemble benchmarking during model selection, providing a cloud-accelerated baseline comparison before registering the final model in MLflow.

The Isolation Forest, also from scikit-learn, provides the anomaly detection layer. Its anomaly scores are interpretable via KernelSHAP, maintaining the explainability contract throughout the system.

#### Amazon Chronos-2 for demand forecasting

Chronos-2 is selected over classical forecasting baselines because it is the state-of-the-art zero-shot time-series foundation model as of late 2025, it natively supports past and future covariates (critical for M5 retail data where 73% of series are intermittent and covariate-blind forecasters fail catastrophically), and it produces native quantile outputs (P10, P50, P90) that feed directly into the uncertainty propagation logic without additional calibration infrastructure.

A clarification on cloud footprint: although Chronos-2 originates from Amazon's research team, it is an **open-source model (Apache 2.0) distributed via Hugging Face and AutoGluon — it is not an AWS-hosted service and creates no AWS dependency.** In this system it is served on an Azure AI Foundry managed endpoint (or a self-hosted AutoGluon inference container locally), keeping the entire stack single-cloud on Azure. Azure-native forecasting foundation models (e.g. Nixtla TimeGEN-1 in the Azure AI Foundry catalog) were evaluated as alternatives; Chronos-2 is retained because its zero-shot covariate support and native quantile outputs map most directly onto the uncertainty-propagation requirements, and because keeping a portable open-source model preserves the teaching goal of a cloud-agnostic architecture.

#### Apache Airflow for data pipelines

Airflow is the dominant open-source workflow orchestration platform and is directly applicable to enterprise data engineering roles. Its DAG model provides clear lineage, retry logic, scheduling, and failure alerting — the operational properties that distinguish a production pipeline from a Jupyter notebook. For the teaching platform, Airflow DAGs are the artefact that shows what real data engineering looks like behind an agentic system.

#### Opik + OpenTelemetry + Prometheus + Grafana for observability and evaluation

OpenTelemetry (OTEL) is the open standard for distributed tracing and is emitted by both LangGraph and Microsoft Agent Framework 1.0. Every agent node, ML inference call, tool invocation, and human approval event emits a span. **Opik (Comet's open-source LLM observability and evaluation platform) ingests these OTEL spans** and provides agent/LLM trace visualisation, prompt and tool-call inspection, and — critically — a single home for the evaluation suite (datasets, experiments, and scoring metrics). Consolidating agent tracing and evaluation in Opik replaces the previously separate Jaeger (trace UI) and LangSmith (evaluation) tools with one open-source component. Prometheus and Grafana are retained for what Opik does not cover: system/infra metrics — Prometheus scrapes metrics from all components, and Grafana provides pre-built dashboards for agent performance, model drift, HITL rates, and pipeline health. Together this stack addresses the governance gap directly and is a complete open-source demonstration of how to instrument *and evaluate* a production agentic AI system.

#### MLflow for ML lifecycle and audit trail

MLflow serves a dual role: as the ML experiment tracking and model registry, and as the structured audit log for agent decisions. Every agent decision — including the ML model inputs, outputs, confidence scores, SHAP values, and routing outcome — is logged as an MLflow run with full parameter and metric metadata. This creates a reproducible, searchable decision history that satisfies SOC2-style audit requirements and provides the Forecast Value Added telemetry needed to measure and improve the system over time.

#### PostgreSQL as the unified data store

PostgreSQL serves as the backend for three components: the Feast feature store (offline and online stores), the MLflow metadata database, and the Airflow scheduler metadata. A single managed PostgreSQL instance (or Azure Database for PostgreSQL in cloud deployment) reduces operational complexity and provides transactional consistency for feature reads and writes.

---

## 5. Data Sources and APIs

### 5.1 Primary open-source datasets

| Dataset | Description | Usage |
|---|---|---|
| **M5 Walmart Competition** (Kaggle) | 42,840 hierarchical daily unit-sales series for 3,049 products across 10 Walmart stores in California, Texas, and Wisconsin, 2011–2016. Includes item-level prices, promotional events, and SNAP calendar flags. | Chronos-2 forecasting; XGBoost stockout-risk classifier training |
| **Corporación Favorita Grocery Sales** (Kaggle) | Ecuador-based grocery chain, 54 million sales records across 54 stores and 33 product families | Supplementary multi-geography model validation |
| **BTS Freight Analysis Framework (FAF) v5.7.1** | US government open data — annual estimates and forecasts of freight tonnage, value, and ton-miles by 132 FAF zones, commodity class, and transport mode through 2050 | Logistics Risk Agent freight volume features |
| **Census Bureau Commodity Flow Survey 2022** | Shipper-level commodity flow data via `api.census.gov` REST API; origin-destination freight matrices at metro area level | Logistics feature enrichment |

---

### 5.2 Live public APIs

| API | Key series / data | Notes |
|---|---|---|
| **FRED API** (Federal Reserve Bank of St. Louis) | `DCOILBRENTEU` (fuel CPI), `MANEMP` (ISM PMI), `T10Y2Y` (yield spread), `UMCSENT` (consumer sentiment), `ICSA` (jobless claims) | Free with API key; 800,000+ series available |
| **MarineTraffic Container Intelligence API** | Terminal-level congestion index, vessel dwell times, 6-week predictive ETAs | Rate-limited to 500 req/min on free tier; responses cached in Feast with 4-hour TTL |
| **Alpha Vantage** | Commodity prices (crude oil, steel, shipping indices), FX rates | Publishes an official MCP server — used as a teaching demonstration of the MCP protocol |

---

## 6. Agent Architecture

### 6.1 Overview

The system comprises five agents, each with a clearly defined role, a set of registered tool calls, and a typed output schema. All agents share a common LangGraph `StateGraph` with checkpointed state persisted in PostgreSQL.

```
                    ┌─────────────────────────────────────────────┐
                    │              Supervisor Agent               │
                    │  Synthesises signals · Routes decisions     │
                    │  Executes ROUTINE/MONITOR autonomously      │
                    │  Routes REVIEW/CRITICAL via interrupt()     │
                    └────────────┬──────────────────┬────────────┘
                                 │                  │
              ┌──────────────────┼──────────────────┼──────────────┐
              ▼                  ▼                  ▼              ▼
   ┌──────────────────┐ ┌──────────────┐ ┌─────────────┐ ┌──────────────────┐
   │  Demand          │ │  Logistics   │ │  Macro      │ │  Stockout-Risk   │
   │  Forecasting     │ │  Risk Agent  │ │  Signal     │ │  Classifier      │
   │  Agent           │ │              │ │  Agent      │ │  Agent           │
   │  Chronos-2       │ │  Isolation   │ │  FRED API   │ │  XGBoost +       │
   │  P10/P50/P90     │ │  Forest +    │ │  Regime     │ │  SHAP + MAPIE    │
   │  quantile output │ │  AIS/BTS     │ │  classifier │ │  calibrated prob │
   └──────────────────┘ └──────────────┘ └─────────────┘ └──────────────────┘
```

---

### 6.2 Agent specifications

#### Demand Forecasting Agent

| Field | Detail |
|---|---|
| **Tool calls** | `chronos_forecast(sku_ids, horizon, covariates)` → `QuantileForecastResult` |
| **Responsibilities** | Retrieves current feature vectors from Feast online store; calls Chronos-2 inference endpoint; returns quantile forecasts with interval width as confidence signal |
| **Output schema** | `sku_id`, `store_id`, `horizon_days`, `p10`, `p50`, `p90`, `interval_width`, `covariate_flags_used` |

#### Logistics Risk Agent

| Field | Detail |
|---|---|
| **Tool calls** | `get_port_congestion(ports)` → `CongestionMetrics`; `detect_freight_anomaly(series_id, window)` → `AnomalyResult` |
| **Responsibilities** | Queries MarineTraffic API for current dwell times; pulls BTS FAF freight series from Feast; runs Isolation Forest inference; returns anomaly score with per-feature SHAP attribution |
| **Output schema** | `port_ids`, `congestion_score`, `anomaly_flag`, `anomaly_score`, `top_features [{feature, shap_value}]` |

#### Macro Signal Agent

| Field | Detail |
|---|---|
| **Tool calls** | `get_fred_series(series_ids, lookback_days)` → `MacroSignals`; `classify_macro_regime(signals)` → `RegimeLabel` |
| **Responsibilities** | Pulls latest FRED observations; runs a lightweight regime classifier (`tightening` / `neutral` / `easing` / `shock`) based on T10Y2Y spread, CPI, and ISM PMI; returns macro overlay for Supervisor context |
| **Output schema** | `series_values`, `regime_label`, `regime_confidence`, `relevant_tariff_flags` |

#### Stockout-Risk Classifier Agent

| Field | Detail |
|---|---|
| **Tool calls** | `classify_stockout_risk(sku_id, store_id, demand_signals, logistics_signals, macro_signals)` → `StockoutRiskResult` |
| **Responsibilities** | Assembles joint feature vector from all three upstream agents; calls XGBoost classifier endpoint; applies isotonic calibration; returns calibrated probability with SHAP attribution and plain-language explanation |
| **Output schema** | `sku_id`, `stockout_probability`, `calibrated`, `confidence_tier`, `shap_values [{feature, value, contribution}]`, `plain_language_brief` |

#### Supervisor Agent

| Field | Detail |
|---|---|
| **Responsibilities** | Receives all four agent outputs; determines overall risk tier using configurable scoring matrix; generates ranked action recommendations (`expedite`, `reroute`, `safety-stock transfer`, `substitute`); executes `ROUTINE` and `MONITOR` actions autonomously; routes `REVIEW` and `CRITICAL` actions to HITL gate via `interrupt()`; logs every decision to MLflow |
| **HITL gate** | `interrupt()` pauses execution and emits a structured `ReviewRequest` to the FastAPI webhook; planner receives a fully explained brief; approval or override resumes the graph; all outcomes are logged |

---

### 6.3 Escalation tier model

| Tier | Trigger condition | Agent action |
|---|---|---|
| `ROUTINE` | Stockout probability < 0.30, interval width narrow, no anomaly | Execute autonomously, log to MLflow |
| `MONITOR` | Stockout probability 0.30–0.55, or single signal anomaly | Execute autonomously, flag in dashboard |
| `REVIEW` | Stockout probability 0.55–0.75, or multi-signal conjunction | Route to planner via `interrupt()` with full brief |
| `CRITICAL` | Stockout probability > 0.75, or anomaly + macro shock + high demand uncertainty | Block autonomous action; escalate immediately with SHAP explanation and counterfactual |

---

## 7. Observability and Governance

### 7.1 Distributed tracing — OpenTelemetry + Opik

Every agent node, tool call, and ML inference invocation emits an OpenTelemetry span with standard attributes:

```
agent_id · action_type · model_id · model_version · input_hash
output_confidence · escalation_tier · human_approval_required
```

Spans are collected by the OTEL Collector container and ingested by Opik for trace visualisation. The complete decision cycle — from Airflow pipeline completion through all four agents, Supervisor synthesis, and HITL outcome — is visible as a single trace in Opik, where the same captured traces also feed the evaluation suite (§4.2, Evaluation).

---

### 7.2 Metrics — Prometheus + Grafana

Prometheus scrapes four metric endpoints:

| Endpoint | Metrics collected |
|---|---|
| LangGraph agent runtime | Decisions per minute, agent latency by node, escalation tier distribution |
| ML inference service | Inference latency, feature drift score, model version in production |
| Airflow pipeline | DAG success rate, task duration, data freshness lag |
| FastAPI layer | HITL response time, approval vs override ratio, recommendations actioned rate |

**Four pre-built Grafana dashboards:**

- **Agent Operations** — decision throughput and escalation rates
- **Model Health** — drift metrics and calibration curves
- **Pipeline Health** — ingestion freshness and error rates
- **Planner Productivity** — HITL latency and Forecast Value Added metrics

---

### 7.3 Audit trail — MLflow

Every Supervisor decision is logged as an MLflow run capturing:

- Input signals from all three ML models (raw quantile outputs and SHAP values)
- Computed stockout probability and confidence tier
- Action recommended and rationale
- Autonomous vs escalated routing decision
- Human reviewer ID and outcome (for escalated actions)
- Eventual inventory outcome (populated by Monitoring Agent on next cycle)

The combination of input signals, model outputs, routing logic, and outcome creates a complete, queryable **Forecast Value Added** record — allowing the system to identify which planners systematically improve or degrade AI recommendations over time.

---

### 7.4 Governance controls

| Control | Implementation |
|---|---|
| **Autonomy thresholds** | Configurable per action class (by SKU category, dollar impact, reversibility class); stored in PostgreSQL and read by Supervisor Agent at runtime |
| **Circuit breakers** | Any agent failing to return a typed result within the configured timeout emits an escalation token; Supervisor treats missing output as maximum uncertainty, routing conservatively rather than hallucinating |
| **Least-privilege agent identity** | Each agent has a named service account with scoped access to only its required tool endpoints; no agent has write access to the audit log |
| **Rollback registry** | Reversible autonomous actions (draft reorder, proposed safety-stock transfer) are registered before execution; Monitoring Agent can trigger rollback within the configurable window if subsequent demand observation contradicts the action |

---

## 8. MLOps and Model Lifecycle

### 8.1 Model training and registration

The XGBoost stockout classifier and Isolation Forest anomaly detector are trained offline on M5 and BTS data using scikit-learn pipelines with full reproducibility:

- Fixed random seeds and logged hyperparameters
- Versioned feature schemas registered in Feast
- Evaluation metrics: AUC, MAPE, F1 at configurable anomaly threshold
- Every training run logged to MLflow; the registered MLflow Model Registry version is the only one callable by agent tool endpoints

Azure AutoML is used during the initial model selection phase to benchmark XGBoost against LightGBM, CatBoost, and random forest ensembles on the same feature set. AutoML experiment results are imported into MLflow for comparison. **Azure AutoML is not used in the production inference path** — the winning scikit-learn pipeline is extracted and served independently.

---

### 8.2 Drift monitoring and retraining

The **Monitoring Agent** runs on a configurable schedule (default: daily):

1. Computes rolling MAPE on Demand Forecasting Agent predictions vs actual sales
2. Computes rolling F1 on Stockout Risk Classifier vs observed stockout events
3. On metric degradation beyond threshold → triggers Optuna hyperparameter optimisation on the current training data window
4. Evaluates candidate model against held-out validation set
5. Registers candidate as a new version in MLflow

**Promotion logic:** Automatic if candidate exceeds production on all metrics by a configurable margin; human approval gate otherwise. This mirrors the same HITL pattern as the operational decision flow — applied to the ML lifecycle.

---

## 9. Deployment Architecture

### 9.1 Local development — Docker Compose

The complete system runs locally via a single `docker-compose up` command. The Compose file defines **twelve services**:

| Service group | Services |
|---|---|
| **Agent runtime** | LangGraph agent runtime · FastAPI API layer |
| **Pipeline** | Apache Airflow webserver · Airflow scheduler · Airflow worker |
| **ML lifecycle** | MLflow tracking server · Feast feature server |
| **Inference** | Chronos-2 inference container · XGBoost + Isolation Forest inference container |
| **Observability** | OpenTelemetry Collector · Opik · Prometheus · Grafana |
| **Data** | PostgreSQL |

A single `.env` file configures all API keys (FRED, MarineTraffic, Azure OpenAI / Azure AI Foundry), database credentials, and autonomy threshold overrides. A `Makefile` provides one-command setup, teardown, and data initialisation targets.

---

### 9.2 Cloud deployment — Azure-optimised

| Component | Azure service |
|---|---|
| LangGraph agent runtime + FastAPI | Azure Container Apps |
| Chronos-2 inference | Azure AI Foundry managed endpoint |
| PostgreSQL | Azure Database for PostgreSQL Flexible Server |
| OTEL telemetry + agent tracing/eval | Azure Monitor Application Insights + Opik (self-hosted or Comet-managed) |
| LLM backbone | Azure OpenAI (GPT-4o) via Azure AI Foundry (primary + secondary deployment) |
| Feature store snapshots + MLflow artefacts | Azure Blob Storage |

The Microsoft Agent Framework 1.0 port deploys to the same container infrastructure, demonstrating that the architecture is framework-portable and cloud-agnostic.

---

## 10. Teaching Platform Design

### 10.1 Curriculum structure

The project is structured as a **six-module progressive curriculum**, each with a standalone notebook, a recorded walkthrough, and a GitHub-tagged release:

| Module | Title | Key content |
|---|---|---|
| **1** | Data pipelines and feature engineering | Airflow DAGs, FRED and AIS ingestion, Feast feature store, M5 data preparation |
| **2** | Real ML models as agent tools | Chronos-2 with covariates, XGBoost + SHAP, Isolation Forest, MLflow registration and serving |
| **3** | Multi-agent architecture in LangGraph | StateGraph design, typed schemas, fan-out to specialist agents, Supervisor synthesis logic |
| **4** | Uncertainty and explainability | Quantile intervals as agent context, SHAP-to-brief translation, four-tier escalation gating, `interrupt()`-based HITL |
| **5** | Production observability and governance | OTEL tracing into Opik, Opik evaluation suite, Prometheus + Grafana infra dashboards, MLflow audit trail, circuit breakers and rollback registry |
| **6** | Framework portability — MAF port | Microsoft Agent Framework 1.0 WorkflowBuilder, Executors, DurableWorkflows, side-by-side LangGraph comparison |

---

### 10.2 Novel teaching contributions

Five architectural patterns taught here that are not documented or demonstrated in any existing course, tutorial, or open-source repository:

1. **ML uncertainty propagation into agent escalation** — passing Chronos-2 P10/P90 interval width and XGBoost calibrated probability as confidence signals that gate autonomous vs supervised action
2. **SHAP-to-brief translation as an agent tool** — wrapping SHAP attribution into a structured tool result that an LLM agent uses to generate planner-readable explanations
3. **Cross-signal supervisor reasoning** — designing a StateGraph that fans out to three independent ML tool calls and synthesises conjoint risk before routing
4. **Production MLOps inside an agentic system** — MLflow as both model registry and agent decision audit log, with drift-triggered retraining as a LangGraph sub-graph
5. **Framework-agnostic agent architecture** — demonstrating that the same multi-agent ML system maps to both LangGraph and Microsoft Agent Framework 1.0 with documented differences

---

## 11. Success Criteria

### 11.1 Technical criteria

| Criterion | Target |
|---|---|
| Chronos-2 with covariates vs M5 naïve baseline (WRMSSE, 2016 test window) | Below baseline on all three US states |
| XGBoost stockout classifier AUC | ≥ 0.85 |
| XGBoost Expected Calibration Error (ECE) | ≤ 0.05 |
| Isolation Forest F1 on injected anomaly scenarios | ≥ 0.70 |
| Supervisor routing reproducibility (identical inputs) | ≥ 95% consistent routing decisions |
| HITL latency from `interrupt()` trigger to planner notification | ≤ 30 seconds end-to-end |
| Docker Compose cold start (all 12 services healthy) | Single `docker-compose up` on standard developer laptop |

---

### 11.2 Portfolio and teaching criteria

| Criterion | Target |
|---|---|
| GitHub repository stars within 90 days of public release | ≥ 500 |
| Curriculum modules complete with notebook + recording + tagged release | All 6 modules |
| Conference citation or published article reference within 12 months | ≥ 1 |
| Framework comparison module engagement (Microsoft + LangChain communities) | Measurable (issues, forks, LinkedIn reposts) |

---

## 12. Out of Scope

The following are explicitly excluded from v1:

- Autonomous financial transactions or purchase order execution without human approval
- Integration with live ERP systems (SAP, Oracle) — the system exposes a FastAPI webhook for ERP integration but does not provide pre-built ERP connectors in v1
- Real-time streaming ingestion (Kafka, Kinesis) — Airflow batch pipelines with configurable frequency are used in v1; streaming ingestion is a documented future extension
- Multi-tenant SaaS deployment — the system is designed for single-organisation deployment; multi-tenancy is a v2 roadmap item
- Natural language conversational interface — all agent outputs are structured typed artefacts; a conversational front-end is a documented future extension
- Healthcare or financial services domain data — the system is scoped to supply chain and logistics data sources in v1

---

## 13. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Chronos-2 underperforms on intermittent M5 SKUs without covariate tuning | Medium | Benchmark against classical baselines (ETS, ARIMA) early; fall back to LightGBM for sparse series if zero-shot accuracy is insufficient |
| MarineTraffic AIS API rate limits constrain real-time logistics features | High | Cache API responses in Feast with a 4-hour TTL; use BTS FAF as the primary logistics signal with AIS as enrichment |
| Azure AutoML costs exceed budget during model selection phase | Low | Cap AutoML experiment budget in config; use local scikit-learn `GridSearchCV` as the fallback selection method |
| LangGraph supervisor routing reproducibility below 95% threshold | Medium | Increase schema specificity for Supervisor inputs; add structured-output enforcement; fall back to deterministic rule-based routing for `CRITICAL` tier |
| MAF port reveals breaking incompatibilities with LangGraph patterns | Low | Document incompatibilities explicitly — this is itself a teaching contribution; scope the MAF port to the Demand + Supervisor sub-graph rather than the full system if necessary |

---

## Appendix — Technology Reference

### Open-source components (all required, no licence fees)

| Component | Version / notes |
|---|---|
| LangGraph | Primary agent framework |
| scikit-learn | ML classification and anomaly detection |
| XGBoost | Gradient-boosted classification with native SHAP support |
| SHAP | Feature attribution for XGBoost and Isolation Forest |
| MAPIE | Conformal prediction intervals (uncertainty quantification) |
| MLflow | ML lifecycle tracking, model registry, audit trail |
| Apache Airflow | Data pipeline orchestration |
| Feast | Open-source feature store (PostgreSQL backend) |
| FastAPI | REST API layer and HITL webhook server |
| PostgreSQL | Unified data store for Feast, MLflow, and Airflow |
| OpenTelemetry Collector | Distributed tracing standard (emission) |
| Opik (Comet) | Agent/LLM trace visualisation **and** evaluation suite (datasets, experiments, scoring); ingests OTEL spans |
| Prometheus | Infra/system metrics collection and alerting |
| Grafana | Infra metrics dashboards |
| Docker / Docker Compose | Full local development stack containerisation |
| Amazon Chronos-2 | Time-series foundation model (Apache 2.0 licence) |

---

### Cloud / managed services (optional — for production deployment)

| Service | Role |
|---|---|
| Azure OpenAI (GPT-4o) via Azure AI Foundry | LLM backbone for agent reasoning; primary + secondary deployment for failover (single provider) |
| Azure AI Foundry | Managed endpoints for **both** Chronos-2 inference **and** the Azure OpenAI LLM deployments |
| Azure Container Apps | Production container orchestration |
| Azure Database for PostgreSQL | Managed database (Feast + MLflow + Airflow) |
| Azure AutoML | Model benchmarking during selection phase only |
| Azure Monitor / Application Insights | OTEL telemetry export for production |
| Azure Blob Storage | MLflow artefacts and feature store snapshots |
| Alpha Vantage API | Commodity prices and macro series |
| FRED API (Federal Reserve) | 800,000+ US economic time series |
| MarineTraffic Container Intelligence API | Port congestion and vessel dwell data |

---

*Supply Chain Resilience Co-Pilot · PRD v1.0 · May 2026 · All core components open source*
