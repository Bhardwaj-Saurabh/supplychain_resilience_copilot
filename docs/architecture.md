# Supply Chain Resilience Co-Pilot — Architecture

> **Companion to** [supply_chain_copilot_PRD.md](supply_chain_copilot_PRD.md) (PRD v1.0). The PRD defines *what* and *why*; this document defines *how* — the modular structure, component boundaries, data and control flow, and the deployment topology that realise the requirements.

---

## Table of contents

1. [Architectural principles](#1-architectural-principles)
2. [System context (C4 L1)](#2-system-context-c4-l1)
3. [Layered / modular view](#3-layered--modular-view)
4. [Module boundaries & separation of concerns](#4-module-boundaries--separation-of-concerns)
5. [Agent orchestration (LangGraph StateGraph)](#5-agent-orchestration-langgraph-stategraph)
6. [End-to-end decision flow](#6-end-to-end-decision-flow)
7. [HITL escalation sequence](#7-hitl-escalation-sequence)
8. [ML-as-Tool serving layer](#8-ml-as-tool-serving-layer)
9. [Data pipeline & feature store](#9-data-pipeline--feature-store)
10. [Observability, evaluation & governance](#10-observability-evaluation--governance)
11. [MLOps: drift & retraining](#11-mlops-drift--retraining)
12. [State & persistence model](#12-state--persistence-model)
13. [Deployment topology](#13-deployment-topology)
14. [Framework portability (LangGraph → MAF)](#14-framework-portability-langgraph--maf)
15. [Proposed repository structure](#15-proposed-repository-structure)

**Part II — Architecture review & gap remediation**

16. [NFR → mechanism traceability](#16-nfr--mechanism-traceability)
17. [LLM reasoning boundary, determinism & safety](#17-llm-reasoning-boundary-determinism--safety)
18. [Security architecture & agent identity](#18-security-architecture--agent-identity)
19. [Failure modes & resilience](#19-failure-modes--resilience)
20. [Decision provenance & schema versioning](#20-decision-provenance--schema-versioning)
21. [Concurrency, scaling & performance budget](#21-concurrency-scaling--performance-budget)
22. [CI/CD & evaluation-as-quality-gate](#22-cicd--evaluation-as-quality-gate)
23. [Architecture Decision Records (seed)](#23-architecture-decision-records-seed)

---

## 1. Architectural principles

The architecture is governed by five principles. Every design decision below traces back to one of them.

| # | Principle | What it means in practice |
|---|---|---|
| **P1** | **ML-as-Tool, never LLM-as-predictor** | Trained models (Chronos-2, XGBoost, Isolation Forest) sit behind a typed tool interface. LLMs only *reason over* and *narrate* model outputs — they never generate predictions. The tool boundary is the hard line that enforces this. |
| **P2** | **Uncertainty & explainability flow through, never stripped** | Quantile spreads, calibrated probabilities, and SHAP attributions are first-class fields on every typed contract from the model layer up to the Supervisor and into the audit log. |
| **P3** | **Governance is a layer, not a feature** | Escalation tiering, HITL gating, audit logging, circuit breakers, and rollback are a distinct cross-cutting module with its own contracts — not scattered `if` statements inside agents. |
| **P4** | **Separation of concerns by dependency direction** | Domain contracts depend on nothing. Agents depend on contracts + tool interfaces (not implementations). Infrastructure (models, DBs, APIs) implements interfaces. Dependencies point *inward* (hexagonal / ports-and-adapters). |
| **P5** | **Framework-portable core** | Agent *reasoning logic* is framework-agnostic; LangGraph (and the MAF port) are adapters at the orchestration edge. Swapping the orchestrator must not touch domain or tool code. |

### Ports-and-adapters at a glance

```mermaid
flowchart LR
    subgraph Core["Domain Core (no external deps)"]
        C["Typed Contracts<br/>(Pydantic schemas)"]
        R["Reasoning / Policies<br/>(escalation, scoring)"]
    end

    subgraph Ports["Ports (interfaces)"]
        TP["ToolPort"]
        FP["FeaturePort"]
        AP["AuditPort"]
        HP["HITLPort"]
    end

    subgraph Adapters["Adapters (infrastructure)"]
        ML["ML inference adapters<br/>Chronos-2 / XGBoost / IsoForest"]
        FE["Feast adapter"]
        AU["MLflow audit adapter"]
        HI["FastAPI HITL adapter"]
        OR["LangGraph / MAF orchestrator"]
    end

    R --> C
    Ports --> Core
    TP -.implemented by.-> ML
    FP -.implemented by.-> FE
    AP -.implemented by.-> AU
    HP -.implemented by.-> HI
    OR -->|drives| R
```

---


## 2. System context (C4 L1)

```mermaid
flowchart TB
    Planner(["Supply Chain Planner<br/>(human reviewer)"])
    ERP(["External ERP / WMS<br/>(via webhook)"])

    subgraph System["Supply Chain Resilience Co-Pilot"]
        APP["Multi-Agent Decision System<br/>+ ML serving + governance"]
    end

    FRED[("FRED API<br/>macro series")]
    MT[("MarineTraffic AIS<br/>port congestion")]
    BTS[("BTS FAF / Census<br/>freight data")]
    M5[("M5 / Favorita<br/>demand datasets")]
    LLM[("Azure OpenAI<br/>via Azure AI Foundry")]

    FRED --> APP
    MT --> APP
    BTS --> APP
    M5 --> APP
    APP <--> LLM
    APP -->|ReviewRequest brief| Planner
    Planner -->|approve / override| APP
    APP -->|recommended actions| ERP
```

**Boundaries:** the system *ingests* external data and *calls* the LLM as a reasoning utility; it *emits* structured action artefacts to humans (HITL) and to downstream systems (webhook). There is no conversational interface and no autonomous write-back to ERP without approval (PRD §12).

---

## 3. Layered / modular view

Six horizontal layers, each with a single responsibility and a strict downward dependency rule. A layer may only depend on the layer(s) below it through published contracts.

```mermaid
flowchart TB
    subgraph L6["6 · Interface Layer"]
        FA["FastAPI: actions, inference, HITL webhook"]
    end
    subgraph L5["5 · Governance Layer (cross-cutting)"]
        GOV["Escalation · HITL gate · Audit · Circuit breaker · Rollback registry"]
    end
    subgraph L4["4 · Orchestration Layer"]
        ORC["LangGraph StateGraph (Supervisor + 4 specialists)<br/>· MAF port adapter"]
    end
    subgraph L3["3 · Tool / Capability Layer"]
        TOOL["Typed tool interfaces (ToolPort) + LLM reasoning utilities"]
    end
    subgraph L2["2 · ML Serving Layer"]
        MLS["Chronos-2 · XGBoost+SHAP · IsoForest · MAPIE · regime classifier"]
    end
    subgraph L1["1 · Data & Feature Layer"]
        DATA["Airflow DAGs · Feast (online/offline) · PostgreSQL"]
    end

    L6 --> L5 --> L4 --> L3 --> L2 --> L1
    L5 -.observes/gates.-> L4
```

| Layer | Concern | Key tech | Must NOT |
|---|---|---|---|
| 1 — Data & Feature | Ingest, validate, materialise features | `scrc.data`, Airflow, Feast, PostgreSQL | know about agents or models |
| 2 — ML Serving | Produce calibrated predictions + attributions | Chronos-2, XGBoost, IsoForest, SHAP, MAPIE | embed business/escalation logic |
| 3 — Tool / Capability | Expose models & LLM as typed tools | Pydantic contracts, LangChain | hold orchestration state |
| 4 — Orchestration | Sequence agents, fan-out/synthesise | LangGraph `StateGraph` | call raw model code directly |
| 5 — Governance | Tier, gate, audit, rollback | MLflow, policy config in PG | live inside agent business logic |
| 6 — Interface | Expose REST + HITL surface | FastAPI | contain decision logic |

---

## 4. Module boundaries & separation of concerns

Each agent and each model is an independently testable module behind a contract. The **typed contract** is the unit of separation: changing a model implementation cannot break an agent as long as the contract holds.

```mermaid
flowchart LR
    subgraph contracts["module: contracts (depends on nothing)"]
        QF["QuantileForecastResult"]
        AR["AnomalyResult"]
        MS["MacroSignals / RegimeLabel"]
        SR["StockoutRiskResult"]
        DEC["SupervisorDecision / ReviewRequest"]
    end

    subgraph agents["module: agents"]
        DA["demand_agent"]
        LA["logistics_agent"]
        MA["macro_agent"]
        SA["stockout_agent"]
        SUP["supervisor"]
    end

    subgraph tools["module: tools (ports)"]
        T1["chronos_forecast()"]
        T2["detect_freight_anomaly()"]
        T3["get_fred_series()"]
        T4["classify_stockout_risk()"]
    end

    DA --> T1 --> QF
    LA --> T2 --> AR
    MA --> T3 --> MS
    SA --> T4 --> SR
    DA & LA & MA & SA --> SUP --> DEC
```

**Rules enforced by this boundary:**
- Agents import *contracts* and *tool interfaces* only — never concrete model classes.
- The Stockout agent consumes the *outputs* (contracts) of the other three, not their internals → conjoint reasoning without coupling.
- The Supervisor consumes only the four typed agent outputs → it can be tested with fixtures, no models needed.

---

## 5. Agent orchestration (LangGraph StateGraph)

The Supervisor fans out to the three independent signal agents in parallel, joins, then runs the Stockout classifier (which needs all three), then synthesises and routes.

```mermaid
flowchart TB
    START(("Start")) --> SUP_IN["Supervisor: ingest request<br/>build run context"]
    SUP_IN --> FANOUT{{fan-out}}

    FANOUT --> DEM["Demand Agent<br/>chronos_forecast()"]
    FANOUT --> LOG["Logistics Agent<br/>port + freight anomaly"]
    FANOUT --> MAC["Macro Agent<br/>FRED + regime"]

    DEM --> JOIN{{join}}
    LOG --> JOIN
    MAC --> JOIN

    JOIN --> STK["Stockout Agent<br/>joint feature vector → XGBoost + SHAP"]
    STK --> SYN["Supervisor: synthesise<br/>score risk tier"]

    SYN --> TIER{escalation tier?}
    TIER -->|ROUTINE / MONITOR| AUTO["Execute autonomously<br/>register rollback if reversible"]
    TIER -->|REVIEW / CRITICAL| INT["interrupt()<br/>emit ReviewRequest"]

    AUTO --> AUDIT["Log decision → MLflow"]
    INT --> AUDIT
    AUDIT --> DONE(("End"))
```

**Resilience rule (P3):** any specialist that fails to return a typed result before timeout emits an escalation token. At the join, a missing output is treated as **maximum uncertainty** — the Supervisor routes conservatively (never hallucinates a value). This is implemented in the join node, not inside each agent.

---

## 6. End-to-end decision flow

The full lifecycle from data freshness to actioned recommendation, spanning all six layers. Note the single OTEL trace that threads the whole cycle.

```mermaid
flowchart LR
    A["Airflow DAG completes<br/>features materialised in Feast"] --> B["Trigger / scheduled<br/>agent run"]
    B --> C["Agents read online features<br/>(Feast)"]
    C --> D["ML serving: predictions<br/>+ uncertainty + SHAP"]
    D --> E["Supervisor synthesis<br/>+ tiering"]
    E --> F{autonomous<br/>or escalate?}
    F -->|auto| G["Action + rollback registration"]
    F -->|escalate| H["HITL via FastAPI webhook"]
    G --> I["MLflow audit run"]
    H --> I
    I --> J["Outcome logged next cycle<br/>(Forecast Value Added)"]

    classDef trace fill:#eef,stroke:#88a;
    class A,B,C,D,E,F,G,H,I trace;
```

> Every node above emits an OpenTelemetry span ingested by **Opik**; the complete cycle is one trace, and the same captured traces feed the evaluation suite.

---

## 7. HITL escalation sequence

`interrupt()` is code-enforced (P1/P3): the graph durably pauses, the planner receives a *fully explained brief* (not a bare prompt), and approval/override resumes the exact checkpoint.

```mermaid
sequenceDiagram
    autonumber
    participant SUP as Supervisor (LangGraph)
    participant GOV as Governance / interrupt()
    participant API as FastAPI HITL webhook
    participant PL as Planner
    participant ML as MLflow audit
    participant CP as Checkpointer (PostgreSQL)

    SUP->>GOV: decision tier = REVIEW/CRITICAL
    GOV->>CP: persist checkpoint (durable pause)
    GOV->>API: emit ReviewRequest<br/>(SHAP brief + counterfactual + signals)
    API-->>PL: notify (≤30s target)
    Note over PL: reviews calibrated prob,<br/>quantile spread, top SHAP features
    PL->>API: approve / override (+ reason)
    API->>GOV: resume(decision)
    GOV->>CP: load checkpoint
    GOV->>SUP: Command(resume) → continue graph
    SUP->>ML: log inputs, outputs, reviewer id, outcome
```

### Escalation policy (governance module, configurable in PostgreSQL)

```mermaid
stateDiagram-v2
    [*] --> Evaluate
    Evaluate --> ROUTINE: prob < 0.30 & narrow interval & no anomaly
    Evaluate --> MONITOR: prob 0.30–0.55 OR single anomaly
    Evaluate --> REVIEW: prob 0.55–0.75 OR multi-signal conjunction
    Evaluate --> CRITICAL: prob > 0.75 OR (anomaly + macro shock + high uncertainty)
    ROUTINE --> Autonomous
    MONITOR --> Autonomous
    REVIEW --> HITL
    CRITICAL --> HITL
    Autonomous --> [*]
    HITL --> [*]
```

---

## 8. ML-as-Tool serving layer

Each model is wrapped as an MLflow-registered, independently deployed inference service exposing a typed tool. **Only registry-promoted versions are callable** — the tool layer resolves the production alias at call time (P1).

```mermaid
flowchart TB
    subgraph ToolIF["Tool interface (ToolPort) — typed in/out"]
        direction LR
        t1["chronos_forecast()"]
        t2["detect_freight_anomaly()"]
        t3["classify_stockout_risk()"]
    end

    subgraph Serving["Inference services (containers)"]
        c1["Chronos-2 service<br/>(Azure AI Foundry endpoint / AutoGluon container)"]
        c2["XGBoost + SHAP + MAPIE service"]
        c3["Isolation Forest + KernelSHAP service"]
    end

    REG[("MLflow Model Registry<br/>prod alias resolution")]
    FEAST[("Feast online store")]

    t1 --> c1
    t2 --> c3
    t3 --> c2
    c1 & c2 & c3 -->|resolve prod version| REG
    c1 & c2 & c3 -->|read features| FEAST
```

**Output contracts always carry uncertainty + attribution (P2):**

| Tool | Returns | Uncertainty field | Explainability field |
|---|---|---|---|
| `chronos_forecast` | `QuantileForecastResult` | `p10/p50/p90`, `interval_width` | `covariate_flags_used` |
| `classify_stockout_risk` | `StockoutRiskResult` | `calibrated` prob, `confidence_tier` (MAPIE) | `shap_values[]`, `plain_language_brief` |
| `detect_freight_anomaly` | `AnomalyResult` | `anomaly_score` | `top_features[{feature, shap_value}]` |

> **Boundary note:** Azure AutoML benchmarks candidate models during selection only — it is never in this inference path. Chronos-2 is open-source (Apache 2.0), served on Azure, so the runtime stack stays single-cloud.

---

## 9. Data pipeline & feature store

Airflow owns ingestion and validation; Feast owns the read contract for both training (offline) and inference (online). Agents never call source APIs directly — they read materialised features, decoupling inference latency from API rate limits.

```mermaid
flowchart LR
    subgraph Sources
        S1["FRED API"]
        S2["MarineTraffic AIS"]
        S3["BTS FAF / Census"]
        S4["M5 / Favorita"]
    end

    subgraph Airflow["Airflow DAGs"]
        D1["ingest → validate schema → transform"]
        D2["feature engineering"]
        D3["materialise"]
    end

    subgraph Feast
        OFF[("Offline store<br/>(training)")]
        ON[("Online store<br/>(low-latency inference)")]
    end

    S1 & S2 & S3 & S4 --> D1 --> D2 --> D3
    D3 --> OFF
    D3 --> ON
    ON -->|read at inference| ML["ML serving / agents"]
    OFF -->|read at training| TR["Training pipelines"]
```

**Rate-limit handling (PRD §13):** AIS responses are cached in Feast with a 4-hour TTL; BTS FAF is the primary logistics signal, AIS is enrichment. All of this lives in the data layer — invisible to agents.

---

## 10. Observability, evaluation & governance

One emission standard (OTEL), two consumers split by concern: **Opik** for agent/LLM traces *and* evaluation; **Prometheus + Grafana** for infra/system metrics. **MLflow** is the durable decision audit log.

```mermaid
flowchart TB
    subgraph Emit["Instrumented components"]
        AG["Agents / tools"]
        MLI["ML inference"]
        AF["Airflow"]
        API["FastAPI"]
    end

    OTEL["OpenTelemetry Collector"]
    AG & MLI & AF --> OTEL

    OTEL -->|spans| OPIK["Opik<br/>trace UI + evaluation suite"]
    AG & MLI & AF & API -->|metrics| PROM["Prometheus"] --> GRAF["Grafana<br/>(infra dashboards)"]

    AG -->|decision runs| MLF["MLflow<br/>audit trail + registry"]

    OPIK -->|datasets / experiments| EVAL["Eval harness:<br/>routing reproducibility ≥95%<br/>disruption replay"]
```

| Concern | Owner | Separation rationale |
|---|---|---|
| Agent/LLM tracing + evaluation | **Opik** | Unifies "what did the agent do" with "was it correct" in one open-source tool (replaces Jaeger + LangSmith) |
| Infra/system metrics | Prometheus + Grafana | Throughput, latency, drift, pipeline health — Opik does not cover infra |
| Decision audit trail | MLflow | Reproducible, queryable Forecast Value Added record (SOC2-style) |
| Policy & autonomy thresholds | PostgreSQL (read by governance) | Config, not code — tunable per SKU/$ impact/reversibility |

---

## 11. MLOps: drift & retraining

The Monitoring Agent applies the *same* HITL governance pattern to the model lifecycle that the Supervisor applies to operations — promotion is gated exactly like a CRITICAL action.

```mermaid
flowchart TB
    M1["Monitoring Agent (scheduled)"] --> M2["Compute rolling MAPE / F1<br/>vs observed outcomes"]
    M2 --> M3{degraded beyond<br/>threshold?}
    M3 -->|no| M1
    M3 -->|yes| M4["Optuna HPO on current window"]
    M4 --> M5["Evaluate vs held-out set"]
    M5 --> M6{beats prod on<br/>all metrics by margin?}
    M6 -->|yes| M7["Auto-register + promote"]
    M6 -->|no| M8["HITL approval gate"]
    M7 --> MLF[("MLflow Registry")]
    M8 --> MLF
```

---

## 12. State & persistence model

A single PostgreSQL instance backs four concerns, kept in separate schemas to preserve separation while sharing operational simplicity (PRD §4.3).

```mermaid
erDiagram
    GRAPH_CHECKPOINT ||--o{ DECISION_RUN : "produces"
    DECISION_RUN ||--o{ AGENT_OUTPUT : "aggregates"
    DECISION_RUN ||--o| REVIEW_REQUEST : "may raise"
    DECISION_RUN ||--o| ROLLBACK_ENTRY : "may register"
    POLICY_CONFIG ||--o{ DECISION_RUN : "governs"

    GRAPH_CHECKPOINT {
        string thread_id PK
        json state
        timestamp updated_at
    }
    DECISION_RUN {
        string run_id PK
        string tier
        float stockout_prob
        string action
        bool autonomous
    }
    AGENT_OUTPUT {
        string id PK
        string agent
        json payload
        float confidence
    }
    REVIEW_REQUEST {
        string id PK
        string reviewer_id
        string outcome
        timestamp responded_at
    }
    ROLLBACK_ENTRY {
        string id PK
        string action_ref
        timestamp window_expires
        bool rolled_back
    }
    POLICY_CONFIG {
        string class PK
        float autonomy_threshold
        string reversibility
    }
```

| Schema | Used by | Concern |
|---|---|---|
| `langgraph_checkpoints` | Orchestration | Durable graph state / resume |
| `mlflow` | Governance | Experiment + audit metadata |
| `airflow` | Data layer | Scheduler metadata |
| `feast` | Data layer | Offline + online feature stores |

---

## 13. Deployment topology

12 services via a single `docker-compose up`; Azure-optimised but cloud-agnostic. Grouped by the layer each service serves.

```mermaid
flowchart TB
    subgraph Interface
        FAPI["FastAPI"]
    end
    subgraph Orchestration
        LG["LangGraph agent runtime"]
    end
    subgraph Inference
        CH["Chronos-2 container"]
        XG["XGBoost + IsoForest container"]
    end
    subgraph Pipeline
        AW["Airflow web"]
        AS["Airflow scheduler"]
        AWK["Airflow worker"]
    end
    subgraph MLLifecycle["ML Lifecycle"]
        MLF["MLflow"]
        FE["Feast server"]
    end
    subgraph Observability
        OT["OTEL Collector"]
        OP["Opik"]
        PR["Prometheus"]
        GR["Grafana"]
    end
    subgraph Data
        PG[("PostgreSQL")]
    end

    FAPI --> LG --> CH & XG
    LG --> FE --> PG
    AW & AS & AWK --> PG
    MLF --> PG
    LG --> OT --> OP
    LG --> PR --> GR
```

**Cloud mapping (production):** LangGraph + FastAPI → Azure Container Apps · Chronos-2 **and** LLM (Azure OpenAI, primary + secondary deployment) → Azure AI Foundry · PostgreSQL → Azure DB for PostgreSQL Flexible Server · OTEL → Azure Monitor + Opik · artefacts → Azure Blob.

---

## 14. Framework portability (LangGraph → MAF)

P5 in action: the orchestrator is an adapter. The domain core, contracts, tools, and governance are reused unchanged; only the orchestration adapter swaps.

```mermaid
flowchart LR
    subgraph Reused["Reused unchanged"]
        CORE["contracts + reasoning policies"]
        TOOLS["typed tools / ports"]
        GOV["governance (escalation, audit, HITL)"]
    end

    subgraph LGAdapter["LangGraph adapter"]
        LG["StateGraph · interrupt() · Command · checkpointer"]
    end
    subgraph MAFAdapter["MAF 1.0 adapter (port)"]
        MAF["WorkflowBuilder · Executors · DurableWorkflows · Magentic-One"]
    end

    CORE --- TOOLS --- GOV
    GOV --> LG
    GOV --> MAF
```

| LangGraph concept | MAF 1.0 equivalent | Notes |
|---|---|---|
| `StateGraph` nodes | `Executors` | one executor per agent |
| `interrupt()` | `DurableWorkflows` pause | both code-enforced HITL |
| `Command` routing | `WorkflowBuilder` edges | conditional routing |
| Checkpointer (PG) | Durable state store | same PostgreSQL backing |

The port is scoped to the Demand + Supervisor sub-graph if full parity proves costly (PRD §13) — and the *differences themselves* are a teaching deliverable (module 6).

---

## 15. Proposed repository structure

A modular monorepo mirroring the layers under a single importable namespace, `scrc`. Dependency direction is enforced by package boundaries (P4): `contracts` ← `tools` ← `agents` ← `orchestration`; `governance` and `observability` are cross-cutting. The rule is not just documented — it is **machine-enforced by `import-linter`** ([.importlinter](../.importlinter)), so a boundary violation fails CI.

```
supplychain_resilience_copilot/
├── docs/                          # PRD, this architecture doc, ADRs, curriculum
├── packages/
│   └── scrc/                      # single importable namespace (e.g. scrc.contracts)
│       ├── contracts/             # Pydantic typed schemas — depends on nothing
│       ├── data/                  # Layer 1: ingestion, transforms, validation
│       ├── ml/                    # training, registration, serving wrappers
│       │   ├── forecasting/       # Chronos-2
│       │   ├── classification/    # XGBoost + SHAP + MAPIE
│       │   └── anomaly/           # Isolation Forest + KernelSHAP
│       ├── tools/                 # ToolPort interfaces + adapters to ml/
│       ├── agents/                # demand/logistics/macro/stockout/supervisor.py
│       ├── orchestration/         # LangGraph graph (state.py, graph.py); maf/ port later
│       ├── governance/            # escalation, HITL, audit, rollback, breakers
│       ├── observability/         # OTEL setup, Opik + Prometheus exporters
│       └── api/                   # FastAPI: actions, inference, HITL webhook
├── pipelines/
│   ├── airflow_dags/              # ingestion + feature DAGs
│   └── feast/                     # feature definitions, repo config
├── eval/                          # Opik datasets, experiments, disruption replay
├── deploy/
│   ├── docker-compose.yml         # service stack (count to reconcile — see §13 note)
│   ├── otel-collector.yaml        # OTEL emission config
│   ├── prometheus.yml             # infra metrics scrape config
│   └── azure/                     # Container Apps / Foundry manifests
├── tests/                         # mirrors packages/ (tests/contracts/ ...)
├── pyproject.toml                 # deps + ruff/mypy/pytest config
├── .importlinter                  # enforces the layer rules above
├── Makefile                       # setup · check · up/down · data-init
└── .env.example
```

> **Namespace note:** §15 originally showed `packages/<layer>/`; the implementation nests these under `packages/scrc/` so imports read `scrc.contracts`, `scrc.tools`, etc. — collision-safe and clean for `import-linter` contracts.

**Test boundary:** each `scrc.*` module is unit-testable in isolation against contracts with fixtures; agents test against fake tools; the Supervisor tests against fake agent outputs; integration tests run the full graph against the Compose stack.

---

# Part II — Architecture review & gap remediation

> A lead-architect pass over Part I identified seven gaps between a *functionally correct* design and a *production-trustworthy, governed* one. This part closes them. Each section states the gap, the design, and where in the layering it lives.

---

## 16. NFR → mechanism traceability

Part I described structure but did not trace the PRD's success criteria to the mechanism that delivers them. Every non-functional requirement now has a single owning mechanism and an enforcement point.

| NFR (PRD §11 / §13) | Target | Mechanism | Enforced in |
|---|---|---|---|
| Routing reproducibility | ≥ 95% identical-input consistency | Deterministic rule-based tiering (LLM does not assign tier) + temperature 0 + structured output + input-hash cache | §17, governance |
| HITL notification latency | ≤ 30 s | Async webhook, durable checkpoint pause, no blocking LLM call on the notify path | §21, §7 |
| Stockout AUC / ECE | ≥ 0.85 / ≤ 0.05 | Eval-as-gate blocks release on regression | §22 |
| Calibrated confidence | isotonic + MAPIE | Calibration in ML serving; tier reads calibrated prob only | §8 |
| Cold start | one `docker-compose up` | Healthchecks + dependency ordering in Compose | §13 |
| Auditability | every decision queryable | Provenance tuple + append-only MLflow; no-audit-no-autonomy invariant | §19, §20 |
| Least-privilege identity | per-agent scoped | Service identity + tool-scoped authz | §18 |

---

## 17. LLM reasoning boundary, determinism & safety

**Gap (G1):** Part I asserted ≥95% routing reproducibility and "LLM never predicts" but gave no enforcing mechanism, and ignored prompt injection from untrusted external data.

### The boundary: what the LLM may and may not do

```mermaid
flowchart TB
    subgraph Deterministic["Deterministic code (no LLM)"]
        TIER["Tier assignment<br/>(scoring matrix on calibrated prob + signals)"]
        GATE["Autonomy gate / HITL routing"]
        PRED["All numeric predictions<br/>(Chronos-2 / XGBoost / IsoForest)"]
    end
    subgraph LLMzone["LLM (constrained)"]
        RANK["Rank/justify candidate actions"]
        BRIEF["SHAP → plain-language brief"]
        SYNTH["Narrative synthesis for reviewer"]
    end
    PRED --> TIER --> GATE
    PRED --> RANK
    RANK -. proposes, never decides tier .-> TIER
    BRIEF --> SYNTH
```

**Invariant:** the *tier and the autonomy decision are pure functions of numeric inputs* — the LLM cannot change a tier. The LLM ranks actions and writes explanations. This is what makes reproducibility achievable: the gate is code, not generation.

### Determinism controls

- **Temperature 0** (or top-1 greedy) for Supervisor synthesis and the Stockout brief.
- **Structured output / strict function-calling** against Pydantic schemas — routing fields are enums; malformed output is rejected and retried, never coerced.
- **Input-hash memoisation**: identical input hash → cached decision (also powers reproducibility eval).
- **CRITICAL tier is fully rule-based** (PRD §13 mitigation) — no LLM in the path for the highest-impact decisions.

### LLM safety & injection defense

```mermaid
flowchart LR
    EXT["Untrusted external data<br/>(AIS text, news, FRED labels)"] --> FEAT["Becomes numeric features<br/>(data layer)"]
    FEAT --> TOOLR["Typed tool results"]
    TOOLR --> PROMPT["LLM prompt:<br/>structured data as DATA, not instructions"]
    PROMPT --> OUT["Schema-constrained output"]
    OUT --> VALIDATE["Validate + reject on schema violation"]
```

- External data reaches the LLM only as **structured, typed values** — never as free-form instruction text. This collapses most of the prompt-injection surface.
- **Deployment failover** via the LangChain abstraction: a primary Azure OpenAI deployment in Azure AI Foundry → a **secondary Azure OpenAI deployment** (alternate region/model) on error/timeout — single-provider, no second cloud. If both are unavailable → treat as max uncertainty → conservative escalation (never silent autonomy). The provider abstraction is retained so the backbone remains swappable.
- LLM output is validated against schema before it can influence any action; the brief is advisory text attached to a decision already gated by code.

---

## 18. Security architecture & agent identity

**Gap (G2):** PRD §7.4 mandates least-privilege agent identity and an append-only audit log; Part I modelled neither, nor any trust zones or HITL authz.

### Trust zones

```mermaid
flowchart TB
    subgraph Public["Public / ingress zone"]
        PL["Planner browser/app"]
        ERPx["External ERP"]
    end
    subgraph Edge["Edge zone (authenticated)"]
        FAPI["FastAPI gateway<br/>authN + RBAC"]
    end
    subgraph App["Application zone (private)"]
        LG["Agent runtime"]
        GOV["Governance"]
    end
    subgraph DataZ["Data zone (no ingress)"]
        PG[("PostgreSQL")]
        FE["Feast"]
        ML["Inference services"]
        MLF["MLflow audit (append-only)"]
    end
    subgraph Egress["Controlled egress"]
        LLM["Azure OpenAI (Azure AI Foundry)"]
        APIs["FRED / AIS / BTS"]
    end

    PL --> FAPI
    ERPx --> FAPI
    FAPI --> LG --> GOV
    LG --> ML
    GOV --> MLF
    LG --> FE --> PG
    LG --> LLM
    APIs -.->|ingestion only via Airflow| DataZ
```

### Controls

| Concern | Control |
|---|---|
| **Agent identity** | Each agent runs under a named service identity scoped to *only* its tool endpoints (least privilege). The Stockout agent cannot call the Chronos endpoint directly; it consumes the Demand agent's contract. |
| **Audit integrity** | Only the **governance layer** writes to the MLflow audit store; it is **append-only / WORM** — no agent has write access. Tampering is out of the agent threat surface. |
| **Secrets** | Local: single `.env` (git-ignored). Cloud: Azure Key Vault + managed identities; no secrets baked into images. |
| **HITL authz** | Approve/override endpoints require authenticated planner identity with RBAC; reviewer ID is captured in the audit run (non-repudiation). |
| **Egress control** | LLM and source-API egress is allow-listed; the data zone has no inbound path except via Airflow ingestion. |

### Threat model (STRIDE-lite, agentic-specific)

| Threat | Vector | Mitigation |
|---|---|---|
| Prompt injection | External text in features | Structured-data-only prompts (§17) |
| Excessive autonomy | Mis-tiered high-impact action | Deterministic CRITICAL routing + autonomy thresholds + rollback registry |
| Tool abuse | Agent calls out-of-scope endpoint | Scoped service identity |
| Audit tampering | Agent rewrites history | Append-only WORM, no agent write access |
| Model poisoning | Bad training data promoted | Eval-as-gate + HITL promotion (§22, §11) |

---

## 19. Failure modes & resilience

**Gap (G3):** resilience was one sentence. A governed system needs an explicit failure taxonomy and one non-negotiable invariant.

> **Invariant — *no audit, no autonomy*.** If a decision cannot be durably persisted to the audit store, the system **must not act autonomously**; it degrades to escalate-only. Autonomy is a privilege contingent on traceability.

```mermaid
flowchart TB
    OK["Normal: act per tier"] --> CHK{dependency healthy?}
    CHK -->|features stale| DEG1["Use last-good + freshness flag;<br/>if too stale → escalate"]
    CHK -->|model endpoint down| DEG2["Circuit breaker →<br/>missing = max uncertainty → escalate"]
    CHK -->|LLM both providers down| DEG3["Rule-based routing only;<br/>brief = raw SHAP table"]
    CHK -->|audit store down| DEG4["BLOCK autonomy →<br/>escalate-only + local queue"]
    CHK -->|checkpointer down| DEG5["No durable pause →<br/>do not act; reject run"]
```

| Component | Failure | Detection | Response |
|---|---|---|---|
| Feast online | stale / unreachable | freshness lag metric | last-good value flagged; escalate if beyond TTL bound |
| Inference service | timeout / 5xx | circuit breaker + OTEL error span | treat output as max uncertainty → conservative route |
| LLM deployment | error / rate limit | deployment error rate | failover to secondary Azure OpenAI deployment (alt region/model); both down → rule-based routing |
| MLflow audit | unreachable | write healthcheck | **block autonomous action**, escalate-only, buffer to local WAL |
| Checkpointer (PG) | unreachable | connection healthcheck | reject new runs (cannot guarantee durable HITL) |
| HITL webhook | delivery fails | ack timeout | retry with idempotency key; fallback notification channel |
| Partial fan-out | 1 of 3 signals missing | join node | proceed with missing = max uncertainty |

**Idempotency:** every decision run and every emitted action carries an idempotency key; rollback entries are registered *before* execution, so a retried run cannot double-act.

---

## 20. Decision provenance & schema versioning

**Gap (G4):** reproducibility and audit are impossible without recording exactly *what produced* a decision; contracts had no evolution strategy.

### Provenance tuple — stamped on every `DECISION_RUN`

```mermaid
flowchart LR
    DEC["DecisionProvenance"] --> A["model_versions{chronos, xgb, isoforest}"]
    DEC --> B["feature_schema_version"]
    DEC --> C["policy_config_version"]
    DEC --> D["prompt_template_version"]
    DEC --> E["llm_model_id"]
    DEC --> F["code_git_sha"]
    DEC --> G["input_hash"]
```

This tuple makes a decision **reconstructable**: given the same input hash and identical versions, the deterministic path reproduces the same tier — the foundation of the ≥95% reproducibility test and of audit defensibility.

### Contract / schema evolution

- Pydantic contracts carry a `schema_version`; evolution is **additive** (new optional fields) within a major version.
- **Consumer-driven contract tests** in CI: agents publish the contract shape they depend on; a producing model change that breaks a consumer fails the build (§22).
- Feature schemas are versioned in Feast (PRD §8.1); a feature schema bump increments `feature_schema_version` in provenance.
- Model output schema and registry version are bound: promoting a model that changes its output contract requires a contract-version bump and consumer re-validation.

---

## 21. Concurrency, scaling & performance budget

**Gap (G5):** no scaling model or latency budget supported the stated targets.

```mermaid
flowchart LR
    Q["Decision requests<br/>(Airflow batch cadence bounds rate)"] --> LB["Stateless agent runtime<br/>(state in checkpointer)"]
    LB --> R1["runtime replica 1"]
    LB --> R2["runtime replica 2"]
    LB --> R3["runtime replica N"]
    R1 & R2 & R3 --> INF["Inference services<br/>(scale independently)"]
    R1 & R2 & R3 --> HITLQ["HITL queue (durable)"]
```

- **Stateless runtime:** all run state lives in the PostgreSQL checkpointer, so agent runtime scales horizontally; a replica can resume any interrupted run.
- **Independent runs** scale out; **within a run**, the three signal agents fan out concurrently (§5).
- **Inference autoscaling** is decoupled — Chronos (Foundry endpoint) and the sklearn container scale on their own latency SLOs.
- **Backpressure:** batch ingestion cadence bounds inbound load; HITL is a durable queue, not a blocking call.

### Latency budget (illustrative, to validate against SLOs)

| Stage | Budget |
|---|---|
| Feature read (Feast online) | ≤ 50 ms |
| 3× parallel inference (max) | ≤ 800 ms |
| Stockout + SHAP | ≤ 400 ms |
| Supervisor synthesis (LLM, temp 0) | ≤ 2 s |
| Tiering + audit write | ≤ 200 ms |
| **→ Decision (auto path)** | **≤ ~3.5 s** |
| `interrupt()` → planner notify | **≤ 30 s** (PRD) — async, off critical path |

---

## 22. CI/CD & evaluation-as-quality-gate

**Gap (G6):** the disruption-replay and reproducibility tests existed but had no enforcement point. Evaluation must be a **release gate**, not a dashboard.

```mermaid
flowchart LR
    PR["PR / commit"] --> S1["lint · type · unit"]
    S1 --> S2["contract tests<br/>(consumer-driven)"]
    S2 --> S3["integration on ephemeral<br/>docker-compose"]
    S3 --> S4{"Eval gate (Opik)"}
    S4 -->|pass| S5["build + push images"]
    S4 -->|fail| BLOCK["block release"]
    S5 --> S6["deploy app"]
    MODELS["Model promotion<br/>(MLflow registry alias)"] -. independent of app deploy .-> S6
```

**Eval gate criteria (block on regression):**

| Check | Threshold |
|---|---|
| Supervisor routing reproducibility | ≥ 95% |
| Disruption replay (COVID, Suez '21, Red Sea '24) | no regression vs golden baseline |
| Stockout AUC / ECE | ≥ 0.85 / ≤ 0.05 |
| IsoForest F1 (injected anomalies) | ≥ 0.70 |

- Disruption scenarios are **versioned golden datasets** in `eval/`, replayed through the full graph; Opik experiments compare against the recorded baseline.
- **Model promotion is decoupled from app deploy:** a new model version flips the registry prod alias (§8) after passing its own eval — no app redeploy needed, and rollback is an alias flip.

---

## 23. Architecture Decision Records

The rationale behind these decisions is recorded as full ADRs in [`adr/`](adr/) (Nygard format — context, decision, consequences, alternatives). See the [ADR index](adr/README.md).

| ADR | Decision | Status |
|---|---|---|
| [0001](adr/0001-deterministic-tiering.md) | Tiering is deterministic code; the LLM never assigns a tier | Accepted |
| [0002](adr/0002-no-audit-no-autonomy.md) | No-audit-no-autonomy invariant | Accepted |
| [0003](adr/0003-hexagonal-orchestrator-adapter.md) | Hexagonal layering; orchestrator (LangGraph/MAF) is an adapter | Accepted |
| [0004](adr/0004-chronos2-over-timegen.md) | Chronos-2 retained (open-source, Azure-hosted) over TimeGEN-1 | Accepted |
| [0005](adr/0005-opik-observability-eval.md) | Opik for tracing + eval; Prometheus/Grafana for infra only | Accepted |
| [0006](adr/0006-single-postgres-schema-separated.md) | Single PostgreSQL, schema-separated, over per-service DBs | Accepted |
| [0007](adr/0007-eval-as-release-gate.md) | Eval-as-release-gate; model promotion decoupled from app deploy | Accepted |
| [0008](adr/0008-azure-openai-via-foundry.md) | Azure OpenAI via Azure AI Foundry; single-provider deployment failover | Accepted |

---

*Architecture v1.1 · Part I (design) + Part II (review & gap remediation) · derived from PRD v1.0 · all diagrams Mermaid · modular hexagonal design with layer-enforced separation of concerns.*
