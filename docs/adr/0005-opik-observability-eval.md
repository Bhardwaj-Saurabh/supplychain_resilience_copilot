# ADR-0005: Opik for agent tracing + evaluation; Prometheus/Grafana for infra only

- **Status:** Accepted
- **Date:** 2026-05-30
- **Deciders:** Lead AI Architect
- **Related:** PRD §4.2, §4.3 (observability rationale), §7 · architecture.md §10

## Context

The system needs three distinct observability capabilities: (1) agent/LLM **trace visualisation** — what each agent did, which tools it called, what the LLM produced; (2) **evaluation** — was the routing correct, did it reproduce, how does it behave on disruption-replay scenarios; and (3) **infra/system metrics** — throughput, latency, drift, pipeline health.

The original PRD draft used Jaeger (trace UI) + LangSmith (evaluation) + Prometheus/Grafana (metrics) — three tools across the agent-observability concern. Opik (Comet's open-source LLM observability and evaluation platform) covers both agent tracing and evaluation in one tool and ingests OpenTelemetry spans, but it does not cover infra/system metrics.

## Decision

We will use **Opik as the single tool for agent/LLM tracing *and* evaluation**, ingesting OpenTelemetry spans emitted by the agents, ML inference, and Airflow. Opik replaces both Jaeger and LangSmith. **Prometheus + Grafana are retained for infra/system metrics only** — the concern Opik does not address. The **OpenTelemetry Collector remains the emission standard**, decoupling instrumentation from the chosen backend. MLflow remains the durable decision audit log (a separate concern from observability).

## Consequences

### Positive
- One open-source tool unifies "what did the agent do" with "was it correct," and the same captured traces feed the evaluation suite (architecture.md §22).
- Fewer moving parts in the observability stack; no LangSmith (managed) dependency.
- OTEL emission means the agent-trace backend can be swapped again later without re-instrumenting code.

### Negative / costs
- Opik does not cover infra metrics, so the stack still runs Prometheus + Grafana — two observability tools remain, split cleanly by concern.
- Dependency on Opik's OTEL ingestion behaving as expected; this should be validated when the Compose stack is first wired (it is the load-bearing integration assumption).

### Neutral / follow-ups
- Disruption-replay golden datasets and reproducibility experiments live as Opik datasets/experiments in `eval/`.
- Cloud deployment uses Azure Monitor Application Insights alongside Opik (self-hosted or Comet-managed) for OTEL telemetry (PRD §9.2).

## Alternatives considered

- **Keep Jaeger + LangSmith + Prometheus/Grafana** — rejected: three tools for the agent-observability concern, including a managed (LangSmith) dependency, with tracing and evaluation split across tools.
- **Opik for everything, drop Prometheus/Grafana** — rejected: loses infra/system metric dashboards (drift, pipeline health, latency) that Opik does not provide.
