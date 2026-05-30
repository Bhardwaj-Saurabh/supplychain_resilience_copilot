# Supply Chain Resilience Co-Pilot

A multi-agent supply chain decision-support system built on the **ML-as-Agent-Tool**
pattern: real trained models (Chronos-2 forecasting, XGBoost stockout classifier,
Isolation Forest anomaly detector) are served as tools that LangGraph agents call —
not LLMs role-playing as forecasters. Uncertainty and explainability flow through the
agents into uncertainty-aware escalation; every decision is audited and gated by
human-in-the-loop where it matters.

> **Status:** Phase 0–1 (foundations + typed contracts). See the roadmap below.

## Documentation

- **What & why:** [docs/supply_chain_copilot_PRD.md](docs/supply_chain_copilot_PRD.md), [docs/problem_statement.md](docs/problem_statement.md)
- **How (design):** [docs/architecture.md](docs/architecture.md)
- **Decisions:** [docs/adr/](docs/adr/)
- **Agent/dev guide:** [CLAUDE.md](CLAUDE.md)

## Getting started

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
make setup        # create venv + install project and dev tools
make check        # lint + type + import-boundaries + tests
```

Common targets (`make help` for the full list):

| Target | Purpose |
|---|---|
| `make fmt` | Auto-format (ruff) |
| `make lint` | Lint without changes |
| `make type` | Static typing (mypy, strict) |
| `make lint-imports` | Enforce layer boundaries (ADR-0003) |
| `make test` | Run the test suite |

## Repository layout

The codebase is a modular hexagonal monorepo; dependencies point inward toward
`contracts` and the rule is enforced by `import-linter` ([.importlinter](.importlinter)).

```
packages/scrc/
  contracts/       typed Pydantic schemas — depends on nothing
  ml/              model training, registration, serving wrappers
  tools/           typed tool interfaces (ports) + adapters to ml/
  agents/          5 agents — import contracts + tools only
  orchestration/   LangGraph graph; maf/ port adapter
  governance/      escalation, HITL, audit, rollback, circuit breakers
  observability/   OTEL setup, Opik + Prometheus exporters
  api/             FastAPI: actions, inference, HITL webhook
pipelines/         Airflow DAGs, Feast feature definitions
eval/              Opik datasets, experiments, disruption replay
deploy/            docker-compose.yml + Azure manifests
```

## Roadmap

| Phase | Scope | State |
|---|---|---|
| 0 | Repo scaffolding, tooling, layer enforcement, Compose skeleton | ✅ |
| 1 | Typed contracts (all schemas) | ✅ |
| 2 | Data layer (`scrc.data`), Feast definitions + Airflow DAGs (Module 1) | ✅ |
| 3 | ML serving (`scrc.ml`) + ML-as-Tool boundary (`scrc.tools`) (Module 2) | ✅ |
| 4 | Governance tiering, 5 agents, LangGraph graph (Module 3) | ✅ |
| 5 | Uncertainty + SHAP-to-brief + `interrupt()` HITL (Module 4) | ⬜ |
| 6 | Observability + governance (Module 5) | ⬜ |
| 7 | Microsoft Agent Framework port (Module 6) | ⬜ |
