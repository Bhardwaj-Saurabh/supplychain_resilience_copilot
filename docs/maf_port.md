# Microsoft Agent Framework Port (Module 6)

The system's primary orchestrator is LangGraph ([scrc.orchestration.graph](../packages/scrc/orchestration/graph.py)). This module is the **framework-portability teaching deliverable**: the *same* agents are re-wired on Microsoft Agent Framework (MAF) 1.0, demonstrating that orchestration is an adapter, not the home of business logic (P5, [ADR-0003](adr/0003-hexagonal-orchestrator-adapter.md)).

## What is reused vs. ported

| Reused unchanged | Ported (adapter only) |
|---|---|
| `scrc.contracts`, `scrc.agents`, `scrc.tools`, `scrc.governance`, `scrc.llm`, `scrc.ml` | `scrc.orchestration.maf.workflow` ([source](../packages/scrc/orchestration/maf/workflow.py)) |

The five agents expose plain `run`/`synthesise` methods and import no orchestrator, so the MAF port wraps them in `Executor`s without touching them. The verified, orchestrator-free reference the port mirrors is [`run_pipeline`](../packages/scrc/orchestration/portable.py).

## Primitive mapping

| Concern | LangGraph | MAF 1.0 |
|---|---|---|
| Node | `StateGraph` node function | `Executor` + `@handler` (async) |
| Fan-out | `add_edge(START, n)` ×3 | `add_fan_out_edges(ingest, [...])` |
| Join | multiple incoming edges (runs once) | `add_fan_in_edges([...], stockout)` (delivers a `list`) |
| Shared state | `GraphState` TypedDict + reducers | `ctx.set_state` / `ctx.get_state` + typed messages |
| HITL | `interrupt()` + `Command(resume=...)` | `ctx.request_info(...)` + `@response_handler` |
| Durable state | `MemorySaver` / `PostgresSaver` | `DurableWorkflows` (Azure-backed) |
| Output | terminal node writes state | `ctx.yield_output(...)` |

## Topology (identical to the LangGraph graph)

```
ingest ─▶ demand ─┐
                  ┼─▶ stockout ─▶ supervisor ─▶ output
        logistics ┤
        macro ────┘
```

`ingest` stores the request in shared state and broadcasts it; the three specialists fan out in parallel; `stockout` joins (receiving the three typed outputs as a list) and reads the request back from state to assemble the joint feature vector; `supervisor` synthesises the decision and yields it. The audit gate and `interrupt()`-based HITL map onto `ctx.request_info` and are the documented extension point in the port.

## Running it

`agent-framework` pulls pre-release sub-packages, so it is **not** a locked extra (it would break `uv` resolution for the whole project). Install it on demand:

```bash
uv pip install "agent-framework>=1.0" --prerelease=allow
uv run pytest tests/orchestration/test_maf.py   # skipped until installed
```

Without the SDK, [test_maf.py](../tests/orchestration/test_maf.py) is skipped; the framework-agnostic behaviour it would exercise is fully covered by [test_portable.py](../tests/orchestration/test_portable.py).

## Documented differences / caveats

- **Message-passing vs shared state.** LangGraph merges partial state dicts; MAF passes messages along edges, so the request is carried in `ctx` shared state for downstream executors. This is the most visible structural difference.
- **Fan-in typing.** MAF delivers fan-in results as an untyped `list`; the port tags each specialist message (`("forecast", ...)`) and rebuilds a dict, whereas LangGraph keys are explicit in `GraphState`.
- **HITL.** Both are code-enforced, but LangGraph's `interrupt()` returns the resume value inline, while MAF splits request/response across `@handler`/`@response_handler`.
- **Verification.** The port targets the documented MAF 1.0 Python API (WorkflowBuilder/Executor); it is validated against the live SDK separately from CI, which runs only the orchestrator-free reference.
