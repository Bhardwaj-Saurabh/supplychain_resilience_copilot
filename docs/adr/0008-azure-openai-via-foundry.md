# ADR-0008: Azure OpenAI via Azure AI Foundry as the LLM backbone; single-provider failover

- **Status:** Accepted
- **Date:** 2026-05-30
- **Deciders:** Lead AI Architect
- **Related:** PRD §4.2, §9.2 · architecture.md §13, §17, §19 · ADR-0004

## Context

The LLM backbone reasons over model outputs, translates SHAP into briefs, and synthesises the Supervisor narrative — but never predicts (ADR-0001). The original PRD specified Azure OpenAI (GPT-4o) **primary** with **Anthropic Claude** as a cross-provider fallback, swappable via the LangChain provider abstraction.

Two facts reshape this:
- Chronos-2 is already served on an **Azure AI Foundry** managed endpoint (ADR-0004). Foundry can also host Azure OpenAI model deployments, so one platform can serve *both* the forecasting endpoint and the LLM backbone.
- A Claude fallback introduces a second provider (separate credentials, egress path, and cloud) on the LLM path — at odds with the single-cloud direction established for the rest of the stack.

The fallback choice is not cosmetic: it is the LLM branch of the resilience design (architecture.md §17, §19).

## Decision

We will use **Azure OpenAI (GPT-4o) accessed through Azure AI Foundry** as the LLM backbone, consolidating it onto the same platform that serves the Chronos-2 endpoint.

Failover will be to a **secondary Azure OpenAI deployment** within Foundry — an alternate region and/or model — **not** a second provider. The Anthropic Claude fallback is dropped. The LangChain provider abstraction is **retained** so the backbone remains swappable in principle (teaching value and future optionality), but the operational topology is single-provider, single-cloud. If both Azure OpenAI deployments are unavailable, the system treats the LLM as unavailable → max uncertainty → conservative escalation; note that tiering and `CRITICAL` routing are deterministic and do not require the LLM (ADR-0001).

## Consequences

### Positive
- Single provider and single cloud across forecasting and LLM; one set of credentials, one egress path, one platform to operate and teach.
- Resilience preserved via redundancy (region/model-level failover) without a second vendor.
- Tiering remains functional during a total LLM outage because it is deterministic (ADR-0001); only the narrative/brief degrades.

### Negative / costs
- **Correlated-failure risk:** a region-wide or account-wide Azure OpenAI outage can take both deployments down at once — a cross-provider fallback would have survived it. Accepted trade-off in favour of single-cloud simplicity.
- Loses the cross-framework/provider-independence demonstration that a Claude fallback offered.

### Neutral / follow-ups
- Configure primary and secondary deployments in distinct Azure regions to decorrelate failures as far as a single provider allows.
- `.env` / Key Vault drops Anthropic credentials; only Azure OpenAI / Foundry credentials remain on the LLM path.
- Because the LangChain abstraction stays, re-introducing a cross-provider fallback later is a configuration + new-ADR change, not a rewrite.

## Alternatives considered

- **Keep Anthropic Claude as cross-provider fallback** — rejected: maximises resilience against an Azure OpenAI outage and offers provider-independence teaching value, but introduces a second provider/cloud on the LLM path, against the single-cloud direction.
- **No fallback (single deployment)** — rejected: simplest, and deterministic tiering survives an LLM outage, but needlessly forgoes cheap region/model redundancy for the narrative path.
