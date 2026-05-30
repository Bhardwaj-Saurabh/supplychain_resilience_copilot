"""Azure OpenAI LLM adapter (ADR-0008).

Azure OpenAI via Azure AI Foundry, single provider. Failover is to a **secondary
Azure OpenAI deployment** (alternate region/model), not a second provider. If
both deployments fail the caller treats the LLM as unavailable and degrades to
the deterministic brief (architecture.md §17/§19) — tiering never depends on the
LLM (ADR-0001).

LangChain is imported lazily so ``scrc.llm`` imports without the ``llm`` extra;
unit tests use a fake ``LLMClient``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AzureDeployment:
    endpoint: str
    api_key: str
    deployment: str
    api_version: str = "2024-10-21"


class AzureOpenAIClient:
    """LLMClient backed by a primary + secondary Azure OpenAI deployment."""

    def __init__(
        self,
        primary: AzureDeployment,
        secondary: AzureDeployment | None = None,
        temperature: float = 0.0,
    ) -> None:
        # temperature 0 for reproducible narration (architecture.md §17).
        self._primary = self._build(primary, temperature)
        self._secondary = self._build(secondary, temperature) if secondary else None

    @staticmethod
    def _build(dep: AzureDeployment, temperature: float) -> object:
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(
            azure_endpoint=dep.endpoint,
            api_key=dep.api_key,
            azure_deployment=dep.deployment,
            api_version=dep.api_version,
            temperature=temperature,
        )

    def complete(self, system: str, user: str) -> str:
        messages = [("system", system), ("human", user)]
        try:
            return str(self._primary.invoke(messages).content)  # type: ignore[attr-defined]
        except Exception:
            if self._secondary is None:
                raise
            return str(self._secondary.invoke(messages).content)  # type: ignore[attr-defined]
