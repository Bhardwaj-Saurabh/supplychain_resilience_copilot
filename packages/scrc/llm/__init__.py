"""LLM capability layer: reasoning/narration utilities (never prediction, P1).

Depends on ``scrc.contracts`` only. The Azure adapter pulls LangChain lazily, so
importing this package needs no LLM dependencies; tests use a fake ``LLMClient``.
"""

from __future__ import annotations

from scrc.llm.azure import AzureDeployment, AzureOpenAIClient
from scrc.llm.brief import BRIEF_SYSTEM_PROMPT, BriefWriter, compose_brief_facts
from scrc.llm.ports import LLMClient

__all__ = [
    "BRIEF_SYSTEM_PROMPT",
    "AzureDeployment",
    "AzureOpenAIClient",
    "BriefWriter",
    "LLMClient",
    "compose_brief_facts",
]
