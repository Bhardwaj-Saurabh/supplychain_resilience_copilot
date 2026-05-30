"""Composition root: assembles tools, models, agents, graph, and the API.

The single place that depends on every layer. ``build_app`` is the uvicorn
factory entrypoint.
"""

from __future__ import annotations

from scrc.app.composition import build_app, build_demo_bundle
from scrc.app.settings import Settings

__all__ = ["Settings", "build_app", "build_demo_bundle"]
