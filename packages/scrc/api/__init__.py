"""Interface layer (Layer 6): FastAPI surface. Contains no decision logic."""

from __future__ import annotations

from scrc.api.app import create_app

__all__ = ["create_app"]
