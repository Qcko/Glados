"""Hybrid local/cloud routing (ARCHITECTURE.md v2.6).

The router picks, per turn, whether the local brain or a cloud brain handles
the request. It ships **deterministic first** — keyword + length + clause-shape
rules — mirroring the effort-router pattern. An LLM-driven router is a later
optimisation layered on top once logged decisions show the rules mis-route.
"""

from __future__ import annotations

from .rules import RouteDecision, Router

__all__ = ["RouteDecision", "Router"]
