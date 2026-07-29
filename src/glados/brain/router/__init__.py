"""Local multi-model routing (ARCHITECTURE.md section 12, v2.6).

The router picks, per turn, whether the primary local brain or a resident
specialist model handles the request. It ships **deterministic first** --
keyword + length + clause-shape rules -- mirroring the effort-router pattern. An
LLM-driven router is a later optimisation layered on top once logged decisions
show the rules mis-route.
"""

from __future__ import annotations

from .rules import RouteDecision, Router

__all__ = ["RouteDecision", "Router"]
