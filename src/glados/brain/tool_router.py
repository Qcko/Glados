"""Deterministic per-turn tool-scoping (ARCH section 13 tiered tool-scoping, v1).

The local 14B model degenerates when shown ~30 flat tools -- it hallucinates
`Call<PascalCase>` pseudo-tool-calls instead of real ones (the
project-callcheck-tooltext leak), and a live experiment confirmed the cause is
tool-list OVERLOAD: with a minimal tool list it calls tools correctly. The
ToolRouter picks, per turn, the subset of tools the model sees.

v1 is deterministic and keyword-based (same whole-word style as
brain/router/rules.py) and **decoupled from the primary/specialist difficulty
router** -- the difficulty router is a likely-dead path that may be removed, so
tool-scope is its own decision on the user text, not nested under that verdict.

Scoping is opt-in per server to stay back-compatible:
- `core` servers are ALWAYS in scope (the "core tools always on" allowlist).
- a server that declares `intent_keywords` is scoped: in scope only when the
  request matches one of them.
- a server with NO keywords and not core is UNSCOPED: always in scope (so an
  un-annotated server behaves exactly as before this feature).

Scope is computed once at turn start, preserving section 7's per-turn frozen registry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..core.adapters import ToolSpec


@dataclass(frozen=True)
class ServerScope:
    """Scoping metadata for one server, sourced from servers.toml."""

    server_id: str
    intent_keywords: tuple[str, ...] = ()
    core: bool = False


@dataclass(frozen=True)
class ToolRouter:
    """Selects the per-turn tool subset. `scopes` is keyed by server id; a
    server absent from it is treated as unscoped (always offered)."""

    scopes: dict[str, ServerScope]

    def scope_for(self, text: str, all_specs: list[ToolSpec]) -> list[ToolSpec]:
        """Return the tools offered this turn. Never returns the whole registry
        by accident: an unmatched annotated server is dropped; only core,
        matched, and unscoped servers survive."""
        matched = self._matched_servers(text)
        return [s for s in all_specs if self._in_scope(s.server, matched)]

    def _in_scope(self, server_id: str, matched: set[str]) -> bool:
        scope = self.scopes.get(server_id)
        if scope is None:
            return True  # unscoped server -> always offered (back-compat)
        if scope.core:
            return True
        if not scope.intent_keywords:
            return True  # declared no keywords -> unscoped
        return server_id in matched

    def _matched_servers(self, text: str) -> set[str]:
        lowered = (text or "").lower()
        matched: set[str] = set()
        for server_id, scope in self.scopes.items():
            for keyword in scope.intent_keywords:
                if re.search(rf"\b{re.escape(keyword.lower())}\b", lowered):
                    matched.add(server_id)
                    break
        return matched
