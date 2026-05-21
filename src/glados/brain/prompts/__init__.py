"""LLM prompt strings. Kept out of `core/organizer.py` so prompt edits
don't churn turn-orchestration code review and so future per-persona /
per-room overrides have a natural home.
"""

from .system import SYSTEM_PROMPT

__all__ = ["SYSTEM_PROMPT"]
