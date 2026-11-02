"""Side-by-side bake-off: WeSpeaker against ECAPA on the same clips.

The decision rule was fixed before any real numbers (SESSION 2026-11-02):
ECAPA is preferred and wins unless (1) it fails the gate while WeSpeaker
passes, (2) WeSpeaker accepts more than one extra genuine clip at its own
pick (counted over every enrolled person, the household total), or
(3) ECAPA's p95 embed exceeds the latency ceiling. EER and the score gap are reported, never deciding. A provisional run still names a
model but cannot enable a room.
"""

from __future__ import annotations

from pathlib import Path

from . import bakeoff
from .embedder import SpeakerEmbedder

PREFERRED = "ecapa"
CHALLENGER = "wespeaker"
BACKENDS = (CHALLENGER, PREFERRED)
MAX_EXTRA_GENUINE = 1
P95_CEILING_MS = 300.0


def run(clip_dir: Path, embedders: dict[str, SpeakerEmbedder], margin: float) -> dict:
    results = {name: bakeoff.run(clip_dir, embedders[name], margin) for name in BACKENDS}
    return {"margin": margin, "backends": results, "verdict": verdict(results)}


def verdict(results: dict[str, dict]) -> dict:
    preferred, challenger = results[PREFERRED], results[CHALLENGER]
    winner, reason = _decide(preferred, challenger)
    provisional = not (preferred["enough_impostors"] and challenger["enough_impostors"])
    return {"winner": winner, "reason": reason, "provisional": provisional}


def _decide(preferred: dict, challenger: dict) -> tuple[str | None, str]:
    if preferred["pick"] is None:
        if challenger["pick"] is None:
            return None, "neither model passes the gate"
        return CHALLENGER, f"{PREFERRED} fails the gate, {CHALLENGER} passes"
    if challenger["pick"] is not None:
        extra = challenger["pick"]["genuine_accepts"] - preferred["pick"]["genuine_accepts"]
        if extra > MAX_EXTRA_GENUINE:
            return CHALLENGER, f"{CHALLENGER} accepts {extra} more genuine clips at its pick"
    p95 = preferred["embed_ms"]["p95"]
    if p95 is not None and p95 > P95_CEILING_MS:
        if challenger["pick"] is None:
            return None, f"{PREFERRED} p95 {p95} ms > {P95_CEILING_MS:.0f} ms and {CHALLENGER} fails the gate"
        return CHALLENGER, f"{PREFERRED} p95 embed {p95} ms > {P95_CEILING_MS:.0f} ms"
    return PREFERRED, f"{PREFERRED} is preferred and no override fired"


def headline(data: dict) -> str:
    lines = [f"{name}: {bakeoff.headline(data['backends'][name])}" for name in BACKENDS]
    lines += [f"{name}: {_separation(data['backends'][name])}" for name in BACKENDS]
    v = data["verdict"]
    tail = " -- PROVISIONAL: decides the model, cannot enable a room" if v["provisional"] else ""
    lines.append(f"winner: {v['winner'] or 'none'} ({v['reason']}){tail}")
    return "\n".join(lines)


def _separation(result: dict) -> str:
    s = result["scores"]
    return (
        f"EER {s['eer']}, weakest genuine {s['weakest_genuine']}, "
        f"worst impostor {s['worst_impostor']}, gap {s['gap']}"
    )
