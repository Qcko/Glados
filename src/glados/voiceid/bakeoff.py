"""Threshold bake-off on household clips (DESIGN-speaker-id.md, Decision 1).

Clip layout: `<dir>/<person>/*.wav` for each person to enroll, and
`<dir>/impostor/*.wav` for voices that must never be accepted (guests, TV,
GLaDOS's own TTS; a `guest-` / `tv-` / `tts-` filename prefix groups them in
the report). Each person is enrolled leave-one-out, so every genuine probe is
a clip its own print never saw. Every probe goes through the identifier's own
`decide`. Two error rates are gated separately: impostors accepted (over
impostor probes) and enrolled people taken for someone else (over genuine
probes).

The pick is the lowest threshold with both rates at most the gate,
at the configured margin: the most real turns identified without breaking
the gate. The report holds scores only, never audio or vectors.
"""

from __future__ import annotations

import json
import os
import time
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .embedder import SAMPLE_RATE, SpeakerEmbedder, unit
from .identifier import decide
from .store import Voiceprint

IMPOSTOR_DIR = "impostor"
FAR_GATE = 0.01
MIN_IMPOSTOR_PROBES = int(1 / FAR_GATE)
THRESHOLDS = tuple(round(0.30 + 0.02 * i, 2) for i in range(35))


@dataclass(frozen=True)
class Probe:
    truth: str | None
    group: str
    embedding: np.ndarray = field(repr=False)
    prints: tuple[Voiceprint, ...] = field(repr=False)


@dataclass(frozen=True)
class Point:
    threshold: float
    genuine_accept: float
    impostor_accept: float
    wrong_person: float
    impostor_accepts: int
    wrong_persons: int
    genuine_accepts: int = 0


def run(clip_dir: Path, embedder: SpeakerEmbedder, margin: float) -> dict:
    clips, latencies = _embed_all(clip_dir, embedder)
    probes = _probes(clips, embedder.model_id)
    points = sweep(probes, margin)
    impostors = [p for p in probes if p.truth is None]
    return {
        "model_id": embedder.model_id,
        "margin": margin,
        "far_gate": FAR_GATE,
        "people": sorted(p for p in clips if p != IMPOSTOR_DIR),
        "clips": {p: len(v) for p, v in clips.items()},
        "impostor_groups": _group_counts(impostors),
        "genuine_probes": len(probes) - len(impostors),
        "enough_impostors": len(impostors) >= MIN_IMPOSTOR_PROBES,
        "scores": score_summary(probes),
        "pick": _as_dict(pick(points)),
        "sweep": [asdict(p) for p in points],
        "embed_ms": {"p50": _percentile(latencies, 50), "p95": _percentile(latencies, 95)},
    }


def sweep(probes: list[Probe], margin: float) -> list[Point]:
    return [_point(t, probes, margin) for t in THRESHOLDS]


def pick(points: list[Point]) -> Point | None:
    """Both error rates inside the gate: an impostor accepted, and an
    enrolled person taken for someone else."""
    passing = [p for p in points if p.impostor_accept <= FAR_GATE and p.wrong_person <= FAR_GATE]
    return min(passing, key=lambda p: p.threshold) if passing else None


def _point(threshold: float, probes: list[Probe], margin: float) -> Point:
    genuine = [p for p in probes if p.truth is not None]
    impostors = [p for p in probes if p.truth is None]
    accepted = {id(p): _decide(p, threshold, margin) for p in probes}
    right = sum(1 for p in genuine if accepted[id(p)] == p.truth)
    wrong = sum(1 for p in genuine if accepted[id(p)] not in (None, p.truth))
    impostor_accepts = sum(1 for p in impostors if accepted[id(p)] is not None)
    return Point(
        threshold=threshold,
        genuine_accept=_rate(right, genuine),
        impostor_accept=_rate(impostor_accepts, impostors),
        wrong_person=_rate(wrong, genuine),
        impostor_accepts=impostor_accepts,
        wrong_persons=wrong,
        genuine_accepts=right,
    )


def score_summary(probes: list[Probe]) -> dict:
    """Threshold-free separation: a genuine probe scores against its own
    print, an impostor against the closest print."""
    genuine = [_score(p, p.truth) for p in probes if p.truth is not None]
    impostor = [_score(p, None) for p in probes if p.truth is None]
    weakest = min(genuine) if genuine else None
    worst = max(impostor) if impostor else None
    return {
        "eer": _eer(genuine, impostor),
        "weakest_genuine": _round(weakest),
        "worst_impostor": _round(worst),
        "gap": _round(weakest - worst) if genuine and impostor else None,
    }


def _score(p: Probe, person: str | None) -> float:
    probe = unit(p.embedding)
    return max(float(np.dot(probe, v.centroid)) for v in p.prints if person in (None, v.person))


def _eer(genuine: list[float], impostor: list[float]) -> float | None:
    if not genuine or not impostor:
        return None
    rates = [(_share(impostor, lambda s: s >= t), _share(genuine, lambda s: s < t)) for t in sorted(genuine + impostor)]
    far, frr = min(rates, key=lambda r: abs(r[0] - r[1]))
    return _round((far + frr) / 2)


def _share(scores: list[float], hit: Callable[[float], bool]) -> float:
    return sum(1 for s in scores if hit(s)) / len(scores)


def _round(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def _decide(p: Probe, threshold: float, margin: float) -> str | None:
    return decide(p.embedding, p.prints, accept_threshold=threshold, margin=margin).person


def _rate(count: int, population: list) -> float:
    return count / len(population) if population else 0.0


def _probes(clips: dict[str, list[tuple[str, np.ndarray]]], model_id: str) -> list[Probe]:
    people = [p for p in clips if p != IMPOSTOR_DIR]
    probes: list[Probe] = []
    for person in people:
        for i, (_, vector) in enumerate(clips[person]):
            prints = _prints(clips, people, model_id, leave_out=(person, i))
            probes.append(Probe(person, "genuine", vector, prints))
    everyone = _prints(clips, people, model_id, leave_out=None)
    for name, vector in clips.get(IMPOSTOR_DIR, []):
        probes.append(Probe(None, _impostor_group(name), vector, everyone))
    return probes


def _prints(clips, people: list[str], model_id: str, leave_out: tuple[str, int] | None) -> tuple[Voiceprint, ...]:
    result = []
    for person in people:
        vectors = [v for i, (_, v) in enumerate(clips[person]) if (person, i) != leave_out]
        if vectors:
            result.append(Voiceprint(person, model_id, unit(np.mean(vectors, axis=0))))
    return tuple(result)


def _embed_all(clip_dir: Path, embedder: SpeakerEmbedder) -> tuple[dict, list[float]]:
    clips: dict[str, list[tuple[str, np.ndarray]]] = {}
    latencies: list[float] = []
    for folder in sorted(d for d in clip_dir.iterdir() if d.is_dir()):
        for path in sorted(folder.glob("*.wav")):
            pcm = read_clip(path)
            started = time.perf_counter()
            clips.setdefault(folder.name, []).append((path.name, embedder.embed(pcm)))
            latencies.append((time.perf_counter() - started) * 1000)
    _check_layout(clips)
    return clips, latencies


def _check_layout(clips: dict) -> None:
    people = [p for p in clips if p != IMPOSTOR_DIR]
    if not people:
        raise ValueError("no person folders with clips")
    thin = [p for p in people if len(clips[p]) < 2]
    if thin:
        raise ValueError(f"need 2+ clips per person for leave-one-out: {', '.join(thin)}")


def read_clip(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            raise ValueError(f"{path.name}: need {SAMPLE_RATE} Hz mono 16-bit WAV")
        return w.readframes(w.getnframes())


def _impostor_group(name: str) -> str:
    prefix = name.split("-", 1)[0].lower()
    return prefix if prefix in ("guest", "tv", "tts") else "other"


def _group_counts(impostors: list[Probe]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for p in impostors:
        counts[p.group] = counts.get(p.group, 0) + 1
    return counts


def _percentile(values: list[float], q: int) -> float | None:
    return round(float(np.percentile(values, q)), 1) if values else None


def _as_dict(point: Point | None) -> dict | None:
    return asdict(point) if point else None


def default_out_dir() -> Path:
    lab = os.environ.get("GLADOS_LAB_DIR")
    return Path(lab) / "glados" / "voiceid_bakeoff" if lab else Path.home() / ".glados" / "voiceid_bakeoff"


def write_report(data: dict, out_dir: Path, now: Callable[[], str] | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = (now or (lambda: time.strftime("%Y-%m-%d_%H%M%S")))()
    path = out_dir / f"voiceid_bakeoff_{stamp}.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def headline(data: dict) -> str:
    chosen = data["pick"]
    if chosen is None:
        return f"FAIL: no threshold keeps false accepts <= {FAR_GATE:.0%} at margin {data['margin']}"
    caveat = "" if data["enough_impostors"] else (
        f" -- PROVISIONAL: under {MIN_IMPOSTOR_PROBES} impostor clips cannot show a "
        f"{FAR_GATE:.0%} impostor rate (0 of n bounds it only near 3/n)"
    )
    return (
        f"pick accept_threshold={chosen['threshold']:.2f} (margin {data['margin']}): "
        f"genuine accept {chosen['genuine_accept']:.0%}, impostor accept {chosen['impostor_accept']:.1%}, "
        f"wrong person {chosen['wrong_person']:.1%}; "
        f"embed p50 {data['embed_ms']['p50']} ms p95 {data['embed_ms']['p95']} ms{caveat}"
    )
