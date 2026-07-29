"""STT benchmark: WER + latency on a manifest of (audio, reference) pairs.

Why: built to baseline `distil-small.en` before the multilingual swap,
which has since been rolled back (ARCH section 13: DEFERRED). The harness is
now the tool for benchmarking *any* future STT change -- model upgrade,
compute_type tweak, language pinning, multilingual retry. Run twice
with two manifests for cross-language compare when needed.

Usage:
    uv run python scripts/stt_benchmark.py \\
        --manifest benchmarks/english.jsonl \\
        --model distil-small.en \\
        --language en \\
        --device cpu \\
        --out benchmarks/distil-small.en.json

Manifest format (JSONL, one record per line):
    {"audio": "path/to/clip.wav", "text": "the reference transcript"}

Audio requirements: 16-kHz, mono, 16-bit PCM WAV. Standard for both
LibriSpeech and Common Voice exports (use `ffmpeg -ar 16000 -ac 1 -c:a
pcm_s16le` to convert anything else).

Output JSON has shape:
    {
        "model": "...", "language": "...", "device": "...",
        "n_clips": int,
        "summary": {
            "wer_mean": float, "wer_p50": float, "wer_p95": float,
            "rtf_mean": float,  # latency / audio_duration
            "latency_ms_p50": float, "latency_ms_p95": float,
            "total_ref_words": int, "total_errors": int,
            "corpus_wer": float  # sum(errors) / sum(ref_words) -- preferred
        },
        "clips": [
            {"audio": "...", "wer": float, "latency_ms": float,
             "audio_ms": float, "hypothesis": "...", "reference": "...",
             "substitutions": int, "deletions": int, "insertions": int},
            ...
        ]
    }

The `corpus_wer` and `corpus_rtf` are the metrics you should report when
comparing runs -- they're the standard ASR-community shape (totals over
totals), robust to varying clip lengths. Per-clip means are also
emitted for visibility but short clips dominate them.

Latency is wall-clock around `await stt.transcribe(pcm)` -- includes the
`asyncio.to_thread` dispatch (microseconds, negligible vs. inference)
and is the user-perceived number GLaDOS cares about. Don't try to time
inside the worker thread.

Warmup: the first clip pays ctranslate2 kernel-graph compilation and
will dominate p95 on small manifests. `--warmup N` (default 1) runs the
first N clips through without recording, so reported numbers reflect
steady-state inference. Bump it for tiny manifests.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import wave
from pathlib import Path

from glados.audio.stt.wer import wer
from glados.core.adapters import STT


def _load_pcm(path: Path) -> tuple[bytes, float]:
    """Return (pcm_bytes, duration_seconds). Raises on unexpected format
    rather than silently up- or down-sampling -- guarantees apples-to-
    apples comparison across runs."""
    with wave.open(str(path), "rb") as wf:
        if wf.getframerate() != 16_000:
            raise ValueError(f"{path}: expected 16-kHz, got {wf.getframerate()}")
        if wf.getnchannels() != 1:
            raise ValueError(f"{path}: expected mono, got {wf.getnchannels()} channels")
        if wf.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit PCM, got {wf.getsampwidth()*8}-bit")
        n_frames = wf.getnframes()
        pcm = wf.readframes(n_frames)
        return pcm, n_frames / 16_000.0


async def _run_one(stt: STT, pcm: bytes) -> tuple[str, float]:
    t0 = time.perf_counter()
    hyp = await stt.transcribe(pcm)
    return hyp, (time.perf_counter() - t0) * 1000.0


async def benchmark(
    stt: STT, manifest: list[dict], *, metadata: dict, warmup: int = 0
) -> dict:
    """Run the manifest through `stt`, return the structured result.

    Pure logic -- no CLI, no model construction. The CLI wraps this so
    the harness itself is testable with a FakeSTT.

    `warmup` runs the first N clips through without recording -- used
    by the CLI to absorb the model's first-inference graph-compile so
    it doesn't blow up p95.
    """
    if warmup > 0:
        for entry in manifest[:warmup]:
            pcm, _ = _load_pcm(Path(entry["audio"]))
            await stt.transcribe(pcm)
    clips: list[dict] = []
    for i, entry in enumerate(manifest, 1):
        audio_path = Path(entry["audio"])
        reference = entry["text"]
        pcm, audio_s = _load_pcm(audio_path)
        hypothesis, latency_ms = await _run_one(stt, pcm)
        r = wer(reference, hypothesis)
        clips.append({
            "audio": str(audio_path),
            "reference": reference.strip(),
            "hypothesis": hypothesis.strip(),
            "wer": r.wer,
            "substitutions": r.substitutions,
            "deletions": r.deletions,
            "insertions": r.insertions,
            "ref_words": r.ref_words,
            "latency_ms": latency_ms,
            "audio_ms": audio_s * 1000.0,
        })
        print(f"[{i}/{len(manifest)}] WER={r.wer:.3f} "
              f"latency={latency_ms:.0f}ms audio={audio_s*1000:.0f}ms "
              f"rtf={latency_ms/(audio_s*1000):.2f} :: {audio_path.name}")

    return {
        **metadata,
        "n_clips": len(clips),
        "summary": _summarise(clips),
        "clips": clips,
    }


def _read_manifest(path: Path) -> list[dict]:
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(json.loads(line))
    return out


def _summarise(clips: list[dict]) -> dict:
    wers = [c["wer"] for c in clips]
    latencies = [c["latency_ms"] for c in clips]
    rtfs = [c["latency_ms"] / c["audio_ms"] for c in clips if c["audio_ms"] > 0]
    total_words = sum(c["ref_words"] for c in clips)
    total_errors = sum(
        c["substitutions"] + c["deletions"] + c["insertions"] for c in clips
    )
    total_latency = sum(latencies)
    total_audio = sum(c["audio_ms"] for c in clips)
    return {
        "wer_mean": statistics.fmean(wers) if wers else 0.0,
        "wer_p50": _percentile(wers, 50),
        "wer_p95": _percentile(wers, 95),
        "latency_ms_p50": _percentile(latencies, 50),
        "latency_ms_p95": _percentile(latencies, 95),
        "rtf_mean": statistics.fmean(rtfs) if rtfs else 0.0,
        "total_ref_words": total_words,
        "total_errors": total_errors,
        # Totals-over-totals -- robust to varying clip lengths; the
        # numbers to report when comparing runs.
        "corpus_wer": (total_errors / total_words) if total_words else 0.0,
        "corpus_rtf": (total_latency / total_audio) if total_audio else 0.0,
    }


def _percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile. statistics.quantiles is awkward
    for ad-hoc p95; this is the same formula numpy uses by default."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="path to JSONL manifest")
    ap.add_argument("--model", default="distil-small.en")
    ap.add_argument("--language", default=None,
                    help="ISO code (e.g. 'en', 'cs'); omit for auto-detect")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--out", required=True, help="path for JSON result")
    ap.add_argument("--warmup", type=int, default=1,
                    help="run the first N clips through without recording so "
                         "first-inference graph-compile doesn't dominate p95")
    args = ap.parse_args()

    from glados.audio.stt.whisper import WhisperSTT  # deferred: heavy import

    manifest = _read_manifest(Path(args.manifest))
    if not manifest:
        print(f"manifest {args.manifest} is empty", file=sys.stderr)
        sys.exit(2)
    stt = WhisperSTT(
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
    )
    metadata = {
        "model": args.model,
        "language": args.language,
        "device": args.device,
        "compute_type": args.compute_type,
    }
    result = asyncio.run(
        benchmark(stt, manifest, metadata=metadata, warmup=args.warmup)
    )
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    s = result["summary"]
    print()
    print(f"corpus WER  : {s['corpus_wer']:.4f}  ({s['total_errors']}/{s['total_ref_words']} words)")
    print(f"corpus RTF  : {s['corpus_rtf']:.3f}")
    print(f"WER  p50/p95: {s['wer_p50']:.4f} / {s['wer_p95']:.4f}")
    print(f"latency p50 : {s['latency_ms_p50']:.0f} ms")
    print(f"latency p95 : {s['latency_ms_p95']:.0f} ms")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
