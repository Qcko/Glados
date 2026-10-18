"""Memory-writer bake-off runner (DESIGN-memory-scorer.md, Part A).

Each case runs through the REAL nightly fold -- `MemoryWriter.run_batch_now`
on a throwaway store, so the prompt, validator, chunk split and day caps are
the ones that ship -- and `scoring` compares what it left behind with what
the case expects. Corpus text is household-shaped, so the run refuses any
model that is not a local Ollama, and refuses to share the GPU with a live
GLaDOS.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import socket
import subprocess
import tempfile
import time
import uuid
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx
from cryptography.fernet import Fernet

from ..core.adapters import LLM
from ..core.config import GladosConfig
from ..core.memory_facts import AcceptedAdd, StagedNote
from ..core.memory_writer import BatchPolicy, MemoryWriter
from ..core.speaker_memory import ChunkWrite, SpeakerMemoryStore
from .scoring import Case, CaseScore, Outcome, Summary, score_case, summarise

KEY = ("bakeoff", "spk1")
GROUP = (*KEY, "day")
RUN_DATE = "2026-01-01"
_REPO = Path(__file__).resolve().parents[3]


class BakeoffRefused(RuntimeError):
    pass


# ---- preflight ---------------------------------------------------------------


def check_local_ollama(cfg: GladosConfig) -> None:
    """Positive locality check, as for injection: an Ollama host that
    resolves to loopback or a private LAN address, nothing else."""
    if cfg.llm.backend != "ollama":
        raise BakeoffRefused(f"[llm] backend is {cfg.llm.backend!r}; the bake-off runs on local ollama only")
    host = urlparse(cfg.llm.host).hostname or ""
    addresses = _resolve(host)
    if not addresses or not all(_is_local(a) for a in addresses):
        raise BakeoffRefused(f"ollama host {host!r} is not on this machine or the LAN")


def _resolve(host: str) -> list:
    """Every address the name has, IPv4 and IPv6. Resolved once here and
    again by httpx, so a hostile resolver could rebind between the two;
    accepted for an operator-run dev tool on the household network."""
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError as e:
        raise BakeoffRefused(f"ollama host {host!r} does not resolve") from e
    return [ipaddress.ip_address(info[4][0].split("%")[0]) for info in infos]


def _is_local(address) -> bool:
    return (address.is_loopback or address.is_private) and not address.is_unspecified


def check_glados_stopped(cfg: GladosConfig) -> None:
    """One GPU slot: a live GLaDOS would both slow and skew the run."""
    try:
        with socket.create_connection(("127.0.0.1", cfg.server.port), timeout=0.5):
            pass
    except OSError:
        return
    raise BakeoffRefused(f"GLaDOS answers on :{cfg.server.port}; stop it before a bake-off")


def model_digest(host: str, model: str) -> str:
    tags = httpx.get(f"{host}/api/tags", timeout=10).json().get("models", [])
    for entry in tags:
        if entry.get("name") == model or entry.get("model") == model:
            return entry.get("digest", "")
    raise BakeoffRefused(f"model {model!r} is not pulled on {host}")


def unload(host: str, model: str) -> None:
    """So the next model never shares VRAM with this one."""
    httpx.post(f"{host}/api/generate", json={"model": model, "keep_alive": 0}, timeout=30)


# ---- the run ---------------------------------------------------------------


async def run_cases(cases: list[Case], llm: LLM, policy: BatchPolicy) -> list[CaseScore]:
    with tempfile.TemporaryDirectory(prefix="glados-bakeoff-") as tmp:
        return [score_case(c, await run_case(c, llm, policy, Path(tmp))) for c in cases]


async def run_case(case: Case, llm: LLM, policy: BatchPolicy, tmp: Path) -> Outcome:
    store = SpeakerMemoryStore(tmp / f"{uuid.uuid4().hex}.sqlite", cipher=Fernet(Fernet.generate_key()))
    try:
        _seed(store, case)
        writer = MemoryWriter(
            llm=llm, store=store, is_busy=lambda: False, is_warm=lambda: True,
            is_enrolled=lambda person: True, policy=_case_policy(policy, case),
        )
        started = time.monotonic()
        await writer.run_batch_now(RUN_DATE)
        elapsed_ms = int((time.monotonic() - started) * 1000)
        after = {f.text for f in store.facts(*KEY)}
        failed = writer.chunks_failed
    finally:
        store.close()
    known = set(case.known)
    return Outcome(
        added=tuple(after - known), dropped=tuple(known - after),
        elapsed_ms=elapsed_ms, failed_chunks=failed,
    )


def _case_policy(policy: BatchPolicy, case: Case) -> BatchPolicy:
    return replace(
        policy,
        idle_before_batch_s=0.0,
        max_notes_per_batch=case.max_notes_per_batch or policy.max_notes_per_batch,
    )


def _seed(store: SpeakerMemoryStore, case: Case) -> None:
    """Known facts, oldest first so the prompt lists them as prod would."""
    for text in case.known:
        store.apply_chunk_sync(
            ChunkWrite(group=(*KEY, "seed"), run_date="seed", note_ids=(),
                       expected=store.stamp(*KEY), adds=(AcceptedAdd(text, (), False),)),
            lambda person: True,
        )
    for note in case.notes:
        reply = None if note.untrusted else note.reply
        store.stage_note(GROUP, uuid.uuid4().hex, StagedNote(0, note.said, reply, note.untrusted))


# ---- report ----------------------------------------------------------------


def report(scores: list[CaseScore], meta: dict) -> dict:
    by_split = {
        name: asdict(summarise([s for s in scores if name == "all" or s.split == name]))
        for name in ("test", "dev", "all")
    }
    for name, summary in by_split.items():
        summary["gate"] = _gate_of(summary)
    return {**meta, "summary": by_split, "cases": [asdict(s) for s in scores]}


def markdown(data: dict) -> str:
    rows = ["| split | cases | gate | F1 | recall | precision | empty-day false adds | unadjudicated | p90 ms |",
            "|---|---|---|---|---|---|---|---|---|"]
    for name, s in data["summary"].items():
        rows.append(
            f"| {name} | {s['cases']} | {s['gate']} | {s['f1']:.2f} "
            f"| {s['recall']:.2f} | {s['precision']:.2f} | {s['empty_day_false_adds']:.2f} "
            f"of {s['empty_days']} | {s['unadjudicated']} | {s['p90_ms']} |"
        )
    head = (f"# Memory bake-off: {data['model']}\n\nDigest `{data['digest']}`, corpus "
            f"`{data['corpus_sha256'][:12]}`, {data['settings']}\n\n")
    detail = ["", "## Per case", "", "| case | split | matched | drops | forbidden | unexpected |", "|---|---|---|---|---|---|"]
    for c in data["cases"]:
        detail.append(
            f"| {c['case_id']} | {c['split']} | {c['matched']}/{c['expected']} "
            f"| {c['drops_correct']}/{c['drops_expected']} (+{c['drops_wrong']} wrong) "
            f"| {_cell(c['forbidden'])} | {_cell(c['unexpected'])} |"
        )
    return head + "\n".join(rows + detail) + "\n"


def _gate_of(summary: dict) -> str:
    """The same verdict as `Summary.gate`, from its dumped fields."""
    if summary["failed_cases"] or not summary["cases"]:
        return "INVALID"
    return "VETO" if summary["forbidden_cases"] else "pass"


def _cell(facts: list[str]) -> str:
    """Model output into a table cell, defanged: corpus payloads are hostile."""
    return "; ".join(_defang(f) for f in facts) or "-"


def _defang(fact: str) -> str:
    flat = " ".join(fact.split())
    for bad, safe in (("|", "/"), ("<", "("), (">", ")"), ("`", "'"), ("[", "("), ("]", ")")):
        flat = flat.replace(bad, safe)
    return flat


def corpus_sha256(texts: list[str]) -> str:
    return hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()


def default_out_dir() -> Path:
    lab = os.environ.get("GLADOS_LAB_DIR")
    return Path(lab) / "glados" / "memory_bakeoff" if lab else Path.home() / ".glados" / "memory_bakeoff"


def write_report(data: dict, out_dir: Path) -> tuple[Path, Path]:
    """Out of every checkout of this repo: a report holds model output
    derived from local cases. Created mode 0600; on Windows that mode is only
    the read-only bit, so the profile or lab directory ACL is what keeps it
    private."""
    out_dir = out_dir.resolve()
    for tree in _checkouts():
        if out_dir == tree or tree in out_dir.parents:
            raise BakeoffRefused(f"report dir {out_dir} is inside the repo")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{datetime.now():%Y-%m-%d_%H%M}-{data['model'].replace(':', '_').replace('/', '_')}"
    paths = (out_dir / f"{stem}.json", out_dir / f"{stem}.md")
    _write_private(paths[0], json.dumps(data, indent=2))
    _write_private(paths[1], markdown(data))
    return paths


def _write_private(path: Path, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)


def headline(summary: Summary) -> str:
    return (f"gate={summary.gate} cases={summary.cases} failed_cases={len(summary.failed_cases)} f1={summary.f1:.2f} "
            f"empty_day_false_adds={summary.empty_day_false_adds:.2f}")


def _checkouts() -> list[Path]:
    """This worktree and the main checkout it hangs off, which a worktree
    under `.claude/worktrees/` would otherwise not see as the repo."""
    trees = [_REPO]
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO), "worktree", "list", "--porcelain"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return trees
    first = next((line[len("worktree "):] for line in out.splitlines() if line.startswith("worktree ")), None)
    if first:
        trees.append(Path(first).resolve())
    return trees


def default_cases() -> Path:
    return _REPO / "benchmarks" / "memory" / "cases.jsonl"


def unload_quietly(host: str, model: str) -> None:
    """Best effort, from a `finally`: never mask the run's own error."""
    try:
        unload(host, model)
    except httpx.HTTPError:
        pass
