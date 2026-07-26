"""G1: the server-level `untrusted` floor, and the memory-append ban.

Both properties exist to contain the same thing: text an attacker authored
elsewhere (a phone writing to the calendar, a scraped page) reaching the model
as something other than data. calendar-mcp's DESIGN.md states them as a
contract on any orchestrator that consumes its rows; this file is GLaDOS's half
of that contract, expressed as checks rather than prose.
"""

from __future__ import annotations

import ast
from pathlib import Path

from glados.core.adapters import ToolSpec
from glados.core.config import ServerEntry, ToolOverlay

_SRC = Path(__file__).resolve().parents[1] / "src" / "glados"


def _spec(name: str = "get_agenda") -> ToolSpec:
    return ToolSpec(server="calendar", name=name, description="", parameters={})


def test_a_tool_with_no_overlay_inherits_the_server_floor():
    """The whole point: a tool nobody listed is still wrapped.

    This is the case that matters for calendar-mcp, whose authoring tools do
    not exist yet. When they arrive, they must be untrusted on the day they
    appear -- not on the day someone remembers to edit a config.
    """
    entry = ServerEntry(id="calendar", command="x", untrusted=True)

    assert entry.apply_flags(_spec("a_tool_added_next_year")).untrusted is True


def test_an_overlay_cannot_lower_a_tool_below_the_floor():
    entry = ServerEntry(
        id="calendar",
        command="x",
        untrusted=True,
        tool_overlays={"get_agenda": ToolOverlay(untrusted=False)},
    )

    assert entry.apply_flags(_spec()).untrusted is True


def test_a_timeout_only_overlay_does_not_un_mark_untrusted():
    """A security downgrade written as a performance edit.

    `ToolOverlay(timeout_s=35)` still has `untrusted=False` in its other
    fields, so a plain assignment would silently clear the floor. This is the
    exact shape of the edit someone makes while tuning a slow server.
    """
    entry = ServerEntry(
        id="dunnes",
        command="x",
        untrusted=True,
        tool_overlays={"search_products": ToolOverlay(timeout_s=35.0)},
    )

    flagged = entry.apply_flags(_spec("search_products"))

    assert flagged.untrusted is True
    assert flagged.timeout_s == 35.0


def test_an_overlay_can_still_raise_a_single_tool():
    """Without a server floor, the per-tool flag must keep working as before."""
    entry = ServerEntry(
        id="mixed",
        command="x",
        tool_overlays={"fetch_page": ToolOverlay(untrusted=True)},
    )

    assert entry.apply_flags(_spec("fetch_page")).untrusted is True
    assert entry.apply_flags(_spec("local_dice")).untrusted is False


def test_the_other_flags_still_come_from_the_overlay():
    entry = ServerEntry(
        id="dunnes",
        command="x",
        untrusted=True,
        tool_overlays={
            "add_to_cart": ToolOverlay(mutating=True, requires_confirmation=True)
        },
    )

    flagged = entry.apply_flags(_spec("add_to_cart"))

    assert flagged.mutating is True
    assert flagged.requires_confirmation is True


def test_a_server_with_no_floor_is_unchanged():
    """Back-compat: existing servers.toml files keep their exact behaviour."""
    entry = ServerEntry(id="toy", command="x")

    assert entry.apply_flags(_spec("roll_dice")).untrusted is False


def _writes_to_a_file(path: Path) -> list[str]:
    """Calls that would open a path for writing, found via AST, not substring.

    A tripwire, not a proof. It does not see `os.replace`, `shutil.copy`,
    `json.dump` into a handle opened elsewhere, or a write delegated to a
    module outside the guarded list. It catches the shape a "remember this"
    feature would actually take, which is the point.
    """
    found = []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        is_method = isinstance(node.func, ast.Attribute)
        name = node.func.attr if is_method else getattr(node.func, "id", "")
        if name in {"write_text", "write_bytes", "writelines"}:
            found.append(f"{path.name}:{node.lineno} {name}")
        elif name == "open":
            # `Path.open("w")` takes the mode first; builtin `open(path, "w")`
            # takes it second. Reading args[0] for the builtin would flag any
            # filename containing "w" -- `.wav`, for one.
            positional = node.args[0:1] if is_method else node.args[1:2]
            modes = list(positional) + [k.value for k in node.keywords if k.arg == "mode"]
            for arg in modes:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if any(m in arg.value for m in ("w", "a", "x", "+")):
                        found.append(f"{path.name}:{node.lineno} open(mode={arg.value!r})")
    return found


def test_glados_never_auto_appends_to_a_memory_file():
    """calendar-mcp's contract item 2, upheld by ABSENCE -- so pinned here.

    The ban is on auto-appending `app`/`provider:*` content to a memory file
    that later gets injected into a TRUSTED system prompt. That would launder
    attacker-authored text from data into instructions, permanently, and the
    hash-approval gate would not help: a blob approved once stays approved.

    GLaDOS currently has no memory-writing path at all, which satisfies the
    contract completely and invisibly. Invisible is the problem -- nobody
    notices an absence being filled in. A "remember this" feature is an
    obvious, attractive thing to add, so make adding one require reading this
    test and thinking about taint first.

    Deliberately NOT a blanket ban on file writes: the audio sink writes wav
    files and should keep doing so. It bans writes from the modules that hold
    memory and prompt content.
    """
    guarded = [
        _SRC / "core" / "memory_gate.py",
        _SRC / "core" / "organizer.py",
        *(_SRC / "brain" / "prompts").glob("*.py"),
    ]
    offenders = []
    for path in guarded:
        offenders.extend(_writes_to_a_file(path))
    assert not offenders, (
        f"a memory/prompt module grew a file write: {offenders}. If this is a "
        "'remember this' feature, it must NOT be able to persist tool content "
        "carrying origin=app or provider:* -- that content is injected into a "
        "TRUSTED prompt later. See calendar-mcp DESIGN.md, orchestrator "
        "contract item 2."
    )
