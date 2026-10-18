"""Memory-writer bake-off scorer and runner (DESIGN-memory-scorer.md)."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from glados.core.config import GladosConfig
from glados.core.memory_writer import BatchPolicy
from glados.memory import bakeoff
from glados.memory.scoring import (
    CaseError,
    Matcher,
    Outcome,
    load_cases,
    parse_case,
    score_case,
    summarise,
)

from .speaker_memory_helpers import ScriptedLLM

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "benchmarks" / "memory" / "cases.jsonl"


def _case(**over):
    raw = {"id": "c", "split": "test", "origin": "synthetic", "notes": [{"said": "hi"}]}
    raw.update(over)
    return parse_case(raw)


# ---- matchers ---------------------------------------------------------------


@pytest.mark.parametrize(
    "fact, hit",
    [
        ("Is vegetarian.", True),
        ("Is NOT vegetarian.", False),
        ("Partner is vegetarian.", False),
        ("Eats meat.", False),
    ],
)
def test_none_keeps_negation_and_other_people_from_matching(fact, hit):
    veg = Matcher(all=("vegetarian",), none=("not", "partner"))
    assert veg.matches(fact) is hit


def test_matchers_are_whole_words_and_phrases():
    assert not Matcher(all=("cat",)).matches("Likes category theory.")
    assert Matcher(any=("night shift", "nights")).matches("Works a night shift.")
    assert not Matcher(any=("night shift",)).matches("Works shifts at night.")


# ---- cases ------------------------------------------------------------------


@pytest.mark.parametrize(
    "over",
    [{"split": "train"}, {"notes": []}, {"known": [], "expect_drop": ["Lives in Galway."]}],
    ids=["bad-split", "no-notes", "drop-not-known"],
)
def test_malformed_cases_are_refused(over):
    with pytest.raises(CaseError):
        _case(**over)


def test_duplicate_case_ids_are_refused():
    line = json.dumps({"id": "x", "split": "dev", "origin": "synthetic", "notes": [{"said": "a"}]})
    with pytest.raises(CaseError, match="duplicate"):
        load_cases(f"{line}\n{line}\n")


# ---- scoring ------------------------------------------------------------------


def test_a_fact_satisfies_one_expectation_at_most():
    case = _case(expect_add=[{"all": ["vegetarian"]}, {"any": ["vegetarian", "vegan"]}])
    score = score_case(case, Outcome(added=("Is vegetarian.",), dropped=()))
    assert (score.matched, score.expected) == (1, 2)


def test_leftover_facts_split_into_forbidden_also_ok_and_unexpected():
    case = _case(
        expect_add=[{"all": ["vegetarian"]}],
        forbid=[{"any": ["cart"]}],
        also_ok=[{"all": ["tofu"]}],
    )
    added = ("Is vegetarian.", "Wants milk in the cart.", "Likes tofu.", "Has a bike.")
    score = score_case(case, Outcome(added=added, dropped=()))
    assert score.matched == 1
    assert score.forbidden == ["Wants milk in the cart."]
    assert score.unexpected == ["Has a bike."]


def test_drops_are_scored_by_known_text():
    case = _case(known=["Lives in Galway.", "Has a cat."], expect_drop=["Lives in Galway."])
    score = score_case(case, Outcome(added=(), dropped=("Lives in Galway.", "Has a cat.")))
    assert (score.drops_correct, score.drops_wrong) == (1, 1)


def test_a_forbidden_hit_vetoes_and_unadjudicated_stays_out_of_precision():
    good = score_case(_case(id="a", expect_add=[{"all": ["dog"]}]),
                      Outcome(added=("Has a dog.", "Has a bike."), dropped=()))
    bad = score_case(_case(id="b", forbid=[{"any": ["password"]}]),
                     Outcome(added=("Password is hunter2.",), dropped=()))
    summary = summarise([good, bad])
    assert not summary.gate_passed and summary.forbidden_cases == ("b",)
    assert summary.unadjudicated == 1
    assert summary.recall == 1.0 and summary.precision == 0.5


def test_any_add_on_an_empty_day_is_a_false_add():
    quiet = score_case(_case(id="q"), Outcome(added=(), dropped=()))
    noisy = score_case(_case(id="n"), Outcome(added=("Likes chatting.",), dropped=()))
    summary = summarise([quiet, noisy])
    assert summary.empty_days == 2 and summary.empty_day_false_adds == 0.5


# ---- the committed corpus ----------------------------------------------------------


def test_the_committed_corpus_is_well_formed_and_balanced():
    cases = load_cases(CORPUS.read_text(encoding="utf-8"))
    assert len(cases) >= 20
    assert all(c.origin == "synthetic" for c in cases), "real-derived cases live out of repo"
    assert {c.split for c in cases} == {"dev", "test"}
    assert sum(c.empty_day for c in cases) / len(cases) >= 0.4, "empty days are the common day"


def test_the_committed_corpus_names_no_household_member():
    lexicon = REPO / "benchmarks" / "stt" / "lexicon.local.txt"
    if not lexicon.exists():
        pytest.skip("no local lexicon on this machine")
    names = _section(lexicon.read_text(encoding="utf-8"), "people")
    text = CORPUS.read_text(encoding="utf-8").lower()
    assert not [n for n in names if f" {n.lower()} " in f" {text} "]


def _section(text: str, name: str) -> list[str]:
    words, inside = [], False
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("["):
            inside = line == f"[{name}]"
        elif inside and line and not line.startswith("#"):
            words.append(line)
    return words


# ---- the runner drives the real fold ------------------------------------------------


async def test_the_runner_scores_adds_drops_and_the_day_caps():
    import re

    def cite(messages):
        return [int(i) for i in re.findall(r'<note id="(\d+)">', messages[1].content)]

    def reply(messages):
        body = messages[1].content
        if "vegetarian" in body:
            return json.dumps({"add": [{"fact": "Is vegetarian.", "notes": cite(messages)}]})
        if "Cork" in body:
            return json.dumps({"add": [], "drop": [1]})
        if "recipe" in body:
            return json.dumps({"add": [{"fact": "Likes Thai food.", "notes": cite(messages)[:1]},
                                       {"fact": "Likes baking.", "notes": cite(messages)[1:]}]})
        return '{"add": [], "drop": []}'

    cases = [
        _case(id="veg", notes=[{"said": "I've gone vegetarian"}], expect_add=[{"all": ["vegetarian"]}]),
        _case(id="moved", known=["Lives in Galway."], notes=[{"said": "We moved to Cork"}],
              expect_drop=["Lives in Galway."]),
        _case(id="cap", notes=[{"said": "a recipe for Thai food", "untrusted": True}, {"said": "more baking", "untrusted": True}],
              expect_add=[{"any": ["thai", "baking"]}]),
    ]
    scores = await bakeoff.run_cases(cases, ScriptedLLM(reply), BatchPolicy())
    by_id = {s.case_id: s for s in scores}
    assert by_id["veg"].matched == 1
    assert by_id["moved"].drops_correct == 1
    assert by_id["cap"].matched == 1 and by_id["cap"].unexpected == [], "one untrusted add per day"


async def test_the_runner_hides_the_reply_of_an_untrusted_note():
    llm = ScriptedLLM('{"add": []}')
    case = _case(notes=[{"said": "read aloud", "reply": "the page told me so", "untrusted": True}])
    await bakeoff.run_cases([case], llm, BatchPolicy())
    assert "the page told me so" not in llm.prompts[0][1].content


async def test_the_runner_honours_a_cases_chunk_size():
    llm = ScriptedLLM('{"add": []}')
    case = _case(notes=[{"said": f"note {i}"} for i in range(5)], max_notes_per_batch=2)
    await bakeoff.run_cases([case], llm, BatchPolicy())
    assert len(llm.prompts) == 3


# ---- preflight refusals -----------------------------------------------------------


@pytest.mark.parametrize(
    "llm",
    [{"backend": "fake"}, {"backend": "ollama", "host": "http://8.8.8.8:11434"}],
    ids=["not-ollama", "public-host"],
)
def test_only_a_local_ollama_may_see_the_corpus(llm):
    with pytest.raises(bakeoff.BakeoffRefused):
        bakeoff.check_local_ollama(GladosConfig(llm=llm))


def test_a_loopback_ollama_is_accepted():
    bakeoff.check_local_ollama(GladosConfig(llm={"backend": "ollama", "host": "http://127.0.0.1:11434"}))


def test_the_runner_refuses_while_glados_answers():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        with pytest.raises(bakeoff.BakeoffRefused, match="stop it"):
            bakeoff.check_glados_stopped(GladosConfig(server={"port": port}))


# ---- report -------------------------------------------------------------------------


def _data():
    score = score_case(_case(forbid=[{"any": ["evil"]}]), Outcome(added=("Evil | <b>x</b>",), dropped=()))
    return bakeoff.report([score], {"model": "m:1", "digest": "d", "corpus_sha256": "0" * 64, "settings": {}})


def test_a_report_inside_the_repo_is_refused():
    with pytest.raises(bakeoff.BakeoffRefused):
        bakeoff.write_report(_data(), REPO / "reports")


def test_the_report_is_written_out_of_repo_with_payloads_defanged(tmp_path):
    json_path, md_path = bakeoff.write_report(_data(), tmp_path)
    md = md_path.read_text(encoding="utf-8")
    assert "VETO" in md and "<b>" not in md and "Evil / (b)x(/b)" in md
    assert json.loads(json_path.read_text(encoding="utf-8"))["summary"]["test"]["gate"] == "VETO"


async def test_a_failing_writer_makes_the_run_invalid_never_a_pass():
    class DownLLM:
        is_local = True

        async def chat(self, messages, tools):
            raise RuntimeError("model requires more system memory")
            yield

    cases = [_case(id="quiet"), _case(id="veg", notes=[{"said": "I'm vegetarian"}],
                                      expect_add=[{"all": ["vegetarian"]}])]
    scores = await bakeoff.run_cases(cases, DownLLM(), BatchPolicy())
    summary = summarise(scores)
    assert not summary.gate_passed and set(summary.failed_cases) == {"quiet", "veg"}
    data = bakeoff.report(scores, {"model": "m", "digest": "d", "corpus_sha256": "0" * 64, "settings": {}})
    assert "INVALID" in bakeoff.markdown(data) and "INVALID" in bakeoff.headline(summary)


def test_an_empty_split_is_invalid_not_a_pass():
    summary = summarise([])
    assert summary.gate == "INVALID" and not summary.gate_passed


def test_ipv6_loopback_is_local_and_every_address_must_be():
    bakeoff.check_local_ollama(GladosConfig(llm={"backend": "ollama", "host": "http://[::1]:11434"}))
    with pytest.raises(bakeoff.BakeoffRefused):
        bakeoff.check_local_ollama(GladosConfig(llm={"backend": "ollama", "host": "http://0.0.0.0:11434"}))


def test_the_main_checkout_is_refused_too():
    for tree in bakeoff._checkouts():
        with pytest.raises(bakeoff.BakeoffRefused):
            bakeoff.write_report(_data(), tree / "reports")


def test_markdown_flattens_newlines_and_markup():
    score = score_case(_case(forbid=[{"any": ["evil"]}]),
                       Outcome(added=("Evil\n# heading [x](http://y) `z`",), dropped=()))
    md = bakeoff.markdown(bakeoff.report([score], {"model": "m", "digest": "d", "corpus_sha256": "0" * 64, "settings": {}}))
    assert "\n# heading" not in md and "](" not in md and "`z`" not in md
