# rule-guard:allow ascii-source - the curly apostrophe is the left-hand side of a normalisation replace(); rewriting it to ASCII turns the call into a silent no-op.
"""Word Error Rate (WER) — Levenshtein distance over word tokens.

Standalone (no external deps). Used by `scripts/stt_benchmark.py` to make
the English baseline comparable against future multilingual-STT runs:
swap the model, re-run the same manifest, diff the WER summaries.

WER = (substitutions + deletions + insertions) / reference_word_count

Tokenization is deliberately tolerant: lowercase, strip punctuation,
collapse whitespace. ASR outputs vary in capitalisation and trailing
punctuation across model versions; counting those as errors would
swamp the meaningful substitutions we actually care about.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PUNCT_RE = re.compile(r"[^\w\s']+", re.UNICODE)
_WS_RE = re.compile(r"\s+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    # U+2019 (curly apostrophe, used by Common Voice exports) → ASCII so
    # cross-corpus comparison doesn't count "don't" vs "don't" as a
    # substitution. faster-whisper emits straight apostrophes.
    text = text.replace("’", "'").lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text.split(" ") if text else []


@dataclass(frozen=True)
class WerResult:
    wer: float
    substitutions: int
    deletions: int
    insertions: int
    ref_words: int
    hyp_words: int


def wer(reference: str, hypothesis: str) -> WerResult:
    ref = tokenize(reference)
    hyp = tokenize(hypothesis)
    s, d, i = _edit_counts(ref, hyp)
    denom = len(ref) or 1  # match jiwer: empty reference yields 0/1=0 not divide-by-zero
    rate = (s + d + i) / denom if ref else (1.0 if hyp else 0.0)
    return WerResult(
        wer=rate,
        substitutions=s,
        deletions=d,
        insertions=i,
        ref_words=len(ref),
        hyp_words=len(hyp),
    )


def _edit_counts(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """Standard DP backtrace returning (substitutions, deletions, insertions).

    Matches the canonical WER decomposition: deletions remove a ref word,
    insertions add a hyp word, substitutions swap a ref word for a hyp word.
    """
    n, m = len(ref), len(hyp)
    # dp[i][j] = edit distance between ref[:i] and hyp[:j].
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # deletion
                dp[i][j - 1] + 1,        # insertion
                dp[i - 1][j - 1] + cost, # match or substitution
            )
    return _backtrace(dp, ref, hyp)


def _backtrace(
    dp: list[list[int]], ref: list[str], hyp: list[str]
) -> tuple[int, int, int]:
    i, j = len(ref), len(hyp)
    s = d = ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            s += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            d += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return s, d, ins
