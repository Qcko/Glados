"""Boot-time proof that the worst-case prompt fits the model's window.

The threat is in `DESIGN-context-flooding.md`: Ollama truncates an over-long
prompt from the FRONT and says nothing, so an oversized tool result deletes the
system prompt carrying the section 7 `<external>` rule -- the defence that
governs the very payload doing the deleting.

`clamp_result_bytes` bounds one result and the organizer's history budget bounds
the tool bytes RETAINED across a session, so "does a retained session fit?" has
an answer at boot -- checked once, where a human is watching, rather than
measured on every turn while a listener waits.

**This is a necessary condition, not a proof, and the gap is named.** History is
capped when a turn COMMITS, so a turn in flight can carry more: the tool loop
runs several passes and each appends its own capped results before any of it is
committed. Bounding that needs the per-hop assertion the design calls B3, which
is not built. Until it is, a passing check here means "a retained session fits",
not "no prompt can overflow".

The history ceiling is what this prices, NOT the per-result cap multiplied by
the turn depth. Those two disagree, and the retained-bytes ceiling is the one
that actually binds: turn count cannot bound a session when the size of each
turn is the attacker's to choose.

Why the worst-case result is PRICED rather than calculated. Converting a byte
ceiling into tokens needs a ratio, and every constant is wrong in one direction:
`tokens <= bytes` is sound but so pessimistic it would force absurdly small
caps, while a prose-derived ratio like 4 bytes per token is exactly the estimate
an attacker gets to choose content against. So the check measures instead: it
builds a payload as token-dense as the clamp permits and asks the model what it
costs. Dense, unbroken, punctuation-heavy ASCII is close to the worst a BPE
tokenizer does with a byte budget -- and, unlike prose, it is what a scraped
page of minified JSON or base64 actually looks like.

Nothing here decides policy. It reports, and the caller refuses to boot.
"""

from __future__ import annotations

from dataclasses import dataclass

from .adapters import LLMMessage, ToolSpec

# Alphabet for the pricing payload. Deliberately nothing like English -- no
# word boundaries for the tokenizer to exploit -- and deliberately NOT a
# repeated unit: a byte-level BPE tokenizer gives every repetition of the same
# string identical merges, so a repeated payload prices at whatever that one
# unit happens to cost and compresses far better than real hostile content.
# That is the direction that silently passes a configuration which overflows in
# production, so the payload is generated non-repeating from a fixed seed.
_DENSE_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789{}[]<>|/_-+=*&^%$#@!?:;,."
_DENSE_SEED = 0x5EED

# A repeated punctuation-heavy unit. Kept alongside the pseudo-random walk
# because MEASUREMENT contradicted the intuition: repetition was expected to
# compress under BPE and price optimistically, but on ministral3:8b-instruct it
# is the DENSER of the two -- 1.14 bytes/token against 1.27 for the random walk
# at 4096 bytes. Neither is provably the worst case, so the budget prices both
# and believes the worse one rather than picking by argument.
_DENSE_UNIT = "x7Q}{~2:_#9aZ|/%3&*8Kv$@1"

# Bytes per token the boot budget assumes. Measured 01-09-2026 on
# ministral3:8b-instruct: dense content 1.14, English prose 4.49. If the
# pricing payload comes out LOOSER than this it is compressing, the budget is
# under-priced, and the check would pass configurations that overflow -- so the
# achieved density is asserted rather than trusted.
_MAX_CREDIBLE_BYTES_PER_TOKEN = 1.6


@dataclass(frozen=True)
class BudgetVerdict:
    fits: bool
    fixed_prefix_tokens: int
    retained_external_tokens: int
    worst_case_total: int
    num_ctx: int
    num_predict: int
    detail: str


def dense_payload(byte_budget: int) -> str:
    """`byte_budget` ASCII bytes that tokenize about as badly as the clamp
    allows. ASCII by construction, so bytes and characters agree.

    A linear congruential walk over the alphabet rather than a repeated unit:
    deterministic (so a boot check is reproducible) but without the repetition
    a BPE tokenizer would merge away.
    """
    if byte_budget <= 0:
        return ""
    out = []
    state = _DENSE_SEED
    for _ in range(byte_budget):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(_DENSE_ALPHABET[state % len(_DENSE_ALPHABET)])
    return "".join(out)


def repeated_payload(byte_budget: int) -> str:
    """The other candidate worst case -- see `_DENSE_UNIT`."""
    if byte_budget <= 0:
        return ""
    repeats = byte_budget // len(_DENSE_UNIT) + 1
    return (_DENSE_UNIT * repeats)[:byte_budget]


def worst_case_total(
    fixed_prefix_tokens: int,
    retained_external_tokens: int,
    num_predict: int,
) -> int:
    """The largest prompt a session can assemble, plus the reply it must leave
    room for.

    `num_predict` is a reservation, never a lever. Shrinking it to make a
    prompt fit is a measured regression, not a saving: at 512 the shipped
    qwen3:4b scored 4/22, and the failure mode is an empty reply that logs as
    a success.
    """
    return fixed_prefix_tokens + retained_external_tokens + num_predict


def verdict_for(
    fixed_prefix_tokens: int,
    retained_external_tokens: int,
    *,
    num_ctx: int,
    num_predict: int,
) -> BudgetVerdict:
    total = worst_case_total(
        fixed_prefix_tokens, retained_external_tokens, num_predict
    )
    fits = total <= num_ctx
    return BudgetVerdict(
        fits=fits,
        fixed_prefix_tokens=fixed_prefix_tokens,
        retained_external_tokens=retained_external_tokens,
        worst_case_total=total,
        num_ctx=num_ctx,
        num_predict=num_predict,
        detail=_detail(total, fits, num_ctx),
    )


def _detail(total: int, fits: bool, num_ctx: int) -> str:
    if fits:
        return f"worst-case prompt {total} tokens fits num_ctx {num_ctx}"
    return (
        f"retained-session prompt {total} tokens EXCEEDS num_ctx {num_ctx}: an "
        f"over-long session would be truncated from the front, silently "
        f"deleting the system prompt. Raise llm.num_ctx, shorten "
        f"llm.system_prompt, or lower the organizer's "
        f"max_history_external_bytes (currently constructor-only, in "
        f"core/organizer.py)."
    )


async def measure_boot_budget(
    llm: object,
    *,
    system_prompt: str,
    tools: list[ToolSpec],
    max_history_external_bytes: int,
    num_ctx: int,
    num_predict: int,
) -> BudgetVerdict | None:
    """Price the two unknowns against the live model, then judge.

    Returns None when the backend cannot price a prompt -- a fake in tests, or
    an adapter whose window is not its own to know. An unknown budget is not a
    failed one; the caller decides what to do about it.
    """
    price = getattr(llm, "price_prompt", None)
    if price is None:
        return None
    fixed = await price([LLMMessage(role="system", content=system_prompt)], tools)
    if fixed is None:
        return None
    # The whole retained-bytes ceiling in one message, priced under BOTH
    # candidate worst cases. Message framing costs a few scaffolding tokens
    # either way; the payload is what has to be right.
    prices = []
    for shape in (dense_payload, repeated_payload):
        loaded = await price(
            [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(
                    role="user", content=shape(max_history_external_bytes)
                ),
            ],
            tools,
        )
        if loaded is None:
            return None
        prices.append(max(0, loaded - fixed))
    retained_tokens = max(prices)
    _reject_a_compressible_price(max_history_external_bytes, retained_tokens)
    return verdict_for(
        fixed,
        retained_tokens,
        num_ctx=num_ctx,
        num_predict=num_predict,
    )


def _reject_a_compressible_price(byte_budget: int, tokens: int) -> None:
    """The whole budget rests on the pricing payload being hostile-dense.

    If it prices looser than real adversarial content, every figure downstream
    is optimistic and the boot check passes configurations that overflow. That
    failure is silent, so it is asserted here rather than assumed.
    """
    if tokens <= 0:
        raise RuntimeError("prompt pricing returned no tokens for a non-empty payload")
    achieved = byte_budget / tokens
    if achieved > _MAX_CREDIBLE_BYTES_PER_TOKEN:
        raise RuntimeError(
            f"prompt pricing payload compressed to {achieved:.2f} bytes/token, "
            f"looser than the {_MAX_CREDIBLE_BYTES_PER_TOKEN} this budget "
            f"assumes -- it is under-pricing the worst case and the check "
            f"would pass a configuration that overflows"
        )
