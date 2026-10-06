"""The cart as shown on the checkout review (DESIGN-checkout-reconcile.md).

Display only. Unlike `cart_verify.CartSnapshot`, this keeps the shop's product
names, so it must never feed anything spoken, the model, history or traces --
it exists to fill one confirm dialog on a cart_view screen.

Same contract and the same all-or-nothing strictness as `CartSnapshot`; names
are clipped by code points and by graphemes; totals are exact decimals.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from glados.core.cart_verify import CartSnapshot
from glados.core.protocols import CartReviewLine, CartReviewPayload

_MAX_NAME_CODEPOINTS = 120
_MAX_NAME_GRAPHEMES = 60
_MAX_MONEY = Decimal(10000)
_CENT = Decimal("0.01")


@dataclass(frozen=True)
class CartReview:
    snapshot: CartSnapshot
    payload: CartReviewPayload
    digest: str

    @classmethod
    def from_content(cls, content: object) -> CartReview | None:
        """The review of one `view_cart` result, or None if any part of it is
        off contract. The quantities are exactly what `CartSnapshot` reads."""
        snapshot = CartSnapshot.from_content(content)
        if snapshot is None or not isinstance(content, dict):
            return None
        names = _names_by_id(content.get("lines"))
        totals = _totals(content)
        if names is None or totals is None:
            return None
        payload = _payload(snapshot, names, totals)
        return cls(snapshot=snapshot, payload=payload, digest=_digest(payload))


def _names_by_id(raw_lines: object) -> dict[str, str] | None:
    if not isinstance(raw_lines, list):
        return None
    names = {}
    for line in raw_lines:
        name = line.get("name", "") if isinstance(line, dict) else None
        if not isinstance(name, str):
            return None
        names[line["productId"]] = _clipped(name)
    return names


def _totals(content: dict) -> dict[str, object] | None:
    totals: dict[str, object] = {}
    for key, out in (("orderValue", "order_value"), ("estimatedTotal", "estimated_total")):
        if key not in content:
            continue
        money = _money(content[key])
        if money is None:
            return None
        totals[out] = str(money)
    count = content.get("itemCount")
    if count is not None:
        if type(count) is not int or not 0 <= count <= 99_999:
            return None
        totals["item_count"] = count
    return totals


def _money(value: object) -> Decimal | None:
    """A non-negative amount under 10000 with at most two decimals, read from
    its JSON text so no float rounding happens on the way."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        amount = Decimal(json.dumps(value))
    except InvalidOperation:
        return None
    if not amount.is_finite() or amount < 0 or amount >= _MAX_MONEY:
        return None
    if amount != amount.quantize(_CENT):
        return None
    return amount.quantize(_CENT)


def _clipped(name: str) -> str:
    """At most 120 code points and 60 graphemes (a base character plus its
    combining marks), so a run of marks cannot tower out of its cell."""
    kept: list[str] = []
    graphemes = 0
    for char in name[:_MAX_NAME_CODEPOINTS]:
        if not unicodedata.combining(char):
            graphemes += 1
            if graphemes > _MAX_NAME_GRAPHEMES:
                break
        kept.append(char)
    return "".join(kept)


def _payload(
    snapshot: CartSnapshot, names: dict[str, str], totals: dict[str, object]
) -> CartReviewPayload:
    lines = [
        CartReviewLine(
            product_id=product_id,
            name=names.get(product_id, ""),
            quantity=line.units,
            pack_of=line.pack_of,
        )
        for product_id, line in snapshot.lines.items()
    ]
    return CartReviewPayload(lines=lines, line_count=len(lines), **totals)


def _digest(payload: CartReviewPayload) -> str:
    """What an approval binds to: every line's id and quantity plus the
    totals. Names are left out -- a shop renaming a product is not a
    different cart."""
    bound = {
        "lines": sorted((l.product_id, l.quantity, l.pack_of) for l in payload.lines),
        "item_count": payload.item_count,
        "order_value": payload.order_value,
        "estimated_total": payload.estimated_total,
    }
    return hashlib.sha256(json.dumps(bound, sort_keys=True).encode()).hexdigest()

