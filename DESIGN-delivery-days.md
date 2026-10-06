# DESIGN - delivery slots by day

Status: APPROVED 06-10-2026 (design panel: architect, security, reliability,
clean-slate; option "server checks the arguments against the page").

## The problem

Live on prod, 06-10-2026: "lets check out" -> `list_delivery_slots` returned one
slot with no date (`8:00am - 10:00am, EUR 9.00, 1 Slot Left`). "What date?" could
not be answered; "Sunday", "Saturday", "Friday" each re-listed and got the same
slot. Nothing was booked.

Cause, in DunnesStoresMCP `CheckoutPage.ReadAvailableSlots`: it reads only the
slot panel of the day tab the checkout page opens on. The page has a day picker
it never reads or clicks, so slots carry no date and other days are unreachable.

## The page (read live over CDP, 06-10-2026)

- Day tabs: `div[role=tablist][aria-label="Available Days"] button[role=tab]`,
  `id="timeslotDay-<Wed|Today|...>"` (the SAME id recurs every week),
  `aria-selected`, two text divs: weekday label ("Wed" / "Today") and month-day
  ("Oct 7"). No year.
- Seven tabs per week; "Load Previous/Next Timeslot Week" buttons page by week.
- Panel: `ul[role=tabpanel][aria-labelledby=<selected tab id>]`; slots inside it
  (`timeslot_gridSlotComponent-div-testId`, element `id` = slotId GUID).
- Opens on the first day with availability. "Show only available times" stays on.
- The same section also shows the delivery ADDRESS -- never read by this code.

## Design

### Dates are decided by tested code, never by the model or the page alone

`DeliveryDates` (pure, no driver, unit-tested with an injected today):
- "Today" is Irish local time (`Europe/Dublin` / `GMT Standard Time`), not the
  host clock's zone.
- A tab's "Oct 7" becomes the date in the window [today - 1, today + 20] --
  the year is whichever makes it fall there; outside the window it is unparsed.
- A request resolves from: omitted, `today`, `tomorrow`, a weekday name (full or
  3-letter; the next such day on or after today), or strict ISO `yyyy-MM-dd`.
  Anything else is refused naming the accepted forms. Bookable range: today to
  today + 13; outside it, refused naming the first and last bookable dates.

### Server (DunnesStoresMCP)

- `CheckoutPage.SelectDay(date)`, used by BOTH tools: from a freshly loaded
  checkout page, pages forward week by week (at most 2) until a tab carries the
  date, clicks it unless already selected, and waits. The wait passes when the
  tab is selected, the panel is labelled by it, and either slots are present
  whose ids are DISJOINT from the ids seen before the click, or the panel has
  stayed empty for the tab settle time (its own constant, measured, not the
  first-load one). Tab ids repeat across weeks, so the id-set change is what
  proves the panel re-rendered. Paging waits for the first tab's date to change.
- Every slot read is scoped to `ul[role=tabpanel]`. Raw page strings never leave
  the server: time becomes `HH:mm-HH:mm` (24 h), the fee a number, "N Slot(s)
  Left" an integer or null. A slot whose time or fee does not parse is dropped
  and logged.
- `list_delivery_slots(date?)`: navigates to checkout every call, selects the
  day (omitted: the day the page opens on). Returns
  `{requested, date, weekday, slots:[{slotId, time, fee, left}], days:[{date, weekday}], bookable:{first, last}}`.
  No slots is `slots: []` for that date -- an answer, never a fallback day.
- `set_delivery_slot(slotId, date, time, fee)`, all required, all copied from a
  listing. slotId must be a GUID, date strict ISO. The server navigates to
  checkout, selects the date, and finds slotId INSIDE that panel. Then:
  - not there -> failed: "no longer offered on <date>", nothing clicked;
  - its time or fee differs from the arguments -> failed naming both, nothing
    clicked;
  - already reserved -> ok "already reserved", nothing clicked;
  - otherwise click Select, then confirm the reservation on the page; unconfirmed
    -> failed ("clicked but could not confirm -- check the browser"), per
    ToolOutcome's rule that an unconfirmed write is a failure.
  The ok result is built from the parsed page fields, never page text.
- ToolGuard never retries, so the click runs at most once per call. A re-send
  after a timeout lands on "already reserved" (or re-selects the same slot --
  the site holds one reservation per basket, so this is not a second booking).

### GLaDOS

- The money modal already lists the call's arguments, so date, time and fee
  are on screen at approval -- and the server books only when all three match
  the page. The approved slot is the booked slot.
- No `confirm_phrase`: a phrase carries at most one free-text argument and
  this call has three, so the generic spoken form names them. Voice never
  authorises it: `money_step` keeps the desk modal as the only gate. No
  organizer change.

```mermaid
flowchart TD
    u["user: book something for Sunday"] --> list["list_delivery_slots<br/>(date='sunday')"]
    list --> resolve{"DeliveryDates: resolve to ISO<br/>(Irish today, 14-day range)"}
    resolve -- "unknown form / out of range" --> refuse["refusal naming accepted<br/>forms or bookable range"]
    resolve --> sel1["SelectDay: fresh checkout,<br/>page weeks, click tab, wait<br/>for new slot ids or settled empty"]
    sel1 --> read["parsed slots of THAT panel:<br/>slotId, HH:mm-HH:mm, fee, left"]
    read --> model["model offers a slot"]
    model --> set["set_delivery_slot(slotId,<br/>date, time, fee)"]
    set --> gate["GLaDOS money_step gate:<br/>modal shows cart + date, time, fee"]
    gate -- denied --> stop["nothing booked"]
    gate -- approved --> sel2["SelectDay(date) from a<br/>fresh checkout page"]
    sel2 --> check{"slotId in this panel,<br/>time + fee match?"}
    check -- "no" --> fail["failed, nothing clicked"]
    check -- "already reserved" --> done["ok: already reserved"]
    check -- "yes" --> click["click Select"]
    click --> confirm{"reservation<br/>confirmed on page?"}
    confirm -- yes --> ok["ok: date + time + fee<br/>from parsed page fields"]
    confirm -- no --> unk["failed: check the browser"]
```

## Open / deferred

- The page's "reserved" marker was never seen live (selecting a slot is the
  money step). The confirm and already-reserved checks are built on the
  observed `#labelReserved` label and the "Continue to Vouchers" button
  enabling; the first real booking verifies them.
- `days` carries no per-day availability: knowing it costs a tab click per day.
- No part-of-day filter ("Saturday morning"); the model filters the list.
- Booking beyond 14 days; Pickup mode; changing store/address.
