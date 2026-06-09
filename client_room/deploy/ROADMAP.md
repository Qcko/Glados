# Deploy & enrollment roadmap

Forward-looking design for how a device becomes a GLaDOS room appliance.
**Vision, not yet built** — the operational how-to for what exists *today* is
[`termux/README.md`](termux/README.md). This note records the target shape so
the small slices we ship stay pointed at it. Touches ARCHITECTURE.md §7
(untrusted content), §9 (privacy & security boundaries), §14 (human-gated
trust).

## Three tools, three jobs

Today's `termux/` scripts collapse several concerns into one self-test bundle.
The target separates them by **secret posture** and **cadence**:

1. **doctor** — *"is this device healthy?"* Today's `go.sh` / `selftest.sh`.
   Read-only diagnostics of the phone's audio plumbing (tools present, Pulse up,
   mic source loads, capture non-silent, playback audible, client imports).
   **Secret-free, reusable, idempotent, re-runnable.** Already essentially this.

2. **install** — *"make this device a GLaDOS appliance."* The persistent-install
   path (extends today's self-deleting bundle): extract the client + deps, wire
   the runit services + Termux:Boot, leave `~/glados` in place. **Should be
   secret-free too** — that is what lets the install artifact stay reusable and
   non-ephemeral, instead of a gitignored secret-bearing bundle.

3. **enroll / config** — *"give this device its identity."* Obtains the room
   token(s) and writes `config.room.toml` + mode-600 token files. This is the
   only secret-bearing step, and in the target it is **transient** — see below.

### Why install and enroll are separate

- **Secret posture** — install is secret-free and identical across devices;
  enroll is per-device. Splitting keeps the install artifact non-ephemeral.
- **Cadence** — install once; re-enroll (rotate tokens, move a device to a new
  room) without reinstalling.
- **It removes the secret-bundle problem.** If tokens are minted server-side at
  enroll time and handed back over the wire, **no secret ever rides in a file
  copied to the device.** Strictly better than baking tokens into a provisioning
  bundle on the dev box.

## Enrollment: out-of-band confirmation pairing

The chosen model is **operator-confirmed pairing** (Bluetooth / Chromecast
"enter the code shown on screen"), *not* a code carried from server to device.
The number is shown on an **already-trusted surface** and typed into the **new**
device — so the trusted side is the authority, and a LAN impostor that can open
the channel still can't see the number on your screen.

Flow:

1. A freshly-installed device opens an **untrusted pairing conversation** with
   GLaDOS and declares its `room_id` + device type (`mic` / `speaker` / both).
2. GLaDOS shows a **pairing number** on an already-connected trusted client /
   admin surface.
3. The operator types that number into the new device. On match, GLaDOS **mints
   the room token(s)**, registers them (rooms.toml + `auth.clients` + keyring),
   and returns them.
4. The new device writes `config.room.toml` + mode-600 token files and runs
   under the enroll/config step.

The operator transfers only a **short, hand-typeable number** — never a secret
file. (Also satisfies the on-device-script standard's "minimize typed commands".)

## Open design points (resolve when this becomes a slice)

1. **The untrusted pre-pairing channel is a new trust-boundary surface (§7).**
   It must do *only* pairing — no tools, audio, or memory access — and be
   rate-limited and TTL'd. This is the part to design most carefully (own duck
   panel).
2. **Bind the number to the specific pending request** — short TTL, single-use,
   attempt-limited. A 6-digit code is brute-forceable without a lockout.
3. **Who may approve** — the surface that displays + confirms must itself hold an
   admin/approve capability, or any paired `ui` client could authorize
   newcomers.
4. **Transport** — the token hand-back rides the wire, so **TLS (`wss://`) is a
   hard prerequisite**, not a competitor to pairing. Encrypt the wire first.

## Dependency order & where we are today

```
TLS (wss://, no embedded creds)  →  pairing/enrollment protocol  →  config.sh
        [in progress]                      [future epic]            [future]
```

- **Auth layer** is already token-based (per-client tokens validated at the WS
  handshake against the keyring). Enrollment changes *how tokens are issued*
  (server-minted at pairing) — it is not a new auth paradigm and does **not**
  replace TLS. Federated identity (OAuth/OIDC) is explicitly out of scope for a
  self-hosted home appliance.
- **Today**, a device is brought up with a **hand-written `config.room.toml` +
  manually-minted tokens** (see termux/README.md). That manual config is the
  exact artifact the enroll step will later generate — writing it by hand first
  de-risks the schema.
