# Design note — WS handshake rate-limiting / throttling

Status: implemented (see `src/glados/core/handshake_gate.py`).
Brief: `ROADMAP.md` § "Future: WS handshake rate-limiting / throttling".
Reviewed by a three-lens design panel (security, concurrency/reliability,
architect) before coding; adjudications below.

## Threat model

Token entropy (256-bit) plus the constant-time compare already make online
brute force infeasible. The remaining exposure with the server on `0.0.0.0`
is **resource exhaustion** (an unauthenticated peer holding many pending
handshakes) and **unobserved probing** (failures nobody notices). Controls
are sized for a home LAN: tens of clients, one operator, no reverse proxy.

## Chosen controls (all three, they cover different gaps)

1. **Handshake timeout** (`asyncio.timeout`, default 10 s) — inline in the
   WS handler, not in the gate: it is a per-handshake concern with no shared
   state. Bounds how long an unauthenticated socket can sit pending.
2. **Concurrent pending-handshake caps** — a global cap (default 8) plus a
   per-source-IP cap (default 2). The per-IP cap stops a single peer from
   holding every global slot and re-acquiring them after each timeout, which
   would otherwise deny reconnects to everyone. Over either cap the socket
   is closed immediately with a retryable error.
3. **Per-source-IP failure lockout** — 5 token-auth failures within 60 s
   locks that IP out for 60 s. Locked-out connects are rejected before the
   hello is read. State is a bounded dict (default 1024 IPs); eviction
   prunes expired entries first and falls back to oldest-activity only when
   still over the cap. A successful handshake clears the IP's failure state.

Only the two credential failures (`unknown client_id`, bad token) count
toward lockout. Malformed hellos and room/role binding mismatches do not:
a binding mismatch means a *valid token holder* is misconfigured, and a
fuzzer sending garbage is bounded by the caps and timeout instead.

## Observability

Every counted auth failure logs a WARNING with the source IP. Lockout
engage and expiry log once each; failures arriving during a lockout do not
log per-attempt, so a probe cannot flood the log. A probe in progress is
therefore visible as: N failure lines, one lockout line.

## Adjudicated panel tensions

- **Lockout can briefly deny a valid client (security must-fix, partially
  rejected).** The security lens asked that a valid token always bypass the
  lockout (verify first, throttle only failures). That would make the
  lockout client-invisible — its only effect would be logging — because
  invalid attempts are rejected with `auth_failed` either way. We keep a
  real pre-hello lockout and mitigate the legit-client risk instead: the
  lockout is short (60 s, self-expiring), it takes five failures in a
  minute to engage (the room client stops on terminal handshake errors, so
  it cannot self-hammer), and the rejection uses a **distinct, retryable
  error** (`rate_limited`, WS close 1013) so an operator debugging a
  misconfigured device sees the truth rather than a misleading
  `auth_failed`. Revealing the throttle to a LAN attacker is acceptable:
  brute force is already infeasible by entropy, and the operator-facing
  clarity is worth more on a home appliance. IP spoofing of an established
  TCP+WS session is not feasible on a switched LAN without an on-path
  position, which would defeat any per-IP scheme regardless.
- **No reverse proxy is assumed.** The gate keys on the transport peer
  address (`ws.client.host`). Behind a proxy/TLS terminator every client
  would share one IP and the per-IP controls would misfire; that deployment
  is out of scope and documented on the config model. The gate takes the
  resolved IP as a plain argument so a trusted-header resolution could be
  added later without touching it.
- **Pairing channel (deferred, by design).** The caps + timeout transfer
  directly to the future pre-auth pairing endpoint; the credential-lockout
  does not (pairing has no token to fail). The gate keeps the two concerns
  independently callable, but no speculative "endpoint" abstraction is
  added for a caller that does not exist yet.

## Config

Nested `[handshake]` table on `glados.toml` (`HandshakeConfig`), env
overrides `GLADOS_HANDSHAKE_*`, safe defaults, on by default:

| field                | default | env override                        |
|----------------------|---------|-------------------------------------|
| `timeout_s`          | 10.0    | `GLADOS_HANDSHAKE_TIMEOUT_S`        |
| `max_pending`        | 8       | `GLADOS_HANDSHAKE_MAX_PENDING`      |
| `max_pending_per_ip` | 2       | `GLADOS_HANDSHAKE_MAX_PENDING_PER_IP` |
| `fail_threshold`     | 5       | `GLADOS_HANDSHAKE_FAIL_THRESHOLD`   |
| `fail_window_s`      | 60.0    | `GLADOS_HANDSHAKE_FAIL_WINDOW_S`    |
| `lockout_s`          | 60.0    | `GLADOS_HANDSHAKE_LOCKOUT_S`        |
| `max_tracked_ips`    | 1024    | `GLADOS_HANDSHAKE_MAX_TRACKED_IPS`  |

## Concurrency shape

The gate is a plain synchronous object (single event loop; every method is
atomic between awaits). The pending slot is acquired in `ws_v1` *before*
`_handshake` is awaited and released in a `finally` anchored in `ws_v1`, so
every exit path — validation error, auth failure, disconnect, timeout,
cancellation at shutdown — releases exactly once. `CancelledError` is never
swallowed. All `ws.close()` calls on reject paths are best-effort (a peer
that already vanished must not turn a reject into a 500). The clock is
injectable (`time.monotonic` by default) so tests drive time directly.
