// Transcript rendering. Pure DOM, no global state besides the bound element.

import type { ServerMessage } from "./protocol";

type RowKind = "user" | "assistant" | "tool" | "system" | "error";

function summariseArgs(args: Record<string, unknown> | undefined): string {
  if (!args || Object.keys(args).length === 0) return "";
  const parts = Object.entries(args).map(([k, v]) => `${k}=${summariseValue(v)}`);
  return parts.join(", ");
}

function summariseResult(content: unknown): string {
  if (content == null) return "ok";
  if (Array.isArray(content)) return `${content.length} item${content.length === 1 ? "" : "s"}`;
  if (typeof content === "object") {
    const obj = content as Record<string, unknown>;
    for (const key of ["text", "message", "summary", "status"]) {
      const v = obj[key];
      if (typeof v === "string" && v.length > 0) return summariseValue(v);
    }
    for (const v of Object.values(obj)) {
      if (Array.isArray(v)) return `${v.length} item${v.length === 1 ? "" : "s"}`;
    }
    return `${Object.keys(obj).length} field${Object.keys(obj).length === 1 ? "" : "s"}`;
  }
  return summariseValue(content);
}

function summariseValue(v: unknown): string {
  if (typeof v === "string") return v.length > 80 ? JSON.stringify(v.slice(0, 77) + "...") : JSON.stringify(v);
  return JSON.stringify(v);
}

export class Transcript {
  private liveBubbles = new Map<string, HTMLDivElement>();
  // Which brain handled each in-flight turn, so the assistant bubble can be
  // colour-coded primary vs. specialist. Set from route_notice, read when the
  // bubble is first created on the next assistant_delta.
  private routeBySession = new Map<string, "primary" | "specialist">();

  constructor(private readonly root: HTMLElement) {}

  ingest(msg: ServerMessage): void {
    switch (msg.type) {
      case "welcome":
        this.row("system", `session ${msg.session_id}`);
        break;
      case "user_transcript": {
        // Server echoes what it believes the user said. Voice-derived
        // text is tagged so the UI can flag STT mistranscriptions at a
        // glance (the whole reason this event exists).
        const bubble = this.row("user", msg.text);
        if (msg.source === "voice") bubble.parentElement?.classList.add("voice");
        break;
      }
      case "assistant_delta": {
        let bubble = this.liveBubbles.get(msg.session_id);
        if (!bubble) {
          bubble = this.row("assistant", "");
          // Tag the bubble with the brain that produced it (router on). With
          // the router off no route_notice arrives, so the bubble keeps its
          // default styling.
          const route = this.routeBySession.get(msg.session_id);
          if (route) bubble.classList.add(`llm-${route}`);
          this.liveBubbles.set(msg.session_id, bubble);
        }
        bubble.textContent = (bubble.textContent ?? "") + msg.text;
        this.scroll();
        break;
      }
      case "tool_call":
        this.toolRow(
          `-> ${msg.server}.${msg.name}(${summariseArgs(msg.args)})`,
          msg.args,
          `call_id ${msg.call_id}`,
        );
        break;
      case "tool_result": {
        const headline = msg.ok
          ? `<- ${summariseResult(msg.content)}`
          : `<- error: ${msg.error}`;
        this.toolRow(headline, msg.content ?? msg.error, `call_id ${msg.call_id}`);
        break;
      }
      case "tts_chunk":
        // Audio frames are handled by the (future) speaker module, not the transcript.
        break;
      case "route_notice": {
        // Remember the brain for this turn so the assistant bubble gets the
        // matching colour when it's created on the next delta.
        this.routeBySession.set(msg.session_id, msg.target);
        // Surface when a turn ran on (or fell through to) the specialist. The
        // primary path is the silent default; no note for it.
        if (msg.escalated) {
          // The primary reply already streamed into a bubble and came back
          // failed. Drop it so the specialist retry starts a clean bubble, and
          // mark the boundary.
          this.liveBubbles.delete(msg.session_id);
          this.row("system", `^ escalated to specialist -- ${msg.reason}`);
        } else if (msg.target === "specialist") {
          this.row("system", `routed to specialist -- ${msg.reason}`);
        }
        break;
      }
      case "turn_outcome":
        // Deterministic verdict on how the turn ended (see
        // core/turn_outcome.py). A `done` turn is the silent default -- no
        // badge. `needs-user` / `failed` get a badge so a turn the model
        // narrated cheerfully but didn't finish is visible at a glance.
        if (msg.outcome !== "done") this.outcomeBadge(msg.session_id, msg.outcome);
        break;
      case "done":
        this.liveBubbles.delete(msg.session_id);
        this.routeBySession.delete(msg.session_id);
        break;
      case "cancelled":
        this.row("system", `cancelled ${msg.session_id}`);
        this.liveBubbles.delete(msg.session_id);
        this.routeBySession.delete(msg.session_id);
        break;
      case "error":
        this.row("error", `${msg.code}: ${msg.message}`);
        break;
    }
  }

  systemNote(text: string): void {
    this.row("system", text);
  }

  private outcomeBadge(sessionId: string, outcome: "needs-user" | "failed"): void {
    const label = outcome === "failed" ? "task not completed" : "awaiting your input";
    const badge = document.createElement("span");
    badge.className = `outcome-badge outcome-${outcome}`;
    badge.textContent = label;
    // Prefer pinning the badge to the assistant bubble for this turn so it
    // reads as a verdict on that reply. If the turn produced no spoken text
    // (tool-only), there's no bubble -- fall back to a centred system row.
    const bubble = this.liveBubbles.get(sessionId);
    if (bubble) {
      bubble.appendChild(badge);
      this.scroll();
    } else {
      const r = this.row("system", "");
      r.appendChild(badge);
    }
  }

  private toolRow(headline: string, payload: unknown, meta: string): void {
    const r = document.createElement("div");
    r.className = "row tool";
    const b = document.createElement("details");
    b.className = "bubble tool-bubble";
    const summary = document.createElement("summary");
    summary.textContent = headline;
    b.appendChild(summary);
    const pre = document.createElement("pre");
    pre.className = "tool-payload";
    pre.textContent = JSON.stringify(payload, null, 2);
    b.appendChild(pre);
    if (meta) {
      const m = document.createElement("div");
      m.className = "meta";
      m.textContent = meta;
      b.appendChild(m);
    }
    r.appendChild(b);
    this.root.appendChild(r);
    this.scroll();
  }

  private row(kind: RowKind, text: string, meta?: string): HTMLDivElement {
    const r = document.createElement("div");
    r.className = `row ${kind}`;
    const b = document.createElement("div");
    b.className = "bubble";
    b.textContent = text;
    if (meta) {
      const m = document.createElement("div");
      m.className = "meta";
      m.textContent = meta;
      b.appendChild(m);
    }
    r.appendChild(b);
    this.root.appendChild(r);
    this.scroll();
    return b;
  }

  private scroll(): void {
    this.root.scrollTop = this.root.scrollHeight;
  }
}
