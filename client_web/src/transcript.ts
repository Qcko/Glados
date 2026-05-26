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
  if (typeof v === "string") return v.length > 80 ? JSON.stringify(v.slice(0, 77) + "…") : JSON.stringify(v);
  return JSON.stringify(v);
}

export class Transcript {
  private liveBubbles = new Map<string, HTMLDivElement>();

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
          this.liveBubbles.set(msg.session_id, bubble);
        }
        bubble.textContent = (bubble.textContent ?? "") + msg.text;
        this.scroll();
        break;
      }
      case "tool_call":
        this.toolRow(
          `→ ${msg.server}.${msg.name}(${summariseArgs(msg.args)})`,
          msg.args,
          `call_id ${msg.call_id}`,
        );
        break;
      case "tool_result": {
        const headline = msg.ok
          ? `← ${summariseResult(msg.content)}`
          : `← error: ${msg.error}`;
        this.toolRow(headline, msg.content ?? msg.error, `call_id ${msg.call_id}`);
        break;
      }
      case "tts_chunk":
        // Audio frames are handled by the (future) speaker module, not the transcript.
        break;
      case "done":
        this.liveBubbles.delete(msg.session_id);
        break;
      case "cancelled":
        this.row("system", `cancelled ${msg.session_id}`);
        this.liveBubbles.delete(msg.session_id);
        break;
      case "error":
        this.row("error", `${msg.code}: ${msg.message}`);
        break;
    }
  }

  systemNote(text: string): void {
    this.row("system", text);
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
