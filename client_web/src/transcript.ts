// Transcript rendering. Pure DOM, no global state besides the bound element.

import type { ServerMessage } from "./protocol";

type RowKind = "user" | "assistant" | "tool" | "system" | "error";

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
        this.row(
          "tool",
          `→ ${msg.server}.${msg.name}(${JSON.stringify(msg.args)})`,
          `call_id ${msg.call_id}`,
        );
        break;
      case "tool_result":
        this.row(
          "tool",
          msg.ok ? `← ${JSON.stringify(msg.content)}` : `← error: ${msg.error}`,
          `call_id ${msg.call_id}`,
        );
        break;
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
