// WebSocket lifecycle, hello handshake, auto-reconnect with backoff.

import type { ClientMessage, ServerMessage } from "./protocol";
import type { Settings } from "./settings";
import type { StateMachine } from "./state";

export type ServerListener = (msg: ServerMessage) => void;
export type FirstReplyListener = (firstReplyMs: number) => void;

const RECONNECT_DELAYS_MS = [500, 1000, 2000, 5000, 10_000];

export class Transport {
  private ws: WebSocket | null = null;
  private settings: Settings | null = null;
  private wantOpen = false;
  private attempt = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private serverListeners = new Set<ServerListener>();
  private firstReplyListeners = new Set<FirstReplyListener>();
  private lastSendAt: number | null = null;

  constructor(private readonly state: StateMachine) {}

  onServerMessage(fn: ServerListener): () => void {
    this.serverListeners.add(fn);
    return () => this.serverListeners.delete(fn);
  }

  onFirstReply(fn: FirstReplyListener): () => void {
    this.firstReplyListeners.add(fn);
    return () => this.firstReplyListeners.delete(fn);
  }

  connect(settings: Settings): void {
    this.clearReconnect();
    this.settings = settings;
    this.wantOpen = true;
    this.attempt = 0;
    this.openSocket();
  }

  disconnect(): void {
    this.wantOpen = false;
    this.clearReconnect();
    if (this.ws) {
      try { this.ws.close(1000, "client disconnect"); } catch { /* ignore */ }
    }
    this.state.set({ kind: "disconnected" });
  }

  send(msg: ClientMessage): boolean {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return false;
    this.lastSendAt = performance.now();
    this.ws.send(JSON.stringify(msg));
    return true;
  }

  sendBinary(data: ArrayBuffer): boolean {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return false;
    this.ws.send(data);
    return true;
  }

  private openSocket(): void {
    if (!this.settings) return;
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const url = `${proto}://${location.host}/ws/v1`;
    this.state.set({ kind: "connecting", attempt: this.attempt });

    const ws = new WebSocket(url);
    this.ws = ws;

    ws.onopen = () => {
      const s = this.settings!;
      ws.send(JSON.stringify({
        type: "hello",
        client_id: s.clientId,
        room_id: s.roomId,
        role: "ui",
        token: s.token,
      }));
      this.attempt = 0;
      this.state.set({ kind: "ready", clientId: s.clientId, roomId: s.roomId });
    };

    ws.onmessage = (ev) => {
      this.recordLatency();
      if (typeof ev.data !== "string") {
        // Binary frames will be claimed by the audio modules in a later slice.
        return;
      }
      try {
        const msg = JSON.parse(ev.data) as ServerMessage;
        for (const fn of this.serverListeners) fn(msg);
      } catch {
        for (const fn of this.serverListeners) {
          fn({ type: "error", code: "bad_payload", message: ev.data.slice(0, 200) });
        }
      }
    };

    ws.onerror = () => {
      // onclose will fire next; reconnect logic lives there.
    };

    ws.onclose = (ev) => {
      this.ws = null;
      if (!this.wantOpen) {
        this.state.set({ kind: "disconnected" });
        return;
      }
      if (ev.code === 1000 || ev.code === 1008) {
        this.state.set({ kind: "closed", reason: ev.reason || `code ${ev.code}` });
        this.wantOpen = false;
        return;
      }
      this.scheduleReconnect();
    };
  }

  private scheduleReconnect(): void {
    const delay = RECONNECT_DELAYS_MS[Math.min(this.attempt, RECONNECT_DELAYS_MS.length - 1)];
    this.attempt += 1;
    this.state.set({ kind: "reconnecting", attempt: this.attempt, nextDelayMs: delay });
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      if (this.wantOpen) this.openSocket();
    }, delay);
  }

  private clearReconnect(): void {
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  // "First reply" — wall-time from the most recent client send to the next
  // inbound frame. Cleared after firing so streamed deltas don't keep
  // re-triggering. Not a true RTT (would need a ping/pong), but a useful
  // proxy for "did the brain hear me yet?" while iterating on the LLM.
  private recordLatency(): void {
    if (this.lastSendAt === null) return;
    const elapsed = performance.now() - this.lastSendAt;
    this.lastSendAt = null;
    for (const fn of this.firstReplyListeners) fn(elapsed);
  }
}
