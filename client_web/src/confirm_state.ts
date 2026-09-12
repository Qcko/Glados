// The confirmation gate's rules, free of the DOM (DESIGN-confirm-modal.md).
// One live request at a time, a deadline stored as a timestamp, and an arming
// time that must pass while the dialog is actually seen and online.

import type { ServerMessage, ToolConfirmRequest, ToolConfirmResolved } from "./protocol";

export const ARMING_MS = 800;
export const DEADLINE_MARGIN_MS = 1000;

export type Clock = () => number;

export interface LiveRequest {
  readonly request: ToolConfirmRequest;
  readonly deadline: number;
  armedAt: number | null;
  unexpanded: number;
}

export interface Offer {
  accepted: boolean;
  superseded: LiveRequest | null;
}

// Frames that can only arrive once the turn has moved past its confirm.
const SESSION_FRAMES: ReadonlySet<string> = new Set([
  "tool_call",
  "tool_result",
  "assistant_delta",
  "route_notice",
  "turn_outcome",
  "done",
  "cancelled",
]);

const REMEMBERED_REQUESTS = 64;

export class ConfirmState {
  private live: LiveRequest | null = null;
  private online = false;
  private visible = true;
  private readonly finished: string[] = [];

  constructor(private readonly clock: Clock = () => performance.now()) {}

  get current(): LiveRequest | null {
    return this.live;
  }

  offer(request: ToolConfirmRequest): Offer {
    if (this.isKnown(request.request_id)) return { accepted: false, superseded: null };
    const superseded = this.end();
    this.live = {
      request,
      deadline: this.clock() + request.ttl_s * 1000 - DEADLINE_MARGIN_MS,
      armedAt: null,
      unexpanded: 0,
    };
    this.rearm();
    return { accepted: true, superseded };
  }

  setUnexpanded(count: number): void {
    if (this.live) this.live.unexpanded = count;
  }

  expandOne(): void {
    if (!this.live) return;
    this.live.unexpanded = Math.max(0, this.live.unexpanded - 1);
    this.rearm();
  }

  setOnline(online: boolean): void {
    this.online = online;
    this.rearm();
  }

  setVisible(visible: boolean): void {
    this.visible = visible;
    this.rearm();
  }

  rearm(): void {
    if (!this.live) return;
    this.live.armedAt = this.online && this.visible ? this.clock() + ARMING_MS : null;
  }

  isOnline(): boolean {
    return this.online;
  }

  isExpired(): boolean {
    return this.live !== null && this.clock() >= this.live.deadline;
  }

  canDeny(): boolean {
    return this.isArmed() && !this.isExpired();
  }

  canAllow(): boolean {
    return this.canDeny() && this.live !== null && this.live.unexpanded === 0;
  }

  pressedAfterArming(pressTime: number): boolean {
    const armedAt = this.live?.armedAt;
    return armedAt !== null && armedAt !== undefined && pressTime >= armedAt;
  }

  msUntilArmed(): number | null {
    const armedAt = this.live?.armedAt;
    if (armedAt === null || armedAt === undefined) return null;
    return Math.max(0, armedAt - this.clock());
  }

  msUntilDeadline(): number | null {
    return this.live ? Math.max(0, this.live.deadline - this.clock()) : null;
  }

  // The server's own word that this request is decided, whichever arm
  // decided it. Any other request's resolution is somebody else's news.
  resolvedBy(msg: ServerMessage): msg is ToolConfirmResolved {
    return (
      msg.type === "tool_confirm_resolved" &&
      this.live !== null &&
      msg.request_id === this.live.request.request_id
    );
  }

  dropsOn(msg: ServerMessage): boolean {
    const live = this.live;
    if (!live) return false;
    // No `error` code means the turn moved on: a short mic frame sends
    // bad_audio_frame while the server is still waiting on this confirm.
    if (msg.type === "welcome") return msg.session_id !== live.request.session_id;
    return SESSION_FRAMES.has(msg.type) && sessionOf(msg) === live.request.session_id;
  }

  end(): LiveRequest | null {
    const live = this.live;
    if (live) this.remember(live.request.request_id);
    this.live = null;
    return live;
  }

  private isArmed(): boolean {
    const armedAt = this.live?.armedAt;
    return armedAt !== null && armedAt !== undefined && this.clock() >= armedAt;
  }

  private isKnown(requestId: string): boolean {
    return this.live?.request.request_id === requestId || this.finished.includes(requestId);
  }

  private remember(requestId: string): void {
    this.finished.push(requestId);
    if (this.finished.length > REMEMBERED_REQUESTS) this.finished.shift();
  }
}

function sessionOf(msg: ServerMessage): string | null {
  return "session_id" in msg ? msg.session_id : null;
}
