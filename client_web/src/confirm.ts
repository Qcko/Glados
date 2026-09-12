// The confirmation dialog (DESIGN-confirm-modal.md). Renders ConfirmState and
// turns deliberate, armed, trusted input into an answer. Nothing the model
// wrote is ever parsed as markup: every string enters through textContent,
// escaped so it cannot fake a row, a button or a reordering.

import { ConfirmState, DEADLINE_MARGIN_MS, type LiveRequest } from "./confirm_state";
import type { ServerMessage, ToolConfirmRequest, ToolConfirmResolved } from "./protocol";

const CLIP_CHARS = 300;
const HIDDEN_TITLE = "(!) Confirmation needed";
const NOTIFICATION_BODY = "GLaDOS needs a confirmation";

// Code points shown as \uXXXX instead of rendered: controls, bidi overrides and
// isolates, zero-width and other invisible formatting characters.
const INVISIBLE_RANGES: ReadonlyArray<readonly [number, number]> = [
  [0x0000, 0x001f],
  [0x007f, 0x009f],
  [0x00ad, 0x00ad],
  [0x061c, 0x061c],
  [0x115f, 0x1160],
  [0x180e, 0x180e],
  [0x200b, 0x200f],
  [0x2028, 0x202e],
  [0x2060, 0x2064],
  [0x2066, 0x2069],
  [0x3164, 0x3164],
  [0xfe00, 0xfe0f],
  [0xfeff, 0xfeff],
  [0xffa0, 0xffa0],
  [0xfff9, 0xfffb],
  [0xe0000, 0xe007f],
];

export interface ConfirmHooks {
  answer(request: ToolConfirmRequest, granted: boolean): boolean;
  stopTurn(sessionId: string): void;
  note(text: string): void;
  restoreFocus(): void;
}

interface PointerPress {
  button: HTMLButtonElement | null;
  time: number;
}

interface KeyPress {
  target: EventTarget | null;
  time: number;
  repeat: boolean;
}

const NO_POINTER: PointerPress = { button: null, time: -Infinity };
const NO_KEY: KeyPress = { target: null, time: -Infinity, repeat: true };

export class ConfirmDialog {
  private readonly state = new ConfirmState();
  private readonly dialog = element("dialog", "confirm") as HTMLDialogElement;
  private readonly heading = element("h2", "confirm-heading");
  private readonly args = element("div", "confirm-args");
  private readonly bar = element("div", "confirm-bar");
  private readonly status = element("p", "confirm-status");
  private readonly stopButton = button("confirm-stop", "stop turn");
  private readonly denyButton = button("confirm-deny", "deny");
  private readonly allowButton = button("confirm-allow", "allow");
  private readonly originalTitle = document.title;
  private armTimer: ReturnType<typeof setTimeout> | null = null;
  private deadlineTimer: ReturnType<typeof setTimeout> | null = null;
  private notification: Notification | null = null;
  private lastPointer: PointerPress = NO_POINTER;
  private lastKey: KeyPress = NO_KEY;

  constructor(private readonly hooks: ConfirmHooks) {
    this.build();
    this.listen();
    this.state.setVisible(!document.hidden);
  }

  observe(msg: ServerMessage): void {
    if (msg.type === "tool_confirm_request") {
      this.open(msg);
      return;
    }
    const live = this.state.current;
    if (!live) return;
    if (this.state.resolvedBy(msg)) {
      this.finish(resolvedNote(live, msg));
    } else if (this.state.dropsOn(msg)) {
      this.finish(`confirmation for ${toolLabel(live)} dropped -- the turn moved on`);
    }
  }

  setOnline(online: boolean): void {
    this.state.setOnline(online);
    this.refresh();
  }

  private build(): void {
    const lede = element("p", "confirm-lede", "GLaDOS wants to run this tool. Nothing happens unless you allow it.");
    const argsLabel = element("div", "confirm-args-label", "arguments (from the model)");
    const countdown = element("div", "confirm-countdown");
    countdown.append(this.bar);
    const footer = element("div", "confirm-footer");
    footer.append(this.stopButton, element("span", "spacer"), this.denyButton, this.allowButton);
    this.heading.id = "confirm-heading";
    this.dialog.setAttribute("aria-labelledby", this.heading.id);
    this.dialog.tabIndex = -1;
    this.dialog.append(this.heading, lede, argsLabel, this.args, countdown, this.status, footer);
    document.body.append(this.dialog);
  }

  private listen(): void {
    this.dialog.addEventListener("keydown", (e) => this.onKeydown(e), true);
    this.dialog.addEventListener("cancel", (e) => e.preventDefault());
    this.dialog.addEventListener("close", () => this.reopenIfStillLive());
    for (const b of [this.denyButton, this.allowButton]) {
      b.addEventListener("pointerdown", (e) => {
        if (e.isTrusted) this.lastPointer = { button: b, time: performance.now() };
      });
    }
    this.allowButton.addEventListener("click", (e) => this.onPress(e, this.allowButton, true));
    this.denyButton.addEventListener("click", (e) => this.onPress(e, this.denyButton, false));
    this.stopButton.addEventListener("click", (e) => this.onStop(e));
    document.addEventListener("visibilitychange", () => this.onVisibility());
    window.addEventListener("focus", () => this.rearmAndRefresh());
    window.addEventListener("resize", () => this.rearmAndRefresh());
  }

  private open(request: ToolConfirmRequest): void {
    const { accepted, superseded } = this.state.offer(request);
    if (!accepted) return;
    if (superseded) {
      this.hooks.note(`confirmation for ${toolLabel(superseded)} dropped -- a newer request replaced it`);
    }
    this.lastPointer = NO_POINTER;
    this.lastKey = NO_KEY;
    this.fill(request);
    if (!this.dialog.open) this.dialog.showModal();
    this.dialog.focus();
    this.signalIfHidden();
    this.refresh();
  }

  private fill(request: ToolConfirmRequest): void {
    this.heading.textContent = `Allow ${visible(request.tool)}?`;
    this.args.replaceChildren();
    const entries = Object.entries(request.args_summary ?? {});
    if (entries.length === 0) this.args.append(element("div", "confirm-arg", "(no arguments)"));
    const clipped = entries.filter(([key, value]) => this.appendArgument(key, value)).length;
    this.state.setUnexpanded(clipped);
  }

  private appendArgument(key: string, value: unknown): boolean {
    const row = element("div", "confirm-arg");
    const keyCell = element("span", "confirm-key", visible(JSON.stringify(key)));
    const valueCell = element("span", "confirm-value");
    keyCell.dir = "ltr";
    valueCell.dir = "ltr";
    row.append(keyCell, valueCell);
    this.args.append(row);
    const text = visible(JSON.stringify(value) ?? String(value));
    const chars = Array.from(text);
    if (chars.length <= CLIP_CHARS) {
      valueCell.textContent = text;
      return false;
    }
    this.clipValue(valueCell, text, chars);
    return true;
  }

  private clipValue(cell: HTMLElement, text: string, chars: string[]): void {
    const more = button("confirm-more", `(${chars.length - CLIP_CHARS} more characters)`);
    cell.textContent = chars.slice(0, CLIP_CHARS).join("");
    cell.append(more);
    more.addEventListener("click", (e) => {
      if (!e.isTrusted) return;
      cell.textContent = text;
      this.state.expandOne();
      this.refresh();
    });
  }

  private onKeydown(e: KeyboardEvent): void {
    if (e.key === "Escape") {
      e.preventDefault();
      if (e.isTrusted && !e.repeat) this.answer(false, performance.now());
      return;
    }
    if (e.key === "Enter" || e.key === " ") {
      this.lastKey = { target: e.target, time: performance.now(), repeat: e.repeat || !e.isTrusted };
    }
  }

  private onPress(e: MouseEvent, pressed: HTMLButtonElement, granted: boolean): void {
    if (!e.isTrusted) return;
    const pressTime = e.detail === 0 ? this.keyPressTime(pressed) : this.pointerPressTime(pressed);
    if (pressTime !== null) this.answer(granted, pressTime);
  }

  private keyPressTime(pressed: HTMLButtonElement): number | null {
    if (this.lastKey.repeat || this.lastKey.target !== pressed) return null;
    return this.lastKey.time;
  }

  private pointerPressTime(pressed: HTMLButtonElement): number | null {
    return this.lastPointer.button === pressed ? this.lastPointer.time : null;
  }

  private answer(granted: boolean, pressTime: number): void {
    const live = this.state.current;
    if (!live) return;
    if (this.state.isExpired()) {
      this.finishExpired(live);
      return;
    }
    const permitted = granted ? this.state.canAllow() : this.state.canDeny();
    if (!permitted || !this.state.pressedAfterArming(pressTime)) return;
    if (!this.hooks.answer(live.request, granted)) {
      this.state.setOnline(false);
      this.refresh();
      return;
    }
    this.finish(`you ${granted ? "allowed" : "denied"} ${toolLabel(live)} -- sent`);
  }

  private onStop(e: MouseEvent): void {
    const live = this.state.current;
    if (e.isTrusted && live) this.hooks.stopTurn(live.request.session_id);
  }

  private onVisibility(): void {
    this.state.setVisible(!document.hidden);
    if (!document.hidden) this.clearSignal();
    this.refresh();
  }

  private rearmAndRefresh(): void {
    this.state.rearm();
    this.refresh();
  }

  private refresh(): void {
    const live = this.state.current;
    if (!live) return;
    if (this.state.isExpired()) {
      this.finishExpired(live);
      return;
    }
    this.allowButton.disabled = !this.state.canAllow();
    this.denyButton.disabled = !this.state.canDeny();
    this.status.textContent = this.statusText(live);
    this.restartCountdown(live);
    this.scheduleTimers();
  }

  private statusText(live: LiveRequest): string {
    if (!this.state.isOnline()) return "reconnecting... answer once the connection is back";
    if (live.unexpanded > 0) return "expand every clipped argument to enable allow";
    return "";
  }

  private restartCountdown(live: LiveRequest): void {
    const remaining = this.state.msUntilDeadline() ?? 0;
    const total = Math.max(1, live.request.ttl_s * 1000 - DEADLINE_MARGIN_MS);
    this.bar.style.animation = "none";
    void this.bar.offsetWidth;
    this.bar.style.transform = `scaleX(${Math.min(1, remaining / total)})`;
    this.bar.style.animation = `confirm-countdown ${remaining}ms linear forwards`;
  }

  private scheduleTimers(): void {
    this.clearTimers();
    const untilArmed = this.state.msUntilArmed();
    if (untilArmed !== null && untilArmed > 0) {
      this.armTimer = setTimeout(() => this.refresh(), untilArmed + 10);
    }
    const untilDeadline = this.state.msUntilDeadline();
    if (untilDeadline !== null) {
      this.deadlineTimer = setTimeout(() => this.refresh(), untilDeadline + 10);
    }
  }

  private finishExpired(live: LiveRequest): void {
    this.finish(`confirmation for ${toolLabel(live)} timed out -- not sent`);
  }

  private finish(note: string): void {
    if (!this.state.end()) return;
    this.clearTimers();
    if (this.dialog.open) this.dialog.close();
    this.clearSignal();
    this.hooks.note(note);
    this.hooks.restoreFocus();
  }

  private reopenIfStillLive(): void {
    if (this.state.current && !this.dialog.open) {
      this.dialog.showModal();
      this.dialog.focus();
      this.rearmAndRefresh();
    }
  }

  private signalIfHidden(): void {
    if (!document.hidden) return;
    document.title = HIDDEN_TITLE;
    if (!("Notification" in window) || Notification.permission !== "granted") return;
    try {
      this.notification?.close();
      this.notification = new Notification("GLaDOS", { body: NOTIFICATION_BODY, tag: "glados-confirm" });
      this.notification.onclick = () => window.focus();
    } catch {
      this.notification = null;
    }
  }

  private clearSignal(): void {
    if (document.title === HIDDEN_TITLE) document.title = this.originalTitle;
    this.notification?.close();
    this.notification = null;
  }

  private clearTimers(): void {
    if (this.armTimer !== null) clearTimeout(this.armTimer);
    if (this.deadlineTimer !== null) clearTimeout(this.deadlineTimer);
    this.armTimer = null;
    this.deadlineTimer = null;
  }
}

function toolLabel(live: LiveRequest): string {
  return visible(live.request.tool);
}

function resolvedNote(live: LiveRequest, msg: ToolConfirmResolved): string {
  const tool = toolLabel(live);
  if (msg.via === "timeout") return `confirmation for ${tool} timed out -- not sent`;
  const verdict = msg.granted ? "allowed" : "denied";
  const by = msg.via === "voice" ? "by voice" : "on screen";
  return `${tool} ${verdict} ${by}`;
}

function visible(text: string): string {
  let out = "";
  for (const ch of text) {
    const code = ch.codePointAt(0) ?? 0;
    out += isInvisible(code) ? escapeCodePoint(code) : ch;
  }
  return out;
}

function escapeCodePoint(code: number): string {
  const hex = code.toString(16);
  return code > 0xffff ? `\\u{${hex}}` : `\\u${hex.padStart(4, "0")}`;
}

function isInvisible(code: number): boolean {
  return INVISIBLE_RANGES.some(([low, high]) => code >= low && code <= high);
}

function element(tag: string, className: string, text?: string): HTMLElement {
  const el = document.createElement(tag);
  el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}

function button(className: string, text: string): HTMLButtonElement {
  const b = element("button", className, text) as HTMLButtonElement;
  b.type = "button";
  return b;
}
