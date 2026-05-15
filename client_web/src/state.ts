// Connection finite-state machine.

export type ConnState =
  | { kind: "disconnected" }
  | { kind: "connecting"; attempt: number }
  | { kind: "ready"; clientId: string; roomId: string }
  | { kind: "reconnecting"; attempt: number; nextDelayMs: number }
  | { kind: "closed"; reason: string };

export type Listener = (s: ConnState) => void;

export class StateMachine {
  private state: ConnState = { kind: "disconnected" };
  private listeners = new Set<Listener>();

  get current(): ConnState {
    return this.state;
  }

  subscribe(fn: Listener): () => void {
    this.listeners.add(fn);
    fn(this.state);
    return () => this.listeners.delete(fn);
  }

  set(next: ConnState): void {
    this.state = next;
    for (const fn of this.listeners) fn(next);
  }
}
