// Main-thread mic capture. Requests permission, wires the AudioWorklet,
// and feeds 16 kHz mono int16 PCM batches to a binary-send callback.
//
// Wire frame layout (must match server-side AUDIO_HEADER_LEN):
//   bytes 0..4  : big-endian uint32 sequence number
//   bytes 4..N  : PCM16-LE samples

// Worklets need a separately-fetchable URL — they can't be inlined into
// the main bundle. Using `?url` on a plain `.js` file makes Vite copy it
// to `dist/assets/` and return that URL. (Vite would otherwise treat a
// JS file imported normally as a module and bundle it away.)
import processorUrl from "./processor.js?url";

const HEADER_BYTES = 4;

export type BinarySender = (data: ArrayBuffer) => boolean;
export type MicEvent =
  | { kind: "starting" }
  | { kind: "running" }
  | { kind: "stopped" }
  | { kind: "error"; message: string };
export type MicListener = (e: MicEvent) => void;

export class Mic {
  private ctx: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private node: AudioWorkletNode | null = null;
  private seq = 0;
  private starting = false;
  private listeners = new Set<MicListener>();

  constructor(private readonly send: BinarySender) {}

  get running(): boolean {
    return this.node !== null;
  }

  subscribe(fn: MicListener): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  async start(): Promise<void> {
    // `running` flips true only after `addModule` resolves; the `starting`
    // flag closes the gap so two rapid clicks don't race into two graphs.
    if (this.running || this.starting) return;
    this.starting = true;
    this.emit({ kind: "starting" });
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      });
      const ctx = new AudioContext();
      await ctx.audioWorklet.addModule(processorUrl);
      const source = ctx.createMediaStreamSource(stream);
      const node = new AudioWorkletNode(ctx, "capture");
      node.port.onmessage = (ev: MessageEvent<ArrayBuffer>) => {
        this.sendFrame(ev.data);
      };
      source.connect(node);
      // AudioWorkletNode must terminate somewhere or the graph is GC'd.
      // ctx.destination is fine for v0 — the worklet outputs nothing so
      // the user hears their own mic at zero volume. Future: speaker
      // playback will replace this terminal.
      node.connect(ctx.destination);
      this.ctx = ctx;
      this.stream = stream;
      this.node = node;
      this.emit({ kind: "running" });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      this.emit({ kind: "error", message });
      await this.stop();
    } finally {
      this.starting = false;
    }
  }

  async stop(): Promise<void> {
    if (this.node) {
      try { this.node.disconnect(); } catch { /* ignore */ }
      this.node.port.onmessage = null;
      this.node = null;
    }
    if (this.stream) {
      for (const track of this.stream.getTracks()) track.stop();
      this.stream = null;
    }
    if (this.ctx) {
      try { await this.ctx.close(); } catch { /* ignore */ }
      this.ctx = null;
    }
    this.seq = 0;
    this.emit({ kind: "stopped" });
  }

  private sendFrame(pcm: ArrayBuffer): void {
    const frame = new Uint8Array(HEADER_BYTES + pcm.byteLength);
    new DataView(frame.buffer).setUint32(0, this.seq, false);
    frame.set(new Uint8Array(pcm), HEADER_BYTES);
    this.seq = (this.seq + 1) >>> 0;
    this.send(frame.buffer);
  }

  private emit(e: MicEvent): void {
    for (const fn of this.listeners) fn(e);
  }
}
