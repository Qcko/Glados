// Gapless playback of int16 PCM chunks streamed from the server.
//
// Each tts_chunk message carries one piper-emitted segment (typically a
// sentence). We schedule each chunk as an AudioBufferSourceNode at a
// running `nextStartTime`, so chunks line up sample-accurate even when
// JS scheduling is jittery.
//
// One AudioContext per session. We pin its sampleRate to the first
// chunk's rate (cori-high = 22050 Hz). Browsers accept arbitrary rates;
// they resample internally.

export type TtsListener = (e: TtsEvent) => void;
export type TtsEvent =
  | { kind: "playing" }
  | { kind: "idle" };

export class TtsPlayer {
  private ctx: AudioContext | null = null;
  private nextStartTime = 0;
  private inFlight = 0;
  // Set by silenceCurrentReply(); cleared by allowPlayback() at the
  // start of the next turn. While set, incoming chunks are dropped so
  // a click late in a reply truly silences the rest of it even though
  // server-side synthesis keeps streaming.
  private suppressed = false;
  private listeners = new Set<TtsListener>();

  subscribe(fn: TtsListener): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  silenceCurrentReply(): void {
    this.suppressed = true;
    this.stop();
  }

  allowPlayback(): void {
    this.suppressed = false;
  }

  enqueue(pcmB64: string, sampleRate: number): void {
    if (this.suppressed) return;
    const pcm = decodePcm16(pcmB64);
    if (pcm.length === 0) return;

    const ctx = this.ensureContext(sampleRate);
    const buffer = ctx.createBuffer(1, pcm.length, sampleRate);
    const channel = buffer.getChannelData(0);
    for (let i = 0; i < pcm.length; i++) channel[i] = pcm[i] / 32768;

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);

    const now = ctx.currentTime;
    const startAt = Math.max(now, this.nextStartTime);
    source.start(startAt);
    this.nextStartTime = startAt + buffer.duration;

    this.inFlight += 1;
    if (this.inFlight === 1) this.emit({ kind: "playing" });
    source.onended = () => {
      this.inFlight -= 1;
      if (this.inFlight === 0) this.emit({ kind: "idle" });
    };
  }

  // Hard-cut active playback. Used on disconnect, silence, or interrupt.
  stop(): void {
    if (!this.ctx) return;
    try { this.ctx.close(); } catch { /* ignore */ }
    this.ctx = null;
    this.nextStartTime = 0;
    if (this.inFlight > 0) {
      this.inFlight = 0;
      this.emit({ kind: "idle" });
    }
  }

  private ensureContext(sampleRate: number): AudioContext {
    if (this.ctx && this.ctx.sampleRate === sampleRate) return this.ctx;
    // Sample-rate switch (rare: would happen if voice config changes
    // mid-session). Tear down and rebuild rather than try to resample.
    if (this.ctx) this.stop();
    this.ctx = new AudioContext({ sampleRate });
    this.nextStartTime = this.ctx.currentTime;
    return this.ctx;
  }

  private emit(e: TtsEvent): void {
    for (const fn of this.listeners) fn(e);
  }
}

function decodePcm16(b64: string): Int16Array {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  // PCM16-LE -- Int16Array on a little-endian view of the bytes.
  return new Int16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 1);
}
