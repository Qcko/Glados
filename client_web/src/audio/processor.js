// AudioWorklet running in AudioWorkletGlobalScope (separate JS realm,
// audio thread). No DOM, no Window. Plain JS so Vite emits it as a
// standalone asset that `audioWorklet.addModule(url)` can fetch --
// .ts files would be inlined into the main bundle.
//
// Job: take 128-sample float32 chunks at the input AudioContext's sample
// rate (typically 48 kHz), downsample to 16 kHz, convert to int16-LE,
// buffer to ~50 ms (800 samples) and post each batch back to the main
// thread as a transferable ArrayBuffer.

const TARGET_SAMPLE_RATE = 16000;
const BATCH_SAMPLES = 800; // 50 ms at 16 kHz

class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.downsampleRatio = sampleRate / TARGET_SAMPLE_RATE;
    this.batch = new Int16Array(BATCH_SAMPLES);
    this.batchFill = 0;
    this.resampleCursor = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const channel = input[0];
    if (!channel) return true;

    while (this.resampleCursor < channel.length) {
      const idx = this.resampleCursor | 0;
      const sample = channel[idx] ?? 0;
      const clamped = Math.max(-1, Math.min(1, sample));
      this.batch[this.batchFill++] = (clamped * 0x7fff) | 0;
      this.resampleCursor += this.downsampleRatio;

      if (this.batchFill >= BATCH_SAMPLES) {
        this.flush();
      }
    }
    this.resampleCursor -= channel.length;
    return true;
  }

  flush() {
    const copy = new Int16Array(this.batchFill);
    copy.set(this.batch.subarray(0, this.batchFill));
    this.port.postMessage(copy.buffer, [copy.buffer]);
    this.batchFill = 0;
  }
}

registerProcessor("capture", CaptureProcessor);
