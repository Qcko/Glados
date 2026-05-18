import "./styles.css";

import { Mic, type MicEvent } from "./audio/mic";
import { TtsPlayer } from "./audio/tts";
import { loadSettings, saveSettings, type Settings } from "./settings";
import { StateMachine, type ConnState } from "./state";
import { Transport } from "./transport";
import { Transcript } from "./transcript";

const QUICK_PROMPTS = [
  "What time is it?",
  "Roll 3d6 for me.",
  "What is 17 plus 25?",
  "Echo: hello GLaDOS",
];

const app = document.getElementById("app")!;
app.innerHTML = `
  <header>
    <h1>GLaDOS</h1>
    <span id="status" class="status">disconnected</span>
    <span id="latency" class="latency" title="time from last send to first server frame"></span>
    <div class="right">
      <input id="clientId" size="10" title="client_id" />
      <input id="roomId" size="6" title="room_id" />
      <input id="token" size="14" title="token" />
      <button id="connectBtn">connect</button>
    </div>
  </header>
  <main id="log"></main>
  <footer>
    <div class="quick" id="quick"></div>
    <form class="input-row" id="form">
      <button id="micBtn" type="button" class="mic" title="capture mic" disabled>
        <span class="dot"></span> mic
      </button>
      <button id="stopBtn" type="button" class="stop" title="interrupt the current reply" disabled>⏹</button>
      <button id="muteBtn" type="button" class="mute" title="mute speaker output">🔊</button>
      <input id="input" placeholder="say something…" autocomplete="off" disabled />
      <button id="sendBtn" type="submit" disabled>send</button>
    </form>
  </footer>
`;

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el as T;
};

const settings = loadSettings();
const clientIdInput = $<HTMLInputElement>("clientId");
const roomIdInput = $<HTMLInputElement>("roomId");
const tokenInput = $<HTMLInputElement>("token");
clientIdInput.value = settings.clientId;
roomIdInput.value = settings.roomId;
tokenInput.value = settings.token;

const statusEl = $<HTMLElement>("status");
const latencyEl = $<HTMLElement>("latency");
const connectBtn = $<HTMLButtonElement>("connectBtn");
const inputEl = $<HTMLInputElement>("input");
const sendBtn = $<HTMLButtonElement>("sendBtn");
const micBtn = $<HTMLButtonElement>("micBtn");
const stopBtn = $<HTMLButtonElement>("stopBtn");
const muteBtn = $<HTMLButtonElement>("muteBtn");
const formEl = $<HTMLFormElement>("form");
const quickEl = $<HTMLElement>("quick");

const log = $<HTMLElement>("log");
const transcript = new Transcript(log);

const state = new StateMachine();
const transport = new Transport(state);
const mic = new Mic((data) => transport.sendBinary(data));
const tts = new TtsPlayer();

mic.subscribe(renderMic);

muteBtn.addEventListener("click", () => {
  tts.setMuted(!tts.isMuted());
});
tts.subscribe((e) => {
  if (e.kind === "muted") muteBtn.textContent = e.value ? "🔇" : "🔊";
});

function renderMic(e: MicEvent): void {
  micBtn.classList.remove("running", "starting", "err");
  switch (e.kind) {
    case "starting":
      micBtn.classList.add("starting");
      micBtn.textContent = "starting…";
      break;
    case "running":
      micBtn.classList.add("running");
      micBtn.innerHTML = '<span class="dot"></span> capturing';
      break;
    case "stopped":
      micBtn.innerHTML = '<span class="dot"></span> mic';
      break;
    case "error":
      micBtn.classList.add("err");
      micBtn.textContent = "mic error";
      transcript.systemNote(`mic: ${e.message}`);
      break;
  }
}

micBtn.addEventListener("click", () => {
  if (mic.running) {
    void mic.stop();
  } else {
    void mic.start();
  }
});

let activeSessionId: string | null = null;

function setActiveSession(id: string | null): void {
  activeSessionId = id;
  stopBtn.disabled = id === null;
}

transport.onServerMessage((msg) => {
  switch (msg.type) {
    case "welcome":
      setActiveSession(msg.session_id);
      break;
    case "tts_chunk":
      tts.enqueue(msg.pcm_b64, msg.sample_rate);
      break;
    case "cancelled":
      tts.stop();
      if (activeSessionId === msg.session_id) setActiveSession(null);
      break;
    case "done":
      if (activeSessionId === msg.session_id) setActiveSession(null);
      break;
    case "error":
      setActiveSession(null);
      break;
  }
  transcript.ingest(msg);
});

stopBtn.addEventListener("click", () => {
  if (activeSessionId === null) return;
  transport.send({ type: "interrupt", session_id: activeSessionId });
  // Optimistic local stop — server's Cancelled will redundantly call stop
  // again, which is idempotent. Without this the user still hears the
  // already-buffered audio for a few hundred ms.
  tts.stop();
});
transport.onFirstReply((ms) => {
  latencyEl.textContent = `1st reply ${ms.toFixed(0)} ms`;
});

state.subscribe(renderState);

function renderState(s: ConnState): void {
  const setInputsDisabled = (locked: boolean) => {
    clientIdInput.disabled = locked;
    roomIdInput.disabled = locked;
    tokenInput.disabled = locked;
  };

  statusEl.classList.remove("live", "warn", "err");
  switch (s.kind) {
    case "disconnected":
      statusEl.textContent = "disconnected";
      setInputsDisabled(false);
      inputEl.disabled = true;
      sendBtn.disabled = true;
      micBtn.disabled = true;
      if (mic.running) void mic.stop();
      setActiveSession(null);
      tts.stop();
      connectBtn.textContent = "connect";
      connectBtn.classList.remove("disconnect");
      latencyEl.textContent = "";
      break;
    case "connecting":
      statusEl.textContent = "connecting…";
      statusEl.classList.add("warn");
      setInputsDisabled(true);
      inputEl.disabled = true;
      sendBtn.disabled = true;
      micBtn.disabled = true;
      if (mic.running) void mic.stop();
      connectBtn.textContent = "cancel";
      connectBtn.classList.add("disconnect");
      break;
    case "ready":
      statusEl.textContent = `connected as ${s.clientId} (${s.roomId})`;
      statusEl.classList.add("live");
      setInputsDisabled(true);
      inputEl.disabled = false;
      sendBtn.disabled = false;
      micBtn.disabled = false;
      connectBtn.textContent = "disconnect";
      connectBtn.classList.add("disconnect");
      inputEl.focus();
      break;
    case "reconnecting":
      statusEl.textContent = `reconnecting (try ${s.attempt}, in ${(s.nextDelayMs / 1000).toFixed(1)}s)`;
      statusEl.classList.add("warn");
      setInputsDisabled(true);
      inputEl.disabled = true;
      sendBtn.disabled = true;
      micBtn.disabled = true;
      if (mic.running) void mic.stop();
      connectBtn.textContent = "cancel";
      connectBtn.classList.add("disconnect");
      break;
    case "closed":
      statusEl.textContent = `closed: ${s.reason}`;
      statusEl.classList.add("err");
      setInputsDisabled(false);
      inputEl.disabled = true;
      sendBtn.disabled = true;
      micBtn.disabled = true;
      if (mic.running) void mic.stop();
      setActiveSession(null);
      tts.stop();
      connectBtn.textContent = "connect";
      connectBtn.classList.remove("disconnect");
      break;
  }
}

function currentSettings(): Settings {
  return {
    clientId: clientIdInput.value.trim(),
    roomId: roomIdInput.value.trim(),
    token: tokenInput.value.trim(),
  };
}

connectBtn.addEventListener("click", () => {
  const s = state.current;
  if (s.kind === "ready" || s.kind === "connecting" || s.kind === "reconnecting") {
    transport.disconnect();
    return;
  }
  const next = currentSettings();
  saveSettings(next);
  transport.connect(next);
});

formEl.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = inputEl.value.trim();
  if (!text) return;
  inputEl.value = "";
  if (transport.send({ type: "user_text", text })) {
    transcript.addUserText(text);
  } else {
    transcript.systemNote("not connected");
  }
});

for (const prompt of QUICK_PROMPTS) {
  const b = document.createElement("button");
  b.type = "button";
  b.textContent = prompt;
  b.addEventListener("click", () => {
    if (transport.send({ type: "user_text", text: prompt })) {
      transcript.addUserText(prompt);
    }
  });
  quickEl.appendChild(b);
}
