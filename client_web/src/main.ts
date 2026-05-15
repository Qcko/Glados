import "./styles.css";

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
const formEl = $<HTMLFormElement>("form");
const quickEl = $<HTMLElement>("quick");

const log = $<HTMLElement>("log");
const transcript = new Transcript(log);

const state = new StateMachine();
const transport = new Transport(state);

transport.onServerMessage((msg) => transcript.ingest(msg));
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
      connectBtn.textContent = "cancel";
      connectBtn.classList.add("disconnect");
      break;
    case "ready":
      statusEl.textContent = `connected as ${s.clientId} (${s.roomId})`;
      statusEl.classList.add("live");
      setInputsDisabled(true);
      inputEl.disabled = false;
      sendBtn.disabled = false;
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
      connectBtn.textContent = "cancel";
      connectBtn.classList.add("disconnect");
      break;
    case "closed":
      statusEl.textContent = `closed: ${s.reason}`;
      statusEl.classList.add("err");
      setInputsDisabled(false);
      inputEl.disabled = true;
      sendBtn.disabled = true;
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
