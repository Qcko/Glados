// Wire-protocol shapes — mirror src/glados/core/protocols.py exactly.

export type Role = "mic" | "speaker" | "ui";

export interface Hello {
  type: "hello";
  client_id: string;
  room_id: string;
  role: Role;
  token: string;
}

export interface UserText {
  type: "user_text";
  text: string;
}

export interface Interrupt {
  type: "interrupt";
  session_id: string;
}

export type ClientMessage = Hello | UserText | Interrupt;

// Audio frames are sent as raw binary WebSocket messages, not JSON.
// Wire layout: big-endian uint32 seq + PCM16-LE samples at 16 kHz mono.
// Keep AUDIO_SAMPLE_RATE in sync with src/glados/core/protocols.py.
export const AUDIO_SAMPLE_RATE = 16_000;
export const AUDIO_HEADER_LEN = 4;

export interface Welcome {
  type: "welcome";
  session_id: string;
}

export interface AssistantDelta {
  type: "assistant_delta";
  session_id: string;
  text: string;
}

export interface ToolCall {
  type: "tool_call";
  session_id: string;
  call_id: string;
  server: string;
  name: string;
  args: Record<string, unknown>;
}

export interface ToolResult {
  type: "tool_result";
  session_id: string;
  call_id: string;
  ok: boolean;
  content?: Record<string, unknown> | null;
  error?: string | null;
}

export interface TtsChunk {
  type: "tts_chunk";
  session_id: string;
  seq: number;
  pcm_b64: string;
}

export interface Done {
  type: "done";
  session_id: string;
}

export interface Cancelled {
  type: "cancelled";
  session_id: string;
}

export interface ServerError {
  type: "error";
  code: string;
  message: string;
}

export type ServerMessage =
  | Welcome
  | AssistantDelta
  | ToolCall
  | ToolResult
  | TtsChunk
  | Done
  | Cancelled
  | ServerError;
