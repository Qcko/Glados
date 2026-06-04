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

export interface ToolConfirmResponse {
  type: "tool_confirm_response";
  request_id: string;
  granted: boolean;
}

export type ClientMessage = Hello | UserText | Interrupt | ToolConfirmResponse;

// Audio frames are sent as raw binary WebSocket messages, not JSON.
// Wire layout: big-endian uint32 seq + PCM16-LE samples at 16 kHz mono.
// Keep AUDIO_SAMPLE_RATE in sync with src/glados/core/protocols.py.
export const AUDIO_SAMPLE_RATE = 16_000;
export const AUDIO_HEADER_LEN = 4;

export interface Welcome {
  type: "welcome";
  session_id: string;
}

export interface UserTranscript {
  type: "user_transcript";
  session_id: string;
  text: string;
  source: "voice" | "text";
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
  sample_rate: number;
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

export interface TurnOutcome {
  type: "turn_outcome";
  session_id: string;
  outcome: "done" | "needs-user" | "failed";
}

export interface ServerError {
  type: "error";
  code: string;
  message: string;
}

export interface RouteNotice {
  type: "route_notice";
  session_id: string;
  target: "primary" | "specialist";
  reason: string;
  escalated: boolean;
}

export interface ToolConfirmRequest {
  type: "tool_confirm_request";
  session_id: string;
  request_id: string;
  tool: string;
  args_summary: Record<string, unknown>;
  ttl_s: number;
}

export type ServerMessage =
  | Welcome
  | UserTranscript
  | AssistantDelta
  | ToolCall
  | ToolResult
  | TtsChunk
  | Done
  | Cancelled
  | TurnOutcome
  | RouteNotice
  | ToolConfirmRequest
  | ServerError;
