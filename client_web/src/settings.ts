// localStorage-backed connection settings.

const KEY = "glados.settings.v1";

export interface Settings {
  clientId: string;
  roomId: string;
  token: string;
}

// No embedded credentials. The bundle is served unauthenticated, so a baked-in
// client_id/token would ship working credentials to anyone who can load the page
// (see deploy/ROADMAP.md and ARCHITECTURE.md §9). The operator types their
// identity into the connect form; it persists to localStorage from there. The
// server validates the token against the OS keyring (scope `glados.client-tokens`,
// username = client id) at the WS handshake — that is the real trust boundary.
const DEFAULTS: Settings = {
  clientId: "",
  roomId: "",
  token: "",
};

export function loadSettings(): Settings {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return { ...DEFAULTS };
    const parsed = JSON.parse(raw) as Partial<Settings>;
    return {
      clientId: parsed.clientId ?? DEFAULTS.clientId,
      roomId: parsed.roomId ?? DEFAULTS.roomId,
      token: parsed.token ?? DEFAULTS.token,
    };
  } catch {
    return { ...DEFAULTS };
  }
}

export function saveSettings(s: Settings): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(s));
  } catch {
    // Quota or disabled storage — silently ignore; in-memory state still works.
  }
}
