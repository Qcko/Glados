// localStorage-backed connection settings.

const KEY = "glados.settings.v1";

export interface Settings {
  clientId: string;
  roomId: string;
  token: string;
}

// Dev-fixture defaults. The server reads real tokens from the OS keyring
// (scope `glados.client-tokens`, username = client id); for this default
// to work, run `python -m glados.secrets set client-tokens desk-ui` on
// the server and store the same value. Override here for any non-dev
// deploy; never reuse the literal below in production.
const DEFAULTS: Settings = {
  clientId: "desk-ui",
  roomId: "desk",
  token: "dev-token-desk",
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
