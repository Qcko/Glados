import { defineConfig } from "vite";

export default defineConfig({
  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: true,
  },
  server: {
    // Loopback only. Mitigates SNYK-JS-VITE-15922213 (.map dir-traversal,
    // unpatched in Vite 5.x) by ensuring the dev server is never reachable
    // from another machine, regardless of network config.
    host: "127.0.0.1",
    strictPort: true,
    port: 5173,
    proxy: {
      "/ws/v1": {
        target: "ws://127.0.0.1:8765",
        ws: true,
      },
      "/healthz": "http://127.0.0.1:8765",
    },
  },
});
