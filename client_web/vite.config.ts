import { defineConfig } from "vite";

export default defineConfig({
  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: true,
  },
  server: {
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
