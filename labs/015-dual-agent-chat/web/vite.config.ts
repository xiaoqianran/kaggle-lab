import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// GitHub Pages project site: https://xiaoqianran.github.io/kaggle-lab/
export default defineConfig({
  plugins: [react()],
  // relative base works for GitHub project pages AND local gateway root
  base: process.env.VITE_BASE || "./",
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy: {
      "/proxy-api": {
        target: process.env.MODEL_PROXY_URL || "https://mp-staging.kaggle.net/models",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/proxy-api/, "/openapi"),
      },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
