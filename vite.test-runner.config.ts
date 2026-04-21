import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import wasm from "vite-plugin-wasm";
import { readFileSync } from "fs";
import { join, resolve } from "path";
import { buildManifest } from "./scripts/test-scripts-manifest.js";

// Builds the browser test-runner as a separate entry. Kept out of the main
// app bundle so production pages don't pay for it, but deployed alongside
// the main site (under /test-runner/) so users can visit and watch tests
// execute.

const PROJECT_ROOT = resolve(__dirname);
const TEST_SCRIPTS_DIR = join(PROJECT_ROOT, "numbl_test_scripts");

// Dev-only: serve /test-scripts/* directly from numbl_test_scripts/ so the
// runner works under `npm run dev:test-runner` without requiring a prior
// copy step. Production builds use dist/test-scripts/ populated by
// scripts/copy-test-scripts.ts.
function devTestScriptsPlugin(): Plugin {
  return {
    name: "numbl-dev-test-scripts",
    apply: "serve",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = req.url ?? "";
        if (!url.startsWith("/test-scripts/")) return next();
        const rel = decodeURIComponent(
          url.slice("/test-scripts/".length).split("?")[0]
        );

        if (rel === "manifest.json" || rel === "") {
          res.setHeader("Content-Type", "application/json");
          res.setHeader("Cache-Control", "no-store");
          res.end(JSON.stringify(buildManifest(TEST_SCRIPTS_DIR)));
          return;
        }

        const filePath = join(TEST_SCRIPTS_DIR, rel);
        if (!filePath.startsWith(TEST_SCRIPTS_DIR + "/")) {
          res.statusCode = 403;
          res.end("forbidden");
          return;
        }
        try {
          res.setHeader("Content-Type", "text/plain; charset=utf-8");
          res.setHeader("Cache-Control", "no-store");
          res.end(readFileSync(filePath));
        } catch {
          res.statusCode = 404;
          res.end("not found");
        }
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), wasm(), devTestScriptsPlugin()],
  root: "src/test-runner",
  // Relative so the entry works regardless of the deploy subpath.
  base: "./",
  define: {
    "import.meta.env.NUMBL_USE_FLOAT32": JSON.stringify(
      process.env.NUMBL_USE_FLOAT32
    ),
  },
  build: {
    outDir: "../../dist/test-runner",
    emptyOutDir: true,
  },
});
