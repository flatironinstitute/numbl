import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import wasm from "vite-plugin-wasm";
import { resolve } from "path";

// Builds the standalone "site viewer": the full IDE workspace wired to load a
// baked-in project bundle (project.zip). Shipped in the npm package as
// dist-site-viewer/ and copied into a user's GitHub Pages deploy by the
// `numbl build-site` CLI command.
//
// base is relative ("./") so the bundle works under any deploy subpath
// (e.g. https://user.github.io/<repo>/) without rebuilding — the same
// approach as the browser test-runner.

export default defineConfig({
  plugins: [
    react(),
    wasm(),
    {
      name: "configure-response-headers",
      configureServer: server => {
        server.middlewares.use((_req, res, next) => {
          // Enable SharedArrayBuffer (synchronous input + cancellation) in dev.
          res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
          res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
          next();
        });
      },
    },
  ],
  worker: {
    plugins: () => [wasm()],
  },
  root: "src/site-viewer",
  base: "./",
  // Use the repo-level public/ so coi-serviceworker.js and favicon.svg are
  // copied into the build.
  publicDir: resolve(__dirname, "public"),
  build: {
    outDir: resolve(__dirname, "dist-site-viewer"),
    emptyOutDir: true,
  },
});
