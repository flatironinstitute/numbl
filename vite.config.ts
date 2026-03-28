import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import wasm from "vite-plugin-wasm";

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    wasm(),
    {
      name: "configure-response-headers",
      configureServer: server => {
        server.middlewares.use((_req, res, next) => {
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
  base: "/",
  define: {
    "import.meta.env.NUMBL_USE_FLOAT32": JSON.stringify(
      process.env.NUMBL_USE_FLOAT32
    ),
  },
  test: {
    exclude: ["**/node_modules/**", "**/dist/**"],
    coverage: {
      exclude: [
        "**/node_modules/**",
        "**/dist/**",
        "**/ts-lapack/**",
        "src/cli*.ts",
        "src/components/**",
        "src/db/**",
        "src/hooks/**",
      ],
    },
  },
});
