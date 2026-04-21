import { defineConfig } from "@playwright/test";

// Browser-side integration tests. The only spec today drives the
// test-runner page (src/test-runner/) which executes every .m script in
// numbl_test_scripts/ through the same executeCode path used by the CLI
// and unit tests, but inside a real Chromium tab.
//
// Prerequisites (handled automatically by `webServer`):
//   - `npm run build` has produced dist/, including dist/test-runner/
//     and dist/test-scripts/manifest.json.
//   - `npm run preview` serves dist/ at http://localhost:4173.
export default defineConfig({
  testDir: "./tests-browser",
  // The runner walks ~680 scripts serially — give the single spec plenty
  // of headroom. Individual scripts finish in milliseconds.
  timeout: 600_000,
  fullyParallel: false,
  workers: 1,
  reporter: process.env.CI ? "github" : "list",
  use: {
    baseURL: "http://localhost:4173",
    trace: "on-first-retry",
  },
  webServer: {
    command: "npm run preview -- --port 4173 --strictPort",
    port: 4173,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
