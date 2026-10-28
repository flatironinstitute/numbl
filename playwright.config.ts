import { defineConfig } from "@playwright/test";

// Browser-side integration tests (tests-browser/*.spec.ts):
//   - test-runner.spec.ts drives the test-runner page (src/test-runner/),
//     which executes every test script in numbl_test_scripts/ through the
//     same executeCode path used by the CLI and unit tests, but inside a
//     real Chromium tab.
//   - ide.spec.ts drives the web IDE on the /share route.
//   - browser-session.spec.ts exercises the published numbl/browser bundle
//     (dist-browser/browser.js).
//
// Prerequisites:
//   - `npm run build` has produced dist/, including dist/test-runner/
//     and dist/test-scripts/manifest.json.
//   - `npm run build:browser` has produced dist-browser/ (for
//     browser-session.spec.ts).
//   - `npm run preview` serves dist/ at http://localhost:4173 (started
//     automatically by `webServer`).
export default defineConfig({
  testDir: "./tests-browser",
  // The test-runner spec walks ~920 scripts serially in a single test, so
  // give each test plenty of headroom. Individual scripts finish in
  // milliseconds.
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
