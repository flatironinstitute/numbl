import { test, expect } from "@playwright/test";

interface TestResults {
  total: number;
  pass: number;
  fail: number;
  skipped: number;
  failed: { path: string; output: string }[];
  durationMs: number;
}

declare global {
  interface Window {
    __numblTestResults?: TestResults;
  }
}

test("every .m script in numbl_test_scripts/ passes in Chromium", async ({
  page,
}) => {
  // Surface in-page errors in Playwright output (helps when CI fails).
  page.on("pageerror", err => console.error("[pageerror]", err.message));
  page.on("console", msg => {
    if (msg.type() === "error") console.error("[console.error]", msg.text());
  });

  await page.goto("/test-runner/", { waitUntil: "load" });

  // Wait until the runner publishes its results. Internal per-test timeouts
  // are bounded by playwright.config.ts `timeout`.
  await page.waitForFunction(() => window.__numblTestResults !== undefined, {
    timeout: 550_000,
    polling: 1000,
  });

  const results = (await page.evaluate(
    () => window.__numblTestResults
  )) as TestResults;

  console.log(
    `browser-runner: total=${results.total} pass=${results.pass} ` +
      `fail=${results.fail} skipped=${results.skipped} ` +
      `duration=${(results.durationMs / 1000).toFixed(1)}s`
  );

  if (results.failed.length > 0) {
    for (const f of results.failed) {
      console.error(`\n── FAIL: ${f.path} ──\n${f.output}`);
    }
  }

  expect(results.failed).toEqual([]);
  expect(results.fail).toBe(0);
  expect(results.pass).toBeGreaterThan(0);
});
