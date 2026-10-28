import { test, expect } from "@playwright/test";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

// Exercises the published numbl/browser bundle (dist-browser/browser.js) in a
// real Chromium tab: the module is imported from a Blob URL so the test runs
// against exactly what npm consumers get. Sessions run with mip and /system
// persistence disabled to stay network-free and hermetic.

const bundlePath = join(
  dirname(fileURLToPath(import.meta.url)),
  "../dist-browser/browser.js"
);

interface SessionScenarioResult {
  bootOutput: string;
  mainCarries: { ok: boolean; output: string };
  replPersists: { ok: boolean; output: string };
  plotTypes: string[];
  errorResult: { ok: boolean; error?: string };
}

test("numbl/browser sessions execute incrementally with figures", async ({
  page,
}) => {
  page.on("pageerror", err => console.error("[pageerror]", err.message));

  await page.goto("/", { waitUntil: "load" });

  const result = (await page.evaluate(
    async bundleSource => {
      const url = URL.createObjectURL(
        new Blob([bundleSource], { type: "text/javascript" })
      );
      const { createNumblSession } = await import(/* @vite-ignore */ url);

      // Boot with a main script; its workspace must carry into execute().
      let bootOutput = "";
      const session = await createNumblSession({
        files: [{ path: "main.m", content: "z = 6 * 7;\ndisp(z)\n" }],
        mainFile: "main.m",
        mip: false,
        persistSystem: false,
        onOutput: (text: string) => {
          bootOutput += text;
        },
      });

      const mainCarries = await session.execute("z2 = z / 2");
      const replPersists = await session.execute("z3 = z2 + 1");
      const plotResult = await session.execute(
        "x = linspace(0, 2*pi, 50); plot(x, sin(x));"
      );
      const errorResult = await session.execute("no_such_function_xyz(1)");
      session.dispose();

      return {
        bootOutput,
        mainCarries: { ok: mainCarries.ok, output: mainCarries.output },
        replPersists: { ok: replPersists.ok, output: replPersists.output },
        plotTypes: plotResult.plotInstructions.map(
          (pi: { type: string }) => pi.type
        ),
        errorResult: { ok: errorResult.ok, error: errorResult.error },
      };
    },
    readFileSync(bundlePath, "utf-8")
  )) as SessionScenarioResult;

  expect(result.bootOutput).toContain("42");

  // Variables persist from the main run and across execute() calls, and
  // unsuppressed results echo REPL-style.
  expect(result.mainCarries.ok).toBe(true);
  expect(result.mainCarries.output).toMatch(/z2\s*=/);
  expect(result.mainCarries.output).toContain("21");
  expect(result.replPersists.ok).toBe(true);
  expect(result.replPersists.output).toContain("22");

  // Non-uihtml plot instructions are returned, not dropped.
  expect(result.plotTypes).toContain("plot");

  // numbl errors resolve with ok:false and a formatted message.
  expect(result.errorResult.ok).toBe(false);
  expect(result.errorResult.error).toBeTruthy();
});

interface InterruptScenarioResult {
  canInput: boolean;
  answered: { ok: boolean; output: string };
  stoppedInput: { aborted?: boolean; ms: number };
  stoppedPause: { aborted?: boolean; ms: number };
  after: { ok: boolean; output: string };
}

test("numbl/browser interrupt() stops a blocked input() or pause", async ({
  page,
}) => {
  page.on("pageerror", err => console.error("[pageerror]", err.message));
  await page.goto("/", { waitUntil: "load" });
  // coi-serviceworker reloads the page once to enable SharedArrayBuffer.
  await page.waitForFunction(() => window.crossOriginIsolated, null, {
    timeout: 30_000,
  });

  const result = (await page.evaluate(
    async bundleSource => {
      const url = URL.createObjectURL(
        new Blob([bundleSource], { type: "text/javascript" })
      );
      const { createNumblSession } = await import(/* @vite-ignore */ url);
      let prompts = 0;
      const session = await createNumblSession({
        mip: false,
        persistSystem: false,
        onInputRequest: () => {
          // Answer the first prompt; leave the second one waiting.
          if (++prompts === 1) session.provideInput("21");
        },
      });

      const answered = await session.execute("x = input('first: '); disp(2*x)");

      const timedInterrupt = async (code: string) => {
        const pending = session.execute(code);
        await new Promise(r => setTimeout(r, 300));
        const t0 = performance.now();
        session.interrupt();
        const r = await pending;
        return { aborted: r.aborted, ms: performance.now() - t0 };
      };
      const stoppedInput = await timedInterrupt("y = input('second: ');");
      const stoppedPause = await timedInterrupt("pause(60)");
      const after = await session.execute("disp(x + 1)");
      const canInput = session.canInput;
      session.dispose();
      return {
        canInput,
        answered: { ok: answered.ok, output: answered.output },
        stoppedInput,
        stoppedPause,
        after: { ok: after.ok, output: after.output },
      };
    },
    readFileSync(bundlePath, "utf-8")
  )) as InterruptScenarioResult;

  expect(result.canInput).toBe(true);
  expect(result.answered.ok).toBe(true);
  expect(result.answered.output).toContain("42");
  // Both waits end promptly, and the workspace survives.
  expect(result.stoppedInput.aborted).toBe(true);
  expect(result.stoppedInput.ms).toBeLessThan(2000);
  expect(result.stoppedPause.aborted).toBe(true);
  expect(result.stoppedPause.ms).toBeLessThan(2000);
  expect(result.after.ok).toBe(true);
  expect(result.after.output).toContain("22");
});
