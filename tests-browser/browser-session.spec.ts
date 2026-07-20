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
