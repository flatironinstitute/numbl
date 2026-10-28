import { test, expect, type Page } from "@playwright/test";
import { zlibSync } from "fflate";

// Drives the IDE on the /share route (files come from the URL hash, so no
// IndexedDB project setup is needed): script output, errors, figures,
// input(), uihtml events in both directions, stopping a runaway script, and
// the REPL.

function shareUrl(files: Record<string, string>, active: string): string {
  const data = {
    files: Object.entries(files).map(([name, content]) => ({ name, content })),
    activeFileName: active,
  };
  const compressed = zlibSync(new TextEncoder().encode(JSON.stringify(data)));
  const b64 = Buffer.from(compressed)
    .toString("base64")
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/, "");
  return `/share#${b64}`;
}

async function openIde(page: Page, files: Record<string, string>) {
  page.on("pageerror", err => console.error("[pageerror]", err.message));
  await page.goto(shareUrl(files, Object.keys(files)[0]));
  // coi-serviceworker reloads the page once to enable SharedArrayBuffer,
  // which input() and cooperative cancellation need.
  await page.waitForFunction(() => window.crossOriginIsolated, null, {
    timeout: 30_000,
  });
  await expect(runButton(page)).toBeVisible({ timeout: 30_000 });
}

const runButton = (page: Page) =>
  page.getByRole("button", { name: "Run", exact: true });
const stopButton = (page: Page) =>
  page.getByRole("button", { name: "Stop", exact: true });
const output = (page: Page) => page.locator("pre").first();

test.describe("IDE", () => {
  test.setTimeout(120_000);

  test("runs a script, shows output, figures and errors", async ({ page }) => {
    await openIde(page, {
      "main.m":
        "disp('hello from main')\n" +
        "y = helper(20);\n" +
        "fprintf('y=%d\\n', y);\n" +
        "plot(1:10, (1:10).^2);\n",
      "helper.m": "function r = helper(x)\n  r = x + 1;\nend\n",
      "bad.m": "disp('before error')\nerror('boom here');\n",
    });
    await runButton(page).click();
    await expect(output(page)).toContainText("hello from main");
    await expect(output(page)).toContainText("y=21");
    await expect(page.getByRole("tab", { name: "Figure 1" })).toBeVisible();
    await expect(runButton(page)).toBeVisible();

    await page.getByText("bad.m", { exact: true }).click();
    await runButton(page).click();
    await expect(output(page)).toContainText("before error");
    await expect(output(page)).toContainText("boom here");
    await expect(runButton(page)).toBeVisible();
  });

  test("answers input() through a prompt", async ({ page }) => {
    await openIde(page, {
      "ask.m": "x = input('Number please: ');\nfprintf('twice=%d\\n', 2*x);\n",
    });
    page.once("dialog", dialog => {
      expect(dialog.message()).toContain("Number please");
      void dialog.accept("21");
    });
    await runButton(page).click();
    await expect(output(page)).toContainText("twice=42");
  });

  test("relays uihtml events both ways", async ({ page }) => {
    const html =
      "<p id=r>waiting</p><script>function setup(c){" +
      "c.addEventListener(''Result'',function(e){" +
      "document.getElementById(''r'').textContent=''result ''+e.Data;});" +
      "c.sendEventToMATLAB(''Square'',7);}</script>";
    await openIde(page, {
      "widget.m":
        "fig = figure;\n" +
        `uihtml(fig, 'HTMLSource', '${html}', 'HTMLEventReceivedFcn', @onEvent);\n` +
        "function onEvent(src, event)\n" +
        "  x = event.HTMLEventData;\n" +
        "  sendEventToHTMLSource(src, 'Result', x^2);\n" +
        "end\n",
    });
    await runButton(page).click();
    const frame = page.frameLocator('iframe[title^="uihtml-"]');
    await expect(frame.locator("#r")).toHaveText("result 49");
  });

  test("stops a runaway script and runs again", async ({ page }) => {
    await openIde(page, {
      "loop.m": "disp('looping')\nk = 0;\nwhile true\n  k = k + 1;\nend\n",
      "hello.m": "disp('after stop')\n",
    });
    await runButton(page).click();
    // Output from a JIT-compiled script only arrives when it finishes, so
    // just give the loop time to start.
    await expect(stopButton(page)).toBeVisible();
    await page.waitForTimeout(500);
    await stopButton(page).click();
    await expect(output(page)).toContainText("--- Execution stopped ---");
    await page.getByText("hello.m", { exact: true }).click();
    await runButton(page).click();
    await expect(output(page)).toContainText("after stop");
  });

  test("cancels cooperatively in persistent mode", async ({ page }) => {
    await openIde(page, {
      "loop.m": "disp('looping')\nk = 0;\nwhile true\n  k = k + 1;\nend\n",
    });
    await page.getByText("1x", { exact: true }).click();
    await expect(page.getByText("persist", { exact: true })).toBeVisible();
    // Only the interpreter polls the cancel flag; JIT-compiled loops do not.
    await page.getByText("jit", { exact: true }).click();
    await expect(page.getByText("no jit", { exact: true })).toBeVisible();
    await runButton(page).click();
    await expect(stopButton(page)).toBeVisible();
    await page.waitForTimeout(500);
    await stopButton(page).click();
    await expect(output(page)).toContainText("Cancelling execution");
    await expect(runButton(page)).toBeVisible();
  });

  test("REPL keeps variables between commands", async ({ page }) => {
    await openIde(page, { "main.m": "disp(1)\n" });
    await page.getByRole("tab", { name: "REPL" }).click();
    const term = page.locator(".xterm");
    await expect(term).toBeVisible();
    await term.click();
    const rows = page.locator(".xterm-rows");
    await page.keyboard.type("a = 6*7;\n");
    await expect(rows).toContainText(/a = 6\*7;\s*>>/);
    await page.keyboard.type("disp(a + 1)\n");
    await expect(rows).toContainText("43");
  });
});
