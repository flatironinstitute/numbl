import { test, expect, type Page } from "@playwright/test";
import { writeFileSync } from "node:fs";

// Smoke test for the figure renderers (drawPlot.ts for 2-D, SurfView.tsx for
// 3-D). The /gallery page runs every script in examples/plots/ in its own
// worker and renders the resulting figure, so between them the examples cover
// every plot kind. The test fails on a script error, a page error, a console
// error, or a figure canvas that renders nothing.
//
// Refactoring aid: with PLOT_GALLERY_HASHES=<file>, the test also writes a
// hash of every 2-D canvas's pixels to <file>. Comparing the file from before
// and after a renderer change (same machine, same browser) shows whether the
// rendered output changed.

/** Pixel statistics for a PNG, decoded in the page. */
async function pngStats(page: Page, png: Buffer) {
  return page.evaluate(async b64 => {
    const blob = await (await fetch(`data:image/png;base64,${b64}`)).blob();
    const bmp = await createImageBitmap(blob);
    const c = document.createElement("canvas");
    c.width = bmp.width;
    c.height = bmp.height;
    const ctx = c.getContext("2d")!;
    ctx.drawImage(bmp, 0, 0);
    const d = ctx.getImageData(0, 0, c.width, c.height).data;
    let ink = 0;
    for (let i = 0; i < d.length; i += 4) {
      if (d[i] < 245 || d[i + 1] < 245 || d[i + 2] < 245) ink++;
    }
    return { ink, total: d.length / 4 };
  }, png.toString("base64"));
}

test("plot gallery renders every example without errors", async ({ page }) => {
  test.setTimeout(240_000);
  const errors: string[] = [];
  page.on("pageerror", err => errors.push(`[pageerror] ${err.message}`));
  page.on("console", msg => {
    if (msg.type() === "error") errors.push(`[console] ${msg.text()}`);
  });

  await page.setViewportSize({ width: 1280, height: 900 });
  await page.goto("/gallery");
  const items = page.getByTestId("gallery-item");
  await expect(items.first()).toBeVisible({ timeout: 30_000 });
  const count = await items.count();
  expect(count).toBeGreaterThan(30);

  // Every script finishes (done or error) ...
  await expect(items.and(page.locator('[data-status="running"]'))).toHaveCount(
    0,
    { timeout: 180_000 }
  );
  // ... and none fails.
  const failed = await items
    .and(page.locator('[data-status="error"]'))
    .evaluateAll(els => els.map(e => e.getAttribute("data-name")));
  expect(failed).toEqual([]);

  // Let the ResizeObservers and the 3-D render loops draw.
  await page.waitForTimeout(1000);

  await expect(items.filter({ hasText: "Plot render error" })).toHaveCount(0);

  // 2-D canvases are read back directly; WebGL canvases (getContext("2d")
  // returns null) are checked from a screenshot below.
  const results = await items.evaluateAll(async els => {
    const out: {
      name: string;
      canvases: { ink: number; total: number; hash: string | null }[];
    }[] = [];
    for (const el of els) {
      const canvases = [];
      for (const c of Array.from(el.querySelectorAll("canvas"))) {
        const ctx = c.getContext("2d");
        if (!ctx) {
          canvases.push({ ink: -1, total: 0, hash: null });
          continue;
        }
        const d = ctx.getImageData(0, 0, c.width, c.height).data;
        let ink = 0;
        for (let i = 0; i < d.length; i += 4) {
          const opaque = d[i + 3] > 0;
          if (opaque && (d[i] < 245 || d[i + 1] < 245 || d[i + 2] < 245)) ink++;
        }
        const digest = await crypto.subtle.digest("SHA-256", d);
        const hash = Array.from(new Uint8Array(digest))
          .map(b => b.toString(16).padStart(2, "0"))
          .join("");
        canvases.push({ ink, total: d.length / 4, hash });
      }
      out.push({ name: el.getAttribute("data-name")!, canvases });
    }
    return out;
  });

  const hashes: Record<string, string[]> = {};
  for (const [i, { name, canvases }] of results.entries()) {
    expect(canvases.length, `${name}: no canvas`).toBeGreaterThan(0);
    hashes[name] = [];
    for (const [k, c] of canvases.entries()) {
      let { ink, total } = c;
      if (c.hash === null) {
        const canvas = items.nth(i).locator("canvas").nth(k);
        await canvas.scrollIntoViewIfNeeded();
        ({ ink, total } = await pngStats(page, await canvas.screenshot()));
      } else {
        hashes[name].push(c.hash);
      }
      expect(ink / total, `${name}: canvas ${k} is blank`).toBeGreaterThan(
        0.005
      );
    }
  }

  expect(errors).toEqual([]);

  const out = process.env.PLOT_GALLERY_HASHES;
  if (out) writeFileSync(out, JSON.stringify(hashes, null, 2) + "\n");
});
