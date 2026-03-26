import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { NATIVE_ADDON_EXPECTED_VERSION } from "../numbl-core/native/lapack-bridge.js";

describe("native fftAlongDim validation", () => {
  const repoRoot = process.cwd();
  const addonPath = join(repoRoot, "build", "Release", "numbl_addon.node");
  const maybeIt = existsSync(addonPath) ? it : it.skip;

  maybeIt("rejects inputs whose length does not match prod(shape)", () => {
    const req = createRequire(import.meta.url);
    const addon = req(addonPath) as {
      addonVersion?: () => number;
      fftAlongDim: (
        re: Float64Array,
        im: Float64Array | null,
        shape: number[],
        dim: number,
        n: number,
        inverse: boolean
      ) => unknown;
    };

    expect(typeof addon.fftAlongDim).toBe("function");
    expect(addon.addonVersion?.()).toBe(NATIVE_ADDON_EXPECTED_VERSION);
    expect(() =>
      addon.fftAlongDim(new Float64Array([1, 2, 3]), null, [2, 2], 1, 2, false)
    ).toThrow(/input length must match prod\(shape\)/);
  });
});
