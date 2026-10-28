import { describe, it, expect, afterEach } from "vitest";
import {
  getLapackBridge,
  providerFor,
  providerNames,
  registerProvider,
  unregisterProvider,
} from "../numbl-core/native/lapack-bridge.js";
import { getTsLapackBridge } from "../numbl-core/native/ts-lapack-bridge.js";
import {
  MATMUL_INLINE_MAX_WORK,
  matmulIsInline,
} from "../numbl-core/native/inline-matmul.js";

const filled = (len: number, v: number) => new Float64Array(len).fill(v);

afterEach(() => {
  unregisterProvider("high");
  unregisterProvider("low");
});

describe("LAPACK provider stack", () => {
  it("asks providers by priority and passes declined calls down", () => {
    registerProvider({
      name: "low",
      priority: 10,
      entries: { matmul: (_a, m, _k, _b, n) => filled(m * n, 1) },
    });
    registerProvider({
      name: "high",
      priority: 20,
      entries: { matmul: (_a, m, _k, _b, n) => filled(m * n, 2) },
      accepts: { matmul: (_a, m) => m >= 50 },
    });
    expect(providerNames()).toEqual(["inline", "high", "low"]);
    const mm = getLapackBridge().matmul;
    const big = (m: number) =>
      mm(filled(m * 40, 0), m, 40, filled(40 * 40, 0), 40);
    expect(big(60)[0]).toBe(2); // high accepts
    expect(big(45)[0]).toBe(1); // high declines, low takes it
    // Small products stay on the inline provider above both.
    const r = mm(new Float64Array([1, 2, 3, 4]), 2, 2, filled(4, 1), 2);
    expect(Array.from(r)).toEqual([4, 6, 4, 6]);
  });

  it("replaces a provider registered under the same name", () => {
    registerProvider({
      name: "low",
      priority: 10,
      entries: { inv: () => filled(1, 1) },
    });
    registerProvider({
      name: "low",
      priority: 10,
      entries: { inv: () => filled(1, 5) },
    });
    expect(getLapackBridge().inv(filled(1, 2), 1)[0]).toBe(5);
    unregisterProvider("low");
    expect(getLapackBridge().inv(filled(1, 2), 1)[0]).toBe(0.5);
  });

  it("an entry without a predicate is the provider's own function", () => {
    const inv = () => filled(1, 1);
    registerProvider({ name: "low", priority: 10, entries: { inv } });
    expect(getLapackBridge().inv).toBe(inv);
  });

  it("providerFor is undefined for a declined native-only entry", () => {
    expect(
      providerFor("fft1dComplex", [filled(4, 0), filled(4, 0), 4, false])
    ).toBeUndefined();
    const fft = () => ({ re: filled(1, 0), im: filled(1, 0) });
    registerProvider({
      name: "low",
      priority: 10,
      entries: { fft1dComplex: fft },
      accepts: { fft1dComplex: (_re, _im, n) => n > 128 },
    });
    expect(
      providerFor("fft1dComplex", [filled(4, 0), filled(4, 0), 4, false])
    ).toBeUndefined();
    const n = 256;
    expect(
      providerFor("fft1dComplex", [filled(n, 0), filled(n, 0), n, false])
    ).toBe(fft);
  });

  it("the inline matmul computes the same bits as ts-lapack", () => {
    const ts = getTsLapackBridge();
    const rnd = (len: number) =>
      Float64Array.from({ length: len }, () => Math.random() - 0.5);
    for (const [m, k, n] of [
      [4, 4, 4],
      [10, 10, 10],
      [1, 1000, 1],
      [7, 3, 9],
    ]) {
      expect(matmulIsInline(m, k, n)).toBe(m * k * n <= MATMUL_INLINE_MAX_WORK);
      const [a, b, ai, bi] = [rnd(m * k), rnd(k * n), rnd(m * k), rnd(k * n)];
      expect(getLapackBridge().matmul(a, m, k, b, n)).toEqual(
        ts.matmul(a, m, k, b, n)
      );
      expect(getLapackBridge().matmulComplex(a, ai, m, k, b, bi, n)).toEqual(
        ts.matmulComplex(a, ai, m, k, b, bi, n)
      );
    }
  });
});
