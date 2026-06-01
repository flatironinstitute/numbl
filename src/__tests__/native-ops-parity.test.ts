/**
 * Drift detection: ensure the pure-TS tensor-op twins (src/numbl-core/ops/*.ts,
 * used in the browser / when no native addon is built) compute the SAME results
 * as the native C addon (native/ops/*.c, used by --opt 0 array kernels once
 * `npm run build:addon` has run).
 *
 * These two implementations are hand-synced and have drifted before (e.g.
 * `sign(NaN)`: the TS twin returned 0 while MATLAB/C return NaN). vitest never
 * builds the addon, so the TS twins are otherwise the LEAST-guarded numeric
 * implementation in the tree — nothing compares them against C. This test
 * closes that gap whenever the addon is present.
 *
 * What is asserted bit-exact vs. tolerance:
 *   - EXACT (Object.is): structural/algebraic ops that IEEE-754 pins to a
 *     single bit pattern on both sides — abs/floor/ceil/round/trunc/sign/sqrt,
 *     all comparisons, real add/sub/mul/div, and max/min/any/all. These are
 *     where
 *     logic drift (NaN handling, sign of zero, branch order) hides.
 *   - TOLERANCE (relative): transcendentals (exp/log/sin/...) where libm (C)
 *     and V8 (JS) legitimately differ at ULP scale (the same inherent
 *     divergence class as jit_parity's C02/A35), plus order-sensitive
 *     reductions (sum/prod/mean) which a `--fast-math` addon build may
 *     reorder. The tolerance is loose enough for ULP noise but tight enough
 *     to catch a wrong formula.
 *
 * Skipped automatically when the native addon isn't built.
 */

import { describe, it, expect } from "vitest";
import { createRequire } from "node:module";
import {
  OpUnary,
  OpRealBin,
  OpCmp,
  OpReduce,
} from "../numbl-core/ops/opCodes.js";
import { tsRealUnaryElemwise } from "../numbl-core/ops/realUnaryElemwise.js";
import {
  tsRealBinaryElemwise,
  tsRealScalarBinaryElemwise,
} from "../numbl-core/ops/realBinaryElemwise.js";
import {
  tsRealComparison,
  tsRealScalarComparison,
} from "../numbl-core/ops/comparison.js";
import { tsRealFlatReduce } from "../numbl-core/ops/reduce.js";

/** The native addon, or null if it hasn't been built. Same load path as
 *  op-codes-sync.test.ts. */
function loadAddon(): NativeOps | null {
  try {
    const req = createRequire(import.meta.url);
    return req("../../build/Release/numbl_addon.node") as NativeOps;
  } catch {
    return null;
  }
}

/** The subset of the native addon surface this test exercises. */
interface NativeOps {
  tensorOpRealUnary?: (
    op: number,
    n: number,
    a: Float64Array,
    out: Float64Array
  ) => void;
  tensorOpRealBinary?: (
    op: number,
    n: number,
    a: Float64Array,
    b: Float64Array,
    out: Float64Array
  ) => void;
  tensorOpRealScalarBinary?: (
    op: number,
    n: number,
    scalar: number,
    arr: Float64Array,
    scalarOnLeft: boolean,
    out: Float64Array
  ) => void;
  tensorOpRealComparison?: (
    op: number,
    n: number,
    a: Float64Array,
    b: Float64Array,
    out: Float64Array
  ) => void;
  tensorOpRealScalarComparison?: (
    op: number,
    n: number,
    scalar: number,
    arr: Float64Array,
    scalarOnLeft: boolean,
    out: Float64Array
  ) => void;
  tensorOpRealFlatReduce?: (
    op: number,
    n: number,
    a: Float64Array,
    out: Float64Array
  ) => void;
}

// Corner-case inputs most likely to expose logic drift: signed zeros,
// halves (round/trunc), out-of-domain (log/sqrt/asin), and non-finites.
const SAMPLES = [
  0,
  -0,
  1,
  -1,
  0.5,
  -0.5,
  2.5,
  -2.5,
  3.7,
  -3.7,
  42,
  -42,
  0.1,
  100,
  1e-12,
  1e12,
  Math.PI,
  -Math.PI,
  NaN,
  Infinity,
  -Infinity,
];
// A second operand vector for binary ops (same length, different values
// including a zero to exercise division by zero).
const SAMPLES_B = [
  2,
  -3,
  0,
  4,
  -0.25,
  7,
  -2.5,
  1.5,
  -1,
  0.5,
  6,
  -8,
  10,
  -0.1,
  1e6,
  -1e-6,
  2,
  -2,
  3,
  NaN,
  -1,
];

const REL_TOL = 1e-9;

/** Exact match: Object.is pins NaN===NaN and distinguishes +0/-0, so it
 *  catches sign-of-zero and NaN-handling drift. */
function bitEqual(a: number, b: number): boolean {
  return Object.is(a, b);
}

/** Approximate match for ops where libm-vs-V8 ULP differences are inherent.
 *  Non-finites must still agree exactly (same NaN-ness / same Inf sign). */
function approxEqual(a: number, b: number): boolean {
  if (Object.is(a, b)) return true;
  if (Number.isNaN(a) || Number.isNaN(b)) {
    return Number.isNaN(a) && Number.isNaN(b);
  }
  if (!Number.isFinite(a) || !Number.isFinite(b)) return a === b;
  return Math.abs(a - b) <= REL_TOL * Math.max(1, Math.abs(a), Math.abs(b));
}

function assertVecEqual(
  label: string,
  ts: Float64Array,
  c: Float64Array,
  cmp: (a: number, b: number) => boolean
): void {
  for (let i = 0; i < ts.length; i++) {
    if (!cmp(ts[i], c[i])) {
      throw new Error(
        `${label}: mismatch at index ${i}: ts=${ts[i]} c=${c[i]}`
      );
    }
  }
}

const addon = loadAddon();

describe("native ops parity (TS twin ↔ C addon)", () => {
  const have = (fn: keyof NativeOps): boolean => !!addon && !!addon[fn];

  // Which real-unary ops must match bit-exactly (vs tolerance).
  const UNARY_EXACT = new Set<number>([
    OpUnary.ABS,
    OpUnary.FLOOR,
    OpUnary.CEIL,
    OpUnary.ROUND,
    OpUnary.TRUNC,
    OpUnary.SIGN,
    OpUnary.SQRT,
  ]);

  it.skipIf(!have("tensorOpRealUnary"))(
    "realUnaryElemwise matches C for every op",
    () => {
      const n = SAMPLES.length;
      const input = Float64Array.from(SAMPLES);
      for (const [name, op] of Object.entries(OpUnary)) {
        const ts = new Float64Array(n);
        const c = new Float64Array(n);
        tsRealUnaryElemwise(op, n, input, ts);
        addon!.tensorOpRealUnary!(op, n, input, c);
        assertVecEqual(
          `unary ${name}`,
          ts,
          c,
          UNARY_EXACT.has(op) ? bitEqual : approxEqual
        );
      }
    }
  );

  it.skipIf(!have("tensorOpRealBinary"))(
    "realBinaryElemwise matches C exactly",
    () => {
      const n = SAMPLES.length;
      const a = Float64Array.from(SAMPLES);
      const b = Float64Array.from(SAMPLES_B);
      for (const [name, op] of Object.entries(OpRealBin)) {
        const ts = new Float64Array(n);
        const c = new Float64Array(n);
        tsRealBinaryElemwise(op, n, a, b, ts);
        addon!.tensorOpRealBinary!(op, n, a, b, c);
        assertVecEqual(`binary ${name}`, ts, c, bitEqual);
      }
    }
  );

  it.skipIf(!have("tensorOpRealScalarBinary"))(
    "realScalarBinaryElemwise matches C exactly (both sides)",
    () => {
      const n = SAMPLES.length;
      const arr = Float64Array.from(SAMPLES);
      const scalar = -2.5;
      for (const [name, op] of Object.entries(OpRealBin)) {
        for (const left of [true, false]) {
          const ts = new Float64Array(n);
          const c = new Float64Array(n);
          tsRealScalarBinaryElemwise(op, n, scalar, arr, left, ts);
          addon!.tensorOpRealScalarBinary!(op, n, scalar, arr, left, c);
          assertVecEqual(`scalar-binary ${name} left=${left}`, ts, c, bitEqual);
        }
      }
    }
  );

  it.skipIf(!have("tensorOpRealComparison"))(
    "realComparison matches C exactly",
    () => {
      const n = SAMPLES.length;
      const a = Float64Array.from(SAMPLES);
      const b = Float64Array.from(SAMPLES_B);
      for (const [name, op] of Object.entries(OpCmp)) {
        const ts = new Float64Array(n);
        const c = new Float64Array(n);
        tsRealComparison(op, n, a, b, ts);
        addon!.tensorOpRealComparison!(op, n, a, b, c);
        assertVecEqual(`cmp ${name}`, ts, c, bitEqual);
      }
    }
  );

  it.skipIf(!have("tensorOpRealScalarComparison"))(
    "realScalarComparison matches C exactly (both sides)",
    () => {
      const n = SAMPLES.length;
      const arr = Float64Array.from(SAMPLES);
      const scalar = 0.5;
      for (const [name, op] of Object.entries(OpCmp)) {
        for (const left of [true, false]) {
          const ts = new Float64Array(n);
          const c = new Float64Array(n);
          tsRealScalarComparison(op, n, scalar, arr, left, ts);
          addon!.tensorOpRealScalarComparison!(op, n, scalar, arr, left, c);
          assertVecEqual(`scalar-cmp ${name} left=${left}`, ts, c, bitEqual);
        }
      }
    }
  );

  it.skipIf(!have("tensorOpRealFlatReduce"))(
    "realFlatReduce matches C (exact for max/min/any/all, ~ for sum/prod/mean)",
    () => {
      const a = Float64Array.from(SAMPLES);
      // A NaN-free vector for sum/prod/mean tolerance checks (NaN swamps them).
      const finite = Float64Array.from([1, -2, 3.5, -0.5, 4, 0.25, -6, 2]);
      const REDUCE_EXACT = new Set<number>([
        OpReduce.MAX,
        OpReduce.MIN,
        OpReduce.ANY,
        OpReduce.ALL,
      ]);
      for (const [name, op] of Object.entries(OpReduce)) {
        const exact = REDUCE_EXACT.has(op);
        const input = exact ? a : finite;
        const ts = new Float64Array(1);
        const c = new Float64Array(1);
        tsRealFlatReduce(op, input.length, input, ts);
        addon!.tensorOpRealFlatReduce!(op, input.length, input, c);
        const ok = exact ? bitEqual(ts[0], c[0]) : approxEqual(ts[0], c[0]);
        expect(ok, `reduce ${name}: ts=${ts[0]} c=${c[0]}`).toBe(true);
      }
    }
  );
});
