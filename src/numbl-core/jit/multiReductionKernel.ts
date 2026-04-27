/**
 * e1 — C kernel for multi-reduction scalar assigns.
 *
 * A MATLAB line like
 *
 *   red_acc = red_acc + (sum(x) + mean(x) + max(x) + min(x));
 *
 * has four reductions over the same vector. The default JS-JIT path
 * emits four `$h.tSum` / `$h.ib_*` helper calls, each of which scans
 * the whole vector. This module emits a single-pass C kernel that
 * computes every requested reduction in one loop and writes results
 * into caller-allocated scalar slots.
 *
 * Specialised per op-set: a group of `{sum, max, min}` compiles to a
 * different kernel than `{sum, mean, max, min}`. Source-addressed by
 * FNV-1a hash so the JS `$h.$kernels[...]` cache dedupes repeated call
 * sites.
 *
 * NaN handling: `-ffast-math` is on for the compile (matches the other
 * e1 kernels), so naive `isnan` is folded to `false`. The kernel uses
 * an inline bit-pattern NaN check to drive MATLAB's omit-NaN semantics
 * for `max`/`min` and records an `any_non_nan` flag the JS side uses
 * to map an all-NaN input to NaN.
 */

import { fnv1a64Hex } from "./hash.js";
import { ompParallelThreshold } from "./heavyOps.js";

/** Reductions we can fuse into one pass. `any` / `all` are excluded
 *  because their short-circuit `break` would prematurely stop the
 *  other accumulators. */
export type MultiReduceOp = "sum" | "prod" | "max" | "min" | "mean";

export interface MultiReductionKernelInfo {
  /** Hash-derived C function name, e.g. `mr_3a7f81b2...`. */
  kernelName: string;
  /** Full C source string. */
  cSource: string;
  /** koffi function signature. */
  koffiSig: string;
  /** Content hash. */
  hash: string;
  /**
   * Output slot layout. Each reduction in the kernel writes to its own
   * Float64 slot, in the order of this array. `any_non_nan` (a 0/1 flag
   * stored as double) is at the end when `hasMinOrMax` is true.
   * The JS caller allocates a `Float64Array(slotCount)` and reads slots
   * by index after the call.
   */
  slotNames: string[];
  /** True when the kernel emits an `any_non_nan` slot at index
   *  `slotNames.length - 1`. */
  hasAnyNonNan: boolean;
}

/**
 * Build a multi-reduction kernel for the given op set. `ops` should be
 * a deduplicated list of reductions to compute (e.g. ["sum", "max"]).
 * The returned `slotNames` preserves insertion order for indexing; if
 * the op set contains `max`/`min`, an extra `any_non_nan` slot is
 * appended (the JS side uses it to override the sentinel max/min with
 * NaN when every input element was NaN).
 *
 * When `par` is true, the per-element loop is emitted as
 * `#pragma omp parallel for simd reduction(...)` with one reduction
 * clause per accumulator and an `if(n >= T)` gate that falls back to
 * serial below the threshold. Requires the caller to link with
 * `-fopenmp`; e1's `install.ts` already does this when libgomp is
 * available. When `par` is false, the loop is emitted as plain
 * `#pragma omp simd` (SIMD-only, single-threaded).
 */
export function emitMultiReductionKernel(
  ops: readonly MultiReduceOp[],
  par: boolean = false
): MultiReductionKernelInfo {
  // Sanitise input: dedupe while preserving first-occurrence order, and
  // drop `mean` from the op-loop list (we piggy-back on `sum` and
  // divide post-loop on the JS side).
  const seen = new Set<MultiReduceOp>();
  const specOps: MultiReduceOp[] = [];
  for (const op of ops) {
    if (!seen.has(op)) {
      seen.add(op);
      specOps.push(op);
    }
  }

  const hasSum = seen.has("sum");
  const hasMean = seen.has("mean");
  const hasProd = seen.has("prod");
  const hasMax = seen.has("max");
  const hasMin = seen.has("min");
  const needSumAcc = hasSum || hasMean;
  const hasAnyNonNan = hasMax || hasMin;

  if (!needSumAcc && !hasProd && !hasMax && !hasMin) {
    throw new Error(
      `multiReductionKernel: empty op set — need at least one reduction`
    );
  }

  // Slot layout: output order is deterministic per op-set. Mean has no
  // slot of its own — the JS side reads the sum slot and divides by n.
  const slotNames: string[] = [];
  const slotOf = new Map<string, number>();
  const addSlot = (name: string) => {
    slotOf.set(name, slotNames.length);
    slotNames.push(name);
  };
  if (needSumAcc) addSlot("sum");
  if (hasProd) addSlot("prod");
  if (hasMax) addSlot("max");
  if (hasMin) addSlot("min");
  if (hasAnyNonNan) addSlot("any_non_nan");

  // Generate C.
  const lines: string[] = [];
  lines.push(`#include <math.h>`);
  lines.push(`#include <stdint.h>`);
  lines.push(`#include <string.h>`);
  lines.push(``);
  lines.push(
    `void __KERNEL_NAME__(int64_t n, const double *in_x, double *out)`
  );
  lines.push(`{`);
  if (needSumAcc) lines.push(`    double acc_sum = 0.0;`);
  if (hasProd) lines.push(`    double acc_prod = 1.0;`);
  if (hasMax) lines.push(`    double acc_max = (-1.0/0.0);`);
  if (hasMin) lines.push(`    double acc_min = (1.0/0.0);`);
  if (hasAnyNonNan) lines.push(`    int acc_any_non_nan = 0;`);

  // Loop pragma. With `par`, upgrade to `parallel for simd` with one
  // reduction clause per accumulator. `acc_any_non_nan` is an int flag
  // combined via bitwise OR (reduction(|:...)) so it ends up set if
  // any thread saw a non-NaN element. The `if(n >= T)` clause keeps
  // small-n fully serial so the thread-spawn cost never exceeds the
  // work.
  if (par) {
    const clauses: string[] = [];
    if (needSumAcc) clauses.push(`reduction(+:acc_sum)`);
    if (hasProd) clauses.push(`reduction(*:acc_prod)`);
    if (hasMax) clauses.push(`reduction(max:acc_max)`);
    if (hasMin) clauses.push(`reduction(min:acc_min)`);
    if (hasAnyNonNan) clauses.push(`reduction(|:acc_any_non_nan)`);
    clauses.push(`if(n >= ${ompParallelThreshold()})`);
    lines.push(`    #pragma omp parallel for simd ${clauses.join(" ")}`);
  } else {
    lines.push(`    #pragma omp simd`);
  }
  lines.push(`    for (int64_t i = 0; i < n; i++) {`);
  lines.push(`        double v = in_x[i];`);
  if (needSumAcc) lines.push(`        acc_sum += v;`);
  if (hasProd) lines.push(`        acc_prod *= v;`);
  if (hasAnyNonNan) {
    // Inline bit-pattern NaN check — survives -ffast-math and is
    // lanewise integer ops (memcpy load + bitmask + compare), so the
    // additive accumulators above can still vectorise on compilers
    // that honour `#pragma omp simd` for the containing loop.
    lines.push(`        uint64_t b;`);
    lines.push(`        memcpy(&b, &v, sizeof(b));`);
    lines.push(
      `        int is_nan = (b & 0x7FFFFFFFFFFFFFFFULL) > 0x7FF0000000000000ULL;`
    );
    lines.push(`        if (!is_nan) {`);
    if (hasMax) lines.push(`            if (v > acc_max) acc_max = v;`);
    if (hasMin) lines.push(`            if (v < acc_min) acc_min = v;`);
    lines.push(`            acc_any_non_nan = 1;`);
    lines.push(`        }`);
  }
  lines.push(`    }`);

  // Write outputs.
  if (needSumAcc) lines.push(`    out[${slotOf.get("sum")!}] = acc_sum;`);
  if (hasProd) lines.push(`    out[${slotOf.get("prod")!}] = acc_prod;`);
  if (hasMax) lines.push(`    out[${slotOf.get("max")!}] = acc_max;`);
  if (hasMin) lines.push(`    out[${slotOf.get("min")!}] = acc_min;`);
  if (hasAnyNonNan) {
    lines.push(
      `    out[${slotOf.get("any_non_nan")!}] = (double)acc_any_non_nan;`
    );
  }
  lines.push(`}`);

  const template = lines.join("\n") + "\n";
  const hash = fnv1a64Hex(template);
  const kernelName = `mr_${hash}`;
  const cSource = template.replace("__KERNEL_NAME__", kernelName);
  const koffiSig = `void ${kernelName}(int64_t, double *, double *)`;

  return {
    kernelName,
    cSource,
    koffiSig,
    hash,
    slotNames,
    hasAnyNonNan,
  };
}
