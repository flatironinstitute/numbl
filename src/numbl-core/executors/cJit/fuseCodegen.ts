/**
 * c-jit-fuse codegen — single-pass element-wise C kernel.
 *
 * Emits one C function whose body is a single `for (long i = 0; i < n; i++)`
 * loop with `out[i] = <fused expression>;`. The fused expression is
 * the AST RHS, with each `Ident` mapped to either `tN[i]` (tensor)
 * or `sN` (scalar) via `emitElemwiseExpr`'s leaf callback.
 *
 * ABI:
 *   void <fnName>(double *out, long n,
 *                 const double *t0, const double *t1, ...,
 *                 double s0, double s1, ...);
 */

import type { FuseClassification } from "./fuseAnalyze.js";
import { emitElemwiseExpr } from "./elemwiseCodegen.js";

/** Emit a complete C source file for a fuse classification. */
export function generateFuseCSource(
  fnName: string,
  cls: FuseClassification
): string {
  const tensorIndex = new Map<string, number>();
  cls.tensorInputs.forEach((n, i) => tensorIndex.set(n, i));
  const scalarIndex = new Map<string, number>();
  cls.scalarInputs.forEach((n, i) => scalarIndex.set(n, i));

  const params: string[] = [];
  // Inputs are NOT restrict-qualified: two distinct names can refer
  // to the same underlying Float64Array (MATLAB copy-on-write makes
  // `b = a` share buffers), so we must allow input aliasing. `out`
  // IS restrict-qualified: the executor always allocates a fresh
  // Float64Array, so no input pointer can alias it. That alone is
  // usually enough to unlock vectorization.
  for (let i = 0; i < cls.tensorInputs.length; i++) {
    params.push(`const double *t${i}`);
  }
  for (let i = 0; i < cls.scalarInputs.length; i++) {
    params.push(`double s${i}`);
  }

  const identToC = (name: string): string => {
    const ti = tensorIndex.get(name);
    if (ti !== undefined) return `t${ti}[i]`;
    const si = scalarIndex.get(name);
    if (si !== undefined) return `s${si}`;
    throw new Error(`fuse codegen: unknown identifier ${name}`);
  };
  const body = emitElemwiseExpr(cls.rhs, identToC);

  const lines: string[] = [];
  lines.push(`#include <math.h>`);
  lines.push(``);
  lines.push(
    `void ${fnName}(double *restrict out, long n${params.length > 0 ? ", " + params.join(", ") : ""}) {`
  );
  // `#pragma omp simd` asks the compiler to vectorize the loop even
  // when it would otherwise be conservative (e.g., due to function-
  // call boundaries on math.h calls). Combined with `restrict out`,
  // GCC/Clang will emit straight-line SIMD code on -march=native.
  lines.push(`  #pragma omp simd`);
  lines.push(`  for (long i = 0; i < n; i++) {`);
  lines.push(`    out[i] = ${body};`);
  lines.push(`  }`);
  lines.push(`}`);
  return lines.join("\n") + "\n";
}
