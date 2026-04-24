/**
 * e2 whole-loop C emission.
 *
 * Given a classified loop body (`BodyStmt[]`) plus the parameter lists
 * that describe how env values flow in and out, emit a single C
 * function that runs the whole `for varName = lo:hi` loop in one call.
 *
 * Three BodyStmt shapes are supported:
 *
 *   scalar_assign    `s = s + sin(i)`          → one C statement per iter
 *   tensor_write     `y(i) = sin(i*0.01)`      → `v_y[(int64_t)idx-1] = ...`
 *   tensor_local     `c = a.*b + i*0.001`      → no code emitted here; its
 *                                                 per-element expression is
 *                                                 substituted into whichever
 *                                                 reduction consumes it
 *
 * Reductions: a `scalar_assign` carries a list of `sum(<tensor_local>)`
 * rewrites that were pulled out of its RHS upstream. Each is emitted as
 * an inline inner `for __j` loop that accumulates the tensor_local's
 * per-element expression into a fresh local. Chained tensor_locals
 * (`d = sqrt(c+1)` where c is itself a tensor_local) fuse through
 * recursively, so no intermediate buffer is materialized.
 */

import type { JitExpr, JitType } from "../jitTypes.js";
import { emitFusedScalarExpr, type FusedTarget } from "../fusedScalarEmit.js";
import { C_SCALAR_TARGET, formatNumberLiteral } from "../c/context.js";
import { fnv1a64Hex } from "../e1/hash.js";
import { getIBuiltin } from "../../interpreter/builtins/index.js";

/** Scalar math builtins we emit as direct C library calls. We bypass
 *  each IBuiltin's `jitEmitC` here because some of those reject based
 *  on type narrowing (e.g. `sqrt` requires `isNonneg` and we don't
 *  propagate sign through Binary ops) — but in a pure-real scalar loop
 *  the C semantics (NaN on negative sqrt, etc.) match what a MATLAB
 *  user gets from `sqrt` on real numeric input.
 *
 *  Exported so the driver's pre-lowering analysis can treat these
 *  names as non-env references. */
export const LOOP_SCALAR_BUILTINS: Record<string, string> = {
  sin: "sin",
  cos: "cos",
  tan: "tan",
  asin: "asin",
  acos: "acos",
  atan: "atan",
  sinh: "sinh",
  cosh: "cosh",
  tanh: "tanh",
  exp: "exp",
  log: "log",
  log2: "log2",
  log10: "log10",
  sqrt: "sqrt",
  abs: "fabs",
  floor: "floor",
  ceil: "ceil",
  round: "round",
  atan2: "atan2",
  pow: "pow",
  fmod: "fmod",
};

/** A fused reduction lifted out of a `scalar_assign`'s RHS.
 *  `sum(<tensorLocal>)` in the source becomes a synthetic scalar ident
 *  `synthName`; the emitter materializes it as an inline inner loop
 *  that accumulates `tensorLocal`'s per-element expression. */
export interface Reduction {
  synthName: string;
  tensorLocal: string;
  op: "sum";
}

/** A body statement in a form ready for C emission. */
export type BodyStmt =
  | {
      kind: "scalar_assign";
      name: string;
      rhs: JitExpr;
      reductions: Reduction[];
    }
  | { kind: "tensor_write"; name: string; idxRhs: JitExpr; rhs: JitExpr }
  | {
      kind: "tensor_local";
      name: string;
      elemExpr: JitExpr;
      lengthTensor: string;
    };

type TensorLocalMap = ReadonlyMap<
  string,
  { elemExpr: JitExpr; lengthTensor: string }
>;

/** Mangle a MATLAB scalar name to a C local-variable name. Prefix keeps
 *  it out of the way of our bookkeeping locals (`lo`, `hi`, `__iv`). */
export function v(name: string): string {
  return `v_${name}`;
}

/** Name for the `int64_t` length companion that travels alongside each
 *  tensor param so inner reductions can bound their inline `__j` loop. */
export function lenN(name: string): string {
  return `__len_${v(name)}`;
}

/** Name for the `double *` output buffer that materializes a tensor_local
 *  on the last iteration so its post-loop value is visible in the env. */
function outBuf(name: string): string {
  return `${v(name)}_out`;
}

/** Names of all tensor_locals in the body, in body-declaration order.
 *  Callers use this to allocate matching output buffers in the same
 *  order as the kernel's param list. */
export function tensorLocalNames(body: BodyStmt[]): string[] {
  const out: string[] = [];
  for (const b of body) if (b.kind === "tensor_local") out.push(b.name);
  return out;
}

export function emitLoopKernel(
  scalarInputVars: string[],
  tensorInputVars: string[],
  tensorInoutVars: string[],
  inoutVars: string[],
  loopVar: string,
  body: BodyStmt[]
): { cSource: string; kernelName: string; koffiSig: string } {
  const tlOutputNames = tensorLocalNames(body);
  const params = buildParams(
    scalarInputVars,
    tensorInputVars,
    tensorInoutVars,
    inoutVars,
    tlOutputNames
  );
  const tensorLocalMap = indexTensorLocals(body);
  const { outerFt, innerFt } = buildFusedTargets(tensorLocalMap);
  const bodyLines = emitBodyLines(
    body,
    loopVar,
    inoutVars,
    tensorLocalMap,
    outerFt,
    innerFt
  );

  const prologue = "#include <math.h>\n#include <stdint.h>\n\n";
  const template =
    prologue +
    `void __KERNEL_NAME__(${params.join(", ")})\n` +
    `{\n${bodyLines.join("\n")}\n}\n`;

  const hash = fnv1a64Hex(template);
  const kernelName = `e2l_${hash}`;
  const cSource = template.replace("__KERNEL_NAME__", kernelName);
  const koffiSig = buildKoffiSig(
    kernelName,
    scalarInputVars,
    tensorInputVars,
    tensorInoutVars,
    inoutVars,
    tlOutputNames
  );
  return { cSource, kernelName, koffiSig };
}

function buildParams(
  scalarInputVars: string[],
  tensorInputVars: string[],
  tensorInoutVars: string[],
  inoutVars: string[],
  tlOutputNames: string[]
): string[] {
  const params = ["int64_t lo", "int64_t hi"];
  for (const n of scalarInputVars) params.push(`double ${v(n)}`);
  // Tensor params each get a length companion `int64_t __len_<v_name>`
  // right after the pointer so inner inline reductions can bound their
  // inner `__j` loop without a separate call-time shape lookup.
  for (const n of tensorInputVars) {
    params.push(`const double *${v(n)}`);
    params.push(`int64_t ${lenN(n)}`);
  }
  for (const n of tensorInoutVars) {
    params.push(`double *${v(n)}`);
    params.push(`int64_t ${lenN(n)}`);
  }
  for (const n of inoutVars) params.push(`double *${v(n)}_ptr`);
  // One `double *` per tensor_local: the caller pre-allocates a buffer
  // sized to the length-input-tensor, and the kernel fills it on the
  // last iteration to give MATLAB-correct post-loop visibility.
  for (const n of tlOutputNames) params.push(`double *${outBuf(n)}`);
  return params;
}

function buildKoffiSig(
  kernelName: string,
  scalarInputVars: string[],
  tensorInputVars: string[],
  tensorInoutVars: string[],
  inoutVars: string[],
  tlOutputNames: string[]
): string {
  const koffiParts: string[] = ["int64_t", "int64_t"];
  for (let i = 0; i < scalarInputVars.length; i++) koffiParts.push("double");
  for (let i = 0; i < tensorInputVars.length; i++) {
    koffiParts.push("double *");
    koffiParts.push("int64_t");
  }
  for (let i = 0; i < tensorInoutVars.length; i++) {
    koffiParts.push("double *");
    koffiParts.push("int64_t");
  }
  for (let i = 0; i < inoutVars.length; i++) koffiParts.push("double *");
  for (let i = 0; i < tlOutputNames.length; i++) koffiParts.push("double *");
  return `void ${kernelName}(${koffiParts.join(", ")})`;
}

function indexTensorLocals(body: BodyStmt[]): TensorLocalMap {
  const map = new Map<string, { elemExpr: JitExpr; lengthTensor: string }>();
  for (const b of body) {
    if (b.kind === "tensor_local") {
      map.set(b.name, { elemExpr: b.elemExpr, lengthTensor: b.lengthTensor });
    }
  }
  return map;
}

/** Build the FusedTarget pair the scalar emitter needs. Two targets
 *  exist because tensor references mean different things in the two
 *  contexts the kernel emits:
 *    - Outer (one scalar iteration): bare tensor Var refs are illegal;
 *      only scalar-index reads (`y(i)`) and reductions (rewritten away
 *      upstream) are valid.
 *    - Inner (inside an inline reduction loop): a tensor Var resolves
 *      to the per-element read `v_<name>[__j]` — unless the name is
 *      itself a tensor_local, in which case we recursively inline its
 *      elemExpr. */
function buildFusedTargets(tensorLocalMap: TensorLocalMap): {
  outerFt: FusedTarget;
  innerFt: FusedTarget;
} {
  const outerFt: FusedTarget = {
    formatNumber: formatNumberLiteral,
    mangle: v,
    tensorElemRead: () => {
      throw new Error(
        "e2 loop kernel: element-wise tensor read not valid in scalar loop"
      );
    },
    tensorScalarIndexRead: (name, idxC) => `${v(name)}[(int64_t)(${idxC}) - 1]`,
    emitBuiltinCall: (name, cargs) => {
      const cFn = LOOP_SCALAR_BUILTINS[name];
      if (cFn) return `${cFn}(${cargs.join(", ")})`;
      const ib = getIBuiltin(name);
      if (!ib?.jitEmitC) return null;
      const argTypes: JitType[] = cargs.map(() => ({ kind: "number" }));
      return ib.jitEmitC(cargs, argTypes);
    },
  };

  const innerFt: FusedTarget = {
    ...outerFt,
    tensorElemRead: name => {
      const tl = tensorLocalMap.get(name);
      if (tl) {
        const inner = emitFusedScalarExpr(
          tl.elemExpr,
          new Set(),
          new Set(),
          C_SCALAR_TARGET,
          innerFt
        );
        return `(${inner})`;
      }
      return `${v(name)}[__j]`;
    },
  };

  return { outerFt, innerFt };
}

function emitBodyLines(
  body: BodyStmt[],
  loopVar: string,
  inoutVars: string[],
  tensorLocalMap: TensorLocalMap,
  outerFt: FusedTarget,
  innerFt: FusedTarget
): string[] {
  const lines: string[] = [];
  // Read inout initial values into plain locals so the body sees them
  // as regular doubles rather than through pointers.
  for (const n of inoutVars) {
    lines.push(`    double ${v(n)} = *${v(n)}_ptr;`);
  }
  lines.push(`    for (int64_t __iv = lo; __iv <= hi; __iv++) {`);
  lines.push(`        double ${v(loopVar)} = (double)__iv;`);

  const declared = new Set<string>(inoutVars);
  for (const b of body) {
    if (b.kind === "tensor_local") {
      // For reductions, the elemExpr is inlined directly into the
      // consumer's inner `__j` loop — no intermediate buffer needed.
      //
      // For post-loop MATLAB visibility (the user may read this
      // variable after the loop), we fill the caller-provided output
      // buffer on the last iteration, at this body-order position —
      // so the captured value reflects the same state MATLAB would
      // compute at its assignment point.
      const elemC = emitFusedScalarExpr(
        b.elemExpr,
        new Set(),
        new Set(),
        C_SCALAR_TARGET,
        innerFt
      );
      lines.push(`        if (__iv == hi) {`);
      lines.push(
        `            for (int64_t __j = 0; __j < ${lenN(b.lengthTensor)}; __j++) {`
      );
      lines.push(`                ${outBuf(b.name)}[__j] = ${elemC};`);
      lines.push(`            }`);
      lines.push(`        }`);
      continue;
    }
    if (b.kind === "scalar_assign") {
      emitReductions(b.reductions, tensorLocalMap, innerFt, lines);
      const rhsC = emitFusedScalarExpr(
        b.rhs,
        new Set(),
        new Set(),
        C_SCALAR_TARGET,
        outerFt
      );
      if (declared.has(b.name)) {
        lines.push(`        ${v(b.name)} = ${rhsC};`);
      } else {
        lines.push(`        double ${v(b.name)} = ${rhsC};`);
        declared.add(b.name);
      }
    } else {
      // tensor_write: y(i) = expr → v_y[(int64_t)(idx) - 1] = rhs;
      const idxC = emitFusedScalarExpr(
        b.idxRhs,
        new Set(),
        new Set(),
        C_SCALAR_TARGET,
        outerFt
      );
      const rhsC = emitFusedScalarExpr(
        b.rhs,
        new Set(),
        new Set(),
        C_SCALAR_TARGET,
        outerFt
      );
      lines.push(`        ${v(b.name)}[(int64_t)(${idxC}) - 1] = ${rhsC};`);
    }
  }
  lines.push(`    }`);
  for (const n of inoutVars) {
    lines.push(`    *${v(n)}_ptr = ${v(n)};`);
  }
  return lines;
}

/** Emit one inline inner loop per reduction consumed by the current
 *  scalar_assign. The accumulator is declared under the synthName's
 *  mangled form (`v_<synthName>`) so the outer emitter — which treats
 *  the synth ident as a normal Var — resolves the reference. */
function emitReductions(
  reductions: readonly Reduction[],
  tensorLocalMap: TensorLocalMap,
  innerFt: FusedTarget,
  lines: string[]
): void {
  for (const r of reductions) {
    const tl = tensorLocalMap.get(r.tensorLocal);
    if (!tl) {
      throw new Error(
        `e2 loop kernel: reduction refers to missing tensor_local '${r.tensorLocal}'`
      );
    }
    const elemC = emitFusedScalarExpr(
      tl.elemExpr,
      new Set(),
      new Set(),
      C_SCALAR_TARGET,
      innerFt
    );
    const accC = v(r.synthName);
    lines.push(`        double ${accC} = 0.0;`);
    lines.push(
      `        for (int64_t __j = 0; __j < ${lenN(tl.lengthTensor)}; __j++) {`
    );
    lines.push(`            ${accC} += ${elemC};`);
    lines.push(`        }`);
  }
}
