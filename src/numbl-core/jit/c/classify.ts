/**
 * C-JIT: unified classification pass for tensor names.
 *
 * Replaces the ten ad-hoc sets/maps (`tensorVars`, `paramTensorNames`,
 * `outputTensorNames`, `localTensorNames`, `assignIndexTargets`,
 * `unshareTensorParams`, `tensorMaxDim`, `freshAllocTensors`,
 * `dynamicOutputs`, `paramOutputTensors`) that the codegen used to
 * build from 7+ body walks. Everything downstream — signature builder,
 * prelude, epilogue, emit helpers, fusion call — now reads from the
 * `TensorMeta` table this pass produces.
 */
import type { JitExpr, JitStmt, JitType } from "../jitTypes.js";
import { collectTensorUsage } from "../js/jitCodegenHoist.js";
import { walkStmts } from "./visit.js";

export type TensorKind =
  /** Tensor param, never appears in the output list. */
  | "param"
  /** Tensor param whose name is also in the output list. */
  | "paramOutput"
  /** Pure tensor output (not a param). */
  | "output"
  /** Tensor local (neither param nor output). */
  | "local";

export interface TensorMeta {
  kind: TensorKind;
  /** Max index arity on this name (1/2/3). 0 means the name is never
   *  used as an Index or AssignIndex base. */
  maxIndexDim: number;
  /** True when any `Assign(name, RHS)` has RHS ∈ {TensorLiteral,
   *  VConcatGrow, Call(zeros|ones), Var(src) where src.hasFreshAlloc,
   *  RangeSliceRead}. Propagated to fixed point. */
  hasFreshAlloc: boolean;
  /** True when any AssignIndex / AssignIndexRange / AssignIndexCol
   *  uses this name as the base. */
  isAssignIndexTarget: boolean;
  /** Derived: `kind === "param"` and (`isAssignIndexTarget` or
   *  `hasFreshAlloc`). Triggers the unshare-at-entry malloc+memcpy
   *  prelude so writes don't leak to the caller's buffer. */
  needsUnshare: boolean;
  /** Derived: `hasFreshAlloc` and (`kind === "output"` or
   *  `kind === "paramOutput"`). Triggers the `double **` dynamic-output
   *  ABI. */
  isDynamicOutput: boolean;
  /** True when this tensor's `JitType.isComplex === true` — either at
   *  the boundary (param / output type) or propagated from a complex
   *  RHS for locals. Drives paired imag-buffer plumbing: every complex
   *  tensor gets a `v_name_data_im` companion pointer, an extra ABI
   *  slot for boundaries, and imag malloc / free / copy parallel to
   *  the existing real path. */
  isComplex: boolean;
}

export interface ClassificationResult {
  /** All names with a tensor role, in insertion order (params first,
   *  then outputs, then locals in body order). */
  tensorNames: string[];
  /** Per-name metadata. */
  meta: Map<string, TensorMeta>;
  /** True iff any name has `hasFreshAlloc`. Lets callers skip the
   *  dynamic-output marshalling when no name uses it. */
  hasAnyDynamic: boolean;
  /** Name → is any tensor role. `meta.has(name)`. */
  tensorVars: Set<string>;
  /** Names with `kind === "param"` or `"paramOutput"`. */
  paramTensorNames: Set<string>;
  /** Names with `kind === "output"` or `"paramOutput"`. */
  outputTensorNames: Set<string>;
  /** Names with `kind === "local"`. */
  localTensorNames: Set<string>;
}

/** Does this Assign RHS allocate a fresh C-owned buffer? */
export function isFreshTensorRhs(expr: JitExpr): boolean {
  if (expr.tag === "TensorLiteral") return true;
  if (expr.tag === "VConcatGrow") return true;
  if (expr.tag === "Call" && (expr.name === "zeros" || expr.name === "ones")) {
    return expr.jitType.kind === "tensor";
  }
  // Tensor-return UserCall transfers ownership via the dynamic-output
  // ABI (C mallocs; caller frees). Feasibility already restricts this
  // to callees whose output[0] is isDynamicOutput, so the LHS receives
  // a freshly-allocated buffer.
  if (expr.tag === "UserCall" && expr.jitType.kind === "tensor") return true;
  return false;
}

export function analyzeTensorUsage(
  body: JitStmt[],
  params: string[],
  argTypes: JitType[],
  outputNames: string[],
  outputTypes: JitType[]
): ClassificationResult {
  const tensorNames: string[] = [];
  const meta = new Map<string, TensorMeta>();

  const ensureMeta = (name: string, defaultKind: TensorKind): TensorMeta => {
    let m = meta.get(name);
    if (!m) {
      m = {
        kind: defaultKind,
        maxIndexDim: 0,
        hasFreshAlloc: false,
        isAssignIndexTarget: false,
        needsUnshare: false,
        isDynamicOutput: false,
        isComplex: false,
      };
      meta.set(name, m);
      tensorNames.push(name);
    }
    return m;
  };

  for (let i = 0; i < params.length; i++) {
    const t = argTypes[i];
    if (t.kind === "tensor") {
      const m = ensureMeta(params[i], "param");
      if (t.isComplex === true) m.isComplex = true;
    }
  }

  for (let i = 0; i < outputNames.length; i++) {
    const t = outputTypes[i];
    if (t?.kind === "tensor") {
      const existing = meta.get(outputNames[i]);
      if (existing) {
        // Promote a tensor param named in the output list to paramOutput.
        existing.kind = "paramOutput";
        if (t.isComplex === true) existing.isComplex = true;
      } else {
        const m = ensureMeta(outputNames[i], "output");
        if (t.isComplex === true) m.isComplex = true;
      }
    }
  }

  // Single pass: seed tensor locals, fresh-alloc seeds, and
  // AssignIndex target flags.
  walkStmts(body, s => {
    switch (s.tag) {
      case "Assign": {
        if (s.expr.jitType.kind === "tensor") {
          const m = ensureMeta(s.name, "local");
          if (isFreshTensorRhs(s.expr)) m.hasFreshAlloc = true;
          if (s.expr.jitType.isComplex === true) m.isComplex = true;
        }
        return;
      }
      case "AssignIndex":
      case "AssignIndexRange":
      case "AssignIndexCol": {
        const m = meta.get(s.baseName);
        if (m) m.isAssignIndexTarget = true;
        return;
      }
      default:
        return;
    }
  });

  const usage = collectTensorUsage(body);
  for (const [name, u] of usage) {
    if (!u.isReal) continue;
    const d = Math.max(u.maxReadDim, u.maxWriteDim);
    const m = meta.get(name);
    if (m && d > 0) m.maxIndexDim = d;
  }

  // Fixed-point propagation of hasFreshAlloc:
  //   - `dst = src` (Var alias): fresh iff src is fresh
  //   - `dst = src(a:b)` (RangeSliceRead): always fresh
  //   - Tensor-producing Unary/Binary/Call: always fresh — the codegen
  //     must malloc a new buffer because the result length tracks the
  //     operand length, which may vary across loop iterations. Marking
  //     these as fresh forces isDynamicOutput, which gates the free+malloc
  //     path in emit.ts; without it, a paramOutput with a caller-aliased
  //     buffer would be overflowed by a larger-sized elemwise result.
  let changed = true;
  while (changed) {
    changed = false;
    walkStmts(body, s => {
      if (s.tag !== "Assign" || s.expr.jitType.kind !== "tensor") return;
      const dst = meta.get(s.name);
      if (!dst) return;
      const e = s.expr;
      // Propagate complexness from the RHS's (unified) jitType. The
      // JIT lowerer unifies Var types across branches and loops, so
      // `e.jitType.isComplex` is authoritative for Var aliases too.
      // Catches `z = conj(x)`, `z = x + 1i*y`, `z = x(:)` (complex x),
      // aliased `b = a` where `a` becomes complex in a later iter, etc.
      if (
        !dst.isComplex &&
        e.jitType.kind === "tensor" &&
        e.jitType.isComplex === true
      ) {
        dst.isComplex = true;
        changed = true;
      }
      if (dst.hasFreshAlloc) return;
      if (e.tag === "Var") {
        // `y = x` is a deep copy at MATLAB semantics. The dst's length
        // tracks the src's, which isn't knowable at JS-wrapper alloc
        // time if src is a param whose size differs from the first
        // tensor param. Force the dynamic-output ABI so C mallocs to
        // the right size instead of overflowing a pre-alloc'd buffer.
        dst.hasFreshAlloc = true;
        changed = true;
      } else if (
        e.tag === "RangeSliceRead" ||
        e.tag === "Unary" ||
        e.tag === "Binary" ||
        e.tag === "Call"
      ) {
        dst.hasFreshAlloc = true;
        changed = true;
      }
    });
  }

  let hasAnyDynamic = false;
  for (const [, m] of meta) {
    m.needsUnshare =
      m.kind === "param" && (m.isAssignIndexTarget || m.hasFreshAlloc);
    m.isDynamicOutput =
      m.hasFreshAlloc && (m.kind === "output" || m.kind === "paramOutput");
    if (m.hasFreshAlloc) hasAnyDynamic = true;
  }

  const tensorVars = new Set<string>(meta.keys());
  const paramTensorNames = new Set<string>();
  const outputTensorNames = new Set<string>();
  const localTensorNames = new Set<string>();
  for (const [name, m] of meta) {
    if (m.kind === "param" || m.kind === "paramOutput")
      paramTensorNames.add(name);
    if (m.kind === "output" || m.kind === "paramOutput")
      outputTensorNames.add(name);
    if (m.kind === "local") localTensorNames.add(name);
  }

  return {
    tensorNames,
    meta,
    hasAnyDynamic,
    tensorVars,
    paramTensorNames,
    outputTensorNames,
    localTensorNames,
  };
}
