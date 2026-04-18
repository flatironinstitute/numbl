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

function forEachStmt(body: JitStmt[], cb: (s: JitStmt) => void): void {
  const visit = (s: JitStmt): void => {
    cb(s);
    switch (s.tag) {
      case "If":
        s.thenBody.forEach(visit);
        s.elseifBlocks.forEach(eb => eb.body.forEach(visit));
        if (s.elseBody) s.elseBody.forEach(visit);
        break;
      case "For":
      case "While":
        s.body.forEach(visit);
        break;
      default:
        break;
    }
  };
  body.forEach(visit);
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
      };
      meta.set(name, m);
      tensorNames.push(name);
    }
    return m;
  };

  for (let i = 0; i < params.length; i++) {
    if (argTypes[i].kind === "tensor") {
      ensureMeta(params[i], "param");
    }
  }

  for (let i = 0; i < outputNames.length; i++) {
    if (outputTypes[i]?.kind === "tensor") {
      const existing = meta.get(outputNames[i]);
      if (existing) {
        // Promote a tensor param named in the output list to paramOutput.
        existing.kind = "paramOutput";
      } else {
        ensureMeta(outputNames[i], "output");
      }
    }
  }

  // Single pass: seed tensor locals, fresh-alloc seeds, and
  // AssignIndex target flags.
  forEachStmt(body, s => {
    switch (s.tag) {
      case "Assign": {
        if (s.expr.jitType.kind === "tensor") {
          const m = ensureMeta(s.name, "local");
          if (isFreshTensorRhs(s.expr)) m.hasFreshAlloc = true;
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

  // Fixed-point propagation of hasFreshAlloc through Var-alias
  // (`dst = src` aliases src's buffer) and RangeSliceRead (`dst = src(a:b)`
  // produces a fresh allocation).
  let changed = true;
  while (changed) {
    changed = false;
    forEachStmt(body, s => {
      if (s.tag !== "Assign" || s.expr.jitType.kind !== "tensor") return;
      const dst = meta.get(s.name);
      if (!dst || dst.hasFreshAlloc) return;
      const e = s.expr;
      if (e.tag === "Var") {
        const src = meta.get(e.name);
        if (src?.hasFreshAlloc) {
          dst.hasFreshAlloc = true;
          changed = true;
        }
      } else if (e.tag === "RangeSliceRead") {
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
