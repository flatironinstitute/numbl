/**
 * e2 — per-assign / chain kernel driver.
 *
 * Entry point `tryE2Assign` is called from `interpreterExec.ts` for
 * every `Assign` statement when `interp.experimental === "e2"`.
 *
 * Multi-LHS chain detection: scans consecutive suppressed Assigns
 * regardless of LHS name. For each chain LHS, uses scope-body liveness
 * (via `interp._currentScopeBody`) to decide whether the LHS escapes
 * (materializes as an `out_<name>` buffer) or is purely chain-local
 * (kept as a per-element stack-local). Reads of a chain LHS before
 * its first assign in the chain become `in_<name>` parameters.
 *
 * On success:
 *   - Single chain assign: handled like a one-stmt chain.
 *   - Multi-stmt chain: one C kernel runs all assigns, only escape
 *     LHSs materialize back to env. The returned `consumed` count
 *     tells the dispatcher how many sibling stmts to skip.
 *
 * Compilation failures are hard errors (RuntimeError). Classification
 * bails (non-classifiable RHS, mismatched lengths, etc.) silently fall
 * through (return null) to the regular interpreter path.
 */

import type { Stmt } from "../../parser/types.js";
import type { Interpreter } from "../../interpreter/interpreter.js";
import { RuntimeError } from "../../runtime/error.js";
import type { JitType } from "../../jit/jitTypes.js";
import { jitTypeKey } from "../../jit/jitTypes.js";
import {
  type RuntimeTensor,
  type RuntimeValue,
  FloatXArray,
  isRuntimeTensor,
} from "../../runtime/types.js";
import { inferJitType } from "../../interpreter/builtins/types.js";
import {
  classifyAssignChain,
  classifyExpr,
  isWorthCompiling,
  matchTrailingReduction,
  type ChainAssignClassification,
  type ChainClassification,
  type ClassifyResult,
  type TrailingReductionMatch,
} from "./classify.js";
import { lowerAstToJitExpr, E2LowerError } from "./astToJitExpr.js";
import { emitE2ChainKernel, type ChainAssignSpec } from "./chainKernelEmit.js";
import { emitE2ReductionKernel } from "./reductionKernelEmit.js";
import {
  emitE2ComplexChainKernel,
  type ComplexChainAssignSpec,
} from "./complexChainKernelEmit.js";
import {
  chainCacheGet,
  chainCacheSet,
  E2_BAILED,
  type E2CacheEntry,
  type E2ComplexInfo,
  type E2ReductionInfo,
} from "./cache.js";
import { getE2CompileFn, e2MinElems } from "./compileFn.js";
import { isNameReferencedOutsideStmts } from "./liveness.js";
import { BinaryOperation } from "../../parser/types.js";
import { tryE2MultiReduction } from "./multiReductionDriver.js";
import { isOpenmpAvailable } from "../../jit/e1/openmpFlag.js";

interface InputDescriptor {
  name: string;
  jitType: JitType;
  /** Concrete runtime value (for marshaling the call). */
  value: unknown;
}

/** Resolve a name to a supported e2 input descriptor, or null if it's
 *  unbound / has an unsupported type. `extraBindings` takes precedence
 *  over the interpreter env (synthetic opaque-root names land there). */
function resolveSupportedInput(
  interp: Interpreter,
  name: string,
  extraBindings: Map<string, unknown>
): InputDescriptor | null {
  let value = extraBindings.get(name);
  if (value === undefined) value = interp.env.get(name);
  if (value === undefined) return null;
  const jt = inferJitType(value);
  if (jt.kind === "tensor") {
    const t = value as RuntimeTensor;
    if (!(t.data instanceof Float64Array)) return null;
    if (jt.isComplex) {
      if (!t.imag || !(t.imag instanceof Float64Array)) return null;
    }
  } else if (
    jt.kind !== "number" &&
    jt.kind !== "boolean" &&
    jt.kind !== "complex_or_number"
  ) {
    return null;
  }
  return { name, jitType: jt, value };
}

/** Resolve every classifier-output Ident to its concrete runtime value
 *  and JitType. Returns null if any name is missing or unsupported. */
function gatherInputs(
  interp: Interpreter,
  envIdents: ReadonlySet<string>,
  extraBindings: Map<string, unknown>
): InputDescriptor[] | null {
  const out: InputDescriptor[] = [];
  for (const name of envIdents) {
    const inp = resolveSupportedInput(interp, name, extraBindings);
    if (!inp) return null;
    out.push(inp);
  }
  return out;
}

/** True iff `expr` references at least one tensor-typed Var (so a
 *  per-element reduction loop has anything to reduce over). Used by
 *  the standalone-reduction detector to filter out e.g.
 *  `acc = sum(2.0)` which would loop over a constant. */
function hasAnyTensorRef(
  expr: import("../../jit/jitTypes.js").JitExpr
): boolean {
  switch (expr.tag) {
    case "Var":
      return expr.jitType.kind === "tensor";
    case "Binary":
      return hasAnyTensorRef(expr.left) || hasAnyTensorRef(expr.right);
    case "Unary":
      return hasAnyTensorRef(expr.operand);
    case "Call":
      return expr.args.some(a => hasAnyTensorRef(a));
    default:
      return false;
  }
}

/** Try to reuse `name`'s existing buffer for the kernel's output —
 *  only safe when it's a unique-reference Float64Array tensor of the
 *  exact length we need. Saves a fresh allocation per kernel call. */
function reuseOrAllocOutBuf(
  prev: RuntimeValue | undefined,
  n: number
): Float64Array {
  if (
    prev !== undefined &&
    isRuntimeTensor(prev) &&
    prev._rc === 1 &&
    prev.data instanceof Float64Array &&
    prev.data.length === n &&
    !prev.imag
  ) {
    return prev.data;
  }
  return new FloatXArray(n) as Float64Array;
}

/** Complex-output variant: reuse the existing complex tensor's paired
 *  data+imag buffers when the env value is a unique-reference complex
 *  tensor of the right length. Avoids two fresh allocations per call
 *  on the hot path (e.g. loop body writes `u1 = z.*z + c` 100 times). */
function reuseOrAllocComplexOutBufs(
  prev: RuntimeValue | undefined,
  n: number
): { data: Float64Array; imag: Float64Array } {
  if (
    prev !== undefined &&
    isRuntimeTensor(prev) &&
    prev._rc === 1 &&
    prev.data instanceof Float64Array &&
    prev.data.length === n &&
    prev.imag instanceof Float64Array &&
    prev.imag.length === n
  ) {
    return { data: prev.data, imag: prev.imag };
  }
  return {
    data: new FloatXArray(n) as Float64Array,
    imag: new FloatXArray(n) as Float64Array,
  };
}

function buildSig(
  inputs: InputDescriptor[],
  inputLhsNames: string[],
  escapeLhsNames: string[],
  chainLength: number
): string {
  const parts: string[] = [
    `len=${chainLength}`,
    `inLhs=${inputLhsNames.join(",")}`,
    `outLhs=${escapeLhsNames.join(",")}`,
  ];
  for (const inp of inputs) {
    parts.push(`${inp.name}=${jitTypeKey(inp.jitType)}`);
  }
  return parts.join("|");
}

function describeKernel(
  file: string,
  line: number,
  assigns: ChainAssignClassification[],
  escapeLhsNames: string[],
  inputs: InputDescriptor[],
  outputN: number,
  trailing: TrailingReductionMatch | null
): string {
  const inputDesc = inputs
    .map(i => {
      if (i.jitType.kind === "tensor") {
        const t = i.value as RuntimeTensor;
        return `${i.name}: tensor double[${t.data.length}]`;
      }
      return `${i.name}: scalar double`;
    })
    .join(", ");
  const outParts: string[] = [];
  for (const n of escapeLhsNames) {
    outParts.push(`${n}: tensor double[${outputN}]`);
  }
  if (trailing) {
    outParts.push(
      `${trailing.accName}: scalar double via ${trailing.reduceName}${
        trailing.hasAccumulate ? " (accumulate)" : ""
      }`
    );
  }
  const note = trailing
    ? assigns.length === 0
      ? `${trailing.accName} = ${trailing.reduceName}(<expr>)`
      : `chain of ${assigns.length} assigns (${assigns
          .map(a => a.stmt.name)
          .join(
            ", "
          )}) + trailing ${trailing.reduceName} into ${trailing.accName}`
    : assigns.length === 1
      ? `${assigns[0].stmt.name} = <expr>`
      : `chain of ${assigns.length} assigns (${assigns
          .map(a => a.stmt.name)
          .join(", ")})`;
  return [
    `e2 kernel: ${note} @ ${file}:${line}`,
    `inputs:  ${inputDesc || "(none)"}`,
    `outputs: ${outParts.join(", ") || "(none)"}`,
  ].join("\n * ");
}

// ── Public entry point ─────────────────────────────────────────────────

/**
 * Try to compile a chain (1+ stmts) starting at `stmt`. Returns the
 * total number of consumed sibling stmts on success (>= 1), or `null`
 * to signal fall-back to the regular interpreter path.
 */
export function tryE2Assign(
  interp: Interpreter,
  stmt: Stmt & { type: "Assign" }
): number | null {
  const siblings = interp._postSiblings;
  const nextIdx = interp._postSiblingsIdx;
  if (!siblings || nextIdx <= 0 || siblings[nextIdx - 1] !== stmt) {
    return null;
  }
  const startIdx = nextIdx - 1;
  // Try chain (with optional trailing reduction or standalone reduction).
  const chained = tryChain(interp, siblings, startIdx);
  if (chained !== null) return chained;
  // Then the multi-reduction path (single Assign with 2+ reductions
  // over the same tensor, e.g. `acc = sum(x) + max(x) + min(x)`).
  if (tryE2MultiReduction(interp, stmt)) return 1;
  return null;
}

function tryChain(
  interp: Interpreter,
  siblings: Stmt[],
  startIdx: number
): number | null {
  // The head may be a standalone reduction `acc = [acc OP] reduce(<expr>)`
  // with a scalar (or unset) accumulator. Prefer that over the chain
  // path: chain treats `reduce(...)` as opaque, evaluates it eagerly
  // (full-tensor materialization), then rejects the stmt for having a
  // scalar LHS. Otherwise classify as a chain (1+ tensor assigns with
  // an optional trailing reduction sibling).
  const standaloneMatch = matchTrailingReduction(siblings[startIdx]);
  let reductionStandaloneCandidate: TrailingReductionMatch | null = null;
  if (standaloneMatch) {
    const accVal = interp.env.get(standaloneMatch.accName);
    if (accVal === undefined || !isRuntimeTensor(accVal)) {
      reductionStandaloneCandidate = standaloneMatch;
    }
  }
  const cls: ChainClassification | null = reductionStandaloneCandidate
    ? null
    : classifyAssignChain(siblings, startIdx);
  if (!cls && !reductionStandaloneCandidate) return null;

  // Phase 1: evaluate opaque subtrees per chain stmt. We evaluate
  // eagerly for the FULL candidate chain; if we later truncate, the
  // extra bindings just go unused.
  const extraBindings = new Map<string, unknown>();
  if (cls) {
    for (const a of cls.assigns) {
      for (const root of a.opaqueRoots) {
        let val;
        try {
          const raw = interp.evalExpr(root.expr);
          val = Array.isArray(raw) ? raw[0] : raw;
        } catch {
          return null;
        }
        extraBindings.set(root.syntheticName, val);
      }
    }
  }

  // Phase 2: lower each candidate chain stmt incrementally. envTypes
  // is built up from env values + chain LHS registrations as we go.
  // We truncate at the first stmt that fails to lower or produces a
  // non-tensor / undefined-type result — that stmt and everything
  // after it stay on the interpreter path. May still match a trailing
  // reduction below.
  //
  // Chain LHS type is propagated from each assign's lowered RHS so
  // later stmts see correct complex-ness when reading a prior LHS.
  // If any stmt in the accepted prefix produces a complex result, the
  // whole chain goes through the paired-buffer complex emitter.
  const envTypes = new Map<string, JitType>();
  const defaultLhsType: JitType = { kind: "tensor", isComplex: false };
  const specs: ChainAssignSpec[] = [];
  const acceptedAssigns: ChainAssignClassification[] = [];
  let chainIsComplex = false;

  // Look up the type for a name before lowering a stmt that references
  // it. Returns false if the name is unbound or has an unsupported
  // runtime value.
  const ensureType = (name: string): boolean => {
    if (envTypes.has(name)) return true;
    const inp = resolveSupportedInput(interp, name, extraBindings);
    if (!inp) return false;
    envTypes.set(name, inp.jitType);
    return true;
  };

  if (cls) {
    for (const a of cls.assigns) {
      let ok = true;
      for (const id of a.envIdents) {
        if (!ensureType(id)) {
          ok = false;
          break;
        }
      }
      if (!ok) break;
      if (!envTypes.has(a.stmt.name)) {
        envTypes.set(a.stmt.name, defaultLhsType);
      }
      let rhs;
      try {
        rhs = lowerAstToJitExpr(a.emittableExpr, envTypes);
      } catch (e) {
        if (e instanceof E2LowerError) break;
        throw e;
      }
      if (rhs.jitType.kind !== "tensor") break;
      if (rhs.jitType.isComplex) chainIsComplex = true;
      specs.push({ lhsName: a.stmt.name, rhs });
      acceptedAssigns.push(a);
      // Merge: once complex, stays complex so subsequent reads widen
      // correctly in the paired-buffer kernel. `isLogical`, by
      // contrast, reflects just the last write (`a = x > 0; a = a + 1`
      // leaves `a` non-logical); the driver reads it off envTypes when
      // materializing the output RuntimeTensor so logical-indexing
      // callers see `class(mask) == "logical"`.
      const prior = envTypes.get(a.stmt.name);
      const priorComplex =
        prior && prior.kind === "tensor" ? prior.isComplex : false;
      const newLhsType: JitType = {
        kind: "tensor",
        isComplex: priorComplex || rhs.jitType.isComplex,
        ...(rhs.jitType.isLogical ? { isLogical: true } : {}),
      };
      envTypes.set(a.stmt.name, newLhsType);
    }
  }

  // Phase 2b: detect a trailing reduction. The candidate stmt is the
  // first sibling AFTER our accepted chain prefix. Two shapes:
  //
  //   (B) Chain + trailing: the chain produced at least one stmt and
  //       the reduce target is `Ident(lastChainLhs)`. Absorbs the
  //       reduction; the last LHS becomes purely chain-local (no
  //       buffer materialization).
  //
  //   (A) Standalone: chain length 0 — the head stmt itself is the
  //       reduction. Target expr can be any classifiable elemwise
  //       expression.
  //
  // The driver picks ONE: if a chain match is good, we don't also
  // consider the standalone form (the chain prefix would have already
  // consumed those stmts).
  const trailingIdx = startIdx + acceptedAssigns.length;
  let trailing: TrailingReductionMatch | null = null;
  let trailingTargetCls: ClassifyResult | null = null;
  let trailingTargetRhs: import("../../jit/jitTypes.js").JitExpr | null = null;
  let trailingIsStandalone = false;

  if (acceptedAssigns.length > 0 && trailingIdx < siblings.length) {
    const m = matchTrailingReduction(siblings[trailingIdx]);
    if (m) {
      // The target expression must be classifiable AND must reference
      // at least one tensor (otherwise the reduction has nothing to
      // loop over). The expression may reference chain-LHS names
      // (which we've registered in envTypes as tensor) and env tensor
      // names — both resolve correctly inside the reduction emitter.
      const tcls = classifyExpr(m.targetExpr);
      if (isWorthCompiling(tcls.emittableExpr) || tcls.envIdents.size > 0) {
        let opaqueOk = true;
        for (const root of tcls.opaqueRoots) {
          let val;
          try {
            const raw = interp.evalExpr(root.expr);
            val = Array.isArray(raw) ? raw[0] : raw;
          } catch {
            opaqueOk = false;
            break;
          }
          extraBindings.set(root.syntheticName, val);
        }
        if (opaqueOk) {
          let typesOk = true;
          for (const id of tcls.envIdents) {
            if (envTypes.has(id)) continue; // chain LHS or already typed
            if (!ensureType(id)) {
              typesOk = false;
              break;
            }
          }
          if (typesOk) {
            try {
              const rhs = lowerAstToJitExpr(tcls.emittableExpr, envTypes);
              if (rhs.jitType.kind === "tensor" || hasAnyTensorRef(rhs)) {
                trailingTargetCls = tcls;
                trailingTargetRhs = rhs;
                trailing = m;
              }
            } catch {
              // fall through
            }
          }
        }
      }
    }
  } else if (acceptedAssigns.length === 0 && reductionStandaloneCandidate) {
    // (A) standalone: classify the target expression and lower it.
    const tcls = classifyExpr(reductionStandaloneCandidate.targetExpr);
    if (isWorthCompiling(tcls.emittableExpr)) {
      // Evaluate any opaque subtrees inside the target.
      let opaqueOk = true;
      for (const root of tcls.opaqueRoots) {
        let val;
        try {
          const raw = interp.evalExpr(root.expr);
          val = Array.isArray(raw) ? raw[0] : raw;
        } catch {
          opaqueOk = false;
          break;
        }
        extraBindings.set(root.syntheticName, val);
      }
      if (opaqueOk) {
        // Look up types for everything the target expr references.
        let typesOk = true;
        for (const id of tcls.envIdents) {
          if (!ensureType(id)) {
            typesOk = false;
            break;
          }
        }
        if (typesOk) {
          try {
            const rhs = lowerAstToJitExpr(tcls.emittableExpr, envTypes);
            // The reduction target must reference at least one tensor —
            // otherwise the kernel would loop over zero elements.
            if (rhs.jitType.kind === "tensor" || hasAnyTensorRef(rhs)) {
              trailingTargetCls = tcls;
              trailingTargetRhs = rhs;
              trailing = reductionStandaloneCandidate;
              trailingIsStandalone = true;
            }
          } catch {
            // fall through
          }
        }
      }
    }
  }

  if (acceptedAssigns.length === 0 && !trailing) return null;

  // Phase 3: liveness — each unique chain LHS escapes if either:
  //   (a) it's a "scope export" (function output, or `varargout`, or
  //       — for top-level scripts where exports is null — always),
  //   (b) it's referenced anywhere in the scope body outside the
  //       accepted chain stmts (and the trailing reduction stmt, if any).
  // If neither, the LHS is purely chain-local: kept as a per-element
  // stack-local with no buffer materialization.
  //
  // When a trailing reduction is absorbed, the reduce-target LHS (last
  // chain stmt's name, in the chain-trailing case) is forced to be
  // chain-local — the kernel folds it into the accumulator instead of
  // writing it to a buffer.
  const scopeBody = interp._currentScopeBody ?? siblings;
  const scopeExports = interp._currentScopeExports; // null ⇒ "all escape"
  const excludeStmts = new Set<Stmt>();
  for (const a of acceptedAssigns) excludeStmts.add(a.stmt);
  if (trailing) excludeStmts.add(trailing.stmt);
  const allLhsSeen = new Set<string>();
  const escapeLhsNames: string[] = [];
  for (const a of acceptedAssigns) {
    if (allLhsSeen.has(a.stmt.name)) continue;
    allLhsSeen.add(a.stmt.name);
    const escapesByExport =
      scopeExports === null || scopeExports.has(a.stmt.name);
    const escapesByUsage =
      !escapesByExport &&
      isNameReferencedOutsideStmts(scopeBody, excludeStmts, a.stmt.name);
    if (escapesByExport || escapesByUsage) {
      escapeLhsNames.push(a.stmt.name);
    }
  }

  // No observable output ⇒ nothing to materialize. With a trailing
  // reduction the accumulator IS the observable output, so we keep
  // going. Without one, fall through to the interpreter.
  if (escapeLhsNames.length === 0 && !trailing) return null;

  // Phase 4: input-needs analysis. A chain LHS needs `in_<name>` iff
  // any chain stmt reads it BEFORE the LHS has been written in the
  // chain (or on the first-write stmt itself if selfReadsLhs).
  const firstWriteOf = new Map<string, number>();
  for (let k = 0; k < acceptedAssigns.length; k++) {
    const name = acceptedAssigns[k].stmt.name;
    if (!firstWriteOf.has(name)) firstWriteOf.set(name, k);
  }
  const inputLhsNames = new Set<string>();
  const writtenSoFar = new Set<string>();
  for (let k = 0; k < acceptedAssigns.length; k++) {
    const a = acceptedAssigns[k];
    for (const id of a.envIdents) {
      if (!firstWriteOf.has(id)) continue; // not a chain LHS
      if (a.stmt.name === id && a.selfReadsLhs && firstWriteOf.get(id) === k) {
        inputLhsNames.add(id);
      } else if (!writtenSoFar.has(id)) {
        inputLhsNames.add(id);
      }
    }
    writtenSoFar.add(a.stmt.name);
  }

  // Phase 5: gather concrete input descriptors. We need each name's
  // runtime value to marshal the kernel call. Skip chain-LHS names
  // that don't need in_<name>. For standalone reduction, also include
  // the target expression's envIdents.
  const allEnvIdents = new Set<string>();
  for (const a of acceptedAssigns) {
    for (const id of a.envIdents) {
      if (allLhsSeen.has(id) && !inputLhsNames.has(id)) continue;
      allEnvIdents.add(id);
    }
  }
  if (trailingIsStandalone && trailingTargetCls) {
    for (const id of trailingTargetCls.envIdents) {
      // Standalone reduction has no chain LHSs, so no chain-local
      // suppression needed.
      allEnvIdents.add(id);
    }
  }
  // Drop the accumulator name from inputs — its OLD value is read by
  // the JS side (not the kernel) when combining the result. The kernel
  // never references it.
  if (trailing) allEnvIdents.delete(trailing.accName);
  const inputs = gatherInputs(interp, allEnvIdents, extraBindings);
  if (!inputs) return null;

  // Determine output element count from the largest tensor input.
  let n = 0;
  let refTensor: RuntimeTensor | null = null;
  for (const inp of inputs) {
    if (inp.jitType.kind === "tensor") {
      const t = inp.value as RuntimeTensor;
      if (t.data.length > n) {
        n = t.data.length;
        refTensor = t;
      }
    }
  }
  if (n === 0 || !refTensor) return null;
  if (n < e2MinElems()) return null;
  for (const inp of inputs) {
    if (inp.jitType.kind === "tensor") {
      const t = inp.value as RuntimeTensor;
      if (t.data.length !== n) return null;
    }
  }

  // Phase 6: partition inputs into kernel signature order.
  const tensorNames: string[] = [];
  const inputLhsOrdered: string[] = [];
  const scalarNames: string[] = [];
  for (const inp of inputs) {
    if (inputLhsNames.has(inp.name)) {
      if (inp.jitType.kind !== "tensor") return null;
      inputLhsOrdered.push(inp.name);
    } else if (inp.jitType.kind === "tensor") {
      tensorNames.push(inp.name);
    } else {
      scalarNames.push(inp.name);
    }
  }

  // Decide complex vs real dispatch. Complex codegen is required
  // whenever ANY input tensor/scalar is complex OR any RHS produces a
  // complex value: reads of complex Vars use paired buffers and the
  // real emitter doesn't know how. We extend chainIsComplex (set
  // during lowering) with the input-driven signal.
  let needComplex = chainIsComplex;
  if (!needComplex) {
    for (const inp of inputs) {
      if (inp.jitType.kind === "tensor" && inp.jitType.isComplex) {
        needComplex = true;
        break;
      }
      if (inp.jitType.kind === "complex_or_number") {
        needComplex = true;
        break;
      }
    }
  }

  // Complex chains can't absorb trailing reductions: the accumulator
  // is a single double, but `sum(complex_tensor)` produces a complex
  // scalar. Drop the trailing match and let the reduction stmt run
  // through the interpreter on its own.
  if (needComplex && trailing) {
    if (trailingIsStandalone) {
      // No chain prefix was accepted — the whole thing falls through.
      return null;
    }
    trailing = null;
    trailingTargetRhs = null;
    trailingTargetCls = null;
  }

  // Partition for the complex path: complex vs real within each bucket.
  // Only populated when `needComplex`. Chain-LHS complex-ness comes from
  // the envTypes map we built during lowering.
  const complexTensorNames: string[] = [];
  const realTensorNames: string[] = [];
  const complexScalarNames: string[] = [];
  const realScalarNames: string[] = [];
  const complexInputLhsNames: string[] = [];
  const realInputLhsNames: string[] = [];
  const complexEscapeLhsNames: string[] = [];
  const realEscapeLhsNames: string[] = [];
  if (needComplex) {
    for (const name of tensorNames) {
      const inp = inputs.find(i => i.name === name)!;
      if (inp.jitType.kind === "tensor" && inp.jitType.isComplex) {
        complexTensorNames.push(name);
      } else {
        realTensorNames.push(name);
      }
    }
    for (const name of scalarNames) {
      const inp = inputs.find(i => i.name === name)!;
      if (inp.jitType.kind === "complex_or_number") {
        complexScalarNames.push(name);
      } else {
        realScalarNames.push(name);
      }
    }
    for (const name of inputLhsOrdered) {
      const t = envTypes.get(name);
      if (t && t.kind === "tensor" && t.isComplex) {
        complexInputLhsNames.push(name);
      } else {
        realInputLhsNames.push(name);
      }
    }
    for (const name of escapeLhsNames) {
      const t = envTypes.get(name);
      if (t && t.kind === "tensor" && t.isComplex) {
        complexEscapeLhsNames.push(name);
      } else {
        realEscapeLhsNames.push(name);
      }
    }
  }

  // Cache key includes everything that distinguishes one specialization
  // from another: input types, partition lists, chain length, and any
  // trailing reduction info (op + accumulate variant).
  const firstStmt =
    acceptedAssigns.length > 0
      ? acceptedAssigns[0].stmt
      : (trailing!.stmt as Stmt);
  const reductionSigPart = trailing
    ? `red=${trailing.reduceName}|acc=${trailing.hasAccumulate ? "1" : "0"}|op=${
        trailing.accOp ?? "-"
      }`
    : "";
  // `--par` produces a different C kernel (parallel for vs simd-only),
  // so it has to participate in the cache key. Complex kernels ignore
  // `--par` but we still include it to keep the sig format uniform.
  const par = interp.par && isOpenmpAvailable();
  const sig =
    buildSig(inputs, inputLhsOrdered, escapeLhsNames, acceptedAssigns.length) +
    "|" +
    reductionSigPart +
    `|par=${par ? "1" : "0"}` +
    `|complex=${needComplex ? "1" : "0"}`;

  let entry = chainCacheGet(firstStmt, sig);
  if (entry === E2_BAILED) return null;

  if (!entry) {
    let emit;
    let complexInfo: E2ComplexInfo | undefined;
    try {
      if (needComplex) {
        // Build per-stmt specs with `rhsIsComplex` so the emitter
        // knows which assigns write a paired pair vs a single real.
        const complexSpecs: ComplexChainAssignSpec[] = specs.map(s => ({
          lhsName: s.lhsName,
          rhs: s.rhs,
          rhsIsComplex:
            s.rhs.jitType.kind === "tensor" && s.rhs.jitType.isComplex,
        }));
        const complexEmit = emitE2ComplexChainKernel(complexSpecs, {
          complexTensorNames,
          realTensorNames,
          complexScalarNames,
          realScalarNames,
          complexInputLhsNames,
          realInputLhsNames,
          complexEscapeLhsNames,
          realEscapeLhsNames,
        });
        emit = complexEmit;
        complexInfo = {
          complexTensorNames: complexEmit.complexInputTensors,
          realTensorNames: complexEmit.realInputTensors,
          complexInputLhsNames: complexEmit.complexInputLhsNames,
          realInputLhsNames: complexEmit.realInputLhsNames,
          complexScalarNames: complexEmit.complexInputScalars,
          realScalarNames: complexEmit.realInputScalars,
          complexEscapeLhsNames: complexEmit.complexEscapeLhsNames,
          realEscapeLhsNames: complexEmit.realEscapeLhsNames,
        };
      } else if (trailing) {
        // Build the per-element value expression for the reduction.
        // For (B) chain+trailing: it's Var(lastChainLhs) which the
        // emitter resolves to the stack-local. For (A) standalone:
        // it's the lowered classifier output.
        const reduceValueExpr = trailingTargetRhs!;
        emit = emitE2ReductionKernel(
          {
            chain: specs,
            reduceName: trailing.reduceName,
            reduceValueExpr,
            inputs: {
              tensorNames,
              scalarNames,
              inputLhsNames: inputLhsOrdered,
              escapeLhsNames,
            },
          },
          par
        );
      } else {
        emit = emitE2ChainKernel(
          specs,
          {
            tensorNames,
            scalarNames,
            inputLhsNames: inputLhsOrdered,
            escapeLhsNames,
          },
          par
        );
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      if (msg.startsWith("fused scalar emitter:")) {
        chainCacheSet(firstStmt, sig, E2_BAILED);
        return null;
      }
      // Complex emitter rejects unsupported ops by throwing Error with
      // a prefix — bail silently (falls through to interpreter).
      if (msg.startsWith("e2 complex kernel:")) {
        chainCacheSet(firstStmt, sig, E2_BAILED);
        return null;
      }
      throw e;
    }

    let fn;
    try {
      fn = getE2CompileFn()(emit.cSource, emit.koffiSig, emit.kernelName, msg =>
        process.stderr.write(`[e2] ${msg}\n`)
      );
    } catch (e) {
      throw new RuntimeError(
        `--opt e2: kernel compilation failed (chain ${acceptedAssigns.length}, reduction=${
          trailing ? trailing.reduceName : "no"
        }): ${e instanceof Error ? e.message : String(e)}`
      );
    }
    if (!fn) {
      throw new RuntimeError(
        `--opt e2: kernel compilation failed (chain ${acceptedAssigns.length}, reduction=${
          trailing ? trailing.reduceName : "no"
        })`
      );
    }

    const file = firstStmt.span?.file ?? interp.currentFile;
    const line = interp.rt.$line ?? 0;
    interp.onCCompile?.(
      describeKernel(
        file,
        line,
        acceptedAssigns,
        escapeLhsNames,
        inputs,
        n,
        trailing
      ),
      emit.cSource
    );

    const reduction: E2ReductionInfo | undefined = trailing
      ? {
          reduceName: trailing.reduceName,
          accName: trailing.accName,
          hasAccumulate: trailing.hasAccumulate,
          accOp: trailing.accOp,
        }
      : undefined;

    if (complexInfo) {
      entry = {
        fn,
        // Combined lists for diagnostics; actual marshaling uses
        // `complex.*` buckets below.
        tensorNames: [
          ...complexInfo.complexTensorNames,
          ...complexInfo.realTensorNames,
        ],
        inputLhsNames: [
          ...complexInfo.complexInputLhsNames,
          ...complexInfo.realInputLhsNames,
        ],
        scalarNames: [
          ...complexInfo.complexScalarNames,
          ...complexInfo.realScalarNames,
        ],
        escapeLhsNames: [
          ...complexInfo.complexEscapeLhsNames,
          ...complexInfo.realEscapeLhsNames,
        ],
        chainLength: emit.chainLength,
        complex: complexInfo,
      };
    } else {
      const realEmit = emit as {
        inputTensors: string[];
        inputLhsNames: string[];
        inputScalars: string[];
        escapeLhsNames: string[];
        chainLength: number;
      };
      entry = {
        fn,
        tensorNames: realEmit.inputTensors,
        inputLhsNames: realEmit.inputLhsNames,
        scalarNames: realEmit.inputScalars,
        escapeLhsNames: realEmit.escapeLhsNames,
        chainLength: realEmit.chainLength,
        ...(reduction ? { reduction } : {}),
      };
    }
    chainCacheSet(firstStmt, sig, entry);
  }

  // Marshal call.
  const cacheEntry: E2CacheEntry = entry;
  const callArgs: unknown[] = [n];
  const byName = new Map<string, InputDescriptor>();
  for (const inp of inputs) byName.set(inp.name, inp);

  // Output buffers, either single Float64Array (real) or paired
  // {data, imag} (complex). Filled for all escape LHSs (both buckets
  // in the complex case); mapped back to env in the same order below.
  const realOutBufs: Float64Array[] = [];
  const complexOutBufs: { data: Float64Array; imag: Float64Array }[] = [];

  if (cacheEntry.complex) {
    const cx = cacheEntry.complex;
    // Complex tensors: push (data, imag) pairs.
    for (const t of cx.complexTensorNames) {
      const inp = byName.get(t)!;
      const rt = inp.value as RuntimeTensor;
      callArgs.push(rt.data);
      callArgs.push(rt.imag!);
    }
    // Real tensors: single pointer.
    for (const t of cx.realTensorNames) {
      const inp = byName.get(t)!;
      callArgs.push((inp.value as RuntimeTensor).data);
    }
    // Complex input LHSs (read before written in chain).
    for (const t of cx.complexInputLhsNames) {
      const inp = byName.get(t)!;
      const rt = inp.value as RuntimeTensor;
      callArgs.push(rt.data);
      callArgs.push(rt.imag!);
    }
    for (const t of cx.realInputLhsNames) {
      const inp = byName.get(t)!;
      callArgs.push((inp.value as RuntimeTensor).data);
    }
    // Complex scalars: (re, im) pair. Handles both `complex_number`
    // runtime values and plain numbers (im = 0, for a name typed as
    // `complex_or_number` but bound to a real scalar).
    for (const s of cx.complexScalarNames) {
      const inp = byName.get(s)!;
      const v = inp.value as
        | { kind: "complex_number"; re: number; im: number }
        | number
        | boolean;
      if (typeof v === "number") {
        callArgs.push(v);
        callArgs.push(0);
      } else if (typeof v === "boolean") {
        callArgs.push(v ? 1 : 0);
        callArgs.push(0);
      } else {
        callArgs.push(v.re);
        callArgs.push(v.im);
      }
    }
    for (const s of cx.realScalarNames) {
      const inp = byName.get(s)!;
      if (typeof inp.value === "boolean") callArgs.push(inp.value ? 1 : 0);
      else callArgs.push(inp.value as number);
    }
    // Complex escape outputs: paired data/imag Float64Arrays.
    for (const e of cx.complexEscapeLhsNames) {
      const pair = reuseOrAllocComplexOutBufs(interp.env.get(e), n);
      complexOutBufs.push(pair);
      callArgs.push(pair.data);
      callArgs.push(pair.imag);
    }
    for (const e of cx.realEscapeLhsNames) {
      const buf = reuseOrAllocOutBuf(interp.env.get(e), n);
      realOutBufs.push(buf);
      callArgs.push(buf);
    }
  } else {
    // Real path — legacy flat lists.
    for (const t of cacheEntry.tensorNames) {
      const inp = byName.get(t)!;
      callArgs.push((inp.value as RuntimeTensor).data);
    }
    for (const t of cacheEntry.inputLhsNames) {
      const inp = byName.get(t)!;
      callArgs.push((inp.value as RuntimeTensor).data);
    }
    for (const s of cacheEntry.scalarNames) {
      const inp = byName.get(s)!;
      if (typeof inp.value === "boolean") callArgs.push(inp.value ? 1 : 0);
      else callArgs.push(inp.value as number);
    }
    for (const e of cacheEntry.escapeLhsNames) {
      const buf = reuseOrAllocOutBuf(interp.env.get(e), n);
      realOutBufs.push(buf);
      callArgs.push(buf);
    }
  }
  // Reduction output: a 1-element Float64Array. The kernel writes
  // `*out_acc = acc;` into slot [0]; the JS side reads it after.
  // Never present for complex kernels (driver rejects absorption).
  let accBuf: Float64Array | null = null;
  if (cacheEntry.reduction) {
    accBuf = new FloatXArray(1) as Float64Array;
    callArgs.push(accBuf);
  }

  cacheEntry.fn(...callArgs);

  // Bind escape LHSs back to env. Use the reference tensor's shape so
  // column/row orientation is preserved. `_isLogical` is stamped when
  // the chain's last write to that LHS produced a logical result —
  // without it, downstream logical-indexing (`pts(:, mask)`) treats
  // the Float64Array as a double index and fails.
  const isLogicalFor = (name: string): boolean => {
    const t = envTypes.get(name);
    return !!(t && t.kind === "tensor" && t.isLogical);
  };
  if (cacheEntry.complex) {
    const cx = cacheEntry.complex;
    for (let k = 0; k < cx.complexEscapeLhsNames.length; k++) {
      const name = cx.complexEscapeLhsNames[k];
      const pair = complexOutBufs[k];
      const newTensor: RuntimeTensor = {
        kind: "tensor",
        data: pair.data,
        imag: pair.imag,
        shape: refTensor.shape.slice(),
        _rc: 1,
      };
      interp.env.set(name, newTensor);
      interp.ans = newTensor;
    }
    for (let k = 0; k < cx.realEscapeLhsNames.length; k++) {
      const name = cx.realEscapeLhsNames[k];
      const newTensor: RuntimeTensor = {
        kind: "tensor",
        data: realOutBufs[k],
        shape: refTensor.shape.slice(),
        _rc: 1,
        ...(isLogicalFor(name) ? { _isLogical: true } : {}),
      };
      interp.env.set(name, newTensor);
      interp.ans = newTensor;
    }
  } else {
    for (let k = 0; k < cacheEntry.escapeLhsNames.length; k++) {
      const name = cacheEntry.escapeLhsNames[k];
      const newTensor: RuntimeTensor = {
        kind: "tensor",
        data: realOutBufs[k],
        shape: refTensor.shape.slice(),
        _rc: 1,
        ...(isLogicalFor(name) ? { _isLogical: true } : {}),
      };
      interp.env.set(name, newTensor);
      interp.ans = newTensor;
    }
  }

  // Trailing reduction: combine kernel's scalar output with env value.
  if (cacheEntry.reduction && accBuf) {
    let red = accBuf[0];
    if (cacheEntry.reduction.reduceName === "mean") {
      red /= n;
    }
    let finalVal: number;
    if (cacheEntry.reduction.hasAccumulate) {
      const prev = interp.env.get(cacheEntry.reduction.accName);
      const prevNum = typeof prev === "number" ? prev : 0;
      switch (cacheEntry.reduction.accOp) {
        case BinaryOperation.Add:
          finalVal = prevNum + red;
          break;
        case BinaryOperation.Sub:
          finalVal = prevNum - red;
          break;
        case BinaryOperation.Mul:
        case BinaryOperation.ElemMul:
          finalVal = prevNum * red;
          break;
        default:
          finalVal = red;
      }
    } else {
      finalVal = red;
    }
    interp.env.set(cacheEntry.reduction.accName, finalVal);
    interp.ans = finalVal;
  }

  // Total sibling stmts consumed: chain assigns + trailing reduction
  // (if absorbed). Returned to the caller as the executor's `consumed`.
  const consumed = acceptedAssigns.length + (trailing ? 1 : 0);
  return consumed;
}
