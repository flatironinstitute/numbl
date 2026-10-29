/**
 * The kernel pass: an IR-to-IR rewrite, run by `compileWith` just before
 * emit for both backends, that turns element-wise tensor assignments into
 * explicit `ElemwiseKernel` nodes (see `ir.ts`), which each backend renders
 * as one loop with no intermediate tensors.
 *
 * Lowering puts every tensor sub-expression in its own ANF temp, so a
 * chain like `y = a.*b + c` arrives as `_t1 = a .* b; y = _t1 + c`. The
 * pass first folds single-use producers into their consumers (`y =
 * (a .* b) + c`), then replaces each fusable RHS with a kernel. A kernel
 * covers literals, variables, and element-wise `Binary` / `Unary` /
 * `Call` nodes over operands of one shape (or scalars), optionally under
 * a full `sum` / `prod` / `mean`.
 *
 * Which builtins a kernel may contain is the target's `FusionPolicy`: C
 * fuses every element-wise builtin; JS fuses only those whose per-element
 * result is exactly specified by IEEE arithmetic (`JS_FUSION`), since its
 * per-op helpers hand large tensors to the host's native kernels and V8's
 * transcendental functions need not match them bit for bit.
 *
 * Inlining gates (all must hold to fold producer P into consumer C):
 *
 *   1. P is an element-wise kernel assignment (`isElemwiseAssign`).
 *   2. P's variable is read exactly once in the whole function (nested
 *      bodies included) and is not an output. Counting function-wide is
 *      what keeps a def inside a loop body that is also read after it.
 *   3. C is the first statement after P in the same body that reads P's
 *      variable, and it is an Assign. Statements in between neither write
 *      P's variable or any variable P reads, nor are control flow. C may
 *      itself write such a variable: an Assign evaluates its RHS first.
 *   4. The read in C is an operand of a Binary / Unary / Call, and C is a
 *      kernel assignment after substitution (an ownership rule: a
 *      non-kernel consumer would get a nested owned operand nothing frees).
 *
 * Use counts are stable under substitution, so one forward sweep per body
 * collapses chains of any depth. The pass never mutates its input: the
 * `Lowerer`'s spec cache is shared by both backends.
 */

import type {
  Assign,
  ElemwiseKernel,
  IRExpr,
  IRFunc,
  IRProgram,
  IRStmt,
  KernelReduce,
  Var,
} from "./ir.js";
import {
  DIM_ONE,
  isDoubleOrLogical,
  isMultiElement,
  isNumeric,
  type NumericType,
  type Type,
} from "./types.js";
import { forEachSubExpr, forEachTopLevelExpr } from "./walk.js";
import { getBuiltin } from "../builtins/index.js";
import { isElementwiseCall } from "../builtins/registry.js";

/** What a backend can put in a kernel. */
export interface FusionPolicy {
  /** The builtin may appear in a kernel body. */
  op(name: string): boolean;
  /** A type a kernel may read, compute or produce. */
  type(t: Type): boolean;
}

/** C: every element-wise builtin (each has a scalar `emitC`). */
export const C_FUSION: FusionPolicy = {
  op: () => true,
  type: () => true,
};

/** Builtins whose JS scalar form computes the same bits as the JS tensor
 *  helpers (and the host kernels those use for large tensors). */
const JS_FUSABLE_OPS = new Set([
  "plus",
  "minus",
  "times",
  "rdivide",
  "mtimes",
  "mrdivide",
  "uminus",
  "eq",
  "ne",
  "lt",
  "le",
  "gt",
  "ge",
  "abs",
  "floor",
  "ceil",
  "fix",
  "sqrt",
  "sum",
  "prod",
  "mean",
]);

/** JS: exact operations on real double or logical values. */
export const JS_FUSION: FusionPolicy = {
  op: name => JS_FUSABLE_OPS.has(name),
  type: t => isNumeric(t) && !t.isComplex && isDoubleOrLogical(t),
};

const REDUCERS = new Set<string>(["sum", "prod", "mean"]);

/** Returns a new program with every function's kernels built. */
export function kernelPass(prog: IRProgram, policy: FusionPolicy): IRProgram {
  const functions = new Map<string, IRFunc>();
  for (const [k, fn] of prog.functions) {
    functions.set(k, kernelizeFunc(fn, policy));
  }
  return { functions };
}

/** Returns `fn` itself when nothing changed. */
export function kernelizeFunc(fn: IRFunc, policy: FusionPolicy): IRFunc {
  const counts = computeUseCounts(fn.body, fn.cOutputs);
  const body = kernelizeBody(fn.body, counts, policy);
  return body === fn.body ? fn : { ...fn, body };
}

// ── Fusability ───────────────────────────────────────────────────────────

/** An Assign whose RHS becomes an element-wise kernel: a real
 *  multi-element result computed slot by slot from operands of its own
 *  shape. */
export function isElemwiseAssign(s: Assign, policy: FusionPolicy): boolean {
  return (
    isNumeric(s.ty) &&
    !s.ty.isComplex &&
    isMultiElement(s.ty) &&
    policy.type(s.ty) &&
    isFusableTree(s.expr, s.ty, policy)
  );
}

/** The reduced operand of an Assign that becomes a reducing kernel:
 *  `sum` / `prod` / `mean` of a non-trivial element-wise expression,
 *  reduced over all of its elements (the result is a real scalar). */
function reduceOperand(s: Assign, policy: FusionPolicy): IRExpr | null {
  const e = s.expr;
  if (e.kind !== "Call" || e.args.length !== 1) return null;
  if (!REDUCERS.has(e.name) || getBuiltin(e.name) === undefined) return null;
  if (!policy.op(e.name)) return null;
  if (!isNumeric(s.ty) || s.ty.isComplex || isMultiElement(s.ty)) return null;
  const x = e.args[0];
  if (x.kind !== "Binary" && x.kind !== "Unary" && x.kind !== "Call") {
    return null;
  }
  if (!isNumeric(x.ty) || x.ty.isComplex || !isMultiElement(x.ty)) {
    return null;
  }
  if (!policy.type(x.ty)) return null;
  return isFusableTree(x, x.ty, policy) ? x : null;
}

/** Any kernel assignment, element-wise or reducing. */
export function isKernelAssign(s: Assign, policy: FusionPolicy): boolean {
  return isElemwiseAssign(s, policy) || reduceOperand(s, policy) !== null;
}

function isFusableTree(
  e: IRExpr,
  target: NumericType,
  policy: FusionPolicy
): boolean {
  return (
    isPureElementwise(e, policy) &&
    everyTensorVarMatchesShape(e, target) &&
    !touchesComplexTensor(e)
  );
}

/** Only NumLit / Var / Binary / Unary / element-wise Call nodes the policy
 *  allows, each of a type it allows. */
function isPureElementwise(e: IRExpr, policy: FusionPolicy): boolean {
  if (!policy.type(e.ty)) return false;
  switch (e.kind) {
    case "NumLit":
    case "Var":
      return true;
    case "Binary": {
      // `Binary` covers matrix ops too; gate on the builtin's flag.
      const b = getBuiltin(e.builtin);
      if (!b || !b.elementwise || !policy.op(e.builtin)) return false;
      // `mtimes` / `mrdivide` are element-wise only with a scalar side.
      if (e.builtin === "mtimes" || e.builtin === "mrdivide") {
        if (isMultiElementNumeric(e.left) && isMultiElementNumeric(e.right)) {
          return false;
        }
      }
      return (
        isPureElementwise(e.left, policy) && isPureElementwise(e.right, policy)
      );
    }
    case "Unary": {
      const b = getBuiltin(e.builtin);
      if (!b || !b.elementwise || !policy.op(e.builtin)) return false;
      return isPureElementwise(e.operand, policy);
    }
    case "Call": {
      const b = getBuiltin(e.name);
      if (
        !b ||
        !policy.op(e.name) ||
        !isElementwiseCall(
          b,
          e.args.map(a => a.ty)
        )
      ) {
        return false;
      }
      // A kernel needs a scalar result per slot. A tensor-producing call
      // made from scalars alone (the shape-constructor form of
      // `nan(2,3)`) flags `elementwise` for its scalar form but returns a
      // fresh tensor; a tensor result from tensor arguments of its own
      // shape (`sqrt(x)`, `max(x, 0)`) is pointwise.
      if (isMultiElementNumeric(e) && !isPointwiseTensorCall(e)) return false;
      return e.args.every(a => isPureElementwise(a, policy));
    }
    default:
      return false;
  }
}

function isMultiElementNumeric(e: IRExpr): boolean {
  return isNumeric(e.ty) && isMultiElement(e.ty);
}

/** A tensor-valued call whose tensor arguments all have its own static
 *  shape, and which has at least one. */
function isPointwiseTensorCall(e: IRExpr & { kind: "Call" }): boolean {
  const out = (e.ty as NumericType).shape;
  if (out === undefined) return false;
  let any = false;
  for (const a of e.args) {
    if (!isNumeric(a.ty) || !isMultiElement(a.ty)) continue;
    const s = a.ty.shape;
    if (
      s === undefined ||
      s.length !== out.length ||
      s.some((d, i) => d !== out[i])
    ) {
      return false;
    }
    any = true;
  }
  return any;
}

/** The kernel addresses tensors by real lane only. */
function touchesComplexTensor(e: IRExpr): boolean {
  let touched = false;
  forEachSubExpr(e, sub => {
    if (isNumeric(sub.ty) && isMultiElement(sub.ty) && sub.ty.isComplex) {
      touched = true;
    }
  });
  return touched;
}

/** Every multi-element Var has the target's static dim pattern (no
 *  broadcasting inside a kernel). */
function everyTensorVarMatchesShape(e: IRExpr, target: NumericType): boolean {
  let ok = true;
  forEachSubExpr(e, sub => {
    if (
      sub.kind === "Var" &&
      isNumeric(sub.ty) &&
      isMultiElement(sub.ty) &&
      !sameStaticShape(sub.ty, target)
    ) {
      ok = false;
    }
  });
  return ok;
}

function sameStaticShape(a: NumericType, b: NumericType): boolean {
  if (a.dims.length !== b.dims.length) return false;
  for (let i = 0; i < a.dims.length; i++) {
    const da = a.dims[i];
    const db = b.dims[i];
    if (da.kind !== db.kind) return false;
    if (da.kind === "exact" && db.kind === "exact" && da.value !== db.value) {
      return false;
    }
  }
  return true;
}

// ── Kernel construction ──────────────────────────────────────────────────

function buildKernel(s: Assign, policy: FusionPolicy): ElemwiseKernel {
  const operand = isElemwiseAssign(s, policy)
    ? s.expr
    : reduceOperand(s, policy);
  if (operand === null) {
    throw new Error(`kernel pass internal: '${s.name}' is not a kernel`);
  }
  const reduce: KernelReduce | undefined =
    operand === s.expr
      ? undefined
      : {
          op: (s.expr as IRExpr & { kind: "Call" }).name as KernelReduce["op"],
        };
  const inputs: Var[] = [];
  const index = new Map<string, number>();
  let anyDynamic = false;
  forEachSubExpr(operand, sub => {
    if (sub.kind !== "Var" || !isMultiElementNumeric(sub)) return;
    if ((sub.ty as NumericType).dims.some(d => d.kind !== "exact")) {
      anyDynamic = true;
    }
    if (!index.has(sub.cName)) {
      index.set(sub.cName, inputs.length);
      inputs.push(sub);
    }
  });
  if (inputs.length === 0) {
    throw new Error(`kernel pass internal: '${s.name}' has no tensor operand`);
  }
  return {
    kind: "ElemwiseKernel",
    inputs,
    body: scalarBody(operand, index),
    // Two or more distinct operands whose extents are not all exact may
    // differ at run time (`ones(1,n) + ones(1,m)`); one operand is its own
    // shape source, and exact dims are already proven equal.
    guard: inputs.length >= 2 && anyDynamic,
    ...(reduce !== undefined ? { reduce } : {}),
    tree: s.expr,
    ty: s.ty,
    span: s.span,
  };
}

/** `e` with every multi-element Var read replaced by a `KernelElem` and
 *  every type made scalar (the per-element view the builtins' scalar
 *  emitters expect). */
function scalarBody(e: IRExpr, index: ReadonlyMap<string, number>): IRExpr {
  switch (e.kind) {
    case "Var":
      if (isMultiElementNumeric(e)) {
        return {
          kind: "KernelElem",
          input: index.get(e.cName)!,
          ty: scalarVersionOf(e.ty),
          span: e.span,
        };
      }
      return e;
    case "Binary":
      return {
        ...e,
        left: scalarBody(e.left, index),
        right: scalarBody(e.right, index),
        ty: scalarVersionOf(e.ty),
      };
    case "Unary":
      return {
        ...e,
        operand: scalarBody(e.operand, index),
        ty: scalarVersionOf(e.ty),
      };
    case "Call":
      return {
        ...e,
        args: e.args.map(a => scalarBody(a, index)),
        ty: scalarVersionOf(e.ty),
      };
    default:
      return e;
  }
}

/** The scalar variant of a multi-element numeric type (same element type,
 *  complexity and sign). Other types pass through unchanged. */
function scalarVersionOf(t: Type): Type {
  if (!isNumeric(t) || !isMultiElement(t)) return t;
  return {
    kind: "Numeric",
    elem: t.elem,
    isComplex: t.isComplex,
    dims: [DIM_ONE, DIM_ONE],
    shape: [1, 1],
    sign: t.sign,
  };
}

// ── Inlining ─────────────────────────────────────────────────────────────

/** Kernelize one body, children first. Returns `stmts` itself when
 *  nothing changed. */
function kernelizeBody(
  stmts: IRStmt[],
  counts: ReadonlyMap<string, number>,
  policy: FusionPolicy
): IRStmt[] {
  let changed = false;
  const cur: IRStmt[] = stmts.map(s => {
    const r = kernelizeNested(s, counts, policy);
    if (r !== s) changed = true;
    return r;
  });

  const removed = new Array<boolean>(cur.length).fill(false);
  for (let i = 0; i < cur.length; i++) {
    const prod = cur[i];
    if (prod.kind !== "Assign") continue;
    if (counts.get(prod.cName) !== 1) continue;
    if (!isElemwiseAssign(prod, policy)) continue;

    const prodFreeVars = collectFreeVarCNames(prod.expr);
    for (let j = i + 1; j < cur.length; j++) {
      const s = cur[j];
      if (isControlFlow(s)) break;
      if (stmtReadsVar(s, prod.cName)) {
        if (s.kind === "Assign" && onlyInOperandPosition(s.expr, prod.cName)) {
          const consumer: Assign = {
            ...s,
            expr: substituteVar(s.expr, prod.cName, prod.expr),
          };
          if (isKernelAssign(consumer, policy)) {
            cur[j] = consumer;
            removed[i] = true;
            changed = true;
          }
        }
        break;
      }
      if (stmtWrites(s, prod.cName, prodFreeVars)) break;
    }
  }

  const out: IRStmt[] = [];
  for (let i = 0; i < cur.length; i++) {
    if (removed[i]) continue;
    const s = cur[i];
    if (s.kind === "Assign" && isKernelAssign(s, policy)) {
      out.push({ ...s, expr: buildKernel(s, policy) });
      changed = true;
    } else {
      out.push(s);
    }
  }
  return changed ? out : stmts;
}

/** Recurse into an If / While / For body. Returns `s` itself when
 *  nothing changed. */
function kernelizeNested(
  s: IRStmt,
  counts: ReadonlyMap<string, number>,
  policy: FusionPolicy
): IRStmt {
  if (s.kind === "If") {
    const thenBody = kernelizeBody(s.thenBody, counts, policy);
    const elseBody = kernelizeBody(s.elseBody, counts, policy);
    if (thenBody === s.thenBody && elseBody === s.elseBody) return s;
    return { ...s, thenBody, elseBody };
  }
  if (s.kind === "While" || s.kind === "For") {
    const body = kernelizeBody(s.body, counts, policy);
    return body === s.body ? s : { ...s, body };
  }
  return s;
}

/** Var reads per cName across the whole function body, nested bodies
 *  included. Outputs get a +1 bump so they never reach count 1. */
function computeUseCounts(
  stmts: ReadonlyArray<IRStmt>,
  outputs: ReadonlyArray<string>
): Map<string, number> {
  const counts = new Map<string, number>();
  const bump = (cName: string) =>
    counts.set(cName, (counts.get(cName) ?? 0) + 1);
  for (const cName of outputs) bump(cName);
  const visit = (s: IRStmt): void => {
    forEachTopLevelExpr(s, e =>
      forEachSubExpr(e, sub => {
        if (sub.kind === "Var") bump(sub.cName);
      })
    );
    if (s.kind === "If") {
      s.thenBody.forEach(visit);
      s.elseBody.forEach(visit);
    } else if (s.kind === "While" || s.kind === "For") {
      s.body.forEach(visit);
    }
  };
  stmts.forEach(visit);
  return counts;
}

function isControlFlow(s: IRStmt): boolean {
  return (
    s.kind === "If" ||
    s.kind === "While" ||
    s.kind === "For" ||
    s.kind === "Break" ||
    s.kind === "Continue" ||
    s.kind === "ReturnFromFunction"
  );
}

function collectFreeVarCNames(e: IRExpr): Set<string> {
  const out = new Set<string>();
  forEachSubExpr(e, sub => {
    if (sub.kind === "Var") out.add(sub.cName);
  });
  return out;
}

/** True iff `s` writes `cName` or any name in `freeVars`: Assign LHS,
 *  base of an indexed / member / cell store, or a bound output of a
 *  MultiAssignCall. */
function stmtWrites(
  s: IRStmt,
  cName: string,
  freeVars: ReadonlySet<string>
): boolean {
  const hits = (n: string) => n === cName || freeVars.has(n);
  switch (s.kind) {
    case "Assign":
      return hits(s.cName);
    case "IndexStore":
    case "IndexSliceStore":
    case "MemberStore":
    case "CellIndexStore":
      return hits(s.base.cName);
    case "MultiAssignCall":
      return s.outputs.some(o => o.binding !== null && hits(o.binding.cName));
    default:
      return false;
  }
}

/** True iff `s` reads `cName` in its top-level expressions (store bases
 *  included). */
function stmtReadsVar(s: IRStmt, cName: string): boolean {
  let found = false;
  forEachTopLevelExpr(s, e => {
    forEachSubExpr(e, sub => {
      if (sub.kind === "Var" && sub.cName === cName) found = true;
    });
  });
  return found;
}

/** True iff every read of `cName` in `e` is reached through Binary /
 *  Unary / Call operands only, where `substituteVar` replaces it. */
function onlyInOperandPosition(e: IRExpr, cName: string): boolean {
  switch (e.kind) {
    case "Var":
      return true;
    case "Binary":
      return (
        onlyInOperandPosition(e.left, cName) &&
        onlyInOperandPosition(e.right, cName)
      );
    case "Unary":
      return onlyInOperandPosition(e.operand, cName);
    case "Call":
      return e.args.every(a => onlyInOperandPosition(a, cName));
    default: {
      let found = false;
      forEachSubExpr(e, sub => {
        if (sub.kind === "Var" && sub.cName === cName) found = true;
      });
      return !found;
    }
  }
}

/** Replace every operand-position `Var(target)` in `e` with
 *  `replacement`, preserving identity for unchanged subtrees. */
function substituteVar(e: IRExpr, target: string, replacement: IRExpr): IRExpr {
  switch (e.kind) {
    case "Var":
      return e.cName === target ? replacement : e;
    case "Binary": {
      const left = substituteVar(e.left, target, replacement);
      const right = substituteVar(e.right, target, replacement);
      if (left === e.left && right === e.right) return e;
      return { ...e, left, right };
    }
    case "Unary": {
      const operand = substituteVar(e.operand, target, replacement);
      return operand === e.operand ? e : { ...e, operand };
    }
    case "Call": {
      let changed = false;
      const args = e.args.map(a => {
        const sub = substituteVar(a, target, replacement);
        if (sub !== a) changed = true;
        return sub;
      });
      return changed ? { ...e, args } : e;
    }
    default:
      return e;
  }
}
