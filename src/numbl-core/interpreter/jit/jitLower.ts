/**
 * AST -> JIT IR lowering with type propagation.
 *
 * Returns null if any unsupported construct is encountered,
 * causing the entire function to fall back to interpretation.
 */

import type { Expr, Stmt } from "../../parser/types.js";
import { BinaryOperation, UnaryOperation } from "../../parser/types.js";
import type { FunctionDef } from "../types.js";
import type { Interpreter } from "../interpreter.js";
import type { CallSite } from "../../runtime/runtimeHelpers.js";
import { resolveFunction } from "../../functionResolve.js";
import type { ItemType } from "../../lowering/itemTypes.js";
import {
  type JitType,
  type JitExpr,
  type JitStmt,
  type SignCategory,
  unifyJitTypes,
  isScalarType,
  isNumericScalarType,
  isTensorType,
  isComplexType,
  isArithmeticType,
  jitTypeKey,
  computeJitFnName,
  signFromNumber,
  flipSign,
} from "./jitTypes.js";
import { generateJS } from "./jitCodegen.js";
import { getIBuiltin } from "../builtins/index.js";
import { offsetToLineFast, buildLineTable } from "../../runtime/error.js";

// ── Known constants ─────────────────────────────────────────────────────

const KNOWN_CONSTANTS: Record<string, number> = {
  pi: Math.PI,
  inf: Infinity,
  Inf: Infinity,
  nan: NaN,
  NaN: NaN,
  eps: 2.220446049250313e-16,
  true: 1,
  false: 0,
};

// ── Type Environment ────────────────────────────────────────────────────

type TypeEnv = Map<string, JitType>;

function cloneEnv(env: TypeEnv): TypeEnv {
  return new Map(env);
}

/** Merge two type environments at a join point. Returns null if any type becomes unknown. */
function mergeEnvs(a: TypeEnv, b: TypeEnv): TypeEnv | null {
  const result = cloneEnv(a);
  for (const [name, typeB] of b) {
    const typeA = result.get(name);
    if (typeA) {
      const unified = unifyJitTypes(typeA, typeB);
      if (unified.kind === "unknown") return null;
      result.set(name, unified);
    } else {
      result.set(name, typeB);
    }
  }
  return result;
}

/** Check if two type environments are identical (by JSON comparison). */
function envsEqual(a: TypeEnv, b: TypeEnv): boolean {
  if (a.size !== b.size) return false;
  for (const [name, type] of a) {
    const other = b.get(name);
    if (!other || JSON.stringify(type) !== JSON.stringify(other)) return false;
  }
  return true;
}

// ── Sign algebra ────────────────────────────────────────────────────────

function addSigns(a: SignCategory, b: SignCategory): SignCategory | undefined {
  if (a === "positive" && b === "positive") return "positive";
  if (a === "negative" && b === "negative") return "negative";
  if (
    (a === "nonneg" || a === "positive") &&
    (b === "nonneg" || b === "positive")
  )
    return "nonneg";
  if (
    (a === "nonpositive" || a === "negative") &&
    (b === "nonpositive" || b === "negative")
  )
    return "nonpositive";
  return undefined;
}

function mulSigns(a: SignCategory, b: SignCategory): SignCategory | undefined {
  // positive * positive -> positive, negative * negative -> positive
  const aPos = a === "positive" || a === "nonneg";
  const aNeg = a === "negative" || a === "nonpositive";
  const bPos = b === "positive" || b === "nonneg";
  const bNeg = b === "negative" || b === "nonpositive";
  const aStrict = a === "positive" || a === "negative";
  const bStrict = b === "positive" || b === "negative";

  if ((aPos && bPos) || (aNeg && bNeg)) {
    return aStrict && bStrict ? "positive" : "nonneg";
  }
  if ((aPos && bNeg) || (aNeg && bPos)) {
    return aStrict && bStrict ? "negative" : "nonpositive";
  }
  return undefined;
}

function combineSigns(
  a: SignCategory | undefined,
  b: SignCategory | undefined,
  op: BinaryOperation
): SignCategory | undefined {
  if (!a || !b) return undefined;
  switch (op) {
    case BinaryOperation.Add:
      return addSigns(a, b);
    case BinaryOperation.Sub:
      return addSigns(a, flipSign(b)!);
    case BinaryOperation.Mul:
    case BinaryOperation.ElemMul:
      return mulSigns(a, b);
    default:
      return undefined;
  }
}

// ── Type propagation for binary operations ──────────────────────────────

function binaryResultType(
  op: BinaryOperation,
  left: JitType,
  right: JitType
): JitType | null {
  // Reject non-arithmetic types (class_instance, struct, string, char, etc.)
  if (!isArithmeticType(left) || !isArithmeticType(right)) return null;

  // Comparisons always produce logical for scalars
  if (
    op === BinaryOperation.Equal ||
    op === BinaryOperation.NotEqual ||
    op === BinaryOperation.Less ||
    op === BinaryOperation.LessEqual ||
    op === BinaryOperation.Greater ||
    op === BinaryOperation.GreaterEqual
  ) {
    if (isScalarType(left) && isScalarType(right)) return { kind: "boolean" };
    // Tensor comparison: real tensors only (MATLAB errors on complex comparisons)
    const anyTensor = isTensorType(left) || isTensorType(right);
    const anyComplex = isComplexType(left) || isComplexType(right);
    if (anyTensor && !anyComplex) {
      const lt = isTensorType(left)
        ? (left as Extract<JitType, { kind: "tensor" }>)
        : undefined;
      const rt = isTensorType(right)
        ? (right as Extract<JitType, { kind: "tensor" }>)
        : undefined;
      const shape = lt?.shape ?? rt?.shape;
      const ndim = shape ? undefined : (lt?.ndim ?? rt?.ndim);
      return {
        kind: "tensor",
        isComplex: false,
        isLogical: true,
        ...(shape ? { shape } : {}),
        ...(ndim !== undefined ? { ndim } : {}),
      };
    }
    return null;
  }

  // Logical operators: scalar only
  if (op === BinaryOperation.AndAnd || op === BinaryOperation.OrOr) {
    if (isScalarType(left) && isScalarType(right)) return { kind: "boolean" };
    return null;
  }

  // Only element-wise arithmetic ops
  switch (op) {
    case BinaryOperation.Add:
    case BinaryOperation.Sub:
    case BinaryOperation.ElemMul:
    case BinaryOperation.ElemDiv:
      break;
    case BinaryOperation.Mul:
      // Matrix multiply (both tensors) not supported
      if (isTensorType(left) && isTensorType(right)) return null;
      break;
    case BinaryOperation.Div:
      // tensor / tensor not supported (use ./ instead)
      if (isTensorType(left) && isTensorType(right)) return null;
      break;
    case BinaryOperation.ElemPow:
      break;
    case BinaryOperation.Pow:
      // Matrix power (both tensors) not supported
      if (isTensorType(left) && isTensorType(right)) return null;
      break;
    default:
      return null;
  }

  // Coerce logical to number for arithmetic
  const effLeft: JitType =
    left.kind === "boolean" ? { kind: "number", sign: "nonneg" } : left;
  const effRight: JitType =
    right.kind === "boolean" ? { kind: "number", sign: "nonneg" } : right;

  // Determine result type based on operand types
  const anyComplex = isComplexType(effLeft) || isComplexType(effRight);
  const anyTensor = isTensorType(effLeft) || isTensorType(effRight);

  if (anyTensor) {
    const lt = isTensorType(effLeft)
      ? (effLeft as Extract<JitType, { kind: "tensor" }>)
      : undefined;
    const rt = isTensorType(effRight)
      ? (effRight as Extract<JitType, { kind: "tensor" }>)
      : undefined;
    const shape = lt?.shape ?? rt?.shape;
    const ndim = shape ? undefined : (lt?.ndim ?? rt?.ndim);
    const isComplex =
      anyComplex || (lt?.isComplex ?? false) || (rt?.isComplex ?? false);
    return {
      kind: "tensor",
      isComplex,
      ...(shape ? { shape } : {}),
      ...(ndim !== undefined ? { ndim } : {}),
    };
  }

  if (anyComplex) {
    // Complex scalar power not yet supported in codegen
    if (op === BinaryOperation.Pow || op === BinaryOperation.ElemPow)
      return null;
    return { kind: "complex_or_number" };
  }

  // Both are numbers (or coerced from logical)
  if (effLeft.kind === "number" && effRight.kind === "number") {
    const sign = combineSigns(effLeft.sign, effRight.sign, op);
    return { kind: "number", ...(sign ? { sign } : {}) };
  }

  return null;
}

function unaryResultType(op: UnaryOperation, operand: JitType): JitType | null {
  switch (op) {
    case UnaryOperation.Plus:
      return operand;
    case UnaryOperation.Minus:
      if (operand.kind === "number") {
        const sign = flipSign(operand.sign);
        return { kind: "number", ...(sign ? { sign } : {}) };
      }
      if (operand.kind === "boolean")
        return { kind: "number", sign: "nonpositive" };
      if (operand.kind === "complex_or_number")
        return { kind: "complex_or_number" };
      if (operand.kind === "tensor")
        return {
          kind: "tensor",
          isComplex: operand.isComplex,
          shape: operand.shape,
          ndim: operand.ndim,
        };
      return null;
    case UnaryOperation.Not:
      if (isNumericScalarType(operand)) return { kind: "boolean" };
      return null;
    default:
      return null; // Transpose not supported
  }
}

// ── Lowering entry point ────────────────────────────────────────────────

export interface LoweringResult {
  body: JitStmt[];
  outputNames: string[];
  localVars: Set<string>;
  hasTensorOps: boolean;
  /** Generated JS code for called user functions: jitName → code */
  generatedFns: Map<string, string>;
  /** Type of the first output variable after lowering */
  outputType: JitType | null;
}

export function lowerFunction(
  fn: FunctionDef,
  argTypes: JitType[],
  nargout: number,
  interp?: Interpreter,
  generatedFns?: Map<string, string>,
  loweringInProgress?: Set<string>
): LoweringResult | null {
  if (argTypes.length !== fn.params.length) return null;

  const env: TypeEnv = new Map();
  const localVars = new Set<string>();

  // Initialize parameters
  for (let i = 0; i < fn.params.length; i++) {
    env.set(fn.params[i], argTypes[i]);
  }

  // Initialize output variables (default to number 0)
  const outputNames = fn.outputs.slice(0, nargout || 1);
  for (const name of outputNames) {
    if (!env.has(name)) {
      env.set(name, { kind: "number", exact: 0, sign: "nonneg" }); // default 0
      localVars.add(name);
    }
  }

  const sharedGeneratedFns = generatedFns ?? new Map<string, string>();
  const sharedInProgress = loweringInProgress ?? new Set<string>();

  // Build line table for offset→line lookup from file sources
  let lineTable: number[] | undefined;
  if (interp && fn.body.length > 0 && fn.body[0].span) {
    const file = fn.body[0].span.file;
    lineTable = interp.lineTableCache.get(file);
    if (!lineTable) {
      const src = interp.fileSources.get(file) ?? "";
      lineTable = buildLineTable(src);
      interp.lineTableCache.set(file, lineTable);
    }
  }

  const ctx: LowerCtx = {
    env,
    localVars,
    params: new Set(fn.params),
    assignedVars: new Set(fn.params),
    sliceAliases: new Map(),
    interp,
    generatedFns: sharedGeneratedFns,
    loweringInProgress: sharedInProgress,
    lineTable,
  };
  const body = lowerStmts(ctx, fn.body);
  if (!body) return null;

  // Bail if any required output variable was never assigned in the body.
  // The interpreter throws a RuntimeError for this case; the JIT would
  // silently return undefined/0 without this check.
  for (const name of outputNames) {
    if (!ctx.params.has(name) && !ctx.assignedVars.has(name)) return null;
  }

  const outputType =
    outputNames.length > 0 ? (ctx.env.get(outputNames[0]) ?? null) : null;

  return {
    body,
    outputNames,
    localVars: ctx.localVars,
    hasTensorOps: ctx._hasTensorOps ?? false,
    generatedFns: sharedGeneratedFns,
    outputType,
  };
}

// ── Internal lowering context ───────────────────────────────────────────

/**
 * A "slice alias" records that a MATLAB local was bound to a colon-slice of
 * a real tensor (e.g. `pt = pts(:, i)`). Rather than materializing the slice
 * as a RuntimeTensor per iteration, we remember the base tensor and a
 * per-dim "template" of indices — scalar expressions captured at bind time
 * for the non-colon dims, and `"colon"` placeholders for the colon dims.
 * Subsequent reads like `pt(k)` substitute `k` into the colon positions and
 * emit a direct scalar read on the base tensor, which compiles cleanly
 * through the existing hoisted `idx{1,2,3}r_h` fast path.
 */
interface SliceAlias {
  baseName: string;
  baseType: JitType;
  /**
   * One entry per index of the original bind expression, in source order.
   * A `"colon"` slot expects to be filled by the read-site's colon indices.
   * An `"expr"` slot carries a JitExpr that will be substituted as-is.
   */
  template: ({ kind: "colon" } | { kind: "expr"; expr: JitExpr })[];
  /** Sizes of the slice's colon dimensions, in source order. */
  sliceShape: number[];
  /** Indices into `template` where colon slots live, in source order. */
  colonPositions: number[];
}

interface LowerCtx {
  env: TypeEnv;
  localVars: Set<string>;
  params: Set<string>;
  /** Variables that are actually assigned in the function body. */
  assignedVars: Set<string>;
  /**
   * Map from a MATLAB local name to its slice alias, if any. A name is
   * present here iff the most recent assignment to it was a whole-tensor
   * colon slice. Reads of the name as a plain Ident bail; reads of
   * `name(...)` substitute through the template and emit a direct scalar
   * read of the base tensor. See `tryLowerAsSliceBind`.
   */
  sliceAliases: Map<string, SliceAlias>;
  _hasTensorOps?: boolean;
  interp?: Interpreter;
  generatedFns: Map<string, string>;
  loweringInProgress: Set<string>;
  /** Pre-built line break table for offset→line lookup. */
  lineTable?: number[];
}

// ── Statement lowering ──────────────────────────────────────────────────

function lowerStmts(ctx: LowerCtx, stmts: Stmt[]): JitStmt[] | null {
  const result: JitStmt[] = [];
  for (const stmt of stmts) {
    const lowered = lowerStmt(ctx, stmt);
    if (!lowered) return null;
    result.push(...lowered);
  }
  return result;
}

function lowerStmt(ctx: LowerCtx, stmt: Stmt): JitStmt[] | null {
  // Emit SetLoc for line tracking if span info is available
  const prefix: JitStmt[] = [];
  if (ctx.lineTable && stmt.span) {
    const line = offsetToLineFast(ctx.lineTable, stmt.span.start);
    prefix.push({ tag: "SetLoc", line });
  }

  let result: JitStmt[] | null;
  switch (stmt.type) {
    case "Assign":
      result = lowerAssign(ctx, stmt);
      break;
    case "AssignLValue":
      result = lowerAssignLValue(ctx, stmt);
      break;
    case "ExprStmt":
      // Bail out: ExprStmt must set `ans` in the environment, which the
      // JIT codegen doesn't do.  Fall back to the interpreter.
      return null;
    case "If":
      result = lowerIf(ctx, stmt);
      break;
    case "For":
      result = lowerFor(ctx, stmt);
      break;
    case "While":
      result = lowerWhile(ctx, stmt);
      break;
    case "Break":
      result = [{ tag: "Break" }];
      break;
    case "Continue":
      result = [{ tag: "Continue" }];
      break;
    case "Return":
      result = [{ tag: "Return" }];
      break;
    case "MultiAssign":
      result = lowerMultiAssign(ctx, stmt);
      break;
    default:
      return null; // unsupported statement
  }
  if (!result) return null;
  return [...prefix, ...result];
}

function lowerAssign(
  ctx: LowerCtx,
  stmt: Stmt & { type: "Assign" }
): JitStmt[] | null {
  // If the RHS is a whole-tensor colon slice (`pts(:, i)`), register a
  // slice alias instead of lowering the RHS directly. See `tryLowerAsSliceBind`.
  const slice = tryLowerAsSliceBind(ctx, stmt.name, stmt.expr);
  if (slice === "bail") return null;
  if (slice !== null) return slice;

  const expr = lowerExpr(ctx, stmt.expr);
  if (!expr) return null;

  // A plain reassignment invalidates any previous slice alias on this name.
  ctx.sliceAliases.delete(stmt.name);

  ctx.env.set(stmt.name, expr.jitType);
  ctx.assignedVars.add(stmt.name);
  if (!ctx.params.has(stmt.name)) ctx.localVars.add(stmt.name);

  return [{ tag: "Assign", name: stmt.name, expr }];
}

/**
 * Try to lower `name = base(...)` as a slice-alias bind instead of a real
 * tensor read+allocation.
 *
 * Returns:
 *   - `null` if the RHS isn't a colon slice of a real tensor (caller falls
 *     through to normal Assign lowering).
 *   - `"bail"` if the RHS IS a colon slice but it's shaped in a way we
 *     can't safely alias (fall back to interpreter).
 *   - A `JitStmt[]` if the slice-alias bind succeeded. The statements are
 *     tmp-variable captures for any non-literal scalar indices in the bind
 *     (so the read-site sees the bind-time value, matching MATLAB semantics).
 */
function tryLowerAsSliceBind(
  ctx: LowerCtx,
  name: string,
  rhs: Expr
): JitStmt[] | null | "bail" {
  // The parser produces either `Index(Ident(base), indices)` or
  // `FuncCall(baseName, args)` depending on disambiguation — both mean
  // "index `base` with these args". We accept both forms and normalize.
  let baseName: string;
  let rawIndices: Expr[];
  if (rhs.type === "Index" && rhs.base.type === "Ident") {
    baseName = rhs.base.name;
    rawIndices = rhs.indices;
  } else if (rhs.type === "FuncCall") {
    baseName = rhs.name;
    rawIndices = rhs.args;
  } else {
    return null;
  }

  const baseType = ctx.env.get(baseName);
  if (!baseType || baseType.kind !== "tensor") return null;

  const hasColon = rawIndices.some(idx => idx.type === "Colon");
  if (!hasColon) return null;

  // From here on, any failure is a hard bail — the caller can't fall back
  // to normal Index lowering because Colon isn't supported there.

  // Real tensors only.
  if (baseType.isComplex === true) return "bail";
  // Fully-known shape required to infer the slice dimensions.
  if (!baseType.shape || baseType.shape.some(d => d === -1)) return "bail";
  // Range indices aren't supported yet (only bare `:`).
  if (rawIndices.some(idx => idx.type === "Range")) return "bail";
  // Require exact-arity multi-dim indexing.
  if (rawIndices.length !== baseType.shape.length) return "bail";
  // Can't rebind a param or an already-assigned local into a slice alias.
  if (ctx.params.has(name)) return "bail";
  if (ctx.assignedVars.has(name) && !ctx.sliceAliases.has(name)) return "bail";

  const template: ({ kind: "colon" } | { kind: "expr"; expr: JitExpr })[] = [];
  const sliceShape: number[] = [];
  const colonPositions: number[] = [];
  const stmts: JitStmt[] = [];

  for (let d = 0; d < rawIndices.length; d++) {
    const idx = rawIndices[d];
    if (idx.type === "Colon") {
      template.push({ kind: "colon" });
      sliceShape.push(baseType.shape[d]);
      colonPositions.push(d);
    } else {
      const lowered = lowerExpr(ctx, idx);
      if (!lowered) return "bail";
      if (
        lowered.jitType.kind !== "number" &&
        lowered.jitType.kind !== "boolean"
      )
        return "bail";
      if (lowered.tag === "NumberLiteral") {
        template.push({ kind: "expr", expr: lowered });
      } else {
        // Capture into a tmp local to freeze the bind-time value. This
        // preserves MATLAB semantics when the source var is reassigned
        // between the bind and a subsequent read.
        const tmpName = `_slice_${name}_d${d}`;
        stmts.push({ tag: "Assign", name: tmpName, expr: lowered });
        ctx.env.set(tmpName, lowered.jitType);
        if (!ctx.params.has(tmpName)) ctx.localVars.add(tmpName);
        ctx.assignedVars.add(tmpName);
        template.push({
          kind: "expr",
          expr: { tag: "Var", name: tmpName, jitType: lowered.jitType },
        });
      }
    }
  }

  // Sanity: at least one colon (hasColon above should guarantee this).
  if (colonPositions.length === 0) return "bail";

  ctx.sliceAliases.set(name, {
    baseName,
    baseType,
    template,
    sliceShape,
    colonPositions,
  });
  // Ensure plain Ident reads of `name` bail — they'd otherwise resolve to
  // a stale env type (or succeed silently with the wrong value).
  ctx.env.delete(name);
  ctx._hasTensorOps = true;

  return stmts;
}

/**
 * Lower a read `alias(...)` where `alias` is a slice-aliased name. Supports
 * two shapes:
 *   - linear indexing with a single index into a 1-colon slice:
 *     `pt(k)` where `pt = pts(:, i)` → `pts(k, i)`;
 *   - multi-indexing matching the number of colons:
 *     `pt(r, c)` where `pt = pts(:, :)` → `pts(r, c)`.
 * Anything else (wrong arity, slice-of-slice, etc.) bails.
 */
function lowerSliceAliasRead(
  ctx: LowerCtx,
  alias: SliceAlias,
  readIndices: Expr[]
): JitExpr | null {
  const ncolon = alias.colonPositions.length;

  const lowered: JitExpr[] = [];
  for (const idx of readIndices) {
    const lo = lowerExpr(ctx, idx);
    if (!lo) return null;
    if (lo.jitType.kind !== "number" && lo.jitType.kind !== "boolean")
      return null;
    lowered.push(lo);
  }

  let readForColon: JitExpr[];
  if (lowered.length === ncolon) {
    readForColon = lowered;
  } else if (lowered.length === 1 && ncolon === 1) {
    readForColon = [lowered[0]];
  } else {
    return null;
  }

  const fullIndices: JitExpr[] = [];
  let colonIdx = 0;
  for (const slot of alias.template) {
    if (slot.kind === "colon") {
      fullIndices.push(readForColon[colonIdx++]);
    } else {
      fullIndices.push(slot.expr);
    }
  }

  const baseExpr: JitExpr = {
    tag: "Var",
    name: alias.baseName,
    jitType: alias.baseType,
  };
  const resultType: JitType = { kind: "number" };
  ctx._hasTensorOps = true;
  return {
    tag: "Index",
    base: baseExpr,
    indices: fullIndices,
    jitType: resultType,
  };
}

/**
 * Lower `t(i) = v` (scalar indexed assignment on a tensor base).
 *
 * Currently supports the narrow case needed for the ptloop pattern:
 *   - lvalue is `Index` with 1..3 **scalar** indices
 *   - lvalue base is a plain `Ident` (no chained indexing or member access)
 *   - base is a real tensor in the type env
 *   - RHS is a scalar numeric expression (number/boolean)
 *
 * Anything else bails to `null`, falling the whole loop back to the
 * interpreter. Notably: slice writes (`t(a:b) = src`) and complex tensors
 * are not yet supported.
 */
function lowerAssignLValue(
  ctx: LowerCtx,
  stmt: Stmt & { type: "AssignLValue" }
): JitStmt[] | null {
  const lv = stmt.lvalue;

  // Only `Index` lvalues — not `Member`, `IndexCell`, etc.
  if (lv.type !== "Index") return null;

  // Only plain Ident bases — no `a.b(i) = v` or `a(j)(i) = v`.
  if (lv.base.type !== "Ident") return null;
  const baseName = lv.base.name;

  // Base must already exist in the type env as a real tensor.
  const baseType = ctx.env.get(baseName);
  if (!baseType || baseType.kind !== "tensor") return null;
  if (baseType.isComplex === true) return null;

  // Range-slice write `dst(a:b) = src(c:d)` is handled by a separate path
  // that generates an `AssignIndexRange` IR node. The current narrow shape
  // is exactly one linear range index — anything more general bails.
  if (lv.indices.length === 1 && lv.indices[0].type === "Range") {
    return tryLowerRangeAssign(
      ctx,
      baseName,
      baseType,
      lv.indices[0],
      stmt.expr
    );
  }

  // 1..3 indices, all scalar numeric.
  if (lv.indices.length < 1 || lv.indices.length > 3) return null;
  const indices: JitExpr[] = [];
  for (const idx of lv.indices) {
    const lowered = lowerExpr(ctx, idx);
    if (!lowered) return null;
    if (lowered.jitType.kind !== "number" && lowered.jitType.kind !== "boolean")
      return null;
    indices.push(lowered);
  }

  // RHS must be a scalar real/bool. Complex scalars into a real tensor
  // would upgrade the tensor to complex at runtime — too invasive for
  // stage 4, bail.
  const value = lowerExpr(ctx, stmt.expr);
  if (!value) return null;
  if (value.jitType.kind !== "number" && value.jitType.kind !== "boolean")
    return null;

  // The base var is both read and written. Mark it assigned so the JIT
  // loop output-set filter keeps it live, and so the hoist logic knows
  // it's a write-target (will go through `unshare` at loop entry).
  ctx.assignedVars.add(baseName);
  // Don't touch ctx.env for baseName — its tensor type is unchanged
  // (same shape, still real — we're only updating one element).
  ctx._hasTensorOps = true;

  return [
    {
      tag: "AssignIndex",
      baseName,
      indices,
      value,
      baseType,
    },
  ];
}

/**
 * Lower `dst(a:b) = src(c:d)` (range-slice write between two real tensors).
 *
 * Currently supports the narrow chunkie grow-and-copy shape:
 *   - LHS lvalue is `Index(Ident(dst), [Range(start, end)])` (1 linear index)
 *   - RHS expression is `Index(Ident(src), [Range(start, end)])` or
 *     `FuncCall(src, [Range(start, end)])` — both forms appear depending on
 *     parser disambiguation
 *   - Both `dst` and `src` are real tensors in the type env
 *   - Range step must be `null` (default step of 1)
 *
 * Anything else bails to `null`. Multi-dim slice writes (`dst(:, j) = ...`),
 * stepped ranges, scalar fills (`dst(a:b) = 0`), and complex tensors are
 * out of scope for stage 6.
 */
function tryLowerRangeAssign(
  ctx: LowerCtx,
  baseName: string,
  baseType: JitType,
  lhsRange: Expr,
  rhsExpr: Expr
): JitStmt[] | null {
  if (lhsRange.type !== "Range") return null;
  if (lhsRange.step !== null) return null;

  // RHS must be a range slice of another real tensor. The parser may
  // produce either Index or FuncCall depending on disambiguation, same
  // as for slice reads (see tryLowerAsSliceBind).
  let srcName: string;
  let srcRange: Expr;
  if (
    rhsExpr.type === "Index" &&
    rhsExpr.base.type === "Ident" &&
    rhsExpr.indices.length === 1 &&
    rhsExpr.indices[0].type === "Range"
  ) {
    srcName = rhsExpr.base.name;
    srcRange = rhsExpr.indices[0];
  } else if (
    rhsExpr.type === "FuncCall" &&
    rhsExpr.args.length === 1 &&
    rhsExpr.args[0].type === "Range"
  ) {
    srcName = rhsExpr.name;
    srcRange = rhsExpr.args[0];
  } else {
    return null;
  }
  if (srcRange.type !== "Range" || srcRange.step !== null) return null;

  const srcType = ctx.env.get(srcName);
  if (!srcType || srcType.kind !== "tensor" || srcType.isComplex === true)
    return null;

  const dstStart = lowerExpr(ctx, lhsRange.start);
  if (!dstStart) return null;
  if (dstStart.jitType.kind !== "number" && dstStart.jitType.kind !== "boolean")
    return null;
  const dstEnd = lowerExpr(ctx, lhsRange.end);
  if (!dstEnd) return null;
  if (dstEnd.jitType.kind !== "number" && dstEnd.jitType.kind !== "boolean")
    return null;

  const srcStart = lowerExpr(ctx, srcRange.start);
  if (!srcStart) return null;
  if (srcStart.jitType.kind !== "number" && srcStart.jitType.kind !== "boolean")
    return null;
  const srcEnd = lowerExpr(ctx, srcRange.end);
  if (!srcEnd) return null;
  if (srcEnd.jitType.kind !== "number" && srcEnd.jitType.kind !== "boolean")
    return null;

  // The dst is both read (existing data preserved outside the range) and
  // written. Mark it assigned so the JIT loop output filter keeps it live
  // and the codegen hoist treats it as a write target (unshare-on-entry).
  ctx.assignedVars.add(baseName);
  // Plain reassignment of dst would invalidate any prior slice alias on
  // it. A range write doesn't change the type — clear the alias defensively
  // to mirror lowerAssign's behavior on this name.
  ctx.sliceAliases.delete(baseName);
  ctx._hasTensorOps = true;

  return [
    {
      tag: "AssignIndexRange",
      baseName,
      baseType,
      dstStart,
      dstEnd,
      srcBaseName: srcName,
      srcType,
      srcStart,
      srcEnd,
    },
  ];
}

function lowerMultiAssign(
  ctx: LowerCtx,
  stmt: Stmt & { type: "MultiAssign" }
): JitStmt[] | null {
  const nargout = stmt.lvalues.length;

  // Only support simple variable lvalues and ~ (Ignore)
  const names: (string | null)[] = [];
  for (const lv of stmt.lvalues) {
    if (lv.type === "Var") names.push(lv.name);
    else if (lv.type === "Ignore") names.push(null);
    else return null; // unsupported lvalue (index, member, etc.)
  }

  // RHS must be a FuncCall (either IBuiltin or user function)
  const rhs = stmt.expr;
  if (rhs.type !== "FuncCall") return null;

  // Lower the arguments
  const args = rhs.args.map(a => lowerExpr(ctx, a));
  if (args.some(a => a === null)) return null;
  const loweredArgs = args as JitExpr[];
  const argJitTypes = loweredArgs.map(a => a.jitType);

  // Try IBuiltin resolution with actual nargout. Per-context JS user
  // functions (.numbl.js) take priority over native builtins.
  const jsEntry = ctx.interp?.ctx.registry.jsUserFunctionsByName.get(rhs.name);
  const ib = jsEntry?.builtin ?? getIBuiltin(rhs.name);
  if (!ib) return null; // user function multi-output not yet supported

  const resolution = ib.resolve(argJitTypes, nargout);
  if (!resolution || resolution.outputTypes.length < nargout) return null;
  const outputTypes = resolution.outputTypes;

  // If any output type is unknown, bail — the builtin likely depends on
  // runtime state and must go through dispatch.
  if (outputTypes.some(t => t.kind === "unknown")) return null;

  // Update type environment for each output variable
  for (let i = 0; i < names.length; i++) {
    const name = names[i];
    if (name !== null) {
      ctx.env.set(name, outputTypes[i]);
      ctx.assignedVars.add(name);
      if (!ctx.params.has(name)) ctx.localVars.add(name);
    }
  }

  ctx._hasTensorOps = true;

  return [
    {
      tag: "MultiAssign",
      names,
      callName: rhs.name,
      args: loweredArgs,
      outputTypes,
    },
  ];
}

function lowerIf(ctx: LowerCtx, stmt: Stmt & { type: "If" }): JitStmt[] | null {
  const cond = lowerExpr(ctx, stmt.cond);
  if (!cond) return null;
  // Only numeric scalar conditions supported
  if (!isScalarType(cond.jitType)) return null;
  if (cond.jitType.kind === "string" || cond.jitType.kind === "char")
    return null;
  if (cond.jitType.kind === "complex_or_number") ctx._hasTensorOps = true;

  const envBefore = cloneEnv(ctx.env);
  // Slice aliases are lexically scoped to the block they're bound in; we
  // snapshot here so aliases created inside a branch don't leak to the
  // post-if code (where the binding may or may not have run).
  const sliceAliasesBefore = new Map(ctx.sliceAliases);

  // Then branch
  ctx.env = cloneEnv(envBefore);
  ctx.sliceAliases = new Map(sliceAliasesBefore);
  const thenBody = lowerStmts(ctx, stmt.thenBody);
  if (!thenBody) return null;
  let mergedEnv = cloneEnv(ctx.env);

  // Elseif branches
  const elseifBlocks: { cond: JitExpr; body: JitStmt[] }[] = [];
  for (const eib of stmt.elseifBlocks) {
    ctx.env = cloneEnv(envBefore);
    ctx.sliceAliases = new Map(sliceAliasesBefore);
    const eibCond = lowerExpr(ctx, eib.cond);
    if (!eibCond) return null;
    if (!isScalarType(eibCond.jitType)) return null;
    if (eibCond.jitType.kind === "string" || eibCond.jitType.kind === "char")
      return null;
    if (eibCond.jitType.kind === "complex_or_number") ctx._hasTensorOps = true;
    const eibBody = lowerStmts(ctx, eib.body);
    if (!eibBody) return null;
    elseifBlocks.push({ cond: eibCond, body: eibBody });

    const merged = mergeEnvs(mergedEnv, ctx.env);
    if (!merged) return null;
    mergedEnv = merged;
  }

  // Else branch
  let elseBody: JitStmt[] | null = null;
  if (stmt.elseBody) {
    ctx.env = cloneEnv(envBefore);
    ctx.sliceAliases = new Map(sliceAliasesBefore);
    elseBody = lowerStmts(ctx, stmt.elseBody);
    if (!elseBody) return null;

    const merged = mergeEnvs(mergedEnv, ctx.env);
    if (!merged) return null;
    mergedEnv = merged;
  } else {
    // No else: merge with pre-if env (variable might not be assigned)
    const merged = mergeEnvs(mergedEnv, envBefore);
    if (!merged) return null;
    mergedEnv = merged;
  }

  ctx.env = mergedEnv;
  // Restore alias state — any aliases created inside a branch are gone.
  ctx.sliceAliases = sliceAliasesBefore;

  return [{ tag: "If", cond, thenBody, elseifBlocks, elseBody }];
}

function lowerFor(
  ctx: LowerCtx,
  stmt: Stmt & { type: "For" }
): JitStmt[] | null {
  // Only Range-based for loops
  if (stmt.expr.type !== "Range") return null;

  const start = lowerExpr(ctx, stmt.expr.start);
  if (!start || !isNumericScalarType(start.jitType)) return null;
  const step = stmt.expr.step ? lowerExpr(ctx, stmt.expr.step) : null;
  if (stmt.expr.step && (!step || !isNumericScalarType(step!.jitType)))
    return null;
  const end = lowerExpr(ctx, stmt.expr.end);
  if (!end || !isNumericScalarType(end.jitType)) return null;

  // Loop variable is always number
  ctx.env.set(stmt.varName, { kind: "number" });
  ctx.assignedVars.add(stmt.varName);
  if (!ctx.params.has(stmt.varName)) ctx.localVars.add(stmt.varName);

  const envBefore = cloneEnv(ctx.env);
  const sliceAliasesBefore = new Map(ctx.sliceAliases);

  // Lower body (first pass)
  ctx.sliceAliases = new Map(sliceAliasesBefore);
  const body = lowerStmts(ctx, stmt.body);
  if (!body) return null;

  // Merge pre-loop and post-body envs (loop might not execute)
  let merged = mergeEnvs(envBefore, ctx.env);
  if (!merged) return null;

  // Fixed-point iteration: re-lower until types stabilize (per-loop budget)
  let finalBody = body;
  let prevEnv = envBefore;
  let repassBudget = 20;
  while (!envsEqual(merged, prevEnv)) {
    if (repassBudget <= 0) return null;
    repassBudget--;
    ctx.env = cloneEnv(merged);
    ctx.env.set(stmt.varName, { kind: "number" });
    ctx.sliceAliases = new Map(sliceAliasesBefore);
    const newBody = lowerStmts(ctx, stmt.body);
    if (!newBody) return null;

    const newMerged = mergeEnvs(merged, ctx.env);
    if (!newMerged) return null;
    prevEnv = merged;
    merged = newMerged;
    finalBody = newBody;
  }

  ctx.env = merged;
  // Slice aliases created inside the body don't leak out of the loop.
  ctx.sliceAliases = sliceAliasesBefore;

  return [
    {
      tag: "For",
      varName: stmt.varName,
      start,
      step,
      end,
      body: finalBody,
    },
  ];
}

function lowerWhile(
  ctx: LowerCtx,
  stmt: Stmt & { type: "While" }
): JitStmt[] | null {
  const envBefore = cloneEnv(ctx.env);
  const sliceAliasesBefore = new Map(ctx.sliceAliases);

  const cond = lowerExpr(ctx, stmt.cond);
  if (!cond) return null;
  if (!isScalarType(cond.jitType)) return null;
  if (cond.jitType.kind === "string" || cond.jitType.kind === "char")
    return null;
  if (cond.jitType.kind === "complex_or_number") ctx._hasTensorOps = true;

  ctx.sliceAliases = new Map(sliceAliasesBefore);
  const body = lowerStmts(ctx, stmt.body);
  if (!body) return null;

  // Merge pre-loop and post-body
  let merged = mergeEnvs(envBefore, ctx.env);
  if (!merged) return null;

  // Fixed-point iteration: re-lower until types stabilize (per-loop budget)
  let finalCond = cond;
  let finalBody = body;
  let prevEnv = envBefore;
  let repassBudget = 20;
  while (!envsEqual(merged, prevEnv)) {
    if (repassBudget <= 0) return null;
    repassBudget--;
    ctx.env = cloneEnv(merged);
    ctx.sliceAliases = new Map(sliceAliasesBefore);
    const newCond = lowerExpr(ctx, stmt.cond);
    if (!newCond) return null;
    const newBody = lowerStmts(ctx, stmt.body);
    if (!newBody) return null;

    const newMerged = mergeEnvs(merged, ctx.env);
    if (!newMerged) return null;
    prevEnv = merged;
    merged = newMerged;
    finalCond = newCond;
    finalBody = newBody;
  }

  ctx.env = merged;
  ctx.sliceAliases = sliceAliasesBefore;

  return [{ tag: "While", cond: finalCond, body: finalBody }];
}

// ── Expression lowering ─────────────────────────────────────────────────

function lowerExpr(ctx: LowerCtx, expr: Expr): JitExpr | null {
  switch (expr.type) {
    case "Number": {
      const value = parseFloat(expr.value);
      return {
        tag: "NumberLiteral",
        value,
        jitType: {
          kind: "number",
          exact: value,
          ...(signFromNumber(value) ? { sign: signFromNumber(value) } : {}),
        },
      };
    }

    case "ImagUnit":
      return {
        tag: "ImagLiteral",
        jitType: { kind: "complex_or_number", pureImaginary: true },
      };

    case "Ident": {
      // Slice alias names can't be read as plain values (they don't
      // correspond to a real tensor at runtime — only to a set of scalar
      // locals and a substitution rule). Bail.
      if (ctx.sliceAliases.has(expr.name)) return null;

      // Known numeric constants
      const constVal = KNOWN_CONSTANTS[expr.name];
      if (constVal !== undefined) {
        const isBool = expr.name === "true" || expr.name === "false";
        return {
          tag: "NumberLiteral",
          value: constVal,
          jitType: isBool
            ? { kind: "boolean", value: expr.name === "true" }
            : {
                kind: "number",
                exact: constVal,
                ...(signFromNumber(constVal)
                  ? { sign: signFromNumber(constVal) }
                  : {}),
              },
        };
      }
      const type = ctx.env.get(expr.name);
      if (!type) return null; // undefined variable
      return { tag: "Var", name: expr.name, jitType: type };
    }

    case "Binary": {
      const left = lowerExpr(ctx, expr.left);
      if (!left) return null;
      const right = lowerExpr(ctx, expr.right);
      if (!right) return null;
      const resultType = binaryResultType(expr.op, left.jitType, right.jitType);
      if (!resultType || resultType.kind === "unknown") return null;

      // Track ops that need $h helpers (tensor or complex)
      if (
        isTensorType(left.jitType) ||
        isTensorType(right.jitType) ||
        left.jitType.kind === "complex_or_number" ||
        right.jitType.kind === "complex_or_number"
      ) {
        ctx._hasTensorOps = true;
      }

      return { tag: "Binary", op: expr.op, left, right, jitType: resultType };
    }

    case "Unary": {
      const operand = lowerExpr(ctx, expr.operand);
      if (!operand) return null;
      const resultType = unaryResultType(expr.op, operand.jitType);
      if (!resultType || resultType.kind === "unknown") return null;

      if (
        isTensorType(operand.jitType) ||
        operand.jitType.kind === "complex_or_number"
      )
        ctx._hasTensorOps = true;

      return { tag: "Unary", op: expr.op, operand, jitType: resultType };
    }

    case "Tensor": {
      const rows: JitExpr[][] = [];
      let hasComplex = false;
      for (const row of expr.rows) {
        const loweredRow: JitExpr[] = [];
        for (const elem of row) {
          const lowered = lowerExpr(ctx, elem);
          if (!lowered) return null;
          // Only scalar elements supported in tensor literals
          if (
            lowered.jitType.kind !== "number" &&
            lowered.jitType.kind !== "boolean" &&
            lowered.jitType.kind !== "complex_or_number"
          )
            return null;
          if (lowered.jitType.kind === "complex_or_number") hasComplex = true;
          loweredRow.push(lowered);
        }
        rows.push(loweredRow);
      }
      const nRows = rows.length;
      const nCols = rows[0]?.length ?? 0;
      ctx._hasTensorOps = true;
      return {
        tag: "TensorLiteral",
        rows,
        nRows,
        nCols,
        jitType: {
          kind: "tensor",
          isComplex: hasComplex,
          shape: [nRows, nCols],
        },
      };
    }

    case "FuncCall": {
      // If the name is a known variable, treat as indexing (MATLAB ambiguity)
      const varType = ctx.env.get(expr.name);
      if (varType) {
        return lowerIndexExpr(ctx, {
          base: { tag: "Var", name: expr.name, jitType: varType },
          indices: expr.args,
        });
      }

      // Slice alias `pt(k)` — parsed as FuncCall because the parser
      // doesn't know `pt` is a variable. Handle it like Index.
      const alias = ctx.sliceAliases.get(expr.name);
      if (alias) return lowerSliceAliasRead(ctx, alias, expr.args);

      // Try user function resolution (nested → local → workspace → class method)
      const userResult = lowerUserFuncCall(ctx, expr);
      if (userResult !== undefined) return userResult;

      // Try IBuiltin resolution (same priority as builtins — last)
      return lowerIBuiltinCall(ctx, expr);
    }

    case "Char": {
      // Strip enclosing quotes and unescape doubled single-quotes
      const charVal = expr.value.slice(1, -1).replaceAll("''", "'");
      return {
        tag: "StringLiteral",
        value: charVal,
        isChar: true,
        jitType: { kind: "char", value: charVal },
      };
    }

    case "String": {
      // Strip surrounding quotes and unescape doubled quotes (same as interpreter)
      let strVal = expr.value.slice(1, expr.value.length - 1);
      strVal = strVal.replaceAll('""', '"');
      return {
        tag: "StringLiteral",
        value: strVal,
        isChar: false,
        jitType: { kind: "string", value: strVal },
      };
    }

    case "Index": {
      // Slice alias intercept: `pt(k)` where `pt = pts(:, i)` → `pts(k, i)`.
      if (expr.base.type === "Ident") {
        const alias = ctx.sliceAliases.get(expr.base.name);
        if (alias) return lowerSliceAliasRead(ctx, alias, expr.indices);
      }
      const base = lowerExpr(ctx, expr.base);
      if (!base) return null;
      return lowerIndexExpr(ctx, { base, indices: expr.indices });
    }

    default:
      return null; // unsupported expression
  }
}

// ── Index expression lowering ───────────────────────────────────────────

function lowerIndexExpr(
  ctx: LowerCtx,
  input: { base: JitExpr; indices: Expr[] }
): JitExpr | null {
  const { base } = input;
  const indices: JitExpr[] = [];
  for (const idx of input.indices) {
    const lowered = lowerExpr(ctx, idx);
    if (!lowered) return null;
    // Only scalar numeric indices supported
    if (lowered.jitType.kind !== "number" && lowered.jitType.kind !== "boolean")
      return null;
    indices.push(lowered);
  }
  if (indices.length === 0) return null;

  // Determine result type based on base type
  let resultType: JitType;
  switch (base.jitType.kind) {
    case "tensor":
      resultType = base.jitType.isComplex
        ? { kind: "complex_or_number" }
        : { kind: "number" };
      break;
    case "number":
    case "boolean":
      resultType = { kind: "number" };
      break;
    case "complex_or_number":
      resultType = { kind: "complex_or_number" };
      break;
    default:
      return null;
  }

  ctx._hasTensorOps = true;
  return { tag: "Index", base, indices, jitType: resultType };
}

// ── User function call resolution ───────────────────────────────────────

/**
 * Try to resolve and compile a user function call.
 * Returns:
 *   JitExpr - successfully compiled
 *   null    - function found but can't compile (bail out of containing function)
 *   undefined - no user function found (fall through to builtins)
 */
function lowerUserFuncCall(
  ctx: LowerCtx,
  expr: Expr & { type: "FuncCall" }
): JitExpr | null | undefined {
  const interp = ctx.interp;
  if (!interp) return undefined;

  // Lower arguments first to determine types
  const args = expr.args.map(a => lowerExpr(ctx, a));
  if (args.some(a => a === null)) return null;
  const loweredArgs = args as JitExpr[];
  const argJitTypes = loweredArgs.map(a => a.jitType);

  // Resolve the function using the same mechanism as the interpreter
  const calleeFn = resolveUserFunction(interp, expr.name, argJitTypes);
  if (!calleeFn) return undefined; // no user function found — fall through to builtins

  // Build identity string for unique naming
  const calleeNargout = 1; // nested calls always expect 1 output
  const typeKey = argJitTypes.map(jitTypeKey).join(":");
  const identity = `${interp.currentFile}:${calleeFn.name}:${calleeNargout}:${typeKey}`;
  const jitName = computeJitFnName(identity, calleeFn.name);

  // Already generated? Reuse.
  if (ctx.generatedFns.has(jitName)) {
    // Need to determine return type - re-lower to get it (or cache it)
    const returnType = getGeneratedFnReturnType(
      calleeFn,
      argJitTypes,
      calleeNargout,
      ctx
    );
    if (!returnType) return null;
    return {
      tag: "UserCall",
      jitName,
      name: calleeFn.name,
      args: loweredArgs,
      jitType: returnType,
    };
  }

  // Recursion guard
  if (ctx.loweringInProgress.has(jitName)) return null;
  ctx.loweringInProgress.add(jitName);

  try {
    // Recursively lower the callee
    const calleeResult = lowerFunction(
      calleeFn,
      argJitTypes,
      calleeNargout,
      interp,
      ctx.generatedFns,
      ctx.loweringInProgress
    );
    if (!calleeResult) return null;

    // Generate JS for the callee and wrap in a named function
    const calleeJS = generateJS(
      calleeResult.body,
      calleeFn.params,
      calleeResult.outputNames,
      calleeNargout,
      calleeResult.localVars,
      interp.currentFile
    );
    const returnType = calleeResult.outputType ?? { kind: "number" as const };
    const paramComments = calleeFn.params
      .map((p, i) => `${p}: ${jitTypeKey(argJitTypes[i])}`)
      .join(", ");
    const outputComments = calleeResult.outputNames
      .map(
        o =>
          `${o}: ${jitTypeKey(calleeResult.outputType ?? { kind: "number" })}`
      )
      .join(", ");
    const comment = [
      `// JIT: ${calleeFn.name}(${paramComments}) -> (${outputComments})`,
      `// from: ${interp.currentFile}`,
    ].join("\n");
    const wrappedJS = `${comment}\nfunction ${jitName}(${calleeFn.params.join(", ")}) {\n${calleeJS}\n}`;
    ctx.generatedFns.set(jitName, wrappedJS);

    // Propagate tensor ops flag
    if (calleeResult.hasTensorOps) ctx._hasTensorOps = true;

    return {
      tag: "UserCall",
      jitName,
      name: calleeFn.name,
      args: loweredArgs,
      jitType: returnType,
    };
  } finally {
    ctx.loweringInProgress.delete(jitName);
  }
}

/** Convert JitType to ItemType for function resolution (only ClassInstance matters). */
function jitTypeToItemType(t: JitType): ItemType {
  if (t.kind === "class_instance") {
    return { kind: "ClassInstance", className: t.className };
  }
  return { kind: "Unknown" };
}

/** Resolve a function name to a FunctionDef using the interpreter's resolution. */
function resolveUserFunction(
  interp: Interpreter,
  name: string,
  argJitTypes: JitType[]
): FunctionDef | null {
  // 1. Check nested functions (mirrors interpreter's callFunction priority)
  const nested = interp.env.getNestedFunction(name);
  if (nested) return nested.fn;

  // 2. Check main local functions
  const localFn = interp.mainLocalFunctions.get(name);
  if (localFn) return localFn;

  // 3. Resolve via function index (for workspace functions, class methods, etc.)
  const callSite: CallSite = {
    file: interp.currentFile,
    ...(interp.currentClassName ? { className: interp.currentClassName } : {}),
    ...(interp.currentMethodName
      ? { methodName: interp.currentMethodName }
      : {}),
  };
  const argItemTypes = argJitTypes.map(jitTypeToItemType);
  const target = resolveFunction(
    name,
    argItemTypes,
    callSite,
    interp.functionIndex
  );
  if (!target) return null;

  if (target.kind === "localFunction" && target.source.from === "main") {
    return interp.mainLocalFunctions.get(target.name) ?? null;
  }

  if (target.kind === "classMethod") {
    const definingClass = interp.ctx.findDefiningClass(
      target.className,
      target.methodName
    );
    const classInfo = interp.ctx.getClassInfo(definingClass);
    if (!classInfo) return null;
    return (
      interp.findMethodInClass(classInfo, target.methodName) ??
      interp.findExternalMethod(classInfo, target.methodName)
    );
  }

  if (target.kind === "workspaceFunction") {
    const dotIdx = target.name.lastIndexOf(".");
    const primaryName =
      dotIdx >= 0 ? target.name.slice(dotIdx + 1) : target.name;
    return interp.findFunctionInWorkspaceFile(target.name, primaryName);
  }

  // jsUserFunction targets are handled by lowerIBuiltinCall (they implement
  // the IBuiltin interface), so this resolver returns null and the caller
  // falls through to the IBuiltin path.

  // Other target kinds (privateFunction, workspaceClassConstructor, etc.) not supported yet
  return null;
}

// ── IBuiltin call resolution ────────────────────────────────────────────

function lowerIBuiltinCall(
  ctx: LowerCtx,
  expr: Expr & { type: "FuncCall" }
): JitExpr | null {
  // Per-context JS user functions (.numbl.js) take priority over native
  // builtins so a .numbl.js file can shadow a builtin of the same name.
  const jsEntry = ctx.interp?.ctx.registry.jsUserFunctionsByName.get(expr.name);
  const ib = jsEntry?.builtin ?? getIBuiltin(expr.name);
  if (!ib) return null;

  const args = expr.args.map(a => lowerExpr(ctx, a));
  if (args.some(a => a === null)) return null;
  const loweredArgs = args as JitExpr[];
  const argJitTypes = loweredArgs.map(a => a.jitType);

  // If any argument is unknown, bail — it could be a class instance at runtime,
  // and class methods take priority over builtins in MATLAB.
  if (argJitTypes.some(t => t.kind === "unknown")) return null;

  const resolution = ib.resolve(argJitTypes, 1);
  if (!resolution || resolution.outputTypes.length === 0) return null;
  const outputTypes = resolution.outputTypes;

  // If the output type is unknown, bail — the builtin likely depends on
  // runtime state (e.g. evalin, assignin) and must go through dispatch.
  if (outputTypes[0].kind === "unknown") return null;

  // IBuiltin calls always go through $h helpers
  ctx._hasTensorOps = true;

  return {
    tag: "Call",
    name: expr.name,
    args: loweredArgs,
    jitType: outputTypes[0],
  };
}

/** Get the return type of an already-generated function. */
function getGeneratedFnReturnType(
  fn: FunctionDef,
  argTypes: JitType[],
  nargout: number,
  ctx: LowerCtx
): JitType | null {
  // Re-lower to determine the return type (this is cheap since the code is already generated)
  const result = lowerFunction(
    fn,
    argTypes,
    nargout,
    ctx.interp,
    ctx.generatedFns,
    ctx.loweringInProgress
  );
  return result?.outputType ?? null;
}
