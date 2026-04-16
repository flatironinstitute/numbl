/**
 * AST -> JIT IR lowering with type propagation.
 *
 * Returns null if any unsupported construct is encountered,
 * causing the entire function to fall back to interpretation.
 *
 * Type-level helpers (sign algebra, result types, env management) are
 * in jitLowerTypes.ts. This file contains the core lowering logic.
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
  isScalarType,
  isNumericScalarType,
  isTensorType,
  isKnownInteger,
  jitTypeKey,
  computeJitFnName,
  signFromNumber,
} from "./jitTypes.js";
import {
  KNOWN_CONSTANTS,
  type TypeEnv,
  cloneEnv,
  mergeEnvs,
  envsEqual,
  binaryResultType,
  unaryResultType,
} from "./jitLowerTypes.js";
import { generateJS } from "./jitCodegen.js";
import { getIBuiltin, inferJitType } from "../builtins/index.js";
import { offsetToLineFast, buildLineTable } from "../../runtime/error.js";
import { isRuntimeFunction } from "../../runtime/types.js";
import type { RuntimeValue } from "../../runtime/types.js";

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

  // Reserve output variables as locals without assigning a default type.
  // An earlier incarnation seeded env[name] = number=0 so every output had
  // *some* type at function exit, but that poisoned the loop-join merge
  // whenever the body's first assignment produced a non-number type
  // (e.g. `chld = T.nodes(i).chld` makes chld a tensor, which can't be
  // unified with the default number=0 that was already in envBefore).
  // Instead we rely on the "bail if output never assigned" check below
  // to catch outputs that the body doesn't initialize — which is the
  // correct failure mode, since reading an unassigned output is an error.
  const outputNames = fn.outputs.slice(0, nargout || 1);
  for (const name of outputNames) {
    if (!env.has(name)) {
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
      // Marker call `assert_jit_compiled()` is elided to nothing — its
      // job is done by the fact that we successfully reached this point
      // in the lowering. Anything else is a bail (an ExprStmt would set
      // `ans` in the env, which the JIT codegen doesn't do).
      if (
        stmt.expr.type === "FuncCall" &&
        stmt.expr.name === "assert_jit_compiled" &&
        stmt.expr.args.length === 0
      ) {
        return prefix;
      }
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

  // A plain reassignment invalidates any previous slice alias on this name,
  // AND any aliases that reference this name as their base tensor.
  ctx.sliceAliases.delete(stmt.name);
  for (const [aliasName, alias] of ctx.sliceAliases) {
    if (alias.baseName === stmt.name) {
      ctx.sliceAliases.delete(aliasName);
    }
  }

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
  // Shape must exist so we can check ndim and colon-dim sizes.
  if (!baseType.shape) return "bail";
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
      // The colon dimension's size must be statically known so the
      // slice alias can substitute read-site indices correctly.
      // Non-colon dimensions are scalar indices whose runtime value
      // is captured at bind time, so they don't need a known size.
      const dimSize = baseType.shape[d];
      if (dimSize === -1 || dimSize === undefined) return "bail";
      template.push({ kind: "colon" });
      sliceShape.push(dimSize);
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
 * Materialize a slice alias as a real tensor. Called when a slice-aliased
 * name is used as a whole value (e.g. `rx .^ 2` where `rx = A(1,:)`).
 * Only supports 2D base tensors with exactly one colon dimension.
 */
function materializeSliceAlias(
  ctx: LowerCtx,
  alias: SliceAlias
): JitExpr | null {
  const bt = alias.baseType;
  if (bt.kind !== "tensor") return null;
  if (!bt.shape || bt.shape.length !== 2) return null;
  if (alias.colonPositions.length !== 1) return null;
  if (alias.template.length !== 2) return null;

  const colonPos = alias.colonPositions[0];
  const fixedSlot = alias.template[colonPos === 0 ? 1 : 0];
  if (fixedSlot.kind !== "expr") return null;

  const sliceLen = alias.sliceShape[0];
  if (sliceLen <= 0) return null;

  // Emit: $h.__extractSlice2d(base, fixedIdx, colonPos, sliceLen)
  ctx._hasTensorOps = true;
  const baseVar: JitExpr = {
    tag: "Var",
    name: alias.baseName,
    jitType: alias.baseType,
  };
  const shape =
    colonPos === 0
      ? [sliceLen, 1] // column slice → Mx1
      : [1, sliceLen]; // row slice → 1xN
  return {
    tag: "Call",
    name: "__extractSlice2d",
    args: [
      baseVar,
      fixedSlot.expr,
      {
        tag: "NumberLiteral",
        value: colonPos,
        jitType: { kind: "number", exact: colonPos },
      },
      {
        tag: "NumberLiteral",
        value: sliceLen,
        jitType: { kind: "number", exact: sliceLen },
      },
    ],
    jitType: { kind: "tensor", isComplex: false, shape, ndim: 2 },
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

  // Cell array write: c{i} = v. Rebind `c` to the helper's return value so
  // the COW-copied cell (when the wrapper is shared, e.g. passed in as a
  // function arg) replaces the local; the caller's original cell stays put.
  if (lv.type === "IndexCell") {
    if (lv.base.type !== "Ident") return null;
    const cellType = ctx.env.get(lv.base.name);
    if (!cellType || cellType.kind !== "cell") return null;
    if (lv.indices.length !== 1) return null;
    const cellIdx = lowerExpr(ctx, lv.indices[0]);
    if (!cellIdx) return null;
    if (cellIdx.jitType.kind !== "number" && cellIdx.jitType.kind !== "boolean")
      return null;
    const rhs = lowerExpr(ctx, stmt.expr);
    if (!rhs) return null;
    ctx._hasTensorOps = true;
    ctx.assignedVars.add(lv.base.name);
    if (!ctx.params.has(lv.base.name)) ctx.localVars.add(lv.base.name);
    return [
      {
        tag: "Assign",
        name: lv.base.name,
        expr: {
          tag: "Call",
          name: "__cellWrite",
          args: [
            { tag: "Var", name: lv.base.name, jitType: cellType },
            cellIdx,
            rhs,
          ],
          jitType: cellType,
        },
      },
    ];
  }

  // Stage 22: Member lvalue `s.f = v` — scalar struct field assign.
  // Three supported cases for the base:
  //   (a) base is already a struct in env → mutate its fields map
  //   (b) base is an empty tensor (from `s = []`) → promote to fresh
  //       struct at runtime, update env type to struct
  //   (c) base is not yet in the env (write-only local) → ditto (b):
  //       compile as fresh-struct-init + field set, update env type
  // All three emit the same IR shape; codegen branches on
  // `needsPromote` to decide whether to emit the struct-init prefix.
  if (lv.type === "Member") {
    return tryLowerMemberAssign(ctx, lv, stmt.expr);
  }

  // Only `Index` lvalues — not `Member`, etc.
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

  // Stage 17: column slice write `dst(:, j) = src` where both are real
  // tensors. LHS shape: [Colon, scalar_j], dst is a 2-D real tensor, RHS
  // is a plain Ident referring to a real-tensor variable. Drives chunkie
  // adapgausskerneval's `vals(:, jj+1) = v2` and chunkerinterior's
  // `rss(:, jj) = rval(:, k)` after the latter's RHS binds to a tensor
  // via the stage 5 slice alias.
  if (
    lv.indices.length === 2 &&
    lv.indices[0].type === "Colon" &&
    lv.indices[1].type !== "Colon" &&
    lv.indices[1].type !== "Range"
  ) {
    const colResult = tryLowerColAssign(
      ctx,
      baseName,
      baseType,
      lv.indices[1],
      stmt.expr
    );
    if (colResult) return colResult;
    // Fall through — if the shape doesn't match the narrow col-write
    // pattern we bail on the whole statement below.
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
 * Lower `dst(a:b) = src(c:d)` or `dst(a:b) = src` (range-slice write
 * between two real tensors).
 *
 * Supports these chunkie grow-and-copy shapes:
 *   - LHS lvalue is `Index(Ident(dst), [Range(start, end)])` (1 linear index)
 *   - RHS expression is either
 *       (a) `Index(Ident(src), [Range(start, end)])` / `FuncCall(src, [Range])`
 *           — explicit source range (stage 6)
 *       (b) `Ident(src)` — whole-tensor source; the runtime length check
 *           compares dst range length against `src.data.length` (stage 9)
 *   - Both `dst` and `src` are real tensors in the type env
 *   - Range step must be `null` (default step of 1)
 *
 * For the whole-tensor case, `srcStart`/`srcEnd` on the IR node are
 * `null` — the codegen substitutes `1` and the source's hoisted length
 * alias so the same helper (`setRange1r_h`) handles both shapes.
 *
 * Anything else bails to `null`. Multi-dim slice writes (`dst(:, j) = ...`),
 * stepped ranges, scalar fills (`dst(a:b) = 0`), and complex tensors are
 * out of scope.
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

  // RHS can be:
  //   - a range slice of another real tensor (stage 6), or
  //   - a plain Ident referencing a whole real tensor (stage 9).
  // The parser produces Index or FuncCall for the range form depending on
  // disambiguation, same as for slice reads (see tryLowerAsSliceBind).
  let srcName: string;
  let srcRange: Expr | null;
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
  } else if (rhsExpr.type === "Ident") {
    // Skip slice-alias names — they don't correspond to a real tensor
    // at runtime.
    if (ctx.sliceAliases.has(rhsExpr.name)) return null;
    srcName = rhsExpr.name;
    srcRange = null;
  } else {
    return null;
  }
  if (
    srcRange !== null &&
    (srcRange.type !== "Range" || srcRange.step !== null)
  )
    return null;

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

  let srcStart: JitExpr | null = null;
  let srcEnd: JitExpr | null = null;
  if (srcRange !== null) {
    srcStart = lowerExpr(ctx, srcRange.start);
    if (!srcStart) return null;
    if (
      srcStart.jitType.kind !== "number" &&
      srcStart.jitType.kind !== "boolean"
    )
      return null;
    srcEnd = lowerExpr(ctx, srcRange.end);
    if (!srcEnd) return null;
    if (srcEnd.jitType.kind !== "number" && srcEnd.jitType.kind !== "boolean")
      return null;
  }

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

/**
 * Lower `dst(:, j) = src` where both are real tensors.
 *
 * Shape constraints:
 *   - LHS is `Index(Ident(dst), [Colon, scalar_j])`
 *   - `dst` is a real tensor with shape.length === 2 and shape[0] known
 *     at compile time (so codegen can check RHS length statically at
 *     lowering time; shape[0] is also available via the hoisted $dst_d0
 *     alias at runtime)
 *   - RHS is a plain `Ident` referring to a real tensor (sliced or
 *     whole — we don't care as long as the linear element count
 *     matches dst.shape[0] at runtime)
 *
 * Anything else returns null so the caller bails.
 */
function tryLowerColAssign(
  ctx: LowerCtx,
  baseName: string,
  baseType: JitType,
  colIdxExpr: Expr,
  rhsExpr: Expr
): JitStmt[] | null {
  if (baseType.kind !== "tensor" || baseType.isComplex === true) return null;
  if (!baseType.shape || baseType.shape.length !== 2) return null;

  // RHS: plain Ident referring to a real-tensor variable in the env.
  // Matches the chunkie `vals(:, jj+1) = v2;` shape. Slice-alias names
  // don't correspond to a real tensor at runtime, so skip them.
  if (rhsExpr.type !== "Ident") return null;
  if (ctx.sliceAliases.has(rhsExpr.name)) return null;
  const srcName = rhsExpr.name;
  const srcType = ctx.env.get(srcName);
  if (!srcType || srcType.kind !== "tensor" || srcType.isComplex === true)
    return null;

  const colIndex = lowerExpr(ctx, colIdxExpr);
  if (!colIndex) return null;
  if (colIndex.jitType.kind !== "number" && colIndex.jitType.kind !== "boolean")
    return null;

  // Mark dst as assigned (write-target → unshare-on-entry hoist); clear
  // any prior slice alias defensively, matching the range-assign path.
  ctx.assignedVars.add(baseName);
  ctx.sliceAliases.delete(baseName);
  ctx._hasTensorOps = true;

  return [
    {
      tag: "AssignIndexCol",
      baseName,
      baseType,
      colIndex,
      srcBaseName: srcName,
      srcType,
    },
  ];
}

/**
 * Stage 21: lower `src(a:b)` on a real-tensor base into a
 * `RangeSliceRead` IR node producing a fresh column-vector tensor.
 *
 * Accepts `Range` with no step (default step 1). `start` and `end`
 * must lower to numeric/boolean scalar exprs. The result is a real
 * tensor with shape `[?, 1]` — the exact length is runtime-dependent.
 *
 * Caller responsibility: match the parent expression shape
 * `Index(Ident(src), [Range])` or `FuncCall(src, [Range])` before
 * calling. Returns null if the source isn't a real tensor or the
 * range isn't the expected shape.
 */
function tryLowerRangeSliceRead(
  ctx: LowerCtx,
  baseName: string,
  rangeExpr: Expr
): JitExpr | null {
  if (rangeExpr.type !== "Range") return null;
  if (rangeExpr.step !== null) return null;

  const srcType = ctx.env.get(baseName);
  if (!srcType || srcType.kind !== "tensor" || srcType.isComplex === true)
    return null;
  // Slice-alias names don't correspond to a real tensor at runtime.
  if (ctx.sliceAliases.has(baseName)) return null;

  const start = lowerExpr(ctx, rangeExpr.start);
  if (!start) return null;
  if (start.jitType.kind !== "number" && start.jitType.kind !== "boolean")
    return null;

  // `end` keyword inside the indexing context refers to the base's
  // linear length. Codegen substitutes the hoisted `.data.length`
  // alias. Any other expression lowers normally.
  let end: JitExpr | null;
  if (rangeExpr.end.type === "EndKeyword") {
    end = null;
  } else {
    end = lowerExpr(ctx, rangeExpr.end);
    if (!end) return null;
    if (end.jitType.kind !== "number" && end.jitType.kind !== "boolean")
      return null;
  }

  ctx._hasTensorOps = true;
  return {
    tag: "RangeSliceRead",
    baseName,
    start,
    end,
    jitType: { kind: "tensor", isComplex: false, shape: [-1, 1] },
  };
}

/**
 * Lower `s.f = v` (Member lvalue). Base must be a plain Ident. Three
 * supported base-type cases:
 *   (a) base is already a `struct` in env → mutate its fields map.
 *   (b) base is a tensor with shape [0, 0] (the `s = []` idiom) →
 *       promote to fresh struct at runtime, update env type to struct.
 *   (c) base has no env type yet (write-only local) → same as (b).
 *
 * After lowering, env type for `baseName` is set to a struct that
 * includes the new field (with `value.jitType`) so subsequent
 * `s.f` reads (stage 12) can resolve it. Field type is re-unified if
 * the field is re-assigned with a different type.
 *
 * RHS must lower to a numeric scalar or real tensor — class instances
 * and cells are out of scope. The actual RHS type is stored in the
 * struct's fields map for subsequent MemberRead lookups.
 */
function tryLowerMemberAssign(
  ctx: LowerCtx,
  lv: { type: "Member"; base: Expr; name: string },
  rhsExpr: Expr
): JitStmt[] | null {
  if (lv.base.type !== "Ident") return null;
  const baseName = lv.base.name;
  const fieldName = lv.name;

  // Reject class instances (their field dispatch may go through
  // user-defined getters/setters that the simple fields-map write
  // would bypass).
  const currentType = ctx.env.get(baseName);
  if (currentType && currentType.kind === "class_instance") return null;

  // Determine whether we need to promote at runtime.
  //   - struct → no promote, mutate fields map directly
  //   - tensor[0x0] (real) → promote, `s = []; s.f = v` idiom
  //   - undefined → promote, write-only local
  //   - anything else → bail (complex tensors, unknown, cells, …)
  let needsPromote: boolean;
  if (currentType === undefined) {
    needsPromote = true;
  } else if (currentType.kind === "struct") {
    needsPromote = false;
  } else if (
    currentType.kind === "tensor" &&
    currentType.isComplex !== true &&
    currentType.shape &&
    currentType.shape.length === 2 &&
    currentType.shape[0] === 0 &&
    currentType.shape[1] === 0
  ) {
    needsPromote = true;
  } else {
    return null;
  }

  // Lower the RHS. Only accept numeric scalars and real tensors.
  const value = lowerExpr(ctx, rhsExpr);
  if (!value) return null;
  const vk = value.jitType.kind;
  if (
    vk !== "number" &&
    vk !== "boolean" &&
    vk !== "tensor" &&
    vk !== "complex_or_number"
  )
    return null;
  if (vk === "tensor" && value.jitType.isComplex === true) return null;

  // Update env: mark baseName as a struct whose fields include the
  // new field. If the base is already a struct, extend (or overwrite)
  // its fields map with the new field's type.
  let newFields: Record<string, JitType>;
  if (currentType && currentType.kind === "struct" && currentType.fields) {
    newFields = { ...currentType.fields, [fieldName]: value.jitType };
  } else {
    newFields = { [fieldName]: value.jitType };
  }
  ctx.env.set(baseName, { kind: "struct", fields: newFields });
  ctx.assignedVars.add(baseName);
  // Clear any slice alias on this name (defensive — struct type supersedes).
  ctx.sliceAliases.delete(baseName);

  return [
    {
      tag: "AssignMember",
      baseName,
      fieldName,
      value,
      needsPromote,
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

  // Save env BEFORE setting the loop variable so that in the zero-iteration
  // case, the merge doesn't leak the loop variable's type into the post-loop
  // env (MATLAB leaves the loop variable undefined after `for i = 1:0`).
  const envBefore = cloneEnv(ctx.env);
  const sliceAliasesBefore = new Map(ctx.sliceAliases);

  // Loop variable is number (integer when start and step are integer)
  const loopVarIsInt =
    isKnownInteger(start.jitType) && (!step || isKnownInteger(step.jitType));
  ctx.env.set(stmt.varName, {
    kind: "number",
    ...(loopVarIsInt ? { isInteger: true } : {}),
  });
  ctx.assignedVars.add(stmt.varName);
  if (!ctx.params.has(stmt.varName)) ctx.localVars.add(stmt.varName);

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
    ctx.env.set(stmt.varName, {
      kind: "number",
      ...(loopVarIsInt ? { isInteger: true } : {}),
    });
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
      const sign = signFromNumber(value);
      return {
        tag: "NumberLiteral",
        value,
        jitType: {
          kind: "number",
          exact: value,
          ...(sign ? { sign } : {}),
          ...(Number.isInteger(value) ? { isInteger: true } : {}),
        },
      };
    }

    case "ImagUnit":
      return {
        tag: "ImagLiteral",
        jitType: { kind: "complex_or_number", pureImaginary: true },
      };

    case "Ident": {
      // When a slice alias is read as a whole value (not indexed), we
      // materialize it into a real tensor via a helper call. This lets
      // patterns like `rx = A(1,:); r2 = rx .^ 2;` work in the JIT.
      if (ctx.sliceAliases.has(expr.name)) {
        const alias = ctx.sliceAliases.get(expr.name)!;
        return materializeSliceAlias(ctx, alias);
      }

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

      // Matrix multiply: tensor * tensor goes through the mtimes IBuiltin
      // rather than the element-wise Binary path (which only handles scalar
      // or broadcast element-wise ops).
      if (
        !resultType &&
        expr.op === BinaryOperation.Mul &&
        isTensorType(left.jitType) &&
        isTensorType(right.jitType)
      ) {
        const lt = left.jitType as Extract<JitType, { kind: "tensor" }>;
        const rt = right.jitType as Extract<JitType, { kind: "tensor" }>;
        // Infer output shape from input shapes when known: (M×K) * (K×N) → (M×N)
        const outShape =
          lt.shape && rt.shape && lt.shape.length === 2 && rt.shape.length === 2
            ? [lt.shape[0], rt.shape[1]]
            : undefined;
        const isComplex = (lt.isComplex || rt.isComplex) ?? false;
        ctx._hasTensorOps = true;
        return {
          tag: "Call",
          name: "__mtimes",
          args: [left, right],
          jitType: {
            kind: "tensor",
            isComplex,
            ...(outShape ? { shape: outShape } : {}),
            ndim: 2,
          },
        };
      }

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
      // Stage 11 vertical-concat-growth fast path: `[base; value]` where
      // `base` is a real tensor (empty or column vector) and `value` is
      // a numeric scalar. Mirrors the chunkie `it = [it; i]` grow-a-list
      // pattern. Must run before the generic TensorLiteral path because
      // the latter rejects any non-scalar row element.
      if (
        expr.rows.length === 2 &&
        expr.rows[0].length === 1 &&
        expr.rows[1].length === 1
      ) {
        const base = lowerExpr(ctx, expr.rows[0][0]);
        if (
          base &&
          base.jitType.kind === "tensor" &&
          base.jitType.isComplex === false
        ) {
          const value = lowerExpr(ctx, expr.rows[1][0]);
          if (
            value &&
            (value.jitType.kind === "number" ||
              value.jitType.kind === "boolean")
          ) {
            ctx._hasTensorOps = true;
            return {
              tag: "VConcatGrow",
              base,
              value,
              jitType: {
                kind: "tensor",
                isComplex: false,
                shape: [-1, 1],
              },
            };
          }
        }
      }

      // Horizontal concat fast path: [a, b] (single row with 2+ elements)
      // where at least one element is a tensor/unknown (NOT string/char).
      // Emits a runtime helper call. Handles the flagself pattern
      // [c{idx}, scalar] → row vector growth.
      if (expr.rows.length === 1 && expr.rows[0].length >= 2) {
        const loweredElems = expr.rows[0].map(e => lowerExpr(ctx, e));
        if (loweredElems.every(e => e !== null)) {
          const elems = loweredElems as JitExpr[];
          // Only match numeric-ish elements (number, boolean, tensor, unknown)
          // — NOT strings or chars (those use char-concat semantics).
          const isNumericIsh = (k: string) =>
            k === "number" ||
            k === "boolean" ||
            k === "complex_or_number" ||
            k === "tensor" ||
            k === "unknown";
          if (
            elems.every(e => isNumericIsh(e.jitType.kind)) &&
            elems.some(
              e => e.jitType.kind === "tensor" || e.jitType.kind === "unknown"
            )
          ) {
            ctx._hasTensorOps = true;
            return {
              tag: "Call",
              name: "__horzcat",
              args: elems,
              jitType: { kind: "tensor", isComplex: false },
            };
          }
        }
      }

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
      // Marker call `assert_jit_compiled()` in expression position is
      // elided to a literal 1. The job of the marker is to fail when the
      // surrounding loop body bails — reaching this point means lowering
      // is succeeding, so just substitute a constant.
      if (
        expr.name === "assert_jit_compiled" &&
        expr.args.length === 0 &&
        !ctx.env.has(expr.name)
      ) {
        return {
          tag: "NumberLiteral",
          value: 1,
          jitType: { kind: "number", exact: 1, sign: "positive" },
        };
      }

      // If the name is a function handle variable, emit an indirect call
      // instead of treating it as indexing. This enables JIT compilation of
      // loops that call function handles (e.g. kern(srcinfo, targinfo) in
      // chunkie's adapgausskerneval).
      const varType = ctx.env.get(expr.name);
      if (varType && varType.kind === "function_handle") {
        const args = expr.args.map(a => lowerExpr(ctx, a));
        if (args.some(a => a === null)) return null;
        const loweredArgs = args as JitExpr[];

        // Determine return type by probing the function handle at JIT
        // compile time: call it once with the actual argument values and
        // inspect the result type. If the probe can't determine a type,
        // bail — we won't guess.
        if (!ctx.interp) return null;
        const returnType = probeFuncHandleReturnType(
          ctx.interp,
          expr.name,
          loweredArgs
        );
        if (!returnType) return null;

        ctx._hasTensorOps = true;
        return {
          tag: "FuncHandleCall",
          name: expr.name,
          args: loweredArgs,
          jitType: returnType,
        };
      }

      // Stage 21: range slice read `src(a:b)` — parser may emit
      // FuncCall when `src` is a variable. Match BEFORE lowerIndexExpr
      // so the Range isn't rejected by the all-scalar check.
      if (varType && expr.args.length === 1 && expr.args[0].type === "Range") {
        const result = tryLowerRangeSliceRead(ctx, expr.name, expr.args[0]);
        if (result) return result;
      }

      // If the name is a known variable, treat as indexing (MATLAB ambiguity)
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

      // Fold the function-call form `and(a, b)` / `or(a, b)` / `not(a)` to
      // the operator form when both args are simple numeric scalars. This
      // mirrors what `&&` / `||` / `~` already lower to (a JS Binary/Unary
      // node), avoiding the per-iter `$h.ib_and(...)` helper hop. The fold
      // is only safe for plain numeric/boolean operands — complex truthiness
      // doesn't match JS truthiness, so for `complex_or_number` we fall
      // through to the IBuiltin path. Variable shadowing is already handled
      // above; we trust no JIT-able workspace function shadows these
      // builtins.
      if (
        (expr.name === "and" || expr.name === "or") &&
        expr.args.length === 2
      ) {
        const left = lowerExpr(ctx, expr.args[0]);
        if (
          left &&
          (left.jitType.kind === "number" || left.jitType.kind === "boolean")
        ) {
          const right = lowerExpr(ctx, expr.args[1]);
          if (
            right &&
            (right.jitType.kind === "number" ||
              right.jitType.kind === "boolean")
          ) {
            return {
              tag: "Binary",
              op:
                expr.name === "and"
                  ? BinaryOperation.AndAnd
                  : BinaryOperation.OrOr,
              left,
              right,
              jitType: { kind: "boolean" },
            };
          }
        }
      }
      if (expr.name === "not" && expr.args.length === 1) {
        const operand = lowerExpr(ctx, expr.args[0]);
        if (
          operand &&
          (operand.jitType.kind === "number" ||
            operand.jitType.kind === "boolean")
        ) {
          return {
            tag: "Unary",
            op: UnaryOperation.Not,
            operand,
            jitType: { kind: "boolean" },
          };
        }
      }

      // Fold `bsxfun(@op, a, b)` where @op is a known arithmetic operator
      // into a direct call to the runtime's broadcasting-aware arithmetic
      // helpers (mSub, mElemDiv, etc.). These handle shape broadcasting
      // correctly, unlike the JIT's element-wise tSub/tDiv which require
      // same-shape operands.
      if (
        expr.name === "bsxfun" &&
        expr.args.length === 3 &&
        expr.args[0].type === "FuncHandle"
      ) {
        const bsxfunHelperMap: Record<string, string> = {
          minus: "__mSub",
          plus: "__mAdd",
          rdivide: "__mElemDiv",
          // ldivide not yet supported
          times: "__mElemMul",
          power: "__mElemPow",
        };
        const helperName = bsxfunHelperMap[expr.args[0].name];
        if (helperName) {
          const left = lowerExpr(ctx, expr.args[1]);
          if (!left) return null;
          const right = lowerExpr(ctx, expr.args[2]);
          if (!right) return null;
          // Compute broadcast result type
          const binOp =
            expr.args[0].name === "minus"
              ? BinaryOperation.Sub
              : expr.args[0].name === "plus"
                ? BinaryOperation.Add
                : expr.args[0].name === "rdivide"
                  ? BinaryOperation.ElemDiv
                  : expr.args[0].name === "ldivide"
                    ? BinaryOperation.ElemLeftDiv
                    : expr.args[0].name === "times"
                      ? BinaryOperation.ElemMul
                      : BinaryOperation.ElemPow;
          const resultType = binaryResultType(
            binOp,
            left.jitType,
            right.jitType
          );
          if (!resultType || resultType.kind === "unknown") return null;
          ctx._hasTensorOps = true;
          return {
            tag: "Call",
            name: helperName,
            args: [left, right],
            jitType: resultType,
          };
        }
      }

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
      // Stage 21: range slice read `src(a:b)` on a real-tensor base
      // returns a fresh column-vector tensor. Must match BEFORE
      // `lowerExpr(expr.base)` so the Range expression isn't lowered
      // as a standalone tensor.
      if (
        expr.base.type === "Ident" &&
        expr.indices.length === 1 &&
        expr.indices[0].type === "Range"
      ) {
        const result = tryLowerRangeSliceRead(
          ctx,
          expr.base.name,
          expr.indices[0]
        );
        if (result) return result;
      }
      const base = lowerExpr(ctx, expr.base);
      if (!base) return null;
      return lowerIndexExpr(ctx, { base, indices: expr.indices });
    }

    case "IndexCell": {
      // Cell array scalar read: c{i}
      if (expr.base.type !== "Ident") return null;
      const cellType = ctx.env.get(expr.base.name);
      if (!cellType || cellType.kind !== "cell") return null;
      if (expr.indices.length !== 1) return null;
      const cellIdx = lowerExpr(ctx, expr.indices[0]);
      if (!cellIdx) return null;
      if (
        cellIdx.jitType.kind !== "number" &&
        cellIdx.jitType.kind !== "boolean"
      )
        return null;
      ctx._hasTensorOps = true;
      // Result type is unknown — the cell element could be any type.
      // Downstream operations that need a specific type (e.g. horzcat)
      // handle this via runtime dispatch in the helper.
      return {
        tag: "Call",
        name: "__cellRead",
        args: [
          { tag: "Var", name: expr.base.name, jitType: cellType },
          cellIdx,
        ],
        jitType: { kind: "unknown" },
      };
    }

    case "Member": {
      // Stage 13: chained struct array member read `T.nodes(i).leaf`.
      // The parser produces this shape in read position as
      //   Member(MethodCall(Ident(T), "nodes", [i]), "leaf")
      // — the middle node is MethodCall, not Index, because the `.` +
      // ident + `(` sequence is parsed as a method-call postfix.
      if (
        expr.base.type === "MethodCall" &&
        expr.base.base.type === "Ident" &&
        expr.base.args.length === 1
      ) {
        const structVarName = expr.base.base.name;
        const structArrayFieldName = expr.base.name;
        const leafFieldName = expr.name;
        const structType = ctx.env.get(structVarName);
        if (structType && structType.kind === "struct" && structType.fields) {
          const arrayFieldType = structType.fields[structArrayFieldName];
          if (
            arrayFieldType &&
            arrayFieldType.kind === "struct_array" &&
            arrayFieldType.elemFields
          ) {
            const leafType = arrayFieldType.elemFields[leafFieldName];
            // Accept scalar numeric fields (read to a scalar local or
            // used inline) or real-tensor fields (assigned to a local
            // which the existing hoist-refresh path picks up).
            const leafOk =
              leafType &&
              (isNumericScalarType(leafType) ||
                (leafType.kind === "tensor" && leafType.isComplex !== true));
            if (leafOk) {
              const idx = lowerExpr(ctx, expr.base.args[0]);
              if (
                idx &&
                (idx.jitType.kind === "number" ||
                  idx.jitType.kind === "boolean")
              ) {
                if (leafType.kind === "tensor") {
                  ctx._hasTensorOps = true;
                }
                return {
                  tag: "StructArrayMemberRead",
                  structVarName,
                  structArrayFieldName,
                  indexExpr: idx,
                  leafFieldName,
                  jitType: leafType,
                };
              }
            }
          }
        }
      }

      // Stage 12: scalar struct field read `s.f` where `s` is an Ident
      // whose type in the env is a struct with a statically-known scalar
      // field. Lowered to a `MemberRead` IR node; codegen hoists each
      // unique `(baseName, fieldName)` pair as a local alias at function
      // entry.
      //
      // The base is restricted to a plain Ident (no chained `a.b.c` yet)
      // and the field type must be a scalar numeric type. Class instances
      // aren't handled because field access may dispatch to a user-defined
      // getter method.
      if (expr.base.type !== "Ident") return null;
      const baseName = expr.base.name;
      const baseType = ctx.env.get(baseName);
      if (!baseType) return null;
      if (baseType.kind !== "struct") return null;
      if (!baseType.fields) return null;
      const fieldType = baseType.fields[expr.name];
      if (!fieldType) return null;
      if (!isNumericScalarType(fieldType)) return null;
      return {
        tag: "MemberRead",
        baseName,
        fieldName: expr.name,
        jitType: fieldType,
      };
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
    indices.push(lowered);
  }
  if (indices.length === 0) return null;

  // Check if all indices are scalar
  const allScalar = indices.every(
    i => i.jitType.kind === "number" || i.jitType.kind === "boolean"
  );

  if (allScalar) {
    // Scalar indexing — result is a scalar
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

  // Tensor indexing: base(tensorIdx) — single tensor index into a tensor base
  // Result is a tensor of the same complexity as the base.
  if (
    indices.length === 1 &&
    indices[0].jitType.kind === "tensor" &&
    base.jitType.kind === "tensor"
  ) {
    const isComplex = base.jitType.isComplex === true;
    ctx._hasTensorOps = true;
    return {
      tag: "Call",
      name: "__tensorIndex",
      args: [base, indices[0]],
      jitType: { kind: "tensor", isComplex },
    };
  }

  return null;
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
    if (!calleeResult) {
      // Stage 24 soft-bail: the callee's body has constructs the JIT
      // can't lower (tensor arithmetic, matrix multiply, bsxfun with a
      // function-handle arg, etc.). Rather than bail the enclosing
      // loop, probe the return type by actually invoking the function
      // once with representative args, then emit a UserDispatchCall
      // that goes through `rt.dispatch` at runtime. The outer loop
      // still JITs — only the callee runs interpreted.
      //
      // Guard: skip the probe+dispatch path for callees whose bodies
      // use caller-aware or frame-sensitive builtins (evalin,
      // assignin, inputname, dbstack, nargin/nargout without arg,
      // etc.). These resolve relative to the MATLAB call stack; a
      // probe call at compile time runs with a different stack than
      // the real runtime call, and going through $h.callUserFunc at
      // runtime may not reproduce the semantics the user expects.
      if (callerAwareBuiltinInBody(calleeFn.body)) {
        return null;
      }
      const returnType = probeUserFuncReturnType(
        interp,
        calleeFn.name,
        loweredArgs
      );
      if (!returnType) return null;
      ctx._hasTensorOps = true;
      return {
        tag: "UserDispatchCall",
        name: calleeFn.name,
        args: loweredArgs,
        jitType: returnType,
      };
    }

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

  // If any argument is unknown, it could be a class instance at runtime,
  // and class methods take priority over builtins in MATLAB. Allow the
  // call only if `resolve` succeeds with a non-unknown output type —
  // this lets builtins like `ismember` work with cell-read results while
  // still bailing for builtins that can't determine the output type.

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

/**
 * Probe a function handle's return type at JIT compile time.
 *
 * Calls the function handle once with representative argument values and
 * inspects the result type via `inferJitType`. This is safe for pure
 * numerical functions (the vast majority of function handles in numerical
 * MATLAB code). At runtime, every call verifies the actual return type
 * matches — a mismatch triggers a bail to the interpreter.
 *
 * If the probe fails for any reason, returns null and the caller bails
 * (falls back to interpretation for the whole loop).
 */
function probeFuncHandleReturnType(
  interp: Interpreter,
  fnName: string,
  loweredArgs: JitExpr[]
): JitType | null {
  try {
    const fnVal = interp.env.get(fnName);
    if (!fnVal || !isRuntimeFunction(fnVal as RuntimeValue)) return null;
    const fn = fnVal as import("../../runtime/types.js").RuntimeFunction;

    // Only probe function handles that have a direct JS closure — these
    // are anonymous functions and named function references. Builtins
    // that require full interpreter dispatch are too expensive to probe.
    if (!fn.jsFn) return null;

    // Collect argument values for the probe. For each lowered arg:
    // - Var: use the actual value from the env, or synthesize a
    //   representative value from its JIT type (handles loop variables
    //   that don't exist in the env yet at JIT compile time)
    // - NumberLiteral: use the literal value
    // - Other: can't cheaply evaluate, bail
    const argVals: unknown[] = [];
    for (const arg of loweredArgs) {
      if (arg.tag === "NumberLiteral") {
        argVals.push(arg.value);
      } else if (arg.tag === "Var") {
        const val = interp.env.get(arg.name);
        if (val !== undefined) {
          argVals.push(val);
        } else {
          // Variable not in env (e.g. loop iterator before loop starts).
          // Synthesize a representative value from its JIT type.
          const rep = representativeValue(arg.jitType);
          if (rep === undefined) return null;
          argVals.push(rep);
        }
      } else {
        return null;
      }
    }

    // Call the function handle once to determine its return type
    const result = fn.jsFnExpectsNargout
      ? fn.jsFn(1, ...argVals)
      : fn.jsFn(...argVals);
    const resultType = inferJitType(result);
    // Don't accept unknown — that would make downstream lowering bail anyway
    if (resultType.kind === "unknown") return null;
    return resultType;
  } catch {
    // Probe failed (function errored) — bail
    return null;
  }
}

/**
 * Stage 24 safety guard: returns true if the function body references
 * any builtin that reads from or writes to the caller's workspace or
 * the MATLAB call stack. Those builtins (evalin, assignin, inputname,
 * dbstack, …) can't be probed safely at JIT compile time because the
 * probe's call stack differs from the real runtime stack, and they
 * may not survive a round-trip through `$h.callUserFunc` depending on
 * how the runtime resolves the caller frame.
 */
const CALLER_AWARE_BUILTINS = new Set<string>([
  "evalin",
  "assignin",
  "inputname",
  "dbstack",
  "dbstop",
  "keyboard",
  "input",
]);

function callerAwareBuiltinInBody(body: Stmt[]): boolean {
  const visitExpr = (e: Expr): boolean => {
    if (!e) return false;
    if (e.type === "FuncCall" && CALLER_AWARE_BUILTINS.has(e.name)) return true;
    switch (e.type) {
      case "Binary":
        return visitExpr(e.left) || visitExpr(e.right);
      case "Unary":
        return visitExpr(e.operand);
      case "FuncCall":
        return e.args.some(visitExpr);
      case "Index":
      case "IndexCell":
        return visitExpr(e.base) || e.indices.some(visitExpr);
      case "Member":
        return visitExpr(e.base);
      case "MethodCall":
        return visitExpr(e.base) || e.args.some(visitExpr);
      case "Range":
        return (
          visitExpr(e.start) ||
          (e.step ? visitExpr(e.step) : false) ||
          visitExpr(e.end)
        );
      case "Tensor":
      case "Cell":
        return e.rows.some(row => row.some(visitExpr));
      case "AnonFunc":
        return visitExpr(e.body);
      default:
        return false;
    }
  };
  const visitStmts = (stmts: Stmt[]): boolean => {
    for (const s of stmts) {
      switch (s.type) {
        case "Assign":
        case "AssignLValue":
        case "ExprStmt":
        case "MultiAssign":
          if (visitExpr(s.expr)) return true;
          break;
        case "If":
          if (visitExpr(s.cond)) return true;
          if (visitStmts(s.thenBody)) return true;
          for (const eib of s.elseifBlocks) {
            if (visitExpr(eib.cond)) return true;
            if (visitStmts(eib.body)) return true;
          }
          if (s.elseBody && visitStmts(s.elseBody)) return true;
          break;
        case "For":
          if (visitExpr(s.expr)) return true;
          if (visitStmts(s.body)) return true;
          break;
        case "While":
          if (visitExpr(s.cond)) return true;
          if (visitStmts(s.body)) return true;
          break;
        case "TryCatch":
          if (visitStmts(s.tryBody)) return true;
          if (visitStmts(s.catchBody)) return true;
          break;
        case "Switch":
          if (visitExpr(s.expr)) return true;
          for (const c of s.cases) {
            if (visitExpr(c.value)) return true;
            if (visitStmts(c.body)) return true;
          }
          if (s.otherwise && visitStmts(s.otherwise)) return true;
          break;
        default:
          break;
      }
    }
    return false;
  };
  return visitStmts(body);
}

/**
 * Stage 24: probe a user function's return type by invoking it once
 * through `rt.dispatch` with representative argument values. Mirrors
 * `probeFuncHandleReturnType` but for named user functions. Called
 * when `lowerFunction` fails on the callee's body — we still want
 * the outer loop to JIT via a UserDispatchCall.
 *
 * Args are mapped to runtime values:
 *   - Var with a value in the current env: use that value.
 *   - Var with no env value (e.g. loop iterator before loop starts):
 *     synthesize a representative from its JIT type.
 *   - NumberLiteral: use the literal value.
 *   - Other exprs: bail — we can't cheaply evaluate.
 *
 * The probe call may have side effects (persistent-var init, etc.),
 * so it's wrapped in try/catch; any failure bails the probe.
 */
function probeUserFuncReturnType(
  interp: Interpreter,
  fnName: string,
  loweredArgs: JitExpr[]
): JitType | null {
  try {
    const argVals: unknown[] = [];
    for (const arg of loweredArgs) {
      if (arg.tag === "NumberLiteral") {
        argVals.push(arg.value);
      } else if (arg.tag === "Var") {
        const val = interp.env.get(arg.name);
        if (val !== undefined) {
          argVals.push(val);
        } else {
          const rep = representativeValue(arg.jitType);
          if (rep === undefined) return null;
          argVals.push(rep);
        }
      } else {
        return null;
      }
    }
    const result = interp.rt.dispatch(fnName, 1, argVals);
    const resultType = inferJitType(result);
    if (resultType.kind === "unknown") return null;
    return resultType;
  } catch {
    return null;
  }
}

/** Create a representative runtime value for a JIT type, for probing. */
function representativeValue(t: JitType): unknown | undefined {
  switch (t.kind) {
    case "number":
      return t.exact ?? 1;
    case "boolean":
      return true;
    case "complex_or_number":
      return 1;
    default:
      // For tensors, structs, etc. we can't cheaply synthesize a value
      // that would be meaningful to an arbitrary function handle.
      return undefined;
  }
}
