/**
 * Statement lowering for the JIT. Paired with jitLowerExpr.ts; the two
 * sides are mutually recursive. jitLower.ts is the orchestrator that ties
 * them together via LowerCtx + lowerFunction.
 */

import type { Expr, Stmt } from "../../../parser/types.js";
import {
  type JitType,
  type JitExpr,
  type JitStmt,
  isScalarType,
  isNumericScalarType,
  isKnownInteger,
} from "../../../jitTypes.js";
import { cloneEnv, mergeEnvs, envsEqual } from "./jitLowerTypes.js";
import { getIBuiltin } from "../../../interpreter/builtins/index.js";
import { offsetToLineFast } from "../../../runtime/error.js";
import type { LowerCtx } from "./jitLower.js";
import {
  lowerExpr,
  lowerIBuiltinCall,
  probeFuncHandleMultiReturnTypes,
} from "./jitLowerExpr.js";
import { JIT_IO_BUILTINS as JIT_VOID_IO_BUILTINS } from "./jitBailSafety.js";

const LOG_CJIT_MISSES =
  typeof process !== "undefined" && !!process.env.NUMBL_LOG_CJIT_MISSES;

/** Returns true/false when the expr is a known constant scalar, else null. */
function constantBool(e: JitExpr): boolean | null {
  if (e.tag === "NumberLiteral" && typeof e.value === "number") {
    return e.value !== 0;
  }
  if (e.jitType.kind === "boolean" && e.jitType.value !== undefined) {
    return e.jitType.value;
  }
  if (e.jitType.kind === "number" && e.jitType.exact !== undefined) {
    return e.jitType.exact !== 0;
  }
  return null;
}

// ── Statement lowering ──────────────────────────────────────────────────

export function lowerStmts(ctx: LowerCtx, stmts: Stmt[]): JitStmt[] | null {
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
  if (LOG_CJIT_MISSES) {
    ctx.lastExprType = `stmt:${stmt.type}`;
    if (stmt.span && ctx.lineTable) {
      ctx.lastExprLine = offsetToLineFast(ctx.lineTable, stmt.span.start);
    }
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
      // Most ExprStmts set `ans` in the env, which the JIT codegen
      // doesn't do — bail. Exceptions:
      //   - `tic;` (no args, no assignment) is a side-effect-only call
      //     that we can JIT. `tic;` parses as either a bare Ident or a
      //     zero-arg FuncCall depending on source; accept both.
      //   - `disp/fprintf/printf/warning(...)` are I/O void calls: emit
      //     as a runtime-dispatched `Call` with an "unknown" jitType.
      //     The bail-safety gate ensures we only keep these if no
      //     mid-execution bail is possible (else the I/O could be
      //     duplicated on interpreter re-run).
      {
        const e = stmt.expr;
        const isTicIdent = e.type === "Ident" && e.name === "tic";
        const isTicCall =
          e.type === "FuncCall" && e.name === "tic" && e.args.length === 0;
        if ((isTicIdent || isTicCall) && !ctx.env.has("tic")) {
          const call: Expr & { type: "FuncCall" } = isTicCall
            ? (e as Expr & { type: "FuncCall" })
            : { type: "FuncCall", name: "tic", args: [], span: e.span };
          const lowered = lowerIBuiltinCall(ctx, call);
          if (lowered) {
            result = [...prefix, { tag: "ExprStmt", expr: lowered }];
            return result;
          }
        }
        if (
          e.type === "FuncCall" &&
          JIT_VOID_IO_BUILTINS.has(e.name) &&
          !ctx.env.has(e.name)
        ) {
          const loweredArgs: JitExpr[] = [];
          for (const arg of e.args) {
            const la = lowerExpr(ctx, arg);
            if (!la) return null;
            loweredArgs.push(la);
          }
          const call: JitExpr = {
            tag: "Call",
            name: e.name,
            args: loweredArgs,
            jitType: { kind: "unknown" },
          };
          result = [...prefix, { tag: "ExprStmt", expr: call }];
          return result;
        }
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
    case "Directive":
      if (stmt.directive === "assert_jit") {
        // Bare assert_jit: JS-JIT lowering succeeded → elide.
        // assert_jit c: under JS-JIT (opt "1") this degrades to a
        // JS-JIT check (since there is no C-JIT path inside JS-JIT
        // bodies); the directive is dropped. Under "e3" the JS-JIT
        // lowering is never reached.
        return prefix;
      }
      // Unknown directives: silently elide in JIT.
      return prefix;
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

  // Single-colon `name = base(:)` is column-vector linearization, not a
  // slice alias bind. Fall through to lowerExpr's `__colonAll` path so
  // shape-mismatched (e.g. 1D-colon on a 2D base) cases lower cleanly
  // instead of hard-bailing.
  if (rawIndices.length === 1 && rawIndices[0].type === "Colon") return null;

  // From here on, any failure is a hard bail — the caller can't fall back
  // to normal Index lowering because Colon isn't supported there.

  // Slice-alias bind is for real tensors only — for complex bases, fall
  // through to normal Index lowering which handles the 2D-colon case via
  // `__extractSlice2d` (preserves the imag part). Returning null (not
  // "bail") triggers the fallthrough.
  if (baseType.isComplex === true) return null;
  // Shape must exist so we can check ndim and colon-dim sizes.
  if (!baseType.shape) return "bail";
  // Range indices aren't supported yet (only bare `:`).
  if (rawIndices.some(idx => idx.type === "Range")) return "bail";
  // Require exact-arity multi-dim indexing.
  if (rawIndices.length !== baseType.shape.length) return "bail";
  // Can't rebind a param or an already-assigned local into a slice alias.
  // Falling through (returning null) lets lowerIndexExpr handle 1D/2D colon
  // slicing via __colonAll / __extractSlice2d when applicable.
  if (ctx.params.has(name)) return null;
  if (ctx.assignedVars.has(name) && !ctx.sliceAliases.has(name)) return null;

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
/**
 * Cell-element-type env update for a literal-index write. Records the
 * rhs type at `idx` so a subsequent literal-index read recovers it
 * (chunkerfunc's `[out{1:3}] = fcurve(ts); r = out{1}; …`). No-op when
 * the rhs has no useful type to track.
 */
function setCellElementType(
  ctx: LowerCtx,
  cellName: string,
  cellType: Extract<JitType, { kind: "cell" }>,
  idx: number,
  rhsType: JitType
): void {
  if (rhsType.kind === "unknown") return;
  if (cellType.elements?.[idx] === rhsType) return;
  ctx.env.set(cellName, {
    kind: "cell",
    ...(cellType.shape ? { shape: cellType.shape } : {}),
    elements: { ...(cellType.elements ?? {}), [idx]: rhsType },
  });
}

/**
 * Drop all per-index element tracking on a cell. Used after a non-
 * literal-index write since the runtime index could touch any slot.
 */
function clearCellElementsTracking(
  ctx: LowerCtx,
  cellName: string,
  cellType: Extract<JitType, { kind: "cell" }>
): void {
  if (!cellType.elements) return;
  ctx.env.set(cellName, {
    kind: "cell",
    ...(cellType.shape ? { shape: cellType.shape } : {}),
  });
}

/**
 * Resolve an Expr to a literal positive-integer index, used for cell
 * literal-index detection. Tries (in order): a Number literal, an
 * Ident whose env type carries an `exact` integer, and finally the
 * live runtime value via `ctx.interp.env`. The runtime fallback is
 * what handles chunkerfunc-style `[out{1:nout}]` where `nout` survives
 * to the loop spec without an `exact` (the lowering pipeline strips
 * `exact` to avoid spec proliferation).
 */
function resolveLiteralInt(ctx: LowerCtx, e: Expr): number | null {
  if (e.type === "Number") {
    const v = parseFloat(e.value);
    return Number.isInteger(v) && v >= 1 ? v : null;
  }
  if (e.type === "Ident") {
    const t = ctx.env.get(e.name);
    if (
      t &&
      t.kind === "number" &&
      t.exact !== undefined &&
      Number.isInteger(t.exact) &&
      t.exact >= 1
    ) {
      return t.exact;
    }
    if (ctx.interp) {
      const v = ctx.interp.env.get(e.name);
      if (typeof v === "number" && Number.isInteger(v) && v >= 1) return v;
    }
  }
  return null;
}

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
    // Per-index element-type tracking. A literal-index write updates
    // the slot's tracked type; a non-literal-index write COULD touch
    // any slot, so we conservatively drop all tracked element types
    // (any of them could now hold the rhs value at runtime). Without
    // this drop, a later literal-index read would use a stale static
    // type and JIT-emit code (e.g. scalar JS `+`) that mishandles the
    // actual runtime value.
    const litIdx = resolveLiteralInt(ctx, lv.indices[0]);
    if (litIdx !== null) {
      setCellElementType(ctx, lv.base.name, cellType, litIdx, rhs.jitType);
    } else {
      clearCellElementsTracking(ctx, lv.base.name, cellType);
    }
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

  // Base must already exist in the type env as a tensor.
  const baseTypeRaw = ctx.env.get(baseName);
  if (!baseTypeRaw || baseTypeRaw.kind !== "tensor") return null;

  // Page-slice write `dst(:, :, k) = rhs` on a 3-D tensor.  Drives
  // chunkie helm2d.green's `grad(:,:,k) = ...` pattern.  Must run BEFORE
  // the `isComplex === true` bail below so the second write
  // (`grad(:,:,2) = ...`) still lowers after the first one promoted
  // grad's env type to complex.
  if (
    lv.indices.length === 3 &&
    lv.indices[0].type === "Colon" &&
    lv.indices[1].type === "Colon" &&
    lv.indices[2].type !== "Colon" &&
    lv.indices[2].type !== "Range"
  ) {
    const pageResult = tryLowerPage3dAssign(
      ctx,
      baseName,
      baseTypeRaw,
      lv.indices[2],
      stmt.expr
    );
    if (pageResult) return pageResult;
  }

  if (baseTypeRaw.isComplex === true) return null;
  const baseType = baseTypeRaw;

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
/** Lower `dst(:, :, k) = rhs` where `dst` is a 3-D tensor and `rhs` is a
 *  2-D tensor expression. If `rhs` is complex but `dst` is real, promote
 *  `dst`'s entry in ctx.env to complex; the runtime helper promotes the
 *  actual storage on write. */
function tryLowerPage3dAssign(
  ctx: LowerCtx,
  baseName: string,
  baseType: JitType,
  pageIdxExpr: Expr,
  rhsExpr: Expr
): JitStmt[] | null {
  if (baseType.kind !== "tensor") return null;
  // Require a known 3-D shape on the base so we know we're writing a page.
  // If shape is undefined the initializer wasn't a zeros-like call with
  // known dims; skip for now (conservative).
  if (!baseType.shape || baseType.shape.length !== 3) return null;

  const pageIndex = lowerExpr(ctx, pageIdxExpr);
  if (!pageIndex) return null;
  if (
    pageIndex.jitType.kind !== "number" &&
    pageIndex.jitType.kind !== "boolean"
  ) {
    return null;
  }

  const value = lowerExpr(ctx, rhsExpr);
  if (!value) return null;
  // RHS must be a tensor. Scalar RHS into a 3-D page would be a broadcast
  // (not supported yet).
  if (value.jitType.kind !== "tensor") return null;

  const rhsComplex = value.jitType.isComplex === true;
  const baseComplex = baseType.isComplex === true;

  // If RHS is complex but base was real-typed, promote the var's env type
  // to complex so downstream reads and the function's output type are right.
  const newBaseType: JitType = {
    kind: "tensor",
    isComplex: baseComplex || rhsComplex,
    shape: baseType.shape,
  };
  if (!baseComplex && rhsComplex) {
    ctx.env.set(baseName, newBaseType);
  }

  ctx.assignedVars.add(baseName);
  ctx.sliceAliases.delete(baseName);
  ctx._hasTensorOps = true;

  return [
    {
      tag: "AssignIndexPage3d",
      baseName,
      baseType: newBaseType,
      pageIndex,
      value,
    },
  ];
}

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
  // Lvalue normalization. Each output slot becomes either a plain Var
  // or a (cell, index) target. Cell-target slots get a synthetic temp
  // var; after the multi-assign runs, a sequence of `cell{idx} = $tmp`
  // writes copies the values into the cell. Two lvalue shapes are
  // supported for cell targets:
  //   (a) per-output IndexCell with a literal positive integer index
  //       — `[out{1}, out{2}, out{3}] = f(...)`.
  //   (b) one IndexCell with a literal `Range(start, end)` index —
  //       `[out{1:3}] = f(...)`. Expanded into N (a)-style slots.
  // The chunkie/chunkerfunc resolve loop uses (b).
  type Slot =
    | { kind: "var"; name: string }
    | { kind: "ignore" }
    | { kind: "cell"; tempName: string; cellName: string; idx: number };
  const slots: Slot[] = [];
  let cellTempCounter = 0;
  for (const lv of stmt.lvalues) {
    if (lv.type === "Var") {
      slots.push({ kind: "var", name: lv.name });
      continue;
    }
    if (lv.type === "Ignore") {
      slots.push({ kind: "ignore" });
      continue;
    }
    if (
      lv.type === "IndexCell" &&
      lv.base.type === "Ident" &&
      lv.indices.length === 1
    ) {
      const cellName = lv.base.name;
      const cellType = ctx.env.get(cellName);
      if (!cellType || cellType.kind !== "cell") return null;
      const idxExpr = lv.indices[0];
      const single = resolveLiteralInt(ctx, idxExpr);
      if (single !== null) {
        slots.push({
          kind: "cell",
          tempName: `$mavc_${cellTempCounter++}`,
          cellName,
          idx: single,
        });
        continue;
      }
      if (idxExpr.type === "Range" && idxExpr.step === null) {
        const a = resolveLiteralInt(ctx, idxExpr.start);
        const b = resolveLiteralInt(ctx, idxExpr.end);
        if (a === null || b === null || b < a) return null;
        for (let k = a; k <= b; k++) {
          slots.push({
            kind: "cell",
            tempName: `$mavc_${cellTempCounter++}`,
            cellName,
            idx: k,
          });
        }
        continue;
      }
      return null;
    }
    return null;
  }
  if (slots.length === 0) return null;
  const nargout = slots.length;
  const names: (string | null)[] = slots.map(s =>
    s.kind === "ignore" ? null : s.kind === "var" ? s.name : s.tempName
  );

  // RHS must be a FuncCall (either IBuiltin or user function)
  const rhs = stmt.expr;
  if (rhs.type !== "FuncCall") return null;

  // Lower the arguments
  const args = rhs.args.map(a => lowerExpr(ctx, a));
  if (args.some(a => a === null)) return null;
  const loweredArgs = args as JitExpr[];
  const argJitTypes = loweredArgs.map(a => a.jitType);

  // Function-handle RHS: `[a, b, c] = fhandle(args...)`. Probe the handle
  // with the requested nargout to discover output types, then emit a
  // MultiAssign with kind="func_handle" so the codegen calls
  // $h.callFuncHandleMulti at runtime. Hot in chunkie-style code where
  // an inner loop binds `fcurve = @(t) starfish(t,...)` and unpacks
  // multiple outputs per iteration.
  // Build the list of cell-write follow-up statements. Each cell-target
  // slot got a synthetic temp Var assigned by the MultiAssign; we now
  // emit `cell{idx} = $tmp` to copy it into the cell. Mirrors the
  // single-write `out{i} = v` lowering in lowerAssignLValue. As a
  // side effect this updates the cell's per-index element types in the
  // env so later literal-index reads see the right type instead of
  // unknown.
  const buildCellWrites = (outputTypes: JitType[]): JitStmt[] => {
    const writes: JitStmt[] = [];
    const cellsTouched = new Set<string>();
    for (let i = 0; i < slots.length; i++) {
      const s = slots[i];
      if (s.kind !== "cell") continue;
      const cellType = ctx.env.get(s.cellName);
      if (!cellType || cellType.kind !== "cell") continue;
      cellsTouched.add(s.cellName);
      const tempType = outputTypes[i];
      // Update env's per-index element type so subsequent writes/reads
      // in this block see the new tracking. Re-read cellType from env
      // before each helper call (prior writes may have updated it).
      setCellElementType(ctx, s.cellName, cellType, s.idx, tempType);
      const updatedCellType = ctx.env.get(s.cellName) as Extract<
        JitType,
        { kind: "cell" }
      >;
      writes.push({
        tag: "Assign",
        name: s.cellName,
        expr: {
          tag: "Call",
          name: "__cellWrite",
          args: [
            { tag: "Var", name: s.cellName, jitType: updatedCellType },
            {
              tag: "NumberLiteral",
              value: s.idx,
              jitType: { kind: "number", exact: s.idx, isInteger: true },
            },
            { tag: "Var", name: s.tempName, jitType: tempType },
          ],
          jitType: updatedCellType,
        },
      });
    }
    for (const c of cellsTouched) {
      ctx.assignedVars.add(c);
      if (!ctx.params.has(c)) ctx.localVars.add(c);
    }
    return writes;
  };

  // Mark slot vars (incl. cell-target temps) as locals + record output
  // types in env. Cell targets keep the cell variable's existing type;
  // the temp's type is the output type from the call.
  const recordOutputTypes = (outputTypes: JitType[]): void => {
    for (let i = 0; i < slots.length; i++) {
      const s = slots[i];
      const t = outputTypes[i];
      if (s.kind === "var") {
        ctx.env.set(s.name, t);
        ctx.assignedVars.add(s.name);
        if (!ctx.params.has(s.name)) ctx.localVars.add(s.name);
      } else if (s.kind === "cell") {
        ctx.env.set(s.tempName, t);
        ctx.assignedVars.add(s.tempName);
        ctx.localVars.add(s.tempName);
      }
    }
  };

  const varType = ctx.interp ? ctx.env.get(rhs.name) : undefined;
  if (varType && varType.kind === "function_handle" && ctx.interp) {
    const types = probeFuncHandleMultiReturnTypes(
      ctx.interp,
      rhs.name,
      loweredArgs,
      nargout
    );
    if (!types) return null;
    recordOutputTypes(types);
    ctx._hasTensorOps = true;
    return [
      {
        tag: "MultiAssign",
        names,
        callName: rhs.name,
        args: loweredArgs,
        outputTypes: types,
        kind: "func_handle",
      },
      ...buildCellWrites(types),
    ];
  }

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

  recordOutputTypes(outputTypes);
  ctx._hasTensorOps = true;

  return [
    {
      tag: "MultiAssign",
      names,
      callName: rhs.name,
      args: loweredArgs,
      outputTypes,
    },
    ...buildCellWrites(outputTypes),
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

  // Dead-branch elimination: if the cond folded to a constant (e.g.
  // `nargout > 1` with nargout inlined to a literal), lower only the live
  // branch so unreachable code can't bail.
  const constBool = constantBool(cond);
  if (constBool === true) {
    const thenBody = lowerStmts(ctx, stmt.thenBody);
    return thenBody ?? null;
  }
  if (constBool === false) {
    for (const eib of stmt.elseifBlocks) {
      const eibCond = lowerExpr(ctx, eib.cond);
      if (!eibCond) return null;
      const eibConst = constantBool(eibCond);
      if (eibConst === true) {
        return lowerStmts(ctx, eib.body) ?? null;
      }
      if (eibConst !== false) {
        // Non-constant elseif with dead then → fall back to normal path
        // rather than half-specializing.
        break;
      }
    }
    if (stmt.elseBody && constBool === false) {
      // Check that all elseifs were also constant-false before taking the else.
      const allElseifFalse = stmt.elseifBlocks.every(eib => {
        const c = lowerExpr(ctx, eib.cond);
        return c !== null && constantBool(c) === false;
      });
      if (allElseifFalse) {
        return lowerStmts(ctx, stmt.elseBody) ?? null;
      }
    }
    // No elseif matched and no else (or else not fully dead): emit nothing.
    if (
      !stmt.elseBody &&
      stmt.elseifBlocks.every(eib => {
        const c = lowerExpr(ctx, eib.cond);
        return c !== null && constantBool(c) === false;
      })
    ) {
      return [];
    }
    // Fall through to normal handling otherwise.
  }

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

  // Statically-empty range elimination. Mirrors dead-branch elimination
  // for `if false`: if the range bounds are both literal numbers and
  // the range is empty, drop the body (and its env-side-effects) so
  // dead bodies can't bail-out the surrounding loop. Hot in
  // chunkerfunc's `for j = nout+1:3 ... end` when nout==3.
  const startLit =
    start.tag === "NumberLiteral" && typeof start.value === "number"
      ? start.value
      : null;
  const endLit =
    end.tag === "NumberLiteral" && typeof end.value === "number"
      ? end.value
      : null;
  const stepLit =
    step === null
      ? 1
      : step.tag === "NumberLiteral" && typeof step.value === "number"
        ? step.value
        : null;
  if (startLit !== null && endLit !== null && stepLit !== null) {
    const empty =
      stepLit === 0 ||
      (stepLit > 0 && startLit > endLit) ||
      (stepLit < 0 && startLit < endLit);
    if (empty) return [];
  }

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
