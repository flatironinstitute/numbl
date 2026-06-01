/**
 * Shared validation helper + slice-arg predicate for the four index
 * lowering helpers (lowerIndexLoad / lowerIndexStore / lowerIndexSlice
 * / lowerIndexSliceStore).
 *
 * `isSliceArg` lets the dispatchers in `lower.ts` choose between the
 * scalar and slice paths without each call site re-discriminating on
 * `Expr` kinds.
 *
 * `resolveIndexBase` performs the common preamble shared by all four
 * helpers: env lookup, numeric/multi-element check, arity check,
 * builds the base `Var` IR node.
 *
 * Adapted from mtoc's `src/lowering/indexResolve.ts`, with the
 * char-tensor / complex / scalar-base branches dropped — mtoc2 v1
 * only indexes real-double multi-element tensors.
 */

import type { Expr, LValue, Span } from "../parser/index.js";
import { unwindMemberChain } from "../parser/astUtils.js";
import { TypeError, UnsupportedConstruct } from "./errors.js";
import type { IRExpr } from "./ir.js";
import {
  fieldType,
  isMultiElement,
  isNumeric,
  isScalar,
  type NumericType,
  type Type,
  typeToString,
} from "./types.js";
import type { Lowerer } from "./lower.js";

/** Operation label that selects operation-specific message text. */
export type IndexOperation = "read" | "write" | "sliceRead" | "sliceWrite";

/** True when an AST expression node is a range or bare colon — the
 *  dispatcher uses this predicate to decide between scalar (IndexLoad /
 *  IndexStore) and slice (IndexSlice / IndexSliceStore) paths. */
export function isSliceArg(a: Expr): boolean {
  return a.type === "Range" || a.type === "Colon";
}

interface ResolveOptions {
  /** Span of the base identifier node — used for the Var IR node and
   *  the "not in scope" diagnostic. Defaults to the outer index span. */
  baseSpan?: Span;
  /** "internal" when the dispatcher already verified the binding exists
   *  (a missing one would be a lowerer bug); "user-facing" when the
   *  caller is a statement-level dispatcher and the variable may
   *  genuinely be undefined. */
  notInScope: "internal" | "user-facing";
  operation: IndexOperation;
}

/** Validate and resolve a member-rooted index lvalue like
 *  `obj.field(i, j) = rhs`. Walks the Member chain to extract the
 *  root Ident, the field path (outermost → innermost), and the leaf
 *  field's NumericType; validates that the leaf is a multi-element
 *  double tensor and that the index arity matches the leaf's ndim.
 *
 *  Returns the same shape as `resolveIndexBase` plus the field path
 *  and leaf type, so the caller can build an `IndexStore` /
 *  `IndexSliceStore` whose `base` Var still names the OWNING root
 *  (struct / class instance) — the codegen path joins `base.cName`
 *  with `fieldPath` to address the field slot. */
export function resolveMemberRootedIndexBase(
  this: Lowerer,
  lvalue: Extract<LValue, { type: "Index" }>,
  span: Span,
  operation: IndexOperation
): {
  rootVar: Extract<IRExpr, { kind: "Var" }>;
  fieldPath: string[];
  leafTy: NumericType;
  slotCName: string;
} {
  if (lvalue.base.type !== "Member") {
    throw new UnsupportedConstruct(
      `internal: resolveMemberRootedIndexBase called with non-Member base ` +
        `(got ${lvalue.base.type})`,
      span
    );
  }
  const unwound = unwindMemberChain(lvalue.base);
  if (unwound === null) {
    throw new UnsupportedConstruct(
      `${opPrefix(operation)} requires a root variable`,
      lvalue.base.span
    );
  }
  const { root, fields: fieldPath } = unwound;
  const rootName = root.name;
  const rootEntry = this.envLookup(rootName);
  if (rootEntry === undefined) {
    throw new TypeError(`use of undefined variable '${rootName}'`, root.span);
  }
  if (rootEntry.ty.kind !== "Struct" && rootEntry.ty.kind !== "Class") {
    throw new UnsupportedConstruct(
      `${opPrefix(operation)} of '${rootName}.${fieldPath.join(".")}' but ` +
        `'${rootName}' is ${typeToString(rootEntry.ty)}; only struct and ` +
        `class fields can be indexed`,
      lvalue.base.span
    );
  }
  let stepTy: Type = rootEntry.ty;
  for (let i = 0; i < fieldPath.length; i++) {
    const fname = fieldPath[i];
    const ft = fieldType(stepTy, fname);
    if (ft === undefined) {
      // If the field is a `Dependent` property of the current class
      // step, surface a more specific error — the field is intended
      // to be read via its `get.X` accessor; an indexed write through
      // it isn't supported in v1 (would need to invoke the getter,
      // splice the rhs into the result, and call the setter).
      if (stepTy.kind === "Class") {
        const reg = this.classReg(stepTy.className);
        if (reg?.dependentProperties.has(fname)) {
          throw new UnsupportedConstruct(
            `${opPrefix(operation)} of dependent property ` +
              `'${rootName}.${fieldPath.slice(0, i + 1).join(".")}' ` +
              `via indexing is not supported in v1`,
            lvalue.base.span
          );
        }
      }
      throw new TypeError(
        `'${rootName}.${fieldPath.slice(0, i + 1).join(".")}': no such ` +
          `field on type ${typeToString(stepTy)}`,
        lvalue.base.span
      );
    }
    stepTy = ft;
  }
  if (!isNumeric(stepTy)) {
    throw new UnsupportedConstruct(
      `${opPrefix(operation)} of '${rootName}.${fieldPath.join(".")}' but ` +
        `the field has type ${typeToString(stepTy)} (only numeric fields ` +
        `can be indexed)`,
      lvalue.base.span
    );
  }
  if (!isMultiElement(stepTy)) {
    throw new UnsupportedConstruct(
      notMultiElementMsg(
        operation,
        `${rootName}.${fieldPath.join(".")}`,
        stepTy
      ),
      span
    );
  }
  if (stepTy.elem !== "double") {
    throw new UnsupportedConstruct(
      `${opPrefix(operation)} into ${typeToString(stepTy)} is not yet supported`,
      span
    );
  }
  const ndim = stepTy.dims.length;
  const argCount = lvalue.indices.length;
  if (argCount === 0 && (operation === "read" || operation === "write")) {
    throw new UnsupportedConstruct(
      operation === "read"
        ? `indexing '${rootName}.${fieldPath.join(".")}' requires at least one index`
        : `indexed write requires at least one index`,
      span
    );
  }
  if (argCount !== 1 && argCount !== ndim) {
    throw new UnsupportedConstruct(arityMsg(operation, argCount, ndim), span);
  }
  const slotCName = `${rootEntry.cName}.${fieldPath.join(".")}`;
  const rootVar: Extract<IRExpr, { kind: "Var" }> = {
    kind: "Var",
    name: rootName,
    cName: rootEntry.cName,
    ty: rootEntry.ty,
    span: lvalue.base.span,
  };
  return { rootVar, fieldPath, leafTy: stepTy, slotCName };
}

/** Unified resolution result for an indexed-write lvalue base —
 *  whether the lvalue's base is a bare `Ident` or a `Member` chain.
 *
 *  `base` always names the OWNING root variable (for liveness +
 *  ANF book-keeping). `baseCName` is the slot path codegen targets
 *  — equal to `base.cName` for bare-Ident, or `<root>.<field>...`
 *  for member-rooted. `baseTy` is the leaf NumericType (the field's
 *  type for member-rooted; identical to `base.ty` otherwise) so
 *  the offset / complex-lane / arity checks downstream don't care
 *  which form the LHS took. `fieldPath` / `leafTy` are populated
 *  only in the member-rooted case so the caller can stamp them
 *  onto the `IndexStore` / `IndexSliceStore` IR node. */
export interface IndexLvalueBase {
  base: Extract<IRExpr, { kind: "Var" }>;
  baseTy: NumericType;
  baseCName: string;
  fieldPath?: string[];
  leafTy?: NumericType;
  /** Source-level display name for error messages — `name` for
   *  bare-Ident bases, `root.f1.f2...` for member-rooted bases. */
  displayName: string;
}

/** Resolve an indexed-write lvalue's base — either a bare `Ident`
 *  or a `Member` chain — to the unified `IndexLvalueBase` shape.
 *  Single chokepoint shared by `lowerIndexStore` and
 *  `lowerIndexSliceStore` so the Ident-vs-Member dispatch lives in
 *  one place. Rejects any other base kind with a clear error. */
export function resolveIndexLvalueBase(
  this: Lowerer,
  lvalue: Extract<LValue, { type: "Index" }>,
  span: Span,
  operation: IndexOperation
): IndexLvalueBase {
  if (lvalue.base.type === "Member") {
    const m = resolveMemberRootedIndexBase.call(this, lvalue, span, operation);
    return {
      base: m.rootVar,
      baseTy: m.leafTy,
      baseCName: m.slotCName,
      fieldPath: m.fieldPath,
      leafTy: m.leafTy,
      displayName: `${m.rootVar.name}.${m.fieldPath.join(".")}`,
    };
  }
  if (lvalue.base.type === "Ident") {
    const name = lvalue.base.name;
    const r = resolveIndexBase.call(this, name, lvalue.indices.length, span, {
      baseSpan: lvalue.base.span,
      notInScope: "user-facing",
      operation,
    });
    return {
      base: r.base,
      baseTy: r.baseTy,
      baseCName: r.baseCName,
      displayName: name,
    };
  }
  throw new UnsupportedConstruct(
    `indexed assignment requires a simple variable or member chain on the left ` +
      `(got ${lvalue.base.type})`,
    span
  );
}

/** Validate and resolve the base variable for an index operation. */
export function resolveIndexBase(
  this: Lowerer,
  name: string,
  argCount: number,
  span: Span,
  opts: ResolveOptions
): {
  baseTy: NumericType;
  baseCName: string;
  base: Extract<IRExpr, { kind: "Var" }>;
} {
  const { baseSpan = span, notInScope, operation } = opts;

  const looked = this.envLookup(name);
  if (looked === undefined) {
    if (notInScope === "internal") {
      const fn = operation === "read" ? "lowerIndexLoad" : "lowerIndexSlice";
      throw new UnsupportedConstruct(
        `internal: ${fn} called for '${name}' which is not in scope`,
        span
      );
    }
    throw new TypeError(`use of undefined variable '${name}'`, baseSpan);
  }

  if (!isNumeric(looked.ty)) {
    throw new UnsupportedConstruct(
      `${opPrefix(operation)} into ${typeToString(looked.ty)} is not yet supported`,
      span
    );
  }

  // The "read" path emits a dedicated "scalar variable" message before
  // the generic multi-element check so the diagnostic names the variable.
  if (operation === "read" && isScalar(looked.ty)) {
    throw new UnsupportedConstruct(
      `indexing into a scalar variable '${name}' is not yet supported`,
      span
    );
  }

  if (!isMultiElement(looked.ty)) {
    throw new UnsupportedConstruct(
      notMultiElementMsg(operation, name, looked.ty),
      span
    );
  }

  // Both real-double and complex-double tensors are supported.
  // Logical / char tensors stay rejected until they're plumbed.
  if (looked.ty.elem !== "double") {
    throw new UnsupportedConstruct(
      `${opPrefix(operation)} into ${typeToString(looked.ty)} is not yet supported`,
      span
    );
  }

  const ndim = looked.ty.dims.length;

  if (argCount === 0 && (operation === "read" || operation === "write")) {
    const msg =
      operation === "read"
        ? `indexing '${name}' requires at least one index`
        : `indexed write requires at least one index`;
    throw new UnsupportedConstruct(msg, span);
  }

  if (argCount !== 1 && argCount !== ndim) {
    throw new UnsupportedConstruct(arityMsg(operation, argCount, ndim), span);
  }

  const baseCName = looked.cName;
  const base: Extract<IRExpr, { kind: "Var" }> = {
    kind: "Var",
    name,
    cName: baseCName,
    ty: looked.ty,
    span: baseSpan,
  };
  return { baseTy: looked.ty, baseCName, base };
}

// ── Message helpers ─────────────────────────────────────────────────────

function opPrefix(op: IndexOperation): string {
  switch (op) {
    case "read":
      return "indexing";
    case "write":
      return "indexed write";
    case "sliceRead":
      return "range/colon indexing";
    case "sliceWrite":
      return "range/colon indexed write";
  }
}

function notMultiElementMsg(
  op: IndexOperation,
  name: string,
  baseTy: NumericType
): string {
  switch (op) {
    case "read":
      return `cannot index variable '${name}' with type ${typeToString(baseTy)}`;
    case "write":
      return `indexed write requires a multi-element tensor (got ${typeToString(baseTy)})`;
    case "sliceRead":
      return `range/colon indexing requires a multi-element tensor (got ${typeToString(baseTy)})`;
    case "sliceWrite":
      return `range/colon indexed write requires a multi-element tensor (got ${typeToString(baseTy)})`;
  }
}

function arityMsg(op: IndexOperation, argCount: number, ndim: number): string {
  switch (op) {
    case "read":
      return (
        `${argCount}-index access into a ${ndim}-D tensor is not yet ` +
        `supported (use 1 linear index or ${ndim} per-axis indices)`
      );
    case "write":
      return (
        `${argCount}-index write into a ${ndim}-D tensor is ` +
        `not yet supported (use 1 linear index or ${ndim} per-axis indices)`
      );
    case "sliceRead":
      return (
        `range/colon indexing of a ${ndim}-D tensor requires either 1 slot ` +
        `(linear) or ${ndim} slots (one per axis); got ${argCount}`
      );
    case "sliceWrite":
      return (
        `range/colon indexed write into a ${ndim}-D tensor requires either 1 ` +
        `slot (linear) or ${ndim} slots (one per axis); got ${argCount}`
      );
  }
}
