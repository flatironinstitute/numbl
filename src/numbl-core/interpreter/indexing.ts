/**
 * Indexing reads (`a(i)`, `c{i}`) and assignment targets (`a(i) = v`,
 * `s.f.g = v`, `c{i}.f(j) = v`).
 */

import type { Expr, LValue } from "../parser/types.js";
import type { RuntimeValue } from "../runtime/types.js";
import {
  isRuntimeClassInstance,
  isRuntimeClassInstanceArray,
  isRuntimeStruct,
  isRuntimeStructArray,
} from "../runtime/types.js";
import { RTV } from "../runtime/constructors.js";
import { ensureRuntimeValue } from "../runtime/runtimeHelpers.js";
import { RuntimeError } from "../runtime/error.js";
import { allocFloat64Array } from "../runtime/alloc.js";
import { cowCopy } from "../runtime/cow.js";
import { isShared } from "../runtime/refcount.js";

import type { Interpreter } from "./interpreter.js";
import { evalExpr } from "./evalExpressions.js";

const emptyTensor = () => RTV.tensor(allocFloat64Array(0), [0, 0]);

/** Whether indexing `base` skips its class's overloaded subsref/subsasgn:
 *  MATLAB uses built-in indexing on an instance of the class whose method
 *  is running. */
export function bypassesOverload(interp: Interpreter, base: unknown): boolean {
  const className = interp.frame.className;
  if (!className || base == null) return false;
  const rv = ensureRuntimeValue(base);
  return isRuntimeClassInstance(rv) && rv.className === className;
}

// ── Reads ────────────────────────────────────────────────────────────────

export function evalIndex(
  interp: Interpreter,
  expr: Extract<Expr, { type: "Index" }>
): unknown {
  let base = evalExpr(interp, expr.base);
  // A brace/member chain can produce a comma-list (array); indexing is
  // only valid when it holds exactly one value
  if (Array.isArray(base)) {
    if (base.length === 1) base = base[0];
    else
      throw new RuntimeError(
        `Expression produced ${base.length} values where one was expected`
      );
  }
  const indices = evalIndicesWithEnd(interp, base, expr.indices);
  return interp.rt.index(base, indices, 1, bypassesOverload(interp, base));
}

export function evalIndexCell(
  interp: Interpreter,
  expr: Extract<Expr, { type: "IndexCell" }>
): unknown {
  const base = evalExpr(interp, expr.base);
  const indices = evalIndicesWithEnd(interp, base, expr.indices);
  return interp.rt.indexCell(base, indices);
}

/** Evaluate subscripts of `base`, with `end` in subscript k meaning the
 *  extent of `base` along dimension k. */
export function evalIndicesWithEnd(
  interp: Interpreter,
  base: unknown,
  indexExprs: Expr[]
): unknown[] {
  const numIndices = indexExprs.length;
  return indexExprs.map((idx, dimIndex) => {
    interp.endContextStack.push({ base, dimIndex, numIndices });
    try {
      return evalExpr(interp, idx);
    } finally {
      interp.endContextStack.pop();
    }
  });
}

// ── Assignment targets ───────────────────────────────────────────────────

export function assignLValue(
  interp: Interpreter,
  lv: LValue,
  value: RuntimeValue
): void {
  const env = interp.frame.env;
  switch (lv.type) {
    case "Var":
      env.set(lv.name, value);
      break;

    case "Ignore":
      break;

    case "Index": {
      // Use undefined for uninitialized Ident so indexStore can detect it
      // (e.g. h(1) = classInstance when h is an uninitialized output var)
      const base =
        lv.base.type === "Ident"
          ? env.get(lv.base.name)
          : evalLValueBase(interp, lv.base, emptyTensor());
      const indices = evalIndicesWithEnd(
        interp,
        base ?? emptyTensor(),
        lv.indices
      );
      const result = interp.rt.indexStore(
        base,
        indices,
        value,
        bypassesOverload(interp, base)
      );
      writeLValueBase(interp, lv.base, ensureRuntimeValue(result));
      break;
    }

    case "IndexCell": {
      const base =
        lv.base.type === "Ident"
          ? (env.get(lv.base.name) ?? RTV.cell([], [0, 0]))
          : evalLValueBase(interp, lv.base, RTV.cell([], [0, 0]));
      const indices = evalIndicesWithEnd(interp, base, lv.indices);
      const result = interp.rt.indexCellStore(base, indices, value);
      writeLValueBase(interp, lv.base, ensureRuntimeValue(result));
      break;
    }

    case "Member": {
      // Walk up the Member chain to find the first non-Member node and collect names
      const names: string[] = [lv.name];
      let cursor: Expr = lv.base;
      while (cursor.type === "Member") {
        names.unshift(cursor.name);
        cursor = cursor.base;
      }
      // cursor is now the root of the member chain (Ident, Index, etc.)
      const rootBase =
        cursor.type === "Ident"
          ? (env.get(cursor.name) ?? RTV.struct({}))
          : evalLValueBase(interp, cursor, RTV.struct({}));
      const rootRv = ensureRuntimeValue(rootBase);
      if (isRuntimeClassInstance(rootRv)) {
        // Use memberChainAssign which routes through subsasgn if needed
        const result = interp.rt.memberChainAssign(rootBase, names, value);
        writeLValueBase(interp, cursor, ensureRuntimeValue(result));
      } else {
        // Non-class: use direct field set with store-back chain
        const base = evalLValueBase(interp, lv.base, RTV.struct({}));
        const result = interp.rt.setMemberReturn(base, lv.name, value);
        writeLValueBase(interp, lv.base, ensureRuntimeValue(result));
      }
      break;
    }

    case "MemberDynamic": {
      const base = evalLValueBase(interp, lv.base, RTV.struct({}));
      const nameVal = evalExpr(interp, lv.nameExpr);
      const result = interp.rt.setMemberDynamicReturn(base, nameVal, value);
      writeLValueBase(interp, lv.base, ensureRuntimeValue(result));
      break;
    }
  }
}

/** Store `value` back into the location `base` denotes, rebuilding each
 *  enclosing container up to the variable. */
function writeLValueBase(
  interp: Interpreter,
  base: Expr,
  value: RuntimeValue
): void {
  if (base.type === "Ident") {
    interp.frame.env.set(base.name, value);
  } else if (base.type === "Member") {
    const parentBase = evalLValueBase(interp, base.base, RTV.struct({}));
    const updatedParent = interp.rt.setMemberReturn(
      parentBase,
      base.name,
      value
    );
    writeLValueBase(interp, base.base, ensureRuntimeValue(updatedParent));
  } else if (base.type === "Index") {
    const baseVal = evalLValueBase(interp, base.base, emptyTensor());
    const indices = evalIndicesWithEnd(interp, baseVal, base.indices);
    // Use builtinIndexStore for compound assignment store-back —
    // this bypasses subsasgn for class instances (the compound decomposition
    // already handled the field set; the store-back should use builtin mechanics)
    const result = interp.rt.builtinIndexStore(baseVal, indices, value);
    writeLValueBase(interp, base.base, ensureRuntimeValue(result));
  } else if (base.type === "IndexCell") {
    const baseVal = evalLValueBase(interp, base.base, RTV.cell([], [0, 0]));
    const indices = evalIndicesWithEnd(interp, baseVal, base.indices);
    const result = interp.rt.indexCellStore(baseVal, indices, value);
    writeLValueBase(interp, base.base, ensureRuntimeValue(result));
  }
}

/** Read the current value at an assignment target's base, creating missing
 *  fields from `defaultVal` and copying shared containers on the way down
 *  (so the write that follows does not leak to another holder). */
function evalLValueBase(
  interp: Interpreter,
  base: Expr,
  defaultVal: RuntimeValue
): unknown {
  if (base.type === "Ident") {
    return interp.frame.env.get(base.name) ?? defaultVal;
  }
  if (base.type === "Member") {
    let parentBase = evalLValueBase(interp, base.base, RTV.struct({}));
    let parentRv = ensureRuntimeValue(parentBase);
    // Refcount-driven COW: if the parent is shared with another
    // binding (env, struct field, cell element, …), copy it now and
    // rebind in its location. After this, the chain from env root to
    // `parent` is uniquely owned, so mutations through this slot won't
    // leak to the other holder.
    if (isShared(parentRv)) {
      const cowed = cowCopy(parentRv);
      writeLValueBase(interp, base.base, cowed);
      parentBase = cowed;
      parentRv = cowed;
    }
    // Objects resolve the member themselves (properties, get methods,
    // subsref); their errors propagate.
    const isObject =
      isRuntimeClassInstance(parentRv) || isRuntimeClassInstanceArray(parentRv);
    const hasField =
      (isRuntimeStruct(parentRv) && parentRv.fields.has(base.name)) ||
      (isRuntimeStructArray(parentRv) &&
        parentRv.fieldNames.includes(base.name));
    if (isObject || hasField) {
      return interp.rt.getMember(parentBase, base.name);
    }
    // Field doesn't exist — auto-create using the caller-supplied
    // default. The default reflects what the next lvalue level needs
    // (empty struct for `.member`, empty tensor for `(...)` /
    // `{...}`), so `s.a(3) = 5` initializes `s.a` as an empty tensor
    // that the subsequent indexStore can grow to `[0 0 5]`, while
    // `s.a.b = 5` keeps the existing empty-struct path.
    const newValue = defaultVal;
    const updatedParent = interp.rt.setMemberReturn(
      parentRv,
      base.name,
      newValue
    );
    if (base.base.type === "Ident") {
      interp.frame.env.set(base.base.name, ensureRuntimeValue(updatedParent));
    }
    return newValue;
  }
  if (base.type === "Index") {
    let baseVal = evalLValueBase(interp, base.base, RTV.struct({}));
    let baseRv = ensureRuntimeValue(baseVal);
    if (isShared(baseRv)) {
      const cowed = cowCopy(baseRv);
      writeLValueBase(interp, base.base, cowed);
      baseVal = cowed;
      baseRv = cowed;
    }
    const indices = evalIndicesWithEnd(interp, baseVal, base.indices);
    try {
      return interp.rt.index(
        baseVal,
        indices,
        1,
        bypassesOverload(interp, baseRv)
      );
    } catch (e) {
      // An index past the end grows the array; an object's own subsref
      // errors propagate.
      if (isRuntimeClassInstance(baseRv) || isRuntimeClassInstanceArray(baseRv))
        throw e;
      return defaultVal;
    }
  }
  if (base.type === "IndexCell") {
    let baseVal = evalLValueBase(interp, base.base, RTV.cell([], [0, 0]));
    let baseRv = ensureRuntimeValue(baseVal);
    if (isShared(baseRv)) {
      const cowed = cowCopy(baseRv);
      writeLValueBase(interp, base.base, cowed);
      baseVal = cowed;
      baseRv = cowed;
    }
    const indices = evalIndicesWithEnd(interp, baseVal, base.indices);
    try {
      const result = interp.rt.indexCell(baseVal, indices);
      // A single cell subscript (`c{i}`) yields a one-element comma-list.
      // As an lvalue base (e.g. `c{i}.field = ...`) it must be the bare
      // value, since the caller feeds it straight to ensureRuntimeValue /
      // the member store-back — it can't accept an array. This mirrors the
      // comma-list base unwrap in evalIndex. A literal index already returns
      // the bare value; unwrapping aligns the tensor-index path (e.g. a
      // find()-derived 1×1 index) with it.
      if (Array.isArray(result)) {
        if (result.length === 1) return result[0];
        if (result.length === 0) return defaultVal;
      }
      return result;
    } catch (e) {
      if (isRuntimeClassInstance(baseRv) || isRuntimeClassInstanceArray(baseRv))
        throw e;
      return defaultVal;
    }
  }
  return evalExpr(interp, base);
}
