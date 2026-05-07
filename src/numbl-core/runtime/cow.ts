/**
 * Refcount-driven copy-on-write helpers.
 *
 * The lvalue write path uses these to make sure the chain from the env
 * root to the leaf is uniquely owned before mutation: at each level, if
 * the container is shared (`isShared`), it's cloned and the clone is
 * rebound in the parent. After the chain is unique, in-place mutation
 * at the leaf is observable only to the LHS variable.
 */

import {
  RuntimeTensor,
  RuntimeCell,
  RuntimeStruct,
  RuntimeClassInstance,
  RuntimeStructArray,
  isRuntimeTensor,
  isRuntimeCell,
  isRuntimeStruct,
  isRuntimeClassInstance,
  isRuntimeStructArray,
  type RuntimeValue,
} from "./types.js";
import { isShared, type RefcountRuntime } from "./refcount.js";
import { allocFloat64Array } from "../executors/jsJit/helpers/alloc.js";

/** Return a fresh wrapper that shadows `v`'s contents but has rc=0
 *  (modulo scope adoption from the constructor). The new wrapper holds
 *  its own copy of the underlying buffers (for tensors) or its own
 *  fields/data array (for containers). Refs to child values are
 *  shared with the original — child wrappers get incref'd by the new
 *  container's constructor. Caller is responsible for binding the
 *  fresh wrapper into the parent slot.
 *
 *  Returns the original value unchanged for kinds that don't need to
 *  be copied (handle-class instances, primitives) — the caller can
 *  `=== v` test if the copy was a no-op. */
export function cowCopy(v: RuntimeValue): RuntimeValue {
  if (isRuntimeTensor(v)) {
    const dataCopy = allocFloat64Array(v.data);
    const imagCopy =
      v.imag !== undefined ? allocFloat64Array(v.imag) : undefined;
    return new RuntimeTensor(dataCopy, [...v.shape], imagCopy, v._isLogical);
  }
  if (isRuntimeStruct(v)) {
    return new RuntimeStruct(new Map(v.fields));
  }
  if (isRuntimeCell(v)) {
    return new RuntimeCell(v.data.slice(), [...v.shape]);
  }
  if (isRuntimeClassInstance(v)) {
    if (v.isHandleClass) {
      // Handle classes have reference semantics — never copied.
      return v;
    }
    return new RuntimeClassInstance(
      v.className,
      new Map(v.fields),
      false,
      v._builtinData
    );
  }
  if (isRuntimeStructArray(v)) {
    return new RuntimeStructArray(v.fieldNames.slice(), v.elements.slice());
  }
  // Other kinds (RuntimeChar, RuntimeComplexNumber, RuntimeFunction,
  // ...) aren't expected to appear in lvalue chains as containers
  // that can be mutated through. Return as-is.
  return v;
}

/** If `v` is shared (rc-driven), return a fresh non-shared copy;
 *  otherwise return `v` unchanged. The new wrapper has refcount 0
 *  (or 1 if the active scope adopts it via the constructor); the
 *  caller must bind it into the parent slot to keep it alive.
 *
 *  Returns the original `v` for handle-class instances and primitives —
 *  callers can `result === v` test to detect a no-op. */
export function cowIfShared(v: RuntimeValue): RuntimeValue {
  if (!isShared(v)) return v;
  return cowCopy(v);
}

// Re-export for convenient access from the runtime surface.
export { isShared } from "./refcount.js";
export type { RefcountRuntime };
