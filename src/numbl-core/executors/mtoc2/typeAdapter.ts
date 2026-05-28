/**
 * Adapter: numbl `JitType` → mtoc2 `Type`.
 *
 * Called by the mtoc2 JIT executor at `propose()` time. Numbl observes
 * each argument's `JitType` from the live runtime value (via
 * `inferJitType`); the adapter translates it into the matching mtoc2
 * Type so `compileSpec` can specialize on the same shape information.
 *
 * Two design rules:
 *
 * 1. **Always strip exact.** Runtime-observed scalars must NOT shard
 *    mtoc2's spec cache by value. We never propagate `exact` through
 *    the adapter — `compileSpec` calls `withoutExact` again
 *    defensively, but doing it here makes the spec key wider one step
 *    earlier.
 * 2. **Reject unsupported kinds via `null`.** The executor's
 *    `propose()` treats `null` from this adapter as "decline this
 *    JIT proposal" — the dispatcher falls through to the next
 *    executor (eventually the interpreter). First-cut coverage is
 *    scalars + dense real/complex tensors + strings + chars. Structs,
 *    cells, class instances, struct arrays, sparse, dictionaries, and
 *    function handles return `null`; they can be added later.
 */

import type { JitType } from "../../jitTypes.js";
import {
  scalarDouble,
  scalarComplex,
  scalarLogical,
  tensorDouble,
  tensorComplex,
  tensorDoubleFromDims,
  tensorComplexFromDims,
  type Type,
  type DimInfo,
} from "../../mtoc2/index.js";

/** Translate a numbl `JitType` to the matching mtoc2 `Type`.
 *  Returns `null` for kinds the mtoc2 JIT path doesn't accept (yet). */
export function jitTypeToMtoc2Type(jt: JitType): Type | null {
  switch (jt.kind) {
    case "number":
      return scalarDouble("unknown");
    case "boolean":
      return scalarLogical();
    case "complex_or_number":
      // The runtime value may be real or complex; specialize for the
      // wider case. A real value passed in will work as a complex with
      // imag=0; the value adapter handles the boundary copy.
      return scalarComplex();
    case "tensor": {
      // Logical tensors are treated as real-double tensors at the
      // mtoc2 type level — mtoc2 has no public logical-tensor factory,
      // and 0/1-valued doubles round-trip correctly through arithmetic.
      if (jt.shape !== undefined && jt.shape.every(d => d >= 0)) {
        return jt.isComplex ? tensorComplex(jt.shape) : tensorDouble(jt.shape);
      }
      // Shape partly or entirely unknown — fall back to per-axis dims.
      const ndim = jt.shape?.length ?? jt.ndim ?? 2;
      const dims: DimInfo[] = jt.shape
        ? jt.shape.map(d =>
            d >= 0
              ? { kind: "exact" as const, value: d }
              : { kind: "unknown" as const }
          )
        : Array.from({ length: ndim }, () => ({ kind: "unknown" as const }));
      return jt.isComplex
        ? tensorComplexFromDims(dims)
        : tensorDoubleFromDims(dims);
    }
    case "string":
      return { kind: "String" };
    case "char":
      return { kind: "Char" };
    // Kinds the JIT path doesn't accept yet — fall through to interp.
    case "struct":
    case "struct_array":
    case "class_instance":
    case "sparse_matrix":
    case "cell":
    case "dictionary":
    case "function_handle":
    case "unknown":
      return null;
  }
}
