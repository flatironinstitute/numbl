/**
 * "Effectively real" helpers.
 *
 * numbl's JIT cannot always prove at compile time that the result of an
 * operation is real (e.g. `sqrt(1 - x.^2/2)` — the argument's sign is not
 * statically provable, so `sqrt` lifts to the complex path). The result is
 * a complex-typed tensor whose imaginary lane is entirely zero — an
 * artifact of static typing, not a value the user asked to be complex.
 *
 * Value-sensitive builtins (`isreal`, `min`, `max`, `sort`, …) must treat
 * such a tensor exactly as they would a real one: otherwise `min`/`max`
 * silently switch to magnitude ordering and `isreal` reports `false`,
 * diverging from the interpreter (`--opt 0`, which never creates the
 * artifact) and from MATLAB on real data. These helpers detect the
 * all-zero imaginary lane and drop it so the real code path runs.
 */

import { RTV } from "../runtime/index.js";
import type { RuntimeValue } from "../runtime/index.js";
import { isRuntimeTensor, type RuntimeTensor } from "../runtime/types.js";

/** True when `imag` is absent or every element is exactly zero. */
export function imagAllZero(imag: ArrayLike<number> | undefined): boolean {
  if (!imag) return true;
  for (let i = 0; i < imag.length; i++) {
    if (imag[i] !== 0) return false;
  }
  return true;
}

/** If `t` carries an all-zero imaginary lane, return a real view sharing
 *  its data; otherwise return `t` unchanged. */
export function stripZeroImagTensor(t: RuntimeTensor): RuntimeTensor {
  if (t.imag && imagAllZero(t.imag)) {
    const out = RTV.tensor(t.data, t.shape);
    if (t._isLogical) out._isLogical = true;
    return out;
  }
  return t;
}

/** Tensor-aware variant for the generic `RuntimeValue` channel. */
export function stripZeroImagValue(v: RuntimeValue): RuntimeValue {
  if (isRuntimeTensor(v)) return stripZeroImagTensor(v);
  return v;
}
