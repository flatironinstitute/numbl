/**
 * `diff(A)` / `diff(A, 1)` / `diff(A, 1, dim)` — first-order forward
 * difference along the operating axis, matching numbl's `diffOnce`
 * (helpers/reduction/cumulative.ts).
 *
 * Scope (v1): first-order (n == 1) real diff of a non-scalar tensor with
 * a statically-known shape. Declines to the interpreter for:
 *   - n != 1 (the n-fold loop recomputes the default axis each pass and
 *     can flip orientation — numbl's own type/value disagree there);
 *   - the `diff(A, [], dim)` empty-weight form (numbl returns A
 *     unchanged via a NaN loop count — a quirk not worth matching);
 *   - complex input, scalar input, non-exact n/dim, out-of-range dim,
 *     and lattice-only (non-concrete) shapes.
 */

import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import {
  DIM_ONE,
  isNumeric,
  isScalar,
  tensorDouble,
  tensorDoubleFromDims,
  type Type,
} from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import { exactDouble } from "../_shared.js";
import { mtoc2_tensor_diff as jsDiff } from "../../runtime/tensor_ops/tensor_diff.js";
import type { RuntimeTensor } from "../../../runtime/value.js";

/** Result shape of one forward difference along the operating axis,
 *  mirroring numbl's `diffOnce` + RTV.tensor trailing-singleton squeeze
 *  (2-D floor). `dim` is 1-based; `undefined` means the default axis. */
function diffOnceShape(shape: number[], dim: number | undefined): number[] {
  let opDim: number;
  if (dim !== undefined) {
    opDim = dim - 1;
  } else if (shape.length <= 1 || (shape.length === 2 && shape[0] === 1)) {
    opDim = shape.length === 2 && shape[0] === 1 ? 1 : 0;
  } else {
    opDim = 0;
  }
  const dimSize = opDim < shape.length ? shape[opDim] : 1;
  const out = shape.slice();
  if (dimSize <= 1) {
    if (opDim < out.length) out[opDim] = 0;
  } else {
    out[opDim] = dimSize - 1;
  }
  while (out.length > 2 && out[out.length - 1] === 1) out.pop();
  return out;
}

/** Parse + validate args. Returns the (concrete) input shape and the
 *  1-based dim (or undefined for the default axis). Throws/declines on
 *  unsupported forms. */
function planDiff(argTypes: Type[]): {
  shape: number[];
  dim: number | undefined;
} {
  if (argTypes.length < 1 || argTypes.length > 3) {
    throw new TypeError(`'diff' expects 1..3 arg(s), got ${argTypes.length}`);
  }
  const input = argTypes[0];
  if (!isNumeric(input)) {
    throw new TypeError(`'diff' arg must be numeric (got ${input.kind})`);
  }
  if (input.isComplex) {
    throw new UnsupportedConstruct(
      `'diff' on complex input is not JIT-compiled`
    );
  }
  if (input.elem !== "double" && input.elem !== "logical") {
    throw new TypeError(
      `'diff' arg must be double or logical (got ${input.elem})`
    );
  }
  if (isScalar(input)) {
    throw new UnsupportedConstruct(`'diff' on a scalar is not JIT-compiled`);
  }
  if (input.shape === undefined) {
    throw new UnsupportedConstruct(
      `'diff' on a tensor with non-concrete shape is not JIT-compiled`
    );
  }

  // n (arg 1): only the first-order case (absent or exactly 1).
  if (argTypes.length >= 2) {
    const nType = argTypes[1];
    if (!isNumeric(nType) || nType.isComplex || !isScalar(nType)) {
      throw new UnsupportedConstruct(
        `'diff' order arg must be the scalar 1 (higher-order / '[]' decline)`
      );
    }
    const nv = exactDouble(nType);
    if (nv === undefined || nv !== 1) {
      throw new UnsupportedConstruct(
        `'diff' only JIT-compiles first-order (n == 1); got ${nv ?? "non-exact"}`
      );
    }
  }

  // dim (arg 2): a statically-known axis within the input's dims.
  let dim: number | undefined;
  if (argTypes.length >= 3) {
    const dimType = argTypes[2];
    if (!isNumeric(dimType) || dimType.isComplex || !isScalar(dimType)) {
      throw new TypeError(`'diff' dim arg must be a scalar real integer`);
    }
    const dv = exactDouble(dimType);
    if (dv === undefined) {
      throw new UnsupportedConstruct(
        `'diff' dim arg must be a statically-known integer`
      );
    }
    if (!Number.isFinite(dv) || !Number.isInteger(dv) || dv < 1) {
      throw new TypeError(
        `'diff' dim arg must be a positive integer (got ${dv})`
      );
    }
    if (dv > input.shape.length) {
      throw new UnsupportedConstruct(
        `'diff' with dim ${dv} > ndims ${input.shape.length} is not JIT-compiled`
      );
    }
    dim = dv;
  }

  return { shape: input.shape, dim };
}

export const diff: Builtin = {
  name: "diff",
  transfer(argTypes, nargout) {
    if (nargout !== 1) {
      throw new UnsupportedConstruct(
        `'diff' does not support multi-output (nargout=${nargout})`
      );
    }
    const { shape, dim } = planDiff(argTypes);
    const rshape = diffOnceShape(shape, dim);
    // A single-element ([1,1]) result collapses to a scalar in the type
    // system, but scalar indexing (v(1)) isn't JIT-compiled — type it as
    // a 1×? row tensor so it stays a tensor (the kernel returns the 1×1).
    if (rshape.every(d => d === 1)) {
      return [tensorDoubleFromDims([DIM_ONE, { kind: "unknown" }])];
    }
    const out = tensorDouble(rshape);
    out.sign = "unknown";
    return [out];
  },
  emitC({ argsC, argTypes, useRuntime }) {
    useRuntime("mtoc2_tensor_diff");
    const { dim } = planDiff(argTypes);
    return `mtoc2_tensor_diff(${argsC[0]}, ${dim ?? 0})`;
  },
  emitJs({ argsJs, argTypes, useRuntime }) {
    useRuntime("mtoc2_tensor_diff");
    const { dim } = planDiff(argTypes);
    return `mtoc2_tensor_diff(${argsJs[0]}, ${dim ?? 0})`;
  },
  call({ args, argTypes }) {
    const { dim } = planDiff(argTypes);
    return [
      jsDiff(args[0] as RuntimeTensor, dim ?? 0) as unknown as RuntimeTensor,
    ];
  },
};
