/**
 * `size` builtin.
 *
 * Three forms supported in mtoc2 v1:
 *
 *   - `size(A)`         → 1×ndim row tensor of dim sizes.
 *   - `size(A, k)`      → scalar (k-th dim, 1-based).
 *   - `[d1, d2, ...]    → one scalar per requested output. Each output
 *     = size(A)`          slot i (0-based) returns `A.dims[i]` when
 *                         i < ndim and 1 otherwise — matches numbl,
 *                         which does NOT collapse trailing dims into
 *                         the last output (unlike MATLAB).
 *
 * Multi-output combined with the `dim` arg is rejected — numbl
 * silently drops the extra outputs in that combo (leaves them
 * unset), and matching that behavior would leak undefined values
 * into mtoc2's typed env.
 */

import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import {
  isMultiElement,
  isNumeric,
  isScalar,
  scalarDouble,
  signFromNumber,
  tensorDouble,
  typeToString,
} from "../../../lowering/types.js";
import type { NumericType } from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import { exactDouble } from "../_shared.js";
import type { RuntimeTensor, RuntimeValue } from "../../../runtime/value.js";
import { isTensor } from "../../../runtime/value.js";
import {
  mtoc2_tensor_from_row,
  mtoc2_tensor_size_row,
} from "../../runtime/snippets.gen.js";

export const size: Builtin = {
  name: "size",
  transfer(argTypes, nargout) {
    if (argTypes.length < 1 || argTypes.length > 2) {
      throw new TypeError(`'size' expects 1..2 arg(s), got ${argTypes.length}`);
    }
    if (nargout < 1) {
      throw new UnsupportedConstruct(
        `'size' requires at least one output (got nargout=${nargout})`
      );
    }
    const a = argTypes[0];
    if (!isNumeric(a)) {
      throw new TypeError(
        `'size' first arg must be numeric (got ${typeToString(a)})`
      );
    }

    if (argTypes.length === 2) {
      if (nargout !== 1) {
        // Numbl quietly drops extra outputs here, leaving them
        // unassigned. Rather than mirror that footgun, reject the
        // combo with a clear message — users should call size with no
        // dim arg if they want per-output dims.
        throw new UnsupportedConstruct(
          `'size(A, k)' is single-output only; drop the 'k' arg and ` +
            `use '[d1, d2, ...] = size(A)' for per-output dims`
        );
      }
      const k = argTypes[1];
      if (!isNumeric(k) || k.isComplex || !isScalar(k)) {
        throw new TypeError(
          `'size' second arg must be a real scalar (got ${typeToString(k)})`
        );
      }
      const kv = exactDouble(k);
      if (kv !== undefined) {
        if (!Number.isFinite(kv) || !Number.isInteger(kv) || kv < 1) {
          throw new TypeError(
            `'size' dim argument must be a positive integer (got ${kv})`
          );
        }
        if (a.shape !== undefined) {
          const dimSize = kv <= a.shape.length ? a.shape[kv - 1] : 1;
          return [scalarDouble(signFromNumber(dimSize), dimSize)];
        }
      }
      return [scalarDouble("nonneg")];
    }

    const ndim = a.dims.length;
    if (nargout >= 2) {
      // One scalar per output slot. Each slot i is A.dims[i] when
      // i < ndim, else 1. When the static shape is known, propagate
      // the exact value (and sign); otherwise nonneg.
      const out: NumericType[] = [];
      for (let i = 0; i < nargout; i++) {
        if (a.shape !== undefined) {
          const v = i < ndim ? a.shape[i] : 1;
          out.push(scalarDouble(signFromNumber(v), v));
        } else if (i < ndim && a.dims[i].kind === "exact") {
          const v = (a.dims[i] as { kind: "exact"; value: number }).value;
          out.push(scalarDouble(signFromNumber(v), v));
        } else if (i >= ndim) {
          out.push(scalarDouble("positive", 1));
        } else {
          out.push(scalarDouble("nonneg"));
        }
      }
      return out;
    }

    if (a.shape !== undefined) {
      const data = new Float64Array(ndim);
      for (let i = 0; i < ndim; i++) data[i] = a.shape[i];
      return [tensorDouble([1, ndim], data)];
    }
    return [tensorDouble([1, ndim])];
  },
  emitC({ argsC, argTypes, nargout, outArgsC, useRuntime }) {
    const a = argTypes[0] as NumericType;
    // Multi-output: emit a comma-expression that writes each requested
    // dim through its out-pointer. Each slot i is `(double)A.dims[i]`
    // when i < ndim, else `1.0`. The framework's wrapping `;` makes
    // the comma-expression a complete statement.
    if (nargout >= 2) {
      const outs = outArgsC ?? [];
      const ndim = a.dims.length;
      const writes: string[] = [];
      for (let i = 0; i < nargout; i++) {
        const rhs = i < ndim ? `(double)${argsC[0]}.dims[${i}]` : `1.0`;
        writes.push(`(*${outs[i]} = ${rhs})`);
      }
      return `((void)(${writes.join(", ")}))`;
    }
    useRuntime("mtoc2_tensor_size_row");
    useRuntime("mtoc2_tensor_from_row");
    if (argTypes.length === 2) {
      const kv = exactDouble(argTypes[1] as NumericType);
      if (kv !== undefined) {
        const ndim = a.dims.length;
        if (kv <= ndim) {
          return `((double)${argsC[0]}.dims[${kv - 1}])`;
        }
        return `1.0`;
      }
      return (
        `({ long _mtoc2_k = (long)(${argsC[1]}); ` +
        `(double)(_mtoc2_k >= 1 && _mtoc2_k <= ${argsC[0]}.ndim ? ${argsC[0]}.dims[_mtoc2_k - 1] : 1); })`
      );
    }

    if (!isMultiElement(a)) {
      return `mtoc2_tensor_from_row((double[]){1.0, 1.0}, 2)`;
    }
    return `mtoc2_tensor_size_row(${argsC[0]})`;
  },
  emitJs({ argsJs, argTypes, nargout, useRuntime }) {
    const a = argTypes[0] as NumericType;
    // Multi-output: return a JS array literal `[d1, d2, ...]` the
    // framework destructures into the out-targets. Each slot i is
    // `<arg>.shape[i] ?? 1`.
    if (nargout >= 2) {
      const ndim = a.dims.length;
      const dims: string[] = [];
      for (let i = 0; i < nargout; i++) {
        dims.push(i < ndim ? `${argsJs[0]}.shape[${i}]` : `1`);
      }
      return `[${dims.join(", ")}]`;
    }
    useRuntime("mtoc2_tensor_size_row");
    useRuntime("mtoc2_tensor_from_row");
    if (argTypes.length === 2) {
      const kv = exactDouble(argTypes[1] as NumericType);
      if (kv !== undefined) {
        const ndim = a.dims.length;
        if (kv <= ndim) return `(${argsJs[0]}.shape[${kv - 1}])`;
        return `1`;
      }
      return (
        `((k => (k >= 1 && k <= ${argsJs[0]}.shape.length ` +
        `? ${argsJs[0]}.shape[k - 1] : 1))(Math.trunc(${argsJs[1]})))`
      );
    }
    if (!isMultiElement(a)) {
      return `mtoc2_tensor_from_row([1, 1], 2)`;
    }
    return `mtoc2_tensor_size_row(${argsJs[0]})`;
  },
  call({ args, argTypes, nargout }) {
    const t = args[0] as RuntimeValue;
    // Multi-output: one scalar per output slot. Scalars (non-tensors)
    // report ndim=2 with [1, 1]; tensors report their actual shape;
    // overflow slots get 1.
    if (nargout >= 2) {
      const shape = isTensor(t) ? t.shape : [1, 1];
      const out: RuntimeValue[] = [];
      for (let i = 0; i < nargout; i++) {
        out.push(i < shape.length ? shape[i] : 1);
      }
      return out;
    }
    if (argTypes.length === 2) {
      const kv =
        typeof args[1] === "number" ? args[1] : Number(args[1] as object);
      const k = Math.trunc(kv);
      if (k < 1) {
        throw new TypeError(
          `'size' dim argument must be a positive integer (got ${k})`
        );
      }
      if (isTensor(t)) return [k <= t.shape.length ? t.shape[k - 1] : 1];
      return [1];
    }
    if (!isTensor(t)) {
      // Scalar: pad to a 2-element row [1, 1].
      return [mtoc2_tensor_from_row([1, 1], 2) as unknown as RuntimeTensor];
    }
    return [mtoc2_tensor_size_row(t) as unknown as RuntimeTensor];
  },
};
