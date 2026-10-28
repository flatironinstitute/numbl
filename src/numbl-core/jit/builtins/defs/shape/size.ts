/**
 * `size` builtin.
 *
 * Three forms supported in the JIT:
 *
 *   - `size(A)`         → 1×ndim row tensor of dim sizes.
 *   - `size(A, k)`      → scalar (k-th dim, 1-based).
 *   - `[d1, d2, ...]    → one scalar per requested output. Each output
 *     = size(A)`          slot i (0-based) returns `A.dims[i]` when
 *                         i < ndim and 1 otherwise, except the last,
 *                         which is the product of the remaining dims
 *                         (as in MATLAB: `[r, c] = size(zeros(2,3,4))`
 *                         gives c = 12).
 *
 * Multi-output combined with the `dim` arg is rejected — numbl
 * silently drops the extra outputs in that combo (leaves them
 * unset), and matching that behavior would leak undefined values
 * into the JIT's typed env.
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
      // i < ndim, else 1; the last slot is the product of the dims from
      // i on. When those dims are statically known, propagate the exact
      // value (and sign); otherwise nonneg.
      const out: NumericType[] = [];
      for (let i = 0; i < nargout; i++) {
        if (i >= ndim) {
          out.push(scalarDouble("positive", 1));
          continue;
        }
        const last = i === nargout - 1 ? ndim : i + 1;
        let v: number | undefined = 1;
        for (let k = i; k < last && v !== undefined; k++) {
          const d = a.dims[k];
          if (a.shape !== undefined) v *= a.shape[k];
          else if (d.kind === "exact")
            v *= (d as { kind: "exact"; value: number }).value;
          else v = undefined;
        }
        out.push(
          v !== undefined
            ? scalarDouble(signFromNumber(v), v)
            : scalarDouble("nonneg")
        );
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
    // A statically-scalar arg is stored as a bare C `double` with no `.dims`
    // member; a scalar is 1x1, so every dim query is 1. Short-circuit before
    // any `.dims[]` access, which would not compile against a double (e.g.
    // `x = 5; size(x,1)`, or `[5, []]` whose empty drops to the scalar 5).
    if (isScalar(a)) {
      if (nargout >= 2) {
        const outs = outArgsC ?? [];
        const writes = outs.slice(0, nargout).map(o => `(*${o} = 1.0)`);
        return `((void)(${writes.join(", ")}))`;
      }
      if (argTypes.length === 2) return `1.0`;
      useRuntime("jit_tensor_size_row");
      useRuntime("jit_tensor_from_row");
      return `jit_tensor_from_row((double[]){1.0, 1.0}, 2)`;
    }
    // Multi-output: emit a comma-expression that writes each requested
    // dim through its out-pointer. Each slot i is `(double)A.dims[i]`
    // when i < ndim, else `1.0`; the last slot is the product of the
    // remaining dims. The framework's wrapping `;` makes the
    // comma-expression a complete statement.
    if (nargout >= 2) {
      const outs = outArgsC ?? [];
      const ndim = a.dims.length;
      const writes: string[] = [];
      for (let i = 0; i < nargout; i++) {
        const last = i === nargout - 1 ? ndim : i + 1;
        const factors: string[] = [];
        for (let k = i; k < last; k++)
          factors.push(`(double)${argsC[0]}.dims[${k}]`);
        const rhs = factors.length > 0 ? factors.join(" * ") : `1.0`;
        writes.push(`(*${outs[i]} = ${rhs})`);
      }
      return `((void)(${writes.join(", ")}))`;
    }
    useRuntime("jit_tensor_size_row");
    useRuntime("jit_tensor_from_row");
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
        `({ long _jit_k = (long)(${argsC[1]}); ` +
        `(double)(_jit_k >= 1 && _jit_k <= ${argsC[0]}.ndim ? ${argsC[0]}.dims[_jit_k - 1] : 1); })`
      );
    }

    if (!isMultiElement(a)) {
      return `jit_tensor_from_row((double[]){1.0, 1.0}, 2)`;
    }
    return `jit_tensor_size_row(${argsC[0]})`;
  },
  emitJs({ argsJs, argTypes, nargout, useRuntime }) {
    const a = argTypes[0] as NumericType;
    // A statically-scalar arg is a bare JS number with no `.shape`; a scalar
    // is 1x1, so every dim query is 1. Short-circuit before any `.shape[]`
    // access (which would throw on a number and force a needless interp bail).
    if (isScalar(a)) {
      if (nargout >= 2) {
        return `[${Array(nargout).fill("1").join(", ")}]`;
      }
      if (argTypes.length === 2) return `1`;
      useRuntime("jit_tensor_size_row");
      useRuntime("jit_tensor_from_row");
      return `jit_tensor_from_row([1, 1], 2)`;
    }
    // Multi-output: return a JS array literal `[d1, d2, ...]` the
    // framework destructures into the out-targets. Each slot i is
    // `<arg>.shape[i] ?? 1`; the last slot is the product of the
    // remaining dims.
    if (nargout >= 2) {
      const ndim = a.dims.length;
      const dims: string[] = [];
      for (let i = 0; i < nargout; i++) {
        const last = i === nargout - 1 ? ndim : i + 1;
        const factors: string[] = [];
        for (let k = i; k < last; k++)
          factors.push(`(${argsJs[0]}.shape[${k}] ?? 1)`);
        dims.push(factors.length > 0 ? factors.join(" * ") : `1`);
      }
      return `[${dims.join(", ")}]`;
    }
    useRuntime("jit_tensor_size_row");
    useRuntime("jit_tensor_from_row");
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
      return `jit_tensor_from_row([1, 1], 2)`;
    }
    return `jit_tensor_size_row(${argsJs[0]})`;
  },
};
