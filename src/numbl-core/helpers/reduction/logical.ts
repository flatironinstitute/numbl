/**
 * Logical reduction builtins: any, all, xor.
 */

import {
  RuntimeValue,
  RTV,
  toNumber,
  RuntimeError,
} from "../../runtime/index.js";
import { ItemType } from "../../lowering/itemTypes.js";
import { register, builtinSingle } from "../registry.js";
import {
  FloatXArray,
  isRuntimeChar,
  isRuntimeComplexNumber,
  isRuntimeLogical,
  isRuntimeNumber,
  isRuntimeSparseMatrix,
  isRuntimeString,
  isRuntimeTensor,
  type RuntimeTensor,
} from "../../runtime/types.js";
import { getBroadcastShape, broadcastIterate } from "../arithmetic.js";
import { rstr } from "../../runtime/runtime.js";
import {
  firstReduceDim,
  scanLogical,
  logicalAlongDim,
  sparseAnyAll,
} from "../reduction-helpers.js";

export function registerLogical(): void {
  /** Factory for any/all logical reductions. */
  const makeAnyAll = (name: string, mode: "any" | "all") => {
    const anyAllCheck = (
      argTypes: ItemType[],
      nargout: number
    ): { outputTypes: ItemType[] } | null => {
      if (nargout !== 1) return null;
      if (argTypes.length === 1) {
        return { outputTypes: [{ kind: "Boolean" }] };
      }
      if (argTypes.length === 2) {
        const arg2 = argTypes[1];
        if (arg2.kind === "Char" || arg2.kind === "String") {
          return { outputTypes: [{ kind: "Boolean" }] };
        }
        return {
          outputTypes: [{ kind: "Tensor" }],
        };
      }
      return null;
    };

    const scalarLogical = (v: RuntimeValue): RuntimeValue | null => {
      if (isRuntimeNumber(v)) return RTV.logical(v !== 0);
      if (isRuntimeLogical(v)) return RTV.logical(v);
      if (isRuntimeComplexNumber(v))
        return RTV.logical(v.re !== 0 || v.im !== 0);
      return null;
    };

    return {
      check: anyAllCheck,
      apply: (args: RuntimeValue[]) => {
        if (args.length < 1)
          throw new RuntimeError(`${name} requires at least 1 argument`);
        const v = args[0];

        // Sparse matrix handling
        if (isRuntimeSparseMatrix(v)) {
          const dim = args.length >= 2 ? Math.round(toNumber(args[1])) : 1;
          return sparseAnyAll(v, dim, mode);
        }

        if (args.length === 1) {
          const scalar = scalarLogical(v);
          if (scalar !== null) return scalar;
          if (isRuntimeTensor(v)) {
            if (v.data.length === 0) return RTV.logical(mode === "all");
            const d = firstReduceDim(v.shape);
            if (d === 0) {
              return RTV.logical(scanLogical(v.data, v.imag, mode));
            }
            return logicalAlongDim(v, d, mode);
          }
          throw new RuntimeError(
            `${name}: argument must be numeric or logical`
          );
        }

        const arg2 = args[1];

        // any/all(A, 'all') — reduce over all elements to a scalar
        if (
          (isRuntimeString(arg2) || isRuntimeChar(arg2)) &&
          rstr(arg2).toLowerCase() === "all"
        ) {
          const scalar = scalarLogical(v);
          if (scalar !== null) return scalar;
          if (isRuntimeTensor(v))
            return RTV.logical(scanLogical(v.data, v.imag, mode));
          throw new RuntimeError(
            `${name}: argument must be numeric or logical`
          );
        }

        // any/all(A, dim) or any/all(A, vecdim)
        const scalar = scalarLogical(v);
        if (scalar !== null) return scalar;
        if (isRuntimeTensor(v)) {
          if (isRuntimeNumber(arg2)) {
            return logicalAlongDim(v, Math.round(arg2), mode);
          }
          if (isRuntimeTensor(arg2)) {
            const dims = Array.from(arg2.data).map(d => Math.round(d));
            let result: RuntimeValue = v;
            for (const dim of dims) {
              if (isRuntimeTensor(result)) {
                result = logicalAlongDim(result, dim, mode);
              }
            }
            return result;
          }
        }
        throw new RuntimeError(`${name}: invalid arguments`);
      },
    };
  };

  register("any", [makeAnyAll("any", "any")]);
  register("all", [makeAnyAll("all", "all")]);

  // ── xor ──────────────────────────────────────────────────────────────

  register(
    "xor",
    builtinSingle(
      args => {
        if (args.length !== 2)
          throw new RuntimeError("xor requires 2 arguments");
        const a = args[0];
        const b = args[1];
        const aIsT = isRuntimeTensor(a);
        const bIsT = isRuntimeTensor(b);
        if (!aIsT && !bIsT) {
          const aVal = isRuntimeLogical(a) ? a : toNumber(a) !== 0;
          const bVal = isRuntimeLogical(b) ? b : toNumber(b) !== 0;
          return RTV.logical(aVal !== bVal);
        }
        // Element-wise xor for tensors
        const aScalar = !aIsT ? (toNumber(a) !== 0 ? 1 : 0) : 0;
        const bScalar = !bIsT ? (toNumber(b) !== 0 ? 1 : 0) : 0;
        if (aIsT && bIsT) {
          if (a.data.length !== b.data.length) {
            const outShape = getBroadcastShape(a.shape, b.shape);
            if (!outShape)
              throw new RuntimeError("xor: incompatible array sizes");
            const n = outShape.reduce((p, c) => p * c, 1);
            const out = new FloatXArray(n);
            broadcastIterate(a.shape, b.shape, outShape, (ai, bi, oi) => {
              out[oi] = (a.data[ai] !== 0) !== (b.data[bi] !== 0) ? 1 : 0;
            });
            const result = RTV.tensor(out, outShape);
            result._isLogical = true;
            return result;
          }
          const n = a.data.length;
          const out = new FloatXArray(n);
          for (let i = 0; i < n; i++) {
            out[i] = (a.data[i] !== 0) !== (b.data[i] !== 0) ? 1 : 0;
          }
          const result = RTV.tensor(out, a.shape);
          result._isLogical = true;
          return result;
        }
        // One tensor, one scalar
        const t = aIsT ? a : (b as RuntimeTensor);
        const s = aIsT ? bScalar : aScalar;
        const n = t.data.length;
        const out = new FloatXArray(n);
        for (let i = 0; i < n; i++) {
          out[i] = (t.data[i] !== 0) !== (s !== 0) ? 1 : 0;
        }
        const result = RTV.tensor(out, t.shape);
        result._isLogical = true;
        return result;
      },
      { outputType: { kind: "Boolean" } }
    )
  );
}
