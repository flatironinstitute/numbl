/**
 * Predicate builtins: isnan, isinf, isfinite, isreal.
 */

import {
  FloatXArray,
  isRuntimeComplexNumber,
  isRuntimeTensor,
} from "../../runtime/types.js";
import type { JitType } from "../jit/jitTypes.js";
import { registerIBuiltin, makeTensor } from "./types.js";

/** Type rule for predicates: any numeric type → produces logical (number nonneg) */
function predicateType(argTypes: JitType[]): JitType[] | null {
  if (argTypes.length !== 1) return null;
  const a = argTypes[0];
  switch (a.kind) {
    case "number":
    case "complex":
      return [{ kind: "number", nonneg: true }];
    case "realTensor":
    case "complexTensor":
      return [{ kind: "realTensor", nonneg: true }];
    default:
      return null;
  }
}

// ── isnan ───────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "isnan",
  typeRule: argTypes => predicateType(argTypes),
  apply: args => {
    const v = args[0];
    if (typeof v === "number") return Number.isNaN(v) ? 1 : 0;
    if (isRuntimeComplexNumber(v))
      return Number.isNaN(v.re) || Number.isNaN(v.im) ? 1 : 0;
    if (isRuntimeTensor(v)) {
      const n = v.data.length;
      const out = new FloatXArray(n);
      if (!v.imag) {
        for (let i = 0; i < n; i++) out[i] = Number.isNaN(v.data[i]) ? 1 : 0;
      } else {
        for (let i = 0; i < n; i++)
          out[i] = Number.isNaN(v.data[i]) || Number.isNaN(v.imag[i]) ? 1 : 0;
      }
      return makeTensor(out, undefined, v.shape.slice());
    }
    throw new Error("isnan: unsupported argument type");
  },
});

// ── isinf ───────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "isinf",
  typeRule: argTypes => predicateType(argTypes),
  apply: args => {
    const v = args[0];
    if (typeof v === "number") return !isFinite(v) && !Number.isNaN(v) ? 1 : 0;
    if (isRuntimeComplexNumber(v))
      return (!isFinite(v.re) && !Number.isNaN(v.re)) ||
        (!isFinite(v.im) && !Number.isNaN(v.im))
        ? 1
        : 0;
    if (isRuntimeTensor(v)) {
      const n = v.data.length;
      const out = new FloatXArray(n);
      if (!v.imag) {
        for (let i = 0; i < n; i++)
          out[i] = !isFinite(v.data[i]) && !Number.isNaN(v.data[i]) ? 1 : 0;
      } else {
        for (let i = 0; i < n; i++)
          out[i] =
            (!isFinite(v.data[i]) && !Number.isNaN(v.data[i])) ||
            (!isFinite(v.imag[i]) && !Number.isNaN(v.imag[i]))
              ? 1
              : 0;
      }
      return makeTensor(out, undefined, v.shape.slice());
    }
    throw new Error("isinf: unsupported argument type");
  },
});

// ── isfinite ────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "isfinite",
  typeRule: argTypes => predicateType(argTypes),
  apply: args => {
    const v = args[0];
    if (typeof v === "number") return isFinite(v) ? 1 : 0;
    if (isRuntimeComplexNumber(v))
      return isFinite(v.re) && isFinite(v.im) ? 1 : 0;
    if (isRuntimeTensor(v)) {
      const n = v.data.length;
      const out = new FloatXArray(n);
      if (!v.imag) {
        for (let i = 0; i < n; i++) out[i] = isFinite(v.data[i]) ? 1 : 0;
      } else {
        for (let i = 0; i < n; i++)
          out[i] = isFinite(v.data[i]) && isFinite(v.imag[i]) ? 1 : 0;
      }
      return makeTensor(out, undefined, v.shape.slice());
    }
    throw new Error("isfinite: unsupported argument type");
  },
});

// ── isreal ──────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "isreal",
  typeRule: argTypes => {
    if (argTypes.length !== 1) return null;
    // isreal always returns a scalar logical
    return [{ kind: "number", nonneg: true }];
  },
  apply: args => {
    const v = args[0];
    if (typeof v === "number") return 1;
    if (typeof v === "boolean") return 1;
    if (isRuntimeComplexNumber(v)) return v.im === 0 ? 1 : 0;
    if (isRuntimeTensor(v)) return !v.imag ? 1 : 0;
    return 1; // strings, chars, etc. are real
  },
});
