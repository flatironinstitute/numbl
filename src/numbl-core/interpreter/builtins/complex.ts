/**
 * Complex number builtins: real, imag, conj, angle.
 */

import {
  FloatXArray,
  isRuntimeComplexNumber,
  isRuntimeTensor,
} from "../../runtime/types.js";
import {
  registerIBuiltin,
  mkc,
  makeTensor,
  unaryAlwaysReal,
  unaryPreserveType,
} from "./types.js";

// ── real ────────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "real",
  resolve: argTypes => {
    const outputTypes = unaryAlwaysReal(argTypes);
    if (!outputTypes) return null;
    return {
      outputTypes,
      apply: args => {
        const v = args[0];
        if (typeof v === "boolean") return v ? 1 : 0;
        if (typeof v === "number") return v;
        if (isRuntimeComplexNumber(v)) return v.re;
        if (isRuntimeTensor(v)) {
          const out = new FloatXArray(v.data.length);
          out.set(v.data);
          return makeTensor(out, undefined, v.shape.slice());
        }
        throw new Error("real: unsupported argument type");
      },
    };
  },
});

// ── imag ────────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "imag",
  resolve: argTypes => {
    const outputTypes = unaryAlwaysReal(argTypes);
    if (!outputTypes) return null;
    return {
      outputTypes,
      apply: args => {
        const v = args[0];
        if (typeof v === "boolean") return 0;
        if (typeof v === "number") return 0;
        if (isRuntimeComplexNumber(v)) return v.im;
        if (isRuntimeTensor(v)) {
          const n = v.data.length;
          if (!v.imag)
            return makeTensor(new FloatXArray(n), undefined, v.shape.slice());
          const out = new FloatXArray(n);
          out.set(v.imag);
          return makeTensor(out, undefined, v.shape.slice());
        }
        throw new Error("imag: unsupported argument type");
      },
    };
  },
});

// ── conj ────────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "conj",
  resolve: argTypes => {
    const outputTypes = unaryPreserveType(argTypes);
    if (!outputTypes) return null;
    return {
      outputTypes,
      apply: args => {
        const v = args[0];
        if (typeof v === "boolean") return v ? 1 : 0;
        if (typeof v === "number") return v;
        if (isRuntimeComplexNumber(v)) return mkc(v.re, -v.im);
        if (isRuntimeTensor(v)) {
          const n = v.data.length;
          const outR = new FloatXArray(n);
          outR.set(v.data);
          if (!v.imag) return makeTensor(outR, undefined, v.shape.slice());
          const outI = new FloatXArray(n);
          for (let i = 0; i < n; i++) outI[i] = -v.imag[i];
          return makeTensor(outR, outI, v.shape.slice());
        }
        throw new Error("conj: unsupported argument type");
      },
    };
  },
});

// ── angle ───────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "angle",
  resolve: argTypes => {
    const outputTypes = unaryAlwaysReal(argTypes);
    if (!outputTypes) return null;
    return {
      outputTypes,
      apply: args => {
        const v = args[0];
        if (typeof v === "boolean") return 0;
        if (typeof v === "number") return v >= 0 ? 0 : Math.PI;
        if (isRuntimeComplexNumber(v)) return Math.atan2(v.im, v.re);
        if (isRuntimeTensor(v)) {
          const n = v.data.length;
          const out = new FloatXArray(n);
          if (!v.imag) {
            for (let i = 0; i < n; i++) out[i] = v.data[i] >= 0 ? 0 : Math.PI;
          } else {
            for (let i = 0; i < n; i++)
              out[i] = Math.atan2(v.imag[i], v.data[i]);
          }
          return makeTensor(out, undefined, v.shape.slice());
        }
        throw new Error("angle: unsupported argument type");
      },
    };
  },
});
