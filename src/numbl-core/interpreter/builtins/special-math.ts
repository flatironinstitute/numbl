/**
 * Special math builtins: airy, ellipj, legendre, besselj, bessely, besseli, besselk.
 */

import {
  FloatXArray,
  isRuntimeChar,
  isRuntimeComplexNumber,
  isRuntimeLogical,
  isRuntimeNumber,
  isRuntimeTensor,
} from "../../runtime/types.js";
import type { RuntimeValue, RuntimeTensor } from "../../runtime/types.js";
import { RTV, RuntimeError } from "../../runtime/index.js";
import { toNumber } from "../../runtime/convert.js";
import { registerIBuiltin } from "./types.js";
import {
  airyAi,
  airyAiPrime,
  airyBi,
  airyBiPrime,
  airyAllComplex,
  besselj,
  bessely,
  besseli,
  besselk,
} from "../../builtins/bessel.js";

// ── Helpers ──────────────────────────────────────────────────────────────

function binaryApply(
  a: RuntimeValue,
  b: RuntimeValue,
  fn: (x: number, y: number) => number
): RuntimeValue {
  const aIsT = isRuntimeTensor(a);
  const bIsT = isRuntimeTensor(b);
  if (aIsT && bIsT) {
    const result = new FloatXArray(a.data.length);
    for (let i = 0; i < a.data.length; i++)
      result[i] = fn(a.data[i], b.data[i]);
    return RTV.tensor(result, a.shape);
  }
  if (aIsT) {
    const bv = toNumber(b);
    const result = new FloatXArray(a.data.length);
    for (let i = 0; i < a.data.length; i++) result[i] = fn(a.data[i], bv);
    return RTV.tensor(result, a.shape);
  }
  if (bIsT) {
    const av = toNumber(a);
    const result = new FloatXArray(b.data.length);
    for (let i = 0; i < b.data.length; i++) result[i] = fn(av, b.data[i]);
    return RTV.tensor(result, b.shape);
  }
  return RTV.num(fn(toNumber(a), toNumber(b)));
}

// ── Bessel functions ─────────────────────────────────────────────────────

const besselDefs: [
  string,
  (nu: number, z: number) => number,
  (z: number) => number,
][] = [
  ["besselj", besselj, z => Math.exp(-Math.abs(z))],
  ["bessely", bessely, z => Math.exp(-Math.abs(z))],
  ["besseli", besseli, z => Math.exp(-Math.abs(z))],
  ["besselk", besselk, z => Math.exp(z)],
];

for (const [name, fn, scaleFn] of besselDefs) {
  registerIBuiltin({
    name,
    resolve: () => ({
      outputTypes: [{ kind: "unknown" }],
      apply: args => {
        if (args.length < 2 || args.length > 3)
          throw new RuntimeError(`${name} requires 2 or 3 arguments`);
        const scale = args.length === 3 ? toNumber(args[2]) : 0;
        return binaryApply(args[0], args[1], (nu, z) => {
          const val = fn(nu, z);
          return scale === 1 ? val * scaleFn(z) : val;
        });
      },
    }),
  });
}

// ── Airy functions ───────────────────────────────────────────────────────

const airyFns = [airyAi, airyAiPrime, airyBi, airyBiPrime];
const airyComplexKeys = ["ai", "aip", "bi", "bip"] as const;

function scaleAiry(n: number, x: number, val: number): number {
  const zeta = (2 / 3) * Math.pow(Math.abs(x), 1.5);
  if (n <= 1) return x >= 0 ? val * Math.exp(zeta) : val;
  return x >= 0 ? val * Math.exp(-zeta) : val;
}

function applyAiryElementwise(
  xArg: RuntimeValue,
  n: number,
  scaled: boolean
): RuntimeValue {
  if (isRuntimeComplexNumber(xArg)) {
    const all = airyAllComplex(xArg.re, xArg.im);
    const r = all[airyComplexKeys[n]];
    return r.im === 0 ? RTV.num(r.re) : RTV.complex(r.re, r.im);
  }
  if (isRuntimeTensor(xArg) && xArg.imag !== undefined) {
    const len = xArg.data.length;
    const resultRe = new FloatXArray(len);
    const resultIm = new FloatXArray(len);
    for (let i = 0; i < len; i++) {
      const all = airyAllComplex(xArg.data[i], xArg.imag[i]);
      const r = all[airyComplexKeys[n]];
      resultRe[i] = r.re;
      resultIm[i] = r.im;
    }
    const isReal = resultIm.every(x => x === 0);
    return RTV.tensor(resultRe, xArg.shape, isReal ? undefined : resultIm);
  }
  const fn = airyFns[n];
  if (isRuntimeTensor(xArg)) {
    const result = new FloatXArray(xArg.data.length);
    for (let i = 0; i < xArg.data.length; i++) {
      const val = fn(xArg.data[i]);
      result[i] = scaled ? scaleAiry(n, xArg.data[i], val) : val;
    }
    return RTV.tensor(result, xArg.shape);
  }
  const x = toNumber(xArg);
  const val = fn(x);
  return RTV.num(scaled ? scaleAiry(n, x, val) : val);
}

registerIBuiltin({
  name: "airy",
  resolve: () => ({
    outputTypes: [{ kind: "unknown" }],
    apply: args => {
      if (args.length < 1 || args.length > 3)
        throw new RuntimeError("airy requires 1 to 3 arguments");

      let nArg: RuntimeValue | undefined;
      let xArg: RuntimeValue;
      let scaled = false;

      if (args.length === 1) {
        xArg = args[0];
      } else {
        nArg = args[0];
        xArg = args[1];
        if (args.length === 3) scaled = toNumber(args[2]) === 1;
      }

      const n = nArg === undefined ? 0 : Math.round(toNumber(nArg));
      if (n < 0 || n > 3) throw new RuntimeError("K must be 0, 1, 2, or 3.");
      return applyAiryElementwise(xArg, n, scaled);
    },
  }),
});

// ── ellipj ───────────────────────────────────────────────────────────────

function ellipjScalar(
  u: number,
  m: number,
  tol: number
): { sn: number; cn: number; dn: number } {
  if (m < 0 || m > 1) return { sn: NaN, cn: NaN, dn: NaN };
  if (m === 0) return { sn: Math.sin(u), cn: Math.cos(u), dn: 1 };
  if (m === 1) {
    const s = Math.tanh(u);
    const c = 1 / Math.cosh(u);
    return { sn: s, cn: c, dn: c };
  }
  const a: number[] = [1];
  const b: number[] = [Math.sqrt(1 - m)];
  const cv: number[] = [Math.sqrt(m)];
  let i = 0;
  while (Math.abs(cv[i]) > tol) {
    a.push((a[i] + b[i]) / 2);
    b.push(Math.sqrt(a[i] * b[i]));
    cv.push((a[i] - b[i]) / 2);
    i++;
    if (i > 100) break;
  }
  const n = i;
  let phi = Math.pow(2, n) * a[n] * u;
  for (let j = n; j >= 1; j--) {
    phi = (phi + Math.asin((cv[j] / a[j]) * Math.sin(phi))) / 2;
  }
  const sn = Math.sin(phi);
  const cn = Math.cos(phi);
  const dn = Math.sqrt(1 - m * sn * sn);
  return { sn, cn, dn };
}

registerIBuiltin({
  name: "ellipj",
  resolve: (_argTypes, nargout) => {
    const outTypes = [];
    for (let i = 0; i < Math.max(nargout, 1); i++)
      outTypes.push({ kind: "unknown" as const });
    return {
      outputTypes: outTypes,
      apply: (args, nargout) => {
        if (args.length < 2 || args.length > 3)
          throw new RuntimeError("ellipj requires 2 or 3 arguments");
        const uArg = args[0];
        const mArg = args[1];
        const tol =
          args.length >= 3 ? toNumber(args[2]) : 2.220446049250313e-16;
        const uIsT = isRuntimeTensor(uArg);
        const mIsT = isRuntimeTensor(mArg);
        const effNargout = Math.max(nargout, 1);

        const buildResult = (
          snArr: InstanceType<typeof FloatXArray>,
          cnArr: InstanceType<typeof FloatXArray>,
          dnArr: InstanceType<typeof FloatXArray>,
          shape: number[]
        ): RuntimeValue | RuntimeValue[] => {
          const results: RuntimeValue[] = [];
          if (effNargout >= 1) results.push(RTV.tensor(snArr, [...shape]));
          if (effNargout >= 2) results.push(RTV.tensor(cnArr, [...shape]));
          if (effNargout >= 3) results.push(RTV.tensor(dnArr, [...shape]));
          return results.length === 1 ? results[0] : results;
        };

        if (!uIsT && !mIsT) {
          const r = ellipjScalar(toNumber(uArg), toNumber(mArg), tol);
          if (effNargout === 1) return RTV.num(r.sn);
          const results: RuntimeValue[] = [RTV.num(r.sn)];
          if (effNargout >= 2) results.push(RTV.num(r.cn));
          if (effNargout >= 3) results.push(RTV.num(r.dn));
          return results;
        }

        if (uIsT && !mIsT) {
          const mv = toNumber(mArg);
          const len = uArg.data.length;
          const snArr = new FloatXArray(len);
          const cnArr = new FloatXArray(len);
          const dnArr = new FloatXArray(len);
          for (let i = 0; i < len; i++) {
            const r = ellipjScalar(uArg.data[i], mv, tol);
            snArr[i] = r.sn;
            cnArr[i] = r.cn;
            dnArr[i] = r.dn;
          }
          return buildResult(snArr, cnArr, dnArr, uArg.shape);
        }

        if (!uIsT && mIsT) {
          const uv = toNumber(uArg);
          const mT = mArg as RuntimeTensor;
          const len = mT.data.length;
          const snArr = new FloatXArray(len);
          const cnArr = new FloatXArray(len);
          const dnArr = new FloatXArray(len);
          for (let i = 0; i < len; i++) {
            const r = ellipjScalar(uv, mT.data[i], tol);
            snArr[i] = r.sn;
            cnArr[i] = r.cn;
            dnArr[i] = r.dn;
          }
          return buildResult(snArr, cnArr, dnArr, mT.shape);
        }

        // tensor-tensor
        const uT = uArg as RuntimeTensor;
        const mT = mArg as RuntimeTensor;
        const len = uT.data.length;
        const snArr = new FloatXArray(len);
        const cnArr = new FloatXArray(len);
        const dnArr = new FloatXArray(len);
        for (let i = 0; i < len; i++) {
          const r = ellipjScalar(uT.data[i], mT.data[i], tol);
          snArr[i] = r.sn;
          cnArr[i] = r.cn;
          dnArr[i] = r.dn;
        }
        return buildResult(snArr, cnArr, dnArr, uT.shape);
      },
    };
  },
});

// ── legendre ─────────────────────────────────────────────────────────────

function factorialVal(n: number): number {
  if (n <= 1) return 1;
  let r = 1;
  for (let i = 2; i <= n; i++) r *= i;
  return r;
}

function legendreAllOrders(n: number, x: number): number[] {
  const result = new Array<number>(n + 1);
  if (n === 0) {
    result[0] = 1;
    return result;
  }
  const sqrtFactor = Math.sqrt(1 - x * x);

  for (let m = 0; m <= n; m++) {
    let pmm = 1.0;
    if (m > 0) {
      let dblFact = 1.0;
      for (let i = 1; i <= m; i++) dblFact *= 2 * i - 1;
      pmm = Math.pow(-1, m) * dblFact * Math.pow(sqrtFactor, m);
    }
    if (m === n) {
      result[m] = pmm;
      continue;
    }
    const pmm1 = x * (2 * m + 1) * pmm;
    if (m + 1 === n) {
      result[m] = pmm1;
      continue;
    }
    let pPrev2 = pmm;
    let pPrev1 = pmm1;
    let pCurr = 0;
    for (let l = m + 2; l <= n; l++) {
      pCurr = (x * (2 * l - 1) * pPrev1 - (l + m - 1) * pPrev2) / (l - m);
      pPrev2 = pPrev1;
      pPrev1 = pCurr;
    }
    result[m] = pCurr;
  }
  return result;
}

registerIBuiltin({
  name: "legendre",
  resolve: () => ({
    outputTypes: [{ kind: "unknown" }],
    apply: args => {
      if (args.length < 2 || args.length > 3)
        throw new RuntimeError("legendre requires 2 or 3 arguments");

      const n = Math.round(toNumber(args[0]));
      if (n < 0 || !isFinite(n))
        throw new RuntimeError("Degree n must be a non-negative integer");

      let normalization = "unnorm";
      if (args.length === 3) {
        const normArg = args[2];
        if (isRuntimeChar(normArg)) {
          normalization = normArg.value;
        } else if (typeof normArg === "string") {
          normalization = normArg;
        } else {
          throw new RuntimeError(
            "Third argument must be a normalization string"
          );
        }
        if (
          normalization !== "unnorm" &&
          normalization !== "sch" &&
          normalization !== "norm"
        )
          throw new RuntimeError(
            "Normalization must be 'unnorm', 'sch', or 'norm'"
          );
      }

      const xArg = args[1];
      let xValues: number[];
      let xShape: number[];
      if (isRuntimeNumber(xArg)) {
        xValues = [xArg as number];
        xShape = [1, 1];
      } else if (isRuntimeLogical(xArg)) {
        xValues = [xArg ? 1 : 0];
        xShape = [1, 1];
      } else if (isRuntimeTensor(xArg)) {
        xValues = Array.from(xArg.data);
        xShape = xArg.shape;
      } else {
        throw new RuntimeError("X must be a numeric value");
      }

      const numX = xValues.length;
      const numOrders = n + 1;
      const outSize = numOrders * numX;
      const result = new FloatXArray(outSize);

      for (let xi = 0; xi < numX; xi++) {
        const x = xValues[xi];
        const pmn = legendreAllOrders(n, x);
        for (let m = 0; m <= n; m++) {
          let val = pmn[m];
          if (normalization === "sch") {
            if (m > 0) {
              val *=
                Math.pow(-1, m) *
                Math.sqrt((2 * factorialVal(n - m)) / factorialVal(n + m));
            }
          } else if (normalization === "norm") {
            val *=
              Math.pow(-1, m) *
              Math.sqrt(
                ((n + 0.5) * factorialVal(n - m)) / factorialVal(n + m)
              );
          }
          result[xi * numOrders + m] = val;
        }
      }

      if (isRuntimeNumber(xArg) || isRuntimeLogical(xArg))
        return RTV.tensor(result, [numOrders, 1]);
      if (xShape.length === 2 && (xShape[0] === 1 || xShape[1] === 1))
        return RTV.tensor(result, [numOrders, numX]);
      return RTV.tensor(result, [numOrders, ...xShape]);
    },
  }),
});
