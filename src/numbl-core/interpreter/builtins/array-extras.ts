/**
 * Array extras: colon, tril, triu, toeplitz, magic,
 * bitwise ops, and coordinate transforms.
 */

import { RTV, toNumber, RuntimeError, mRange } from "../../runtime/index.js";
import {
  FloatXArray,
  type FloatXArrayType,
  type RuntimeValue,
  isRuntimeNumber,
  isRuntimeTensor,
  isRuntimeComplexNumber,
  isRuntimeSparseMatrix,
} from "../../runtime/types.js";
import { defineBuiltin } from "./types.js";

// ── colon ────────────────────────────────────────────────────────────

defineBuiltin({
  name: "colon",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length < 2 || argTypes.length > 3) return null;
        return [{ kind: "unknown" }];
      },
      apply: args => {
        if (args.length === 2)
          return mRange(toNumber(args[0]), 1, toNumber(args[1]));
        if (args.length === 3)
          return mRange(
            toNumber(args[0]),
            toNumber(args[1]),
            toNumber(args[2])
          );
        throw new RuntimeError("colon requires 2 or 3 arguments");
      },
    },
  ],
});

// ── triu / tril ──────────────────────────────────────────────────────

function triPart(
  args: RuntimeValue[],
  keepFn: (i: number, j: number, k: number) => boolean,
  name: string
): RuntimeValue {
  if (args.length < 1 || args.length > 2)
    throw new RuntimeError(`${name} requires 1 or 2 arguments`);
  const k = args.length === 2 ? Math.round(toNumber(args[1])) : 0;
  const v = args[0];
  if (isRuntimeNumber(v)) return keepFn(0, 0, k) ? v : RTV.num(0);
  if (isRuntimeSparseMatrix(v)) {
    const isComplex = v.pi !== undefined;
    const irArr: number[] = [];
    const prArr: number[] = [];
    const piArr: number[] = [];
    const jc = new Int32Array(v.n + 1);
    for (let c = 0; c < v.n; c++) {
      jc[c] = irArr.length;
      for (let kk = v.jc[c]; kk < v.jc[c + 1]; kk++) {
        if (keepFn(v.ir[kk], c, k)) {
          irArr.push(v.ir[kk]);
          prArr.push(v.pr[kk]);
          if (isComplex) piArr.push(v.pi![kk]);
        }
      }
    }
    jc[v.n] = irArr.length;
    return RTV.sparseMatrix(
      v.m,
      v.n,
      new Int32Array(irArr),
      jc,
      new Float64Array(prArr),
      isComplex ? new Float64Array(piArr) : undefined
    );
  }
  if (!isRuntimeTensor(v))
    throw new RuntimeError(`${name}: argument must be a matrix`);
  const nrows = v.shape[0] ?? 1;
  const ncols = v.shape.length >= 2 ? v.shape[1] : 1;
  const data = new FloatXArray(nrows * ncols);
  const idata = v.imag ? new FloatXArray(nrows * ncols) : undefined;
  for (let j = 0; j < ncols; j++) {
    for (let i = 0; i < nrows; i++) {
      if (keepFn(i, j, k)) {
        const idx = i + j * nrows;
        data[idx] = v.data[idx];
        if (idata && v.imag) idata[idx] = v.imag[idx];
      }
    }
  }
  return RTV.tensor(data, [nrows, ncols], idata);
}

defineBuiltin({
  name: "triu",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length < 1 || argTypes.length > 2) return null;
        return [{ kind: "unknown" }];
      },
      apply: args => triPart(args, (i, j, k) => j - i >= k, "triu"),
    },
  ],
});

defineBuiltin({
  name: "tril",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length < 1 || argTypes.length > 2) return null;
        return [{ kind: "unknown" }];
      },
      apply: args => triPart(args, (i, j, k) => i - j >= -k, "tril"),
    },
  ],
});

// ── toeplitz ─────────────────────────────────────────────────────────

defineBuiltin({
  name: "toeplitz",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length < 1 || argTypes.length > 2) return null;
        return [{ kind: "unknown" }];
      },
      apply: args => {
        function vecData(v: RuntimeValue): {
          re: FloatXArrayType;
          im: FloatXArrayType | undefined;
          len: number;
        } {
          if (isRuntimeNumber(v))
            return {
              re: new FloatXArray([v as number]),
              im: undefined,
              len: 1,
            };
          if (isRuntimeComplexNumber(v))
            return {
              re: new FloatXArray([v.re]),
              im: new FloatXArray([v.im]),
              len: 1,
            };
          if (isRuntimeTensor(v))
            return { re: v.data, im: v.imag, len: v.data.length };
          throw new RuntimeError("toeplitz: arguments must be numeric vectors");
        }

        let colRe: FloatXArrayType,
          colIm: FloatXArrayType | undefined,
          colLen: number,
          rowRe: FloatXArrayType,
          rowIm: FloatXArrayType | undefined,
          rowLen: number;

        if (args.length === 1) {
          const { re, im, len } = vecData(args[0]);
          rowRe = re;
          rowIm = im;
          rowLen = len;
          colRe = new FloatXArray(re);
          colLen = len;
          if (im) {
            const ci = new FloatXArray(im);
            for (let k = 1; k < len; k++) ci[k] = -im[k];
            colIm = ci;
          }
        } else {
          const c = vecData(args[0]);
          const r = vecData(args[1]);
          colRe = c.re;
          colIm = c.im;
          colLen = c.len;
          rowRe = r.re;
          rowIm = r.im;
          rowLen = r.len;
        }

        const m = colLen;
        const n = rowLen;
        const isComplex = colIm !== undefined || rowIm !== undefined;
        const data = new FloatXArray(m * n);
        const idata = isComplex ? new FloatXArray(m * n) : undefined;

        for (let j = 0; j < n; j++) {
          for (let i = 0; i < m; i++) {
            const idx = i + j * m;
            const diag = i - j;
            if (diag >= 0) {
              data[idx] = colRe[diag];
              if (idata) idata[idx] = colIm ? colIm[diag] : 0;
            } else {
              const k = -diag;
              data[idx] = rowRe[k];
              if (idata) idata[idx] = rowIm ? rowIm[k] : 0;
            }
          }
        }

        return RTV.tensor(data, [m, n], idata);
      },
    },
  ],
});

// ── magic ────────────────────────────────────────────────────────────

defineBuiltin({
  name: "magic",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length !== 1) return null;
        return [{ kind: "unknown" }];
      },
      apply: args => {
        const n = Math.round(toNumber(args[0]));
        if (n < 3)
          throw new RuntimeError("Order N must be greater than or equal to 3");

        const M = new Array<number>(n * n).fill(0);
        const set = (r: number, c: number, v: number) => {
          M[c * n + r] = v;
        };
        const get = (r: number, c: number) => M[c * n + r];

        if (n % 2 === 1) {
          let i = 0;
          let j = Math.floor(n / 2);
          for (let k = 1; k <= n * n; k++) {
            set(i, j, k);
            const ni = (i - 1 + n) % n;
            const nj = (j + 1) % n;
            if (get(ni, nj) !== 0) {
              i = (i + 1) % n;
            } else {
              i = ni;
              j = nj;
            }
          }
        } else if (n % 4 === 0) {
          for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
              set(i, j, i * n + j + 1);
            }
          }
          for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
              const ii = i % 4;
              const jj = j % 4;
              if (
                ((ii === 0 || ii === 3) && (jj === 0 || jj === 3)) ||
                ((ii === 1 || ii === 2) && (jj === 1 || jj === 2))
              ) {
                set(i, j, n * n + 1 - get(i, j));
              }
            }
          }
        } else {
          const p = n / 2;
          const sub = new Array<number>(p * p).fill(0);
          const sset = (r: number, c: number, v: number) => {
            sub[c * p + r] = v;
          };
          const sget = (r: number, c: number) => sub[c * p + r];

          let si = 0;
          let sj = Math.floor(p / 2);
          for (let k = 1; k <= p * p; k++) {
            sset(si, sj, k);
            const ni = (si - 1 + p) % p;
            const nj = (sj + 1) % p;
            if (sget(ni, nj) !== 0) {
              si = (si + 1) % p;
            } else {
              si = ni;
              sj = nj;
            }
          }

          for (let i = 0; i < p; i++) {
            for (let j = 0; j < p; j++) {
              const v = sget(i, j);
              set(i, j, v);
              set(i, j + p, v + 2 * p * p);
              set(i + p, j, v + 3 * p * p);
              set(i + p, j + p, v + p * p);
            }
          }

          const k = Math.floor((n - 2) / 4);
          for (let i = 0; i < p; i++) {
            for (let j = 0; j < k; j++) {
              if (j === 0) {
                if (i === Math.floor(p / 2)) {
                  const tmp = get(i, 1);
                  set(i, 1, get(i + p, 1));
                  set(i + p, 1, tmp);
                } else {
                  const tmp = get(i, 0);
                  set(i, 0, get(i + p, 0));
                  set(i + p, 0, tmp);
                }
              } else {
                const tmp = get(i, j);
                set(i, j, get(i + p, j));
                set(i + p, j, tmp);
              }
            }
            for (let j = n - k + 2; j < n; j++) {
              const tmp = get(i, j);
              set(i, j, get(i + p, j));
              set(i + p, j, tmp);
            }
          }
        }

        return RTV.tensor(new FloatXArray(M), [n, n]);
      },
    },
  ],
});

// ── Bitwise ops ──────────────────────────────────────────────────────

function bitwiseOp(
  a: RuntimeValue,
  b: RuntimeValue,
  op: (a: number, b: number) => number,
  name: string
): RuntimeValue {
  if (isRuntimeNumber(a) && isRuntimeNumber(b)) {
    return RTV.num(op(Math.round(a as number), Math.round(b as number)));
  }
  if (isRuntimeTensor(a) && isRuntimeNumber(b)) {
    const bv = Math.round(b as number);
    const result = new FloatXArray(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      result[i] = op(Math.round(a.data[i]), bv);
    }
    return RTV.tensor(result, [...a.shape]);
  }
  if (isRuntimeNumber(a) && isRuntimeTensor(b)) {
    const av = Math.round(a as number);
    const result = new FloatXArray(b.data.length);
    for (let i = 0; i < b.data.length; i++) {
      result[i] = op(av, Math.round(b.data[i]));
    }
    return RTV.tensor(result, [...b.shape]);
  }
  if (isRuntimeTensor(a) && isRuntimeTensor(b)) {
    if (a.data.length !== b.data.length)
      throw new RuntimeError(`${name}: arrays must be the same size`);
    const result = new FloatXArray(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      result[i] = op(Math.round(a.data[i]), Math.round(b.data[i]));
    }
    return RTV.tensor(result, [...a.shape]);
  }
  throw new RuntimeError(`${name}: arguments must be numeric`);
}

defineBuiltin({
  name: "bitand",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length !== 2) return null;
        return [{ kind: "unknown" }];
      },
      apply: args => bitwiseOp(args[0], args[1], (a, b) => a & b, "bitand"),
    },
  ],
});

defineBuiltin({
  name: "bitor",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length !== 2) return null;
        return [{ kind: "unknown" }];
      },
      apply: args => bitwiseOp(args[0], args[1], (a, b) => a | b, "bitor"),
    },
  ],
});

defineBuiltin({
  name: "bitxor",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length !== 2) return null;
        return [{ kind: "unknown" }];
      },
      apply: args => bitwiseOp(args[0], args[1], (a, b) => a ^ b, "bitxor"),
    },
  ],
});

defineBuiltin({
  name: "bitshift",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length < 2 || argTypes.length > 3) return null;
        return [{ kind: "unknown" }];
      },
      apply: args => {
        const shift = (a: number, k: number): number => {
          const ai = Math.round(a);
          const ki = Math.round(k);
          return ki >= 0 ? ai << ki : ai >>> -ki;
        };

        const aVal = args[0];
        const kVal = args[1];

        if (isRuntimeNumber(aVal) && isRuntimeNumber(kVal)) {
          return RTV.num(shift(aVal as number, kVal as number));
        }
        if (isRuntimeTensor(aVal) && isRuntimeNumber(kVal)) {
          const result = new FloatXArray(aVal.data.length);
          for (let i = 0; i < aVal.data.length; i++) {
            result[i] = shift(aVal.data[i], kVal as number);
          }
          return RTV.tensor(result, [...aVal.shape]);
        }
        if (isRuntimeNumber(aVal) && isRuntimeTensor(kVal)) {
          const result = new FloatXArray(kVal.data.length);
          for (let i = 0; i < kVal.data.length; i++) {
            result[i] = shift(aVal as number, kVal.data[i]);
          }
          return RTV.tensor(result, [...kVal.shape]);
        }
        if (isRuntimeTensor(aVal) && isRuntimeTensor(kVal)) {
          if (aVal.data.length !== kVal.data.length)
            throw new RuntimeError("bitshift: arrays must be the same size");
          const result = new FloatXArray(aVal.data.length);
          for (let i = 0; i < aVal.data.length; i++) {
            result[i] = shift(aVal.data[i], kVal.data[i]);
          }
          return RTV.tensor(result, [...aVal.shape]);
        }

        throw new RuntimeError("bitshift: arguments must be numeric");
      },
    },
  ],
});

// ── Coordinate transforms ────────────────────────────────────────────

function coordTransform(
  name: string,
  nArgs: 2 | 3 | "2or3",
  nOut: number,
  fn: (...vals: number[]) => number[]
): void {
  defineBuiltin({
    name,
    cases: [
      {
        match: argTypes => {
          const minArgs = nArgs === "2or3" ? 2 : nArgs;
          const maxArgs = nArgs === "2or3" ? 3 : nArgs;
          if (argTypes.length < minArgs || argTypes.length > maxArgs)
            return null;
          return [{ kind: "unknown" }];
        },
        apply: (args, nargout) => {
          const n = args.length;
          const tensors = args.map(a => (isRuntimeTensor(a) ? a : null));
          const anyTensor = tensors.some(t => t !== null);

          if (anyTensor) {
            const refT = tensors.find(t => t !== null)!;
            const shape = refT.shape;
            const len = refT.data.length;
            const datas = tensors.map(t => (t ? t.data : null));
            const scalars = args.map((a, i) => (datas[i] ? 0 : toNumber(a)));
            const outArrays = Array.from(
              { length: nOut },
              () => new FloatXArray(len)
            );
            for (let i = 0; i < len; i++) {
              const vals = datas.map((d, j) => (d ? d[i] : scalars[j]));
              const result = fn(...vals);
              for (let k = 0; k < nOut; k++) outArrays[k][i] = result[k];
            }
            const effOut = nArgs === "2or3" && n === 3 ? 3 : nOut;
            const outTensors = outArrays.map(d => RTV.tensor(d, shape));
            if (nArgs === "2or3" && n === 3 && nOut === 2) {
              const zd = datas[2];
              const zOut = new FloatXArray(len);
              for (let i = 0; i < len; i++) zOut[i] = zd ? zd[i] : scalars[2];
              outTensors.push(RTV.tensor(zOut, shape));
            }
            if (nargout <= 1) return outTensors[0];
            return outTensors.slice(0, Math.min(nargout, effOut));
          }

          const vals = args.map(a => toNumber(a));
          const result = fn(...vals);
          const effOut = nArgs === "2or3" && n === 3 ? 3 : nOut;
          const outVals = result.map(v => RTV.num(v));
          if (nArgs === "2or3" && n === 3 && nOut === 2) {
            outVals.push(RTV.num(vals[2]));
          }
          if (nargout <= 1) return outVals[0];
          return outVals.slice(0, Math.min(nargout, effOut));
        },
      },
    ],
  });
}

coordTransform("cart2sph", 3, 3, (x, y, z) => {
  const hypotxy = Math.sqrt(x * x + y * y);
  return [
    Math.atan2(y, x),
    Math.atan2(z, hypotxy),
    Math.sqrt(x * x + y * y + z * z),
  ];
});

coordTransform("sph2cart", 3, 3, (az, el, r) => {
  const rcosel = r * Math.cos(el);
  return [rcosel * Math.cos(az), rcosel * Math.sin(az), r * Math.sin(el)];
});

coordTransform("cart2pol", "2or3", 2, (x, y) => [
  Math.atan2(y, x),
  Math.sqrt(x * x + y * y),
]);

coordTransform("pol2cart", "2or3", 2, (theta, rho) => [
  rho * Math.cos(theta),
  rho * Math.sin(theta),
]);
