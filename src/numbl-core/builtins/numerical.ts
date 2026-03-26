/**
 * Numerical/polynomial/signal processing builtin functions:
 * conv, deconv, polyval, polyfit, trapz, cumtrapz, gradient, accumarray,
 * interp1, roots, poly, cov, corrcoef, bitand, bitor, bitxor, bitshift
 */

import {
  RuntimeValue,
  RTV,
  toNumber,
  RuntimeError,
  tensorSize2D,
} from "../runtime/index.js";
import {
  FloatXArray,
  FloatXArrayType,
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeTensor,
  isRuntimeChar,
  isRuntimeString,
  isRuntimeFunction,
} from "../runtime/types.js";
import { register, builtinSingle } from "./registry.js";
import { getLapackBridge } from "../native/lapack-bridge.js";
import { linsolveLapack } from "./linear-algebra/linsolve.js";

/** Convert a runtime value to a Float64 array (vector). */
function toFloatArray(v: RuntimeValue): FloatXArrayType {
  if (isRuntimeNumber(v)) return new FloatXArray([v]);
  if (isRuntimeLogical(v)) return new FloatXArray([v ? 1 : 0]);
  if (isRuntimeTensor(v)) return new FloatXArray(v.data);
  throw new RuntimeError("Expected numeric argument");
}

/** Check if value is a column vector */
function isColumnVector(v: RuntimeValue): boolean {
  if (isRuntimeTensor(v))
    return v.shape[0] > 1 && (v.shape.length < 2 || v.shape[1] === 1);
  return false;
}

/** Compute eps(x): distance from abs(x) to the next larger double. */
function epsOfScalar(x: number): number {
  x = Math.abs(x);
  if (!isFinite(x) || isNaN(x)) return NaN;
  if (x === 0) return Number.MIN_VALUE;
  const buf = new Float64Array(1);
  const view = new DataView(buf.buffer);
  buf[0] = x;
  const bits = view.getBigUint64(0, true);
  view.setBigUint64(0, bits + 1n, true);
  return buf[0] - x;
}

export function registerNumericalFunctions(): void {
  // ── conv ──────────────────────────────────────────────────────────────

  register(
    "conv",
    builtinSingle(args => {
      if (args.length < 2 || args.length > 3)
        throw new RuntimeError("conv requires 2 or 3 arguments");

      const a = toFloatArray(args[0]);
      const b = toFloatArray(args[1]);
      const m = a.length;
      const n = b.length;

      // Full convolution
      const fullLen = m + n - 1;
      const full = new FloatXArray(fullLen);
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
          full[i + j] += a[i] * b[j];
        }
      }

      // Determine shape
      let shape: string = "full";
      if (args.length === 3) {
        const opt = args[2];
        if (isRuntimeChar(opt)) shape = opt.value.toLowerCase();
        else if (isRuntimeString(opt)) shape = opt.toLowerCase();
        else throw new RuntimeError("conv: third argument must be a string");
      }

      let result: FloatXArrayType;
      if (shape === "full") {
        result = full;
      } else if (shape === "same") {
        // Same size as first input
        const start = Math.floor(n / 2);
        result = new FloatXArray(m);
        for (let i = 0; i < m; i++) {
          result[i] = full[start + i];
        }
      } else if (shape === "valid") {
        const validLen = Math.max(m, n) - Math.min(m, n) + 1;
        const start = Math.min(m, n) - 1;
        result = new FloatXArray(validLen);
        for (let i = 0; i < validLen; i++) {
          result[i] = full[start + i];
        }
      } else {
        throw new RuntimeError(`conv: unknown shape '${shape}'`);
      }

      return RTV.tensor(result, [1, result.length]);
    })
  );

  // ── deconv ────────────────────────────────────────────────────────────

  register(
    "deconv",
    builtinSingle((args, nargout) => {
      if (args.length !== 2)
        throw new RuntimeError("deconv requires exactly 2 arguments");

      const b = toFloatArray(args[0]); // dividend
      const a = toFloatArray(args[1]); // divisor
      const nb = b.length;
      const na = a.length;

      if (na === 0 || a[0] === 0)
        throw new RuntimeError(
          "deconv: leading coefficient of divisor is zero"
        );

      const nq = nb - na + 1;
      if (nq <= 0) {
        // Quotient is zero, remainder is the dividend
        if (nargout <= 1) return RTV.tensor(new FloatXArray([0]), [1, 1]);
        return [
          RTV.tensor(new FloatXArray([0]), [1, 1]),
          RTV.tensor(new FloatXArray(b), [1, nb]),
        ];
      }

      // Polynomial long division
      const r = new FloatXArray(b); // remainder (copy)
      const q = new FloatXArray(nq);

      for (let i = 0; i < nq; i++) {
        q[i] = r[i] / a[0];
        for (let j = 0; j < na; j++) {
          r[i + j] -= q[i] * a[j];
        }
      }

      if (nargout <= 1) return RTV.tensor(q, [1, nq]);
      return [RTV.tensor(q, [1, nq]), RTV.tensor(r, [1, nb])];
    })
  );

  // ── polyval ───────────────────────────────────────────────────────────

  register(
    "polyval",
    builtinSingle(args => {
      if (args.length < 2 || args.length > 3)
        throw new RuntimeError("polyval requires 2 or 3 arguments");

      const p = toFloatArray(args[0]);
      const x = args[1];
      const np = p.length;

      // Evaluate using Horner's method
      const horner = (xVal: number): number => {
        let result = p[0];
        for (let i = 1; i < np; i++) {
          result = result * xVal + p[i];
        }
        return result;
      };

      if (isRuntimeNumber(x)) {
        return RTV.num(horner(x));
      }

      if (isRuntimeTensor(x)) {
        const result = new FloatXArray(x.data.length);
        for (let i = 0; i < x.data.length; i++) {
          result[i] = horner(x.data[i]);
        }
        return RTV.tensor(result, [...x.shape]);
      }

      throw new RuntimeError("polyval: second argument must be numeric");
    })
  );

  // ── polyfit ───────────────────────────────────────────────────────────

  register(
    "polyfit",
    builtinSingle(args => {
      if (args.length !== 3)
        throw new RuntimeError("polyfit requires exactly 3 arguments");

      const xArr = toFloatArray(args[0]);
      const yArr = toFloatArray(args[1]);
      const n = Math.round(toNumber(args[2])); // polynomial degree
      const m = xArr.length;

      if (xArr.length !== yArr.length)
        throw new RuntimeError("polyfit: x and y must have the same length");
      if (n < 0) throw new RuntimeError("polyfit: degree must be non-negative");

      const ncols = n + 1;

      if (ncols > m) {
        console.warn(
          "Warning: Polynomial is not unique; degree >= number of data points."
        );
      }

      // Build Vandermonde matrix V (m x ncols) in column-major
      // V[i, j] = x[i]^(n-j), so highest power first
      const V = new FloatXArray(m * ncols);
      for (let j = 0; j < ncols; j++) {
        const power = n - j;
        for (let i = 0; i < m; i++) {
          V[j * m + i] = Math.pow(xArr[i], power);
        }
      }

      // Solve V * p = y via linsolve (uses QR/LQ via LAPACK for non-square)
      const B = new FloatXArray(m);
      for (let i = 0; i < m; i++) B[i] = yArr[i];

      const X = linsolveLapack(V, m, ncols, B, 1);
      if (!X) throw new RuntimeError("polyfit: LAPACK bridge unavailable");

      // Check for badly conditioned result (any NaN/Inf in output)
      let badlyConditioned = false;
      for (let i = 0; i < ncols; i++) {
        if (!isFinite(X[i])) {
          badlyConditioned = true;
          break;
        }
      }
      if (!badlyConditioned) {
        // Estimate condition: check if residuals are suspiciously large
        // relative to the data, indicating ill-conditioning
        let maxX = 0;
        for (let i = 0; i < ncols; i++) maxX = Math.max(maxX, Math.abs(X[i]));
        let maxY = 0;
        for (let i = 0; i < m; i++) maxY = Math.max(maxY, Math.abs(yArr[i]));
        if (maxY > 0 && maxX / maxY > 1e10) badlyConditioned = true;
      }

      if (badlyConditioned) {
        console.warn(
          "Warning: Polynomial is badly conditioned. Add points with distinct X values, " +
            "reduce the degree of the polynomial, or try centering and scaling as described in HELP POLYFIT."
        );
      }

      const result = new FloatXArray(ncols);
      for (let i = 0; i < ncols; i++) result[i] = X[i];

      return RTV.tensor(result, [1, ncols]);
    })
  );

  // ── trapz ─────────────────────────────────────────────────────────────

  register(
    "trapz",
    builtinSingle(args => {
      if (args.length < 1 || args.length > 2)
        throw new RuntimeError("trapz requires 1 or 2 arguments");

      let x: FloatXArrayType | null = null;
      let y: FloatXArrayType;

      if (args.length === 1) {
        y = toFloatArray(args[0]);
      } else {
        x = toFloatArray(args[0]);
        y = toFloatArray(args[1]);
        if (x.length !== y.length)
          throw new RuntimeError("trapz: x and y must have the same length");
      }

      if (y.length <= 1) return RTV.num(0);

      let sum = 0;
      for (let i = 0; i < y.length - 1; i++) {
        const dx = x ? x[i + 1] - x[i] : 1;
        sum += ((y[i] + y[i + 1]) * dx) / 2;
      }

      return RTV.num(sum);
    })
  );

  // ── gradient ──────────────────────────────────────────────────────────

  register(
    "gradient",
    builtinSingle((args, nargout) => {
      if (args.length < 1 || args.length > 2)
        throw new RuntimeError("gradient requires 1 or 2 arguments");

      const f = args[0];

      // Determine spacing: scalar number or coordinate vector
      let hScalar: number | null = 1;
      let hVec: FloatXArrayType | null = null;
      if (args.length === 2) {
        const spacing = args[1];
        if (isRuntimeNumber(spacing)) {
          hScalar = spacing;
        } else if (isRuntimeTensor(spacing)) {
          // Non-uniform spacing (coordinate vector)
          hVec = new FloatXArray(spacing.data);
          hScalar = null;
        } else {
          hScalar = toNumber(spacing);
        }
      }

      // Helper: compute 1D gradient with uniform spacing
      const grad1dUniform = (
        data: FloatXArrayType,
        len: number,
        spacing: number
      ): FloatXArrayType => {
        const result = new FloatXArray(len);
        if (len < 2) {
          result[0] = 0;
          return result;
        }
        result[0] = (data[1] - data[0]) / spacing;
        for (let i = 1; i < len - 1; i++) {
          result[i] = (data[i + 1] - data[i - 1]) / (2 * spacing);
        }
        result[len - 1] = (data[len - 1] - data[len - 2]) / spacing;
        return result;
      };

      // Helper: compute 1D gradient with non-uniform spacing
      const grad1dNonUniform = (
        data: FloatXArrayType,
        len: number,
        coords: FloatXArrayType
      ): FloatXArrayType => {
        const result = new FloatXArray(len);
        if (len < 2) {
          result[0] = 0;
          return result;
        }
        result[0] = (data[1] - data[0]) / (coords[1] - coords[0]);
        for (let i = 1; i < len - 1; i++) {
          result[i] =
            (data[i + 1] - data[i - 1]) / (coords[i + 1] - coords[i - 1]);
        }
        result[len - 1] =
          (data[len - 1] - data[len - 2]) / (coords[len - 1] - coords[len - 2]);
        return result;
      };

      // Check if input is a vector or matrix
      if (!isRuntimeTensor(f)) {
        return nargout >= 2 ? [RTV.num(0), RTV.num(0)] : RTV.num(0);
      }

      const nRows = f.shape[0] || 1;
      const nCols = f.data.length / nRows;
      const isVec = nRows === 1 || nCols === 1;

      if (isVec) {
        const n = f.data.length;
        if (n < 2)
          throw new RuntimeError(
            "gradient: input must have at least 2 elements"
          );
        if (hVec) {
          if (hVec.length !== n)
            throw new RuntimeError(
              "gradient: coordinate vector must match input length"
            );
          const result = grad1dNonUniform(f.data, n, hVec);
          const shape = isColumnVector(f) ? [n, 1] : [1, n];
          return RTV.tensor(result, shape);
        }
        const result = grad1dUniform(f.data, n, hScalar!);
        const shape = isColumnVector(f) ? [n, 1] : [1, n];
        return RTV.tensor(result, shape);
      }

      // 2D case: compute fx (along columns) and fy (along rows)
      // Data is column-major: element at (r,c) is data[c * nRows + r]
      const sp = hScalar ?? 1;

      // fx: gradient in x-direction (across columns) for each row
      const fxData = new FloatXArray(nRows * nCols);
      for (let r = 0; r < nRows; r++) {
        const rowSlice = new FloatXArray(nCols);
        for (let c = 0; c < nCols; c++) {
          rowSlice[c] = f.data[c * nRows + r];
        }
        const rowGrad = grad1dUniform(rowSlice, nCols, sp);
        for (let c = 0; c < nCols; c++) {
          fxData[c * nRows + r] = rowGrad[c];
        }
      }
      const fx = RTV.tensor(fxData, [nRows, nCols]);

      if (nargout >= 2) {
        // fy: gradient in y-direction (across rows) for each column
        const fyData = new FloatXArray(nRows * nCols);
        for (let c = 0; c < nCols; c++) {
          const colSlice = new FloatXArray(nRows);
          for (let r = 0; r < nRows; r++) {
            colSlice[r] = f.data[c * nRows + r];
          }
          const colGrad = grad1dUniform(colSlice, nRows, sp);
          for (let r = 0; r < nRows; r++) {
            fyData[c * nRows + r] = colGrad[r];
          }
        }
        const fy = RTV.tensor(fyData, [nRows, nCols]);
        return [fx, fy];
      }

      return fx;
    })
  );

  // ── accumarray ────────────────────────────────────────────────────────

  register(
    "accumarray",
    builtinSingle(args => {
      if (args.length < 2)
        throw new RuntimeError("accumarray requires at least 2 arguments");

      const subsArg = args[0];
      const valsArg = args[1];

      // Parse subscripts (column vector of indices)
      const subs = Array.from(toFloatArray(subsArg), (x: number) =>
        Math.round(x)
      );
      const nSubs = subs.length;

      // Parse values
      let vals: FloatXArrayType;
      if (isRuntimeNumber(valsArg)) {
        // Scalar val is replicated
        vals = new FloatXArray(nSubs);
        vals.fill(valsArg);
      } else {
        vals = toFloatArray(valsArg);
        if (vals.length !== nSubs)
          throw new RuntimeError(
            "accumarray: subs and vals must have the same length"
          );
      }

      // Determine max index
      let maxIdx = 0;
      for (let i = 0; i < nSubs; i++) {
        if (subs[i] > maxIdx) maxIdx = subs[i];
        if (subs[i] < 1)
          throw new RuntimeError(
            "accumarray: subscript indices must be positive integers"
          );
      }

      // Parse optional size argument
      if (args.length >= 3 && args[2] !== undefined) {
        const szArg = args[2];
        if (isRuntimeTensor(szArg) && szArg.data.length > 0) {
          const requestedMax = Math.round(szArg.data[0]);
          if (requestedMax > maxIdx) maxIdx = requestedMax;
        } else if (isRuntimeNumber(szArg)) {
          if (szArg > maxIdx) maxIdx = Math.round(szArg);
        }
      }

      // Parse optional function handle
      let reduceFn: (group: number[]) => number;
      if (args.length >= 4 && isRuntimeFunction(args[3])) {
        // We'll handle common function handles
        const fnName = args[3].name;
        switch (fnName) {
          case "sum":
            reduceFn = g => g.reduce((a, b) => a + b, 0);
            break;
          case "max":
            reduceFn = g => Math.max(...g);
            break;
          case "min":
            reduceFn = g => Math.min(...g);
            break;
          case "mean":
            reduceFn = g => g.reduce((a, b) => a + b, 0) / g.length;
            break;
          case "prod":
            reduceFn = g => g.reduce((a, b) => a * b, 1);
            break;
          case "numel":
            reduceFn = g => g.length;
            break;
          default:
            throw new RuntimeError(
              `accumarray: unsupported function handle @${fnName}`
            );
        }
      } else {
        // Default: sum
        reduceFn = g => g.reduce((a, b) => a + b, 0);
      }

      // Group values by subscript
      const groups: number[][] = new Array(maxIdx);
      for (let i = 0; i < maxIdx; i++) groups[i] = [];
      for (let i = 0; i < nSubs; i++) {
        groups[subs[i] - 1].push(vals[i]);
      }

      // Apply reduction
      const result = new FloatXArray(maxIdx);
      for (let i = 0; i < maxIdx; i++) {
        if (groups[i].length > 0) {
          result[i] = reduceFn(groups[i]);
        }
        // else: result[i] = 0 (default fill)
      }

      return RTV.tensor(result, [maxIdx, 1]);
    })
  );

  // ── cumtrapz ──────────────────────────────────────────────────────────

  register(
    "cumtrapz",
    builtinSingle(args => {
      if (args.length < 1 || args.length > 2)
        throw new RuntimeError("cumtrapz requires 1 or 2 arguments");

      let x: FloatXArrayType | null = null;
      let y: FloatXArrayType;
      const yArg = args.length === 1 ? args[0] : args[1];

      if (args.length === 1) {
        y = toFloatArray(args[0]);
      } else {
        x = toFloatArray(args[0]);
        y = toFloatArray(args[1]);
        if (x.length !== y.length)
          throw new RuntimeError("cumtrapz: x and y must have the same length");
      }

      const n = y.length;
      const result = new FloatXArray(n);
      result[0] = 0;
      for (let i = 1; i < n; i++) {
        const dx = x ? x[i] - x[i - 1] : 1;
        result[i] = result[i - 1] + ((y[i - 1] + y[i]) * dx) / 2;
      }

      if (isColumnVector(yArg)) {
        return RTV.tensor(result, [n, 1]);
      }
      return RTV.tensor(result, [1, n]);
    })
  );

  // ── interp1 ─────────────────────────────────────────────────────────

  register(
    "interp1",
    builtinSingle(args => {
      if (args.length < 3 || args.length > 5)
        throw new RuntimeError("interp1 requires 3 to 5 arguments");

      const xArr = toFloatArray(args[0]);
      const yArr = toFloatArray(args[1]);
      const xqArg = args[2];
      const nn = xArr.length;

      if (xArr.length !== yArr.length)
        throw new RuntimeError("interp1: x and y must have the same length");

      // Parse method
      let method = "linear";
      let doExtrap = false;
      for (let i = 3; i < args.length; i++) {
        const a = args[i];
        if (isRuntimeChar(a)) {
          const s = a.value.toLowerCase();
          if (s === "extrap") doExtrap = true;
          else method = s;
        } else if (isRuntimeString(a)) {
          const s = a.toLowerCase();
          if (s === "extrap") doExtrap = true;
          else method = s;
        }
      }

      const interpOne = (xq: number): number => {
        // Find interval via binary search
        if (xq < xArr[0] || xq > xArr[nn - 1]) {
          if (!doExtrap) return NaN;
          // Linear extrapolation
          if (method === "linear") {
            if (xq < xArr[0]) {
              const slope = (yArr[1] - yArr[0]) / (xArr[1] - xArr[0]);
              return yArr[0] + slope * (xq - xArr[0]);
            } else {
              const slope =
                (yArr[nn - 1] - yArr[nn - 2]) / (xArr[nn - 1] - xArr[nn - 2]);
              return yArr[nn - 1] + slope * (xq - xArr[nn - 1]);
            }
          }
        }

        // Exact endpoint match
        if (xq <= xArr[0]) return yArr[0];
        if (xq >= xArr[nn - 1]) return yArr[nn - 1];

        // Binary search for interval
        let lo = 0,
          hi = nn - 1;
        while (hi - lo > 1) {
          const mid = (lo + hi) >> 1;
          if (xArr[mid] <= xq) lo = mid;
          else hi = mid;
        }

        if (method === "nearest") {
          return xq - xArr[lo] < xArr[hi] - xq ? yArr[lo] : yArr[hi];
        }

        // Linear interpolation
        const t = (xq - xArr[lo]) / (xArr[hi] - xArr[lo]);
        return yArr[lo] + t * (yArr[hi] - yArr[lo]);
      };

      if (isRuntimeNumber(xqArg)) {
        return RTV.num(interpOne(xqArg));
      }

      if (isRuntimeTensor(xqArg)) {
        const result = new FloatXArray(xqArg.data.length);
        for (let i = 0; i < xqArg.data.length; i++) {
          result[i] = interpOne(xqArg.data[i]);
        }
        return RTV.tensor(result, [...xqArg.shape]);
      }

      throw new RuntimeError("interp1: query points must be numeric");
    })
  );

  // ── roots ───────────────────────────────────────────────────────────

  register(
    "roots",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("roots requires exactly 1 argument");

      const p = toFloatArray(args[0]);
      const n = p.length - 1; // degree

      if (n <= 0) return RTV.tensor(new FloatXArray(0), [0, 0]);

      // Build companion matrix (n x n) in column-major
      const C = new FloatXArray(n * n);
      for (let i = 0; i < n - 1; i++) {
        C[(i + 1) * n + i] = 1; // sub-diagonal
      }
      for (let i = 0; i < n; i++) {
        C[i] = -p[i + 1] / p[0]; // first row
      }

      // Get eigenvalues of companion matrix
      const bridge = getLapackBridge();
      if (bridge && bridge.eig) {
        const f64 = C instanceof Float64Array ? C : new Float64Array(C);
        const eigResult = bridge.eig(
          f64 as Float64Array,
          n,
          false,
          false,
          true
        );
        if (eigResult) {
          // Check if any eigenvalues are complex
          let hasComplex = false;
          for (let i = 0; i < n; i++) {
            if (Math.abs(eigResult.wi[i]) > 0) {
              hasComplex = true;
              break;
            }
          }

          if (hasComplex) {
            const realPart = new FloatXArray(n);
            const imagPart = new FloatXArray(n);
            for (let i = 0; i < n; i++) {
              realPart[i] = eigResult.wr[i];
              imagPart[i] = eigResult.wi[i];
            }
            return RTV.tensor(realPart, [n, 1], imagPart);
          }

          const result = new FloatXArray(n);
          for (let i = 0; i < n; i++) {
            result[i] = eigResult.wr[i];
          }
          return RTV.tensor(result, [n, 1]);
        }
      }

      // JS fallback for small polynomials
      if (n === 1) {
        return RTV.tensor(new FloatXArray([-p[1] / p[0]]), [1, 1]);
      }
      if (n === 2) {
        const a = p[0],
          b = p[1],
          c = p[2];
        const disc = b * b - 4 * a * c;
        if (disc >= 0) {
          const r1 = (-b + Math.sqrt(disc)) / (2 * a);
          const r2 = (-b - Math.sqrt(disc)) / (2 * a);
          return RTV.tensor(new FloatXArray([r1, r2]), [2, 1]);
        } else {
          const realPart = new FloatXArray([-b / (2 * a), -b / (2 * a)]);
          const imagPart = new FloatXArray([
            Math.sqrt(-disc) / (2 * a),
            -Math.sqrt(-disc) / (2 * a),
          ]);
          return RTV.tensor(realPart, [2, 1], imagPart);
        }
      }

      // General JS fallback: Durand-Kerner method
      return durandKernerRoots(p, n);
    })
  );

  // ── poly ────────────────────────────────────────────────────────────

  register(
    "poly",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("poly requires exactly 1 argument");

      const A = args[0];

      if (isRuntimeNumber(A)) {
        // poly(scalar) = [1, -scalar]
        return RTV.tensor(new FloatXArray([1, -A]), [1, 2]);
      }

      if (!isRuntimeTensor(A))
        throw new RuntimeError("poly: argument must be numeric");

      const [m, n] = tensorSize2D(A);

      if (m === 1 || n === 1) {
        // Vector of roots -> polynomial coefficients
        const roots = toFloatArray(A);
        const nr = roots.length;
        // Start with [1], then multiply by (x - r_i) for each root
        let coeffs = new FloatXArray(nr + 1);
        coeffs[0] = 1;
        for (let i = 0; i < nr; i++) {
          const newCoeffs = new FloatXArray(nr + 1);
          newCoeffs[0] = coeffs[0];
          for (let j = 1; j <= i + 1; j++) {
            newCoeffs[j] = coeffs[j] - roots[i] * coeffs[j - 1];
          }
          coeffs = newCoeffs;
        }
        return RTV.tensor(coeffs, [1, nr + 1]);
      }

      // Square matrix -> characteristic polynomial
      if (m !== n) throw new RuntimeError("poly: matrix must be square");

      // Get eigenvalues and compute poly from them
      const bridge = getLapackBridge();
      if (bridge && bridge.eig) {
        const f64 =
          A.data instanceof Float64Array ? A.data : new Float64Array(A.data);
        const eigResult = bridge.eig(
          f64 as Float64Array,
          n,
          false,
          false,
          true
        );
        if (eigResult) {
          const roots = new FloatXArray(n);
          for (let i = 0; i < n; i++) {
            roots[i] = eigResult.wr[i];
          }
          // Build polynomial from roots
          let coeffs = new FloatXArray(n + 1);
          coeffs[0] = 1;
          for (let i = 0; i < n; i++) {
            const newCoeffs = new FloatXArray(n + 1);
            newCoeffs[0] = coeffs[0];
            for (let j = 1; j <= i + 1; j++) {
              newCoeffs[j] = coeffs[j] - roots[i] * coeffs[j - 1];
            }
            coeffs = newCoeffs;
          }
          return RTV.tensor(coeffs, [1, n + 1]);
        }
      }

      // JS fallback: compute eigenvalues and build poly from them
      // For small matrices, compute eigenvalues directly
      if (n === 2) {
        // 2x2: eigenvalues from trace and determinant
        const a00 = A.data[0],
          a10 = A.data[1],
          a01 = A.data[m],
          a11 = A.data[m + 1];
        const tr = a00 + a11;
        const dt = a00 * a11 - a01 * a10;
        // char poly: x^2 - trace*x + det
        return RTV.tensor(new FloatXArray([1, -tr, dt]), [1, 3]);
      }

      throw new RuntimeError(
        "poly: matrix input requires LAPACK eigenvalue support for matrices larger than 2x2"
      );
    })
  );

  // ── cov ─────────────────────────────────────────────────────────────

  register(
    "cov",
    builtinSingle(args => {
      if (args.length < 1 || args.length > 2)
        throw new RuntimeError("cov requires 1 or 2 arguments");

      if (
        args.length === 2 &&
        isRuntimeTensor(args[0]) &&
        isRuntimeTensor(args[1])
      ) {
        // cov(x, y) — two vectors
        const x = toFloatArray(args[0]);
        const y = toFloatArray(args[1]);
        const n = x.length;
        if (n !== y.length)
          throw new RuntimeError("cov: x and y must have the same length");

        let mx = 0,
          my = 0;
        for (let i = 0; i < n; i++) {
          mx += x[i];
          my += y[i];
        }
        mx /= n;
        my /= n;

        let sxx = 0,
          sxy = 0,
          syy = 0;
        for (let i = 0; i < n; i++) {
          sxx += (x[i] - mx) * (x[i] - mx);
          sxy += (x[i] - mx) * (y[i] - my);
          syy += (y[i] - my) * (y[i] - my);
        }
        const d = n - 1;
        const result = new FloatXArray([sxx / d, sxy / d, sxy / d, syy / d]);
        return RTV.tensor(result, [2, 2]);
      }

      // cov(X) — single matrix or vector
      const X = args[0];
      if (isRuntimeNumber(X)) return RTV.num(0);

      if (!isRuntimeTensor(X))
        throw new RuntimeError("cov: argument must be numeric");

      const [m, ncols] = tensorSize2D(X);

      if (m === 1 || ncols === 1) {
        // Vector: return scalar variance
        const arr = toFloatArray(X);
        const n = arr.length;
        let mean = 0;
        for (let i = 0; i < n; i++) mean += arr[i];
        mean /= n;
        let s = 0;
        for (let i = 0; i < n; i++) s += (arr[i] - mean) * (arr[i] - mean);
        return RTV.num(s / (n - 1));
      }

      // Matrix: columns are variables, rows are observations
      // Compute means
      const means = new FloatXArray(ncols);
      for (let j = 0; j < ncols; j++) {
        let s = 0;
        for (let i = 0; i < m; i++) s += X.data[j * m + i];
        means[j] = s / m;
      }

      // Compute covariance matrix (ncols x ncols)
      const result = new FloatXArray(ncols * ncols);
      for (let p = 0; p < ncols; p++) {
        for (let q = 0; q < ncols; q++) {
          let s = 0;
          for (let i = 0; i < m; i++) {
            s +=
              (X.data[p * m + i] - means[p]) * (X.data[q * m + i] - means[q]);
          }
          result[q * ncols + p] = s / (m - 1);
        }
      }
      return RTV.tensor(result, [ncols, ncols]);
    })
  );

  // ── corrcoef ────────────────────────────────────────────────────────

  register(
    "corrcoef",
    builtinSingle(args => {
      if (args.length < 1 || args.length > 2)
        throw new RuntimeError("corrcoef requires 1 or 2 arguments");

      // Build the data matrix (each column is a variable)
      let data: FloatXArrayType;
      let m: number, ncols: number;

      if (args.length === 2) {
        const x = toFloatArray(args[0]);
        const y = toFloatArray(args[1]);
        const n = x.length;
        if (n !== y.length)
          throw new RuntimeError("corrcoef: x and y must have the same length");
        m = n;
        ncols = 2;
        data = new FloatXArray(n * 2);
        for (let i = 0; i < n; i++) {
          data[i] = x[i]; // col 0
          data[n + i] = y[i]; // col 1
        }
      } else {
        const X = args[0];
        if (!isRuntimeTensor(X))
          throw new RuntimeError("corrcoef: argument must be numeric");
        [m, ncols] = tensorSize2D(X);
        if (m === 1 || ncols === 1) {
          // Single vector: correlation is 1
          return RTV.num(1);
        }
        data = new FloatXArray(X.data);
      }

      // Compute means
      const means = new FloatXArray(ncols);
      for (let j = 0; j < ncols; j++) {
        let s = 0;
        for (let i = 0; i < m; i++) s += data[j * m + i];
        means[j] = s / m;
      }

      // Compute covariance matrix
      const covMat = new FloatXArray(ncols * ncols);
      for (let p = 0; p < ncols; p++) {
        for (let q = 0; q < ncols; q++) {
          let s = 0;
          for (let i = 0; i < m; i++) {
            s += (data[p * m + i] - means[p]) * (data[q * m + i] - means[q]);
          }
          covMat[q * ncols + p] = s / (m - 1);
        }
      }

      // Normalize to correlation: R(i,j) = C(i,j) / sqrt(C(i,i) * C(j,j))
      const result = new FloatXArray(ncols * ncols);
      for (let p = 0; p < ncols; p++) {
        for (let q = 0; q < ncols; q++) {
          const denom = Math.sqrt(
            covMat[p * ncols + p] * covMat[q * ncols + q]
          );
          result[q * ncols + p] =
            denom === 0 ? 0 : covMat[q * ncols + p] / denom;
        }
      }
      return RTV.tensor(result, [ncols, ncols]);
    })
  );

  // ── bitand ──────────────────────────────────────────────────────────

  register(
    "bitand",
    builtinSingle(args => {
      if (args.length !== 2)
        throw new RuntimeError("bitand requires exactly 2 arguments");
      return bitwiseOp(args[0], args[1], (a, b) => a & b);
    }),
    2
  );

  // ── bitor ───────────────────────────────────────────────────────────

  register(
    "bitor",
    builtinSingle(args => {
      if (args.length !== 2)
        throw new RuntimeError("bitor requires exactly 2 arguments");
      return bitwiseOp(args[0], args[1], (a, b) => a | b);
    }),
    2
  );

  // ── bitxor ──────────────────────────────────────────────────────────

  register(
    "bitxor",
    builtinSingle(args => {
      if (args.length !== 2)
        throw new RuntimeError("bitxor requires exactly 2 arguments");
      return bitwiseOp(args[0], args[1], (a, b) => a ^ b);
    }),
    2
  );

  // ── bitshift ────────────────────────────────────────────────────────

  register(
    "bitshift",
    builtinSingle(args => {
      if (args.length < 2 || args.length > 3)
        throw new RuntimeError("bitshift requires 2 or 3 arguments");

      const aVal = args[0];
      const kVal = args[1];

      const shift = (a: number, k: number): number => {
        const ai = Math.round(a);
        const ki = Math.round(k);
        return ki >= 0 ? ai << ki : ai >>> -ki;
      };

      if (isRuntimeNumber(aVal) && isRuntimeNumber(kVal)) {
        return RTV.num(shift(aVal, kVal));
      }
      if (isRuntimeTensor(aVal) && isRuntimeNumber(kVal)) {
        const result = new FloatXArray(aVal.data.length);
        for (let i = 0; i < aVal.data.length; i++) {
          result[i] = shift(aVal.data[i], kVal);
        }
        return RTV.tensor(result, [...aVal.shape]);
      }
      if (isRuntimeNumber(aVal) && isRuntimeTensor(kVal)) {
        const result = new FloatXArray(kVal.data.length);
        for (let i = 0; i < kVal.data.length; i++) {
          result[i] = shift(aVal, kVal.data[i]);
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
    }),
    2
  );

  // ── eps ──────────────────────────────────────────────────────────────

  register(
    "eps",
    builtinSingle(args => {
      const DOUBLE_EPS = 2.220446049250313e-16;
      const SINGLE_EPS = 1.1920928955078125e-7;
      if (args.length === 0) return RTV.num(DOUBLE_EPS);
      const v = args[0];
      if (isRuntimeChar(v)) {
        if (v.value === "double") return RTV.num(DOUBLE_EPS);
        if (v.value === "single") return RTV.num(SINGLE_EPS);
        throw new RuntimeError(`eps: unknown data type '${v.value}'`);
      }
      if (isRuntimeString(v)) {
        if (v === "double") return RTV.num(DOUBLE_EPS);
        if (v === "single") return RTV.num(SINGLE_EPS);
        throw new RuntimeError(`eps: unknown data type '${v}'`);
      }
      if (isRuntimeNumber(v)) return RTV.num(epsOfScalar(v as number));
      if (isRuntimeTensor(v)) {
        const n = v.data.length;
        const out = new FloatXArray(n);
        for (let i = 0; i < n; i++) out[i] = epsOfScalar(v.data[i]);
        return RTV.tensor(out, v.shape.slice());
      }
      throw new RuntimeError("eps: unsupported argument type");
    }),
    1
  );
}

/** Helper for bitwise binary operations (bitand, bitor, bitxor) */
function bitwiseOp(
  a: RuntimeValue,
  b: RuntimeValue,
  op: (a: number, b: number) => number
): RuntimeValue {
  if (isRuntimeNumber(a) && isRuntimeNumber(b)) {
    return RTV.num(op(Math.round(a), Math.round(b)));
  }
  if (isRuntimeTensor(a) && isRuntimeNumber(b)) {
    const bv = Math.round(b);
    const result = new FloatXArray(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      result[i] = op(Math.round(a.data[i]), bv);
    }
    return RTV.tensor(result, [...a.shape]);
  }
  if (isRuntimeNumber(a) && isRuntimeTensor(b)) {
    const av = Math.round(a);
    const result = new FloatXArray(b.data.length);
    for (let i = 0; i < b.data.length; i++) {
      result[i] = op(av, Math.round(b.data[i]));
    }
    return RTV.tensor(result, [...b.shape]);
  }
  if (isRuntimeTensor(a) && isRuntimeTensor(b)) {
    if (a.data.length !== b.data.length)
      throw new RuntimeError("Bitwise operation: arrays must be the same size");
    const result = new FloatXArray(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      result[i] = op(Math.round(a.data[i]), Math.round(b.data[i]));
    }
    return RTV.tensor(result, [...a.shape]);
  }
  throw new RuntimeError("Bitwise operation: arguments must be numeric");
}

/** Durand-Kerner method for finding polynomial roots (JS fallback) */
function durandKernerRoots(p: FloatXArrayType, n: number): RuntimeValue {
  // Normalize so leading coefficient is 1
  const a = new Float64Array(n + 1);
  for (let i = 0; i <= n; i++) a[i] = p[i] / p[0];

  // Initial guesses: spread around a circle
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  const radius = 1 + Math.max(...Array.from(a).map(Math.abs));
  for (let i = 0; i < n; i++) {
    const angle = (2 * Math.PI * i) / n + 0.4;
    re[i] = radius * Math.cos(angle);
    im[i] = radius * Math.sin(angle);
  }

  const maxIter = 1000;
  const tol = 1e-14;

  for (let iter = 0; iter < maxIter; iter++) {
    let maxDelta = 0;

    for (let i = 0; i < n; i++) {
      // Evaluate polynomial at z_i
      let pRe = a[0],
        pIm = 0;
      for (let k = 1; k <= n; k++) {
        const tmpRe = pRe * re[i] - pIm * im[i] + a[k];
        const tmpIm = pRe * im[i] + pIm * re[i];
        pRe = tmpRe;
        pIm = tmpIm;
      }

      // Compute product of (z_i - z_j) for j != i
      let dRe = 1,
        dIm = 0;
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        const diffRe = re[i] - re[j];
        const diffIm = im[i] - im[j];
        const tmpRe = dRe * diffRe - dIm * diffIm;
        const tmpIm = dRe * diffIm + dIm * diffRe;
        dRe = tmpRe;
        dIm = tmpIm;
      }

      // delta = p(z_i) / prod(z_i - z_j)
      const denom = dRe * dRe + dIm * dIm;
      if (denom < 1e-30) continue;
      const deltaRe = (pRe * dRe + pIm * dIm) / denom;
      const deltaIm = (pIm * dRe - pRe * dIm) / denom;

      re[i] -= deltaRe;
      im[i] -= deltaIm;

      const mag = Math.sqrt(deltaRe * deltaRe + deltaIm * deltaIm);
      if (mag > maxDelta) maxDelta = mag;
    }

    if (maxDelta < tol) break;
  }

  // Check if all roots are real
  let allReal = true;
  for (let i = 0; i < n; i++) {
    if (Math.abs(im[i]) > 1e-10) {
      allReal = false;
      break;
    }
  }

  if (allReal) {
    const result = new FloatXArray(n);
    for (let i = 0; i < n; i++) result[i] = re[i];
    return RTV.tensor(result, [n, 1]);
  }

  const realPart = new FloatXArray(re);
  const imagPart = new FloatXArray(im);
  return RTV.tensor(realPart, [n, 1], imagPart);
}
