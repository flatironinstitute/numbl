/**
 * Introspection builtin functions
 */

import { RuntimeValue, RTV, toNumber, RuntimeError } from "../runtime/index.js";
import {
  FloatXArray,
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeString,
  isRuntimeChar,
  isRuntimeTensor,
  isRuntimeCell,
  isRuntimeStruct,
  isRuntimeFunction,
  isRuntimeClassInstance,
  isRuntimeComplexNumber,
  isRuntimeDummyHandle,
  isRuntimeStructArray,
  isRuntimeSparseMatrix,
  RuntimeStruct,
} from "../runtime/types.js";
import { register, builtinSingle } from "./registry.js";
import { sparseToDense } from "./sparse-arithmetic.js";

/** Extract string content from either a RuntimeChar or RuntimeString. */
function charOrStringValue(v: RuntimeValue): string {
  if (isRuntimeChar(v)) return v.value;
  if (isRuntimeString(v)) return v;
  throw new RuntimeError("Expected char or string");
}

/** Get numeric scalar value from number, logical, or 1-element real tensor. Returns NaN if not a numeric scalar. */
function numericScalarVal(v: RuntimeValue): number {
  if (isRuntimeNumber(v)) return v;
  if (isRuntimeLogical(v)) return v ? 1 : 0;
  if (isRuntimeTensor(v) && v.data.length === 1 && !v.imag) return v.data[0];
  return NaN;
}

/** Convert a number/logical to a 1x1 tensor for uniform tensor comparison. */
function asTensor(
  v: RuntimeValue
): import("../runtime/types.js").RuntimeTensor {
  if (isRuntimeTensor(v)) return v;
  const data = new FloatXArray(1);
  data[0] = isRuntimeNumber(v) ? v : v ? 1 : 0;
  return { kind: "tensor" as const, data, shape: [1, 1], _rc: 1 };
}

function valuesEqual(a: RuntimeValue, b: RuntimeValue): boolean {
  // char and string with the same text are considered equal by isequal
  if (
    (isRuntimeChar(a) || isRuntimeString(a)) &&
    (isRuntimeChar(b) || isRuntimeString(b))
  ) {
    return charOrStringValue(a) === charOrStringValue(b);
  }
  // Densify sparse for comparison
  if (isRuntimeSparseMatrix(a)) return valuesEqual(sparseToDense(a), b);
  if (isRuntimeSparseMatrix(b)) return valuesEqual(a, sparseToDense(b));
  // Numeric types (number, logical, tensor) are all comparable by value
  const aNum = isRuntimeNumber(a) || isRuntimeLogical(a) || isRuntimeTensor(a);
  const bNum = isRuntimeNumber(b) || isRuntimeLogical(b) || isRuntimeTensor(b);
  if (aNum && bNum) {
    // Fast path: both are scalars
    const sa = numericScalarVal(a);
    const sb = numericScalarVal(b);
    if (!isNaN(sa) && !isNaN(sb)) return sa === sb;
    // General path: compare as tensors
    const ta = asTensor(a);
    const tb = asTensor(b);
    if (ta.data.length !== tb.data.length) return false;
    if (ta.shape.length !== tb.shape.length) return false;
    for (let i = 0; i < ta.shape.length; i++) {
      if (ta.shape[i] !== tb.shape[i]) return false;
    }
    for (let i = 0; i < ta.data.length; i++) {
      if (ta.data[i] !== tb.data[i]) return false;
    }
    if (ta.imag || tb.imag) {
      const ai = ta.imag;
      const bi = tb.imag;
      if (!ai || !bi) return false;
      for (let i = 0; i < ai.length; i++) {
        if (ai[i] !== bi[i]) return false;
      }
    }
    return true;
  }
  if (isRuntimeCell(a) && isRuntimeCell(b)) {
    if (a.shape.length !== b.shape.length) return false;
    for (let i = 0; i < a.shape.length; i++) {
      if (a.shape[i] !== b.shape[i]) return false;
    }
    if (a.data.length !== b.data.length) return false;
    for (let i = 0; i < a.data.length; i++) {
      if (!valuesEqual(a.data[i], b.data[i])) return false;
    }
    return true;
  }
  if (isRuntimeStruct(a) && isRuntimeStruct(b)) {
    if (a.fields.size !== b.fields.size) return false;
    for (const [key, val] of a.fields) {
      if (!b.fields.has(key)) return false;
      if (!valuesEqual(val, b.fields.get(key)!)) return false;
    }
    return true;
  }
  if (isRuntimeComplexNumber(a) && isRuntimeComplexNumber(b)) {
    return a.re === b.re && a.im === b.im;
  }
  return false;
}

/** Extract a numeric array from a RuntimeValue (for sparse triplet construction). */
function toNumericArray(v: RuntimeValue, name: string): number[] {
  if (isRuntimeNumber(v)) return [v];
  if (isRuntimeLogical(v)) return [v ? 1 : 0];
  if (isRuntimeTensor(v)) return Array.from(v.data);
  if (isRuntimeComplexNumber(v)) return [v.re];
  throw new RuntimeError(`${name}: arguments must be numeric`);
}

/** Extract imaginary parts from a RuntimeValue (undefined if all real). */
function toImagArray(v: RuntimeValue): number[] | undefined {
  if (isRuntimeComplexNumber(v)) return [v.im];
  if (isRuntimeTensor(v) && v.imag) return Array.from(v.imag);
  return undefined;
}

/** Build a sparse matrix from COO triplets (1-based i, j; summing duplicates). */
function buildSparseFromTriplets(
  iArr: number[],
  jArr: number[],
  vArr: number[],
  m: number,
  n: number,
  vImag?: number[]
): import("../runtime/types.js").RuntimeSparseMatrix {
  const nnz = iArr.length;
  const isComplex = vImag !== undefined;
  // Build (col0, row0, value) triplets with 0-based indices
  const triplets: { col: number; row: number; re: number; im: number }[] = [];
  for (let k = 0; k < nnz; k++) {
    triplets.push({
      col: jArr[k] - 1,
      row: iArr[k] - 1,
      re: vArr[k],
      im: isComplex ? vImag[k] : 0,
    });
  }
  // Sort by (col, row)
  triplets.sort((a, b) => a.col - b.col || a.row - b.row);
  // Merge duplicates by summing, tracking column for each merged entry
  const mergedIr: number[] = [];
  const mergedPr: number[] = [];
  const mergedPi: number[] = [];
  const mergedCols: number[] = [];
  let prevCol = -1;
  let prevRow = -1;
  for (const t of triplets) {
    if (t.col === prevCol && t.row === prevRow) {
      mergedPr[mergedPr.length - 1] += t.re;
      if (isComplex) mergedPi[mergedPi.length - 1] += t.im;
    } else {
      mergedIr.push(t.row);
      mergedPr.push(t.re);
      if (isComplex) mergedPi.push(t.im);
      mergedCols.push(t.col);
      prevCol = t.col;
      prevRow = t.row;
    }
  }
  // Build jc from column list
  const jc = new Int32Array(n + 1);
  let ci = 0;
  for (let c = 0; c < n; c++) {
    jc[c] = ci;
    while (ci < mergedCols.length && mergedCols[ci] === c) ci++;
  }
  jc[n] = ci;
  return RTV.sparseMatrix(
    m,
    n,
    new Int32Array(mergedIr),
    jc,
    new Float64Array(mergedPr),
    isComplex ? new Float64Array(mergedPi) : undefined
  );
}

export function registerIntrospectionFunctions(): void {
  register(
    "size",
    builtinSingle((args, nargout) => {
      if (args.length < 1)
        throw new RuntimeError("size requires at least 1 argument");
      const v = args[0];
      let shape: number[];
      if (isRuntimeNumber(v) || isRuntimeLogical(v)) shape = [1, 1];
      else if (isRuntimeSparseMatrix(v)) shape = [v.m, v.n];
      else if (isRuntimeTensor(v))
        shape = v.shape.length >= 2 ? v.shape : [1, ...v.shape];
      else if (isRuntimeCell(v)) shape = v.shape;
      else if (isRuntimeChar(v))
        shape =
          v.shape ?? (v.value.length === 0 ? [0, 0] : [1, v.value.length]);
      else if (isRuntimeString(v)) shape = [1, 1];
      else shape = [1, 1];

      if (args.length === 2) {
        const dim = Math.round(toNumber(args[1]));
        return RTV.num(dim > 0 && dim <= shape.length ? shape[dim - 1] : 1);
      }

      if (nargout > 1) {
        // Return multiple outputs
        const result: RuntimeValue[] = [];
        for (let i = 0; i < nargout; i++) {
          result.push(RTV.num(i < shape.length ? shape[i] : 1));
        }
        return result;
      }

      return RTV.tensor(new FloatXArray(shape), [1, shape.length]);
    })
  );

  register(
    "length",
    builtinSingle(
      args => {
        if (args.length !== 1)
          throw new RuntimeError("length requires 1 argument");
        const v = args[0];
        if (isRuntimeNumber(v) || isRuntimeLogical(v)) return RTV.num(1);
        if (isRuntimeSparseMatrix(v)) return RTV.num(Math.max(v.m, v.n));
        if (isRuntimeTensor(v))
          return RTV.num(v.data.length === 0 ? 0 : Math.max(...v.shape));
        if (isRuntimeCell(v))
          return RTV.num(v.data.length === 0 ? 0 : Math.max(...v.shape));
        if (isRuntimeChar(v)) {
          const s =
            v.shape ?? (v.value.length === 0 ? [0, 0] : [1, v.value.length]);
          return RTV.num(s.length === 0 ? 0 : Math.max(...s));
        }
        if (isRuntimeString(v)) return RTV.num(1);
        if (isRuntimeStructArray(v)) return RTV.num(v.elements.length);
        return RTV.num(1);
      },
      { outputType: { kind: "Number" } }
    )
  );

  register(
    "numel",
    builtinSingle(
      args => {
        if (args.length !== 1)
          throw new RuntimeError("numel requires 1 argument");
        const v = args[0];
        if (isRuntimeNumber(v) || isRuntimeLogical(v)) return RTV.num(1);
        if (isRuntimeSparseMatrix(v)) return RTV.num(v.m * v.n);
        if (isRuntimeTensor(v)) return RTV.num(v.data.length);
        if (isRuntimeCell(v)) return RTV.num(v.data.length);
        if (isRuntimeChar(v)) return RTV.num(v.value.length);
        if (isRuntimeString(v)) return RTV.num(1);
        if (isRuntimeStructArray(v)) return RTV.num(v.elements.length);
        return RTV.num(1);
      },
      { outputType: { kind: "Number" } }
    )
  );

  register(
    "ndims",
    builtinSingle(
      args => {
        if (args.length !== 1)
          throw new RuntimeError("ndims requires 1 argument");
        const v = args[0];
        if (isRuntimeTensor(v)) return RTV.num(Math.max(2, v.shape.length));
        return RTV.num(2);
      },
      { outputType: { kind: "Number" } }
    )
  );

  register(
    "isempty",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("isempty requires 1 argument");
      const v = args[0];
      if (isRuntimeSparseMatrix(v)) return RTV.logical(v.m === 0 || v.n === 0);
      if (isRuntimeTensor(v)) return RTV.logical(v.data.length === 0);
      if (isRuntimeCell(v)) return RTV.logical(v.data.length === 0);
      if (isRuntimeChar(v)) return RTV.logical(v.value.length === 0);
      if (isRuntimeString(v)) return RTV.logical(false); // string scalar is never empty
      if (isRuntimeStructArray(v)) return RTV.logical(v.elements.length === 0);
      return RTV.logical(false);
    })
  );

  // ── Type predicate factory ───────────────────────────────────────────
  const typePred = (name: string, pred: (v: RuntimeValue) => boolean) =>
    register(
      name,
      builtinSingle(args => {
        if (args.length !== 1)
          throw new RuntimeError(`${name} requires 1 argument`);
        return RTV.logical(pred(args[0]));
      })
    );

  typePred(
    "isscalar",
    v =>
      isRuntimeNumber(v) ||
      isRuntimeLogical(v) ||
      isRuntimeComplexNumber(v) ||
      (isRuntimeTensor(v) && v.data.length === 1)
  );

  /** Helper: get the effective 2D+ shape of a value */
  const getShape = (v: RuntimeValue): number[] => {
    if (isRuntimeNumber(v) || isRuntimeLogical(v)) return [1, 1];
    if (isRuntimeTensor(v))
      return v.shape.length >= 2 ? v.shape : [1, ...v.shape];
    if (isRuntimeChar(v))
      return v.shape ?? (v.value.length === 0 ? [0, 0] : [1, v.value.length]);
    return [1, 1];
  };

  typePred("isvector", v => {
    const shape = getShape(v);
    return shape.filter(d => d > 1).length <= 1;
  });
  typePred("isrow", v => {
    const shape = getShape(v);
    return shape.length === 2 && shape[0] === 1;
  });
  typePred("iscolumn", v => {
    const shape = getShape(v);
    return shape.length === 2 && shape[1] === 1;
  });
  typePred("ismatrix", v => getShape(v).length <= 2);
  typePred(
    "isfloat",
    v =>
      isRuntimeNumber(v) ||
      isRuntimeTensor(v) ||
      isRuntimeComplexNumber(v) ||
      isRuntimeSparseMatrix(v)
  );
  typePred("isinteger", () => false);
  typePred(
    "isnumeric",
    v =>
      isRuntimeNumber(v) ||
      isRuntimeTensor(v) ||
      isRuntimeComplexNumber(v) ||
      isRuntimeSparseMatrix(v) ||
      (isRuntimeClassInstance(v) && v._builtinData !== undefined)
  );
  typePred(
    "islogical",
    v => isRuntimeLogical(v) || (isRuntimeTensor(v) && v._isLogical === true)
  );
  typePred("ischar", v => isRuntimeChar(v));
  typePred("isstring", v => isRuntimeString(v));
  typePred("iscell", v => isRuntimeCell(v));
  typePred("isstruct", v => isRuntimeStruct(v) || isRuntimeStructArray(v));

  register(
    "isequal",
    builtinSingle(args => {
      if (args.length < 2)
        throw new RuntimeError("isequal requires at least 2 arguments");
      for (let i = 1; i < args.length; i++) {
        if (!valuesEqual(args[0], args[i])) return RTV.logical(false);
      }
      return RTV.logical(true);
    })
  );

  register(
    "class",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("class requires 1 argument");
      const v = args[0];
      if (isRuntimeNumber(v)) return RTV.string("double");
      if (isRuntimeLogical(v)) return RTV.string("logical");
      if (isRuntimeString(v)) return RTV.string("string");
      if (isRuntimeChar(v)) return RTV.string("char");
      if (isRuntimeTensor(v))
        return RTV.string(v._isLogical ? "logical" : "double");
      if (isRuntimeCell(v)) return RTV.string("cell");
      if (isRuntimeStruct(v)) return RTV.string("struct");
      if (isRuntimeFunction(v)) return RTV.string("function_handle");
      if (isRuntimeClassInstance(v)) return RTV.string(v.className);
      if (isRuntimeComplexNumber(v)) return RTV.string("double");
      if (isRuntimeSparseMatrix(v)) return RTV.string("double");
      if (isRuntimeDummyHandle(v)) return RTV.string("dummy_handle");
      if (isRuntimeStructArray(v)) return RTV.string("struct");
      return RTV.string("unknown");
    })
  );

  // isa - stub for IR type inference; actual implementation is $rt.isa() via codegen
  register(
    "isa",
    builtinSingle(args => {
      if (args.length !== 2) throw new RuntimeError("isa requires 2 arguments");
      throw new RuntimeError("isa should be handled by $rt.isa() via codegen");
    })
  );

  register(
    "issparse",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("issparse requires 1 argument");
      return RTV.logical(isRuntimeSparseMatrix(args[0]));
    })
  );

  register(
    "full",
    builtinSingle(args => {
      if (args.length !== 1) throw new RuntimeError("full requires 1 argument");
      const v = args[0];
      if (!isRuntimeSparseMatrix(v)) return v; // passthrough for non-sparse
      const { m, n, ir, jc, pr, pi } = v;
      const data = new FloatXArray(m * n);
      const imag = pi ? new FloatXArray(m * n) : undefined;
      for (let col = 0; col < n; col++) {
        for (let k = jc[col]; k < jc[col + 1]; k++) {
          data[col * m + ir[k]] = pr[k]; // column-major
          if (imag && pi) imag[col * m + ir[k]] = pi[k];
        }
      }
      return RTV.tensor(data, [m, n], imag);
    })
  );

  register(
    "sparse",
    builtinSingle(args => {
      if (args.length === 1) {
        const v = args[0];
        if (isRuntimeSparseMatrix(v)) return v;
        // dense to sparse: scan column-major for nonzeros
        if (isRuntimeNumber(v)) {
          if (v === 0)
            return RTV.sparseMatrix(
              1,
              1,
              new Int32Array(0),
              new Int32Array(2),
              new Float64Array(0)
            );
          return RTV.sparseMatrix(
            1,
            1,
            new Int32Array([0]),
            new Int32Array([0, 1]),
            new Float64Array([v])
          );
        }
        if (isRuntimeLogical(v)) {
          const nv = v ? 1 : 0;
          if (nv === 0)
            return RTV.sparseMatrix(
              1,
              1,
              new Int32Array(0),
              new Int32Array(2),
              new Float64Array(0)
            );
          return RTV.sparseMatrix(
            1,
            1,
            new Int32Array([0]),
            new Int32Array([0, 1]),
            new Float64Array([nv])
          );
        }
        if (isRuntimeComplexNumber(v)) {
          if (v.re === 0 && v.im === 0)
            return RTV.sparseMatrix(
              1,
              1,
              new Int32Array(0),
              new Int32Array(2),
              new Float64Array(0)
            );
          return RTV.sparseMatrix(
            1,
            1,
            new Int32Array([0]),
            new Int32Array([0, 1]),
            new Float64Array([v.re]),
            v.im !== 0 ? new Float64Array([v.im]) : undefined
          );
        }
        if (!isRuntimeTensor(v))
          throw new RuntimeError("sparse: argument must be numeric");
        const rows = v.shape[0] || 1;
        const cols = v.shape.length >= 2 ? v.shape[1] : 1;
        const hasImag = v.imag !== undefined;
        const irList: number[] = [];
        const prList: number[] = [];
        const piList: number[] | undefined = hasImag ? [] : undefined;
        const jcArr = new Int32Array(cols + 1);
        for (let c = 0; c < cols; c++) {
          jcArr[c] = irList.length;
          for (let r = 0; r < rows; r++) {
            const idx = c * rows + r;
            const re = v.data[idx];
            const im = hasImag ? v.imag![idx] : 0;
            if (re !== 0 || im !== 0) {
              irList.push(r);
              prList.push(re);
              if (piList) piList.push(im);
            }
          }
        }
        jcArr[cols] = irList.length;
        return RTV.sparseMatrix(
          rows,
          cols,
          new Int32Array(irList),
          jcArr,
          new Float64Array(prList),
          piList ? new Float64Array(piList) : undefined
        );
      }
      if (args.length === 2) {
        // sparse(m, n) — sparse zero matrix
        const m = Math.round(toNumber(args[0]));
        const n = Math.round(toNumber(args[1]));
        return RTV.sparseMatrix(
          m,
          n,
          new Int32Array(0),
          new Int32Array(n + 1),
          new Float64Array(0)
        );
      }
      if (args.length >= 3) {
        // sparse(i, j, v) or sparse(i, j, v, m, n) or sparse(i, j, v, m, n, nzmax)
        const iArg = args[0];
        const jArg = args[1];
        const vArg = args[2];
        // Extract index arrays (1-based)
        const iArr = toNumericArray(iArg, "sparse");
        const jArr = toNumericArray(jArg, "sparse");
        const len = Math.max(iArr.length, jArr.length);
        // v can be scalar (broadcast) or array
        let vArr: number[];
        let vImag: number[] | undefined;
        if (isRuntimeNumber(vArg)) {
          vArr = new Array(len).fill(vArg);
        } else if (isRuntimeComplexNumber(vArg)) {
          vArr = new Array(len).fill(vArg.re);
          vImag = new Array(len).fill(vArg.im);
        } else if (isRuntimeLogical(vArg)) {
          vArr = new Array(len).fill(vArg ? 1 : 0);
        } else {
          vArr = toNumericArray(vArg, "sparse");
          vImag = toImagArray(vArg);
        }
        if (iArr.length !== jArr.length || iArr.length !== vArr.length) {
          // Allow scalar i or j to broadcast
          if (iArr.length === 1 && jArr.length === vArr.length) {
            const iv = iArr[0];
            iArr.length = 0;
            for (let k = 0; k < vArr.length; k++) iArr.push(iv);
          } else if (jArr.length === 1 && iArr.length === vArr.length) {
            const jv = jArr[0];
            jArr.length = 0;
            for (let k = 0; k < vArr.length; k++) jArr.push(jv);
          } else {
            throw new RuntimeError("sparse: i, j, v must have the same length");
          }
        }
        // Determine dimensions
        let m: number, n: number;
        if (args.length >= 5) {
          m = Math.round(toNumber(args[3]));
          n = Math.round(toNumber(args[4]));
        } else {
          m = 0;
          n = 0;
          for (let k = 0; k < iArr.length; k++) {
            if (iArr[k] > m) m = iArr[k];
            if (jArr[k] > n) n = jArr[k];
          }
        }
        // nzmax (args[5]) is ignored — it's a pre-allocation hint
        return buildSparseFromTriplets(iArr, jArr, vArr, m, n, vImag);
      }
      throw new RuntimeError("sparse: unsupported call signature");
    })
  );

  // spconvert: convert triplet-format matrix to sparse
  // spconvert(S) where S is n×3 [i j v] or n×4 [i j re im]
  register(
    "spconvert",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("spconvert requires 1 argument");
      const S = args[0];
      if (!isRuntimeTensor(S))
        throw new RuntimeError("spconvert: argument must be a matrix");
      const nrows = S.shape[0];
      const ncols = S.shape.length >= 2 ? S.shape[1] : 1;
      if (ncols < 3)
        throw new RuntimeError("spconvert: input must have at least 3 columns");

      const iArr: number[] = [];
      const jArr: number[] = [];
      const vArr: number[] = [];
      let m = 0;
      let n = 0;

      for (let k = 0; k < nrows; k++) {
        const i = S.data[k]; // column-major: col 0
        const j = S.data[k + nrows]; // col 1
        const v = S.data[k + 2 * nrows]; // col 2
        if (i > m) m = i;
        if (j > n) n = j;
        if (v !== 0) {
          iArr.push(i);
          jArr.push(j);
          vArr.push(v);
        }
      }
      return buildSparseFromTriplets(iArr, jArr, vArr, m, n);
    })
  );

  // This is a placeholder that will be specially handled by the executor
  register(
    "builtin",
    builtinSingle(args => {
      if (args.length < 1)
        throw new RuntimeError("builtin requires at least 1 argument");
      throw new RuntimeError(
        "builtin function should be handled specially by the executor"
      );
    })
  );

  // verLessThan: always return false (numbl aims to be like modern MATLAB)
  register(
    "verLessThan",
    builtinSingle(args => {
      if (args.length !== 2)
        throw new RuntimeError("verLessThan requires 2 arguments");
      return RTV.logical(false);
    })
  );

  // substruct: create subscript structure for subsref/subsasgn
  // substruct('.','field','()','indices',...) -> struct_array with {type, subs}
  register(
    "substruct",
    builtinSingle(args => {
      if (args.length < 2 || args.length % 2 !== 0)
        throw new RuntimeError(
          "substruct requires pairs of (type, subs) arguments"
        );
      const elements: RuntimeStruct[] = [];
      for (let i = 0; i < args.length; i += 2) {
        const typeArg = args[i];
        const subsArg = args[i + 1];
        if (!isRuntimeChar(typeArg) && !isRuntimeString(typeArg))
          throw new RuntimeError("substruct: type must be a string");
        const typeStr = charOrStringValue(typeArg);
        if (typeStr !== "." && typeStr !== "()" && typeStr !== "{}")
          throw new RuntimeError(
            `substruct: type must be '.', '()', or '{}', got '${typeStr}'`
          );
        // For '.', subs is a field name string
        // For '()' and '{}', subs should be wrapped in a cell array
        let subs: RuntimeValue;
        if (typeStr === ".") {
          subs = subsArg;
        } else {
          // Wrap in cell if not already a cell
          if (isRuntimeCell(subsArg)) {
            subs = subsArg;
          } else {
            subs = RTV.cell([subsArg], [1, 1]);
          }
        }
        elements.push(RTV.struct({ type: RTV.char(typeStr), subs }));
      }
      return RTV.structArray(["type", "subs"], elements);
    })
  );

  // subsref / subsasgn: registered here so the lowerer recognises them as
  // builtins (prevents the single-candidate optimisation from bypassing
  // runtime dispatch when a class overloads these).  The actual runtime
  // implementations are in runtime.ts (initBuiltins) and override these.
  register(
    "subsref",
    builtinSingle(() => {
      throw new RuntimeError("subsref: should be handled by runtime override");
    })
  );
  register(
    "subsasgn",
    builtinSingle(() => {
      throw new RuntimeError("subsasgn: should be handled by runtime override");
    })
  );
}
