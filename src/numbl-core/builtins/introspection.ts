/**
 * Introspection builtin functions
 */

import {
  RuntimeValue,
  RTV,
  toNumber,
  RuntimeError,
  colMajorIndex,
  tensorSize2D,
} from "../runtime/index.js";
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
  RuntimeStruct,
} from "../runtime/types.js";
import { register, builtinSingle } from "./registry.js";

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

export function registerIntrospectionFunctions(): void {
  register(
    "size",
    builtinSingle((args, nargout) => {
      if (args.length < 1)
        throw new RuntimeError("size requires at least 1 argument");
      const v = args[0];
      let shape: number[];
      if (isRuntimeNumber(v) || isRuntimeLogical(v)) shape = [1, 1];
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
      if (isRuntimeTensor(v)) return RTV.logical(v.data.length === 0);
      if (isRuntimeCell(v)) return RTV.logical(v.data.length === 0);
      if (isRuntimeChar(v)) return RTV.logical(v.value.length === 0);
      if (isRuntimeString(v)) return RTV.logical(false); // string scalar is never empty
      if (isRuntimeStructArray(v)) return RTV.logical(v.elements.length === 0);
      return RTV.logical(false);
    })
  );

  register(
    "isscalar",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("isscalar requires 1 argument");
      const v = args[0];
      if (
        isRuntimeNumber(v) ||
        isRuntimeLogical(v) ||
        isRuntimeComplexNumber(v)
      )
        return RTV.logical(true);
      if (isRuntimeTensor(v)) return RTV.logical(v.data.length === 1);
      return RTV.logical(false);
    })
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

  register(
    "isvector",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("isvector requires 1 argument");
      const shape = getShape(args[0]);
      // A vector has at most one dimension > 1
      const dimsGt1 = shape.filter(d => d > 1).length;
      return RTV.logical(dimsGt1 <= 1);
    })
  );

  register(
    "isrow",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("isrow requires 1 argument");
      const shape = getShape(args[0]);
      // Row: exactly 2D with shape[0] === 1
      return RTV.logical(shape.length === 2 && shape[0] === 1);
    })
  );

  register(
    "iscolumn",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("iscolumn requires 1 argument");
      const shape = getShape(args[0]);
      // Column: exactly 2D with shape[1] === 1
      return RTV.logical(shape.length === 2 && shape[1] === 1);
    })
  );

  register(
    "ismatrix",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("ismatrix requires 1 argument");
      const shape = getShape(args[0]);
      // Matrix: ndims <= 2 (no dimension beyond 2nd)
      return RTV.logical(shape.length <= 2);
    })
  );

  register(
    "isfloat",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("isfloat requires 1 argument");
      const v = args[0];
      // In numbl, numbers and tensors are always double (float)
      return RTV.logical(
        isRuntimeNumber(v) || isRuntimeTensor(v) || isRuntimeComplexNumber(v)
      );
    })
  );

  register(
    "isinteger",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("isinteger requires 1 argument");
      // numbl has no integer types (int8, uint16, etc.), everything is double
      return RTV.logical(false);
    })
  );

  register(
    "isnumeric",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("isnumeric requires 1 argument");
      const v = args[0];
      return RTV.logical(
        isRuntimeNumber(v) ||
          isRuntimeTensor(v) ||
          isRuntimeComplexNumber(v) ||
          (isRuntimeClassInstance(v) && v._builtinData !== undefined)
      );
    })
  );

  register(
    "islogical",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("islogical requires 1 argument");
      const v = args[0];
      return RTV.logical(
        isRuntimeLogical(v) || (isRuntimeTensor(v) && v._isLogical === true)
      );
    })
  );

  register(
    "ischar",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("ischar requires 1 argument");
      return RTV.logical(isRuntimeChar(args[0]));
    })
  );

  register(
    "isstring",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("isstring requires 1 argument");
      return RTV.logical(isRuntimeString(args[0]));
    })
  );

  register(
    "iscell",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("iscell requires 1 argument");
      return RTV.logical(isRuntimeCell(args[0]));
    })
  );

  register(
    "isstruct",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("isstruct requires 1 argument");
      return RTV.logical(
        isRuntimeStruct(args[0]) || isRuntimeStructArray(args[0])
      );
    })
  );

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

  // Sparse matrices are not supported; issparse always returns false
  register(
    "issparse",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("issparse requires 1 argument");
      return RTV.logical(false);
    })
  );

  // Sparse matrices are not supported; full() is a passthrough
  register(
    "full",
    builtinSingle(args => {
      if (args.length !== 1) throw new RuntimeError("full requires 1 argument");
      return args[0];
    })
  );

  // Sparse matrices are not supported; sparse() is a passthrough/zeros constructor.
  // sparse(S)    → returns S unchanged (dense passthrough)
  // sparse(m, n) → returns an m-by-n zero matrix
  register(
    "sparse",
    builtinSingle(args => {
      if (args.length === 1) return args[0]; // passthrough
      if (args.length === 2) {
        const m = Math.round(toNumber(args[0]));
        const n = Math.round(toNumber(args[1]));
        return RTV.tensor(new FloatXArray(m * n), [m, n]);
      }
      throw new RuntimeError("sparse: unsupported call signature");
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

  // issymmetric(A) — true if A == A.' (within tolerance)
  // issymmetric(A, 'skew') — true if A == -A.'
  register(
    "issymmetric",
    builtinSingle(args => {
      if (args.length < 1 || args.length > 2)
        throw new RuntimeError("issymmetric requires 1 or 2 arguments");

      const A = args[0];

      // Scalar is always symmetric
      if (isRuntimeNumber(A) || isRuntimeLogical(A)) return RTV.logical(true);
      if (isRuntimeComplexNumber(A)) return RTV.logical(true);

      if (!isRuntimeTensor(A))
        throw new RuntimeError("issymmetric: argument must be numeric");

      const [m, n] = tensorSize2D(A);
      if (m !== n) return RTV.logical(false);

      // Parse optional 'skew' flag
      let skew = false;
      if (args.length === 2) {
        const opt = args[1];
        let s: string | undefined;
        if (isRuntimeString(opt)) s = opt.toLowerCase();
        else if (isRuntimeChar(opt)) s = opt.value.toLowerCase();
        if (s === "skew") skew = true;
        else if (s !== "nonskew")
          throw new RuntimeError(
            "issymmetric: second argument must be 'skew' or 'nonskew'"
          );
      }

      // Check A(i,j) == A(j,i) or A(i,j) == -A(j,i)
      for (let i = 0; i < n; i++) {
        for (let j = i; j < n; j++) {
          const aij = A.data[colMajorIndex(i, j, m)];
          const aji = A.data[colMajorIndex(j, i, m)];
          if (skew) {
            if (aij !== -aji) return RTV.logical(false);
          } else {
            if (aij !== aji) return RTV.logical(false);
          }
          // Also check imaginary parts if complex
          if (A.imag) {
            const aijIm = A.imag[colMajorIndex(i, j, m)];
            const ajiIm = A.imag[colMajorIndex(j, i, m)];
            if (skew) {
              if (aijIm !== -ajiIm) return RTV.logical(false);
            } else {
              if (aijIm !== ajiIm) return RTV.logical(false);
            }
          }
        }
      }
      return RTV.logical(true);
    })
  );

  // ishermitian(A) — true if A == A' (conjugate transpose, within tolerance)
  // ishermitian(A, 'skew') — true if A == -A'
  register(
    "ishermitian",
    builtinSingle(args => {
      if (args.length < 1 || args.length > 2)
        throw new RuntimeError("ishermitian requires 1 or 2 arguments");

      const A = args[0];

      // Scalar is always Hermitian
      if (isRuntimeNumber(A) || isRuntimeLogical(A)) return RTV.logical(true);
      if (isRuntimeComplexNumber(A)) {
        // Hermitian scalar: im == 0; skew-Hermitian: re == 0
        let skew = false;
        if (args.length === 2) {
          const opt = args[1];
          let s: string | undefined;
          if (isRuntimeString(opt)) s = opt.toLowerCase();
          else if (isRuntimeChar(opt)) s = opt.value.toLowerCase();
          if (s === "skew") skew = true;
        }
        if (skew) return RTV.logical(A.re === 0);
        return RTV.logical(A.im === 0);
      }

      if (!isRuntimeTensor(A))
        throw new RuntimeError("ishermitian: argument must be numeric");

      const [m, n] = tensorSize2D(A);
      if (m !== n) return RTV.logical(false);

      // Parse optional 'skew' flag
      let skew = false;
      if (args.length === 2) {
        const opt = args[1];
        let s: string | undefined;
        if (isRuntimeString(opt)) s = opt.toLowerCase();
        else if (isRuntimeChar(opt)) s = opt.value.toLowerCase();
        if (s === "skew") skew = true;
        else if (s !== "nonskew")
          throw new RuntimeError(
            "ishermitian: second argument must be 'skew' or 'nonskew'"
          );
      }

      // Check A(i,j) == conj(A(j,i)) or A(i,j) == -conj(A(j,i))
      const hasImag = A.imag !== undefined;
      for (let i = 0; i < n; i++) {
        for (let j = i; j < n; j++) {
          const aijRe = A.data[colMajorIndex(i, j, m)];
          const ajiRe = A.data[colMajorIndex(j, i, m)];
          const aijIm = hasImag ? A.imag![colMajorIndex(i, j, m)] : 0;
          const ajiIm = hasImag ? A.imag![colMajorIndex(j, i, m)] : 0;
          // conj(A(j,i)) = (ajiRe, -ajiIm)
          if (skew) {
            // Check A(i,j) == -conj(A(j,i)) => aijRe == -ajiRe, aijIm == ajiIm
            if (aijRe !== -ajiRe) return RTV.logical(false);
            if (aijIm !== ajiIm) return RTV.logical(false);
          } else {
            // Check A(i,j) == conj(A(j,i)) => aijRe == ajiRe, aijIm == -ajiIm
            if (aijRe !== ajiRe) return RTV.logical(false);
            if (aijIm !== -ajiIm) return RTV.logical(false);
          }
        }
      }
      return RTV.logical(true);
    })
  );
}
