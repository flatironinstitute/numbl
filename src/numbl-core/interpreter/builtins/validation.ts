/**
 * Validation builtins: mustBe* family.
 */

import type { RuntimeValue } from "../../runtime/types.js";
import {
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeTensor,
  isRuntimeCell,
  isRuntimeChar,
  isRuntimeString,
  isRuntimeComplexNumber,
  isRuntimeSparseMatrix,
} from "../../runtime/types.js";
import { toNumber, RuntimeError } from "../../runtime/index.js";
import { defineBuiltin } from "./types.js";
import type { JitType } from "../../jit/jitTypes.js";

function isNumeric(v: RuntimeValue): boolean {
  return isRuntimeNumber(v) || isRuntimeTensor(v) || isRuntimeLogical(v);
}

function numericElements(v: RuntimeValue): number[] {
  if (isRuntimeNumber(v)) return [v as number];
  if (isRuntimeLogical(v)) return [v ? 1 : 0];
  if (isRuntimeTensor(v)) return Array.from(v.data);
  return [];
}

function numelVal(v: RuntimeValue): number {
  if (isRuntimeNumber(v) || isRuntimeLogical(v)) return 1;
  if (isRuntimeTensor(v)) return v.data.length;
  if (isRuntimeChar(v)) return v.value.length;
  if (isRuntimeString(v)) return 1;
  if (isRuntimeCell(v)) return v.data.length;
  return 0;
}

function validatorMatch(argTypes: JitType[]): JitType[] | null {
  if (argTypes.length < 1) return null;
  return [{ kind: "number" }];
}

function makeElemValidator(
  name: string,
  predicate: (n: number) => boolean,
  message: string
): void {
  defineBuiltin({
    name,
    cases: [
      {
        match: validatorMatch,
        apply: args => {
          const v = args[0];
          if (!v) throw new RuntimeError(`${name}: missing argument`);
          if (!isNumeric(v)) throw new RuntimeError(message);
          for (const n of numericElements(v)) {
            if (!predicate(n)) throw new RuntimeError(message);
          }
          return 0;
        },
      },
    ],
  });
}

defineBuiltin({
  name: "mustBeNumeric",
  cases: [
    {
      match: validatorMatch,
      apply: args => {
        const v = args[0];
        if (!v) throw new RuntimeError("mustBeNumeric: missing argument");
        if (!isNumeric(v)) throw new RuntimeError("Value must be numeric.");
        return 0;
      },
    },
  ],
});

makeElemValidator(
  "mustBeInteger",
  n => Number.isFinite(n) && Math.floor(n) === n,
  "Value must be integer."
);
makeElemValidator("mustBePositive", n => n > 0, "Value must be positive.");
makeElemValidator(
  "mustBeNonnegative",
  n => n >= 0,
  "Value must be nonnegative."
);
makeElemValidator("mustBeNonzero", n => n !== 0, "Value must be nonzero.");
makeElemValidator(
  "mustBeFinite",
  n => Number.isFinite(n),
  "Value must be finite."
);

defineBuiltin({
  name: "mustBeNonempty",
  cases: [
    {
      match: validatorMatch,
      apply: args => {
        const v = args[0];
        if (!v) throw new RuntimeError("mustBeNonempty: missing argument");
        if (numelVal(v) === 0)
          throw new RuntimeError("Value must be nonempty.");
        return 0;
      },
    },
  ],
});

defineBuiltin({
  name: "mustBeScalarOrEmpty",
  cases: [
    {
      match: validatorMatch,
      apply: args => {
        const v = args[0];
        if (!v) throw new RuntimeError("mustBeScalarOrEmpty: missing argument");
        const n = numelVal(v);
        if (n !== 0 && n !== 1)
          throw new RuntimeError("Value must be scalar or empty.");
        return 0;
      },
    },
  ],
});

defineBuiltin({
  name: "mustBeVector",
  cases: [
    {
      match: validatorMatch,
      apply: args => {
        const v = args[0];
        if (!v) throw new RuntimeError("mustBeVector: missing argument");
        if (isRuntimeTensor(v)) {
          const isVec =
            v.shape.length <= 2 &&
            (v.shape[0] === 1 || v.shape[1] === 1 || v.data.length === 0);
          if (!isVec) throw new RuntimeError("Value must be a vector.");
        } else if (!isRuntimeNumber(v) && !isRuntimeLogical(v)) {
          throw new RuntimeError("Value must be a vector.");
        }
        return 0;
      },
    },
  ],
});

defineBuiltin({
  name: "mustBeMember",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length < 2) return null;
        return [{ kind: "number" }];
      },
      apply: args => {
        const v = args[0];
        const set = args[1];
        if (!v || !set)
          throw new RuntimeError("mustBeMember: requires 2 arguments");
        const vElems = isRuntimeString(v) ? [v] : numericElements(v);
        const setElems = isRuntimeCell(set)
          ? set.data
          : isRuntimeString(set)
            ? [set]
            : numericElements(set);
        for (const elem of vElems) {
          const found = (setElems as unknown[]).some(s =>
            typeof s === "object" &&
            s !== null &&
            "kind" in s &&
            isRuntimeString(s as RuntimeValue)
              ? (s as { kind: string; value: string }).value === String(elem)
              : s === elem
          );
          if (!found)
            throw new RuntimeError(
              "Value must be a member of the specified set."
            );
        }
        return 0;
      },
    },
  ],
});

defineBuiltin({
  name: "mustBeInRange",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length < 1) return null;
        return [{ kind: "number" }];
      },
      apply: args => {
        const v = args[0];
        const lower = args[1] ? toNumber(args[1]) : -Infinity;
        const upper = args[2] ? toNumber(args[2]) : Infinity;
        if (!v) throw new RuntimeError("mustBeInRange: missing argument");
        if (!isNumeric(v)) throw new RuntimeError("Value must be in range.");
        for (const n of numericElements(v)) {
          if (n < lower || n > upper) {
            throw new RuntimeError(
              `Value must be in range [${lower}, ${upper}].`
            );
          }
        }
        return 0;
      },
    },
  ],
});

// ── validateattributes ─────────────────────────────────────────────────

function getValClassName(v: RuntimeValue): string {
  if (typeof v === "number") return "double";
  if (typeof v === "boolean") return "logical";
  if (typeof v === "string") return "string";
  const obj = v as { kind: string; _isLogical?: boolean; className?: string };
  switch (obj.kind) {
    case "tensor":
      return obj._isLogical ? "logical" : "double";
    case "complex_number":
      return "double";
    case "char":
      return "char";
    case "cell":
      return "cell";
    case "struct":
    case "struct_array":
      return "struct";
    case "function":
      return "function_handle";
    case "class_instance":
    case "class_instance_array":
      return obj.className ?? "unknown";
    case "sparse_matrix":
      return "double";
    case "dictionary":
      return "dictionary";
    default:
      return "unknown";
  }
}

function matchesOneClass(v: RuntimeValue, cls: string): boolean {
  const vc = getValClassName(v);
  if (vc === cls) return true;
  if (cls === "numeric" && vc === "double") return true;
  return false;
}

function getShapeOf(v: RuntimeValue): number[] {
  if (typeof v === "number" || typeof v === "boolean" || typeof v === "string")
    return [1, 1];
  const obj = v as Record<string, unknown>;
  switch (obj.kind) {
    case "tensor":
      return ((obj.shape as number[]) ?? []).length === 0
        ? [1, 1]
        : (obj.shape as number[]);
    case "complex_number":
      return [1, 1];
    case "char":
      return (obj.shape as number[]) ?? [1, (obj.value as string).length || 0];
    case "cell":
      return obj.shape as number[];
    case "sparse_matrix":
      return [obj.m as number, obj.n as number];
    case "struct":
      return [1, 1];
    case "struct_array":
      return [1, (obj.elements as unknown[]).length];
    default:
      return [1, 1];
  }
}

function extractStr(v: RuntimeValue): string {
  if (typeof v === "string") return v;
  if (isRuntimeChar(v)) return v.value;
  throw new RuntimeError("Expected character vector or string");
}

function checkColumnMonotonicity(
  A: RuntimeValue,
  shape: number[],
  cmp: (prev: number, curr: number) => boolean,
  desc: string,
  errFn: (msg: string) => never
): void {
  if (isRuntimeComplexNumber(A)) errFn(`be ${desc}.`);
  const elems = numericElements(A);
  if (elems.length <= 1) return;
  for (const e of elems) {
    if (Number.isNaN(e)) errFn(`be ${desc}.`);
  }
  const nrows = shape[0] ?? 1;
  const ncols = Math.floor(elems.length / nrows);
  for (let c = 0; c < ncols; c++) {
    for (let r = 1; r < nrows; r++) {
      if (!cmp(elems[c * nrows + r - 1], elems[c * nrows + r])) {
        errFn(`be ${desc}.`);
      }
    }
  }
}

defineBuiltin({
  name: "validateattributes",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length < 3) return null;
        return [{ kind: "number" }];
      },
      apply: args => {
        const A = args[0];
        const classesArg = args[1];
        const attrsArg = args[2];

        // Parse optional arguments
        let funcName: string | undefined;
        let varName: string | undefined;
        let argIndex: number | undefined;

        if (args.length === 4) {
          if (isRuntimeNumber(args[3])) {
            argIndex = args[3] as number;
          } else {
            funcName = extractStr(args[3]);
          }
        } else if (args.length === 5) {
          funcName = extractStr(args[3]);
          varName = extractStr(args[4]);
        } else if (args.length >= 6) {
          funcName = extractStr(args[3]);
          varName = extractStr(args[4]);
          argIndex = toNumber(args[5]);
        }

        if (funcName === "") funcName = undefined;
        if (varName === "") varName = undefined;

        // Build error subject
        let subject = "input";
        if (argIndex !== undefined && varName) {
          subject = `input number ${argIndex}, ${varName},`;
        } else if (argIndex !== undefined) {
          subject = `input number ${argIndex}`;
        } else if (varName) {
          subject = `input '${varName}'`;
        }

        const errFn = (msg: string): never => {
          throw new RuntimeError(`Expected ${subject} to ${msg}`);
        };

        // --- Class check ---
        let classNames: string[] = [];
        if (isRuntimeCell(classesArg)) {
          classNames = classesArg.data.map(extractStr);
        } else if (isRuntimeChar(classesArg)) {
          classNames = [classesArg.value];
        } else if (isRuntimeString(classesArg)) {
          classNames = [classesArg as string];
        }

        if (classNames.length > 0) {
          if (!classNames.some(cls => matchesOneClass(A, cls))) {
            const vc = getValClassName(A);
            errFn(
              `be one of these types:\n\n${classNames.join(", ")}\n\nInstead its type was ${vc}.`
            );
          }
        }

        // --- Attribute check ---
        if (!isRuntimeCell(attrsArg)) {
          throw new RuntimeError(
            "validateattributes: attributes must be a cell array"
          );
        }

        const shape = getShapeOf(A);
        // Effective ndims: remove trailing singletons, min 2
        let effNdims = shape.length;
        while (effNdims > 2 && shape[effNdims - 1] === 1) effNdims--;
        effNdims = Math.max(effNdims, 2);

        const nrows = shape[0] ?? 1;
        const ncols = shape.length >= 2 ? shape[1] : 1;
        const numel = shape.reduce((a, b) => a * b, 1);
        const attrData = attrsArg.data;

        let ai = 0;
        while (ai < attrData.length) {
          const attrName = extractStr(attrData[ai]);
          ai++;

          switch (attrName) {
            // ── Shape / size ──
            case "2d":
              if (effNdims > 2) errFn("be two-dimensional.");
              break;
            case "3d":
              if (effNdims > 3) errFn("have 3 or fewer dimensions.");
              break;
            case "column":
              if (effNdims > 2 || ncols !== 1) errFn("be a column vector.");
              break;
            case "row":
              if (effNdims > 2 || nrows !== 1) errFn("be a row vector.");
              break;
            case "scalar":
              if (numel !== 1) errFn("be scalar.");
              break;
            case "scalartext": {
              let ok = false;
              if (isRuntimeString(A)) ok = true;
              else if (isRuntimeChar(A)) {
                const cs = A.shape ?? [1, A.value.length];
                if (cs.length <= 2 && cs[0] === 1) ok = true;
              }
              if (!ok) errFn("be a character vector or string scalar.");
              break;
            }
            case "vector": {
              const isVec =
                numel <= 1 || (effNdims <= 2 && (nrows === 1 || ncols === 1));
              if (!isVec) errFn("be a vector.");
              break;
            }
            case "size": {
              if (ai >= attrData.length)
                throw new RuntimeError(
                  "validateattributes: 'size' requires a value"
                );
              const sizeVal = attrData[ai];
              ai++;
              let expectedSize: number[];
              if (isRuntimeTensor(sizeVal)) {
                expectedSize = Array.from(sizeVal.data);
              } else if (isRuntimeNumber(sizeVal)) {
                expectedSize = [sizeVal as number];
              } else {
                throw new RuntimeError(
                  "validateattributes: 'size' value must be numeric"
                );
              }
              for (let j = 0; j < expectedSize.length; j++) {
                if (Number.isNaN(expectedSize[j])) continue;
                const actual = j < shape.length ? shape[j] : 1;
                if (actual !== expectedSize[j]) {
                  const expStr = expectedSize
                    .map(x => (Number.isNaN(x) ? ":" : String(x)))
                    .join("x");
                  errFn(
                    `be of size ${expStr} when it is actually size ${shape.join("x")}.`
                  );
                }
              }
              break;
            }
            case "numel": {
              if (ai >= attrData.length)
                throw new RuntimeError(
                  "validateattributes: 'numel' requires a value"
                );
              const n = toNumber(attrData[ai]);
              ai++;
              if (numel !== n) errFn(`have ${n} elements.`);
              break;
            }
            case "ncols": {
              if (ai >= attrData.length)
                throw new RuntimeError(
                  "validateattributes: 'ncols' requires a value"
                );
              const n = toNumber(attrData[ai]);
              ai++;
              if (ncols !== n) errFn(`have ${n} columns.`);
              break;
            }
            case "nrows": {
              if (ai >= attrData.length)
                throw new RuntimeError(
                  "validateattributes: 'nrows' requires a value"
                );
              const n = toNumber(attrData[ai]);
              ai++;
              if (nrows !== n) errFn(`have ${n} rows.`);
              break;
            }
            case "ndims": {
              if (ai >= attrData.length)
                throw new RuntimeError(
                  "validateattributes: 'ndims' requires a value"
                );
              const n = toNumber(attrData[ai]);
              ai++;
              if (effNdims !== n) errFn(`be ${n}-dimensional.`);
              break;
            }
            case "square":
              if (effNdims !== 2 || nrows !== ncols) errFn("be square.");
              break;
            case "diag": {
              if (effNdims > 2 || nrows !== ncols) errFn("be diagonal.");
              const diagElems = numericElements(A);
              for (let r = 0; r < nrows; r++) {
                for (let c = 0; c < ncols; c++) {
                  if (r !== c && diagElems[c * nrows + r] !== 0)
                    errFn("be diagonal.");
                }
              }
              break;
            }
            case "nonempty":
              if (numel === 0) errFn("be nonempty.");
              break;
            case "nonsparse":
              if (isRuntimeSparseMatrix(A)) errFn("be nonsparse.");
              break;

            // ── Range ──
            case ">": {
              if (ai >= attrData.length)
                throw new RuntimeError(
                  "validateattributes: '>' requires a value"
                );
              const n = toNumber(attrData[ai]);
              ai++;
              if (isRuntimeComplexNumber(A))
                errFn(`be an array with all of the values > ${n}.`);
              for (const e of numericElements(A)) {
                if (!(e > n))
                  errFn(`be an array with all of the values > ${n}.`);
              }
              break;
            }
            case ">=": {
              if (ai >= attrData.length)
                throw new RuntimeError(
                  "validateattributes: '>=' requires a value"
                );
              const n = toNumber(attrData[ai]);
              ai++;
              if (isRuntimeComplexNumber(A))
                errFn(`be an array with all of the values >= ${n}.`);
              for (const e of numericElements(A)) {
                if (!(e >= n))
                  errFn(`be an array with all of the values >= ${n}.`);
              }
              break;
            }
            case "<": {
              if (ai >= attrData.length)
                throw new RuntimeError(
                  "validateattributes: '<' requires a value"
                );
              const n = toNumber(attrData[ai]);
              ai++;
              if (isRuntimeComplexNumber(A))
                errFn(`be an array with all of the values < ${n}.`);
              for (const e of numericElements(A)) {
                if (!(e < n))
                  errFn(`be an array with all of the values < ${n}.`);
              }
              break;
            }
            case "<=": {
              if (ai >= attrData.length)
                throw new RuntimeError(
                  "validateattributes: '<=' requires a value"
                );
              const n = toNumber(attrData[ai]);
              ai++;
              if (isRuntimeComplexNumber(A))
                errFn(`be an array with all of the values <= ${n}.`);
              for (const e of numericElements(A)) {
                if (!(e <= n))
                  errFn(`be an array with all of the values <= ${n}.`);
              }
              break;
            }
            case "finite":
              for (const e of numericElements(A)) {
                if (!Number.isFinite(e)) errFn("be finite.");
              }
              break;
            case "nonnan":
              for (const e of numericElements(A)) {
                if (Number.isNaN(e)) errFn("have no NaN values.");
              }
              break;

            // ── Type / value ──
            case "binary":
              for (const e of numericElements(A)) {
                if (e !== 0 && e !== 1) errFn("be binary.");
              }
              break;
            case "even":
              for (const e of numericElements(A)) {
                if (!Number.isFinite(e) || Math.floor(e) !== e || e % 2 !== 0)
                  errFn("be even.");
              }
              break;
            case "odd":
              for (const e of numericElements(A)) {
                if (
                  !Number.isFinite(e) ||
                  Math.floor(e) !== e ||
                  Math.abs(e) % 2 !== 1
                )
                  errFn("be odd.");
              }
              break;
            case "integer":
              if (isRuntimeComplexNumber(A)) errFn("be integer.");
              for (const e of numericElements(A)) {
                if (!Number.isFinite(e) || Math.floor(e) !== e)
                  errFn("be integer.");
              }
              break;
            case "real":
              if (isRuntimeComplexNumber(A) && (A as { im: number }).im !== 0)
                errFn("be real.");
              if (isRuntimeTensor(A) && A.imag) {
                for (const v of A.imag) {
                  if (v !== 0) errFn("be real.");
                }
              }
              break;
            case "nonnegative":
              if (isRuntimeComplexNumber(A)) errFn("be nonnegative.");
              for (const e of numericElements(A)) {
                if (e < 0) errFn("be nonnegative.");
              }
              break;
            case "nonzero":
              for (const e of numericElements(A)) {
                if (e === 0) errFn("be nonzero.");
              }
              break;
            case "positive":
              if (isRuntimeComplexNumber(A)) errFn("be positive.");
              for (const e of numericElements(A)) {
                if (e <= 0) errFn("be positive.");
              }
              break;

            // ── Monotonicity (column-wise) ──
            case "increasing":
              checkColumnMonotonicity(
                A,
                shape,
                (a, b) => b > a,
                "strictly increasing",
                errFn
              );
              break;
            case "decreasing":
              checkColumnMonotonicity(
                A,
                shape,
                (a, b) => b < a,
                "strictly decreasing",
                errFn
              );
              break;
            case "nondecreasing":
              checkColumnMonotonicity(
                A,
                shape,
                (a, b) => b >= a,
                "nondecreasing",
                errFn
              );
              break;
            case "nonincreasing":
              checkColumnMonotonicity(
                A,
                shape,
                (a, b) => b <= a,
                "nonincreasing",
                errFn
              );
              break;

            default:
              throw new RuntimeError(
                `validateattributes: unknown attribute '${attrName}'`
              );
          }
        }

        return 0;
      },
    },
  ],
});
