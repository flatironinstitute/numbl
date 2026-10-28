/**
 * Size, class and validator checks shared by `arguments` blocks and classdef
 * property declarations (`name (dims) class {validators}`).
 *
 * A declaration is compiled once into a `CompiledCheck`; `validateValue`
 * applies it to a value in MATLAB's order: class conversion, then size
 * (with MATLAB's scalar expansion and vector reshaping), then the
 * validator functions in order.
 */

import type { Expr, ValidationSpec } from "../parser/types.js";
import {
  isRuntimeCell,
  isRuntimeChar,
  isRuntimeClassInstance,
  isRuntimeClassInstanceArray,
  isRuntimeComplexNumber,
  isRuntimeFunction,
  isRuntimeLogical,
  isRuntimeNumber,
  isRuntimeSparseMatrix,
  isRuntimeString,
  isRuntimeStringArray,
  isRuntimeStruct,
  isRuntimeStructArray,
  isRuntimeTensor,
  type RuntimeValue,
} from "../runtime/types.js";
import { RTV } from "../runtime/constructors.js";
import { ensureRuntimeValue } from "../runtime/runtimeHelpers.js";
import { CancellationError, RuntimeError } from "../runtime/error.js";
import { invokeBuiltin } from "../runtime/runtimeDispatch.js";
import { INTEGER_CLASSES, valueShape } from "../runtime/kinds.js";

import type { Interpreter } from "./interpreter.js";
import { callFunction, callResolved } from "./calls.js";
import { resolveName } from "../resolution/index.js";
import type { FunctionIndex } from "../workspace/functionIndex.js";
import { evalExprNargout } from "./evalExpressions.js";
import { instantiateClass } from "./oop.js";

/** A declaration's constraints, ready to apply. */
export interface CompiledCheck {
  /** Required size per dimension (null for `:`), or null for any size. */
  readonly dims: readonly (number | null)[] | null;
  readonly className: string | null;
  readonly validators: readonly Expr[];
  /** Some validator is a call expression that reads the declared name. */
  readonly needsBinding: boolean;
  /** The size spec admits a 1x1 value as is. */
  readonly scalarFits: boolean;
}

/** Compile a parsed declaration; null when it constrains nothing. Size
 *  entries other than integer literals (MATLAB allows only those) are
 *  treated as `:`. */
export function compileCheck(spec: ValidationSpec): CompiledCheck | null {
  if (!spec.dimensions && !spec.className && spec.validators.length === 0)
    return null;
  let dims: (number | null)[] | null = null;
  if (spec.dimensions) {
    dims = spec.dimensions.map(e =>
      e.type === "Number" ? parseFloat(e.value) : null
    );
    // Trailing fixed 1s beyond the second dimension constrain nothing.
    while (dims.length > 2 && dims[dims.length - 1] === 1) dims.pop();
    if (dims.length === 1) dims.push(null);
  }
  return {
    dims,
    className: spec.className,
    validators: spec.validators,
    needsBinding: spec.validators.some(v => v.type !== "Ident"),
    scalarFits: dims === null || dims.every(d => d === null || d === 1),
  };
}

/** The prefix of a validation error message, e.g. "Invalid argument at
 *  position 1. ", built only when an error is raised. */
export type ErrorPrefix = () => string;

/** Apply `check` to `value` and return the (possibly converted) value.
 *  Errors are raised with `prefix()` prepended and the original identifier
 *  kept. `bind` makes the converted value visible under the declared name
 *  before call-expression validators run (they reference it, as in
 *  `mustBeInRange(x,0,10)`). */
export function validateValue(
  interp: Interpreter,
  check: CompiledCheck,
  value: RuntimeValue,
  prefix: ErrorPrefix,
  bind: ((v: RuntimeValue) => void) | null
): RuntimeValue {
  let v = value;
  if (check.className) {
    // Fast path: a real double scalar needs no conversion to double.
    if (!(typeof v === "number" && check.className === "double"))
      v = convertToClass(interp, v, check.className, prefix);
  }
  if (check.dims && !(check.scalarFits && isScalarPrimitive(v)))
    v = coerceSize(interp, v, check.dims, prefix);
  if (check.validators.length > 0) {
    if (check.needsBinding && bind) bind(v);
    for (const validator of check.validators) {
      try {
        if (validator.type === "Ident") callValidator(interp, validator, v);
        else evalExprNargout(interp, validator, 0);
      } catch (e) {
        throw prefixed(e, prefix());
      }
    }
  }
  return v;
}

/** Validators named by a bare name that resolved to a builtin, with the
 *  function index and file they were resolved against. */
const builtinValidators = new WeakMap<
  Expr,
  { index: FunctionIndex; file: string }
>();

/** Call a validator given by name on `v`. A name that resolves to a
 *  builtin (the common case: mustBePositive, ...) is resolved once per
 *  function index and calling file, so repeated calls skip resolution. */
function callValidator(
  interp: Interpreter,
  validator: Extract<Expr, { type: "Ident" }>,
  v: RuntimeValue
): void {
  // An object argument may select a class method of the same name.
  const isObject = isRuntimeClassInstance(v) || isRuntimeClassInstanceArray(v);
  const index = interp.functionIndex;
  const file = interp.frame.file;
  if (!isObject) {
    const cached = builtinValidators.get(validator);
    if (cached && cached.index === index && cached.file === file) {
      invokeBuiltin(interp.rt, validator.name, 0, [v]);
      return;
    }
  }
  const r = resolveName(validator.name, interp.frame, {
    args: [v],
    variables: false,
  });
  if (r.kind === "builtin" && !isObject)
    builtinValidators.set(validator, { index, file });
  callResolved(interp, r, [v], 0);
}

/** A JS-primitive value (number, logical or string scalar): always 1x1. */
function isScalarPrimitive(v: RuntimeValue): boolean {
  const t = typeof v;
  return t === "number" || t === "boolean" || t === "string";
}

/** Re-raise a validator's error with `prefix` prepended. */
function prefixed(e: unknown, prefix: string): unknown {
  if (!(e instanceof RuntimeError)) return e;
  return new RuntimeError(prefix + e.message, null, {
    identifier: e.identifier,
  });
}

function validationError(id: string, message: string): RuntimeError {
  return new RuntimeError(message, null, { identifier: id });
}

// ── Size ────────────────────────────────────────────────────────────────

function fitsDims(shape: number[], dims: readonly (number | null)[]): boolean {
  const n = Math.max(shape.length, dims.length);
  for (let i = 0; i < n; i++) {
    const s = i < shape.length ? shape[i] : 1;
    if (i >= dims.length) {
      if (s !== 1) return false;
      continue;
    }
    const d = dims[i];
    if (d !== null && d !== s) return false;
  }
  return true;
}

/** Check `v` against a size spec, applying MATLAB's compatible-size
 *  conversions: a scalar expands to a fully specified size (`:` taken as
 *  1), and a vector of either orientation (or an empty) is reshaped to a
 *  row/column spec such as `(1,:)` or `(3,1)`. */
function coerceSize(
  interp: Interpreter,
  v: RuntimeValue,
  dims: readonly (number | null)[],
  prefix: ErrorPrefix
): RuntimeValue {
  const shape = valueShape(v);
  if (fitsDims(shape, dims)) return v;
  const numel = shape.reduce((a, b) => a * b, 1);
  if (numel === 1) {
    const target = dims.map(d => d ?? 1);
    return ensureRuntimeValue(
      invokeBuiltin(interp.rt, "repmat", 1, [v, RTV.row(target)])
    );
  }
  if (dims.length === 2 && (dims[0] === 1 || dims[1] === 1)) {
    const nonSingleton = shape.filter(s => s !== 1).length;
    if (nonSingleton <= 1 || numel === 0) {
      const target =
        dims[0] === 1
          ? dims[1] === null || dims[1] === numel
            ? [1, numel]
            : null
          : dims[0] === null || dims[0] === numel
            ? [numel, 1]
            : null;
      if (target)
        return ensureRuntimeValue(
          invokeBuiltin(interp.rt, "reshape", 1, [v, RTV.row(target)])
        );
    }
  }
  throw validationError(
    "MATLAB:validation:IncompatibleSize",
    prefix() + sizeMessage(dims)
  );
}

/** MATLAB's description of a size spec, e.g. "Value must be a vector." */
function sizeMessage(dims: readonly (number | null)[]): string {
  const fmt = (d: number | null) => (d === null ? "*" : String(d));
  const anyColon = dims.some(d => d === null);
  if (dims.length === 2) {
    const [r, c] = dims;
    if (r === 1 && c === 1) return "Value must be a scalar.";
    if (r === 1 || c === 1) {
      const other = r === 1 ? c : r;
      return other === null
        ? "Value must be a vector."
        : `Value must be a vector with ${other} elements.`;
    }
    if (r === null && c === null) return "Value must be a matrix.";
    if (c === null) return `Value must be a matrix with ${r} rows.`;
    if (r === null) return `Value must be a matrix with ${c} columns.`;
    return `Value must be a matrix of size ${r}-by-${c}.`;
  }
  if (dims.every(d => d === 1)) return "Value must be a scalar.";
  if (dims.every(d => d === null))
    return `Value must be a ${dims.length}-dimensional array.`;
  return anyColon
    ? `Value must be an array of size ${dims.map(fmt).join("-by-")} where * indicates a dimension of any size.`
    : `Value must be an array of size ${dims.map(fmt).join("-by-")}.`;
}

// ── Class ───────────────────────────────────────────────────────────────

function isNumericOrLogical(v: RuntimeValue): boolean {
  return (
    isRuntimeNumber(v) ||
    isRuntimeLogical(v) ||
    isRuntimeTensor(v) ||
    isRuntimeComplexNumber(v) ||
    isRuntimeSparseMatrix(v)
  );
}

function isLogicalValue(v: RuntimeValue): boolean {
  return isRuntimeLogical(v) || (isRuntimeTensor(v) && !!v._isLogical);
}

function isStringValue(v: RuntimeValue): boolean {
  return isRuntimeString(v) || isRuntimeStringArray(v);
}

/** Convert `v` to class `cls` the way MATLAB converts an argument or
 *  property value, raising MATLAB's error when it cannot. Class names
 *  numbl does not know (other than workspace classes) are not checked. */
function convertToClass(
  interp: Interpreter,
  v: RuntimeValue,
  cls: string,
  prefix: ErrorPrefix
): RuntimeValue {
  const fail = (): RuntimeError =>
    validationError(
      "MATLAB:validation:UnableToConvert",
      `${prefix()}Value must be of type ${cls} or be convertible to ${cls}.`
    );
  const builtinConvert = (name: string, x: RuntimeValue): RuntimeValue => {
    try {
      return ensureRuntimeValue(invokeBuiltin(interp.rt, name, 1, [x]));
    } catch (e) {
      if (e instanceof CancellationError) throw e;
      throw fail();
    }
  };

  // A class instance converts through its own converter method, if any.
  if (isRuntimeClassInstance(v) || isRuntimeClassInstanceArray(v)) {
    if (v.className === cls) return v;
    const info = interp.code.classes.get(cls);
    if (info && toBool(interp.rt.isa(v, RTV.char(cls)))) return v;
    const own = interp.code.classes.get(v.className);
    if (own && own.methodNames.has(cls))
      return ensureRuntimeValue(callFunction(interp, cls, [v], 1));
    if (info) return constructFrom(interp, v, cls, fail);
    if (KNOWN_BUILTIN_CLASSES.has(cls)) throw fail();
    return v;
  }

  switch (cls) {
    case "double":
      if (isLogicalValue(v)) return builtinConvert("double", v);
      if (isNumericOrLogical(v)) return v;
      if (isRuntimeChar(v) || isStringValue(v))
        return builtinConvert("double", v);
      throw fail();
    case "single":
      if (isNumericOrLogical(v) || isRuntimeChar(v))
        return builtinConvert("single", v);
      if (isStringValue(v))
        return builtinConvert("single", builtinConvert("double", v));
      throw fail();
    case "logical": {
      if (isLogicalValue(v)) return v;
      if (isRuntimeChar(v)) {
        const codes = ensureRuntimeValue(
          invokeBuiltin(interp.rt, "double", 1, [v])
        );
        return builtinConvert("logical", codes);
      }
      if (
        isRuntimeComplexNumber(v) ||
        (isRuntimeTensor(v) && v.imag !== undefined)
      )
        throw fail();
      if (isRuntimeNumber(v)) {
        if (Number.isNaN(v)) throw fail();
        return v !== 0;
      }
      if (isRuntimeTensor(v) || isRuntimeSparseMatrix(v))
        return builtinConvert("logical", v);
      throw fail();
    }
    case "char":
      if (isRuntimeChar(v)) return v;
      if (isStringValue(v) || isNumericOrLogical(v))
        return builtinConvert("char", v);
      if (isRuntimeCell(v)) {
        for (const e of v.data) {
          const ev = ensureRuntimeValue(e);
          if (!isRuntimeChar(ev) && !isRuntimeString(ev)) throw fail();
        }
        return builtinConvert("char", v);
      }
      throw fail();
    case "string":
      if (isStringValue(v)) return v;
      if (isRuntimeChar(v) || isNumericOrLogical(v) || isRuntimeCell(v))
        return builtinConvert("string", v);
      throw fail();
    case "cell":
      if (isRuntimeCell(v)) return v;
      throw fail();
    case "struct":
      if (isRuntimeStruct(v) || isRuntimeStructArray(v)) return v;
      throw fail();
    case "function_handle":
      if (isRuntimeFunction(v)) return v;
      throw fail();
  }
  if (INTEGER_CLASSES.has(cls)) {
    if (isNumericOrLogical(v) || isRuntimeChar(v))
      return builtinConvert(cls, v);
    throw fail();
  }
  // A workspace class: an instance (of it or a subclass) passes, anything
  // else goes through the class constructor.
  if (interp.code.classes.get(cls)) return constructFrom(interp, v, cls, fail);
  return v;
}

/** Built-in class names whose instances are never class_instance values. */
const KNOWN_BUILTIN_CLASSES = new Set([
  "double",
  "single",
  "logical",
  "char",
  "string",
  "cell",
  "struct",
  "function_handle",
  ...INTEGER_CLASSES,
]);

function constructFrom(
  interp: Interpreter,
  v: RuntimeValue,
  cls: string,
  fail: () => RuntimeError
): RuntimeValue {
  try {
    return ensureRuntimeValue(instantiateClass(interp, cls, [v], 1));
  } catch (e) {
    if (e instanceof CancellationError) throw e;
    throw fail();
  }
}

function toBool(v: unknown): boolean {
  if (typeof v === "boolean") return v;
  if (typeof v === "number") return v !== 0;
  const rv = ensureRuntimeValue(v);
  return isRuntimeTensor(rv) ? rv.data[0] !== 0 : !!rv;
}
