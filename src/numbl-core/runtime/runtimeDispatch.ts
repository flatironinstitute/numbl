/**
 * Runtime dispatch, registration, and higher-order functions.
 *
 * Standalone functions that take the Runtime instance as the first parameter.
 * Handles function/class dispatch, arrayfun/cellfun/bsxfun, and subsref/subsasgn.
 */

import {
  type RuntimeValue,
  type RuntimeLogical,
  RuntimeFunction,
  RTV,
  toNumber,
  toString,
  mGetField,
  RuntimeError,
} from "../runtime/index.js";
import {
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeTensor,
  isRuntimeCell,
  isRuntimeChar,
  isRuntimeString,
  isRuntimeStruct,
  isRuntimeFunction,
  isRuntimeStructArray,
  isRuntimeComplexNumber,
  isRuntimeClassInstance,
  isRuntimeClassInstanceArray,
  isRuntimeDictionary,
  isRuntimeStringArray,
} from "../runtime/types.js";
import { isaValue, kstr, valueClass, valueShape } from "./kinds.js";
import { incref, decref } from "./refcount.js";
import {
  getClassMethod,
  getIBuiltin,
  getIBuiltinNargin,
  isBuiltin,
} from "./builtinRegistry.js";
import { COLON_SENTINEL } from "./sentinels.js";
import {
  getBroadcastShape,
  mAdd,
  mSub,
  mElemMul,
  mElemDiv,
} from "../../numbl-core/helpers/arithmetic.js";
import { ensureRuntimeValue } from "./runtimeHelpers.js";
import type { Runtime } from "./runtime.js";
import { allocFloat64Array } from "./alloc.js";
import { readableProperty } from "./runtimeMemberAccess.js";
import { invalidHandleError } from "./classAccess.js";
import { coerceToTensor } from "../helpers/shape-utils.js";

// ── Builtin invocation ──────────────────────────────────────────────────

/** Returned by `invokeBuiltin` when `name` is not a builtin. */
export const NOT_A_BUILTIN: unique symbol = Symbol("NOT_A_BUILTIN");

/** Call the builtin `name`: an execution-specific custom builtin if one is
 *  set, else the registered IBuiltin. Returns NOT_A_BUILTIN if neither
 *  exists. A builtin with no outputs errors when outputs are requested. */
export function invokeBuiltin(
  rt: Runtime,
  name: string,
  nargout: number,
  args: unknown[]
): unknown {
  const custom = rt.customBuiltins[name];
  if (custom) return custom(nargout, args);
  const ib = getIBuiltin(name);
  if (!ib) return NOT_A_BUILTIN;
  const margs = args.map(a => ensureRuntimeValue(a));
  if (!rt.profilingEnabled) return ib.call(margs, nargout);
  rt.profileEnter("builtin:interp:" + name);
  try {
    return ib.call(margs, nargout);
  } finally {
    rt.profileLeave();
  }
}

/** Call the method `methodName` a TypeScript-implemented class registered
 *  (`registerClassMethod`). */
export function invokeClassMethod(
  rt: Runtime,
  className: string,
  methodName: string,
  nargout: number,
  args: unknown[]
): unknown {
  const ib = getClassMethod(className, methodName);
  if (!ib)
    throw new RuntimeError(
      `No method '${methodName}' for class '${className}'`
    );
  const margs = args.map(a => ensureRuntimeValue(a));
  if (!rt.profilingEnabled) return ib.call(margs, nargout);
  rt.profileEnter(`builtin:interp:${className}.${methodName}`);
  try {
    return ib.call(margs, nargout);
  } finally {
    rt.profileLeave();
  }
}

// ── arrayfun/cellfun result packing ──────────────────────────────────────

/** Pack uniform-output results into a tensor (if all scalar) or a cell. */
function packResults(results: RuntimeValue[], shape: number[]): RuntimeValue {
  const allScalar = results.every(
    r =>
      isRuntimeNumber(r) ||
      isRuntimeLogical(r) ||
      (isRuntimeTensor(r) && r.data.length === 1)
  );
  if (allScalar) {
    const data = allocFloat64Array(results.length);
    const allLogical = results.every(r => isRuntimeLogical(r));
    for (let i = 0; i < results.length; i++) {
      data[i] = toNumber(results[i]);
    }
    const result = RTV.tensor(data, [...shape]);
    if (allLogical) result._isLogical = true;
    return result;
  }
  return RTV.cell(results, [...shape]);
}

// ── Scalar/complex → tensor coercion ─────────────────────────────────────

// ── Function handles ────────────────────────────────────────────────────

export function getFuncHandle(name: string): RuntimeFunction {
  const handle = RTV.func(name, isBuiltin(name) ? "builtin" : "user");
  const nargin = getIBuiltinNargin(name);
  if (nargin !== undefined) {
    handle.nargin = nargin;
  }
  return handle;
}

export function makeUserFuncHandle(
  jsFn: (...args: unknown[]) => unknown,
  nargin?: number
): RuntimeFunction {
  return new RuntimeFunction("", "user", [], jsFn, true, nargin);
}

// ── isa ─────────────────────────────────────────────────────────────────

export function isa(
  rt: Runtime,
  value: unknown,
  classNameArg: unknown
): RuntimeLogical {
  const mv = ensureRuntimeValue(value);
  const clsName =
    typeof classNameArg === "string"
      ? classNameArg
      : toString(ensureRuntimeValue(classNameArg));
  return isaValue(mv, clsName, cls => rt.classAncestors(cls));
}

// ── callSuperConstructor ─────────────────────────────────────────────────

export function callSuperConstructor(
  target: unknown,
  superInstance: unknown,
  rt?: Runtime
): RuntimeValue {
  const targetObj = ensureRuntimeValue(target);
  const superObj = ensureRuntimeValue(superInstance);
  if (isRuntimeClassInstance(targetObj) && isRuntimeClassInstance(superObj)) {
    // Class-to-class inheritance: merge fields from super into target.
    // bindField does decref-old / incref-new so refcounts stay accurate
    // even when the inherited field's value was already held by `superObj`.
    if (rt) {
      for (const [key, val] of superObj.fields) {
        targetObj.bindField(rt, key, val);
      }
    } else {
      for (const [key, val] of superObj.fields) {
        targetObj.fields.set(key, val);
      }
    }
  } else if (isRuntimeClassInstance(targetObj)) {
    // Built-in type superclass (e.g. classdef Foo < double):
    // store the result as _builtinData. Incref-new / decref-old keeps
    // the ownership transfer accounted for.
    if (rt) {
      const old = targetObj._builtinData;
      incref(superObj);
      targetObj._builtinData = superObj;
      if (old !== undefined) decref(rt, old);
    } else {
      targetObj._builtinData = superObj;
    }
  }
  return targetObj;
}

// ── createClassInstance ─────────────────────────────────────────────────

export function createClassInstance(
  _rt: Runtime,
  className: string,
  propertyNames: string[],
  defaults?: Record<string, unknown>,
  isHandleClass = false
): RuntimeValue {
  let defaultsMap: Map<string, RuntimeValue> | undefined;
  if (defaults) {
    defaultsMap = new Map();
    for (const [key, val] of Object.entries(defaults)) {
      defaultsMap.set(key, ensureRuntimeValue(val));
    }
  }
  return RTV.classInstance(
    className,
    propertyNames,
    isHandleClass,
    defaultsMap
  );
}

// ── Dispatch ────────────────────────────────────────────────────────────

export function dispatch(
  rt: Runtime,
  name: string,
  nargout: number,
  args: unknown[],
  targetClassName?: string
): unknown {
  const callerFile = rt.$file;
  const callerLine = rt.$line;
  try {
    // 1. User functions and class methods, resolved from the running
    //    code's scope. This runs BEFORE builtins so that class method
    //    overloads (e.g., horzcat, vertcat, plus) take precedence.
    if (rt.resolveNamedCall) {
      if (rt.profilingEnabled) rt.profileEnter("jit");
      const fn = rt.resolveNamedCall(name, args, targetClassName);
      if (fn) {
        const dispName = targetClassName ? `${targetClassName}.${name}` : name;
        if (rt.profilingEnabled) {
          rt.profileLeave(); // leave jit
          rt.profileEnter("dispatch:" + dispName);
          const result = fn(nargout, ...args);
          rt.profileLeave(); // leave dispatch
          return result;
        }
        return fn(nargout, ...args);
      }
      if (rt.profilingEnabled) rt.profileLeave(); // leave jit
    }

    // 2. Builtins
    const result = invokeBuiltin(rt, name, nargout, args);
    if (result !== NOT_A_BUILTIN) return result;

    if (targetClassName) {
      throw new RuntimeError(
        `No method '${name}' for class '${targetClassName}'`
      );
    }
    throw new RuntimeError(`Undefined function or variable: '${name}'`, null, {
      identifier: "MATLAB:UndefinedFunction",
    });
  } finally {
    rt.$file = callerFile;
    rt.$line = callerLine;
  }
}

// ── callBuiltin ──────────────────────────────────────────────────────────

/** Direct builtin call — skips local/workspace/registry lookup. */
export function callBuiltin(
  rt: Runtime,
  name: string,
  nargout: number,
  args: unknown[]
): unknown {
  const result = invokeBuiltin(rt, name, nargout, args);
  if (result !== NOT_A_BUILTIN) return result;
  throw new RuntimeError(`'${name}' is not a builtin function`);
}

// ── Class method hierarchy lookup ────────────────────────────────────────

// ── callClassMethod ──────────────────────────────────────────────────────

/** Direct class method call — delegates to unified dispatch with targetClassName. */
export function callClassMethod(
  rt: Runtime,
  className: string,
  methodName: string,
  nargout: number,
  args: unknown[]
): unknown {
  return dispatch(rt, methodName, nargout, args, className);
}

// ── methodDispatch ──────────────────────────────────────────────────────

/** Flatten comma-separated lists: a JS array in the argument list represents
 *  a cs-list (e.g. from `c{:}` or `varargin{:}`) and expands into individual
 *  arguments. Single runtime values pass through unchanged. */
function flattenCsLists(args: unknown[]): unknown[] {
  const out: unknown[] = [];
  for (const a of args) {
    if (Array.isArray(a)) out.push(...a);
    else out.push(a);
  }
  return out;
}

export function methodDispatch(
  rt: Runtime,
  name: string,
  nargout: number,
  args: unknown[]
): unknown {
  const callerFile = rt.$file;
  const callerLine = rt.$line;
  try {
    // Struct field: function handle or indexed field access
    if (args.length > 0) {
      const first = args[0];
      let firstRV: RuntimeValue | null = null;
      try {
        firstRV = ensureRuntimeValue(first);
      } catch {
        // fall through
      }
      // Dictionary method dispatch: d.keys(), d.values(), d.numEntries(), etc.
      if (firstRV && isRuntimeDictionary(firstRV)) {
        return dispatch(rt, name, nargout, args);
      }
      // A 1x1 struct array behaves like a scalar struct: s.field(args)
      if (
        firstRV &&
        isRuntimeStructArray(firstRV) &&
        firstRV.elements.length === 1 &&
        firstRV.fieldNames.includes(name)
      ) {
        firstRV = firstRV.elements[0];
      }
      if (firstRV && isRuntimeStruct(firstRV)) {
        const fieldVal = mGetField(firstRV, name);
        const remaining = args.slice(1);
        const fieldRV = ensureRuntimeValue(fieldVal);
        if (isRuntimeFunction(fieldRV)) {
          // Flatten comma-separated lists (e.g. `s.fn(a, c{:})`) into
          // individual arguments. The indexing branch below routes through
          // rt.index, which already expands cs-lists; the function-handle
          // call here must do the same before invoking the handle.
          const callArgs = flattenCsLists(remaining);
          if (fieldRV.jsFn) {
            return fieldRV.jsFnExpectsNargout
              ? fieldRV.jsFn(nargout, ...callArgs)
              : fieldRV.jsFn(...callArgs);
          }
          return dispatch(rt, fieldRV.name, nargout, callArgs);
        }
        if (remaining.length > 0) {
          return rt.index(fieldVal, remaining, nargout);
        }
        return fieldVal;
      }
      // Class instance: property access or method call
      if (firstRV && isRuntimeClassInstance(firstRV)) {
        // A property the running code may not read goes to an overloaded
        // subsref (below), or raises MATLAB's access error.
        const prop = readableProperty(rt, firstRV.className, name);
        if (firstRV._deleted && prop) throw invalidHandleError();
        if (prop !== "subsref" && prop?.constant) {
          const value = rt.constantPropertyValue!(prop);
          const remaining = args.slice(1);
          return remaining.length > 0
            ? rt.index(value, remaining, nargout)
            : value;
        }
        // Try field access first
        if (prop !== "subsref" && firstRV.fields.has(name)) {
          const fieldVal = mGetField(firstRV, name);
          const remaining = args.slice(1);
          if (remaining.length > 0) {
            return rt.index(fieldVal, remaining, nargout);
          }
          return fieldVal;
        }
        // Dependent property with a get.<name> accessor: call the getter
        // to obtain the value, then index the result with any remaining
        // args.  This makes `obj.depProp(i,j)` work, distinguishing it
        // from a same-named method (which would take the extra args).
        {
          const accessorKey = `${firstRV.className}.get.${name}`;
          if (!rt.activeAccessors.has(accessorKey)) {
            const getter = rt.cachedResolveClassMethod(
              firstRV.className,
              `get.${name}`
            );
            if (getter) {
              rt.activeAccessors.add(accessorKey);
              let gotVal: unknown;
              try {
                gotVal = getter(1, first);
              } finally {
                rt.activeAccessors.delete(accessorKey);
              }
              const remaining = args.slice(1);
              if (remaining.length > 0) {
                return rt.index(gotVal, remaining, nargout);
              }
              return gotVal;
            }
          }
        }
        // If the name is NOT a real method of the class but the class
        // overloads subsref, route the `obj.name(...)` through subsref.
        // This handles cases where `name` is resolved via custom subsref
        // (e.g., chebfunpref.tech() where tech is a field of an internal
        // struct). We must check method existence *first* — otherwise an
        // error thrown from inside a real method would be masked by a
        // spurious subsref reroute (e.g. ultraSEM's obj.initialize(...)).
        const methodExists =
          rt.cachedResolveClassMethod(firstRV.className, name) !== null;
        if (!methodExists) {
          const guardKey = `${firstRV.className}.subsref`;
          if (!rt.activeAccessors.has(guardKey)) {
            const subsrefFn = rt.cachedResolveClassMethod(
              firstRV.className,
              "subsref"
            );
            if (subsrefFn) {
              const remaining = args.slice(1);
              // Build compound subsref indices: [{'.', name}, {'()', {args...}}]
              const sEntries = [
                RTV.struct({
                  type: RTV.char("."),
                  subs: RTV.char(name),
                }),
                RTV.struct({
                  type: RTV.char("()"),
                  subs: RTV.cell(
                    remaining.map(a => ensureRuntimeValue(a)),
                    [1, remaining.length]
                  ),
                }),
              ];
              const S = RTV.structArray(["type", "subs"], sEntries);
              rt.activeAccessors.add(guardKey);
              try {
                return subsrefFn(nargout, first, S);
              } finally {
                rt.activeAccessors.delete(guardKey);
              }
            }
          }
        }
        // The method exists (call it; errors propagate untouched) or there's
        // no subsref to handle a missing name (callClassMethod throws the
        // proper "No method 'name' for class" error).
        return callClassMethod(rt, firstRV.className, name, nargout, args);
      }
    }

    // Builtins
    const result = invokeBuiltin(rt, name, nargout, args);
    if (result !== NOT_A_BUILTIN) return result;

    throw new RuntimeError(`No method '${name}' found`);
  } finally {
    rt.$file = callerFile;
    rt.$line = callerLine;
  }
}

// ── arrayfun / cellfun ──────────────────────────────────────────────────

/** Shared implementation for arrayfun/cellfun — handles UniformOutput name-value pair */
function arrayfunCellfunImpl(
  rt: Runtime,
  name: string,
  nargout: number,
  args: unknown[]
): unknown {
  if (args.length < 2)
    throw new RuntimeError(`${name} requires at least 2 arguments`);
  const fnArg = args[0];

  // Separate input arrays from name-value pairs.
  // Input arrays come first (args[1], args[2], ...), then name-value pairs
  // start when we encounter a string argument.
  const inputArgs: RuntimeValue[] = [];
  let nvStart = args.length; // index where name-value pairs begin
  for (let i = 1; i < args.length; i++) {
    const rv = ensureRuntimeValue(args[i]);
    if (
      (isRuntimeChar(rv) || isRuntimeString(rv)) &&
      !isRuntimeCell(rv) &&
      !isRuntimeTensor(rv)
    ) {
      nvStart = i;
      break;
    }
    inputArgs.push(rv);
  }

  // Parse name-value pairs (e.g., 'UniformOutput', false)
  let uniformOutput = true;
  for (let i = nvStart; i + 1 < args.length; i += 2) {
    const key = ensureRuntimeValue(args[i]);
    if (
      (isRuntimeChar(key) || isRuntimeString(key)) &&
      (typeof key === "string" ? key : key.value).toLowerCase() ===
        "uniformoutput"
    ) {
      const val = ensureRuntimeValue(args[i + 1]);
      uniformOutput = isRuntimeLogical(val)
        ? val
        : isRuntimeNumber(val)
          ? val !== 0
          : isRuntimeTensor(val)
            ? val.data[0] !== 0
            : true;
    }
  }

  if (inputArgs.length === 0)
    throw new RuntimeError(`${name} requires at least one input array`);

  const arrArg = inputArgs[0];
  const extraInputs = inputArgs.slice(1);

  const callFn = (elems: unknown[]): unknown => {
    if (typeof fnArg === "function") {
      return (fnArg as (...a: unknown[]) => unknown)(...elems);
    }
    const mv = ensureRuntimeValue(fnArg);
    if (isRuntimeFunction(mv)) {
      return rt.index(mv, elems, nargout);
    }
    throw new RuntimeError(`${name}: first argument must be a function`);
  };

  // Helper to get the i-th element from an input array (cell, tensor,
  // struct array, object array or string array)
  const getElem = (arr: RuntimeValue, i: number): unknown => {
    if (isRuntimeCell(arr)) return arr.data[i];
    if (isRuntimeTensor(arr)) return arr.data[i];
    if (isRuntimeStructArray(arr) || isRuntimeClassInstanceArray(arr))
      return arr.elements[i];
    if (isRuntimeStringArray(arr)) return arr.data[i];
    return arr; // scalar - broadcast
  };

  // Helper to get length of an input array
  const getLen = (arr: RuntimeValue): number => {
    if (isRuntimeCell(arr)) return arr.data.length;
    if (isRuntimeTensor(arr)) return arr.data.length;
    if (isRuntimeStructArray(arr) || isRuntimeClassInstanceArray(arr))
      return arr.elements.length;
    if (isRuntimeStringArray(arr)) return arr.data.length;
    return 1;
  };

  // Collect element arguments for index i across all input arrays
  const collectArgs = (i: number): unknown[] => {
    const elems: unknown[] = [getElem(arrArg, i)];
    for (const extra of extraInputs) {
      elems.push(getElem(extra, i));
    }
    return elems;
  };

  if (isRuntimeTensor(arrArg) && extraInputs.length === 0 && nargout <= 1) {
    if (!uniformOutput) {
      const results: RuntimeValue[] = [];
      for (let i = 0; i < arrArg.data.length; i++) {
        const r = callFn(collectArgs(i));
        results.push(ensureRuntimeValue(r));
      }
      return RTV.cell(results, [...arrArg.shape]);
    }
    const resultData = allocFloat64Array(arrArg.data.length);
    let allLogical = true;
    for (let i = 0; i < arrArg.data.length; i++) {
      const r = callFn(collectArgs(i));
      if (typeof r === "number") {
        resultData[i] = r;
        allLogical = false;
      } else {
        const rv = ensureRuntimeValue(r);
        resultData[i] = toNumber(rv);
        if (!isRuntimeLogical(rv)) allLogical = false;
      }
    }
    const result = RTV.tensor(resultData, [...arrArg.shape]);
    if (allLogical && arrArg.data.length > 0) result._isLogical = true;
    return result;
  }
  if (
    isRuntimeCell(arrArg) ||
    isRuntimeStructArray(arrArg) ||
    isRuntimeClassInstanceArray(arrArg) ||
    isRuntimeStringArray(arrArg) ||
    extraInputs.length > 0 ||
    nargout > 1
  ) {
    const len = getLen(arrArg);
    const shape = [...valueShape(arrArg)];

    // Handle multiple outputs
    if (nargout > 1) {
      const allResults: RuntimeValue[][] = Array.from(
        { length: nargout },
        () => []
      );
      for (let i = 0; i < len; i++) {
        const r = callFn(collectArgs(i));
        // r should be an array (CSL) for multiple outputs
        const rArr = Array.isArray(r) ? r : [r];
        for (let o = 0; o < nargout; o++) {
          allResults[o].push(ensureRuntimeValue(o < rArr.length ? rArr[o] : 0));
        }
      }
      if (!uniformOutput) {
        const outputs = allResults.map(results =>
          RTV.cell(results, [...shape])
        );
        return outputs;
      }
      return allResults.map(results => packResults(results, shape));
    }

    const results: RuntimeValue[] = [];
    for (let i = 0; i < len; i++) {
      const r = callFn(collectArgs(i));
      results.push(ensureRuntimeValue(r));
    }
    if (!uniformOutput) {
      return RTV.cell(results, [...shape]);
    }
    return packResults(results, shape);
  }
  // Scalar input
  const r = callFn(collectArgs(0));
  if (typeof r === "number") return r;
  return ensureRuntimeValue(r);
}

export function arrayfunImpl(
  rt: Runtime,
  nargout: number,
  args: unknown[]
): unknown {
  return arrayfunCellfunImpl(rt, "arrayfun", nargout, args);
}

export function cellfunImpl(
  rt: Runtime,
  nargout: number,
  args: unknown[]
): unknown {
  // Handle string-based cellfun: cellfun('isempty', C), cellfun('isclass', C, 'className'), etc.
  if (args.length >= 2) {
    const fnArg = ensureRuntimeValue(args[0]);
    if (isRuntimeChar(fnArg) || isRuntimeString(fnArg)) {
      const fnName = typeof fnArg === "string" ? fnArg : fnArg.value;
      const cellArg = ensureRuntimeValue(args[1]);
      if (!isRuntimeCell(cellArg))
        throw new RuntimeError("cellfun: second argument must be a cell array");
      if (fnName === "isclass") {
        if (args.length < 3)
          throw new RuntimeError(
            "cellfun('isclass', ...) requires a class name argument"
          );
        const classNameRv = ensureRuntimeValue(args[2]);
        if (!isRuntimeChar(classNameRv) && !isRuntimeString(classNameRv))
          throw new RuntimeError(
            "cellfun('isclass', ...) class name must be a string"
          );
        const className =
          typeof classNameRv === "string" ? classNameRv : classNameRv.value;
        const data = allocFloat64Array(cellArg.data.length);
        for (let i = 0; i < cellArg.data.length; i++) {
          data[i] = valueClass(cellArg.data[i]) === className ? 1 : 0;
        }
        const result = RTV.tensor(data, [...cellArg.shape]);
        result._isLogical = true;
        return result;
      }
      // Convert string function name to function handle and delegate
      const handle = getFuncHandle(fnName);
      return arrayfunCellfunImpl(rt, "cellfun", nargout, [
        handle,
        ...args.slice(1),
      ]);
    }
  }
  return arrayfunCellfunImpl(rt, "cellfun", nargout, args);
}

export function structfunImpl(
  rt: Runtime,
  _nargout: number,
  args: unknown[]
): unknown {
  if (args.length < 2)
    throw new RuntimeError("structfun requires at least 2 arguments");
  const fnArg = args[0];
  const sArg = ensureRuntimeValue(args[1]);
  if (!isRuntimeStruct(sArg))
    throw new RuntimeError(
      "structfun: second argument must be a scalar struct"
    );

  // Parse name-value pairs
  let uniformOutput = true;
  for (let i = 2; i + 1 < args.length; i += 2) {
    const key = ensureRuntimeValue(args[i]);
    if (
      (isRuntimeChar(key) || isRuntimeString(key)) &&
      (typeof key === "string" ? key : key.value).toLowerCase() ===
        "uniformoutput"
    ) {
      const val = ensureRuntimeValue(args[i + 1]);
      uniformOutput = isRuntimeLogical(val)
        ? val
        : isRuntimeNumber(val)
          ? val !== 0
          : isRuntimeTensor(val)
            ? val.data[0] !== 0
            : true;
    }
  }

  const callFn = (elem: unknown): unknown => {
    if (typeof fnArg === "function") {
      return (fnArg as (...a: unknown[]) => unknown)(elem);
    }
    const mv = ensureRuntimeValue(fnArg);
    if (isRuntimeFunction(mv)) {
      return rt.index(mv, [elem]);
    }
    throw new RuntimeError("structfun: first argument must be a function");
  };

  const fieldNames = [...sArg.fields.keys()];

  if (uniformOutput) {
    // Collect results into a column vector
    const data = allocFloat64Array(fieldNames.length);
    for (let i = 0; i < fieldNames.length; i++) {
      const val = sArg.fields.get(fieldNames[i])!;
      const r = callFn(val);
      data[i] = toNumber(ensureRuntimeValue(r));
    }
    return RTV.tensor(data, [fieldNames.length, 1]);
  } else {
    // Collect results into a struct with the same field names
    const fields = new Map<string, RuntimeValue>();
    for (const name of fieldNames) {
      const val = sArg.fields.get(name)!;
      const r = callFn(val);
      fields.set(name, ensureRuntimeValue(r));
    }
    return RTV.struct(fields);
  }
}

/** Map known operator names to their direct arithmetic implementations. */
const bsxfunOpMap: Record<
  string,
  (a: RuntimeValue, b: RuntimeValue) => RuntimeValue
> = {
  plus: mAdd,
  minus: mSub,
  times: mElemMul,
  rdivide: mElemDiv,
};

/**
 * If fnArg is a known operator handle (@times, @plus, etc.), return the
 * direct arithmetic function that already handles broadcasting natively.
 */
function resolveKnownBsxfunOp(
  fnArg: unknown
): ((a: RuntimeValue, b: RuntimeValue) => RuntimeValue) | undefined {
  if (typeof fnArg === "function") {
    // JS function — can't identify by name
    return undefined;
  }
  const mv = ensureRuntimeValue(fnArg);
  if (isRuntimeFunction(mv) && mv.impl === "builtin") {
    return bsxfunOpMap[mv.name];
  }
  return undefined;
}

export function bsxfunImpl(
  rt: Runtime,
  _nargout: number,
  args: unknown[]
): unknown {
  if (args.length !== 3)
    throw new RuntimeError("bsxfun requires exactly 3 arguments");

  const fnArg = args[0];
  let callFn: (a: unknown, b: unknown) => unknown;
  if (typeof fnArg === "function") {
    callFn = (a, b) => (fnArg as (...a: unknown[]) => unknown)(a, b);
  } else {
    const mv = ensureRuntimeValue(fnArg);
    if (isRuntimeFunction(mv)) {
      callFn = (a, b) => rt.index(mv, [a, b]);
    } else {
      throw new RuntimeError(
        "bsxfun: first argument must be a function handle"
      );
    }
  }

  // Fast path: known operator handles bypass element-by-element dispatch
  const knownOp = resolveKnownBsxfunOp(fnArg);
  if (knownOp) {
    return knownOp(ensureRuntimeValue(args[1]), ensureRuntimeValue(args[2]));
  }

  const rawA = ensureRuntimeValue(args[1]);
  const rawB = ensureRuntimeValue(args[2]);
  const a = coerceToTensor(rawA, "bsxfun");
  const b = coerceToTensor(rawB, "bsxfun");

  const outShape = getBroadcastShape(a.shape, b.shape);
  if (!outShape)
    throw new RuntimeError(
      "bsxfun: non-singleton dimensions of the two input arrays must match each other"
    );

  const totalElems = outShape.reduce((acc, d) => acc * d, 1);
  const resultData = allocFloat64Array(totalElems);
  let resultImag: Float64Array | undefined;

  const aIsReal = !a.imag;
  const bIsReal = !b.imag;
  const ndim = outShape.length;
  const aPadded =
    a.shape.length >= ndim
      ? a.shape
      : [...a.shape, ...new Array(ndim - a.shape.length).fill(1)];
  const bPadded =
    b.shape.length >= ndim
      ? b.shape
      : [...b.shape, ...new Array(ndim - b.shape.length).fill(1)];

  // Precompute strides — zero for broadcast (size-1) dimensions
  const aStrides = new Array(ndim);
  const bStrides = new Array(ndim);
  let aStr = 1,
    bStr = 1;
  for (let d = 0; d < ndim; d++) {
    aStrides[d] = aPadded[d] === 1 ? 0 : aStr;
    bStrides[d] = bPadded[d] === 1 ? 0 : bStr;
    aStr *= aPadded[d];
    bStr *= bPadded[d];
  }

  const subs = new Array(ndim).fill(0);
  let aIdx = 0,
    bIdx = 0;
  for (let i = 0; i < totalElems; i++) {
    let r;
    if (aIsReal && bIsReal) {
      r = callFn(a.data[aIdx], b.data[bIdx]);
    } else if (aIsReal && !bIsReal) {
      r = callFn(a.data[aIdx], RTV.complex(b.data[bIdx], b.imag![bIdx]));
    } else if (!aIsReal && bIsReal) {
      r = callFn(RTV.complex(a.data[aIdx], a.imag![aIdx]), b.data[bIdx]);
    } else {
      r = callFn(
        RTV.complex(a.data[aIdx], a.imag![aIdx]),
        RTV.complex(b.data[bIdx], b.imag![bIdx])
      );
    }
    if (typeof r === "number") {
      resultData[i] = r;
    } else {
      const rv = ensureRuntimeValue(r);
      if (isRuntimeComplexNumber(rv)) {
        resultData[i] = rv.re;
        if (!resultImag) resultImag = allocFloat64Array(totalElems);
        resultImag[i] = rv.im;
      } else {
        resultData[i] = toNumber(rv);
      }
    }
    for (let d = 0; d < ndim; d++) {
      subs[d]++;
      aIdx += aStrides[d];
      bIdx += bStrides[d];
      if (subs[d] < outShape[d]) break;
      aIdx -= subs[d] * aStrides[d];
      bIdx -= subs[d] * bStrides[d];
      subs[d] = 0;
    }
  }

  return RTV.tensor(resultData, outShape, resultImag);
}

// ── subsref / subsasgn builtins ─────────────────────────────────────────

function extractSubsType(val: unknown): string {
  const rv = ensureRuntimeValue(val);
  if (isRuntimeChar(rv) || isRuntimeString(rv))
    return typeof rv === "string" ? rv : rv.value;
  throw new RuntimeError(`Expected string, got ${kstr(rv)}`);
}

function extractSubsIndices(subs: unknown): unknown[] {
  const rv = ensureRuntimeValue(subs);
  if (isRuntimeCell(rv)) {
    return rv.data.map(v =>
      isRuntimeChar(v) && v.value === ":" ? COLON_SENTINEL : v
    );
  }
  return [subs];
}

/** Apply a single subscript entry (shared by subsref and subsasgn). */
function applySubsEntry(
  rt: Runtime,
  entry: { type: string; subs: RuntimeValue },
  target: unknown,
  nargout: number,
  fnName: string
): unknown {
  if (entry.type === ".") {
    return rt.getMember(target, extractSubsType(entry.subs));
  }
  if (entry.type === "()") {
    return rt.index(target, extractSubsIndices(entry.subs), nargout);
  }
  if (entry.type === "{}") {
    return rt.indexCell(target, extractSubsIndices(entry.subs));
  }
  throw new RuntimeError(
    `${fnName}: unsupported subscript type '${entry.type}'`
  );
}

export function subsrefBuiltin(
  rt: Runtime,
  nargout: number,
  args: unknown[]
): unknown {
  if (args.length !== 2)
    throw new RuntimeError("subsref requires exactly 2 arguments");
  const obj = args[0];
  const S = ensureRuntimeValue(args[1]);

  type SubsEntry = { type: string; subs: RuntimeValue };
  const elements: SubsEntry[] = [];
  if (isRuntimeStruct(S)) {
    const typeVal = S.fields.get("type");
    const subsVal = S.fields.get("subs");
    if (!typeVal || !subsVal)
      throw new RuntimeError("subsref: S must have 'type' and 'subs' fields");
    elements.push({ type: extractSubsType(typeVal), subs: subsVal });
  } else if (isRuntimeStructArray(S)) {
    for (const el of S.elements) {
      const typeVal = el.fields.get("type");
      const subsVal = el.fields.get("subs");
      if (!typeVal || !subsVal)
        throw new RuntimeError("subsref: S must have 'type' and 'subs' fields");
      elements.push({ type: extractSubsType(typeVal), subs: subsVal });
    }
  } else {
    throw new RuntimeError(
      "subsref: second argument must be a subscript structure"
    );
  }

  let result: unknown = obj;
  for (const entry of elements) {
    result = applySubsEntry(rt, entry, result, nargout, "subsref");
  }
  return result;
}

export function subsasgnBuiltin(
  rt: Runtime,
  _nargout: number,
  args: unknown[]
): unknown {
  if (args.length !== 3)
    throw new RuntimeError("subsasgn requires exactly 3 arguments");
  const obj = args[0];
  const S = ensureRuntimeValue(args[1]);
  const B = args[2];

  type SubsEntry = { type: string; subs: RuntimeValue };
  const elements: SubsEntry[] = [];
  if (isRuntimeStruct(S)) {
    const typeVal = S.fields.get("type");
    const subsVal = S.fields.get("subs");
    if (!typeVal || !subsVal)
      throw new RuntimeError("subsasgn: S must have 'type' and 'subs' fields");
    elements.push({ type: extractSubsType(typeVal), subs: subsVal });
  } else if (isRuntimeStructArray(S)) {
    for (const el of S.elements) {
      const typeVal = el.fields.get("type");
      const subsVal = el.fields.get("subs");
      if (!typeVal || !subsVal)
        throw new RuntimeError(
          "subsasgn: S must have 'type' and 'subs' fields"
        );
      elements.push({ type: extractSubsType(typeVal), subs: subsVal });
    }
  } else {
    throw new RuntimeError(
      "subsasgn: second argument must be a subscript structure"
    );
  }

  // Navigate to the parent of the leaf, collecting intermediates
  const intermediates: unknown[] = [obj];
  let current: unknown = obj;
  for (let i = 0; i < elements.length - 1; i++) {
    current = applySubsEntry(rt, elements[i], current, 1, "subsasgn");
    intermediates.push(current);
  }

  // Perform the final assignment
  const last = elements[elements.length - 1];
  let result: unknown;
  if (last.type === ".") {
    const fieldName = extractSubsType(last.subs);
    result = rt.setMemberReturn(current, fieldName, B);
  } else if (last.type === "()") {
    const indices = extractSubsIndices(last.subs);
    result = rt.indexStore(current, indices, B);
  } else if (last.type === "{}") {
    const indices = extractSubsIndices(last.subs);
    result = rt.indexCellStore(current, indices, B);
  } else {
    throw new RuntimeError(
      `subsasgn: unsupported subscript type '${last.type}'`
    );
  }

  // Propagate changes back up the chain
  for (let i = elements.length - 2; i >= 0; i--) {
    const entry = elements[i];
    if (entry.type === ".") {
      const fieldName = extractSubsType(entry.subs);
      result = rt.setMemberReturn(intermediates[i], fieldName, result);
    } else if (entry.type === "()") {
      const indices = extractSubsIndices(entry.subs);
      result = rt.indexStore(intermediates[i], indices, result, false, true);
    } else if (entry.type === "{}") {
      const indices = extractSubsIndices(entry.subs);
      result = rt.indexCellStore(intermediates[i], indices, result);
    }
  }
  return result;
}
