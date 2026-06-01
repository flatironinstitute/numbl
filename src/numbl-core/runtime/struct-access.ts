/**
 * Struct and class instance field access.
 */

import {
  RuntimeClassInstance,
  isRuntimeClassInstance,
  isRuntimeGraphicsHandle,
  isRuntimeNumber,
  isRuntimeChar,
  isRuntimeString,
  isRuntimeTensor,
  isRuntimeStruct,
  isRuntimeStructArray,
  kstr,
  type RuntimeValue,
} from "./types.js";
import { isShared, type RefcountRuntime } from "./refcount.js";
import { RuntimeError } from "./error.js";
import { RTV } from "./constructors.js";
import { allocFloat64Array } from "./alloc.js";
import { horzcat } from "./tensor-construction.js";

/** Convert a plain JS value stored on a graphics-handle trace into a runtime
 *  value when read as a property (e.g. `H.LineWidth`). */
function handlePropToRuntime(v: unknown): RuntimeValue {
  if (typeof v === "number") return RTV.num(v);
  if (typeof v === "boolean") return RTV.logical(v);
  if (typeof v === "string") return RTV.char(v);
  if (Array.isArray(v)) {
    const data = allocFloat64Array(v as number[]);
    return RTV.tensor(data, [1, v.length]);
  }
  return v as RuntimeValue;
}

/** Default values for documented handle properties not stored on the trace
 *  (e.g. `q.LineWidth` before any width is set), keyed by trace type and the
 *  MATLAB (PascalCase) property name. */
const HANDLE_DEFAULTS: Record<string, Record<string, unknown>> = {
  quiver3: {
    LineWidth: 0.5,
    LineStyle: "-",
    ShowArrowHead: true,
    AutoScale: true,
    AutoScaleFactor: 0.9,
    Marker: "none",
    Color: [0, 0.447, 0.741],
  },
};

/** Resolve a graphics-handle property name to the key actually stored on the
 *  trace. MATLAB uses PascalCase property names (`LineWidth`); numbl traces
 *  store camelCase fields (`lineWidth`). Returns the matching key, or null. */
function resolveHandleKey(
  trace: Record<string, unknown>,
  field: string
): string | null {
  if (field in trace) return field;
  const camel = field.charAt(0).toLowerCase() + field.slice(1);
  if (camel in trace) return camel;
  // MATLAB data properties (XData, UData, …) map to short trace fields.
  if (field.endsWith("Data")) {
    const base = field.slice(0, -4);
    const short = base.charAt(0).toLowerCase() + base.slice(1);
    if (short in trace) return short;
  }
  return null;
}

/** Coerce a runtime value to the JS value stored on a graphics-handle trace,
 *  matching the existing field's type. Booleans accept MATLAB on/off
 *  semantics ('off'/'false'/'0' → false). */
function runtimeToHandleValue(value: RuntimeValue, current: unknown): unknown {
  if (typeof current === "boolean") {
    if (typeof value === "boolean") return value;
    if (typeof value === "number") return value !== 0;
    if (isRuntimeString(value) || isRuntimeChar(value)) {
      const s = (isRuntimeChar(value) ? value.value : value).toLowerCase();
      return !(s === "off" || s === "false" || s === "0");
    }
    return true;
  }
  if (typeof current === "string") {
    return isRuntimeChar(value)
      ? value.value
      : isRuntimeString(value)
        ? value
        : String(value as unknown);
  }
  if (typeof current === "number" || current === undefined) {
    if (typeof value === "number") return value;
    if (isRuntimeNumber(value)) return value;
  }
  if (Array.isArray(current) && isRuntimeTensor(value)) {
    return Array.from(value.data);
  }
  return value;
}

export function getRTValueField(
  base: RuntimeValue,
  field: string
): RuntimeValue {
  if (isRuntimeStruct(base) || isRuntimeClassInstance(base)) {
    const val = base.fields.get(field);
    if (val === undefined) {
      throw new RuntimeError(
        `No field '${field}' on ${isRuntimeClassInstance(base) ? base.className : "struct"}`
      );
    }
    return val;
  }
  if (isRuntimeStructArray(base)) {
    if (!base.fieldNames.includes(field)) {
      throw new RuntimeError(`No field '${field}' on struct array`);
    }
    const values = base.elements.map(el => {
      const val = el.fields.get(field);
      if (val === undefined) {
        throw new RuntimeError(`No field '${field}' on struct element`);
      }
      return val;
    });
    return horzcat(...values);
  }
  if (isRuntimeGraphicsHandle(base)) {
    const key = resolveHandleKey(base._trace, field);
    if (key !== null) return handlePropToRuntime(base._trace[key]);
    const dflt = HANDLE_DEFAULTS[base._traceType]?.[field];
    if (dflt !== undefined) return handlePropToRuntime(dflt);
    throw new RuntimeError(
      `No property '${field}' on ${base._traceType} handle`
    );
  }
  throw new RuntimeError(`Cannot access field ${field} on ${kstr(base)}`);
}

export function setRTValueField(
  base: RuntimeValue,
  field: string,
  value: RuntimeValue,
  rt?: RefcountRuntime
): RuntimeValue {
  if (isRuntimeClassInstance(base)) {
    if (base.isHandleClass) {
      // Handle class: mutate shared fields in place and return same reference.
      // bindField does decref-old / incref-new so the slot's refcount stays
      // accurate. If rt isn't provided the bookkeeping is skipped (lax-mode
      // — phase 6 will require rt at every call site).
      if (rt) {
        base.bindField(rt, field, value);
      } else {
        base.fields.set(field, value);
      }
      return base;
    }
    // Value class. Mutate in place if uniquely owned (the lvalue chain
    // walker has already COWed any shared ancestor); otherwise return a
    // new instance with copied fields.
    if (rt && !isShared(base)) {
      base.bindField(rt, field, value);
      return base;
    }
    const newFields = new Map(base.fields);
    newFields.set(field, value);
    return new RuntimeClassInstance(
      base.className,
      newFields,
      false,
      base._builtinData
    );
  }
  if (isRuntimeStruct(base)) {
    if (rt && !isShared(base)) {
      base.bindField(rt, field, value);
      return base;
    }
    const newFields = new Map(base.fields);
    newFields.set(field, value);
    return RTV.struct(newFields);
  }
  if (isRuntimeStructArray(base)) {
    if (base.elements.length === 1) {
      // Scalar struct array: set field on the single element, return as plain struct
      const newFields = new Map(base.elements[0].fields);
      newFields.set(field, value);
      return RTV.struct(newFields);
    }
    throw new RuntimeError(
      `Cannot assign field '${field}' on a non-scalar struct array without indexing`
    );
  }
  if (isRuntimeGraphicsHandle(base)) {
    // Mutate the trace in place. Because the handle wraps the same trace
    // object stored in the plot instruction, the change is reflected when
    // the figure renders (e.g. `q.ShowArrowHead = 'off'`).
    const key =
      resolveHandleKey(base._trace, field) ??
      field.charAt(0).toLowerCase() + field.slice(1);
    const current =
      base._trace[key] ?? HANDLE_DEFAULTS[base._traceType]?.[field];
    base._trace[key] = runtimeToHandleValue(value, current);
    return base;
  }
  // Auto-create struct from num 0 or undefined
  if (isRuntimeNumber(base) && base === 0) {
    return RTV.struct(new Map([[field, value]]));
  }
  // Auto-create struct
  return RTV.struct(new Map([[field, value]]));
}
