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
import { resolveColor } from "./plotUtils.js";

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
  line: {
    Color: [0, 0.447, 0.741],
    LineStyle: "-",
    LineWidth: 0.5,
    Marker: "none",
    MarkerSize: 6,
  },
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
  // MATLAB property names are case-insensitive (`XData` ≡ `xdata`), so match
  // any existing trace key ignoring case.
  const lower = field.toLowerCase();
  for (const k of Object.keys(trace)) {
    if (k.toLowerCase() === lower) return k;
  }
  // MATLAB data properties (XData, YData, ZData, UData, …) map to the short
  // trace fields (x, y, z, u, …). Handled case-insensitively.
  if (lower.endsWith("data")) {
    const short = lower.slice(0, -4);
    for (const k of Object.keys(trace)) {
      if (k.toLowerCase() === short) return k;
    }
  }
  return null;
}

/** Coerce a runtime value to the JS value stored on a graphics-handle trace,
 *  matching the existing field's type. Booleans accept MATLAB on/off
 *  semantics ('off'/'false'/'0' → false). */
function runtimeToHandleValue(value: RuntimeValue, current: unknown): unknown {
  // A named/short color string assigned to an RGB-triplet property (e.g.
  // `pl.Color = 'green'`) resolves to its [r,g,b] triplet, since the renderer
  // expects a triplet — not the raw string.
  if (
    Array.isArray(current) &&
    current.length === 3 &&
    (isRuntimeString(value) || isRuntimeChar(value))
  ) {
    const rgb = resolveColor(value);
    if (rgb) return rgb;
  }
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
  if (Array.isArray(current)) {
    // Coordinate/data properties (XData, YData, …) hold number arrays. A tensor
    // becomes its flat data; a scalar becomes a one-element array.
    if (isRuntimeTensor(value)) return Array.from(value.data);
    if (typeof value === "number") return [value];
  }
  return value;
}

/** Apply one property to a graphics-handle trace, mutating it in place.
 *  Returns the resolved trace key and the stored JS value so callers (e.g.
 *  `set`) can re-emit the change as an `update_trace` instruction. Shared by
 *  dot-assignment (`h.Prop = v`) and `set(h,'Prop',v,...)`. */
export function applyHandleProperty(
  trace: Record<string, unknown>,
  traceType: string,
  field: string,
  value: RuntimeValue
): { key: string; value: unknown } {
  const key =
    resolveHandleKey(trace, field) ??
    field.charAt(0).toLowerCase() + field.slice(1);
  const current = trace[key] ?? HANDLE_DEFAULTS[traceType]?.[field];
  const stored = runtimeToHandleValue(value, current);
  trace[key] = stored;
  return { key, value: stored };
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
    applyHandleProperty(base._trace, base._traceType, field, value);
    return base;
  }
  // Auto-create struct from num 0 or undefined
  if (isRuntimeNumber(base) && base === 0) {
    return RTV.struct(new Map([[field, value]]));
  }
  // Auto-create struct
  return RTV.struct(new Map([[field, value]]));
}
