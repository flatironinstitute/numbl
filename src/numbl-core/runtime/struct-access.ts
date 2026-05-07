/**
 * Struct and class instance field access.
 */

import {
  RuntimeClassInstance,
  isRuntimeClassInstance,
  isRuntimeNumber,
  isRuntimeStruct,
  isRuntimeStructArray,
  kstr,
  type RuntimeValue,
} from "./types.js";
import { isShared, type RefcountRuntime } from "./refcount.js";
import { RuntimeError } from "./error.js";
import { RTV } from "./constructors.js";
import { horzcat } from "./tensor-construction.js";

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
  // Auto-create struct from num 0 or undefined
  if (isRuntimeNumber(base) && base === 0) {
    return RTV.struct(new Map([[field, value]]));
  }
  // Auto-create struct
  return RTV.struct(new Map([[field, value]]));
}
