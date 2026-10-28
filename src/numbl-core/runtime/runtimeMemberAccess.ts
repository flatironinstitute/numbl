/**
 * Member access operations for the runtime.
 */

import {
  type RuntimeValue,
  RTV,
  toString,
  mGetField,
  mSetField,
  RuntimeError,
} from "../runtime/index.js";
import {
  isRuntimeStruct,
  isRuntimeClassInstance,
  isRuntimeClassInstanceArray,
  type RuntimeClassInstance,
} from "../runtime/types.js";
import { horzcat } from "./tensor-construction.js";
import { ensureRuntimeValue } from "./runtimeHelpers.js";
import type { Runtime } from "./runtime.js";
import {
  MEXCEPTION_CLASS,
  MEXCEPTION_PROPERTIES,
  isMException,
} from "./mexception.js";
import type { PropertyInfo } from "../workspace/classInfo.js";
import {
  invalidHandleError,
  noSetMethodError,
  propertyGetError,
  propertySetError,
  unrecognizedMemberError,
  unrecognizedPropertyError,
} from "./classAccess.js";

/** Whether `obj.name` on an instance of `className` may go to the class's
 *  overloaded subsref (or subsasgn): the class has one and is not already
 *  running it. */
function hasOverload(
  rt: Runtime,
  className: string,
  method: "subsref" | "subsasgn"
): boolean {
  return (
    !rt.activeAccessors.has(`${className}.${method}`) &&
    rt.cachedResolveClassMethod(className, method) !== null
  );
}

/** The declaration of property `name` of a `className` object, checked
 *  against the running code's access. When the code may not read it, the
 *  class's overloaded subsref handles the access ("subsref" is returned),
 *  or MATLAB's error is thrown. */
export function readableProperty(
  rt: Runtime,
  className: string,
  name: string
): PropertyInfo | undefined | "subsref" {
  const table = rt.classTable;
  const prop = table?.findProperty(className, name);
  if (!prop) return undefined;
  const denied = propertyGetError(table!, rt.codeSite(), className, prop);
  if (denied) {
    if (hasOverload(rt, className, "subsref")) return "subsref";
    throw denied;
  }
  return prop;
}

export function getMember(
  rt: Runtime,
  base: unknown,
  name: string,
  nargout = 1
): unknown {
  const mv = ensureRuntimeValue(base);
  if (isRuntimeClassInstance(mv)) {
    const prop = readableProperty(rt, mv.className, name);
    if (mv._deleted && prop) throw invalidHandleError();
    if (prop === "subsref") return subsrefCall(rt, base, [name]);
    if (prop?.constant) return rt.constantPropertyValue!(prop);
    // A GetObservable property triggers PreGet and PostGet around the read
    // (not inside its own get method).
    if (
      prop?.getObservable &&
      rt.propertyEvent &&
      !rt.activeAccessors.has(`${mv.className}.get.${name}`)
    ) {
      rt.propertyEvent(mv, prop, "PreGet");
      const value = classMember(rt, mv, base, name, nargout);
      rt.propertyEvent(mv, prop, "PostGet");
      return value;
    }
    return classMember(rt, mv, base, name, nargout);
  }
  // Property access on an object array yields a comma-separated list; numbl
  // materializes it by horzcat-ing the per-element values (mirroring struct
  // arrays, e.g. `[obj_array.prop]`). Route each element through getMember so
  // getters / Dependent properties are respected.
  if (isRuntimeClassInstanceArray(mv)) {
    const values = mv.elements.map(el =>
      ensureRuntimeValue(getMember(rt, el, name))
    );
    return horzcat(...values);
  }
  return mGetField(mv, name);
}

/** `obj.name` on a classdef object once access is checked: its get method,
 *  stored value, or a method called without arguments. */
function classMember(
  rt: Runtime,
  mv: RuntimeClassInstance,
  base: unknown,
  name: string,
  nargout: number
): unknown {
  // Check for property getter method (get.PropertyName)
  const accessorKey = `${mv.className}.get.${name}`;
  if (!rt.activeAccessors.has(accessorKey)) {
    const getter = rt.cachedResolveClassMethod(mv.className, `get.${name}`);
    if (getter) {
      rt.activeAccessors.add(accessorKey);
      try {
        return getter(1, base);
      } finally {
        rt.activeAccessors.delete(accessorKey);
      }
    }
  }
  // Check stored field
  const field = mv.fields.get(name);
  if (field !== undefined) return field;
  // Fall back to calling as a no-argument method (walks inheritance chain);
  // nargout carries through so `[a,b] = obj.method;` binds all outputs
  const method = rt.cachedResolveClassMethod(mv.className, name);
  if (method) {
    return method(nargout, base);
  }
  // Runtime fallback: if the class has a subsref method and we're not
  // already inside one, route through subsref (handles cases where
  // compile-time type info was unavailable)
  const guardKey = `${mv.className}.subsref`;
  if (!rt.activeAccessors.has(guardKey)) {
    const subsrefFn = rt.cachedResolveClassMethod(mv.className, "subsref");
    if (subsrefFn) {
      return subsrefCall(rt, base, [name]);
    }
  }
  throw unrecognizedMemberError(name, mv.className);
}

export function getMemberDynamic(
  rt: Runtime,
  base: unknown,
  nameExpr: unknown
): RuntimeValue {
  const name = toString(ensureRuntimeValue(nameExpr));
  // Route through getMember so property getters (get.Prop) and object-array
  // expansion are honored for dynamic field access `obj.(name)`, exactly as
  // for static `obj.Prop`.
  return ensureRuntimeValue(getMember(rt, base, name));
}

export function getMemberOrEmpty(base: unknown, name: string): RuntimeValue {
  try {
    const mv = ensureRuntimeValue(base);
    if (isRuntimeStruct(mv) || isRuntimeClassInstance(mv)) {
      const field = mv.fields.get(name);
      if (field !== undefined) return field;
    }
  } catch {
    // fall through
  }
  return RTV.struct(new Map());
}

export function setMemberReturn(
  rt: Runtime,
  base: unknown,
  name: string,
  rhs: unknown
): unknown {
  const mv = ensureRuntimeValue(base);
  if (isRuntimeClassInstance(mv)) {
    if (
      MEXCEPTION_PROPERTIES.includes(name) &&
      isMException(mv, c => rt.classAncestors(c))
    ) {
      // MATLAB's message doubles the quotes around the class name.
      throw new RuntimeError(
        `Unable to set the '${name}' property of class ''${MEXCEPTION_CLASS}'' because it is read-only.`,
        null,
        { identifier: "MATLAB:class:SetProhibited" }
      );
    }
    const table = rt.classTable;
    const prop = table?.findProperty(mv.className, name);
    if (mv._deleted && prop) throw invalidHandleError();
    if (prop) {
      const denied = propertySetError(
        table!,
        rt.codeSite(),
        mv.className,
        prop
      );
      if (denied) {
        // An overloaded subsasgn handles assignments the class does not
        // expose.
        if (hasOverload(rt, mv.className, "subsasgn"))
          return subsasgnCall(rt, base, [name], rhs);
        throw denied;
      }
    } else if (
      table?.get(mv.className)?.ast &&
      !mv.fields.has(name) &&
      !table.isSubclassOf(mv.className, "dynamicprops") &&
      rt.cachedResolveClassMethod(mv.className, "subsasgn") === null
    ) {
      // A classdef object has only its declared (or dynamic) properties.
      throw unrecognizedPropertyError(name, mv.className);
    }
    // Property validation runs before any set.PropertyName method.
    if (rt.validateProperty)
      rhs = rt.validateProperty(mv.className, name, ensureRuntimeValue(rhs));
    // A SetObservable property triggers PreSet and PostSet around the
    // assignment (not inside its own set method).
    const accessorKey = `${mv.className}.set.${name}`;
    const inSetter = rt.activeAccessors.has(accessorKey);
    if (prop?.setObservable && rt.propertyEvent && !inSetter) {
      rt.propertyEvent(mv, prop, "PreSet");
      const result = assignProperty(rt, mv, base, name, rhs, prop);
      rt.propertyEvent(mv, prop, "PostSet");
      return result;
    }
    if (!inSetter) return assignProperty(rt, mv, base, name, rhs, prop);
  }
  const rhsMv = ensureRuntimeValue(rhs);
  return mSetField(mv, name, rhsMv, rt);
}

/** `obj.name = rhs` on a classdef object once access is checked and the
 *  value validated: through its set method, or stored. */
function assignProperty(
  rt: Runtime,
  mv: RuntimeClassInstance,
  base: unknown,
  name: string,
  rhs: unknown,
  prop: PropertyInfo | undefined
): unknown {
  const accessorKey = `${mv.className}.set.${name}`;
  const setter = rt.cachedResolveClassMethod(mv.className, `set.${name}`);
  if (!setter && prop?.dependent) throw noSetMethodError(prop);
  if (setter) {
    rt.activeAccessors.add(accessorKey);
    try {
      // Call with nargout=0: a handle-class setter declares no output
      // (`set.Prop(obj, v)`), so requesting an output would raise "Too
      // many output arguments". A value-class setter
      // (`obj = set.Prop(obj, v)`) still returns its first output even at
      // nargout=0, so its modified object is captured below.
      const result = setter(0, base, rhs);
      return result !== undefined ? result : base;
    } finally {
      rt.activeAccessors.delete(accessorKey);
    }
  }
  return mSetField(mv, name, ensureRuntimeValue(rhs), rt);
}

export function setMemberDynamicReturn(
  rt: Runtime,
  base: unknown,
  nameExpr: unknown,
  rhs: unknown
): RuntimeValue {
  const name = toString(ensureRuntimeValue(nameExpr));
  // Route through setMemberReturn so property setters (set.Prop) are honored
  // for dynamic field assignment `obj.(name) = v`, exactly as for static
  // `obj.Prop = v`.
  return ensureRuntimeValue(setMemberReturn(rt, base, name, rhs));
}

/**
 * Call a user-defined subsref method on a class instance.
 * Constructs S = struct('type', '.', 'subs', name) for each name in the chain.
 */
export function subsrefCall(
  rt: Runtime,
  base: unknown,
  names: string[]
): unknown {
  const mv = ensureRuntimeValue(base);
  if (!isRuntimeClassInstance(mv)) {
    // Not a class instance — fall back to chained getMember
    let result: unknown = base;
    for (const name of names) result = getMember(rt, result, name);
    return result;
  }

  // Recursion guard: prevent infinite recursion when subsref accesses obj.field
  const guardKey = `${mv.className}.subsref`;
  if (rt.activeAccessors.has(guardKey)) {
    let result: unknown = base;
    for (const name of names) result = getMember(rt, result, name);
    return result;
  }

  // Look up the subsref method
  const subsrefFn = rt.cachedResolveClassMethod(mv.className, "subsref");
  if (!subsrefFn) {
    let result: unknown = base;
    for (const name of names) result = getMember(rt, result, name);
    return result;
  }

  // Construct S as a struct array with {type: '.', subs: name} entries
  const S = RTV.structArray(
    ["type", "subs"],
    names.map(n => RTV.struct({ type: RTV.char("."), subs: RTV.char(n) }))
  );

  rt.activeAccessors.add(guardKey);
  try {
    return subsrefFn(1, base, S);
  } finally {
    rt.activeAccessors.delete(guardKey);
  }
}

/**
 * Call a user-defined subsasgn method on a class instance.
 * Constructs S = struct('type', '.', 'subs', name) for each name in the chain.
 */
export function subsasgnCall(
  rt: Runtime,
  base: unknown,
  names: string[],
  rhs: unknown
): unknown {
  const mv = ensureRuntimeValue(base);
  if (!isRuntimeClassInstance(mv)) {
    return subsasgnFallback(rt, base, names, rhs);
  }

  // Recursion guard
  const guardKey = `${mv.className}.subsasgn`;
  if (rt.activeAccessors.has(guardKey)) {
    return subsasgnFallback(rt, base, names, rhs);
  }

  // Look up the subsasgn method
  const subsasgnFn = rt.cachedResolveClassMethod(mv.className, "subsasgn");
  if (!subsasgnFn) {
    return subsasgnFallback(rt, base, names, rhs);
  }

  // Construct S as a struct array with {type: '.', subs: name} entries
  const S = RTV.structArray(
    ["type", "subs"],
    names.map(n => RTV.struct({ type: RTV.char("."), subs: RTV.char(n) }))
  );

  rt.activeAccessors.add(guardKey);
  try {
    const result = subsasgnFn(1, base, S, ensureRuntimeValue(rhs));
    return result !== undefined ? result : base;
  } finally {
    rt.activeAccessors.delete(guardKey);
  }
}

/** Fallback for subsasgnCall: chained getMember + final setMemberReturn */
function subsasgnFallback(
  rt: Runtime,
  base: unknown,
  names: string[],
  rhs: unknown
): unknown {
  if (names.length === 1) {
    return setMemberReturn(rt, base, names[0], rhs);
  }
  // Multi-level: navigate to leaf, set, propagate back
  const intermediates: unknown[] = [base];
  let current: unknown = base;
  for (let i = 0; i < names.length - 1; i++) {
    current = getMemberOrEmpty(current, names[i]);
    intermediates.push(current);
  }
  // Set the leaf
  let result: unknown = setMemberReturn(
    rt,
    intermediates[intermediates.length - 1],
    names[names.length - 1],
    rhs
  );
  // Propagate back up
  for (let i = names.length - 2; i >= 0; i--) {
    result = setMemberReturn(rt, intermediates[i], names[i], result);
  }
  return result;
}

/**
 * Runtime dispatch for member chain assignment when the type is unknown
 * at compile time. If the base is a class instance with overloaded
 * subsasgn, routes through subsasgnCall; otherwise uses normal chain.
 */
export function memberChainAssign(
  rt: Runtime,
  base: unknown,
  names: string[],
  rhs: unknown
): unknown {
  const mv = ensureRuntimeValue(base);
  if (isRuntimeClassInstance(mv)) {
    // If the top-level field is a declared property, set directly
    const isDeclaredProperty = mv.fields.has(names[0]);
    if (!isDeclaredProperty) {
      const guardKey = `${mv.className}.subsasgn`;
      if (!rt.activeAccessors.has(guardKey)) {
        const subsasgnFn = rt.cachedResolveClassMethod(
          mv.className,
          "subsasgn"
        );
        if (subsasgnFn) {
          return subsasgnCall(rt, base, names, rhs);
        }
      }
    }
  }
  return subsasgnFallback(rt, base, names, rhs);
}
