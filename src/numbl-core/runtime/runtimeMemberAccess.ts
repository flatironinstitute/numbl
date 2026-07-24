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
} from "../runtime/types.js";
import { horzcat } from "./tensor-construction.js";
import { ensureRuntimeValue } from "./runtimeHelpers.js";
import type { Runtime } from "./runtime.js";

export function getMember(
  rt: Runtime,
  base: unknown,
  name: string,
  nargout = 1
): unknown {
  const mv = ensureRuntimeValue(base);
  // Check for property getter method (get.PropertyName)
  if (isRuntimeClassInstance(mv)) {
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
    throw new RuntimeError(
      `No property or method '${name}' for class '${mv.className}'`
    );
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
  // Check for property setter method (set.PropertyName)
  if (isRuntimeClassInstance(mv)) {
    const accessorKey = `${mv.className}.set.${name}`;
    if (!rt.activeAccessors.has(accessorKey)) {
      const setter = rt.cachedResolveClassMethod(mv.className, `set.${name}`);
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
    }
  }
  const rhsMv = ensureRuntimeValue(rhs);
  return mSetField(mv, name, rhsMv, rt);
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
