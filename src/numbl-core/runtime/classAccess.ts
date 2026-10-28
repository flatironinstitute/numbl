/**
 * Access checks for class members: whether the running code may read or set
 * a property or call a method, given the member's attributes and the class
 * of the running code. Each check returns MATLAB's error, or null when the
 * access is allowed.
 */

import { RuntimeError } from "./error.js";
import type { MethodInfo, PropertyInfo } from "../workspace/classInfo.js";
import type { ClassTable } from "../workspace/classTable.js";

/** Where code runs: the class and method whose code it is. */
export interface AccessSite {
  className?: string;
  methodName?: string;
}

function classError(identifier: string, message: string): RuntimeError {
  return new RuntimeError(message, null, { identifier });
}

export function unrecognizedMemberError(
  name: string,
  className: string
): RuntimeError {
  return classError(
    "MATLAB:noSuchMethodOrField",
    `Unrecognized method, property, or field '${name}' for class '${className}'.`
  );
}

/** Assignment to a property a classdef object does not have. */
export function unrecognizedPropertyError(
  name: string,
  className: string
): RuntimeError {
  return classError(
    "MATLAB:noPublicFieldForClass",
    `Unrecognized property '${name}' for class '${className}'.`
  );
}

/** MATLAB hides a superclass's private members from subclass code: they
 *  are reported as unknown rather than inaccessible. */
function hiddenFromSubclass(
  table: ClassTable,
  isPrivate: boolean,
  definingClass: string,
  context: string | undefined
): boolean {
  return (
    isPrivate &&
    context !== undefined &&
    context !== definingClass &&
    table.isSubclassOf(context, definingClass)
  );
}

/** The error for reading `prop` of an object of class `className` from
 *  `site`, or null when allowed. */
export function propertyGetError(
  table: ClassTable,
  site: AccessSite,
  className: string,
  prop: PropertyInfo
): RuntimeError | null {
  if (prop.getAccess === "public") return null;
  const ctx = site.className;
  if (table.canAccess(prop.getAccess, prop.definingClass, ctx)) return null;
  if (
    hiddenFromSubclass(
      table,
      prop.getAccess === "private",
      prop.definingClass,
      ctx
    )
  )
    return unrecognizedMemberError(prop.name, className);
  return classError(
    "MATLAB:class:GetProhibited",
    `No public property '${prop.name}' for class ''${className}''.`
  );
}

/** The error for setting `prop` of an object of class `className` from
 *  `site`, or null when allowed. An immutable property can be set only in
 *  the constructor of its defining class; a Constant property never. */
export function propertySetError(
  table: ClassTable,
  site: AccessSite,
  className: string,
  prop: PropertyInfo
): RuntimeError | null {
  const access = prop.setAccess;
  if (access === "public") return null;
  const ctx = site.className;
  if (access === "immutable") {
    if (
      ctx === prop.definingClass &&
      site.methodName !== undefined &&
      site.methodName === table.get(ctx)?.constructorName
    )
      return null;
  } else if (table.canAccess(access, prop.definingClass, ctx)) {
    return null;
  }
  if (
    hiddenFromSubclass(
      table,
      prop.getAccess === "private",
      prop.definingClass,
      ctx
    )
  )
    return unrecognizedPropertyError(prop.name, className);
  if (!table.canAccess(prop.getAccess, prop.definingClass, ctx))
    return classError(
      "MATLAB:class:SetProhibited",
      `Setting the '${prop.name}' property of class ''${className}'' is not supported.`
    );
  return classError(
    "MATLAB:class:SetProhibited",
    `Unable to set the '${prop.name}' property of class ''${className}'' because it is read-only.`
  );
}

/** The error for calling `method` (found through class `className`) from
 *  `site`, or null when allowed. */
export function methodAccessError(
  table: ClassTable,
  site: AccessSite,
  className: string,
  method: MethodInfo
): RuntimeError | null {
  if (method.access === "public") return null;
  const ctx = site.className;
  if (table.canAccess(method.access, method.definingClass, ctx)) return null;
  if (
    hiddenFromSubclass(
      table,
      method.access === "private",
      method.definingClass,
      ctx
    )
  )
    return unrecognizedMemberError(method.name, className);
  return classError(
    "MATLAB:class:MethodRestricted",
    `Cannot access method '${method.name}' in class '${className}'.`
  );
}

/** Using a property of a deleted handle object. */
export function invalidHandleError(): RuntimeError {
  return classError("MATLAB:class:InvalidHandle", "Invalid or deleted object.");
}

/** The error for assigning a Dependent property that has no set method. */
export function noSetMethodError(prop: PropertyInfo): RuntimeError {
  return classError(
    "MATLAB:class:noSetMethod",
    `In class '${prop.definingClass}', no set method is defined for dependent property '${prop.name}'. A dependent property needs a set method to assign its value.`
  );
}
