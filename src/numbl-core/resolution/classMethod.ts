/**
 * Method lookup by class name, for calls on an object (`obj.method(...)`,
 * property accessors, subsref/subsasgn overloads).
 */

import type { ClassTable } from "../workspace/classTable.js";
import type { ResolvedTarget } from "./resolveFunction.js";

/** The method `methodName` of `className` (an instance or static method, the
 *  constructor, or an `@Class` method file, own or inherited), as a target
 *  on its defining class. Null if the class has no such method. */
export function resolveClassMethod(
  classes: ClassTable,
  className: string,
  methodName: string
): {
  target: Extract<ResolvedTarget, { kind: "classMethod" }>;
  isStatic: boolean;
} | null {
  const info = classes.get(className);
  const exists =
    classes.hasMethod(className, methodName) ||
    classes.hasStaticMethod(className, methodName) ||
    (!!info &&
      (info.constructorName === methodName ||
        classes
          .lineage(info)
          .some(c => c.externalMethodFiles.has(methodName))));
  if (!exists) return null;
  const definingClass = classes.findDefiningClass(className, methodName);
  return {
    target: {
      kind: "classMethod",
      className: definingClass,
      methodName,
      stripInstance: false,
    },
    isStatic: classes.hasStaticMethod(definingClass, methodName),
  };
}
