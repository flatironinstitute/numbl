/**
 * Classdef property validation: `Prop (dims) class {validators} = default`.
 * Applied to default values when an instance is built and to every
 * assignment of the property (before a `set.Prop` method runs).
 */

import type { RuntimeValue } from "../runtime/types.js";
import { RTV } from "../runtime/constructors.js";
import { ensureRuntimeValue } from "../runtime/runtimeHelpers.js";
import { invokeBuiltin } from "../runtime/runtimeDispatch.js";
import type { ClassInfo } from "../workspace/classInfo.js";

import { Environment } from "./types.js";
import { CallFrame } from "./callFrame.js";
import type { Interpreter } from "./interpreter.js";
import {
  compileCheck,
  validateValue,
  type CompiledCheck,
  type ErrorPrefix,
} from "./valueValidation.js";

interface PropertyCheck {
  readonly check: CompiledCheck;
  /** The class whose file declares the property (validators resolve
   *  there). */
  readonly owner: ClassInfo;
}

/** Compiled property checks of a class, including inherited properties;
 *  null when no property is validated. */
const classChecks = new WeakMap<
  ClassInfo,
  ReadonlyMap<string, PropertyCheck> | null
>();

export function propertyChecks(
  interp: Interpreter,
  info: ClassInfo
): ReadonlyMap<string, PropertyCheck> | null {
  let checks = classChecks.get(info);
  if (checks !== undefined) return checks;
  const map = new Map<string, PropertyCheck>();
  // Subclass declarations take precedence (lineage starts at the class).
  for (const cls of interp.code.classes.lineage(info)) {
    for (const [prop, spec] of cls.propertyValidations) {
      if (map.has(prop)) continue;
      const check = compileCheck(spec);
      if (check) map.set(prop, { check, owner: cls });
    }
  }
  checks = map.size > 0 ? map : null;
  classChecks.set(info, checks);
  return checks;
}

/** Validate `value` for property `prop` of an instance of `className`,
 *  returning the converted value (unchanged when the property has no
 *  declaration). */
export function validatePropertyValue(
  interp: Interpreter,
  className: string,
  prop: string,
  value: RuntimeValue
): RuntimeValue {
  const info = interp.code.classes.get(className);
  if (!info) return value;
  const pc = propertyChecks(interp, info)?.get(prop);
  if (!pc) return value;
  return runCheck(
    interp,
    pc,
    prop,
    value,
    () => `Error setting property '${prop}' of class '${className}'. `
  );
}

/** Apply a property check in a frame at the declaring class's file, with
 *  the property name bound to the value for call-expression validators
 *  (`{mustBeMember(Color, {'red','blue'})}`). */
function runCheck(
  interp: Interpreter,
  pc: PropertyCheck,
  prop: string,
  value: RuntimeValue,
  prefix: ErrorPrefix
): RuntimeValue {
  if (pc.check.validators.length === 0)
    return validateValue(interp, pc.check, value, prefix, null);
  const env = new Environment();
  env.rt = interp.rt;
  const saved = interp.frame;
  interp.frame = new CallFrame(
    interp.code,
    pc.owner.fileName,
    pc.owner.qualifiedName,
    undefined,
    env,
    saved,
    undefined
  );
  try {
    return validateValue(interp, pc.check, value, prefix, v =>
      env.set(prop, v)
    );
  } finally {
    env.clearLocals();
    interp.frame = saved;
  }
}

/** Apply property declarations to a class's evaluated defaults: convert
 *  and validate given defaults, and give a declared property without a
 *  default the default of its class and size (zeros for numeric classes,
 *  false for logical). */
export function applyPropertyDefaults(
  interp: Interpreter,
  info: ClassInfo,
  className: string,
  defaults: Map<string, RuntimeValue>
): void {
  const checks = propertyChecks(interp, info);
  if (!checks) return;
  for (const [prop, pc] of checks) {
    const given = defaults.get(prop);
    if (given !== undefined) {
      defaults.set(
        prop,
        runCheck(
          interp,
          pc,
          prop,
          given,
          () =>
            `Invalid default value for property '${prop}' in class '${className}':\n`
        )
      );
      continue;
    }
    const implicit = implicitDefault(interp, pc.check);
    if (implicit !== undefined) defaults.set(prop, implicit);
  }
}

const ZERO_CLASSES = new Set([
  "double",
  "single",
  "int8",
  "int16",
  "int32",
  "int64",
  "uint8",
  "uint16",
  "uint32",
  "uint64",
]);

/** MATLAB's default for a declared property without a default value: an
 *  array of the declared class and size (`:` taken as 0, no size as 0x0)
 *  of zeros, false, char(0), "" or [] cells. Other classes keep numbl's
 *  plain `[]` default. */
function implicitDefault(
  interp: Interpreter,
  check: CompiledCheck
): RuntimeValue | undefined {
  const cls = check.className;
  const dims = RTV.row(check.dims ? check.dims.map(d => d ?? 0) : [0, 0]);
  const call = (fn: string, args: unknown[]) =>
    ensureRuntimeValue(invokeBuiltin(interp.rt, fn, 1, args));
  if (cls === null) return check.dims ? call("zeros", [dims]) : undefined;
  if (ZERO_CLASSES.has(cls)) return call("zeros", [dims]);
  switch (cls) {
    case "logical":
      return call("false", [dims]);
    case "char":
      return call("char", [call("zeros", [dims])]);
    case "string":
      return call("strings", [dims]);
    case "cell":
      return call("cell", [dims]);
  }
  return undefined;
}
