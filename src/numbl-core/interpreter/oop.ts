/**
 * Classes: instantiation, constructors, superclass constructor calls,
 * class methods, enumerations and class introspection.
 */

import type { Expr } from "../parser/types.js";
import type { RuntimeValue } from "../runtime/types.js";
import {
  isRuntimeChar,
  isRuntimeString,
  isRuntimeTensor,
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeClassInstance,
  isRuntimeClassInstanceArray,
  RuntimeClassInstance,
} from "../runtime/types.js";
import { RTV } from "../runtime/constructors.js";
import { ensureRuntimeValue } from "../runtime/runtimeHelpers.js";
import { RuntimeError, CancellationError } from "../runtime/error.js";
import { toNumber } from "../runtime/convert.js";
import { allocFloat64Array } from "../runtime/alloc.js";
import {
  functionBodyFor,
  constructorBodyFor,
  oldStyleConstructorBodyFor,
  type ResolvedTarget,
} from "../resolution/index.js";
import type { ClassInfo } from "../workspace/classInfo.js";
import {
  MEXCEPTION_CLASS,
  MEXCEPTION_PROPERTIES,
} from "../runtime/mexception.js";

import type { Interpreter } from "./interpreter.js";
import { evalArgs, evalExpr } from "./evalExpressions.js";
import { callBody } from "./callConvention.js";

/** Built-in classes a classdef can subclass by value (`classdef Foo <
 *  double`, `enumeration` over `uint32`): the superclass constructor
 *  produces the instance's underlying value. */
const BUILTIN_VALUE_CLASSES: ReadonlySet<string> = new Set([
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
  "logical",
  "char",
]);

// ── Instantiation ────────────────────────────────────────────────────────

/** `ClassName(args)`: construct an instance (or convert to an enumeration
 *  member). */
export function instantiateClass(
  interp: Interpreter,
  className: string,
  args: unknown[],
  nargout: number
): unknown {
  const classInfo = interp.code.classes.get(className);
  if (!classInfo) {
    return interp.rt.callClassMethod(className, className, nargout, args);
  }

  // Enumeration class: `ClassName(x)` converts x to a member (or member array).
  if (classInfo.enumMembers && args.length > 0) {
    return convertToEnum(interp, className, classInfo, args);
  }

  // Old-style (pre-classdef) class: the constructor is a plain @Name/Name.m
  // function that builds its own instance via `class(struct, 'Name')`. Call it
  // directly with the user args — do NOT pre-create/prepend an instance.
  if (classInfo.isOldStyle) {
    const body = oldStyleConstructorBodyFor(interp.code, classInfo);
    if (!body)
      throw new RuntimeError(
        `Constructor for old-style class '${className}' not found`
      );
    return callBody(interp, body, args, nargout);
  }

  const { propertyNames, defaults } = propertyDefaults(
    interp,
    className,
    classInfo
  );
  const instance = RTV.classInstance(
    className,
    propertyNames,
    interp.code.classes.isHandleClass(classInfo),
    defaults
  );

  if (classInfo.constructorName) {
    return interpretConstructor(
      interp,
      classInfo,
      [instance, ...args],
      nargout
    );
  }
  return instance;
}

/** Run a classdef constructor on `args` (the new instance, then the user
 *  arguments). Without a constructor the instance is returned as is. */
function interpretConstructor(
  interp: Interpreter,
  classInfo: ClassInfo,
  args: unknown[],
  nargout: number
): unknown {
  const body = constructorBodyFor(interp.code, classInfo);
  if (!body) return args[0];
  // nargin in the constructor counts the user args, not the instance.
  return callBody(interp, body, args, nargout, args.length - 1);
}

/** `obj@Super(args)` (a superclass constructor call) or `method@Super(args)`
 *  (a superclass method call). */
export function evalSuperMethodCall(
  interp: Interpreter,
  expr: Extract<Expr, { type: "SuperMethodCall" }>,
  nargout: number
): unknown {
  const args = evalArgs(interp, expr.args);
  const objVal = interp.frame.env.get(expr.methodName);
  if (objVal === undefined) {
    return interp.rt.callClassMethod(
      expr.superClassName,
      expr.methodName,
      nargout,
      args
    );
  }
  const superClassInfo = interp.code.classes.get(expr.superClassName);
  if (superClassInfo && superClassInfo.constructorName) {
    const superResult = interpretConstructor(
      interp,
      superClassInfo,
      [objVal, ...args],
      1
    );
    return interp.rt.callSuperConstructor(objVal, superResult);
  }
  // Built-in super class (e.g. classdef Foo < double): pass the raw arg to
  // callSuperConstructor, which stores it as _builtinData.
  if (BUILTIN_VALUE_CLASSES.has(expr.superClassName)) {
    const data =
      args.length > 0
        ? ensureRuntimeValue(args[0])
        : RTV.tensor(allocFloat64Array(0), [0, 0]);
    return interp.rt.callSuperConstructor(objVal, data);
  }
  // `obj@MException(id, msg, ...)`: the built-in MException constructor
  // provides the exception properties.
  if (!superClassInfo && expr.superClassName === MEXCEPTION_CLASS) {
    const superInstance = interp.rt.callBuiltin(MEXCEPTION_CLASS, 1, args);
    return interp.rt.callSuperConstructor(objVal, superInstance);
  }
  // A superclass without a constructor: an instance with its defaults.
  const { propertyNames, defaults } = superClassInfo
    ? propertyDefaults(interp, expr.superClassName, superClassInfo)
    : { propertyNames: [], defaults: new Map<string, RuntimeValue>() };
  const superInstance = RTV.classInstance(
    expr.superClassName,
    propertyNames,
    false,
    defaults
  );
  return interp.rt.callSuperConstructor(objVal, superInstance);
}

// ── Properties ───────────────────────────────────────────────────────────

/** A class's property names, including inherited ones, and their evaluated
 *  default values. `className` names the class being built, for errors. */
function propertyDefaults(
  interp: Interpreter,
  className: string,
  classInfo: ClassInfo
): { propertyNames: string[]; defaults: Map<string, RuntimeValue> } {
  const propertyNames: string[] = [...classInfo.propertyNames];
  const defaultExprs = new Map(classInfo.propertyDefaults);
  for (const parentInfo of interp.code.classes.lineage(classInfo).slice(1)) {
    for (const propName of parentInfo.propertyNames) {
      if (!propertyNames.includes(propName)) {
        propertyNames.push(propName);
        const defaultExpr = parentInfo.propertyDefaults.get(propName);
        if (defaultExpr && !defaultExprs.has(propName)) {
          defaultExprs.set(propName, defaultExpr);
        }
      }
    }
  }
  const defaults = new Map<string, RuntimeValue>();
  for (const [propName, defaultExpr] of defaultExprs) {
    defaults.set(
      propName,
      evalPropertyDefault(interp, className, propName, defaultExpr)
    );
  }
  return { propertyNames, defaults };
}

/** Evaluate a property's default value expression. MATLAB reports an
 *  invalid default as an error rather than leaving the property unset. */
function evalPropertyDefault(
  interp: Interpreter,
  className: string,
  propName: string,
  defaultExpr: Expr
): RuntimeValue {
  try {
    return ensureRuntimeValue(evalExpr(interp, defaultExpr));
  } catch (e) {
    if (e instanceof CancellationError) throw e;
    const msg = e instanceof Error ? e.message : String(e);
    throw new RuntimeError(
      `Invalid default value for property '${propName}' in class '${className}':\n${msg}`
    );
  }
}

// ── Methods ──────────────────────────────────────────────────────────────

export function interpretClassMethod(
  interp: Interpreter,
  target: Extract<ResolvedTarget, { kind: "classMethod" }>,
  args: unknown[],
  nargout: number
): unknown {
  const { className, methodName } = target;
  const definingClass = interp.code.classes.findDefiningClass(
    className,
    methodName
  );
  const classInfo = interp.code.classes.get(definingClass);
  if (!classInfo) throw new RuntimeError(`Class '${definingClass}' not found`);

  if (methodName === classInfo.constructorName) {
    return interpretConstructor(interp, classInfo, args, nargout);
  }

  const body = functionBodyFor(interp.code, target);
  if (body) {
    const actualArgs =
      target.stripInstance && args.length > 0 ? args.slice(1) : args;
    return callBody(interp, body, actualArgs, nargout);
  }
  // Implicit static `ClassName.empty(m,n)`: every MATLAB class inherits
  // an `empty` method producing an empty object array.
  if (methodName === "empty") {
    const dims: number[] = [];
    for (const a of args) {
      const rv = ensureRuntimeValue(a);
      if (isRuntimeTensor(rv)) for (const d of rv.data) dims.push(d);
      else dims.push(toNumber(rv));
    }
    if (dims.length === 1) dims.push(dims[0]);
    if (dims.length === 0) dims.push(0, 0);
    if (!dims.some(d => d === 0)) {
      throw new RuntimeError(
        `${className}.empty: at least one dimension must be zero`
      );
    }
    return RTV.classInstanceArray(className, [], [dims[0], dims[1]]);
  }
  throw new RuntimeError(`No method '${methodName}' for class '${className}'`);
}

// ── Enumerations ─────────────────────────────────────────────────────────

/** Underlying (superclass) value of an enumeration member from its declared
 *  constructor args. MATLAB's default enumeration constructor passes the
 *  member's arguments to the first superclass constructor. */
function enumUnderlyingValue(
  interp: Interpreter,
  classInfo: ClassInfo,
  argExprs: Expr[]
): RuntimeValue | undefined {
  if (argExprs.length === 0) return undefined;
  const first = ensureRuntimeValue(evalExpr(interp, argExprs[0]));
  const sc = classInfo.superClasses[0];
  if (sc && BUILTIN_VALUE_CLASSES.has(sc)) {
    return ensureRuntimeValue(interp.rt.callBuiltin(sc, 1, [first]));
  }
  return first;
}

/** Resolve `ClassName.MemberName` to an enumeration-member instance. Returns
 *  null when `className` is not an enumeration or has no such member. */
export function interpretEnumMember(
  interp: Interpreter,
  className: string,
  memberName: string
): RuntimeValue | null {
  const classInfo = interp.code.classes.get(className);
  if (!classInfo || !classInfo.enumMembers) return null;
  const memberDef = classInfo.enumMembers.find(m => m.name === memberName);
  if (!memberDef) return null;
  const value = enumUnderlyingValue(interp, classInfo, memberDef.args);
  return new RuntimeClassInstance(
    className,
    new Map(),
    false,
    value,
    memberName
  );
}

/** Enumeration constructor-as-converter: `ClassName(x)` for an enumeration
 *  with no explicit constructor. Converts x — a member, a numeric value, a
 *  char/string member name, or an array thereof — to member(s). */
function convertToEnum(
  interp: Interpreter,
  className: string,
  classInfo: ClassInfo,
  args: unknown[]
): RuntimeValue {
  if (args.length === 0) {
    throw new RuntimeError(
      `${className}: enumeration requires an argument to construct`
    );
  }
  const val = ensureRuntimeValue(args[0]);

  // Already an instance / array of this enumeration → identity (value classes
  // are immutable, so no copy is needed).
  if (isRuntimeClassInstance(val) && val._enumMember !== undefined) return val;
  if (isRuntimeClassInstanceArray(val)) return val;

  const members = classInfo.enumMembers!.map(m => ({
    name: m.name,
    value: enumUnderlyingValue(interp, classInfo, m.args),
  }));
  const makeMember = (name: string, value: RuntimeValue | undefined) =>
    new RuntimeClassInstance(className, new Map(), false, value, name);
  const fromName = (text: string): RuntimeClassInstance => {
    const m = members.find(mm => mm.name === text);
    if (!m)
      throw new RuntimeError(
        `'${text}' is not a valid member of enumeration '${className}'`
      );
    return makeMember(m.name, m.value);
  };
  const fromNumber = (num: number): RuntimeClassInstance => {
    const m = members.find(
      mm => mm.value !== undefined && toNumber(mm.value) === num
    );
    if (!m)
      throw new RuntimeError(
        `${num} is not a valid underlying value for enumeration '${className}'`
      );
    return makeMember(m.name, m.value);
  };

  if (isRuntimeChar(val)) return fromName(val.value);
  if (isRuntimeString(val)) return fromName(val);
  if (isRuntimeNumber(val)) return fromNumber(val);
  if (isRuntimeLogical(val)) return fromNumber(val ? 1 : 0);
  if (isRuntimeTensor(val)) {
    const elems = Array.from(val.data, x => fromNumber(x));
    if (elems.length === 1) return elems[0];
    const shape: [number, number] = [val.shape[0] ?? 0, val.shape[1] ?? 1];
    return RTV.classInstanceArray(className, elems, shape);
  }
  throw new RuntimeError(`Cannot convert value to enumeration '${className}'`);
}

// ── Introspection ────────────────────────────────────────────────────────

/** Public (GetAccess public) property names of a class, including inherited
 *  ones, in declaration order. A Properties block whose `Access` or
 *  `GetAccess` is `private`/`protected` is excluded — matching MATLAB's
 *  `properties(obj)`, which lists only publicly-gettable properties. Returns
 *  undefined when `className` is not a known class. */
export function classPublicProperties(
  interp: Interpreter,
  className: string
): string[] | undefined {
  if (className === MEXCEPTION_CLASS) return [...MEXCEPTION_PROPERTIES];
  const self = interp.code.classes.get(className);
  if (!self) return undefined;
  const out: string[] = [];
  const seen = new Set<string>();
  for (const info of interp.code.classes.lineage(self)) {
    const ast = info.ast;
    if (ast) {
      for (const member of ast.members) {
        if (member.type !== "Properties") continue;
        const hidden = member.attributes.some(a => {
          const n = a.name.toLowerCase();
          if (n !== "access" && n !== "getaccess") return false;
          const val = (a.value ?? "").toLowerCase();
          return val === "private" || val === "protected";
        });
        if (hidden) continue;
        for (const pn of member.names) {
          if (!seen.has(pn)) {
            seen.add(pn);
            out.push(pn);
          }
        }
      }
    }
  }
  // A subclass of MException inherits its properties.
  if (interp.code.classes.ancestors(className).includes(MEXCEPTION_CLASS)) {
    for (const pn of MEXCEPTION_PROPERTIES) if (!seen.has(pn)) out.push(pn);
  }
  return out;
}

/** Methods inherited from the built-in `handle` class. */
const HANDLE_CLASS_METHODS = [
  "addlistener",
  "delete",
  "eq",
  "findobj",
  "findprop",
  "ge",
  "gt",
  "isvalid",
  "le",
  "listener",
  "lt",
  "ne",
  "notify",
];

/** Public method names of a class (instance and static methods, the
 *  constructor, @folder method files), including inherited ones. Methods in
 *  a block with private/protected Access are excluded, matching MATLAB's
 *  `ismethod`. Returns undefined when `className` is not a known class. */
export function classPublicMethods(
  interp: Interpreter,
  className: string
): Set<string> | undefined {
  const self = interp.code.classes.get(className);
  if (!self) return undefined;
  const out = new Set<string>();
  if (self.constructorName) out.add(self.constructorName);
  for (const info of interp.code.classes.lineage(self)) {
    const ast = info.ast;
    if (ast) {
      for (const member of ast.members) {
        if (member.type !== "Methods") continue;
        const hidden = member.attributes.some(a => {
          if (a.name.toLowerCase() !== "access") return false;
          const val = (a.value ?? "").toLowerCase();
          return val !== "public";
        });
        if (hidden) continue;
        const names = [
          ...member.body.flatMap(m => (m.type === "Function" ? [m.name] : [])),
          ...(member.signatures ?? []).map(sig => sig.name),
        ];
        for (const n of names) {
          // skip property accessors (get.X / set.X) and superclass ctors
          if (n.includes(".") || (n === info.name && n !== className)) continue;
          out.add(n);
        }
      }
    } else {
      for (const n of info.methodNames) out.add(n);
      for (const n of info.staticMethodNames) out.add(n);
    }
    for (const n of info.externalMethodFiles.keys())
      if (n !== info.name || info.qualifiedName === className) out.add(n);
  }
  if (interp.code.classes.isHandleClass(self))
    for (const n of HANDLE_CLASS_METHODS) out.add(n);
  return out;
}
