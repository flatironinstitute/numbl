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
import { applyPropertyDefaults } from "./propertyValidation.js";
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
import type { ClassInfo, PropertyInfo } from "../workspace/classInfo.js";
import {
  MEXCEPTION_CLASS,
  MEXCEPTION_PROPERTIES,
} from "../runtime/mexception.js";
import { methodAccessError, propertyGetError } from "../runtime/classAccess.js";
import { incref } from "../runtime/refcount.js";
import { HANDLE_CLASS_METHODS } from "../workspace/classTable.js";

import type { Interpreter } from "./interpreter.js";
import { Environment } from "./types.js";
import { CallFrame } from "./callFrame.js";
import { evalArgs, evalExpr } from "./evalExpressions.js";
import { callBody } from "./callConvention.js";
import { createListeners, destroyHandle } from "./events.js";

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

  // `event.listener(src, ...)`: the listener classes' constructor is
  // numbl's.
  if (args.length > 0 && LISTENER_CLASSES.has(className))
    return createListeners(
      interp,
      args.map(a => ensureRuntimeValue(a)),
      false,
      className
    );

  checkInstantiable(interp, className, classInfo);
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

const LISTENER_CLASSES: ReadonlySet<string> = new Set([
  "event.listener",
  "event.proplistener",
]);

/** Classes already checked by `checkClassDefinition`. */
const checkedClasses = new WeakSet<ClassInfo>();

/** Raise MATLAB's error when the running code cannot construct an instance
 *  of `className`: the class is abstract, its constructor is not accessible
 *  from here, or its definition is invalid (`checkClassDefinition`). */
function checkInstantiable(
  interp: Interpreter,
  className: string,
  info: ClassInfo
): void {
  if (!checkedClasses.has(info)) {
    checkClassDefinition(interp, info);
    checkedClasses.add(info);
  }
  const table = interp.code.classes;
  if (info.abstract || table.unimplementedAbstractMembers(className).length) {
    const declares =
      info.abstract ||
      [...info.properties.values(), ...info.methods.values()].some(
        m => m.abstract
      );
    throw new RuntimeError(
      declares
        ? `Abstract classes cannot be instantiated. Class '${className}' defines abstract methods and/or properties.`
        : `Abstract classes cannot be instantiated. Class '${className}' inherits abstract methods or properties but does not implement them. Use matlab.metadata.abstractDetails to see the list of methods and properties that '${className}' must implement if you do not intend the class to be abstract.`,
      null,
      { identifier: "MATLAB:class:abstract" }
    );
  }
  const ctor =
    info.constructorName && table.findMethod(className, info.constructorName);
  if (ctor) {
    const denied = methodAccessError(table, interp.frame, className, ctor);
    if (denied) throw denied;
  }
}

/** Raise MATLAB's error when a class subclasses a Sealed class or overrides
 *  a Sealed method. */
function checkClassDefinition(interp: Interpreter, info: ClassInfo): void {
  const lineage = interp.code.classes.lineage(info);
  for (let i = 1; i < lineage.length; i++) {
    const parent = lineage[i];
    if (parent.sealed)
      throw new RuntimeError(
        `Class '${parent.qualifiedName}' is Sealed and may not be used as a superclass.`,
        null,
        { identifier: "MATLAB:class:sealed" }
      );
    for (const m of parent.methods.values()) {
      if (!m.sealed) continue;
      const sub = lineage.slice(0, i).find(c => c.methods.has(m.name));
      if (sub)
        throw new RuntimeError(
          `Method '${m.name}' in class '${sub.qualifiedName}' conflicts with the sealed method in superclass '${parent.qualifiedName}'. Overriding a sealed method is not supported.`,
          null,
          { identifier: "MATLAB:class:methodOverrideSealed" }
        );
    }
  }
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

/** A class's instance property names, including inherited ones, and their
 *  evaluated default values. `className` names the class being built, for
 *  errors. A default is evaluated in the context of the class declaring it. */
function propertyDefaults(
  interp: Interpreter,
  className: string,
  classInfo: ClassInfo
): { propertyNames: string[]; defaults: Map<string, RuntimeValue> } {
  const propertyNames: string[] = [];
  const defaultExprs = new Map<string, { expr: Expr; info: ClassInfo }>();
  for (const info of interp.code.classes.lineage(classInfo)) {
    for (const propName of info.propertyNames) {
      if (!propertyNames.includes(propName)) propertyNames.push(propName);
      const expr = info.propertyDefaults.get(propName);
      if (expr && !defaultExprs.has(propName))
        defaultExprs.set(propName, { expr, info });
    }
  }
  const defaults = new Map<string, RuntimeValue>();
  for (const [propName, { expr, info }] of defaultExprs) {
    defaults.set(
      propName,
      evalPropertyDefault(interp, className, info, propName, expr)
    );
  }
  applyPropertyDefaults(interp, classInfo, className, defaults);
  return { propertyNames, defaults };
}

/** A frame running code of class `info` outside any method (property
 *  defaults), with an empty workspace of its own. */
function classContextFrame(interp: Interpreter, info: ClassInfo): CallFrame {
  const env = new Environment();
  env.rt = interp.rt;
  return new CallFrame(
    interp.code,
    info.fileName,
    info.qualifiedName,
    undefined,
    env,
    interp.frame,
    undefined
  );
}

/** Evaluate a property's default value expression in the context of the
 *  class `info` that declares it. MATLAB reports an invalid default as an
 *  error rather than leaving the property unset. */
function evalPropertyDefault(
  interp: Interpreter,
  className: string,
  info: ClassInfo,
  propName: string,
  defaultExpr: Expr
): RuntimeValue {
  try {
    return ensureRuntimeValue(
      interp.inFrame(classContextFrame(interp, info), () =>
        evalExpr(interp, defaultExpr)
      )
    );
  } catch (e) {
    if (e instanceof CancellationError) throw e;
    const msg = e instanceof Error ? e.message : String(e);
    throw new RuntimeError(
      `Invalid default value for property '${propName}' in class '${className}':\n${msg}`
    );
  }
}

/** `ClassName.Prop` for a Constant property, checked against the running
 *  code's access. */
export function classConstantValue(
  interp: Interpreter,
  className: string,
  propName: string
): RuntimeValue {
  const table = interp.code.classes;
  const prop = table.findProperty(className, propName)!;
  const denied = propertyGetError(table, interp.frame, className, prop);
  if (denied) throw denied;
  return constantPropertyValue(interp, prop);
}

/** A property's default value (a Constant property's value), or undefined
 *  when it declares none. */
export function propertyDefaultValue(
  interp: Interpreter,
  prop: PropertyInfo
): RuntimeValue | undefined {
  if (prop.constant) return constantPropertyValue(interp, prop);
  const info = interp.code.classes.get(prop.definingClass);
  if (!prop.defaultExpr || !info) return undefined;
  return evalPropertyDefault(
    interp,
    prop.definingClass,
    info,
    prop.name,
    prop.defaultExpr
  );
}

/** The value of a Constant property: its default, evaluated on first use
 *  and then shared by every access (`Class.Prop`, `obj.Prop`). */
export function constantPropertyValue(
  interp: Interpreter,
  prop: PropertyInfo
): RuntimeValue {
  const key = `${prop.definingClass}\0${prop.name}`;
  const cached = interp.constantValues.get(key);
  if (cached !== undefined) return cached;
  if (interp.constantsInProgress.has(key))
    throw new RuntimeError(
      `Circular definition of the Constant property '${prop.name}' in class '${prop.definingClass}'.`
    );
  const info = interp.code.classes.get(prop.definingClass)!;
  interp.constantsInProgress.add(key);
  let value: RuntimeValue;
  try {
    value = prop.defaultExpr
      ? evalPropertyDefault(
          interp,
          prop.definingClass,
          info,
          prop.name,
          prop.defaultExpr
        )
      : RTV.tensor(allocFloat64Array(0), [0, 0]);
  } finally {
    interp.constantsInProgress.delete(key);
  }
  // The cache is a slot holding the value for the rest of the run.
  incref(value);
  interp.constantValues.set(key, value);
  return value;
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

  const method = interp.code.classes.findMethod(className, methodName);
  if (method) {
    const denied = methodAccessError(
      interp.code.classes,
      interp.frame,
      className,
      method
    );
    if (denied) throw denied;
  }

  if (methodName === classInfo.constructorName) {
    return interpretConstructor(interp, classInfo, args, nargout);
  }

  const body = functionBodyFor(interp.code, target);
  if (body) {
    const actualArgs =
      target.stripInstance && args.length > 0 ? args.slice(1) : args;
    // A handle class's delete method: the object is destroyed around it.
    const self = actualArgs[0];
    if (
      methodName === "delete" &&
      self instanceof RuntimeClassInstance &&
      self.isHandleClass
    )
      return destroyHandle(interp, self, () =>
        callBody(interp, body, actualArgs, nargout)
      );
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

/** The properties `properties` and `fieldnames` list: publicly readable,
 *  not Hidden, own before inherited (Constant and Dependent included). A
 *  subclass of MException also has MException's properties. Undefined when
 *  `className` is not a known class. */
export function classPublicProperties(
  interp: Interpreter,
  className: string
): string[] | undefined {
  if (className === MEXCEPTION_CLASS) return [...MEXCEPTION_PROPERTIES];
  const classes = interp.code.classes;
  if (!classes.get(className)) return undefined;
  const out = classes.publicProperties(className).map(p => p.name);
  if (classes.ancestors(className).includes(MEXCEPTION_CLASS))
    for (const pn of MEXCEPTION_PROPERTIES) if (!out.includes(pn)) out.push(pn);
  return out;
}

/** Methods of the built-in classes that have no classdef: those of MATLAB's
 *  methods that numbl implements. */
const BUILTIN_CLASS_METHODS = new Map<string, readonly string[]>([
  [
    MEXCEPTION_CLASS,
    [
      "MException",
      "addCause",
      "addCorrection",
      "getReport",
      "isequal",
      "rethrow",
      "throw",
      "throwAsCaller",
    ],
  ],
  [
    "graph",
    [
      "addedge",
      "conncomp",
      "degree",
      "graph",
      "laplacian",
      "numedges",
      "numnodes",
    ],
  ],
  [
    "dictionary",
    [
      "dictionary",
      "entries",
      "insert",
      "isConfigured",
      "isKey",
      "keys",
      "length",
      "lookup",
      "ndims",
      "numEntries",
      "numel",
      "remove",
      "size",
      "types",
      "values",
    ],
  ],
]);

/** The methods `methods` and `ismethod` report (see
 *  `ClassTable.publicMethodNames`), including those of the built-in classes
 *  above; a subclass of MException also has MException's methods. Undefined
 *  for an unknown class. */
export function classPublicMethods(
  interp: Interpreter,
  className: string
): Set<string> | undefined {
  const classes = interp.code.classes;
  if (className === "handle") return new Set(HANDLE_CLASS_METHODS);
  const out = classes.publicMethodNames(className);
  if (!out) {
    const builtinMethods = BUILTIN_CLASS_METHODS.get(className);
    return builtinMethods && new Set(builtinMethods);
  }
  if (classes.ancestors(className).includes(MEXCEPTION_CLASS))
    for (const m of BUILTIN_CLASS_METHODS.get(MEXCEPTION_CLASS) ?? [])
      if (m !== MEXCEPTION_CLASS) out.add(m);
  return out;
}

/** The public static methods of a classdef class, including inherited
 *  ones (a subset of `classPublicMethods`). */
export function classStaticMethods(
  interp: Interpreter,
  className: string
): Set<string> {
  const classes = interp.code.classes;
  const out = new Set<string>();
  for (const n of classes.publicMethodNames(className) ?? [])
    if (classes.findMethod(className, n)?.static) out.add(n);
  return out;
}

/** The methods of `handle` a handle class inherits without redefining them
 *  (empty for other classes). `methods(obj)` lists them, but its display
 *  leaves them out and points to `methods('handle')`, as MATLAB does. */
export function inheritedHandleMethods(
  interp: Interpreter,
  className: string
): Set<string> {
  const classes = interp.code.classes;
  const info = classes.get(className);
  const out = new Set<string>();
  if (!info || !classes.isHandleClass(info)) return out;
  const lineage = classes.lineage(info);
  for (const n of HANDLE_CLASS_METHODS)
    if (
      !lineage.some(
        c =>
          c.methodNames.has(n) ||
          c.externalMethodFiles.has(n) ||
          c.staticMethodNames.has(n)
      )
    )
      out.add(n);
  return out;
}
