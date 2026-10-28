/**
 * `?ClassName` and `metaclass(obj)`: a small meta.class model. The objects
 * are handle instances of `matlab.metadata.Class`, `.Property`, `.Method`
 * and `.EnumeratedValue` (the R2024+ names; `isa(mc, 'meta.class')` holds)
 * with the commonly used fields. One object per class and run, so
 * `DefiningClass` refers back to the same object.
 */

import type { RuntimeValue } from "../runtime/types.js";
import {
  RuntimeClassInstance,
  type RuntimeClassInstanceArray,
} from "../runtime/types.js";
import { RTV } from "../runtime/constructors.js";
import { incref } from "../runtime/refcount.js";
import { allocFloat64Array } from "../runtime/alloc.js";
import type {
  ClassInfo,
  EventInfo,
  MemberAccess,
  MethodInfo,
  PropertyInfo,
} from "../workspace/classInfo.js";
import {
  HANDLE_CLASS_METHODS,
  OBJECT_BEING_DESTROYED,
} from "../workspace/classTable.js";
import {
  MEXCEPTION_CLASS,
  MEXCEPTION_PROPERTIES,
} from "../runtime/mexception.js";

import type { Interpreter } from "./interpreter.js";
import { propertyDefaultValue } from "./oop.js";

const CLASS = "matlab.metadata.Class";
const PROPERTY = "matlab.metadata.Property";
const METHOD = "matlab.metadata.Method";
const ENUM_VALUE = "matlab.metadata.EnumeratedValue";
const EVENT = "matlab.metadata.Event";

/** Ancestors of the meta classes, for `isa` (`meta.class` is an alias). */
export const META_CLASS_ALIASES: Record<string, string[]> = {
  [CLASS]: ["meta.class", "handle"],
  [PROPERTY]: ["meta.property", "handle"],
  [METHOD]: ["meta.method", "handle"],
  [ENUM_VALUE]: ["meta.EnumeratedValue", "handle"],
  [EVENT]: ["meta.event", "handle"],
};

/** Built-in classes `?Name` describes (with no members). */
const BUILTIN_CLASSES = new Set([
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
  "string",
  "cell",
  "struct",
  "function_handle",
  "handle",
  MEXCEPTION_CLASS,
]);

const EMPTY = () => RTV.tensor(allocFloat64Array(0), [0, 0]);

/** A column list of meta objects: a 0x1 array, the object itself, or an
 *  n×1 array. */
function objectList(
  className: string,
  items: RuntimeValue[]
): RuntimeClassInstance | RuntimeClassInstanceArray {
  const elems = items as RuntimeClassInstance[];
  if (elems.length === 1) return elems[0];
  return RTV.classInstanceArray(className, elems, [elems.length, 1]);
}

function metaObject(
  interp: Interpreter,
  className: string,
  fields: [string, RuntimeValue][],
  obj = new RuntimeClassInstance(className, new Map(), true)
): RuntimeClassInstance {
  for (const [name, value] of fields) obj.bindField(interp.rt, name, value);
  return obj;
}

/** The meta.class object for `name`: a class definition, a built-in class,
 *  or (for an unknown name, as MATLAB) an empty 0x0 Class array. */
export function metaClassFor(interp: Interpreter, name: string): RuntimeValue {
  const cached = interp.metaClasses.get(name);
  if (cached !== undefined) return cached;
  const info = interp.code.classes.get(name);
  if (!info && !BUILTIN_CLASSES.has(name))
    return RTV.classInstanceArray(CLASS, [], [0, 0]);
  // Cache the (still empty) object first: member lists refer back to it.
  const obj = new RuntimeClassInstance(CLASS, new Map(), true);
  incref(obj);
  interp.metaClasses.set(name, obj);
  if (info) fillClass(interp, obj, info);
  else fillBuiltinClass(interp, obj, name);
  return obj;
}

function accessValue(interp: Interpreter, access: MemberAccess): RuntimeValue {
  if (typeof access === "string") return RTV.char(access);
  const classes = access.classes.map(c => metaClassFor(interp, c));
  return RTV.cell(classes, [1, classes.length]);
}

function classFlags(
  name: string,
  flags: { hidden: boolean; sealed: boolean; abstract: boolean },
  enumeration: boolean,
  handle: boolean
): [string, RuntimeValue][] {
  return [
    ["Name", RTV.char(name)],
    ["Description", RTV.char("")],
    ["DetailedDescription", RTV.char("")],
    ["Hidden", RTV.logical(flags.hidden)],
    ["Sealed", RTV.logical(flags.sealed)],
    ["Abstract", RTV.logical(flags.abstract)],
    ["Enumeration", RTV.logical(enumeration)],
    ["ConstructOnLoad", RTV.logical(false)],
    ["HandleCompatible", RTV.logical(handle)],
  ];
}

function fillBuiltinClass(
  interp: Interpreter,
  obj: RuntimeClassInstance,
  name: string
): void {
  const handle = name === "handle";
  const methods = handle
    ? HANDLE_CLASS_METHODS.map(m => handleMethod(interp, m, obj))
    : [];
  metaObject(
    interp,
    CLASS,
    [
      ...classFlags(
        name,
        { hidden: false, sealed: false, abstract: handle },
        false,
        handle
      ),
      ["InferiorClasses", objectList(CLASS, [])],
      ["RestrictsSubclassing", RTV.logical(false)],
      [
        "PropertyList",
        objectList(
          PROPERTY,
          name === MEXCEPTION_CLASS ? mexceptionProperties(interp, obj) : []
        ),
      ],
      ["MethodList", objectList(METHOD, methods)],
      [
        "EventList",
        objectList(
          EVENT,
          handle ? [eventMeta(interp, OBJECT_BEING_DESTROYED, obj)] : []
        ),
      ],
      ["EnumerationMemberList", objectList(ENUM_VALUE, [])],
      ["SuperclassList", objectList(CLASS, [])],
    ],
    obj
  );
}

/** MException's public properties (it has no classdef). */
function mexceptionProperties(
  interp: Interpreter,
  definingClass: RuntimeValue
): RuntimeClassInstance[] {
  return MEXCEPTION_PROPERTIES.map(name =>
    metaObject(interp, PROPERTY, [
      ["Name", RTV.char(name)],
      ["Description", RTV.char("")],
      ["DetailedDescription", RTV.char("")],
      ["GetAccess", RTV.char("public")],
      // As MATLAB reports them.
      ["SetAccess", RTV.char(name === "message" ? "protected" : "private")],
      ["Dependent", RTV.logical(false)],
      ["Constant", RTV.logical(false)],
      ["Abstract", RTV.logical(false)],
      ["Transient", RTV.logical(false)],
      ["Hidden", RTV.logical(false)],
      ["GetObservable", RTV.logical(false)],
      ["SetObservable", RTV.logical(false)],
      ["AbortSet", RTV.logical(false)],
      ["NonCopyable", RTV.logical(false)],
      ["HasDefault", RTV.logical(false)],
      ["GetMethod", EMPTY()],
      ["SetMethod", EMPTY()],
      ["DefiningClass", definingClass],
    ])
  );
}

function fillClass(
  interp: Interpreter,
  obj: RuntimeClassInstance,
  info: ClassInfo
): void {
  const table = interp.code.classes;
  const name = info.qualifiedName;
  const handle = table.isHandleClass(info);
  // A superclass's private properties (neither readable nor settable
  // outside it) are not listed.
  const restricted = (a: MemberAccess) =>
    a === "private" || a === "immutable" || a === "none";
  const properties = table
    .allProperties(name)
    .filter(
      p =>
        p.definingClass === name ||
        !restricted(p.getAccess) ||
        !restricted(p.setAccess)
    )
    .map(p => metaPropertyFor(interp, p));
  // Likewise a superclass's events that only it may listen to and notify.
  const events = table
    .allEvents(name)
    .filter(
      e =>
        e.definingClass === name ||
        !restricted(e.listenAccess) ||
        !restricted(e.notifyAccess)
    )
    .map(e =>
      eventMeta(
        interp,
        e,
        e.definingClass === name ? obj : metaClassFor(interp, e.definingClass)
      )
    );
  const enumMembers = (info.enumMembers ?? []).map(m =>
    metaObject(interp, ENUM_VALUE, [
      ["Name", RTV.char(m.name)],
      ["Description", RTV.char("")],
      ["DetailedDescription", RTV.char("")],
      ["Hidden", RTV.logical(false)],
      ["DefiningClass", obj],
    ])
  );
  const abstract =
    info.abstract || table.unimplementedAbstractMembers(name).length > 0;
  metaObject(
    interp,
    CLASS,
    [
      ...classFlags(
        name,
        { ...info, abstract },
        info.enumMembers !== null,
        handle
      ),
      [
        "InferiorClasses",
        objectList(
          CLASS,
          info.inferiorClasses.map(c => metaClassFor(interp, c))
        ),
      ],
      ["RestrictsSubclassing", RTV.logical(false)],
      ["PropertyList", objectList(PROPERTY, properties)],
      ["MethodList", objectList(METHOD, methodMetas(interp, info, obj))],
      ["EventList", objectList(EVENT, events)],
      ["EnumerationMemberList", objectList(ENUM_VALUE, enumMembers)],
      [
        "SuperclassList",
        objectList(
          CLASS,
          info.superClasses.map(c => metaClassFor(interp, c))
        ),
      ],
    ],
    obj
  );
}

/** The meta.property object describing a property, one per property and
 *  run (as `PropertyList` and a property event's `Source` hold it). */
export function metaPropertyFor(
  interp: Interpreter,
  p: PropertyInfo
): RuntimeClassInstance {
  let cached = interp.metaProperties.get(p);
  if (cached) return cached;
  // Describing the defining class describes its properties.
  metaClassFor(interp, p.definingClass);
  cached = interp.metaProperties.get(p);
  if (cached) return cached;
  const obj = propertyMeta(interp, p);
  incref(obj);
  interp.metaProperties.set(p, obj);
  return obj;
}

function propertyMeta(
  interp: Interpreter,
  p: PropertyInfo
): RuntimeClassInstance {
  const value = propertyDefaultValue(interp, p);
  return metaObject(interp, PROPERTY, [
    ["Name", RTV.char(p.name)],
    ["Description", RTV.char("")],
    ["DetailedDescription", RTV.char("")],
    ["GetAccess", accessValue(interp, p.getAccess)],
    ["SetAccess", accessValue(interp, p.setAccess)],
    ["Dependent", RTV.logical(p.dependent)],
    ["Constant", RTV.logical(p.constant)],
    ["Abstract", RTV.logical(p.abstract)],
    ["Transient", RTV.logical(p.transient)],
    ["Hidden", RTV.logical(p.hidden)],
    ["GetObservable", RTV.logical(p.getObservable)],
    ["SetObservable", RTV.logical(p.setObservable)],
    ["AbortSet", RTV.logical(false)],
    ["NonCopyable", RTV.logical(false)],
    ["HasDefault", RTV.logical(value !== undefined)],
    ...(value !== undefined
      ? ([["DefaultValue", value]] as [string, RuntimeValue][])
      : []),
    ["GetMethod", EMPTY()],
    ["SetMethod", EMPTY()],
    ["DefiningClass", metaClassFor(interp, p.definingClass)],
  ]);
}

function eventMeta(
  interp: Interpreter,
  e: EventInfo,
  definingClass: RuntimeValue
): RuntimeClassInstance {
  return metaObject(interp, EVENT, [
    ["Name", RTV.char(e.name)],
    ["Description", RTV.char("")],
    ["DetailedDescription", RTV.char("")],
    ["Hidden", RTV.logical(e.hidden)],
    ["NotifyAccess", accessValue(interp, e.notifyAccess)],
    ["ListenAccess", accessValue(interp, e.listenAccess)],
    ["DefiningClass", definingClass],
  ]);
}

function methodMeta(
  interp: Interpreter,
  m: MethodInfo,
  definingClass: RuntimeValue
): RuntimeClassInstance {
  const names = (list: string[]) =>
    RTV.cell(
      list.map(n => RTV.char(n)),
      [list.length, 1]
    );
  return metaObject(interp, METHOD, [
    ["Name", RTV.char(m.name)],
    ["Description", RTV.char("")],
    ["DetailedDescription", RTV.char("")],
    ["Access", accessValue(interp, m.access)],
    ["Static", RTV.logical(m.static)],
    ["Abstract", RTV.logical(m.abstract)],
    ["Sealed", RTV.logical(m.sealed)],
    ["Hidden", RTV.logical(m.hidden)],
    ["InputNames", names(m.inputs)],
    ["OutputNames", names(m.outputs)],
    ["DefiningClass", definingClass],
  ]);
}

function publicMethod(
  name: string,
  definingClass: string,
  extra: Partial<MethodInfo> = {}
): MethodInfo {
  return {
    name,
    definingClass,
    access: "public",
    static: false,
    hidden: false,
    abstract: false,
    sealed: false,
    inputs: [],
    outputs: [],
    ...extra,
  };
}

function handleMethod(
  interp: Interpreter,
  name: string,
  handleMeta: RuntimeValue
): RuntimeClassInstance {
  return methodMeta(
    interp,
    publicMethod(name, "handle", { inputs: ["obj"] }),
    handleMeta
  );
}

/** MethodList: the class's own methods (with the constructor, static and
 *  abstract methods), inherited methods that are not private (without
 *  superclass constructors), the implicit static `empty`, and for a handle
 *  class the methods of `handle`. */
function methodMetas(
  interp: Interpreter,
  info: ClassInfo,
  self: RuntimeValue
): RuntimeClassInstance[] {
  const table = interp.code.classes;
  const out: RuntimeClassInstance[] = [];
  const seen = new Set<string>();
  const add = (m: MethodInfo) => {
    if (seen.has(m.name)) return;
    seen.add(m.name);
    out.push(
      methodMeta(
        interp,
        m,
        m.definingClass === info.qualifiedName
          ? self
          : metaClassFor(interp, m.definingClass)
      )
    );
  };
  // A class without an explicit constructor has an implicit one.
  if (!info.constructorName)
    add(publicMethod(info.name, info.qualifiedName, { outputs: ["obj"] }));
  for (const c of table.lineage(info)) {
    const own = c === info;
    for (const m of c.methods.values()) {
      if (!own && (m.access === "private" || m.name === c.constructorName))
        continue;
      add(m);
    }
    for (const n of c.externalMethodFiles.keys()) {
      if (!own && n === c.name) continue;
      add(publicMethod(n, c.qualifiedName));
    }
  }
  add(
    publicMethod("empty", info.qualifiedName, { static: true, hidden: true })
  );
  if (table.isHandleClass(info)) {
    const handleMeta = metaClassFor(interp, "handle");
    for (const n of HANDLE_CLASS_METHODS)
      if (!seen.has(n)) out.push(handleMethod(interp, n, handleMeta));
  }
  return out;
}
