/**
 * Class information types and extraction utilities.
 */

import {
  type Attr,
  type Expr,
  type Stmt,
  type ValidationSpec,
} from "../parser/index.js";

/** Built-in base classes that confer handle (reference) semantics. */
export const HANDLE_BASE_CLASSES = new Set([
  "handle",
  "dynamicprops",
  "matlab.mixin.Copyable",
  "matlab.mixin.SetGet",
  "matlab.mixin.SetGetExactNames",
  "hgsetget",
]);

/** Who may use a class member. A class list (`Access = {?A, ?B}`) admits
 *  the listed classes and their subclasses, besides the defining class.
 *  `immutable` (SetAccess only) admits the defining class's constructor;
 *  `none` (SetAccess only, e.g. of a Constant property) admits no one. */
export type MemberAccess =
  | "public"
  | "protected"
  | "private"
  | "immutable"
  | "none"
  | { classes: string[] };

/** A property declared in a classdef, with its block's attributes. */
export interface PropertyInfo {
  name: string;
  /** Qualified name of the class that declares the property. */
  definingClass: string;
  defaultExpr: Expr | null;
  getAccess: MemberAccess;
  setAccess: MemberAccess;
  constant: boolean;
  dependent: boolean;
  hidden: boolean;
  abstract: boolean;
  /** Parsed only (numbl does not serialize objects). */
  transient: boolean;
  /** PreSet/PostSet (SetObservable) and PreGet/PostGet (GetObservable)
   *  listeners may be attached to the property. */
  setObservable: boolean;
  getObservable: boolean;
}

/** An event declared in a classdef `events` block. */
export interface EventInfo {
  name: string;
  /** Qualified name of the class that declares the event. */
  definingClass: string;
  /** Who may add listeners (`addlistener`, `listener`). */
  listenAccess: MemberAccess;
  /** Who may trigger the event (`notify`). */
  notifyAccess: MemberAccess;
  hidden: boolean;
}

/** A method declared in a classdef (with a body or as a signature), with its
 *  block's attributes. Property accessors (`get.X`/`set.X`) are not listed. */
export interface MethodInfo {
  name: string;
  /** Qualified name of the class that declares the method. */
  definingClass: string;
  access: MemberAccess;
  static: boolean;
  hidden: boolean;
  abstract: boolean;
  sealed: boolean;
  inputs: string[];
  outputs: string[];
}

/** Metadata about a class definition. */
export interface ClassInfo {
  name: string; // e.g. "Rectangle_"
  qualifiedName: string; // e.g. "Rectangle_" or "geometry.Circle"
  fileName: string; // e.g. "Rectangle_.m"
  source: string; // raw .m file source
  /** Direct superclasses in declaration order (`classdef A < B & C`). */
  superClasses: string[];
  /** Properties stored on each instance (not Dependent, Constant or
   *  Abstract), in declaration order. */
  propertyNames: string[];
  propertyDefaults: Map<string, Expr>; // property name → default AST expr
  /** Every property the class itself declares, in declaration order. */
  properties: Map<string, PropertyInfo>;
  /** Every method the class itself declares (including the constructor and
   *  static methods), in declaration order. Methods that only exist as
   *  `@folder` files, without a signature in the classdef, are not listed;
   *  they are public. */
  methods: Map<string, MethodInfo>;
  /** Every event the class itself declares, in declaration order. */
  events: Map<string, EventInfo>;
  /** Size/class/validator declarations of properties that have them. */
  propertyValidations: Map<string, ValidationSpec>;
  methodNames: Set<string>; // instance method names (non-static, non-constructor)
  staticMethodNames: Set<string>;
  constructorName: string | null; // method name matching class name
  /** Class attributes. */
  abstract: boolean;
  sealed: boolean;
  hidden: boolean;
  /** Enumeration members in declaration order (name + constructor-arg
   *  exprs), or null when the class is not an enumeration. */
  enumMembers: { name: string; args: Expr[] }[] | null;
  ast: (Stmt & { type: "ClassDef" }) | null; // the parsed ClassDef AST (null for old-style @folder classes)
  isOldStyle?: boolean; // true for pre-classdef @folder classes (constructor is a plain function calling class(s,'name'))
  inferiorClasses: string[]; // classes declared inferior via InferiorClasses attribute
  externalMethodFiles: Map<string, { fileName: string; source: string }>; // method name → file info (for @folder methods)
}

// ── Attributes ──────────────────────────────────────────────────────────

/** An attribute's value with any quotes removed (`'private'` → private). */
function attrText(value: string): string {
  const v = value.trim();
  if (
    v.length >= 2 &&
    (v[0] === "'" || v[0] === '"') &&
    v[v.length - 1] === v[0]
  )
    return v.slice(1, -1);
  return v;
}

/** A logical attribute: bare (`Static`) is true, `~Static` or `= false` is
 *  false. */
function attrFlag(value: string | null): boolean {
  if (value === null) return true;
  const v = attrText(value).toLowerCase();
  return v !== "false" && v !== "0";
}

/** An access attribute's value: a level, or a class list `{?A, ?B}`. An
 *  empty list (`{}`) admits only the defining class, as `private` does. */
function attrAccess(value: string | null): MemberAccess {
  if (value === null) return "public";
  const v = attrText(value);
  if (v.startsWith("{")) {
    const classes = [...v.matchAll(/\?([\w.]+)/g)].map(m => m[1]);
    return classes.length > 0 ? { classes } : "private";
  }
  switch (v.toLowerCase()) {
    case "private":
      return "private";
    case "protected":
      return "protected";
    case "immutable":
      return "immutable";
    case "none":
      return "none";
    default:
      return "public";
  }
}

interface BlockAttributes {
  getAccess: MemberAccess;
  setAccess: MemberAccess;
  /** Logical attributes that are true, lower-cased. */
  flags: Set<string>;
}

/** The attributes of a properties or methods block. `Access` sets both get
 *  and set access (a method's access is `getAccess`); a later `GetAccess` or
 *  `SetAccess` overrides one of them. */
function blockAttributes(attrs: Attr[]): BlockAttributes {
  let getAccess: MemberAccess = "public";
  let setAccess: MemberAccess = "public";
  const flags = new Set<string>();
  for (const a of attrs) {
    const n = a.name.toLowerCase();
    if (n === "access") getAccess = setAccess = attrAccess(a.value);
    else if (n === "getaccess") getAccess = attrAccess(a.value);
    else if (n === "setaccess") setAccess = attrAccess(a.value);
    else if (attrFlag(a.value)) flags.add(n);
  }
  return { getAccess, setAccess, flags };
}

/**
 * Extract ClassInfo from a parsed ClassDef AST node.
 */
export function extractClassInfo(
  classDef: Stmt & { type: "ClassDef" },
  qualifiedName: string,
  fileName: string,
  source: string
): ClassInfo {
  const propertyNames: string[] = [];
  const propertyDefaults = new Map<string, Expr>();
  const properties = new Map<string, PropertyInfo>();
  const methods = new Map<string, MethodInfo>();
  const events = new Map<string, EventInfo>();
  const propertyValidations = new Map<string, ValidationSpec>();
  const methodNames = new Set<string>();
  const staticMethodNames = new Set<string>();
  let constructorName: string | null = null;
  let enumMembers: { name: string; args: Expr[] }[] | null = null;

  // The base class name (last segment for qualified names like "pkg.ClassName")
  const dotIdx = qualifiedName.lastIndexOf(".");
  const baseName =
    dotIdx >= 0 ? qualifiedName.slice(dotIdx + 1) : qualifiedName;

  for (const member of classDef.members) {
    if (member.type === "Properties") {
      const { getAccess, setAccess, flags } = blockAttributes(
        member.attributes
      );
      const constant = flags.has("constant");
      const dependent = flags.has("dependent");
      const abstract = flags.has("abstract");
      for (let i = 0; i < member.names.length; i++) {
        const name = member.names[i];
        const defaultExpr = member.defaultValues[i] ?? null;
        properties.set(name, {
          name,
          definingClass: qualifiedName,
          defaultExpr,
          getAccess,
          setAccess: constant ? "none" : setAccess,
          constant,
          dependent,
          hidden: flags.has("hidden"),
          abstract,
          transient: flags.has("transient"),
          setObservable: flags.has("setobservable"),
          getObservable: flags.has("getobservable"),
        });
        if (constant || dependent || abstract) continue;
        propertyNames.push(name);
        if (defaultExpr) propertyDefaults.set(name, defaultExpr);
        const validation = member.validations[i];
        if (validation) propertyValidations.set(name, validation);
      }
    } else if (member.type === "Methods") {
      const { getAccess, flags } = blockAttributes(member.attributes);
      const isStatic = flags.has("static");
      const addMethod = (name: string, inputs: string[], outputs: string[]) => {
        if (isStatic) {
          staticMethodNames.add(name);
        } else if (name === classDef.name || name === baseName) {
          constructorName = name;
        } else {
          methodNames.add(name);
        }
        // Property accessors are not methods in their own right.
        if (name.includes(".")) return;
        methods.set(name, {
          name,
          definingClass: qualifiedName,
          access: getAccess,
          static: isStatic,
          hidden: flags.has("hidden"),
          abstract: flags.has("abstract"),
          sealed: flags.has("sealed"),
          inputs,
          outputs,
        });
      };
      for (const methodStmt of member.body) {
        if (methodStmt.type === "Function")
          addMethod(methodStmt.name, methodStmt.params, methodStmt.outputs);
      }
      // Also process method signatures (prototype declarations without
      // function keyword, e.g. `result = myMethod(x);`). These are used
      // for external method files declared in the classdef, and for
      // abstract methods.
      for (const sig of member.signatures ?? [])
        addMethod(sig.name, sig.params, sig.outputs);
    } else if (member.type === "Events") {
      let listenAccess: MemberAccess = "public";
      let notifyAccess: MemberAccess = "public";
      let hidden = false;
      for (const a of member.attributes) {
        const n = a.name.toLowerCase();
        if (n === "listenaccess") listenAccess = attrAccess(a.value);
        else if (n === "notifyaccess") notifyAccess = attrAccess(a.value);
        else if (n === "hidden") hidden = attrFlag(a.value);
      }
      for (const name of member.names)
        events.set(name, {
          name,
          definingClass: qualifiedName,
          listenAccess,
          notifyAccess,
          hidden,
        });
    } else if (member.type === "Enumeration") {
      enumMembers = member.members.map(m => ({ name: m.name, args: m.args }));
    }
  }

  const classFlags = blockAttributes(classDef.classAttributes).flags;
  // Extract InferiorClasses from class attributes
  const inferiorClasses: string[] = [];
  for (const attr of classDef.classAttributes) {
    if (attr.name.toLowerCase() === "inferiorclasses" && attr.value) {
      // Parse class names from {?Class1, ?Class2, ...}
      const matches = attr.value.matchAll(/\?(\w+)/g);
      for (const m of matches) {
        inferiorClasses.push(m[1]);
      }
    }
  }

  return {
    name: classDef.name,
    qualifiedName,
    fileName,
    source,
    superClasses: classDef.superClasses,
    propertyNames,
    propertyDefaults,
    properties,
    methods,
    events,
    propertyValidations,
    methodNames,
    staticMethodNames,
    constructorName,
    abstract: classFlags.has("abstract"),
    sealed: classFlags.has("sealed"),
    hidden: classFlags.has("hidden"),
    enumMembers,
    inferiorClasses,
    ast: classDef,
    externalMethodFiles: new Map(),
  };
}

/**
 * Build ClassInfo for an old-style (pre-classdef) `@ClassName` folder class.
 * There is no classdef AST: the constructor is the `@Name/Name.m` plain
 * function (which calls `obj = class(struct, 'Name')`), and every other `.m`
 * in the folder is an external method dispatched on instances. Fields are
 * dynamic (defined by the constructor's struct), so there are no declared
 * property names.
 */
export function makeOldStyleClassInfo(
  qualifiedName: string,
  baseName: string,
  constructorFile: { fileName: string; source: string },
  methodFiles: { name: string; fileName: string; source: string }[]
): ClassInfo {
  const externalMethodFiles = new Map<
    string,
    { fileName: string; source: string }
  >();
  externalMethodFiles.set(baseName, {
    fileName: constructorFile.fileName,
    source: constructorFile.source,
  });
  const methodNames = new Set<string>();
  for (const mf of methodFiles) {
    externalMethodFiles.set(mf.name, {
      fileName: mf.fileName,
      source: mf.source,
    });
    methodNames.add(mf.name);
  }
  return {
    name: baseName,
    qualifiedName,
    fileName: constructorFile.fileName,
    source: constructorFile.source,
    superClasses: [],
    propertyNames: [],
    propertyDefaults: new Map(),
    properties: new Map(),
    methods: new Map(),
    events: new Map(),
    propertyValidations: new Map(),
    methodNames,
    staticMethodNames: new Set(),
    constructorName: baseName,
    abstract: false,
    sealed: false,
    hidden: false,
    enumMembers: null,
    inferiorClasses: [],
    ast: null,
    isOldStyle: true,
    externalMethodFiles,
  };
}
