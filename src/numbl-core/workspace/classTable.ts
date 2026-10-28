/**
 * The classes known to a run (workspace classdef files, `@folder` classes and
 * classdefs in the main file) and inheritance and member queries over them.
 */

import {
  HANDLE_BASE_CLASSES,
  type ClassInfo,
  type EventInfo,
  type MemberAccess,
  type MethodInfo,
  type PropertyInfo,
} from "./classInfo.js";

/** Public methods every handle class inherits from `handle`. */
export const HANDLE_CLASS_METHODS = [
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

/** The event every handle class inherits from `handle`, triggered when an
 *  object is deleted. */
export const OBJECT_BEING_DESTROYED: EventInfo = {
  name: "ObjectBeingDestroyed",
  definingClass: "handle",
  listenAccess: "public",
  notifyAccess: "private",
  hidden: false,
};

export class ClassTable {
  /** Workspace classes: qualified name → ClassInfo. */
  readonly workspaceClasses = new Map<string, ClassInfo>();
  /** Classes defined in the main file: name → ClassInfo. Not cleared when
   *  the workspace is rebuilt after addpath/rmpath/cd. */
  readonly localClasses = new Map<string, ClassInfo>();

  /** Member lookups by "class\0member", cleared when the table changes. */
  private readonly propertyCache = new Map<string, PropertyInfo | null>();
  private readonly methodCache = new Map<string, MethodInfo | null>();
  private readonly eventsCache = new Map<string, EventInfo[]>();

  /** Register a workspace class. First wins: earlier search paths take
   *  priority, as for workspace functions. */
  addWorkspaceClass(info: ClassInfo): void {
    if (!this.workspaceClasses.has(info.qualifiedName)) {
      this.workspaceClasses.set(info.qualifiedName, info);
      this.clearMemberCaches();
    }
  }

  addLocalClass(info: ClassInfo): void {
    this.localClasses.set(info.qualifiedName, info);
    this.clearMemberCaches();
  }

  clearWorkspaceClasses(): void {
    this.workspaceClasses.clear();
    this.clearMemberCaches();
  }

  private clearMemberCaches(): void {
    this.propertyCache.clear();
    this.methodCache.clear();
    this.eventsCache.clear();
  }

  /** Workspace classes, then main-file classes. */
  *entries(): IterableIterator<[string, ClassInfo]> {
    yield* this.workspaceClasses;
    yield* this.localClasses;
  }

  get(name: string): ClassInfo | null {
    return (
      this.workspaceClasses.get(name) ?? this.localClasses.get(name) ?? null
    );
  }

  /** Ancestors of a class in method-resolution order: depth-first, left to
   *  right through each `<` list, each name once. Built-in bases such as
   *  `handle` are included even though they have no ClassInfo; a built-in
   *  handle mixin (e.g. `matlab.mixin.Copyable`) is followed by `handle`. */
  superclassNames(info: ClassInfo): string[] {
    const out: string[] = [];
    const seen = new Set([info.qualifiedName]);
    const visit = (supers: string[]) => {
      for (const s of supers) {
        if (seen.has(s)) continue;
        seen.add(s);
        out.push(s);
        const parent = this.get(s);
        if (parent) visit(parent.superClasses);
        else if (HANDLE_BASE_CLASSES.has(s) && s !== "handle")
          visit(["handle"]);
      }
    };
    visit(info.superClasses);
    return out;
  }

  /** `superclassNames` by class name; empty for an unknown class. */
  ancestors(className: string): string[] {
    const info = this.get(className);
    return info ? this.superclassNames(info) : [];
  }

  /** Whether `className` is `ancestor` or derives from it. */
  isSubclassOf(className: string, ancestor: string): boolean {
    return (
      className === ancestor || this.ancestors(className).includes(ancestor)
    );
  }

  /** A class followed by its ancestors that have class definitions, in
   *  method-resolution order. */
  lineage(info: ClassInfo): ClassInfo[] {
    const out = [info];
    for (const name of this.superclassNames(info)) {
      const parent = this.get(name);
      if (parent) out.push(parent);
    }
    return out;
  }

  /** True if the class derives from a handle base class. */
  isHandleClass(info: ClassInfo): boolean {
    return this.superclassNames(info).some(s => HANDLE_BASE_CLASSES.has(s));
  }

  /** Whether a class (or any ancestor) has a given instance method. */
  hasMethod(className: string, methodName: string): boolean {
    const info = this.get(className);
    return (
      !!info && this.lineage(info).some(c => c.methodNames.has(methodName))
    );
  }

  /** Whether a class (or any ancestor) has a given static method. */
  hasStaticMethod(className: string, methodName: string): boolean {
    const info = this.get(className);
    return (
      !!info &&
      this.lineage(info).some(c => c.staticMethodNames.has(methodName))
    );
  }

  /** The class in the inheritance chain that defines a method (instance,
   *  external file, static or constructor); `className` itself if none. */
  findDefiningClass(className: string, methodName: string): string {
    const info = this.get(className);
    if (!info) return className;
    for (const c of this.lineage(info)) {
      if (
        c.methodNames.has(methodName) ||
        c.externalMethodFiles.has(methodName) ||
        c.staticMethodNames.has(methodName) ||
        methodName === c.constructorName
      ) {
        return c === info ? className : c.qualifiedName;
      }
    }
    return className;
  }

  // ── Members and access ────────────────────────────────────────────────

  /** The declaration of property `name` that instances of `className` see:
   *  the class's own, else the first ancestor's in method-resolution order.
   *  Undefined for an unknown class or property. */
  findProperty(className: string, name: string): PropertyInfo | undefined {
    const key = `${className}\0${name}`;
    let found = this.propertyCache.get(key);
    if (found === undefined) {
      found = null;
      const info = this.get(className);
      if (info) {
        for (const c of this.lineage(info)) {
          const p = c.properties.get(name);
          if (p) {
            found = p;
            break;
          }
        }
      }
      this.propertyCache.set(key, found);
    }
    return found ?? undefined;
  }

  /** The declaration of method `name` (instance, static or constructor)
   *  that `className` dispatches to, like `findProperty`. Undefined when no
   *  classdef declares it (e.g. a plain `@folder` method file). */
  findMethod(className: string, name: string): MethodInfo | undefined {
    const key = `${className}\0${name}`;
    let found = this.methodCache.get(key);
    if (found === undefined) {
      found = null;
      const info = this.get(className);
      if (info) {
        for (const c of this.lineage(info)) {
          const m = c.methods.get(name);
          if (m) {
            found = m;
            break;
          }
          // An @folder method file without a classdef signature is public
          // and overrides any inherited declaration.
          if (c.externalMethodFiles.has(name)) break;
        }
      }
      this.methodCache.set(key, found);
    }
    return found ?? undefined;
  }

  /** The declaration of event `name` that instances of `className` see
   *  (own, else inherited; `ObjectBeingDestroyed` for a handle class).
   *  Undefined for an unknown class or event. */
  findEvent(className: string, name: string): EventInfo | undefined {
    return this.allEvents(className).find(e => e.name === name);
  }

  /** All events of `className`: its own, then inherited ones in
   *  method-resolution order, then `ObjectBeingDestroyed` for a handle
   *  class (MATLAB's order). */
  allEvents(className: string): EventInfo[] {
    const cached = this.eventsCache.get(className);
    if (cached) return cached;
    const info = this.get(className);
    if (!info) return [];
    const out: EventInfo[] = [];
    const seen = new Set<string>();
    for (const c of this.lineage(info)) {
      for (const e of c.events.values()) {
        if (seen.has(e.name)) continue;
        seen.add(e.name);
        out.push(e);
      }
    }
    if (this.isHandleClass(info) && !seen.has(OBJECT_BEING_DESTROYED.name))
      out.push(OBJECT_BEING_DESTROYED);
    this.eventsCache.set(className, out);
    return out;
  }

  /** Whether code of class `context` (undefined: code outside any class)
   *  may use a member declared by `definingClass` with `access`. Private
   *  admits the defining class; protected, also its subclasses; a class
   *  list, also the listed classes and their subclasses. (`immutable` is
   *  treated as private here; the constructor-only rule is the caller's.) */
  canAccess(
    access: MemberAccess,
    definingClass: string,
    context: string | undefined
  ): boolean {
    if (access === "public") return true;
    if (access === "none" || context === undefined) return false;
    if (context === definingClass) return true;
    if (access === "protected")
      return this.isSubclassOf(context, definingClass);
    if (typeof access === "object")
      return access.classes.some(c => this.isSubclassOf(context, c));
    return false;
  }

  /** All properties instances of `className` have (own first, then
   *  inherited in method-resolution order), one declaration per name. */
  allProperties(className: string): PropertyInfo[] {
    const info = this.get(className);
    if (!info) return [];
    const out: PropertyInfo[] = [];
    const seen = new Set<string>();
    for (const c of this.lineage(info)) {
      for (const p of c.properties.values()) {
        if (seen.has(p.name)) continue;
        seen.add(p.name);
        out.push(p);
      }
    }
    return out;
  }

  /** The properties `properties`, `fieldnames` and object display list:
   *  publicly readable and not Hidden. */
  publicProperties(className: string): PropertyInfo[] {
    return this.allProperties(className).filter(
      p => p.getAccess === "public" && !p.hidden
    );
  }

  /** The methods `methods` and `ismethod` report: the public, non-Hidden
   *  instance and static methods of the class and its ancestors (classdef
   *  and `@folder` methods), the class's constructor, and the methods of
   *  `handle` for a handle class. Undefined for an unknown class. */
  publicMethodNames(className: string): Set<string> | undefined {
    const self = this.get(className);
    if (!self) return undefined;
    const lineage = this.lineage(self);
    const ctorName = self.constructorName ?? self.name;
    const candidates = new Set<string>([ctorName]);
    for (const c of lineage) {
      for (const n of c.methods.keys()) candidates.add(n);
      for (const n of c.methodNames) candidates.add(n);
      for (const n of c.staticMethodNames) candidates.add(n);
      for (const n of c.externalMethodFiles.keys()) candidates.add(n);
    }
    // Superclass constructors are not methods of the class.
    const superCtors = new Set(lineage.slice(1).map(c => c.name));
    const out = new Set<string>();
    for (const n of candidates) {
      if (n.includes(".")) continue; // get.X / set.X accessors
      if (n !== ctorName && superCtors.has(n)) continue;
      const m = this.findMethod(className, n);
      if (m && (m.access !== "public" || m.hidden)) continue;
      out.add(n);
    }
    if (this.isHandleClass(self))
      for (const n of HANDLE_CLASS_METHODS) out.add(n);
    return out;
  }

  /** Abstract members `className` declares or inherits without a concrete
   *  definition anywhere in its lineage. Empty for a concrete class. */
  unimplementedAbstractMembers(className: string): string[] {
    const info = this.get(className);
    if (!info) return [];
    const abstract = new Set<string>();
    const concrete = new Set<string>();
    for (const c of this.lineage(info)) {
      for (const p of c.properties.values())
        (p.abstract ? abstract : concrete).add(p.name);
      for (const m of c.methods.values())
        (m.abstract ? abstract : concrete).add(m.name);
      for (const name of c.externalMethodFiles.keys()) concrete.add(name);
    }
    return [...abstract].filter(n => !concrete.has(n));
  }
}
