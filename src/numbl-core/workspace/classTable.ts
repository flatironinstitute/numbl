/**
 * The classes known to a run (workspace classdef files, `@folder` classes and
 * classdefs in the main file) and inheritance queries over them.
 */

import { HANDLE_BASE_CLASSES, type ClassInfo } from "./classInfo.js";

export class ClassTable {
  /** Workspace classes: qualified name → ClassInfo. */
  readonly workspaceClasses = new Map<string, ClassInfo>();
  /** Classes defined in the main file: name → ClassInfo. Not cleared when
   *  the workspace is rebuilt after addpath/rmpath/cd. */
  readonly localClasses = new Map<string, ClassInfo>();

  /** Register a workspace class. First wins: earlier search paths take
   *  priority, as for workspace functions. */
  addWorkspaceClass(info: ClassInfo): void {
    if (!this.workspaceClasses.has(info.qualifiedName)) {
      this.workspaceClasses.set(info.qualifiedName, info);
    }
  }

  addLocalClass(info: ClassInfo): void {
    this.localClasses.set(info.qualifiedName, info);
  }

  clearWorkspaceClasses(): void {
    this.workspaceClasses.clear();
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
}
