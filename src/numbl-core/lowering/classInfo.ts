/**
 * Class information types and extraction utilities.
 */

import { type Expr, type Stmt } from "../parser/index.js";

/**
 * Metadata about a class definition.
 * The ctx field is typed as 'unknown' to avoid circular dependencies with LoweringContext.
 * At runtime, it will hold a LoweringContext instance.
 */
export interface ClassInfo {
  name: string; // e.g. "Rectangle_"
  qualifiedName: string; // e.g. "Rectangle_" or "geometry.Circle"
  fileName: string; // e.g. "Rectangle_.m"
  source: string; // raw .m file source
  superClass: string | null;
  propertyNames: string[];
  propertyDefaults: Map<string, Expr>; // property name → default AST expr
  methodNames: Set<string>; // instance method names (non-static, non-constructor)
  staticMethodNames: Set<string>;
  constructorName: string | null; // method name matching class name
  ast: (Stmt & { type: "ClassDef" }) | null; // the parsed ClassDef AST (null for old-style @folder classes)
  isOldStyle?: boolean; // true for pre-classdef @folder classes (constructor is a plain function calling class(s,'name'))
  inferiorClasses: string[]; // classes declared inferior via InferiorClasses attribute
  externalMethodFiles: Map<string, { fileName: string; source: string }>; // method name → file info (for @folder methods)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ctx?: any; // lazily created lowering context (typed as any to avoid circular deps)
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
  const methodNames = new Set<string>();
  const staticMethodNames = new Set<string>();
  let constructorName: string | null = null;

  // The base class name (last segment for qualified names like "pkg.ClassName")
  const dotIdx = qualifiedName.lastIndexOf(".");
  const baseName =
    dotIdx >= 0 ? qualifiedName.slice(dotIdx + 1) : qualifiedName;

  for (const member of classDef.members) {
    if (member.type === "Properties") {
      const isDependent = member.attributes.some(
        a =>
          a.name.toLowerCase() === "dependent" &&
          (a.value === null || a.value === "true")
      );
      if (!isDependent) {
        for (let i = 0; i < member.names.length; i++) {
          propertyNames.push(member.names[i]);
          if (member.defaultValues[i]) {
            propertyDefaults.set(member.names[i], member.defaultValues[i]!);
          }
        }
      }
    } else if (member.type === "Methods") {
      const isStatic = member.attributes.some(
        a =>
          a.name.toLowerCase() === "static" &&
          (a.value === null || a.value === "true")
      );
      for (const methodStmt of member.body) {
        if (methodStmt.type === "Function") {
          if (isStatic) {
            staticMethodNames.add(methodStmt.name);
          } else if (
            methodStmt.name === classDef.name ||
            methodStmt.name === baseName
          ) {
            constructorName = methodStmt.name;
          } else {
            methodNames.add(methodStmt.name);
          }
        }
      }
      // Also process method signatures (prototype declarations without
      // function keyword, e.g. `result = myMethod(x);`). These are used
      // for external method files declared in the classdef.
      if (member.signatures) {
        for (const sig of member.signatures) {
          if (isStatic) {
            staticMethodNames.add(sig.name);
          } else if (sig.name === classDef.name || sig.name === baseName) {
            constructorName = sig.name;
          } else {
            methodNames.add(sig.name);
          }
        }
      }
    }
  }

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
    superClass: classDef.superClass,
    propertyNames,
    propertyDefaults,
    methodNames,
    staticMethodNames,
    constructorName,
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
    superClass: null,
    propertyNames: [],
    propertyDefaults: new Map(),
    methodNames,
    staticMethodNames: new Set(),
    constructorName: baseName,
    inferiorClasses: [],
    ast: null,
    isOldStyle: true,
    externalMethodFiles,
  };
}
