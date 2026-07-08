/**
 * Class-definition registry. The `Workspace` walks each known
 * `Stmt.ClassDef` (main file + every workspace file containing a
 * classdef) and calls `registerClassDef` to validate and infer
 * property types from the declared default expressions. Results are
 * keyed by class name.
 *
 * Scope for v1:
 *   - Single `classdef Foo properties ... methods ... end end`
 *     blocks; no class attributes accepted.
 *   - No inheritance (`< Parent`), no handle classes, no
 *     `Events`/`Enumeration`/`Arguments` blocks, no operator
 *     overloads, no `get.`/`set.` accessors. Every rejected form
 *     surfaces with a clean `UnsupportedConstruct` carrying the
 *     class's source span.
 *   - Properties may declare a default-value expression. When set,
 *     the default's type drives the property's storage type
 *     eagerly. When unset, the property's type is inferred at first
 *     constructor specialization from the first
 *     `obj.<prop> = <rhs>` write at the top level of the
 *     constructor body — see `Lowerer.resolveClassType` in
 *     [lower.ts](./lower.ts). The C typedef hash uses
 *     `cFieldTypeStr` (one C-type string per property), so the
 *     typedef stays stable even if writes evolve the internal type.
 *   - Methods can be the constructor (named same as class, returns
 *     the receiver), an instance method, or a static method
 *     (declared inside `methods (Static)` block). v1 only supports
 *     methods with 0 or 1 outputs (same as user functions).
 */
import type { Span, Stmt, Expr } from "../parser/index.js";
import { UnaryOperation } from "../parser/index.js";
import { UnsupportedConstruct, TypeError } from "./errors.js";
import {
  type Type,
  type ClassType,
  classType,
  scalarDouble,
  scalarLogical,
  signFromNumber,
  tensorDouble,
} from "./types.js";

type ClassDefStmt = Extract<Stmt, { type: "ClassDef" }>;
type FuncStmt = Extract<Stmt, { type: "Function" }>;

export interface ClassRegistration {
  /** Source-level name. */
  className: string;
  /** File the classdef lives in (used to salt specialization keys
   *  and for diagnostics). */
  file: string;
  /** Declared property names in source order (sorted version lives on
   *  `ty.properties` once resolved). */
  propertyNames: string[];
  /** Properties with an explicit default-value expression. Each
   *  carries the lowered default's precise type (sign, exact, shape
   *  preserved). The C typedef hash uses `cFieldTypeStr` so precision
   *  differences across writes don't shard typedefs, but downstream
   *  reads / spec keying benefit from precision. */
  defaults: Map<string, { expr: Expr; ty: Type }>;
  /** Properties WITHOUT a default. Their C-level type is inferred at
   *  first constructor specialization from direct
   *  `obj.<prop> = <rhs>` writes in the constructor body. Disjoint
   *  from `defaults`. */
  pendingProperties: Set<string>;
  /** Class instance type. Eagerly resolved when every property has a
   *  default (`pendingProperties` empty). Otherwise null at
   *  registration and filled in by the lowerer's
   *  `resolveClassType()` at first constructor specialization. */
  ty: ClassType | null;
  /** Constructor (a method whose name matches the class). May be
   *  null when the class has no constructor — in that case the
   *  default-valued receiver is the constructor's "body" and
   *  `Foo()` (no args) is the only legal call. Required when
   *  `pendingProperties` is non-empty. */
  constructor: FuncStmt | null;
  /** Instance methods, keyed by source name. The constructor is NOT
   *  included here. Accessors (`get.X` / `set.X`) live in `getters`
   *  / `setters` and are NOT in this map. */
  methods: Map<string, FuncStmt>;
  /** Static methods, keyed by source name. Called as
   *  `ClassName.method(args)` — the receiver is not passed. */
  staticMethods: Map<string, FuncStmt>;
  /** `Dependent` properties — no per-instance storage. Reads route
   *  through `getters.get(name)`; writes route through
   *  `setters.get(name)` if a setter exists. Disjoint from
   *  `pendingProperties` and `defaults`; included in `propertyNames`
   *  for name-collision checks. */
  dependentProperties: Set<string>;
  /** Property getters declared as `function val = get.X(obj)`.
   *  Keyed by the property name (without the `get.` prefix). v1
   *  requires the property to be `Dependent`. */
  getters: Map<string, FuncStmt>;
  /** Property setters declared as `function obj = set.X(obj, val)`.
   *  Keyed by the property name (without the `set.` prefix). v1
   *  requires the property to be `Dependent`. */
  setters: Map<string, FuncStmt>;
}

/** Validate one `classdef` AST and infer its property types. Throws
 *  `UnsupportedConstruct` for any v1-unsupported form (inheritance,
 *  class attributes, get/set accessors, events/enumeration/arguments
 *  blocks, operator overloads, etc.). The `file` argument is stored
 *  on the registration so cross-file specialization keys can salt by
 *  it.
 *
 *  `externalMethods` carries instance methods discovered by numbl
 *  in an `@ClassName/` folder — one `<methodName>.m` per file. Numbl's
 *  `registerWorkspaceFiles` already collected them in
 *  `ClassInfo.externalMethodFiles`; the `Workspace` finalizer extracts
 *  each file's primary Function AST and passes them here so they
 *  participate in the same validation and method-map as in-body
 *  methods. */
export function registerClassDef(
  s: ClassDefStmt,
  file: string,
  externalMethods?: ReadonlyMap<string, FuncStmt>
): ClassRegistration {
  // Enumeration classes run exclusively on the interpreter (member access,
  // conversion, and ==/~=/switch semantics live in the runtime — see
  // `runtimeEnum.ts`). Decline them here so the JIT falls back to the
  // interpreter cleanly at the first use rather than trying to codegen an
  // enumeration member or its numeric superclass.
  if (s.members.some(m => m.type === "Enumeration")) {
    throw new UnsupportedConstruct(
      `classdef '${s.name}' is an enumeration; enumeration classes are ` +
        `interpreter-only in the JIT`,
      s.span
    );
  }
  if (s.classAttributes.length > 0) {
    throw new UnsupportedConstruct(
      `classdef '${s.name}' has class attributes; not supported in v1`,
      s.span
    );
  }
  if (s.superClass !== null) {
    throw new UnsupportedConstruct(
      `classdef '${s.name}' has a superclass; inheritance is not supported in v1`,
      s.span
    );
  }

  const propertyNames: string[] = [];
  const propsWithDefault: { name: string; ty: Type }[] = [];
  const defaults = new Map<string, { expr: Expr; ty: Type }>();
  const pendingProperties = new Set<string>();
  const methods = new Map<string, FuncStmt>();
  const staticMethods = new Map<string, FuncStmt>();
  const dependentProperties = new Set<string>();
  const getters = new Map<string, FuncStmt>();
  const setters = new Map<string, FuncStmt>();
  // Method names that appear as `name(args)` rows inside a
  // `methods (Static)` block — i.e. signature-only declarations
  // (no `function` keyword, body lives in `@<Cls>/<name>.m`).
  // When `collectExternalMethods` later hands us the external file's
  // FuncStmt, presence of the name in this set routes it to
  // `staticMethods` rather than to `methods`.
  const staticSignatureNames = new Set<string>();
  let constructor: FuncStmt | null = null;

  for (const m of s.members) {
    switch (m.type) {
      case "Properties": {
        // Property-block attributes. mtoc2 doesn't enforce visibility,
        // so `Access` / `GetAccess` / `SetAccess` / `Hidden` are
        // accepted silently — matches numbl, which ignores them in
        // `extractClassInfo`. `Dependent` strips per-instance storage
        // and routes reads/writes through `get.X` / `set.X` accessor
        // methods. Everything else is rejected by name.
        let isDependent = false;
        for (const attr of m.attributes) {
          const lower = attr.name.toLowerCase();
          if (
            lower === "access" ||
            lower === "getaccess" ||
            lower === "setaccess" ||
            lower === "hidden"
          ) {
            continue;
          }
          if (lower === "dependent") {
            if (attr.value === null || attr.value === "true") {
              isDependent = true;
              continue;
            }
            // `Dependent = false` is legal but does nothing.
            if (attr.value === "false") {
              continue;
            }
          }
          throw new UnsupportedConstruct(
            `'properties' block attribute '${attr.name}' is not supported in v1`,
            s.span
          );
        }
        for (let i = 0; i < m.names.length; i++) {
          const name = m.names[i];
          const def = m.defaultValues[i];
          if (
            propertyNames.includes(name) ||
            pendingProperties.has(name) ||
            propsWithDefault.some(p => p.name === name)
          ) {
            throw new UnsupportedConstruct(
              `duplicate property '${name}' on class '${s.name}'`,
              s.span
            );
          }
          propertyNames.push(name);
          if (isDependent) {
            // Dependent properties have no per-instance storage. A
            // default value would be meaningless (the getter computes
            // the value); reject so the user can fix the source.
            if (def !== null) {
              throw new UnsupportedConstruct(
                `dependent property '${name}' on class '${s.name}' ` +
                  `cannot have a default value`,
                s.span
              );
            }
            dependentProperties.add(name);
            continue;
          }
          if (def === null) {
            // No default. Type will be inferred at first constructor
            // specialization from the first `obj.<name> = <rhs>` write
            // in the constructor body.
            pendingProperties.add(name);
            continue;
          }
          // Carry the default's precise type (sign, exact, shape) on
          // the property. The C typedef hash uses `cFieldTypeStr` so
          // precision differences across writes don't shard typedefs,
          // but downstream reads / spec keying benefit from precision.
          const ty = inferDefaultType(def, name);
          propsWithDefault.push({ name, ty });
          defaults.set(name, { expr: def, ty });
        }
        break;
      }
      case "Methods": {
        // The only `methods (...)` block attribute supported in v1 is
        // `Static = true` (or bare `Static`). Anything else (Access,
        // Sealed, Hidden, ...) is rejected so unsupported semantics
        // don't silently slip through.
        let isStatic = false;
        for (const attr of m.attributes) {
          if (
            attr.name.toLowerCase() === "static" &&
            (attr.value === null || attr.value === "true")
          ) {
            isStatic = true;
            continue;
          }
          throw new UnsupportedConstruct(
            `'methods' block attribute '${attr.name}' is not supported in v1`,
            s.span
          );
        }
        // Method signatures — prototype rows like
        // `[a, b] = foo(obj, x)` without a `function` keyword. The
        // body lives in an `@<Cls>/<name>.m` external file (or, for
        // some chunkie-style classes, in a workspace-level `<name>.m`
        // that the class signature is declaring for documentation
        // only). Numbl mirrors this in `extractClassInfo`: signatures
        // populate `methodNames` / `staticMethodNames` but don't carry
        // a body themselves. We use them for one thing only — knowing
        // whether an external method file should be registered as
        // `staticMethods` vs `methods`. Output-arity, input-arity, and
        // declared-but-unimplemented signatures are accepted silently
        // (matches numbl); the call site is what surfaces a missing
        // implementation.
        for (const sig of m.signatures ?? []) {
          if (isStatic) {
            staticSignatureNames.add(sig.name);
          }
        }
        for (const stmt of m.body) {
          if (stmt.type !== "Function") {
            throw new UnsupportedConstruct(
              `non-Function statement inside 'methods' block`,
              stmt.span
            );
          }
          // Property accessor methods (`get.X` / `set.X`). Validated
          // here so a bad accessor surfaces at classdef-registration
          // time rather than at the first call site. The property X
          // must be declared in some `properties(...)` block of this
          // class. v1 only routes through accessors when the property
          // is `Dependent` — accessors on storage-backed properties
          // are rejected (deferred until a real use case surfaces).
          if (stmt.name.startsWith("get.") || stmt.name.startsWith("set.")) {
            if (isStatic) {
              throw new UnsupportedConstruct(
                `accessor '${stmt.name}' cannot be declared in a 'methods (Static)' block`,
                stmt.span
              );
            }
            const isGet = stmt.name.startsWith("get.");
            const propName = stmt.name.slice(4);
            const kind = isGet ? "getter" : "setter";
            if (!propertyNames.includes(propName)) {
              throw new UnsupportedConstruct(
                `${kind} '${stmt.name}' on class '${s.name}': ` +
                  `'${propName}' is not a declared property`,
                stmt.span
              );
            }
            // Storage-backed properties with paired accessors (the
            // chunkgraph pattern: the property carries real storage
            // but reads/writes are intercepted by `get.X` / `set.X`)
            // are accepted. The storage stays — matching numbl, which
            // always routes through the accessor when one exists and
            // ignores the field even if present. mtoc2's lowerer /
            // interpreter already do this dispatch.
            //
            // No recursion guard in v1: a getter that reads its own
            // property (`get.X` body containing `obj.X`) would
            // infinite-loop. Real codebases use the accessor to
            // compute from OTHER fields, so this hasn't surfaced.
            if (stmt.outputs.length !== 1) {
              throw new UnsupportedConstruct(
                `${kind} '${stmt.name}' must declare exactly one output`,
                stmt.span
              );
            }
            const expectedParams = isGet ? 1 : 2;
            if (stmt.params.length !== expectedParams) {
              throw new UnsupportedConstruct(
                `${kind} '${stmt.name}' must take ${expectedParams} ` +
                  `parameter${expectedParams === 1 ? "" : "s"} ` +
                  `(${isGet ? "obj" : "obj, val"}), got ${stmt.params.length}`,
                stmt.span
              );
            }
            const target = isGet ? getters : setters;
            if (target.has(propName)) {
              throw new UnsupportedConstruct(
                `duplicate ${kind} for property '${propName}' on class '${s.name}'`,
                stmt.span
              );
            }
            target.set(propName, stmt);
            continue;
          }
          if (isStatic) {
            // Static methods can't double as the constructor.
            if (stmt.name === s.name) {
              throw new UnsupportedConstruct(
                `constructor '${stmt.name}' cannot be declared in a 'methods (Static)' block`,
                stmt.span
              );
            }
            if (
              staticMethods.has(stmt.name) ||
              methods.has(stmt.name) ||
              stmt.name === s.name
            ) {
              throw new UnsupportedConstruct(
                `duplicate method '${stmt.name}' on class '${s.name}'`,
                stmt.span
              );
            }
            staticMethods.set(stmt.name, stmt);
            continue;
          }
          if (stmt.name === s.name) {
            if (constructor !== null) {
              throw new UnsupportedConstruct(
                `duplicate constructor '${stmt.name}' on class '${s.name}'`,
                stmt.span
              );
            }
            if (stmt.outputs.length !== 1) {
              throw new UnsupportedConstruct(
                `constructor '${stmt.name}' must declare exactly one output (the receiver)`,
                stmt.span
              );
            }
            constructor = stmt;
          } else {
            if (methods.has(stmt.name) || staticMethods.has(stmt.name)) {
              throw new UnsupportedConstruct(
                `duplicate method '${stmt.name}' on class '${s.name}'`,
                stmt.span
              );
            }
            methods.set(stmt.name, stmt);
          }
        }
        break;
      }
      case "Events":
      case "Enumeration":
      case "Arguments":
        throw new UnsupportedConstruct(
          `'${m.type}' blocks are not supported in v1`,
          s.span
        );
    }
  }

  // External method files (`@ClassName/<methodName>.m`). Same validation
  // as in-body instance methods — the source location just lives in a
  // different file. Accessor methods (`get.X` / `set.X`) are not
  // permitted in external method files (MATLAB itself disallows this).
  if (externalMethods) {
    for (const [methodName, stmt] of externalMethods) {
      if (stmt.name.startsWith("get.") || stmt.name.startsWith("set.")) {
        throw new UnsupportedConstruct(
          `accessor method '${stmt.name}' cannot be declared in an external method file; ` +
            `accessors must live in the classdef`,
          stmt.span
        );
      }
      if (methodName === s.name) {
        // MATLAB allows a constructor in an external file; mtoc2's
        // type-inference path keys off the classdef-file constructor,
        // so we reject this for now.
        throw new UnsupportedConstruct(
          `class '${s.name}': external constructor file is not supported; ` +
            `declare the constructor inside the classdef`,
          stmt.span
        );
      }
      if (methods.has(methodName) || staticMethods.has(methodName)) {
        throw new UnsupportedConstruct(
          `duplicate method '${methodName}' on class '${s.name}' ` +
            `(declared both in the classdef and in an external file)`,
          stmt.span
        );
      }
      // A signature inside a `methods (Static)` block routes the
      // external file to `staticMethods` instead of `methods` — the
      // class-side declaration is the only thing that distinguishes
      // an external file's static-vs-instance status (the file
      // itself just contains a plain `function`).
      if (staticSignatureNames.has(methodName)) {
        staticMethods.set(methodName, stmt);
      } else {
        methods.set(methodName, stmt);
      }
    }
  }

  // When every property has a default, build the ClassType eagerly so
  // downstream code can read `reg.ty` without a resolve step. When at
  // least one property lacks a default, leave `ty` null — the lowerer
  // fills it in at first constructor specialization.
  const ty =
    pendingProperties.size === 0 ? classType(s.name, propsWithDefault) : null;
  // A class with pending properties must declare a constructor — that's
  // where the inference reads from. Without one, the class is
  // un-constructible (no zero-arg path can produce a typed instance).
  if (pendingProperties.size > 0 && constructor === null) {
    throw new UnsupportedConstruct(
      `class '${s.name}' has property without a default (${[...pendingProperties].join(", ")}) ` +
        `but declares no constructor; either add defaults or define a constructor that ` +
        `assigns each property at the top level of its body`,
      s.span
    );
  }
  // Every dependent property must have a getter — without one it is
  // unreadable. (`set.X` remains optional; write-only is uncommon and
  // chunker has read-only dependents like `k` / `dim`.)
  for (const name of dependentProperties) {
    if (!getters.has(name)) {
      throw new UnsupportedConstruct(
        `dependent property '${name}' on class '${s.name}' has no getter; ` +
          `declare a 'function val = get.${name}(obj)' method`,
        s.span
      );
    }
  }
  return {
    className: s.name,
    file,
    propertyNames,
    defaults,
    pendingProperties,
    ty,
    constructor,
    methods,
    staticMethods,
    dependentProperties,
    getters,
    setters,
  };
}

/** Shallow type inference for a property default-value expression.
 *  Accepts numeric literals, signed numeric literals, and tensor
 *  literals whose cells are all numeric. Anything richer (function
 *  call, identifier, etc.) is rejected — v1 requires defaults that
 *  the registry can type without running the full lowerer. */
function inferDefaultType(e: Expr, propName: string): Type {
  const span: Span = e.span;
  switch (e.type) {
    case "Number": {
      const v = Number(e.value);
      if (!Number.isFinite(v)) {
        throw new UnsupportedConstruct(
          `property '${propName}': default value '${e.value}' is non-finite`,
          span
        );
      }
      return scalarDouble(signFromNumber(v), v);
    }
    case "Ident": {
      // The boolean constants `true` / `false` parse as Idents
      // (they're not literal tokens in the grammar — they're nullary
      // builtin functions). Recognize them by name so property
      // defaults like `isnan = false` get a scalar logical type at
      // registration time. Matches numbl, which lazily evaluates
      // `true` / `false` to scalar logical at construction.
      if (e.name === "true") return scalarLogical(true);
      if (e.name === "false") return scalarLogical(false);
      throw new UnsupportedConstruct(
        `property '${propName}': default must be a literal numeric, ` +
          `tensor, or 'true'/'false' (got identifier '${e.name}')`,
        span
      );
    }
    case "Unary": {
      // Accept unary +/- on a numeric literal.
      const inner = inferDefaultType(e.operand, propName);
      if (inner.kind !== "Numeric" || typeof inner.exact !== "number") {
        throw new UnsupportedConstruct(
          `property '${propName}': default must be a numeric literal (or its negation)`,
          span
        );
      }
      if (e.op === UnaryOperation.Plus) return inner;
      if (e.op === UnaryOperation.Minus) {
        const v = -inner.exact;
        return scalarDouble(signFromNumber(v), v);
      }
      throw new UnsupportedConstruct(
        `property '${propName}': unsupported unary op in default`,
        span
      );
    }
    case "Tensor": {
      // Empty `[]` → 0×0 tensor.
      if (e.rows.length === 0) {
        return tensorDouble([0, 0]);
      }
      const rows = e.rows.length;
      const cols = e.rows[0].length;
      for (const r of e.rows) {
        if (r.length !== cols) {
          throw new UnsupportedConstruct(
            `property '${propName}': tensor default has ragged rows`,
            span
          );
        }
      }
      // Verify every cell is a numeric literal (or +/-numeric).
      const data = new Float64Array(rows * cols);
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const cellTy = inferDefaultType(e.rows[r][c], propName);
          if (cellTy.kind !== "Numeric" || typeof cellTy.exact !== "number") {
            throw new TypeError(
              `property '${propName}': tensor default cell must be a numeric literal`,
              span
            );
          }
          data[c * rows + r] = cellTy.exact;
        }
      }
      // 1×1 collapses to scalar (matches MATLAB).
      if (rows === 1 && cols === 1) {
        return scalarDouble(signFromNumber(data[0]), data[0]);
      }
      return tensorDouble([rows, cols], data);
    }
    default:
      throw new UnsupportedConstruct(
        `property '${propName}': default must be a literal numeric or tensor ` +
          `(got '${e.type}')`,
        span
      );
  }
}
