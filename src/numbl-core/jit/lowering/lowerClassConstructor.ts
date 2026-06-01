/**
 * Class-constructor lowering.
 *
 * Three responsibilities:
 *
 *   - `lowerClassConstructorCall` — produces the `Call` IR node that
 *     invokes the user-declared constructor, after settling the
 *     class's `ClassType` and synthesizing the default-valued
 *     receiver that the constructor body sees on entry.
 *
 *   - `resolveClassType` — for classes whose property types weren't
 *     fully inferred at registration (because at least one property
 *     declares no default), pre-scan the constructor body for the
 *     first top-level `<receiver>.<prop> = <rhs>` write per pending
 *     property and lower its RHS in a temp env to derive the
 *     property's type. The first call wins; subsequent calls reuse
 *     `reg.ty` and validate via the normal `MemberStore` storage-
 *     equivalence check.
 *
 *   - `makeInitialClassReceiver` — synthesize a `StructLit` whose
 *     ty is the just-resolved `ClassType` and whose field values are
 *     either the registered defaults or a zero/empty placeholder for
 *     pending properties. The constructor body's first write
 *     overwrites the placeholder.
 */

import type { Expr, Span, Stmt } from "../parser/index.js";
import { TypeError, UnsupportedConstruct } from "./errors.js";
import type { IRExpr } from "./ir.js";
import {
  type ClassType,
  type Type,
  classType,
  isMultiElement,
  scalarDouble,
  tensorDouble,
  typeToString,
} from "./types.js";
import type { ClassRegistration } from "./classDefs.js";
import type { Lowerer } from "./lower.js";
import { cIdentForUserName } from "./lower.js";
import { specializeUserFunction } from "./specialize.js";

/** Lower `ClassName(args)` to the `Call` IR node that invokes the
 *  user-declared constructor. The constructor's first statement gets
 *  a synthetic `obj = <default-receiver>` prepended (via
 *  `specializeUserFunction`'s `preSeedOutput`) so the body's reads
 *  and writes against the receiver work against an initialized
 *  slot. */
export function lowerClassConstructorCall(
  this: Lowerer,
  reg: ClassRegistration,
  args: Expr[],
  span: Span
): IRExpr {
  if (reg.constructor === null) {
    // No constructor declared: only a zero-arg call is valid; the
    // value IS the default-valued receiver. (Classes with pending
    // properties are required to declare a constructor at
    // registration time, so reg.ty is non-null on this branch.)
    if (args.length !== 0) {
      throw new TypeError(
        `class '${reg.className}' has no constructor; cannot pass arguments`,
        span
      );
    }
    const ty = resolveClassType.call(this, reg, [], span);
    return makeInitialClassReceiver.call(this, reg, ty, span);
  }
  const userArgs = args.map(a => this.lowerExpr(a));
  for (const a of userArgs) {
    this.requireValueType(a, `argument to constructor '${reg.className}'`);
  }
  const argTypes = userArgs.map(a => a.ty);
  const classTy = resolveClassType.call(this, reg, argTypes, span);
  const initialReceiver = makeInitialClassReceiver.call(
    this,
    reg,
    classTy,
    span
  );
  const outName = reg.constructor.outputs[0];
  const spec = specializeUserFunction.call(
    this,
    reg.constructor,
    argTypes,
    reg.className,
    reg.file,
    { name: outName, ty: classTy, initExpr: initialReceiver },
    1,
    span
  );
  // Constructor must return one output (validated at registration).
  const ty: Type = spec.outputTypes[0] ?? classTy;
  return {
    kind: "Call",
    cName: spec.cName,
    name: reg.className,
    args: userArgs,
    ty,
    span,
  };
}

/** Settle the class's `ClassType`. For a class with every property
 *  declaring a default, `reg.ty` is already filled in at
 *  registration and we just return it. For a class with pending
 *  properties, we pre-scan the constructor body for direct
 *  `obj.<prop> = <rhs>` writes (where `obj` is the constructor's
 *  output receiver) and lower each first-write RHS in a temp env
 *  bound to the call's `argTypes`. The first call wins — subsequent
 *  specs validate against the cached type via the normal
 *  `MemberStore` storage-equivalence check. */
export function resolveClassType(
  this: Lowerer,
  reg: ClassRegistration,
  argTypes: Type[],
  span: Span
): ClassType {
  if (reg.ty !== null) return reg.ty;
  // `registerClassDef` enforces that pendingProperties.size > 0
  // implies a constructor is declared. Defensive assertion.
  if (reg.constructor === null) {
    throw new UnsupportedConstruct(
      `internal: class '${reg.className}' has pending properties but no constructor`,
      span
    );
  }
  const decl = reg.constructor;
  // Match the user-function rule: too-many args is an error, fewer
  // is fine — the body's `if nargin < N` branch (folded by the
  // exact `nargin` fold) handles the unbound trailing params.
  if (argTypes.length > decl.params.length) {
    throw new TypeError(
      `constructor '${reg.className}' expects at most ${decl.params.length} arg(s), got ${argTypes.length}`,
      span
    );
  }
  const receiverName = decl.outputs[0];

  // Save outer lowering state — we're going to lower the first-write
  // RHSs in a fresh env that mirrors the constructor's entry state
  // (params bound, no other locals).
  const savedEnv = this.env;
  const savedTempCounter = this.tempCounter;
  const savedCurrentFile = this.currentFile;
  this.env = new Map();
  this.tempCounter = 0;
  this.currentFile = reg.file;
  for (let i = 0; i < argTypes.length; i++) {
    this.env.set(decl.params[i], {
      cName: cIdentForUserName(decl.params[i]),
      ty: argTypes[i],
    });
  }

  const props: { name: string; ty: Type }[] = [];
  try {
    for (const propName of reg.propertyNames) {
      // Dependent properties have no storage — they are not
      // materialized as ClassType fields. Reads route through
      // `get.<prop>`, writes through `set.<prop>`. Skip the
      // default/first-write inference path for them entirely.
      if (reg.dependentProperties.has(propName)) {
        continue;
      }
      const def = reg.defaults.get(propName);
      if (def !== undefined) {
        // Default-having property: use the type already inferred at
        // registration (it's literal-derived, so it's stable).
        props.push({ name: propName, ty: def.ty });
        continue;
      }
      // Pending property: find the first top-level direct write in
      // the constructor body and lower its RHS for its static type.
      const rhs = findFirstPropertyWrite(decl.body, receiverName, propName);
      if (rhs === null) {
        throw new UnsupportedConstruct(
          `class '${reg.className}' property '${propName}' has no default ` +
            `and is not directly assigned at the top level of the ` +
            `constructor body (\`${receiverName}.${propName} = <expr>;\`); ` +
            `either add a default value or add such an assignment`,
          decl.span
        );
      }
      const inferred = this.lowerExpr(rhs);
      this.requireValueType(
        inferred,
        `inferring type of '${reg.className}.${propName}'`
      );
      props.push({ name: propName, ty: inferred.ty });
    }
  } finally {
    this.env = savedEnv;
    this.tempCounter = savedTempCounter;
    this.currentFile = savedCurrentFile;
  }

  const ty = classType(reg.className, props);
  reg.ty = ty;
  return ty;
}

/** Synthesize a `StructLit` whose ty is `classTy` and whose field
 *  values are the property defaults from `reg.defaults` — for any
 *  property without a default, synthesize a zero-value matching the
 *  inferred C-level type (the constructor body's first write
 *  overwrites it anyway; the zero is just a typed placeholder so
 *  the C designated initializer is well-formed and any
 *  read-before-write reads as 0 / empty). */
export function makeInitialClassReceiver(
  this: Lowerer,
  reg: ClassRegistration,
  classTy: ClassType,
  span: Span
): IRExpr {
  const fields: { name: string; value: IRExpr }[] = [];
  for (const p of classTy.properties) {
    const def = reg.defaults.get(p.name);
    let value: IRExpr;
    if (def !== undefined) {
      // Defaults are restricted to literals, so an empty env is
      // sufficient to lower them.
      const savedEnv = this.env;
      this.env = new Map();
      value = this.lowerExpr(def.expr);
      this.env = savedEnv;
    } else {
      value = synthesizeZeroValue(p.ty, reg.className, p.name, span);
    }
    fields.push({ name: p.name, value });
  }
  return {
    kind: "StructLit",
    fields,
    ty: classTy,
    span,
  };
}

/** Scan a constructor body for the FIRST top-level direct
 *  `<receiver>.<propName> = <rhs>` assignment. Returns the rhs Expr,
 *  or `null` if no such assignment is found. Conditional / loop /
 *  nested-block writes are intentionally NOT considered — for v1,
 *  property-type inference relies on writes that the body
 *  unconditionally performs. */
function findFirstPropertyWrite(
  body: Stmt[],
  receiverName: string,
  propName: string
): Expr | null {
  for (const s of body) {
    if (
      s.type === "AssignLValue" &&
      s.lvalue.type === "Member" &&
      s.lvalue.name === propName &&
      s.lvalue.base.type === "Ident" &&
      s.lvalue.base.name === receiverName
    ) {
      return s.expr;
    }
  }
  return null;
}

/** Build a zero / empty IR value of `ty`'s C-level shape. Used by
 *  `makeInitialClassReceiver` to fill the `StructLit` slot for a
 *  property that lacks an explicit default — the constructor body's
 *  first write overwrites it anyway, but the C designated
 *  initializer still needs a syntactic value, and a read-before-
 *  write should observe a stable zero. Only Numeric types are
 *  supported in v1; struct / class / handle / string properties
 *  without defaults raise `UnsupportedConstruct` here. */
function synthesizeZeroValue(
  ty: Type,
  className: string,
  propName: string,
  span: Span
): IRExpr {
  if (ty.kind === "Numeric") {
    if (isMultiElement(ty)) {
      // Empty 0×0 tensor — matches MATLAB's `[]` initial value. The
      // first constructor write replaces it.
      return {
        kind: "TensorBuild",
        elements: [],
        shape: [0, 0],
        ty: tensorDouble([0, 0]),
        span,
      };
    }
    return {
      kind: "NumLit",
      value: 0,
      ty: scalarDouble("zero", 0),
      span,
    };
  }
  throw new UnsupportedConstruct(
    `class '${className}' property '${propName}' is inferred to type ` +
      `${typeToString(ty)}, but v1 can only synthesize a zero placeholder ` +
      `for numeric properties; provide an explicit default value`,
    span
  );
}
