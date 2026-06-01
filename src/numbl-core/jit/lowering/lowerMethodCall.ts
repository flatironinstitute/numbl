/**
 * Method-call lowering — every `<base>.<name>(args)` AST node:
 *
 *   - **Package call**: `pkg.foo(args)` and `pkg.sub.foo(args)` —
 *     leftmost segment is not an in-scope variable. Routes to the
 *     userFunction / classConstructor verdict from the resolver.
 *   - **Static method call**: `ClassName.staticMethod(args)` — base
 *     resolves to a class name. Receiver is not present in the arg
 *     tuple.
 *   - **Instance method call**: `obj.method(args)` — base evaluates
 *     to a class instance. The receiver becomes the first arg when
 *     numbl's resolver decides this is an instance dispatch
 *     (`stripInstance === false`).
 *   - **Member-rooted indexing**: `obj.field(args)` where `field` is
 *     a property (not a method). The field load lands in a fresh
 *     temp and the args run through the standard `lowerIndexLoad` /
 *     `lowerIndexSlice` path.
 */

import type { Expr, Span } from "../parser/index.js";
import { TypeError, UnsupportedConstruct } from "./errors.js";
import type { IRExpr } from "./ir.js";
import {
  type Type,
  classMethodSpecSource,
  fieldType,
  isClass,
  typeToString,
} from "./types.js";
import { isSliceArg } from "./indexResolve.js";
import { lowerIndexLoad } from "./lowerIndexLoad.js";
import { lowerIndexSlice } from "./lowerIndexSlice.js";
import { lowerClassConstructorCall } from "./lowerClassConstructor.js";
import { buildUserFunctionCall } from "./specialize.js";
import type { Lowerer } from "./lower.js";
import { tryExtractDottedName } from "./lower.js";

export function lowerMethodCall(
  this: Lowerer,
  e: Extract<Expr, { type: "MethodCall" }>
): IRExpr {
  // Package function / qualified class call. The base is a chain of
  // Ident/Member (no calls, no index) whose leftmost segment is not
  // an in-scope variable.
  const dottedBase = tryExtractDottedName(e.base);
  if (dottedBase && !this.env.has(dottedBase.split(".")[0])) {
    const qname = `${dottedBase}.${e.name}`;
    // `pkg.Foo(args)` — packaged class constructor.
    if (this.workspace.isClass(qname)) {
      const reg = this.workspace.requireClass(qname);
      if (reg === undefined) {
        throw new UnsupportedConstruct(
          `internal: class '${qname}' missing from workspace registry`,
          e.span
        );
      }
      return lowerClassConstructorCall.call(this, reg, e.args, e.span);
    }
    // `ClassName.staticMethod(args)` where `ClassName` is either a
    // bare class or a qualified one (`pkg.Foo.staticMethod(...)`).
    if (this.workspace.isClass(dottedBase)) {
      return lowerStaticMethodCall.call(this, dottedBase, e, e.span);
    }
    // `pkg.foo(args)` — packaged workspace function. Let the
    // resolver decide; we route the userFunction verdict through the
    // same path as `lowerFuncCall`.
    const args = e.args.map(a => this.lowerExpr(a));
    for (const a of args) {
      this.requireValueType(a, `argument to '${qname}'`);
    }
    const argTypes = args.map(a => a.ty);
    const target = this.workspace.resolve(
      qname,
      argTypes,
      this.callSite(),
      e.span
    );
    if (target?.kind === "userFunction") {
      // Expression-context call site requests exactly 1 output (or 0
      // for a void-declared function); see `buildUserFunctionCall`
      // for the discipline.
      return buildUserFunctionCall.call(this, target.ast, args, qname, e.span, {
        specSource: qname,
        definingFile: target.file,
      });
    }
    if (target?.kind === "classConstructor") {
      const reg = this.classReg(target.className);
      if (reg === undefined) {
        throw new UnsupportedConstruct(
          `internal: class '${target.className}' missing from workspace registry`,
          e.span
        );
      }
      return lowerClassConstructorCall.call(this, reg, e.args, e.span);
    }
    // Numbl's resolver only returns these dotted-route verdicts for
    // qualified names; if we got something else (or nothing) for a
    // dotted chain that's clearly not a class, fail with a clear
    // message rather than fall through to instance dispatch (which
    // would try to lower `pkg` as an Ident and crash).
    throw new UnsupportedConstruct(`unknown function '${qname}'`, e.span);
  }

  // Instance dispatch: the base lowers to a value and must be a
  // class instance.
  const base = this.lowerExpr(e.base);
  this.requireValueType(base, `method call '.${e.name}'`);
  if (!isClass(base.ty)) {
    throw new UnsupportedConstruct(
      `method call '.${e.name}' on a value of type ` +
        `${typeToString(base.ty)} is not supported (v1: classes only)`,
      e.span
    );
  }

  // Property-rooted indexing: `obj.field(args)` where `field` is a
  // class property (not a method). MATLAB semantics are "load the
  // field, then index it" — distinct from method dispatch. We
  // pre-hoist the field load to a fresh temp so the downstream
  // `IndexLoad` / `IndexSlice` has a real `Var` to anchor on (the
  // temp also gives `end`-keyword resolution a concrete `dims[k]` to
  // query). Dependent properties take a parallel path: the source
  // value is a getter Call instead of a `MemberLoad`.
  const reg = this.classReg(base.ty.className);
  const classProperties = base.ty.properties;
  const isProperty = classProperties.some(p => p.name === e.name);
  const isDependent = reg?.dependentProperties.has(e.name) ?? false;
  const isMethod = reg
    ? reg.methods.has(e.name) || reg.staticMethods.has(e.name)
    : false;
  if (isProperty && !isMethod) {
    return lowerMemberRootedIndex.call(this, base, e.name, e.args, e.span);
  }
  if (isDependent && reg !== undefined) {
    return lowerDependentPropertyIndex.call(
      this,
      base,
      reg,
      e.name,
      e.args,
      e.span
    );
  }
  const args = e.args.map(a => this.lowerExpr(a));
  for (const a of args) {
    this.requireValueType(a, `argument to method '${e.name}'`);
  }
  // Build the type tuple the resolver inspects: receiver + user
  // args. The resolver decides whether `e.name` is an instance or
  // static method of `base.ty.className` and toggles `stripInstance`
  // accordingly.
  const argTypesForResolve: Type[] = [base.ty, ...args.map(a => a.ty)];
  const target = this.workspace.resolve(
    e.name,
    argTypesForResolve,
    { ...this.callSite(), targetClassName: base.ty.className },
    e.span
  );
  if (target?.kind !== "classMethod") {
    throw new TypeError(
      `class '${base.ty.className}' has no method '${e.name}'`,
      e.span
    );
  }
  const targetReg = this.classReg(target.className);
  if (targetReg === undefined) {
    throw new UnsupportedConstruct(
      `internal: class '${target.className}' missing from workspace registry`,
      e.span
    );
  }
  const method = target.stripInstance
    ? targetReg.staticMethods.get(target.methodName)
    : targetReg.methods.get(target.methodName);
  if (method === undefined) {
    throw new TypeError(
      `class '${target.className}' has no ${target.stripInstance ? "static " : ""}method '${target.methodName}'`,
      e.span
    );
  }
  // Multi-output methods are accepted: single-output dispatch
  // (`x = obj.method(...)`) specializes the method at nargout=1
  // through `buildUserFunctionCall`, truncating the body's extra
  // outputs. `[a, b] = obj.method(...)` (multi-output instance
  // dispatch) remains rejected at `lowerMultiAssign`.
  const allArgs: IRExpr[] = target.stripInstance ? args : [base, ...args];
  return buildUserFunctionCall.call(
    this,
    method,
    allArgs,
    `${target.className}.${target.methodName}`,
    e.span,
    {
      specSource: classMethodSpecSource(target.className, target.methodName),
      // External method files own per-method-file local helpers, and
      // the resolver scopes them by the executing method's path. Use
      // the method FuncStmt's source file (the external file for an
      // `@<Cls>/<m>.m` method, the classdef file for an in-body
      // method) rather than always salting on the classdef file.
      definingFile: method.span.file ?? targetReg.file,
    }
  );
}

/** Lowers `obj.depProp(args)` where `depProp` is a `Dependent`
 *  property: call the getter to produce the value, hoist the result
 *  to a fresh temp, then run the args through the standard
 *  `lowerIndexLoad` / `lowerIndexSlice` path on the temp. v1
 *  rejects the corresponding write form (`obj.depProp(args) = rhs`)
 *  at `lowerAssignLValue` — splicing through the getter then setter
 *  is deferred. */
export function lowerDependentPropertyIndex(
  this: Lowerer,
  base: IRExpr,
  reg: import("./classDefs.js").ClassRegistration,
  propName: string,
  argExprs: ReadonlyArray<Expr>,
  span: Span
): IRExpr {
  const getter = reg.getters.get(propName);
  if (getter === undefined) {
    // `registerClassDef` guarantees every Dependent property has a
    // getter, so this should be unreachable.
    throw new UnsupportedConstruct(
      `internal: dependent property '${reg.className}.${propName}' has no getter`,
      span
    );
  }
  const getterCall = buildUserFunctionCall.call(
    this,
    getter,
    [base],
    `${reg.className}.get.${propName}`,
    span,
    {
      specSource: classMethodSpecSource(reg.className, `get.${propName}`),
      definingFile: getter.span.file ?? reg.file,
    }
  );
  const tempName = this.freshTempName();
  this.env.set(tempName, { cName: tempName, ty: getterCall.ty });
  this.pendingExprHoists.push({
    kind: "Assign",
    name: tempName,
    cName: tempName,
    ty: getterCall.ty,
    expr: getterCall,
    span,
  });
  if (argExprs.some(isSliceArg)) {
    return lowerIndexSlice.call(this, tempName, argExprs, span);
  }
  return lowerIndexLoad.call(this, tempName, argExprs, span);
}

/** Lowers `obj.field(args)` where `field` is a class property (not a
 *  method): load the field into a fresh temp, then run the args
 *  through the standard `lowerIndexLoad` / `lowerIndexSlice` path
 *  using the temp's name. The synthetic `Assign(temp = MemberLoad)`
 *  is queued on `pendingExprHoists` so `lowerStmt` prepends it to
 *  the emitted statement. */
export function lowerMemberRootedIndex(
  this: Lowerer,
  base: IRExpr,
  field: string,
  argExprs: ReadonlyArray<Expr>,
  span: Span
): IRExpr {
  const ft = fieldType(base.ty, field);
  if (ft === undefined) {
    throw new TypeError(
      `no field '${field}' on type ${typeToString(base.ty)}`,
      span
    );
  }
  // Only owned (multi-element / non-numeric owned) properties make
  // sense to index. Scalar real properties hit the
  // `requireMultiElement` check inside `resolveIndexBase`, so the
  // diagnostic still points at the original source span.
  const memberLoad: IRExpr = {
    kind: "MemberLoad",
    base,
    field,
    ty: ft,
    span,
  };
  const tempName = this.freshTempName();
  this.env.set(tempName, { cName: tempName, ty: ft });
  this.pendingExprHoists.push({
    kind: "Assign",
    name: tempName,
    cName: tempName,
    ty: ft,
    expr: memberLoad,
    span,
  });
  if (argExprs.some(isSliceArg)) {
    return lowerIndexSlice.call(this, tempName, argExprs, span);
  }
  return lowerIndexLoad.call(this, tempName, argExprs, span);
}

/** `ClassName.staticMethod(args)` — static method called via class
 *  name. The receiver is not present; arg types feed the resolver
 *  directly. */
export function lowerStaticMethodCall(
  this: Lowerer,
  className: string,
  e: Extract<Expr, { type: "MethodCall" }>,
  span: Span
): IRExpr {
  const args = e.args.map(a => this.lowerExpr(a));
  for (const a of args) {
    this.requireValueType(a, `argument to static method '${e.name}'`);
  }
  const argTypes = args.map(a => a.ty);
  const target = this.workspace.resolve(
    e.name,
    argTypes,
    { ...this.callSite(), targetClassName: className },
    span
  );
  if (target?.kind !== "classMethod") {
    throw new TypeError(
      `class '${className}' has no static method '${e.name}'`,
      span
    );
  }
  const reg = this.classReg(target.className);
  if (reg === undefined) {
    throw new UnsupportedConstruct(
      `internal: class '${target.className}' missing from workspace registry`,
      span
    );
  }
  // Numbl's `stripInstance` only fires on the `targetClassName`
  // branch when args[0] is a ClassInstance (i.e. the
  // `obj.staticMethod(...)` syntax). For the `ClassName.method(...)`
  // syntax we never prepend a receiver, so `stripInstance` is always
  // false — we instead look up `staticMethods` directly to
  // disambiguate static vs. instance.
  const method = reg.staticMethods.get(target.methodName);
  if (method === undefined) {
    if (reg.methods.has(target.methodName)) {
      throw new TypeError(
        `'${target.className}.${target.methodName}' is an instance method; ` +
          `call it on an instance (e.g. 'obj.${target.methodName}(...)')`,
        span
      );
    }
    throw new TypeError(
      `class '${target.className}' has no static method '${target.methodName}'`,
      span
    );
  }
  if (method.outputs.length >= 2) {
    throw new UnsupportedConstruct(
      `static class method '${target.className}.${target.methodName}' ` +
        `has ${method.outputs.length} outputs; multi-output methods are ` +
        `not supported yet`,
      span
    );
  }
  return buildUserFunctionCall.call(
    this,
    method,
    args,
    `${target.className}.${target.methodName}`,
    span,
    {
      specSource: classMethodSpecSource(target.className, target.methodName),
      definingFile: method.span.file ?? reg.file,
    }
  );
}
