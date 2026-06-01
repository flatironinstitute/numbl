/**
 * Function-handle lowering:
 *   - `@name`           → `lowerFuncHandle` (HandleLit with no captures)
 *   - `@(p1,...) body`  → `lowerAnonFunc`   (HandleLit with captures
 *                                            of the surrounding scope)
 *   - `h(args)`          → `dispatchHandleCall` (specialize + Call IR)
 *
 * Captures are deep-copied into the handle struct at the `@(...)`
 * site (MATLAB by-value snapshot). Owned-typed captures (tensor,
 * struct, class instance, another handle) ship per-shape
 * `_empty/_copy/_assign/_free` helpers so they participate in the
 * standard scope-exit-free / early-free lifecycle.
 */

import type { Expr, LValue, Span, Stmt } from "../parser/index.js";
import { TypeError, UnsupportedConstruct } from "./errors.js";
import type { IRExpr, IRStmt } from "./ir.js";
import { handleType, typeToString, VOID } from "./types.js";
import type { HandleCapture, HandleType, Type } from "./types.js";
import { getBuiltin } from "../builtins/index.js";
import type { EnvEntry, Lowerer } from "./lower.js";
import { buildUserFunctionCall, specializeUserFunction } from "./specialize.js";
import { buildMultiOutputSlots } from "./lowerMultiAssign.js";

type FuncStmt = Extract<Stmt, { type: "Function" }>;

/** `@name` — a named handle to a top-level user function. Builds the
 *  handle type, captures empty, and returns a HandleLit. Builtin
 *  targets (`@disp`, `@sin`) are rejected — mtoc2 v1 doesn't support
 *  them. Class methods (`@SomeClass.method`) aren't reachable via
 *  this AST (the parser emits a different shape for those). The
 *  workspace resolver finds local + cross-file function targets. */
export function lowerFuncHandle(
  this: Lowerer,
  e: Extract<Expr, { type: "FuncHandle" }>
): IRExpr {
  // Names shadowed by an in-scope variable: numbl forbids `@name` on
  // a non-function name (it's always a function reference, never a
  // var read).
  if (this.env.has(e.name)) {
    throw new TypeError(
      `function-handle target '@${e.name}' refers to an in-scope variable, ` +
        `not a function`,
      e.span
    );
  }
  if (getBuiltin(e.name)) {
    throw new UnsupportedConstruct(
      `builtin function handles (e.g. '@${e.name}') are not supported`,
      e.span
    );
  }
  // Pass `[]` argTypes — the resolver doesn't need them to decide a
  // function vs. classMethod for a bare `@name`; if the name is a
  // class instance method, the resolver returns its `classMethod`
  // verdict but we reject it (handles to class methods aren't
  // supported in v1).
  const target = this.workspace.resolve(e.name, [], this.callSite(), e.span);
  if (target?.kind !== "userFunction") {
    throw new UnsupportedConstruct(
      `unresolved function-handle target '@${e.name}'`,
      e.span
    );
  }
  // Use the source-level reference name (e.g. `pkg.foo`, `sq`) — NOT
  // `target.ast.name` (the basename) — so two handles to differently-
  // qualified functions with the same basename (`@pkg.foo` vs
  // `@other.foo`) produce distinct canonical types and don't unify.
  const ty = handleType(e.name, target.ast, []);
  return { kind: "HandleLit", captures: [], ty, span: e.span };
}

/** `@(p1, ..., pN) <body>` — an anonymous function. Detects every
 *  free Ident in the body that's bound in the enclosing scope (and
 *  not in the param list) as a capture, then synthesizes a top-level
 *  `function out = anon_<N>(p1, ..., pN, c1, ..., cM)` whose body
 *  assigns the source body expression to the synthesized output. The
 *  synthesized AST is parked on `handleTy.ast` so every call site
 *  routes through the same specialization cache used for
 *  user-declared functions.
 *
 *  Captures may be scalar real numeric, tensor, struct, class
 *  instance, or another handle — the handle's C struct ships with
 *  per-shape `_empty/_copy/_assign/_free` helpers (just like
 *  structs/classes), so owned-typed fields participate in the
 *  standard scope-exit-free / early-free lifecycle. String / Void /
 *  Unknown captures are rejected with `UnsupportedConstruct`. */
export function lowerAnonFunc(
  this: Lowerer,
  e: Extract<Expr, { type: "AnonFunc" }>
): IRExpr {
  const paramSet = new Set(e.params);
  const captureNames: string[] = [];
  const captureSet = new Set<string>();
  collectAnonCaptures(this.env, e.body, paramSet, captureNames, captureSet);

  const captures: HandleCapture[] = [];
  const captureValues: { name: string; value: IRExpr }[] = [];
  for (const cname of captureNames) {
    if (paramSet.has(cname)) {
      throw new UnsupportedConstruct(
        `anonymous-function parameter '${cname}' shadows a captured ` +
          `variable; rename the parameter`,
        e.span
      );
    }
    const entry = this.env.get(cname);
    if (entry === undefined) {
      throw new UnsupportedConstruct(
        `internal: capture '${cname}' lost between detection and lowering`,
        e.span
      );
    }
    if (
      entry.ty.kind !== "Numeric" &&
      entry.ty.kind !== "Struct" &&
      entry.ty.kind !== "Class" &&
      entry.ty.kind !== "Handle"
    ) {
      throw new UnsupportedConstruct(
        `anonymous function captures '${cname}' of unsupported type ` +
          `${typeToString(entry.ty)} (string / void / unknown captures ` +
          `are not supported)`,
        e.span
      );
    }
    captures.push({ name: cname, ty: entry.ty });
    captureValues.push({
      name: cname,
      value: {
        kind: "Var",
        name: cname,
        cName: entry.cName,
        ty: entry.ty,
        span: e.span,
      },
    });
  }

  const idx = this.anonCounter++;
  const synthName = `anon_${idx}`;
  const outName = `anonOut_${idx}`;
  const synthAst: FuncStmt = {
    type: "Function",
    name: synthName,
    functionId: synthName,
    params: [...e.params, ...captureNames],
    outputs: [outName],
    body: [
      {
        type: "Assign",
        name: outName,
        expr: e.body,
        suppressed: true,
        span: e.span,
      },
    ],
    argumentsBlocks: [],
    span: e.span,
  };
  // The synth AST is reachable only via `handleTy.ast` at call sites
  // (`dispatchHandleCall` passes it straight to
  // `specializeUserFunction`); it never needs name-based lookup, so
  // we don't park it anywhere external.
  const ty = handleType(synthName, synthAst, captures);
  return {
    kind: "HandleLit",
    captures: captureValues,
    ty,
    span: e.span,
  };
}

/** Dispatch `h(args)` where `h` is an in-scope handle variable.
 *  Reads the handle's `ast` off its type, lowers the user-supplied
 *  args, appends per-capture `HandleCaptureLoad` reads, specializes
 *  the underlying function on the combined arg-type tuple, and emits
 *  a direct call to the mangled name. */
export function dispatchHandleCall(
  this: Lowerer,
  handleName: string,
  handleEntry: EnvEntry,
  argExprs: Expr[],
  span: Span
): IRExpr {
  const handleTy = handleEntry.ty as HandleType;
  const userArgs = argExprs.map(a => this.lowerExpr(a));
  for (const a of userArgs) {
    this.requireValueType(a, `argument to handle '${handleName}'`);
  }
  const baseVar: Extract<IRExpr, { kind: "Var" }> = {
    kind: "Var",
    name: handleName,
    cName: handleEntry.cName,
    ty: handleTy,
    span,
  };
  const captureArgs: IRExpr[] = handleTy.captures.map(c => ({
    kind: "HandleCaptureLoad",
    base: baseVar,
    captureName: c.name,
    ty: c.ty,
    span,
  }));
  const allArgs = [...userArgs, ...captureArgs];
  // The handle's stored AST carries its own source span, which
  // identifies the file the function was defined in — that's the
  // right file to salt the spec key with. `buildUserFunctionCall`
  // specializes the target with nargout=1 (single-output ABI), which
  // is the correct truncation when the handle targets a multi-output
  // function but the call site only wants one output. The multi-
  // output call-site path lives in `dispatchHandleMultiAssign`.
  return buildUserFunctionCall.call(
    this,
    handleTy.ast,
    allArgs,
    handleTy.targetName,
    span,
    { definingFile: handleTy.ast.span.file }
  );
}

/** `[a, b, ...] = h(args)` where `h` is an in-scope handle variable.
 *  Mirrors `dispatchHandleCall`'s user-arg + capture-load wiring, then
 *  specializes the target at the caller's nargout. Emits a Call (when
 *  the spec truncates to 1 output) or a MultiAssignCall (N>=2 outputs);
 *  drop-all bare-statement use (`h(args);`) passes 0 lvalues and emits
 *  an ExprStmt. Returns the IR statement(s) to splice into the parent
 *  block. */
export function dispatchHandleMultiAssign(
  this: Lowerer,
  handleName: string,
  handleEntry: EnvEntry,
  argExprs: Expr[],
  lvalues: ReadonlyArray<LValue>,
  span: Span
): IRStmt | IRStmt[] {
  const handleTy = handleEntry.ty as HandleType;
  const userArgs = argExprs.map(a => this.lowerExpr(a));
  for (const a of userArgs) {
    this.requireValueType(a, `argument to handle '${handleName}'`);
  }
  const baseVar: Extract<IRExpr, { kind: "Var" }> = {
    kind: "Var",
    name: handleName,
    cName: handleEntry.cName,
    ty: handleTy,
    span,
  };
  const captureArgs: IRExpr[] = handleTy.captures.map(c => ({
    kind: "HandleCaptureLoad",
    base: baseVar,
    captureName: c.name,
    ty: c.ty,
    span,
  }));
  const allArgs = [...userArgs, ...captureArgs];
  // ANF args to scalar-or-Var — same discipline lowerMultiAssign uses
  // for owned-producing arg expressions. The captures themselves are
  // bare HandleCaptureLoad nodes that read from the handle struct;
  // these are owned values for tensor/struct/etc captures, so route
  // them through ANF too.
  const argHoists: IRStmt[] = [];
  const anfArgs = allArgs.map(a => this.anfRequireScalarOrVar(a, argHoists));
  const argTypes = allArgs.map(a => a.ty);
  if (lvalues.length > handleTy.ast.outputs.length) {
    throw new UnsupportedConstruct(
      `handle '${handleName}' targets '${handleTy.targetName}', which ` +
        `returns ${handleTy.ast.outputs.length} output(s); ${lvalues.length} ` +
        `were requested`,
      span
    );
  }
  if (handleTy.ast.outputs.length === 0 && lvalues.length > 0) {
    throw new UnsupportedConstruct(
      `handle '${handleName}' targets '${handleTy.targetName}', which has ` +
        `no outputs and cannot be assigned`,
      span
    );
  }
  const callNargout = lvalues.length;
  const spec = specializeUserFunction.call(
    this,
    handleTy.ast,
    argTypes,
    handleTy.targetName,
    handleTy.ast.span.file,
    undefined,
    callNargout,
    span
  );
  // 1-output spec (declared multi-output truncated, or declared
  // single-output): classic return-by-value ABI.
  if (spec.outputTypes.length === 1) {
    const callExpr: IRExpr = {
      kind: "Call",
      cName: spec.cName,
      name: handleTy.targetName,
      args: anfArgs,
      ty: spec.outputTypes[0] ?? { kind: "Unknown" },
      span,
    };
    const lv = lvalues[0];
    if (lv === undefined || lv.type !== "Var") {
      const stmt: IRStmt = { kind: "ExprStmt", expr: callExpr, span };
      return argHoists.length === 0 ? stmt : [...argHoists, stmt];
    }
    const stmt = this.recordAssignment(lv.name, callExpr, span);
    return argHoists.length === 0 ? stmt : [...argHoists, stmt];
  }
  // 0-output spec: drop-all bare-statement path on a 0-output target.
  if (spec.outputTypes.length === 0) {
    const callExpr: IRExpr = {
      kind: "Call",
      cName: spec.cName,
      name: handleTy.targetName,
      args: anfArgs,
      ty: VOID,
      span,
    };
    const stmt: IRStmt = { kind: "ExprStmt", expr: callExpr, span };
    return argHoists.length === 0 ? stmt : [...argHoists, stmt];
  }
  // N>=2 outputs.
  const slotTypes: Type[] = spec.outputTypes.map(t => t ?? { kind: "Unknown" });
  const outputs = buildMultiOutputSlots.call(
    this,
    handleTy.targetName,
    slotTypes,
    lvalues,
    i => `output '${handleTy.ast.outputs[i]}'`,
    span
  );
  const mac: IRStmt = {
    kind: "MultiAssignCall",
    cName: spec.cName,
    name: handleTy.targetName,
    args: anfArgs,
    outputs,
    span,
  };
  return argHoists.length === 0 ? mac : [...argHoists, mac];
}

/** Walk the body of an `@(...)` anonymous function and collect free
 *  Ident references that are bound in the OUTER scope (and not in
 *  the param list). The captured names become the handle's
 *  capture-fields, in source-encountered order — the order is fixed
 *  by the captures' position in `handleType.captures`, which drives
 *  the synthetic-function's parameter layout, so the call site can
 *  match positions.
 *
 *  Names that hit a registered builtin OR a top-level user function
 *  are NOT captures — they're function references resolved at the
 *  call site. A bare-Ident reference to such a name without a call
 *  is not yet meaningful in mtoc2 (only `@name` produces a handle),
 *  so we conservatively treat any in-scope variable as a capture.
 *
 *  Nested `@(...)` and `@name` inside the body do NOT contribute to
 *  the OUTER anonymous's captures — `@name` resolves at body-
 *  lowering time, and a nested `@(...)`'s captures are detected
 *  when that inner anonymous itself is lowered. */
function collectAnonCaptures(
  outerEnv: ReadonlyMap<string, EnvEntry>,
  e: Expr,
  params: ReadonlySet<string>,
  names: string[],
  seen: Set<string>
): void {
  const register = (name: string): void => {
    if (params.has(name)) return;
    if (seen.has(name)) return;
    if (getBuiltin(name)) return;
    if (!outerEnv.has(name)) return;
    seen.add(name);
    names.push(name);
  };
  switch (e.type) {
    case "Ident":
      register(e.name);
      return;
    case "Number":
      return;
    case "Binary":
      collectAnonCaptures(outerEnv, e.left, params, names, seen);
      collectAnonCaptures(outerEnv, e.right, params, names, seen);
      return;
    case "Unary":
      collectAnonCaptures(outerEnv, e.operand, params, names, seen);
      return;
    case "Range":
      collectAnonCaptures(outerEnv, e.start, params, names, seen);
      if (e.step) collectAnonCaptures(outerEnv, e.step, params, names, seen);
      collectAnonCaptures(outerEnv, e.end, params, names, seen);
      return;
    case "FuncCall":
      // A bare `name(args)` may refer to a captured handle variable
      // OR to a top-level user function / builtin. The same
      // `register` predicate filters: only bound-in-outer-scope
      // names become captures.
      register(e.name);
      for (const a of e.args)
        collectAnonCaptures(outerEnv, a, params, names, seen);
      return;
    case "Tensor":
      for (const row of e.rows) {
        for (const cell of row) {
          collectAnonCaptures(outerEnv, cell, params, names, seen);
        }
      }
      return;
    case "AnonFunc": {
      const nested = new Set(params);
      for (const p of e.params) nested.add(p);
      collectAnonCaptures(outerEnv, e.body, nested, names, seen);
      return;
    }
    case "FuncHandle":
      // `@name` resolves to a function reference at body-lowering
      // time — not a capture of the outer scope.
      return;
    case "Member":
      collectAnonCaptures(outerEnv, e.base, params, names, seen);
      return;
    case "Index":
      collectAnonCaptures(outerEnv, e.base, params, names, seen);
      for (const idx of e.indices) {
        collectAnonCaptures(outerEnv, idx, params, names, seen);
      }
      return;
    case "MethodCall":
      // `pkg.foo(args)` and `obj.method(args)` both parse as
      // MethodCall. The base chain may reference a captured variable
      // (the leftmost ident of `obj.method(...)`) or a workspace name
      // (the leftmost ident of `pkg.foo(...)`) — `register` filters
      // out workspace names because they aren't bound in outerEnv.
      collectAnonCaptures(outerEnv, e.base, params, names, seen);
      for (const a of e.args)
        collectAnonCaptures(outerEnv, a, params, names, seen);
      return;
    case "SuperMethodCall":
      for (const a of e.args)
        collectAnonCaptures(outerEnv, a, params, names, seen);
      return;
    default:
      // Other expression kinds remaining (literals, etc.) carry no
      // captures; any unsupported expression in the body fails when
      // the body itself is lowered.
      return;
  }
}
