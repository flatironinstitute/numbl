/**
 * Multi-assign statement lowering: `[a, b] = foo(x);` (with mix of
 * named lvalues, `~` discards, and trailing-omitted slots) and the
 * drop-all bare-statement variant `foo(x);` where `foo` has N≥2
 * outputs. Handles both user-function and multi-output builtin (e.g.
 * `[v, i] = sort(...)`) targets.
 *
 * 1-output declarations reached through this path (because the call
 * site has lvalues.length === 1) take the classic single-output ABI
 * — return-by-value via a `Call` IR node and an `Assign` / `ExprStmt`
 * wrapper. Only N≥2 outputs produce a `MultiAssignCall` IR node
 * (which carries the sret out-pointer ABI).
 */

import type { Expr, Span, Stmt, LValue } from "../parser/index.js";
import { UnsupportedConstruct } from "./errors.js";
import type { IRExpr, IRStmt } from "./ir.js";
import {
  type Type,
  VOID,
  isHandle,
  isMultiOutputSlotType,
  typeToString,
} from "./types.js";
import { getBuiltin } from "../builtins/index.js";
import type { Builtin } from "../builtins/registry.js";
import { withSpan } from "./errors.js";
import type { Lowerer } from "./lower.js";
import { tryExtractDottedName } from "./lower.js";
import { specializeUserFunction } from "./specialize.js";
import { dispatchHandleMultiAssign } from "./lowerHandle.js";
import { rewriteFevalToDirectCall } from "./lowerFuncCall.js";

export function lowerMultiAssign(
  this: Lowerer,
  s: Extract<Stmt, { type: "MultiAssign" }>
): IRStmt | IRStmt[] {
  // Allow two AST shapes on the RHS:
  //   - `FuncCall` for plain `foo(...)` / `helper(...)`
  //   - `MethodCall` whose base is a dotted name with a non-in-scope
  //     leftmost segment — i.e. a package-function call like
  //     `lege.exps(...)`. Same routing as `lowerMethodCall`'s package
  //     branch. (Instance methods stay 1-output-only; multi-output
  //     method dispatch is a separate followup.)
  let callName: string;
  let argExprs: Expr[];
  let callSpan: { file: string; start: number; end: number };
  let specSource: string | undefined;
  if (s.expr.type === "FuncCall") {
    // `[a, b] = feval(handle_or_name, args...)` — rewrite the RHS
    // to a direct call on the underlying name before any of the
    // downstream resolution logic runs. Only the unshadowed form
    // qualifies; a local `feval` variable would have taken precedence
    // via the (rejected-here) "in-scope variable on the right of
    // [...] = ..." path below.
    let rhs = s.expr;
    if (rhs.name === "feval" && this.env.get("feval") === undefined) {
      rhs = rewriteFevalToDirectCall(rhs);
    }
    callName = rhs.name;
    argExprs = rhs.args;
    callSpan = rhs.span;
    specSource = undefined; // default: decl.name
  } else if (s.expr.type === "MethodCall") {
    const dottedBase = tryExtractDottedName(s.expr.base);
    if (dottedBase === null || this.env.has(dottedBase.split(".")[0])) {
      throw new UnsupportedConstruct(
        `multi-assign right-hand side must be a user-function or ` +
          `package-function call (got instance-method dispatch, not ` +
          `yet supported in '[...] = ...')`,
        s.span
      );
    }
    callName = `${dottedBase}.${s.expr.name}`;
    argExprs = s.expr.args;
    callSpan = s.expr.span;
    // Salt the spec key with the qualified name so a packaged `foo`
    // doesn't share its specialization slot with a workspace-level
    // `foo` of the same shape.
    specSource = callName;
  } else {
    throw new UnsupportedConstruct(
      `multi-assign right-hand side must be a user-function call`,
      s.span
    );
  }
  const fc = { name: callName, args: argExprs, span: callSpan };
  // Expand a single `[out{1:nout}]` style brace-with-range lvalue
  // into N separate `out{k}` IndexCell lvalues — comma-list LHS
  // expansion (phase C of cells_plan.md). The range bounds must be
  // statically known at lowering so the arity is fixed.
  const expandedLvalues = expandRangeCellLvalues.call(this, s.lvalues, s.span);

  // Validate lvalues up-front. Var / Ignore are the direct sret
  // targets; Member, Index, and IndexCell lvalues are accepted via
  // a temp-substitute pass that lands the call into freshly-named
  // locals and then runs a synthesized per-slot AssignLValue through
  // the regular write paths. MemberDynamic is out of scope.
  for (const lv of expandedLvalues) {
    if (
      lv.type !== "Var" &&
      lv.type !== "Ignore" &&
      lv.type !== "Member" &&
      lv.type !== "Index" &&
      lv.type !== "IndexCell"
    ) {
      throw new UnsupportedConstruct(
        `multi-assign lvalue must be a simple identifier, '~' ignore, ` +
          `a member access, an indexed write, or a brace-indexed cell ` +
          `write (got '${lv.type}')`,
        s.span
      );
    }
  }
  // Switch over to the expanded lvalues for the rest of this lower.
  s = { ...s, lvalues: expandedLvalues };
  // Handle multi-output via in-scope handle variable: route to the
  // handle dispatch path (it specializes the underlying target at the
  // caller's nargout and emits the same Call / MultiAssignCall IR
  // shape user-function calls produce).
  if (s.expr.type === "FuncCall") {
    const envEntry = this.env.get(callName);
    if (envEntry !== undefined && isHandle(envEntry.ty)) {
      return dispatchHandleMultiAssign.call(
        this,
        callName,
        envEntry,
        argExprs,
        s.lvalues,
        s.span
      );
    }
  }
  // Reject in-scope variable names and class names — only user
  // functions (or handles, handled above) can sit on the right of
  // `[...] = ...` in v1.
  if (this.env.get(callName) !== undefined) {
    throw new UnsupportedConstruct(
      `multi-assign of '${callName}': name resolves to an in-scope ` +
        `variable, not a function`,
      s.span
    );
  }
  if (this.workspace.isClass(callName)) {
    throw new UnsupportedConstruct(
      `multi-assign of '${callName}': class constructors have a single ` +
        `output`,
      s.span
    );
  }
  // Lower the args. Same pattern as `lowerFuncCall`.
  const args = fc.args.map(a => this.lowerExpr(a));
  for (const a of args) {
    this.requireValueType(a, `argument to '${callName}'`);
  }
  const argTypes = args.map(a => a.ty);
  // ANF each arg to scalar-or-Var — same discipline as `anfChildren`
  // for a regular `Call` node. Without this, an owned-producing arg
  // like `times_ts(ones_nd(...), 2.0)` would leave the inner
  // `ones_nd` unfreed (the outer helper doesn't consume its tensor
  // arg). For a user-function call (Call or MultiAssignCall), the
  // callee owns each arg, so the OUTER producer is fine — only the
  // grandchild needs hoisting — but mirroring Call's discipline
  // (hoist top-level owned non-Vars too) is simpler and the temp
  // gets early-freed after the call so cost is nil.
  const argHoists: IRStmt[] = [];
  const anfArgs = args.map(a => this.anfRequireScalarOrVar(a, argHoists));
  const target = this.workspace.resolve(
    callName,
    argTypes,
    this.callSite(),
    fc.span
  );

  // Builtin multi-output path. A builtin opts into `[...] = f(x)` by
  // accepting an `nargout > 1` in its `transfer` hook. Numbl's
  // resolver returns `kind: "builtin"`; we re-fetch from mtoc2's own
  // registry and route through the same `MultiAssignCall` IR shape
  // user functions use. Single-output `b = f(x)` still flows through
  // `lowerFuncCall` → `transfer`/`emit` — this hook only fires for
  // true multi-output uses.
  if (target?.kind === "builtin") {
    const builtin = getBuiltin(callName);
    if (builtin !== undefined) {
      return buildBuiltinMultiAssign.call(
        this,
        callName,
        builtin,
        anfArgs,
        argTypes,
        argHoists,
        s
      );
    }
  }

  if (target?.kind !== "userFunction") {
    throw new UnsupportedConstruct(
      `multi-assign of '${callName}': only user-defined functions can ` +
        `appear on the right of '[...] = ...' (or as a bare multi-output ` +
        `statement)`,
      s.span
    );
  }
  const fnAst = target.ast;
  const fnFile = target.file;
  if (s.lvalues.length > fnAst.outputs.length) {
    throw new UnsupportedConstruct(
      `function '${callName}' returns ${fnAst.outputs.length} output(s) ` +
        `but ${s.lvalues.length} were requested`,
      s.span
    );
  }
  if (fnAst.outputs.length === 0 && s.lvalues.length > 0) {
    throw new UnsupportedConstruct(
      `function '${callName}' has no outputs and cannot be assigned`,
      s.span
    );
  }
  // Caller's requested `nargout`: count of lvalues (0 for the
  // bare-drop-all path that `lowerExprStmt` routes here).
  const callNargout = s.lvalues.length;
  const spec = specializeUserFunction.call(
    this,
    fnAst,
    argTypes,
    specSource,
    fnFile,
    undefined,
    callNargout,
    s.span
  );

  // 1-output spec: route to the classic single-output ABI (return-
  // by-value) so we don't introduce a redundant sret path. Note this
  // is `spec.outputTypes.length`, not `fnAst.outputs.length` — a
  // multi-output declared callee specialized with nargout=1 (because
  // the call site only requested one output) truncates its spec to a
  // single output and emits the return-by-value shape, even though
  // `fnAst.outputs.length > 1`.
  if (spec.outputTypes.length === 1) {
    const callExpr: IRExpr = {
      kind: "Call",
      cName: spec.cName,
      name: callName,
      args: anfArgs,
      ty: spec.outputTypes[0] ?? { kind: "Unknown" },
      span: s.span,
    };
    // 0 or 1 lvalues: lvalues.length === 0 → drop-all reached via the
    // ExprStmt routing (caller passed empty lvalues for a bare
    // 1-output call); lvalues.length === 1 → either named, `~`,
    // or a Member / Index lvalue that lands the call into a temp
    // and then runs a synthesized AssignLValue to move into the slot.
    const lv = s.lvalues[0];
    if (lv === undefined || lv.type === "Ignore") {
      const stmt: IRStmt = { kind: "ExprStmt", expr: callExpr, span: s.span };
      return argHoists.length === 0 ? stmt : [...argHoists, stmt];
    }
    if (lv.type === "Var") {
      const stmt = this.recordAssignment(lv.name, callExpr, s.span);
      return argHoists.length === 0 ? stmt : [...argHoists, stmt];
    }
    // Member / Index lvalue: land the call into a temp, then run a
    // synthesized AssignLValue through the regular write path.
    const sub = substituteNonVarLvalues.call(
      this,
      [lv],
      [spec.outputTypes[0] ?? { kind: "Unknown" }],
      s.span
    );
    const tempLv = sub.effectiveLvalues[0] as Extract<LValue, { type: "Var" }>;
    const assignStmt = this.recordAssignment(tempLv.name, callExpr, s.span);
    const postStmts = lowerPostStores.call(this, sub.postStores);
    const head: IRStmt[] =
      argHoists.length === 0 ? [assignStmt] : [...argHoists, assignStmt];
    return [...head, ...postStmts];
  }

  // 0-output spec with no lvalues: bare-statement routing path
  // (`f(...);` with f declared 0-output, or a multi-output declared f
  // called bare which truncates the spec to 0 outputs and emits a
  // void function). Pass through as ExprStmt(Call) with Void type —
  // the same shape `lowerFuncCall` would produce for any 0-output
  // bare call.
  if (spec.outputTypes.length === 0) {
    const callExpr: IRExpr = {
      kind: "Call",
      cName: spec.cName,
      name: callName,
      args: anfArgs,
      ty: VOID,
      span: s.span,
    };
    const stmt: IRStmt = { kind: "ExprStmt", expr: callExpr, span: s.span };
    return argHoists.length === 0 ? stmt : [...argHoists, stmt];
  }

  // N≥2 outputs. Build a MultiAssignCall via the shared slot-binding
  // helper so the user-function and builtin paths share one place
  // for the slot-type validation and the `recordAssignment`
  // discipline. Non-Var lvalues are substituted with temps; the
  // synthesized AssignLValue per slot runs after the MAC.
  const slotTypes: Type[] = spec.outputTypes.map(t => t ?? { kind: "Unknown" });
  const sub = substituteNonVarLvalues.call(this, s.lvalues, slotTypes, s.span);
  const outputs = buildMultiOutputSlots.call(
    this,
    callName,
    slotTypes,
    sub.effectiveLvalues,
    i => `output '${fnAst.outputs[i]}'`,
    s.span
  );
  const mac: IRStmt = {
    kind: "MultiAssignCall",
    cName: spec.cName,
    name: callName,
    args: anfArgs,
    outputs,
    span: s.span,
  };
  const postStmts = lowerPostStores.call(this, sub.postStores);
  const head: IRStmt[] = argHoists.length === 0 ? [mac] : [...argHoists, mac];
  return [...head, ...postStmts];
}

/** Substitute non-Var lvalues with synthetic `Var` lvalues that
 *  reference fresh temps, and return the synthesized `AssignLValue`
 *  AST statements that move each temp into the original Member /
 *  Index lvalue.
 *
 *  Lets multi-output paths land call results into freshly-named
 *  temps, then run a normal per-slot write through
 *  `lowerAssignLValue` — reusing every existing type-check, ANF
 *  rule, and env refresh without duplicating the Member / Index
 *  write logic here.
 *
 *  The temp is registered in env with the slot's type so the
 *  subsequent `lowerAssignLValue` lowering sees the right type
 *  when it lowers the synthetic `Ident` RHS. Var / Ignore slots
 *  pass through unchanged. */
function substituteNonVarLvalues(
  this: Lowerer,
  lvalues: ReadonlyArray<LValue>,
  slotTypes: ReadonlyArray<Type>,
  span: Span
): { effectiveLvalues: LValue[]; postStores: Stmt[] } {
  const effective: LValue[] = [];
  const postStores: Stmt[] = [];
  for (let i = 0; i < lvalues.length; i++) {
    const lv = lvalues[i];
    if (lv.type === "Var" || lv.type === "Ignore") {
      effective.push(lv);
      continue;
    }
    const tempName = this.freshTempName();
    const slotTy = slotTypes[i] ?? { kind: "Unknown" as const };
    this.env.set(tempName, { cName: tempName, ty: slotTy });
    effective.push({ type: "Var", name: tempName });
    const tempIdent: Expr = { type: "Ident", name: tempName, span };
    postStores.push({
      type: "AssignLValue",
      lvalue: lv,
      expr: tempIdent,
      suppressed: true,
      span,
    });
  }
  return { effectiveLvalues: effective, postStores };
}

/** Lower each synthesized post-store AssignLValue through the
 *  regular write path and flatten the result. */
function lowerPostStores(this: Lowerer, postStores: Stmt[]): IRStmt[] {
  const out: IRStmt[] = [];
  for (const s of postStores) {
    if (s.type !== "AssignLValue") continue;
    const lowered = this.lowerAssignLValue(s);
    if (Array.isArray(lowered)) out.push(...lowered);
    else out.push(lowered);
  }
  return out;
}

/** Validate each multi-output slot type, register the lvalue
 *  bindings in env (or mark them as discarded), and return the
 *  `MultiAssignCall.outputs` list. Single chokepoint for the
 *  user-function and builtin multi-output paths.
 *
 *  `slotLabel(i)` returns a per-slot description for the error
 *  message when a slot's type isn't supported — e.g. `output 'a'`
 *  for declared user-function outputs or `output slot 1` for builtin
 *  slots that have no source-level name. */
export function buildMultiOutputSlots(
  this: Lowerer,
  callName: string,
  slotTypes: ReadonlyArray<Type>,
  lvalues: ReadonlyArray<LValue>,
  slotLabel: (i: number) => string,
  span: Span
): { ty: Type; binding: { name: string; cName: string } | null }[] {
  const outputs: {
    ty: Type;
    binding: { name: string; cName: string } | null;
  }[] = [];
  for (let i = 0; i < slotTypes.length; i++) {
    const slotTy = slotTypes[i];
    // Accept scalar real numeric or any owned type (tensor / struct /
    // class / handle / Char / String). Owned slots transfer ownership
    // via the kind's `_assign` helper at the callee's sret write
    // site; scalar slots use a bare struct copy. Void / Unknown stay
    // rejected — no C representation that fits the sret slot.
    if (!isMultiOutputSlotType(slotTy)) {
      throw new UnsupportedConstruct(
        `multi-output call '${callName}': ${slotLabel(i)} has type ` +
          `${typeToString(slotTy)}; this type isn't supported in a ` +
          `multi-output slot`,
        span
      );
    }
    const lv: LValue | undefined = lvalues[i];
    if (lv === undefined || lv.type !== "Var") {
      outputs.push({ ty: slotTy, binding: null });
      continue;
    }
    // Named slot. We reuse `recordAssignment` for its side effects
    // (env update, cName allocation). The synthetic Var expression it
    // builds the Assign around is discarded — we only consume the
    // returned cName.
    const synthRhs: IRExpr = {
      kind: "Var",
      name: lv.name,
      cName: "<placeholder>",
      ty: slotTy,
      span,
    };
    const rec = this.recordAssignment(lv.name, synthRhs, span);
    outputs.push({
      ty: slotTy,
      binding: { name: lv.name, cName: rec.cName },
    });
  }
  return outputs;
}

/** Routes a `[v, i, ...] = builtin(args)` call through
 *  `MultiAssignCall`. The builtin's `transfer(argTypes, nargout)`
 *  validates arity / arg types / output count and returns one type
 *  per requested output; the builtin's `emit` hook (called at codegen
 *  time) produces the full C call string including out-pointer args.
 *  Reuses the same output-slot bookkeeping as the user-function path
 *  (named slots go through `recordAssignment` for env / cName setup;
 *  `~` / trailing-omitted slots become discard temps in the
 *  emitter). */
function buildBuiltinMultiAssign(
  this: Lowerer,
  callName: string,
  builtin: Builtin,
  anfArgs: IRExpr[],
  argTypes: Type[],
  argHoists: IRStmt[],
  s: Extract<Stmt, { type: "MultiAssign" }>
): IRStmt | IRStmt[] {
  const nargout = s.lvalues.length;
  const outTys = withSpan(s.span, () => builtin.transfer(argTypes, nargout));
  if (outTys.length !== nargout) {
    // Builtin author contract violation. Surfaces with the call-site
    // span so the bad builtin name + invocation are easy to locate.
    throw new UnsupportedConstruct(
      `internal: builtin '${callName}' transfer returned ` +
        `${outTys.length} type(s) for nargout=${nargout}`,
      s.span
    );
  }
  // Non-Var lvalues land into temps, then a synthesized
  // AssignLValue per slot runs through the regular write path.
  const sub = substituteNonVarLvalues.call(this, s.lvalues, outTys, s.span);
  const outputs = buildMultiOutputSlots.call(
    this,
    callName,
    outTys,
    sub.effectiveLvalues,
    i => `output slot ${i + 1}`,
    s.span
  );
  const mac: IRStmt = {
    kind: "MultiAssignCall",
    // cName is unused for builtin MAC at codegen — the builtin's emit
    // hook produces the full call string. Keep the source-level name
    // here for prettyIR / debugging.
    cName: callName,
    name: callName,
    isBuiltin: true,
    args: anfArgs,
    outputs,
    span: s.span,
  };
  const postStmts = lowerPostStores.call(this, sub.postStores);
  const head: IRStmt[] = argHoists.length === 0 ? [mac] : [...argHoists, mac];
  return [...head, ...postStmts];
}

/** Expand `[out{1:nout}]` / `[out{start:end}]` style multi-assign
 *  LHS into N separate `out{k}` IndexCell lvalues — the LHS shape
 *  of cell comma-list expansion (phase C of docs/cells_plan.md).
 *
 *  The expansion only fires when:
 *    - the LHS list has exactly one entry, AND
 *    - that entry is an `IndexCell` with exactly one index, AND
 *    - the index is a `Range` whose bounds resolve statically.
 *
 *  Everything else passes through unchanged. */
function expandRangeCellLvalues(
  this: Lowerer,
  lvalues: ReadonlyArray<LValue>,
  span: Span
): LValue[] {
  if (lvalues.length !== 1) return lvalues.slice();
  const lv = lvalues[0];
  if (lv.type !== "IndexCell") return lvalues.slice();
  if (lv.indices.length !== 1) return lvalues.slice();
  const idx = lv.indices[0];
  if (idx.type !== "Range") return lvalues.slice();
  const startTy = this.lowerExpr(idx.start).ty;
  const endTy = this.lowerExpr(idx.end).ty;
  const stepTy = idx.step !== null ? this.lowerExpr(idx.step).ty : null;
  const startExact =
    startTy.kind === "Numeric" && typeof startTy.exact === "number"
      ? startTy.exact
      : null;
  const endExact =
    endTy.kind === "Numeric" && typeof endTy.exact === "number"
      ? endTy.exact
      : null;
  const stepExact =
    stepTy === null
      ? 1
      : stepTy.kind === "Numeric" && typeof stepTy.exact === "number"
        ? stepTy.exact
        : null;
  if (startExact === null || endExact === null || stepExact === null) {
    throw new UnsupportedConstruct(
      `multi-assign LHS '[<cell>{<range>}]': the range bounds must be ` +
        `statically known at compile time (got non-exact bounds)`,
      span
    );
  }
  if (
    !Number.isInteger(startExact) ||
    !Number.isInteger(endExact) ||
    !Number.isInteger(stepExact) ||
    stepExact === 0
  ) {
    throw new UnsupportedConstruct(
      `multi-assign LHS '[<cell>{<range>}]': range bounds must be ` +
        `non-zero integers (got ${startExact}:${stepExact}:${endExact})`,
      span
    );
  }
  const slotIdxs: number[] = [];
  if (stepExact > 0) {
    for (let v = startExact; v <= endExact; v += stepExact) slotIdxs.push(v);
  } else {
    for (let v = startExact; v >= endExact; v += stepExact) slotIdxs.push(v);
  }
  if (slotIdxs.length === 0) {
    throw new UnsupportedConstruct(
      `multi-assign LHS '[<cell>{<range>}]': empty range (no slots to assign)`,
      span
    );
  }
  const out: LValue[] = slotIdxs.map(k => ({
    type: "IndexCell",
    base: lv.base,
    indices: [{ type: "Number", value: String(k), span }],
  }));
  return out;
}
