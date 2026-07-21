/**
 * mtoc2 lowerer. Walks the numbl AST, threads a type env (with exact-
 * value tracking), and produces typed IR. The same pass:
 *   - resolves every function-call site through the shared
 *     `Workspace` (which delegates to numbl's `resolveFunction`),
 *     so MATLAB precedence rules (local > workspace > builtin, plus
 *     class-method dispatch on `obj.method(args)` and
 *     `ClassName.method(args)`) are inherited from numbl wholesale;
 *   - allocates per-call function specializations (mangled by the
 *     FNV-1a hash of the canonicalized arg-type tuple, salted by
 *     the defining file so two files defining a subfunction with
 *     the same name get distinct C names);
 *   - merges types across control-flow joins;
 *   - widens variables assigned inside loop bodies (strips exact)
 *     before lowering the body, so the one-pass lowering doesn't
 *     bake the entry-state value into the emitted code.
 *
 * Exact-value tracking threads through the type system (builtin
 * transfer fns still compute exact results), but the only place the
 * lowerer substitutes a literal for a computation is the `if` /
 * `elseif` condition — when the cond's type has a `number` exact, the
 * branch is statically taken or dropped. Arithmetic, comparisons,
 * builtin calls, and Ident reads all emit runtime IR even when their
 * exact value is known.
 *
 * MVP scope: scalar real double + arithmetic + comparisons + disp +
 * if/while/for + user functions (0 or 1 outputs; 0-output calls return
 * `Void` and are only valid as the expression of an `ExprStmt`) +
 * classes (instance methods, static methods, constructors), with
 * resolution against sibling `.m` files in the workspace via numbl's
 * vendored resolver. Anything outside that throws
 * `UnsupportedConstruct` with a span.
 */

import type {
  AbstractSyntaxTree,
  Expr,
  LValue,
  Stmt,
  Span,
} from "../parser/index.js";
import { BinaryOperation, UnaryOperation } from "../parser/index.js";
import { offsetToLineCol } from "../parser/sourceLoc.js";
import {
  tryExtractDottedName,
  unwindMemberLvalue,
} from "../parser/astUtils.js";
import { UnsupportedConstruct, TypeError } from "./errors.js";
import {
  type Type,
  type Sign,
  type NumericType,
  scalarDouble,
  scalarLogical,
  scalarComplex,
  tensorDouble,
  signFromExactArray,
  signFromNumber,
  unifySign,
  isScalarRealNumeric,
  isMultiElement,
  isNumeric,
  isVoid,
  isOwned,
  fieldType,
  classMethodSpecSource,
  shapeNumel,
  typeToString,
  VOID,
  unify,
  storageEquivalent,
  stripExactFromEnv,
  widenAfterIndexedWrite,
  widenMemberLeafAfterIndexedWrite,
  withPathTypeUpdated,
  withoutExact,
} from "./types.js";
import type { ClassRegistration } from "./classDefs.js";
import type { Workspace } from "../workspace/workspace.js";
import type { CallSite } from "../numbl/index.js";

/** Hook for the `%!numbl:printtype` directive's compile-time output.
 *  Defaults to `console.error` (so emitted lines go to stderr and
 *  don't mix with the cross-runner's stdout comparison). Tests
 *  replace this with a capturing function. */
export let printTypeSink: (line: string) => void = line => console.error(line);
export function setPrintTypeSink(sink: (line: string) => void): void {
  printTypeSink = sink;
}
export function resetPrintTypeSink(): void {
  printTypeSink = line => console.error(line);
}
import {
  type Assign,
  type IRExpr,
  type IRStmt,
  type IRFunc,
  type IRProgram,
} from "./ir.js";
import { rangeCountFromExactEnds } from "./rangeCount.js";
import { columnMajorOffsetFromIndices } from "./indexFold.js";
import {
  getBuiltin,
  binaryOpBuiltin,
  unaryOpBuiltin,
} from "../builtins/index.js";
import { withSpan } from "./errors.js";
import { exactComplex } from "../builtins/defs/_shared.js";
import { isSliceArg } from "./indexResolve.js";
import { lowerTensorLit } from "./lowerTensorLit.js";
import { lowerCellLit } from "./lowerCellLit.js";
import { lowerCellIndexLoad, lowerCellIndexStore } from "./lowerCellIndex.js";
import { lowerMultiAssign } from "./lowerMultiAssign.js";
import { lowerAnonFunc, lowerFuncHandle } from "./lowerHandle.js";
import { lowerMethodCall } from "./lowerMethodCall.js";
import { buildUserFunctionCall } from "./specialize.js";
import { lowerFuncCall } from "./lowerFuncCall.js";
import { lowerIndexStore } from "./lowerIndexStore.js";
import { lowerIndexSliceStore } from "./lowerIndexSliceStore.js";

export interface EnvEntry {
  cName: string;
  ty: Type;
  /** True when the variable is assigned on only SOME arms of an
   *  enclosing `if` (not all), so a read may hit a never-assigned value.
   *  MATLAB raises "Undefined ..." at runtime on the unassigned path; the
   *  JIT predeclares the local with a default and would silently read it,
   *  so a read of a flagged var (or returning it as an output) declines
   *  to the interpreter, which performs the runtime definite-assignment
   *  check. */
  maybeUnassigned?: boolean;
}

/** Conservative purity check used by the short-circuit fold: returns true only
 *  for IR expressions that provably have no side effects (no calls, no stores,
 *  no indexing that could throw), so they are safe to drop. Anything else
 *  (calls, loads, builds) is assumed to potentially have side effects. */
function isSideEffectFreeExpr(e: IRExpr): boolean {
  switch (e.kind) {
    case "NumLit":
    case "ImagLit":
    case "StringLit":
    case "Var":
      return true;
    case "Binary":
      return isSideEffectFreeExpr(e.left) && isSideEffectFreeExpr(e.right);
    case "Unary":
      return isSideEffectFreeExpr(e.operand);
    default:
      return false;
  }
}

export class Lowerer {
  // The members below are formally `package-internal` — used by the
  // extracted helpers in `./lower*.ts` and `./specialize.ts` via the
  // same `this: Lowerer` pattern as `lowerIndexLoad` etc. Treat them
  // as private to the lowering module; external callers should go
  // through `lowerProgram` / `lowerExpr`.
  env: Map<string, EnvEntry> = new Map();
  specializations: Map<string, IRFunc> = new Map();
  /** Stack of contexts for resolving the `end` keyword inside an
   *  index slot. Pushed by the index-lowering helpers (lowerIndexLoad
   *  / lowerIndexStore / lowerIndexSlice / lowerIndexSliceStore)
   *  around each slot they lower; consumed by the `EndKeyword` arm of
   *  `lowerExpr`. `axis === "linear"` flags the single-slot form
   *  whose `end` resolves to `numel(base)`. Outside an index slot the
   *  stack is empty and `end` raises `UnsupportedConstruct`. */
  endStack: Array<{
    baseCName: string;
    baseTy: NumericType;
    axis: number | "linear";
  }> = [];
  /** Monotonic counter for synthesizing `_mtoc2_t1`, `_mtoc2_t2`, ...
   *  hoist-temp names. Reset per function specialization. */
  tempCounter: number = 0;
  /** Monotonic counter for the synthetic suffix used when an
   *  incompatible reassignment is split into a fresh binding
   *  (`_mtoc2_<cName>__v<N>`). See `recordAssignment`. Per-Lowerer, so
   *  each function specialization / script scope numbers independently. */
  splitCounter: number = 0;
  /** Nesting depth inside `if` / `while` / `for` bodies. Bumped by
   *  `withControlDepth` around each body lowering. An incompatible
   *  reassignment can only be *split* at depth 0 (straight-line scope);
   *  inside a branch or loop the env merge can't reconcile two storage
   *  categories under one name, so it throws instead. */
  controlDepth: number = 0;
  /** Expression-level hoist statements queued by sub-lowerings that
   *  can't pass a `hoists` array up the IR-expression return chain.
   *  Used today by member-rooted indexing (`obj.field(args)`) to push a
   *  fresh `Assign(temp = MemberLoad)` so the downstream `IndexLoad` /
   *  `IndexSlice` has a real `Var` to anchor `end`-keyword resolution
   *  on. `lowerStmt` drains this list around every statement boundary
   *  and prepends the hoists to whatever the inner lowering emitted. */
  pendingExprHoists: IRStmt[] = [];
  /** Monotonic counter for synthesizing anonymous-function names
   *  (`anon_0`, `anon_1`, ...). Shared across the whole program so two
   *  textually distinct `@(...)` expressions get distinct identities. */
  anonCounter: number = 0;
  /** Source file the lowerer is currently inside. Defaults to the
   *  workspace's main file at construction; pushed/popped by
   *  `specializeUserFunction` so a call from inside `helper.m`'s
   *  subfunction reports the right file in its `CallSite`. */
  currentFile: string;
  /** Per-specialization `nargin` / `nargout` values. Pushed by
   *  `specializeUserFunction` before lowering a function body; popped
   *  after. Read by the matching identifier arms of `lowerIdent`. Empty
   *  at top level — MATLAB rejects `nargin` / `nargout` outside a
   *  function body, which we mirror by leaving the reference
   *  unresolved (which falls through to the "undefined" error). */
  callFrameStack: { nargin: number; nargout: number }[] = [];
  /** Spec keys whose still-empty placeholder was consumed via a cache
   *  hit while lowering some other (or the same) spec. Tracked so
   *  `specializeUserFunction` knows when a recursive self-call used a
   *  stale `outputTypes` seed and a second pass is needed to refine
   *  the recursive-call IR's `ty`. The set is populated by
   *  `specializeUserFunction`'s cache-hit branch and consumed by the
   *  same function after body lowering. */
  recursiveSpecsConsumed: Set<string> = new Set();
  /** Number of distinct specializations created so far per
   *  `<sourceName>|<definingFile>`. Read by
   *  `specializeUserFunction`'s cache-miss branch; when the count for
   *  any one source-level function crosses `MAX_SPECS_PER_FUNCTION`,
   *  translation aborts with a clear error rather than letting the
   *  spec map (and the emitted C / JS) grow without bound. Exact-
   *  value tracking shards the cache by arg value, so a tight call
   *  with hundreds of distinct literal args (or value-keyed loop
   *  bodies) is the typical trigger. */
  specCountPerSource: Map<string, number> = new Map();

  constructor(public workspace: Workspace) {
    this.currentFile = workspace.mainFile;
  }

  /** Built `CallSite` for the vendored numbl resolver. New
   *  resolver-relevant fields added later (`className`, `methodName`,
   *  ...) only need to be threaded here. */
  callSite(): CallSite {
    return { file: this.currentFile };
  }

  /** Public env lookup used by the index-lowering helpers. Returns
   *  the env entry (cName + ty) if a binding exists in the current
   *  scope, or undefined otherwise. */
  envLookup(name: string): EnvEntry | undefined {
    return this.env.get(name);
  }

  /** Run `fn` with `controlDepth` incremented; restored on exit. Used by
   *  `lowerIf` / `lowerWhile` / `lowerFor` to mark "we're inside a
   *  branch or loop body" so an incompatible reassignment throws rather
   *  than splitting (see `recordAssignment`). */
  withControlDepth<T>(fn: () => T): T {
    this.controlDepth++;
    try {
      return fn();
    } finally {
      this.controlDepth--;
    }
  }

  /** Look up a registered class (workspace or local) by name.
   *  Re-throws the saved `UnsupportedConstruct` if numbl knew the
   *  class but mtoc2's `registerClassDef` rejected it in the JIT
   *  bridge's `fromExistingContext` mode — surfacing the error at
   *  the first actual use rather than at workspace finalize. */
  classReg(name: string): ClassRegistration | undefined {
    return this.workspace.requireClass(name);
  }

  lowerProgram(ast: AbstractSyntaxTree): IRProgram {
    // Build numbl's function index + mtoc2's class registry. The
    // Workspace has already had every file added; `finalize()` is
    // idempotent so a double-call from a caller that explicitly
    // finalized first is harmless.
    this.workspace.finalize();
    // Lower top-level statements (Function / ClassDef stmts return
    // null and are filtered out). The active file's top-level body
    // is the script entry; sibling-file bodies (workspace functions)
    // are lowered lazily by `specializeUserFunction` from call sites.
    const topLevelStmts = this.lowerStmts(ast.body);
    return { topLevelStmts, functions: this.specializations };
  }

  // ── Statement lowering ────────────────────────────────────────────────

  lowerStmts(stmts: Stmt[]): IRStmt[] {
    const out: IRStmt[] = [];
    for (const s of stmts) {
      const lowered = this.lowerStmt(s);
      if (lowered === null) continue;
      if (Array.isArray(lowered)) out.push(...lowered);
      else out.push(lowered);
      const tail = out[out.length - 1];
      if (
        tail !== undefined &&
        (tail.kind === "ReturnFromFunction" ||
          tail.kind === "Break" ||
          tail.kind === "Continue")
      ) {
        break;
      }
    }
    return out;
  }

  private lowerStmt(s: Stmt): IRStmt | IRStmt[] | null {
    const hoistMark = this.pendingExprHoists.length;
    const inner = this.lowerStmtInner(s);
    if (this.pendingExprHoists.length === hoistMark) return inner;
    const drained = this.pendingExprHoists.splice(hoistMark);
    if (inner === null) return drained;
    if (Array.isArray(inner)) return [...drained, ...inner];
    return [...drained, inner];
  }

  private lowerStmtInner(s: Stmt): IRStmt | IRStmt[] | null {
    switch (s.type) {
      case "Function":
        return null; // pre-scanned, specialized on demand at call sites
      case "ClassDef":
        return null; // pre-scanned into classDefs registry
      case "ExprStmt":
        return this.lowerExprStmt(s);
      case "Assign":
        return this.lowerAssign(s);
      case "AssignLValue":
        return this.lowerAssignLValue(s);
      case "MultiAssign":
        return lowerMultiAssign.call(this, s);
      case "If":
        return this.lowerIf(s);
      case "While":
        return this.lowerWhile(s);
      case "For":
        return this.lowerFor(s);
      case "Return":
        return { kind: "ReturnFromFunction", span: s.span };
      case "Break":
        return { kind: "Break", span: s.span };
      case "Continue":
        return { kind: "Continue", span: s.span };
      case "Directive":
        return this.lowerDirective(s);
      default:
        throw new UnsupportedConstruct(
          `statement type '${s.type}' not supported`,
          s.span
        );
    }
  }

  /** Numbl directives (`%!numbl:<name> <args>`). Numbl interprets a
   *  small set (e.g. `assert_jit`) and ignores the rest. Mtoc2 reuses
   *  the same parsed-directive AST node to host translator-side hints
   *  that numbl silently passes over:
   *
   *  - `%!numbl:opaque <var> [<var>...]` — strip `exact` from each
   *    named variable in the current env. Used in test scripts to
   *    force the runtime codegen path on values mtoc2 would otherwise
   *    fold at compile time.
   *  - `%!numbl:showtype <var> [<var>...]` — snapshot each variable's
   *    type and emit a `/_ type ... _/` comment in the generated C
   *    at this position. Debug aid; no runtime effect.
   *  - `%!numbl:printtype <var> [<var>...]` — same snapshot, but
   *    written to stderr (via `printTypeSink`) at compile time. Fires
   *    once per function specialization. No IR node, no codegen
   *    interaction.
   *
   *  Numbl ignores all three directives, so cross-runner output is
   *  unaffected. */
  private lowerDirective(
    s: Extract<Stmt, { type: "Directive" }>
  ): IRStmt | null {
    if (s.directive === "opaque") {
      for (const name of s.args) {
        const entry = this.env.get(name);
        if (entry === undefined) {
          throw new UnsupportedConstruct(
            `'%!numbl:opaque' references unknown variable '${name}'`,
            s.span
          );
        }
        // Strip exact from the env. Delegates to `withoutExact` so
        // struct fields, class properties, and char/string scalars
        // also get their exact stripped — otherwise an opaque'd
        // struct would still let the if-cond folder see precise
        // field values (e.g. `s = struct('x', 5); %!numbl:opaque s;
        // if s.x == 5 …` would still take the branch). The
        // variable's prior Assign already materialized in C
        // (always-materialize), so the runtime path can read its
        // current buffer contents — no synthetic re-assignment
        // needed here.
        const stripped = withoutExact(entry.ty);
        if (stripped !== entry.ty) {
          this.env.set(name, { cName: entry.cName, ty: stripped });
        }
      }
      return null;
    }
    if (s.directive === "showtype") {
      const entries = this.snapshotTypeEntries(s.args, s.span, "showtype");
      return { kind: "TypeComment", entries, span: s.span };
    }
    if (s.directive === "printtype") {
      const entries = this.snapshotTypeEntries(s.args, s.span, "printtype");
      const fileSource = this.workspace.sourceOf(s.span.file) ?? "";
      const { line, column } = offsetToLineCol(fileSource, s.span.start);
      for (const e of entries) {
        printTypeSink(
          `${s.span.file}:${line}:${column}: type ${e.name} :: ${typeToString(e.ty)}`
        );
      }
      return null;
    }
    // Unknown directives are silently ignored — keeps mtoc2 forward-
    // compatible with numbl directives that don't translate (e.g.
    // `assert_jit`).
    return null;
  }

  /** Look up each name in `args` in the current env, snapshotting its
   *  `cName` and current `Type`. Throws `UnsupportedConstruct` with
   *  the directive's span on the first unknown name. Used by the
   *  `showtype` and `printtype` directive branches. */
  private snapshotTypeEntries(
    args: string[],
    span: Span,
    directiveName: string
  ): { name: string; cName: string; ty: Type }[] {
    const entries: { name: string; cName: string; ty: Type }[] = [];
    for (const name of args) {
      const entry = this.env.get(name);
      if (entry === undefined) {
        throw new UnsupportedConstruct(
          `'%!numbl:${directiveName}' references unknown variable '${name}'`,
          span
        );
      }
      entries.push({ name, cName: entry.cName, ty: entry.ty });
    }
    return entries;
  }

  private lowerExprStmt(
    s: Extract<Stmt, { type: "ExprStmt" }>
  ): IRStmt | IRStmt[] | null {
    // Special case: bare `toc;` / `toc();` / `toc(t0);` ExprStmt routes
    // to the printing form of toc. Numbl uses `nargout === 0` as the
    // discriminator; mtoc2 uses "this Call is the sole expression of
    // an ExprStmt". The shadow checks ensure a user `toc = 5; toc;`
    // (reading a local) or a `classdef toc` (constructor call) doesn't
    // get hijacked. We synthesize a Void-typed Call to the runtime
    // helper directly so we don't go through builtin codegen at all.
    if (this.env.get("toc") === undefined && !this.workspace.isClass("toc")) {
      if (s.expr.type === "Ident" && s.expr.name === "toc") {
        return {
          kind: "ExprStmt",
          expr: {
            kind: "Call",
            cName: "mtoc2_toc_print",
            name: "toc_print",
            args: [],
            ty: VOID,
            span: s.expr.span,
          },
          span: s.span,
        };
      }
      if (s.expr.type === "FuncCall" && s.expr.name === "toc") {
        if (s.expr.args.length === 0) {
          return {
            kind: "ExprStmt",
            expr: {
              kind: "Call",
              cName: "mtoc2_toc_print",
              name: "toc_print",
              args: [],
              ty: VOID,
              span: s.expr.span,
            },
            span: s.span,
          };
        }
        if (s.expr.args.length === 1) {
          const arg = this.lowerExpr(s.expr.args[0]);
          if (!isScalarRealNumeric(arg.ty)) {
            throw new TypeError(
              `'toc' tic-handle argument must be a scalar real numeric ` +
                `(the value returned by 'tic'); got ${arg.ty.kind}`,
              s.expr.args[0].span
            );
          }
          return {
            kind: "ExprStmt",
            expr: {
              kind: "Call",
              cName: "mtoc2_toc_handle_print",
              name: "toc_handle_print",
              args: [arg],
              ty: VOID,
              span: s.expr.span,
            },
            span: s.span,
          };
        }
      }
    }
    // Multi-output user-function bare-statement form: `foo(x);` where
    // `foo` returns N≥2 outputs. The expression-position lowering of
    // a multi-output user function is rejected (the C ABI is `void` +
    // out-pointers, no return value to consume), so we peek at the
    // resolver and route to `lowerMultiAssign` with zero lvalues —
    // "drop every output" semantics, matching numbl. We use the
    // empty-args resolver call (no arg types yet) since for a bare
    // name the verdict doesn't depend on arg types.
    if (s.expr.type === "FuncCall") {
      const fc = s.expr;
      const envEntry = this.env.get(fc.name);
      if (envEntry === undefined && !this.workspace.isClass(fc.name)) {
        const target = this.workspace.resolve(
          fc.name,
          [],
          this.callSite(),
          fc.span
        );
        if (target?.kind === "userFunction" && target.ast.outputs.length >= 2) {
          return lowerMultiAssign.call(this, {
            type: "MultiAssign",
            lvalues: [],
            expr: fc,
            suppressed: s.suppressed,
            span: s.span,
          });
        }
      }
    }
    // Same drop-all peek for a bare `pkg.foo(x);` whose target is an
    // N≥2-output packaged user function. Without this, `lowerExpr`
    // routes through `lowerMethodCall`, which throws on `>= 2`
    // outputs because there is no value to consume.
    if (s.expr.type === "MethodCall") {
      const mc = s.expr;
      const dottedBase = tryExtractDottedName(mc.base);
      if (dottedBase && !this.env.has(dottedBase.split(".")[0])) {
        const qname = `${dottedBase}.${mc.name}`;
        if (!this.workspace.isClass(qname)) {
          const target = this.workspace.resolve(
            qname,
            [],
            this.callSite(),
            mc.span
          );
          if (
            target?.kind === "userFunction" &&
            target.ast.outputs.length >= 2
          ) {
            return lowerMultiAssign.call(this, {
              type: "MultiAssign",
              lvalues: [],
              expr: mc,
              suppressed: s.suppressed,
              span: s.span,
            });
          }
        }
      }
    }
    const expr = this.lowerExpr(s.expr);
    // A folded literal has no side effect, so a *suppressed* one (`42;`) is
    // dead code we can drop. But an *unsuppressed* bare literal (`42`) still
    // sets `ans` and echoes "ans = 42" — keep it: as a non-Void bare ExprStmt
    // it makes `assertNoNonVoidBareExprStmts` decline the spec, so the
    // interpreter performs the `ans` echo (the same path `2+3` / `pi` take).
    // Dropping it here silently loses that output and the `ans` update — which
    // also breaks `eval('42')` and numeric `input()` of a bare number.
    if ((expr.kind === "NumLit" || expr.kind === "ImagLit") && s.suppressed) {
      return null;
    }
    // Void-typed call (zero-output user function, fprintf-style
    // side-effecting builtin): the top-level expression itself can't
    // be hoisted (Void has no value), but its OWN-producing
    // sub-expressions still need ANF so they land in named locals
    // that codegen can take the address of / scope-exit free.
    if (isVoid(expr.ty)) {
      const hoists: IRStmt[] = [];
      const hoisted = this.anfChildren(expr, hoists);
      if (hoists.length > 0) {
        return [...hoists, { kind: "ExprStmt", expr: hoisted, span: s.span }];
      }
      return { kind: "ExprStmt", expr, span: s.span };
    }
    // A-normalize: hoist every owned-producing non-Var sub-expression
    // to a fresh temp Assign. After ANF, owned-producing expressions
    // appear only as Assign RHSs (so codegen has a single uniform
    // consume site), and every freshly-allocated tensor's lifetime is
    // tied to a named local that the scope-exit free walk releases.
    const hoists: IRStmt[] = [];
    const hoisted = this.anfRequireScalarOrVar(expr, hoists);
    if (hoists.length > 0) {
      return [...hoists, { kind: "ExprStmt", expr: hoisted, span: s.span }];
    }
    return { kind: "ExprStmt", expr, span: s.span };
  }

  /** Reject Void in a value-consuming context with a clear span. The
   *  Void type tags a call to a zero-output user function; it has no
   *  representation as a value and is only valid as the direct
   *  expression of an `ExprStmt`. */
  requireValueType(e: IRExpr, what: string): void {
    if (isVoid(e.ty)) {
      throw new UnsupportedConstruct(
        `${what}: cannot use the result of a zero-output function as a value`,
        e.span
      );
    }
  }

  // ── ANF (owned-producing-expression hoisting) ─────────────────────────

  /** Recursively rewrite sub-expressions of `e`, hoisting any owned-
   *  producing non-Var sub-expression to a fresh temp Assign. The
   *  top-level `e` itself is NOT hoisted by this function — the caller
   *  decides what context `e` sits in (Assign RHS at an owned consume
   *  site can keep an owned producer; everywhere else requires
   *  `anfRequireScalarOrVar`). */
  private anfChildren(e: IRExpr, hoists: IRStmt[]): IRExpr {
    switch (e.kind) {
      case "NumLit":
      case "ImagLit":
      case "StringLit":
      case "Var":
      case "HandleLit":
      case "HandleCaptureLoad":
      case "EndRef":
        return e;
      case "TensorBuild":
        return {
          ...e,
          elements: e.elements.map(el =>
            this.anfRequireScalarOrVar(el, hoists)
          ),
        };
      case "TensorConcat":
        return {
          ...e,
          cells: e.cells.map(row =>
            row.map(cell => this.anfRequireScalarOrVar(cell, hoists))
          ),
        };
      case "CellLit":
        // Each slot value is consumed by the freshly-allocated cell
        // (parallel to TensorBuild / StructLit). Owned slot values must
        // be fresh producers (ANF guarantees this) and stay in place;
        // non-owned slot values pass through scalar-or-Var ANF.
        return {
          ...e,
          elements: e.elements.map(el =>
            isOwned(el.ty)
              ? this.anfChildren(el, hoists)
              : this.anfRequireScalarOrVar(el, hoists)
          ),
        };
      case "CellEmpty":
        return {
          ...e,
          dims: e.dims.map(d => this.anfRequireScalarOrVar(d, hoists)),
        };
      case "CellIndexLoad":
        return {
          ...e,
          base: this.anfRequireScalarOrVar(e.base, hoists),
          indices: e.indices.map(i => this.anfRequireScalarOrVar(i, hoists)),
        };
      case "Binary":
        return {
          ...e,
          left: this.anfRequireScalarOrVar(e.left, hoists),
          right: this.anfRequireScalarOrVar(e.right, hoists),
        };
      case "Unary":
        return {
          ...e,
          operand: this.anfRequireScalarOrVar(e.operand, hoists),
        };
      case "Call":
        return {
          ...e,
          args: e.args.map(a => this.anfRequireScalarOrVar(a, hoists)),
        };
      case "StructLit":
        return {
          ...e,
          fields: e.fields.map(f => ({
            name: f.name,
            value: isOwned(f.value.ty)
              ? // Owned field values land inside a designated initializer
                // — they must be fresh producers. Recurse through ANF on
                // their children but leave the top-level producer in
                // place (it's at a direct consume site, just like an
                // owned Assign RHS).
                this.anfChildren(f.value, hoists)
              : this.anfRequireScalarOrVar(f.value, hoists),
          })),
        };
      case "MemberLoad":
        return {
          ...e,
          base: this.anfRequireScalarOrVar(e.base, hoists),
        };
      case "IndexLoad":
        // The base is usually a `Var` (resolveIndexBase returns one for
        // bare-name indexing) but can be a `MemberLoad` when the source
        // form is `obj.field(i)`. ANF the base too so the post-ANF IR
        // always has a Var here. Each scalar index slot also ANFs.
        return {
          ...e,
          base: this.anfRequireScalarOrVar(e.base, hoists),
          indices: e.indices.map(i => this.anfRequireScalarOrVar(i, hoists)),
        };
      case "IndexSlice":
        // Slice slots' sub-expressions are scalar (start/step/end of a
        // Range, or the Scalar slot's expr). Run them through scalar-
        // or-Var ANF to keep them simple — they evaluate per loop
        // iteration in codegen. The base mirrors IndexLoad: usually a
        // `Var`, but may be a `MemberLoad` in the property-rooted
        // `obj.field(args)` form, which ANF then hoists.
        return {
          ...e,
          base: this.anfRequireScalarOrVar(e.base, hoists),
          index: e.index.map(slot => {
            if (slot.kind === "Range") {
              return {
                ...slot,
                start: this.anfRequireScalarOrVar(slot.start, hoists),
                step: this.anfRequireScalarOrVar(slot.step, hoists),
                end: this.anfRequireScalarOrVar(slot.end, hoists),
              };
            }
            if (slot.kind === "Scalar") {
              return {
                ...slot,
                expr: this.anfRequireScalarOrVar(slot.expr, hoists),
              };
            }
            if (slot.kind === "IndexVec") {
              return {
                ...slot,
                expr: this.anfRequireScalarOrVar(slot.expr, hoists),
              };
            }
            if (slot.kind === "LogicalMask") {
              return {
                ...slot,
                expr: this.anfRequireScalarOrVar(slot.expr, hoists),
              };
            }
            return slot;
          }),
        };
      case "MakeRange":
        return {
          ...e,
          start: this.anfRequireScalarOrVar(e.start, hoists),
          step: this.anfRequireScalarOrVar(e.step, hoists),
          end: this.anfRequireScalarOrVar(e.end, hoists),
        };
    }
  }

  /** Walk `e` and ensure the returned expression is either non-owned or
   *  a Var. Recursively ANFs children; if `e` itself is owned-producing
   *  (multi-element tensor / struct / class / handle / char / string,
   *  in non-Var form), hoist it to a fresh temp. The ANF invariant is
   *  that owned-producing expressions appear only as direct Assign RHSs
   *  at owned consume sites; every other position holds a Var read. */
  anfRequireScalarOrVar(e: IRExpr, hoists: IRStmt[]): IRExpr {
    const rewritten = this.anfChildren(e, hoists);
    if (isOwned(rewritten.ty) && rewritten.kind !== "Var") {
      return this.hoistToTemp(rewritten, hoists);
    }
    return rewritten;
  }

  private hoistToTemp(e: IRExpr, hoists: IRStmt[]): IRExpr {
    const tempName = this.freshTempName();
    this.env.set(tempName, { cName: tempName, ty: e.ty });
    hoists.push({
      kind: "Assign",
      name: tempName,
      cName: tempName,
      ty: e.ty,
      expr: e,
      span: e.span,
    });
    return {
      kind: "Var",
      name: tempName,
      cName: tempName,
      ty: e.ty,
      span: e.span,
    };
  }

  freshTempName(): string {
    this.tempCounter += 1;
    return `_mtoc2_t${this.tempCounter}`;
  }

  private lowerAssign(s: Extract<Stmt, { type: "Assign" }>): IRStmt | IRStmt[] {
    const expr = this.lowerExpr(s.expr);
    this.requireValueType(expr, `assigning to '${s.name}'`);
    // ANF the RHS. When the RHS is itself owned-producing and the
    // LHS is also owned, the RHS is at a direct consume site — recurse
    // into its CHILDREN only (the top stays as the Assign's RHS).
    // Otherwise the RHS lands at a non-consume site (scalar Assign,
    // mismatched ownership) and the top-level itself may need hoisting.
    // Uses `isOwned` (not just `isMultiElement`) so Struct/Class/Handle/
    // String/Char RHSs reach the consume-site path — matches
    // `lowerAssignLValue`'s `leafOwned` check and the documented
    // ANF invariant ("owned-producing exprs appear only as direct
    // Assign RHSs at owned consume sites").
    const hoists: IRStmt[] = [];
    const lhsOwned = isOwned(expr.ty);
    const rhsOwnedDirectProducer = lhsOwned && expr.kind !== "Var";
    const newExpr = rhsOwnedDirectProducer
      ? this.anfChildren(expr, hoists)
      : this.anfRequireScalarOrVar(expr, hoists);
    const main = this.recordAssignment(s.name, newExpr, s.span);
    if (hoists.length === 0) return main;
    return [...hoists, main];
  }

  /** `s.f = rhs` or chained `s.inner.f = rhs`. Lowers to a single
   *  `MemberStore` IR node with the root `Var` as `base` and the
   *  field chain in `fieldPath`. v1 requires the root variable to be
   *  already in env (no implicit struct introduction — use
   *  `s = struct('f', v, ...)` or a class constructor). Every step
   *  of the chain must already exist on the corresponding nested
   *  struct/class type.
   *
   *  Type discipline: the rhs and the field must occupy the same
   *  C-level slot — checked by `storageEquivalent`, which calls
   *  `cFieldTypeStr` on both. Scalar↔tensor, different elem, or
   *  different struct/class typedef name → rejected. Otherwise the
   *  write is accepted and env is updated so subsequent reads of
   *  the field report the rhs's full internal type (the C typedef
   *  is unaffected — the typedef hash sees only the C-level type
   *  via `cFieldTypeStr`, not the internal lattice precision). */
  // Package-internal (used by `lowerMultiAssign` so it can route
  // synthesized AssignLValue stmts back through this path for the
  // Member / Index lvalue cases). Treat as private outside the
  // lowering module.
  lowerAssignLValue(
    s: Extract<Stmt, { type: "AssignLValue" }>
  ): IRStmt | IRStmt[] {
    const lv = s.lvalue;
    if (lv.type === "IndexCell") {
      return lowerCellIndexStore.call(this, lv, s.expr, s.span);
    }
    if (lv.type === "Index") {
      // `<indexed> = []` is element/row/column DELETION, which dynamically
      // shrinks the base. The JIT compiles indexed stores against the static
      // pre-write shape and can't reallocate mid-spec — a masked/slice store
      // against the 0-element [] either reads NaN (JS) or aborts on a size
      // mismatch (C). Decline to the interpreter, which implements MATLAB's
      // full deletion semantics.
      if (s.expr.type === "Tensor" && s.expr.rows.length === 0) {
        throw new UnsupportedConstruct(
          "indexed deletion `x(idx) = []` is not JIT-compiled; resizing the " +
            "base requires the interpreter",
          s.span
        );
      }
      const result = lv.indices.some(isSliceArg)
        ? lowerIndexSliceStore.call(this, lv, s.expr, s.span)
        : lowerIndexStore.call(this, lv, s.expr, s.span);
      // The runtime tensor was mutated in place, but the env entry
      // for the base variable still carries the pre-write `exact`
      // and `sign` (set by e.g. `zeros(N,M)` at construction).
      // Subsequent transfer fns (sum, etc.), domain checks (sqrt /
      // log), and the if-cond folder would read those stale fields
      // and silently mis-emit. Drop `exact` and widen `sign` toward
      // the rhs sign so the env reflects the post-write reality.
      // Bare-Ident base: try to refresh the exact Float64Array in
      // place when the store's indices and rhs are both compile-time-
      // known and the base already carries exact data. This keeps
      // downstream builtin transfers (`zeros(sz)` after `sz(1) = ...`)
      // able to see the post-write shape. If any precondition fails,
      // fall back to widening (strip exact).
      if (lv.base.type === "Ident") {
        const refreshed = tryRefreshExactAfterIndexedWrite(
          this.env,
          lv.base.name,
          result
        );
        if (!refreshed) {
          const rhsSign = rhsSignFromStoreResult(result);
          widenAfterIndexedWrite(this.env, lv.base.name, rhsSign);
        }
      } else if (lv.base.type === "Member") {
        // Member-rooted: strip the leaf field's `exact` carrier (the
        // construction-site Float64Array is stale after the write).
        // In-place refresh of the field's exact data is doable when
        // indices + rhs are all known, but the MVP just widens; the
        // optimisation can land later if a use case needs it.
        const last = Array.isArray(result) ? result[result.length - 1] : result;
        if (
          last !== undefined &&
          (last.kind === "IndexStore" || last.kind === "IndexSliceStore") &&
          last.fieldPath !== undefined &&
          last.leafTy !== undefined
        ) {
          widenMemberLeafAfterIndexedWrite(
            this.env,
            last.base.name,
            last.fieldPath,
            last.leafTy
          );
        }
      }
      return result;
    }
    if (lv.type !== "Member") {
      throw new UnsupportedConstruct(
        `assignment lvalue '${lv.type}' is not supported`,
        s.span
      );
    }

    // Walk the Member chain to find the root Ident and the field
    // path (outermost → innermost). Reject any non-Ident root or
    // `MemberDynamic` step.
    const unwound = unwindMemberLvalue(lv);
    if (unwound === null) {
      throw new UnsupportedConstruct(
        `assignment lvalue must be rooted at a named variable`,
        s.span
      );
    }
    const { rootName, fields: fieldPath } = unwound;
    const rootEntry = this.env.get(rootName);
    if (rootEntry === undefined) {
      throw new UnsupportedConstruct(
        `assignment to '${rootName}.${fieldPath.join(".")}' but '${rootName}' is not yet defined ` +
          `(struct variables must first be introduced via 'struct(...)' or a class constructor call)`,
        s.span
      );
    }

    // Dependent-property routing: single-level `obj.X = rhs` where the
    // class declares a `set.X` setter rewrites to
    // `obj = set.X(obj, rhs)` — a method call returning self. The
    // existing call machinery handles ANF, env update, and ownership
    // bookkeeping (same shape as `obj = obj.method(args)`).
    if (fieldPath.length === 1 && rootEntry.ty.kind === "Class") {
      const reg = this.classReg(rootEntry.ty.className);
      const propName = fieldPath[0];
      const setter = reg?.setters.get(propName);
      if (setter !== undefined && reg !== undefined) {
        const rhsRaw = this.lowerExpr(s.expr);
        this.requireValueType(
          rhsRaw,
          `assignment to '${rootName}.${propName}'`
        );
        const baseVar: IRExpr = {
          kind: "Var",
          name: rootName,
          cName: rootEntry.cName,
          ty: rootEntry.ty,
          span: s.span,
        };
        const callExpr = buildUserFunctionCall.call(
          this,
          setter,
          [baseVar, rhsRaw],
          `${rootEntry.ty.className}.set.${propName}`,
          s.span,
          {
            specSource: classMethodSpecSource(
              rootEntry.ty.className,
              `set.${propName}`
            ),
            definingFile: setter.span.file ?? reg.file,
          }
        );
        // ANF the call's children the same way `lowerAssign` does — the
        // call is owned-producing and lands at a direct consume site
        // (the receiver assignment), so we recurse through children
        // only.
        const hoists: IRStmt[] = [];
        const lhsOwned = isOwned(callExpr.ty);
        const rhsOwnedDirectProducer = lhsOwned && callExpr.kind !== "Var";
        const newExpr = rhsOwnedDirectProducer
          ? this.anfChildren(callExpr, hoists)
          : this.anfRequireScalarOrVar(callExpr, hoists);
        const main = this.recordAssignment(rootName, newExpr, s.span);
        return hoists.length === 0 ? main : [...hoists, main];
      }
      // Dependent property with no setter is read-only at the language
      // level. Flag it early — otherwise `fieldType` returns undefined
      // and the user sees the generic "no such field" message.
      if (reg?.dependentProperties.has(propName)) {
        throw new UnsupportedConstruct(
          `dependent property '${rootEntry.ty.className}.${propName}' ` +
            `has no setter; cannot assign`,
          s.span
        );
      }
    }

    // Walk the field path, checking that each step exists. Track the
    // leaf type the rhs is going to overwrite.
    let stepTy: Type = rootEntry.ty;
    for (let i = 0; i < fieldPath.length; i++) {
      const fname = fieldPath[i];
      const ft = fieldType(stepTy, fname);
      if (ft === undefined) {
        throw new TypeError(
          `'${rootName}.${fieldPath.slice(0, i + 1).join(".")}': no such ` +
            `field on type ${typeToString(stepTy)}`,
          s.span
        );
      }
      stepTy = ft;
    }
    const leafTy = stepTy;

    // Lower the rhs. Then ANF the rhs the same way Assign does — owned
    // direct producers can stay; everything else hoists to a temp.
    const rhsRaw = this.lowerExpr(s.expr);
    this.requireValueType(
      rhsRaw,
      `assignment to '${rootName}.${fieldPath.join(".")}'`
    );

    // Storage compatibility: the rhs and leaf must occupy the same
    // C-level slot. `storageEquivalent` reduces both sides via
    // `cFieldTypeStr` — so all `mtoc2_tensor_t` slots accept any
    // multi-element tensor, all `double` slots accept any scalar
    // real numeric, and struct/class slots only accept values with
    // the same typedef name. Internal lattice precision (sign,
    // exact, tensor shape) is preserved on the field afterward —
    // the typedef hash already ignores it.
    if (!storageEquivalent(rhsRaw.ty, leafTy)) {
      throw new TypeError(
        `assignment to '${rootName}.${fieldPath.join(".")}': field has ` +
          `C type ${typeToString(leafTy)} but rhs has ` +
          `type ${typeToString(rhsRaw.ty)}`,
        s.span
      );
    }

    // ANF the rhs.
    const hoists: IRStmt[] = [];
    const leafOwned = isOwned(leafTy);
    const rhsOwnedDirectProducer = leafOwned && rhsRaw.kind !== "Var";
    const rhs = rhsOwnedDirectProducer
      ? this.anfChildren(rhsRaw, hoists)
      : this.anfRequireScalarOrVar(rhsRaw, hoists);

    const baseVar: Extract<IRExpr, { kind: "Var" }> = {
      kind: "Var",
      name: rootName,
      cName: rootEntry.cName,
      ty: rootEntry.ty,
      span: s.span,
    };
    const store: IRStmt = {
      kind: "MemberStore",
      base: baseVar,
      fieldPath,
      leafTy: rhsRaw.ty,
      rhs,
      span: s.span,
    };

    // Update env so subsequent reads of the touched field see the
    // post-write rhs type — not the construction-site type. The
    // typedef hash uses `cFieldTypeStr` (one C-type string per
    // field), so the C typedef name stays the same regardless of
    // how the field's internal precision evolves through writes.
    // CFG-join `unify` widens further when different branches
    // assign different types to the same path.
    const updatedRootTy = withPathTypeUpdated(
      rootEntry.ty,
      fieldPath,
      rhsRaw.ty
    );
    this.env.set(rootName, { cName: rootEntry.cName, ty: updatedRootTy });

    if (hoists.length === 0) return store;
    return [...hoists, store];
  }

  // ── MultiAssign + buildMultiOutputSlots lives in `./lowerMultiAssign.ts`. ─

  recordAssignment(name: string, expr: IRExpr, span: Span): Assign {
    const existing = this.env.get(name);
    // Type-compat check: catch reassignments that would invalidate the
    // C-side declaration. The function-top `collectOwnedLocals` walk
    // records only the FIRST seen owned typedef per cName, so a later
    // owned-typed reassignment of `name` with a different storage shape
    // (different struct field set, different class, different handle
    // capture-shape, scalar↔tensor, etc.) would emit a call to a
    // `_assign` helper whose signature doesn't match the pre-declared
    // local — i.e. a C compile error rather than a clean span-attributed
    // translate-time error. `storageEquivalent` is the same predicate
    // used for `MemberStore` writes — it compares `cFieldTypeStr` so
    // it catches every C-level slot mismatch.
    if (existing && !storageEquivalent(existing.ty, expr.ty)) {
      // Incompatible reassignment: the new value can't share the prior
      // binding's C variable (scalar↔tensor, char↔double, different
      // struct/class/handle storage, ...). At the top level of a scope
      // we SPLIT — allocate a fresh C local for the new value and
      // repoint the name's env entry at it. Subsequent reads see the new
      // binding; earlier reads of the old one stay valid. The predeclare
      // walk (`collectAllLocals`) keys by cName, so each binding is
      // declared, lived, and freed independently. Inside control flow we
      // can't split: the branch / loop env merge would have to reconcile
      // two storage categories under one name (`mergeBranchEnvs` keeps a
      // single cName per key), so we throw with a span instead.
      if (this.controlDepth > 0) {
        if (isMultiElement(existing.ty) !== isMultiElement(expr.ty)) {
          throw new UnsupportedConstruct(
            `cannot reassign '${name}' across scalar/tensor boundary ` +
              `inside control flow`,
            span
          );
        }
        throw new UnsupportedConstruct(
          `cannot reassign '${name}': new value's C storage ` +
            `(${typeToString(expr.ty)}) is incompatible with the ` +
            `existing binding (${typeToString(existing.ty)}) inside ` +
            `control flow`,
          span
        );
      }
      const splitCName = `_mtoc2_${existing.cName}__v${++this.splitCounter}`;
      this.env.set(name, { cName: splitCName, ty: expr.ty });
      return {
        kind: "Assign",
        name,
        cName: splitCName,
        ty: expr.ty,
        expr,
        span,
      };
    }
    const cName = existing?.cName ?? cIdentForUserName(name);
    this.env.set(name, { cName, ty: expr.ty });
    return {
      kind: "Assign",
      name,
      cName,
      ty: expr.ty,
      expr,
      span,
    };
  }

  private lowerIf(s: Extract<Stmt, { type: "If" }>): IRStmt | IRStmt[] {
    const cond = this.lowerExpr(s.cond);
    this.requireScalarCondType(cond.ty, "if condition", s.span);

    // If-fold: when the top cond is exact, take/drop the then-arm and
    // recurse on the remaining elseif chain. The cond's COMPUTATION is
    // still emitted (as an `ExprStmt` prefix) so any side effects in
    // the cond — e.g. a `log_then_5() > 0` where the user function's
    // body has a `disp` and its spec returns an exact 5 — still run at
    // runtime. Only the BRANCH decision is folded out. `condIsPure`
    // skips the prefix when no side effect is reachable from cond
    // (NumLit / Var / EndRef / pure arithmetic on those) so the
    // common literal-cond case stays clean.
    const folded = condToBool(cond);
    const condPrefix: IRStmt[] = condIsPure(cond)
      ? []
      : [{ kind: "ExprStmt", expr: cond, span: s.cond.span }];
    if (folded === true) return [...condPrefix, ...this.lowerStmts(s.thenBody)];
    if (folded === false) {
      if (s.elseifBlocks.length === 0) {
        return [
          ...condPrefix,
          ...(s.elseBody ? this.lowerStmts(s.elseBody) : []),
        ];
      }
      const [first, ...rest] = s.elseifBlocks;
      // Reshape: `elseif first ... rest else B` becomes a fresh If.
      const synthetic: Extract<Stmt, { type: "If" }> = {
        type: "If",
        cond: first.cond,
        thenBody: first.body,
        elseifBlocks: rest,
        elseBody: s.elseBody,
        span: first.cond.span,
      };
      const inner = this.lowerIf(synthetic);
      return [...condPrefix, ...(Array.isArray(inner) ? inner : [inner])];
    }

    // Non-folded path.
    const envBefore = new Map(this.env);
    const branchEnvs: Map<string, EnvEntry>[] = [];

    return this.withControlDepth(() => {
      // Then-branch.
      this.env = new Map(envBefore);
      const thenBody = this.lowerStmts(s.thenBody);
      branchEnvs.push(this.env);

      // Else chain.
      this.env = new Map(envBefore);
      const elseBody = this.lowerElseChain(
        s.elseifBlocks,
        s.elseBody,
        envBefore,
        branchEnvs
      );

      // Merge. `true` → flag variables assigned on only some arms as
      // maybe-unassigned (definite-assignment check, see EnvEntry).
      this.env = this.mergeBranchEnvs(branchEnvs, true);
      return {
        kind: "If",
        cond,
        thenBody,
        elseBody,
        span: s.span,
      };
    });
  }

  private lowerElseChain(
    elseifs: { cond: Expr; body: Stmt[] }[],
    elseBody: Stmt[] | null,
    envBefore: Map<string, EnvEntry>,
    branchEnvs: Map<string, EnvEntry>[]
  ): IRStmt[] {
    if (elseifs.length === 0) {
      if (elseBody === null) {
        branchEnvs.push(new Map(envBefore));
        return [];
      }
      this.env = new Map(envBefore);
      const b = this.lowerStmts(elseBody);
      branchEnvs.push(this.env);
      return b;
    }
    const [first, ...rest] = elseifs;
    this.env = new Map(envBefore);
    const ec = this.lowerExpr(first.cond);
    this.requireScalarCondType(ec.ty, "elseif condition", first.cond.span);
    const beforeBody = new Map(this.env);

    this.env = beforeBody;
    const thenBody = this.lowerStmts(first.body);
    branchEnvs.push(this.env);

    this.env = new Map(envBefore);
    const innerElse = this.lowerElseChain(
      rest,
      elseBody,
      envBefore,
      branchEnvs
    );

    return [
      {
        kind: "If",
        cond: ec,
        thenBody,
        elseBody: innerElse,
        span: first.cond.span,
      },
    ];
  }

  private lowerWhile(s: Extract<Stmt, { type: "While" }>): IRStmt {
    const envBefore = new Map(this.env);
    // Strip exact for body-mutated vars BEFORE lowering cond — the body
    // might re-enter the cond after a back-edge, so even the cond sees
    // post-loop values.
    stripExactFromEnv(this.env, collectAssignedNames(s.body));
    const cond = this.lowerExpr(s.cond);
    this.requireScalarCondType(cond.ty, "while condition", s.span);
    const envBodyInput = new Map(this.env);
    const body = this.withControlDepth(() => this.lowerStmts(s.body));
    assertLoopBodyShapeStable(envBodyInput, this.env, s.body, s.span);
    this.env = this.mergeBranchEnvs([envBefore, this.env]);
    return { kind: "While", cond, body, span: s.span };
  }

  private lowerFor(s: Extract<Stmt, { type: "For" }>): IRStmt {
    if (s.expr.type !== "Range") {
      throw new UnsupportedConstruct(
        `for-loop iterables other than ranges are not yet supported`,
        s.span
      );
    }
    const start = this.lowerExpr(s.expr.start);
    const end = this.lowerExpr(s.expr.end);
    this.requireScalarReal(start.ty, "for-loop start", s.expr.start.span);
    this.requireScalarReal(end.ty, "for-loop end", s.expr.end.span);

    let step = 1;
    if (s.expr.step) {
      const stepExpr = this.lowerExpr(s.expr.step);
      this.requireScalarReal(stepExpr.ty, "for-loop step", s.expr.step.span);
      // Step must be a compile-time-known scalar — read its exact from
      // the type (no IR-level fold runs anymore, but the transfer fns
      // still propagate exact through e.g. unary-minus on a literal).
      const stepVal =
        isNumeric(stepExpr.ty) && typeof stepExpr.ty.exact === "number"
          ? stepExpr.ty.exact
          : undefined;
      if (stepVal === undefined) {
        throw new UnsupportedConstruct(
          `for-loop step must be a compile-time-known numeric literal`,
          s.expr.step.span
        );
      }
      if (stepVal === 0) {
        throw new UnsupportedConstruct(
          `for-loop step must be non-zero`,
          s.expr.step.span
        );
      }
      step = stepVal;
    }

    const envBefore = new Map(this.env);
    // Loop var sign: the values are start, start+step, ..., end, so the
    // sign lattice must cover every value in [min(start,end),
    // max(start,end)]. Unifying startSign and endSign captures that —
    // it correctly classifies a descending loop with positive bounds
    // (`10:-1:1`) as "positive", an ascending loop with negative
    // bounds as "negative", and a range spanning zero as "unknown".
    const startSign = isNumeric(start.ty) ? start.ty.sign : "unknown";
    const endSign = isNumeric(end.ty) ? end.ty.sign : "unknown";
    const kSign: NumericType["sign"] = unifySign(startSign, endSign);

    const cVar = cIdentForUserName(s.varName);
    this.env.set(s.varName, {
      cName: cVar,
      ty: scalarDouble(kSign),
    });

    stripExactFromEnv(this.env, collectAssignedNames(s.body));

    const envBodyInput = new Map(this.env);
    const body = this.withControlDepth(() => this.lowerStmts(s.body));
    assertLoopBodyShapeStable(envBodyInput, this.env, s.body, s.span);
    this.env = this.mergeBranchEnvs([envBefore, this.env]);
    return {
      kind: "For",
      varName: s.varName,
      cVar,
      start,
      step,
      end,
      body,
      span: s.span,
    };
  }

  // ── Expression lowering ───────────────────────────────────────────────

  lowerExpr(e: Expr): IRExpr {
    switch (e.type) {
      case "Number": {
        const v = Number(e.value);
        if (!Number.isFinite(v)) {
          throw new UnsupportedConstruct(
            `non-finite numeric literal '${e.value}'`,
            e.span
          );
        }
        return {
          kind: "NumLit",
          value: v,
          ty: scalarDouble(signFromNumber(v), v),
          span: e.span,
        };
      }
      case "ImagUnit":
        // Bare `i` / `j` postfixed onto a Number turns into
        // `Mul(Number, ImagUnit)` by the parser; that's caught by
        // the collapse in `lowerBinary`. A standalone `ImagUnit`
        // (rare — e.g. `1i / 2` after parser binding) lowers here.
        return {
          kind: "ImagLit",
          value: 1,
          ty: scalarComplex({ re: 0, im: 1 }),
          span: e.span,
        };
      case "Ident":
        return this.lowerIdent(e);
      case "Binary":
        return this.lowerBinary(e);
      case "Unary":
        return this.lowerUnary(e);
      case "FuncCall":
        return lowerFuncCall.call(this, e);
      case "Tensor":
        return lowerTensorLit.call(this, e);
      case "Cell":
        return lowerCellLit.call(this, e);
      case "IndexCell":
        return lowerCellIndexLoad.call(this, e);
      case "FuncHandle":
        return lowerFuncHandle.call(this, e);
      case "AnonFunc":
        return lowerAnonFunc.call(this, e);
      case "Member":
        return this.lowerMember(e);
      case "MethodCall":
        return lowerMethodCall.call(this, e);
      case "EndKeyword":
        return this.lowerEndKeyword(e);
      case "Range":
        return this.lowerRangeAsValue(e);
      case "Char": {
        // Single-quoted MATLAB char literal — lexeme keeps the
        // surrounding `'` delimiters and doubled-quote escapes.
        // Strip them so the type's `exact` mirrors numbl's runtime
        // string. v1 only consumes these inside reducer builtins
        // (`sum(A, 'all')`, etc.); every other site rejects via
        // the builtin's transfer or `requireValueType`-adjacent check.
        const raw = e.value;
        const inner = raw.slice(1, raw.length - 1).replaceAll("''", "'");
        return {
          kind: "StringLit",
          value: inner,
          ty: { kind: "Char", exact: inner },
          span: e.span,
        };
      }
      case "String": {
        // Double-quoted MATLAB string literal — scalar string handle
        // (`length("hi") == 1`); codegen emits via
        // `mtoc2_string_from_literal`. Same stripping rule as Char but
        // with `""` as the in-literal escape.
        const raw = e.value;
        const inner = raw.slice(1, raw.length - 1).replaceAll('""', '"');
        return {
          kind: "StringLit",
          value: inner,
          ty: { kind: "String", exact: inner },
          span: e.span,
        };
      }
      case "Colon":
        throw new UnsupportedConstruct(
          `bare ':' is only valid inside an index expression`,
          e.span
        );
      default:
        throw new UnsupportedConstruct(
          `expression type '${e.type}' not supported`,
          e.span
        );
    }
  }

  /** `end` keyword inside an index slot — resolves through the top of
   *  `endStack`. Outside an index slot it raises with a span. When the
   *  axis size is statically known, emit a `NumLit`; otherwise emit
   *  an `EndRef` IR node and let codegen render the runtime axis-size
   *  read. */
  private lowerEndKeyword(e: Extract<Expr, { type: "EndKeyword" }>): IRExpr {
    if (this.endStack.length === 0) {
      throw new UnsupportedConstruct(
        `'end' is only valid inside an index expression`,
        e.span
      );
    }
    const top = this.endStack[this.endStack.length - 1];
    // Try a static value first.
    if (top.axis === "linear") {
      if (top.baseTy.shape !== undefined) {
        const n = shapeNumel(top.baseTy.shape);
        return {
          kind: "NumLit",
          value: n,
          ty: scalarDouble(n > 0 ? "positive" : "nonneg", n),
          span: e.span,
        };
      }
    } else {
      const axis = top.axis;
      if (top.baseTy.shape !== undefined && axis < top.baseTy.shape.length) {
        const n = top.baseTy.shape[axis];
        return {
          kind: "NumLit",
          value: n,
          ty: scalarDouble(n > 0 ? "positive" : "nonneg", n),
          span: e.span,
        };
      }
    }
    return {
      kind: "EndRef",
      baseCName: top.baseCName,
      baseTy: top.baseTy,
      axis: top.axis,
      ty: scalarDouble("nonneg"),
      span: e.span,
    };
  }

  /** `a : b` or `a : s : b` used as a value (outside an index slot
   *  and outside a for-loop iterable). Emits a `MakeRange` IR node
   *  whose result is a freshly-allocated row vector. */
  private lowerRangeAsValue(e: Extract<Expr, { type: "Range" }>): IRExpr {
    const start = this.lowerExpr(e.start);
    const end = this.lowerExpr(e.end);
    let step: IRExpr;
    if (e.step === null) {
      step = {
        kind: "NumLit",
        value: 1,
        ty: scalarDouble("positive", 1),
        span: e.span,
      };
    } else {
      step = this.lowerExpr(e.step);
    }
    this.requireScalarReal(start.ty, "range start", e.start.span);
    this.requireScalarReal(end.ty, "range end", e.end.span);
    this.requireScalarReal(step.ty, "range step", e.step?.span ?? e.span);

    // Static-shape detection: when start / step / end are all exact
    // and step is finite-nonzero, compute the count at compile time.
    let resultTy: Type;
    const sExact =
      isNumeric(start.ty) && typeof start.ty.exact === "number"
        ? start.ty.exact
        : undefined;
    const tExact =
      isNumeric(step.ty) && typeof step.ty.exact === "number"
        ? step.ty.exact
        : undefined;
    const eExact =
      isNumeric(end.ty) && typeof end.ty.exact === "number"
        ? end.ty.exact
        : undefined;
    if (
      sExact !== undefined &&
      tExact !== undefined &&
      eExact !== undefined &&
      tExact !== 0 &&
      Number.isFinite(sExact) &&
      Number.isFinite(tExact) &&
      Number.isFinite(eExact)
    ) {
      // Mirrors numbl's `makeRangeTensor` count formula via the
      // shared `rangeCountFromExactEnds` helper — same formula as
      // `lowerIndexSlice.exactRangeCount` and the C-side
      // `mtoc2_loop_count`.
      const n = rangeCountFromExactEnds(sExact, tExact, eExact);
      // Length-1 collapse: a `1:1`-style range is a single-element
      // tensor with dims [1, 1]. The type system classifies that as
      // scalar (both dims `one`), so the LHS would be declared
      // `double` — but the IR is `MakeRange`, which always emits a
      // tensor-returning helper. Returning `start` directly resolves
      // the type/IR mismatch and matches MATLAB / numbl, which both
      // treat `1:1` as the scalar `1`. Side effects in `start` are
      // preserved (it's the returned IR); side effects in `step` /
      // `end` are dropped — accepted limitation since both must have
      // exact values for us to know the length statically, and
      // exact-bearing IR is overwhelmingly NumLit / Var / pure
      // arithmetic in practice.
      if (n === 1) return start;
      resultTy = tensorDouble([1, n]);
    } else {
      // Build a dim-only type when length is runtime.
      resultTy = {
        kind: "Numeric",
        elem: "double",
        isComplex: false,
        dims: [{ kind: "exact", value: 1 }, { kind: "unknown" }],
        sign: "unknown",
      };
    }

    return {
      kind: "MakeRange",
      start,
      step,
      end,
      ty: resultTy,
      span: e.span,
    };
  }

  /** `s.f` or chained `s.inner.f`. Lowers each nesting level into a
   *  fresh `MemberLoad` whose `base` is the inner load. The leaf
   *  type is the field's static type on the immediate container.
   *
   *  Dependent-property routing: when `base` is a class instance and
   *  the class declares a `get.<name>` accessor, the read is replaced
   *  by a single-output method call. The result is an owned `Call`
   *  IRExpr; ANF in the surrounding context hoists it like any other
   *  owned producer. */
  private lowerMember(e: Extract<Expr, { type: "Member" }>): IRExpr {
    const base = this.lowerExpr(e.base);
    this.requireValueType(base, `field access '.${e.name}'`);
    if (base.ty.kind === "Class") {
      const reg = this.classReg(base.ty.className);
      const getter = reg?.getters.get(e.name);
      if (getter !== undefined && reg !== undefined) {
        return buildUserFunctionCall.call(
          this,
          getter,
          [base],
          `${base.ty.className}.get.${e.name}`,
          e.span,
          {
            specSource: classMethodSpecSource(
              base.ty.className,
              `get.${e.name}`
            ),
            definingFile: getter.span.file ?? reg.file,
          }
        );
      }
    }
    const ft = fieldType(base.ty, e.name);
    if (ft === undefined) {
      throw new TypeError(
        `no field '${e.name}' on type ${typeToString(base.ty)}`,
        e.span
      );
    }
    return {
      kind: "MemberLoad",
      base,
      field: e.name,
      ty: ft,
      span: e.span,
    };
  }

  // ── Method calls live in `./lowerMethodCall.ts`. ──────────────────────

  private lowerIdent(e: Extract<Expr, { type: "Ident" }>): IRExpr {
    const entry = this.env.get(e.name);
    if (entry !== undefined) {
      if (entry.maybeUnassigned) {
        // Assigned on only some `if` arms — reading it may hit the
        // predeclared default. Decline so the interpreter applies
        // MATLAB's runtime definite-assignment check (an error on the
        // unassigned path) instead of the JIT silently reading 0/NaN.
        throw new UnsupportedConstruct(
          `'${e.name}' may be used before it is assigned (assigned only on ` +
            `some branches)`,
          e.span
        );
      }
      return {
        kind: "Var",
        name: e.name,
        cName: entry.cName,
        ty: entry.ty,
        span: e.span,
      };
    }
    // `nargout` / `nargin` are MATLAB pseudo-variables that fold to a
    // compile-time constant per specialization. The `callFrameStack`
    // is pushed by `specializeUserFunction` before lowering the body
    // and popped after; a reference outside a function body finds it
    // empty and falls through to the "undefined" error.
    if (
      (e.name === "nargout" || e.name === "nargin") &&
      this.callFrameStack.length > 0
    ) {
      const frame = this.callFrameStack[this.callFrameStack.length - 1];
      const v = e.name === "nargout" ? frame.nargout : frame.nargin;
      return {
        kind: "NumLit",
        value: v,
        ty: scalarDouble(signFromNumber(v), v),
        span: e.span,
      };
    }
    // Identifier read of a builtin name with no parens. MATLAB treats
    // this as a 0-arg call when the name isn't shadowed by a local.
    // Supports the fixed-0-arity system builtins (`tic`, `toc`) and
    // the variadic plot-dispatch names whose arity range admits 0
    // (`figure;`, `hold;`, `drawnow;`, …). User-function
    // ident-as-call and class references are left to dedicated paths.
    const b = getBuiltin(e.name);
    if (b !== undefined) {
      // Probe: try `transfer([], 1)`. If the builtin accepts 0 args,
      // return the lowered Call; otherwise fall through to the
      // undefined-variable error (matches MATLAB behavior of treating
      // bare `sqrt` as an undefined identifier read rather than a
      // missing-args call).
      let tys: ReturnType<typeof b.transfer> | undefined;
      try {
        tys = b.transfer([], 1);
      } catch {
        tys = undefined;
      }
      if (tys !== undefined) {
        return {
          kind: "Call",
          cName: e.name,
          name: e.name,
          args: [],
          ty: tys[0],
          span: e.span,
        };
      }
    }
    throw new UnsupportedConstruct(
      `undefined variable '${e.name}' (or unsupported reference)`,
      e.span
    );
  }

  // ── Tensor literal lowering lives in `./lowerTensorLit.ts`. ────────────

  private lowerBinary(e: Extract<Expr, { type: "Binary" }>): IRExpr {
    // Parser produces `1i` as `Mul(Number(1), ImagUnit)` (and `2.5i`
    // as `Mul(Number(2.5), ImagUnit)`). Collapse that to a single
    // `ImagLit` so the rest of the pipeline doesn't see an artificial
    // multiplication and so the resulting type carries an exact
    // `{re: 0, im: v}` for downstream folding.
    if (
      e.op === BinaryOperation.Mul &&
      e.left.type === "Number" &&
      e.right.type === "ImagUnit"
    ) {
      const v = Number(e.left.value);
      if (Number.isFinite(v)) {
        return {
          kind: "ImagLit",
          value: v,
          ty: scalarComplex({ re: 0, im: v }),
          span: e.span,
        };
      }
    }
    // Short-circuit `||` / `&&`: lower the LHS first, fold if it
    // already determines the result, and only lower the RHS when
    // needed. This matches the runtime semantics (numbl + C's
    // native short-circuit) and — load-bearing — keeps the RHS
    // from being lowered when it reads an unbound name in a dead
    // branch (e.g. `nargin < 1 || isempty(pref)` inside a 0-arg
    // spec where `pref` is unfilled).
    if (e.op === BinaryOperation.OrOr || e.op === BinaryOperation.AndAnd) {
      const lhs = this.lowerExpr(e.left);
      this.requireValueType(lhs, "binary operator operand");
      // Pull `exact` if the LHS is a scalar real numeric / logical.
      let lhsExact: number | undefined;
      if (
        isNumeric(lhs.ty) &&
        !lhs.ty.isComplex &&
        isScalarRealNumeric(lhs.ty) &&
        typeof lhs.ty.exact === "number"
      ) {
        lhsExact = lhs.ty.exact;
      }
      const isOr = e.op === BinaryOperation.OrOr;
      if (lhsExact !== undefined) {
        // NaN is excluded from BOTH folds: numbl treats a nonzero NaN as
        // truthy (`!= 0` in C, `!== 0` in JS, `toBool` in the interpreter —
        // see `truthy()` in emitJs and the `andand`/`oror` builtins), so
        // `NaN || x` and `NaN && x` must defer to the builtin's RHS
        // evaluation rather than short-circuit. Folding `NaN && x` to false
        // here diverged opt1/opt2 (→ false) from opt0 (→ x's truthiness).
        if (
          (isOr && lhsExact !== 0 && !Number.isNaN(lhsExact)) ||
          (!isOr && lhsExact === 0)
        ) {
          const folded: IRExpr = {
            kind: "NumLit",
            value: isOr ? 1 : 0,
            ty: scalarLogical(isOr),
            span: e.span,
          };
          // The LHS short-circuits the result. The LHS is ALWAYS evaluated
          // (only the RHS is skipped), so we must not discard a side-effecting
          // LHS — e.g. `sideEffect() && x` where `sideEffect()` is a folded
          // user-function call. If the LHS is side-effect-free we fold to a
          // literal (enables downstream constant use, and is load-bearing for
          // `nargin < 1 || isempty(unfilled)` in 0-arg specs). Otherwise keep
          // the LHS evaluated, paired with a literal RHS that native `&&`/`||`
          // short-circuits away — crucially WITHOUT lowering the real RHS.
          if (isSideEffectFreeExpr(lhs)) {
            return folded;
          }
          return {
            kind: "Binary",
            builtin: isOr ? "oror" : "andand",
            op: e.op,
            left: lhs,
            right: folded,
            ty: scalarLogical(isOr),
            span: e.span,
          };
        }
      }
      // RHS needs to be evaluated; defer to the standard builtin
      // path so transfer-time exact propagation still fires.
    }
    const left = this.lowerExpr(e.left);
    this.requireValueType(left, "binary operator operand");
    const right = this.lowerExpr(e.right);
    this.requireValueType(right, "binary operator operand");
    const name = binaryOpBuiltin(e.op, e.span);
    const b = getBuiltin(name);
    if (!b) {
      throw new UnsupportedConstruct(
        `builtin '${name}' not registered`,
        e.span
      );
    }
    const ty = withSpan(e.span, () => b.transfer([left.ty, right.ty], 1))[0];
    return {
      kind: "Binary",
      builtin: name,
      op: e.op,
      left,
      right,
      ty,
      span: e.span,
    };
  }

  private lowerUnary(e: Extract<Expr, { type: "Unary" }>): IRExpr {
    const operand = this.lowerExpr(e.operand);
    this.requireValueType(operand, "unary operator operand");
    // `'` (conjugate transpose) on a complex operand lowers to
    // `transpose(conj(z))` per the plan — no native ctranspose
    // helper. For real operands `'` and `.'` are identical (both
    // route to `transpose`).
    if (
      e.op === UnaryOperation.Transpose &&
      isNumeric(operand.ty) &&
      operand.ty.isComplex
    ) {
      const conjB = getBuiltin("conj");
      const transB = getBuiltin("transpose");
      if (!conjB || !transB) {
        throw new UnsupportedConstruct(
          `internal: 'conj'/'transpose' must be registered for complex ctranspose`,
          e.span
        );
      }
      const conjTy = withSpan(e.span, () => conjB.transfer([operand.ty], 1))[0];
      const conjCall: IRExpr = {
        kind: "Call",
        cName: "conj",
        name: "conj",
        args: [operand],
        ty: conjTy,
        span: e.span,
      };
      const transTy = withSpan(e.span, () => transB.transfer([conjTy], 1))[0];
      return {
        kind: "Call",
        cName: "transpose",
        name: "transpose",
        args: [conjCall],
        ty: transTy,
        span: e.span,
      };
    }
    const name = unaryOpBuiltin(e.op, e.span);
    const b = getBuiltin(name);
    if (!b) {
      throw new UnsupportedConstruct(
        `builtin '${name}' not registered`,
        e.span
      );
    }
    const ty = withSpan(e.span, () => b.transfer([operand.ty], 1))[0];
    return {
      kind: "Unary",
      builtin: name,
      op: e.op,
      operand,
      ty,
      span: e.span,
    };
  }

  // ── Bare-name calls live in `./lowerFuncCall.ts`. ─────────────────────
  // ── Class constructors live in `./lowerClassConstructor.ts`. ──────────
  // ── Function handles live in `./lowerHandle.ts`. ──────────────────────
  // ── Function specialization lives in `./specialize.ts`. ───────────────

  // ── Helpers ───────────────────────────────────────────────────────────

  private requireScalarReal(
    t: Type,
    what: string,
    span: Span,
    detail?: string
  ): void {
    if (!isScalarRealNumeric(t)) {
      const suffix = detail ? ` ${detail}` : "";
      throw new UnsupportedConstruct(
        `${what} must be a scalar real numeric${suffix}`,
        span
      );
    }
  }

  /** Like `requireScalarReal` but also accepts scalar complex —
   *  used by `if` / `while` / `elseif` conds, where MATLAB's `toBool`
   *  rule (`creal(z) != 0 || cimag(z) != 0`) gives the boolean. */
  private requireScalarCondType(t: Type, what: string, span: Span): void {
    if (
      isNumeric(t) &&
      (t.elem === "double" || t.elem === "logical") &&
      t.dims.length === 2 &&
      t.dims[0].kind === "exact" &&
      t.dims[0].value === 1 &&
      t.dims[1].kind === "exact" &&
      t.dims[1].value === 1
    ) {
      return;
    }
    throw new UnsupportedConstruct(`${what} must be a scalar numeric`, span);
  }

  private mergeBranchEnvs(
    envs: Map<string, EnvEntry>[],
    flagUnassigned = false
  ): Map<string, EnvEntry> {
    // Collect all keys present in any branch. Variables assigned in
    // only some branches stay in scope after the merge — MATLAB's rule
    // is "declared if any path declared it; reading without that path
    // having run is a runtime error". Codegen pre-declares such locals
    // at function top (see `collectHoistedScalarLocals` in emit.ts) so
    // a later C-level read survives the block's lexical scope.
    const allKeys = new Set<string>();
    for (const e of envs) for (const k of e.keys()) allKeys.add(k);
    const out = new Map<string, EnvEntry>();
    for (const k of allKeys) {
      const present = envs
        .map(e => e.get(k))
        .filter((x): x is EnvEntry => x !== undefined);
      if (present.length === 0) continue;
      let ty: Type = present[0].ty;
      for (let i = 1; i < present.length; i++) {
        ty = unify(ty, present[i].ty);
      }
      // Drop `exact` if the key is missing in any branch — the
      // runtime value can't be guaranteed to match.
      if (present.length < envs.length) {
        ty = withoutExact(ty);
      }
      // Flag a variable that some (but not all) branches assigned: a
      // later read may hit the predeclared default instead of a real
      // value, which MATLAB treats as a runtime "Undefined" error. Only
      // `if`-merges flag (flagUnassigned) — loop merges don't, to keep
      // the established loop-var / loop-body-var-after-loop behavior.
      // Always propagate an inbound flag so nested conditionals carry it.
      const maybeUnassigned =
        (flagUnassigned && present.length < envs.length) ||
        present.some(p => p.maybeUnassigned);
      const entry: EnvEntry = { cName: present[0].cName, ty };
      if (maybeUnassigned) entry.maybeUnassigned = true;
      out.set(k, entry);
    }
    return out;
  }
}

// ── Helpers (free functions) ────────────────────────────────────────────

/** Reserved words that are legal numbl identifiers but would collide
 *  with a C or JS keyword on the emit side. mtoc2 maps every user
 *  variable / param name through `cIdentForUserName` at declaration
 *  time so the same `cName` is safe in BOTH the C-AOT and JS-AOT emit
 *  paths (which share the cName — see `emitJs.ts` vs `emit.ts`). A
 *  variable named `struct` (C keyword) or `in` (JS keyword) becomes
 *  `v_struct` / `v_in` and survives `new Function(source)()` plus the
 *  C compiler. `mtoc2_` is reserved for the translator's own synthetic
 *  names. */
const C_RESERVED_NAMES: ReadonlySet<string> = new Set([
  // ── C keywords ──────────────────────────────────────────────────────
  "auto",
  "break",
  "case",
  "char",
  "const",
  "continue",
  "default",
  "do",
  "double",
  "else",
  "enum",
  "extern",
  "float",
  "for",
  "goto",
  "if",
  "inline",
  "int",
  "long",
  "register",
  "restrict",
  "return",
  "short",
  "signed",
  "sizeof",
  "static",
  "struct",
  "switch",
  "typedef",
  "union",
  "unsigned",
  "void",
  "volatile",
  "while",
  "main",
  // C99/C11 keywords seen in our runtime headers
  "_Bool",
  "_Complex",
  "_Imaginary",
  // ── JS-only keywords / reserved words ───────────────────────────────
  // The JS emit path uses the same cName, so any reserved word that
  // could appear as a user identifier needs escaping too. `in` is the
  // one that bit us in practice (chunkerinterior.m has `function [in]
  // = chunkerinterior(...)`); the rest are listed for completeness so
  // we don't regress one keyword at a time.
  "async",
  "await",
  "catch",
  "class",
  "debugger",
  "delete",
  "export",
  "extends",
  "false",
  "finally",
  "function",
  "implements",
  "import",
  "in",
  "instanceof",
  "interface",
  "let",
  "new",
  "null",
  "package",
  "private",
  "protected",
  "public",
  "super",
  "this",
  "throw",
  "true",
  "try",
  "typeof",
  "var",
  "with",
  "yield",
  // Magic JS bindings: not reserved words per se, but using either as
  // a `let`-declared local would shadow the function-scope `arguments`
  // object / blow up `eval`-bound calls. Cheap to escape.
  "arguments",
  "eval",
]);

/** Map a user-source variable name to a safe C / JS identifier.
 *  Reserved words get a `v_` prefix; other names pass through unchanged
 *  so emitted code remains readable for the common case. The numbl
 *  parser already rejects identifiers starting with `_`, so a user
 *  `for` only collides with the keyword (not with `v_for` from any
 *  other source). */
export function cIdentForUserName(name: string): string {
  if (C_RESERVED_NAMES.has(name)) return `v_${name}`;
  return name;
}

// Re-export so existing call sites (`lowerMethodCall`,
// `lowerMultiAssign`, and `lowerProgram` itself) keep working without
// import-path churn.
export { tryExtractDottedName } from "../parser/astUtils.js";

/** If the lowered cond's type carries an exact scalar value, return
 *  its boolean interpretation; otherwise null. This is the ONLY place
 *  the lowerer turns a known exact value into a compile-time decision —
 *  the resulting branch is taken/dropped before codegen. Arithmetic /
 *  comparisons / builtin calls and Ident reads all produce runtime IR
 *  even when their `ty.exact` is known. */
function condToBool(cond: IRExpr): boolean | null {
  if (!isNumeric(cond.ty)) return null;
  const x = cond.ty.exact;
  if (typeof x === "number") {
    if (!Number.isFinite(x)) return null;
    return x !== 0;
  }
  // Scalar complex exact `{re, im}` — truthy iff either part is nonzero.
  const sx = exactComplex(cond.ty);
  if (sx !== undefined) {
    if (!Number.isFinite(sx.re) || !Number.isFinite(sx.im)) return null;
    return sx.re !== 0 || sx.im !== 0;
  }
  return null;
}

/** True when `e` is guaranteed side-effect-free: NumLit, Var, EndRef,
 *  HandleCaptureLoad, and pure compositions of those via Binary /
 *  Unary. Calls (user functions and any builtin — `disp`, `tic`,
 *  bounds-checked indexing) are conservatively treated as impure.
 *  Used by `lowerIf`'s fold path: when the cond folds to a constant,
 *  pure conds are dropped entirely; impure ones get prepended as an
 *  `ExprStmt` so their side effects still run. */
function condIsPure(e: IRExpr): boolean {
  switch (e.kind) {
    case "NumLit":
    case "ImagLit":
    case "Var":
    case "EndRef":
    case "HandleCaptureLoad":
      return true;
    case "Unary":
      return condIsPure(e.operand);
    case "Binary":
      return condIsPure(e.left) && condIsPure(e.right);
    default:
      return false;
  }
}

/** Strip the surrounding `'` or `"` quotes the numbl parser stores as
 *  part of a `Char`/`String` literal's lexeme. */
export function stripQuotes(s: string): string {
  if (s.length >= 2) {
    const first = s[0];
    const last = s[s.length - 1];
    if ((first === "'" && last === "'") || (first === '"' && last === '"')) {
      return s.slice(1, -1);
    }
  }
  return s;
}

/** Refresh the env entry's `exact: Float64Array` in place after a
 *  scalar IndexStore when both the indices and the rhs are compile-
 *  time-known. Returns true on success (env updated), false if any
 *  precondition fails — the caller falls back to widening (strip
 *  exact). Only handles `IndexStore` (scalar lvalue); slice stores
 *  keep the existing strip-exact behavior. */
function tryRefreshExactAfterIndexedWrite(
  env: Map<string, { cName: string; ty: Type }>,
  name: string,
  result: IRStmt | IRStmt[]
): boolean {
  const last = Array.isArray(result) ? result[result.length - 1] : result;
  if (last.kind !== "IndexStore") return false;
  const e = env.get(name);
  if (e === undefined) return false;
  if (e.ty.kind !== "Numeric") return false;
  const ty = e.ty;
  const shape = ty.shape;
  const data = ty.exact;
  if (shape === undefined || !(data instanceof Float64Array)) return false;
  if (!isNumeric(last.rhs.ty) || typeof last.rhs.ty.exact !== "number") {
    return false;
  }
  const rhsVal = last.rhs.ty.exact;
  const idxVals: number[] = [];
  for (const ix of last.indices) {
    if (!isNumeric(ix.ty) || typeof ix.ty.exact !== "number") return false;
    const v = ix.ty.exact;
    if (!Number.isFinite(v) || !Number.isInteger(v) || v < 1) return false;
    idxVals.push(v);
  }
  const offset = columnMajorOffsetFromIndices(shape, idxVals);
  if (offset === undefined) return false;
  const newData = new Float64Array(data);
  newData[offset] = rhsVal;
  const newSign = signFromExactArray(newData);
  env.set(name, {
    cName: e.cName,
    ty: { ...ty, exact: newData, sign: newSign },
  });
  return true;
}

/** Extract the rhs sign from an IndexStore / IndexSliceStore result so
 *  `widenAfterIndexedWrite` can unify it into the base's lattice sign.
 *  The store is always the last stmt in the result (lowerIndexSliceStore
 *  may prepend hoists). */
function rhsSignFromStoreResult(result: IRStmt | IRStmt[]): Sign {
  const last = Array.isArray(result) ? result[result.length - 1] : result;
  if (last.kind !== "IndexStore" && last.kind !== "IndexSliceStore") {
    return "unknown";
  }
  return isNumeric(last.rhs.ty) ? last.rhs.ty.sign : "unknown";
}

/** Reject loops whose body changes the shape of a self-referenced
 *  variable. mtoc2 lowers the body once with the entry-iteration type,
 *  so if iteration 1 produces a different shape than iteration 0
 *  carried in (e.g. `w = [w, i]` growing `w` each pass), the emitted
 *  code uses stale shape constants and produces wrong runtime output.
 *  Detect this by comparing the env entry the body actually lowered
 *  with (`envBodyInput[n]`) against the post-body env entry
 *  (`envAfter[n]`) for every variable assigned inside the body. */
function assertLoopBodyShapeStable(
  envBodyInput: Map<string, EnvEntry>,
  envAfter: Map<string, EnvEntry>,
  bodyStmts: Stmt[],
  loopSpan: Span
): void {
  for (const name of collectAssignedNames(bodyStmts)) {
    const before = envBodyInput.get(name);
    const after = envAfter.get(name);
    if (before === undefined || after === undefined) continue;
    if (numericShapeChanged(before.ty, after.ty)) {
      throw new UnsupportedConstruct(
        `loop body changes the shape of '${name}' across iterations ` +
          `(entry shape differs from end-of-iter shape); ` +
          `mtoc2 lowers the body once per spec and cannot specialize ` +
          `per-iteration growing tensors`,
        loopSpan
      );
    }
  }
}

/** True iff `a` and `b` are both Numeric types whose shape information
 *  is observably different. Compares both the lattice `dims` (exact
 *  values only — `unknown` matches anything) and the concrete `shape`
 *  array when present. Non-numeric types and same-shape types return
 *  false. */
function numericShapeChanged(a: Type, b: Type): boolean {
  if (!isNumeric(a) || !isNumeric(b)) return false;
  if (a.dims.length !== b.dims.length) return true;
  for (let i = 0; i < a.dims.length; i++) {
    const da = a.dims[i];
    const db = b.dims[i];
    if (da.kind === "exact" && db.kind === "exact" && da.value !== db.value) {
      return true;
    }
    // Pre-body had an exact dim but post-body widened to unknown:
    // iteration 2 would feed in a value whose actual shape doesn't
    // match the body's emitted shape constants.
    if (da.kind === "exact" && db.kind === "unknown") return true;
  }
  const sa = a.shape;
  const sb = b.shape;
  if (sa !== undefined && sb !== undefined) {
    if (sa.length !== sb.length) return true;
    for (let i = 0; i < sa.length; i++) {
      if (sa[i] !== sb[i]) return true;
    }
  } else if (sa !== undefined && sb === undefined) {
    return true;
  }
  return false;
}

function collectAssignedNames(stmts: Stmt[]): Set<string> {
  const out = new Set<string>();
  // Add the root variable name of any LValue that points at a
  // mutable binding (Var, or any chain rooted at a Var via
  // Member / Index). Used uniformly by Assign / AssignLValue /
  // MultiAssign so the loop-entry exact-strip stays in lockstep
  // across all assignment forms.
  const addLValueRoot = (lv: LValue): void => {
    if (lv.type === "Var") {
      out.add(lv.name);
      return;
    }
    if (lv.type === "Member" || lv.type === "Index") {
      let cur: Expr | null = lv.base;
      while (cur !== null && cur.type === "Member") cur = cur.base;
      if (cur !== null && cur.type === "Ident") out.add(cur.name);
    }
    // Ignore / IndexCell / MemberDynamic are either non-mutating
    // (Ignore) or rejected at lowering with an UnsupportedConstruct.
    // Either way the loop-widen pass doesn't need them.
  };
  const walk = (ss: Stmt[]): void => {
    for (const s of ss) {
      switch (s.type) {
        case "Assign":
          out.add(s.name);
          break;
        case "AssignLValue":
          // `s.f.g = rhs` and `s(i) = rhs` inside a loop body both
          // mutate `s` — so the loop entry needs to strip exact from
          // `s` (and its recursive struct/class field types) just
          // like a plain `s = ...` reassignment would.
          addLValueRoot(s.lvalue);
          break;
        case "MultiAssign":
          // Each output slot can be a Var, an Ignore, OR a Member /
          // Index lvalue rooted at a var (`[a, b.f] = ...`,
          // `[v(i), w] = ...`). The exact-strip pass needs to walk
          // ALL forms so a per-slot Member / Index write doesn't
          // sneak past the loop-entry widen.
          for (const lv of s.lvalues) addLValueRoot(lv);
          break;
        case "If":
          walk(s.thenBody);
          for (const eb of s.elseifBlocks) walk(eb.body);
          if (s.elseBody) walk(s.elseBody);
          break;
        case "While":
          walk(s.body);
          break;
        case "For":
          out.add(s.varName);
          walk(s.body);
          break;
        case "Switch":
          for (const c of s.cases) walk(c.body);
          if (s.otherwise) walk(s.otherwise);
          break;
        case "TryCatch":
          walk(s.tryBody);
          walk(s.catchBody);
          break;
      }
    }
  };
  walk(stmts);
  return out;
}
