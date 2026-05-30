/**
 * Bare-name function-call lowering: `name(args)`.
 *
 * Resolution priority — checked in this order:
 *   0. The literal name `struct` (no shadowing local) → `StructLit`.
 *   1. The literal name `bsxfun` (no shadowing local) → rewrite to
 *      `fn(A, B)` when the first arg is `@<knownElementwiseBinary>`.
 *   2. An in-scope `HandleType` variable → `dispatchHandleCall`.
 *   3. An in-scope class name with no shadowing local → class
 *      constructor.
 *   4. An in-scope variable: multi-element numeric routes through
 *      the index helpers; other types raise UnsupportedConstruct.
 *   5. Zero-arity builtin paren-form (`pi()`, `Inf()`).
 *   6. Numbl resolver verdict — builtin / user-function / class
 *      method.
 *   7. Fallback to mtoc2's own builtin registry for plot-drawing
 *      builtins numbl wires through runtime dispatch.
 */

import type { Expr } from "../parser/index.js";
import { TypeError, UnsupportedConstruct } from "./errors.js";
import type { IRExpr } from "./ir.js";
import {
  DIM_ONE,
  EXACT_ARRAY_MAX_ELEMENTS,
  MTOC2_MAX_NDIM,
  cellTuple,
  cellUniform,
  classMethodSpecSource,
  emptyDoubleTensorType,
  isHandle,
  isMultiElement,
  isNumeric,
  isScalar,
  structType,
  typeToString,
} from "./types.js";
import type { DimInfo, Type } from "./types.js";
import { getBuiltin } from "../builtins/index.js";
import { withSpan } from "./errors.js";
import { isSliceArg } from "./indexResolve.js";
import { lowerIndexLoad } from "./lowerIndexLoad.js";
import { lowerIndexSlice } from "./lowerIndexSlice.js";
import { lowerClassConstructorCall } from "./lowerClassConstructor.js";
import { dispatchHandleCall } from "./lowerHandle.js";
import { buildUserFunctionCall } from "./specialize.js";
import type { Lowerer } from "./lower.js";
import { stripQuotes } from "./lower.js";

export function lowerFuncCall(
  this: Lowerer,
  e: Extract<Expr, { type: "FuncCall" }>
): IRExpr {
  // Look up the env BEFORE the `struct(...)` constructor shortcut so
  // `struct = [1 2 3]; struct(2)` reads the local (yields `2`) rather
  // than dispatching to the `struct(...)` constructor and erroring on
  // "expects an even number of args". MATLAB precedence is env >
  // builtin; this honors it for the one builtin name that has
  // special-cased lowering. Other in-scope-variable cases (handle,
  // multi-element numeric, scalar) are handled below.
  const envEntry = this.env.get(e.name);
  if (envEntry === undefined && e.name === "struct") {
    return lowerStructConstructor.call(this, e);
  }
  if (envEntry === undefined && e.name === "cell") {
    return lowerCellConstructor.call(this, e);
  }
  if (envEntry === undefined && e.name === "bsxfun") {
    const rewritten = tryRewriteBsxfun.call(this, e);
    if (rewritten !== null) return rewritten;
  }
  if (envEntry === undefined && e.name === "feval") {
    return lowerFevalCall.call(this, e);
  }
  if (envEntry !== undefined && isHandle(envEntry.ty)) {
    return dispatchHandleCall.call(this, e.name, envEntry, e.args, e.span);
  }
  if (envEntry === undefined && this.workspace.isClass(e.name)) {
    const reg = this.workspace.requireClass(e.name);
    if (reg === undefined) {
      throw new UnsupportedConstruct(
        `internal: class '${e.name}' missing from workspace registry`,
        e.span
      );
    }
    return lowerClassConstructorCall.call(this, reg, e.args, e.span);
  }
  if (envEntry !== undefined) {
    // MATLAB's "workspace shadows functions" rule: `v(i)` reads as an
    // indexed access when `v` is in scope. Multi-element numeric
    // bases route through the index helpers; scalar variables get a
    // clearer error than "unknown function". Other types (handle is
    // handled above, struct / class / string) keep the existing
    // "cannot be called" error.
    if (isNumeric(envEntry.ty) && isMultiElement(envEntry.ty)) {
      if (e.args.some(isSliceArg)) {
        return lowerIndexSlice.call(this, e.name, e.args, e.span);
      }
      return lowerIndexLoad.call(this, e.name, e.args, e.span);
    }
    throw new UnsupportedConstruct(
      `'${e.name}' is an in-scope variable of type ` +
        `${typeToString(envEntry.ty)}; cannot be called as a function ` +
        `(scalar indexing and dynamically-typed handles are not supported)`,
      e.span
    );
  }

  const args = e.args.map(a => this.lowerExpr(a));
  for (const a of args) {
    this.requireValueType(a, `argument to '${e.name}'`);
  }
  const argTypes = args.map(a => a.ty);

  // Zero-arity mtoc2 builtins like `pi()` / `Inf()` / `NaN()`. Numbl
  // resolves these through a separate constants table
  // (`BUILTIN_CONSTANTS`) not in `index.builtins`, so
  // `workspace.resolve` returns null. The bare-name read path in
  // `lowerIdent` already handles `pi` (no parens); this branch
  // handles the paren-form. `e.args.length === 0` is the gate so we
  // don't accidentally claim a 1-arg call like `pi(2,3)` (which
  // numbl/MATLAB treat as a fill constructor — out of scope for
  // mtoc2 v1).
  if (args.length === 0) {
    const b = getBuiltin(e.name);
    if (b !== undefined) {
      // Probe: if `transfer([], 1)` accepts, treat as a 0-arg call.
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
  }

  const target = this.workspace.resolve(
    e.name,
    argTypes,
    this.callSite(),
    e.span
  );
  if (!target) {
    // Fall back to mtoc2's builtin registry when numbl exposes the
    // name via a non-index surface (e.g. plot drawing primitives like
    // `plot`/`surf`/`imagesc`/`bar`, which numbl wires through its
    // runtime dispatch rather than `index.builtins`). The
    // validate-then-route shape is identical to the standard builtin
    // branch below; we just don't have numbl's blessing.
    const fallback = getBuiltin(e.name);
    if (fallback !== undefined) {
      const ty = withSpan(e.span, () => fallback.transfer(argTypes, 1))[0];
      return {
        kind: "Call",
        cName: e.name,
        name: e.name,
        args,
        ty,
        span: e.span,
      };
    }
    throw new UnsupportedConstruct(`unknown function '${e.name}'`, e.span);
  }
  switch (target.kind) {
    case "builtin": {
      // Numbl agreed it's a builtin; mtoc2 still requires the builtin
      // to be registered in its own table.
      const b = getBuiltin(e.name);
      if (!b) {
        throw new UnsupportedConstruct(
          `builtin '${e.name}' is not supported by mtoc2`,
          e.span
        );
      }
      const ty = withSpan(e.span, () => b.transfer(argTypes, 1))[0];
      return {
        kind: "Call",
        cName: e.name,
        name: e.name,
        args,
        ty,
        span: e.span,
      };
    }
    case "userFunction": {
      // Expression-context: request nargout=1 (the call site's single
      // lvalue). A multi-output declared function specializes with
      // truncated output list — see `buildUserFunctionCall`.
      return buildUserFunctionCall.call(
        this,
        target.ast,
        args,
        e.name,
        e.span,
        {
          definingFile: target.file,
        }
      );
    }
    case "classMethod": {
      // `method(obj, args)` syntax — the resolver decided this name
      // is a class method because one of the arg types is a
      // ClassInstance. Route through the same path as the dot form.
      const reg = this.classReg(target.className);
      if (reg === undefined) {
        throw new UnsupportedConstruct(
          `internal: class '${target.className}' missing from workspace registry`,
          e.span
        );
      }
      const method = target.stripInstance
        ? reg.staticMethods.get(target.methodName)
        : reg.methods.get(target.methodName);
      if (method === undefined) {
        throw new TypeError(
          `class '${target.className}' has no ${target.stripInstance ? "static " : ""}method '${target.methodName}'`,
          e.span
        );
      }
      if (method.outputs.length >= 2) {
        throw new UnsupportedConstruct(
          `class method '${target.className}.${target.methodName}' has ` +
            `${method.outputs.length} outputs; multi-output methods can ` +
            `only be called via '[a, b, ...] = ...' (not yet supported ` +
            `for class methods) or as a bare statement`,
          e.span
        );
      }
      const callArgs = target.stripInstance ? args.slice(1) : args;
      return buildUserFunctionCall.call(
        this,
        method,
        callArgs,
        `${target.className}.${target.methodName}`,
        e.span,
        {
          specSource: classMethodSpecSource(
            target.className,
            target.methodName
          ),
          definingFile: method.span.file ?? reg.file,
        }
      );
    }
    case "classConstructor": {
      // Shouldn't fire because we short-circuit above on
      // `isClass(name)`, but kept for completeness.
      const reg = this.classReg(target.className);
      if (reg === undefined) {
        throw new UnsupportedConstruct(
          `internal: class '${target.className}' missing from workspace registry`,
          e.span
        );
      }
      return lowerClassConstructorCall.call(this, reg, e.args, e.span);
    }
    case "mtoc2UserFunction": {
      // A `.mtoc2.js` user function. The evaluated `Builtin` lives on
      // the workspace; route through its `transfer` exactly like a
      // global builtin call. `emit` runs at codegen time via the
      // workspace lookup the emitter consults (see
      // `lookupBuiltinForEmit` in `src/codegen/runtime.ts`).
      const userBuiltin = this.workspace.getUserBuiltin(target.name);
      if (!userBuiltin) {
        throw new UnsupportedConstruct(
          `internal: workspace resolved '${target.name}' to a .mtoc2.js ` +
            `function but no Builtin is loaded`,
          e.span
        );
      }
      const ty = withSpan(e.span, () => userBuiltin.transfer(argTypes, 1))[0];
      return {
        kind: "Call",
        cName: e.name,
        name: e.name,
        args,
        ty,
        span: e.span,
      };
    }
  }
}

/** `feval(handle_or_name, args...)` — invoke a function handle or a
 *  function named by a char literal. Rewrites the call into a direct
 *  call on the underlying name, then recursively lowers that:
 *    - `feval(@foo, x, y)`  → `foo(x, y)` (FuncHandle literal)
 *    - `feval(f, x, y)`      → `f(x, y)` where `f` is an in-scope
 *                              handle variable (Ident path; the
 *                              recursive lowerFuncCall picks the
 *                              env-handle dispatch branch)
 *    - `feval('foo', x, y)`  → `foo(x, y)` (Char literal)
 *
 *  Runtime-computed handle expressions (`feval(get_h(), ...)`,
 *  `feval(handles{k}, ...)`, inline `@(x) ...`) are rejected with
 *  `UnsupportedConstruct` — mtoc2's AOT call sites are specialized
 *  at lowering time, so the handle's target name must be statically
 *  visible. The user-level workaround is to bind the value to a
 *  local first (`f = ...; feval(f, ...)`). */
function lowerFevalCall(
  this: Lowerer,
  e: Extract<Expr, { type: "FuncCall" }>
): IRExpr {
  const synth = rewriteFevalToDirectCall(e);
  return lowerFuncCall.call(this, synth);
}

/** Shared AST rewrite used by both `lowerFevalCall` (single-output
 *  expression context) and the multi-assign path (`lowerMultiAssign`
 *  calls this to rewrite the RHS before its own dispatch). Returns a
 *  fresh `FuncCall` node naming the resolved target with `args[0]`
 *  consumed. */
export function rewriteFevalToDirectCall(
  e: Extract<Expr, { type: "FuncCall" }>
): Extract<Expr, { type: "FuncCall" }> {
  if (e.args.length < 1) {
    throw new UnsupportedConstruct(
      `'feval' expects at least 1 argument (the function handle or name), ` +
        `got 0`,
      e.span
    );
  }
  const first = e.args[0];
  const rest = e.args.slice(1);
  let targetName: string;
  switch (first.type) {
    case "FuncHandle":
      // `feval(@foo, args)` → `foo(args)`.
      targetName = first.name;
      break;
    case "Ident":
      // `feval(f, args)` where `f` is in scope as a handle variable.
      // The recursive lowerFuncCall will check `env.get(f)` and route
      // through `dispatchHandleCall`. If `f` is bound to a non-handle
      // (number, tensor, …) the recursive call surfaces the same
      // "cannot be called" message the user would see for `f(args)`.
      targetName = first.name;
      break;
    case "Char":
    case "String":
      // `feval('foo', args)` / `feval("foo", args)` — the numbl
      // parser keeps the surrounding quotes in `.value`. Strip them
      // and validate identifier shape so we don't synthesize a
      // FuncCall whose `name` carries quotes / spaces / dots.
      targetName = stripQuotes(first.value);
      if (!/^[A-Za-z_][A-Za-z0-9_.]*$/.test(targetName)) {
        throw new UnsupportedConstruct(
          `'feval' name argument '${targetName}' is not a valid function ` +
            `identifier`,
          first.span
        );
      }
      break;
    case "AnonFunc":
      throw new UnsupportedConstruct(
        `'feval' on an inline anonymous handle ('@(...) ...') is not ` +
          `supported; bind the handle to a local variable first ` +
          `(f = @(...) ...; feval(f, ...))`,
        e.span
      );
    default:
      throw new UnsupportedConstruct(
        `'feval' first argument must be a function-handle literal (@name), ` +
          `an in-scope handle variable, or a char/string function name; ` +
          `got '${first.type}' (runtime-computed handle expressions are ` +
          `not supported — bind to a local first)`,
        first.span
      );
  }
  return {
    type: "FuncCall",
    name: targetName,
    args: rest,
    span: e.span,
  };
}

/** `bsxfun(@fn, A, B)` — when `@fn` is a function-handle literal
 *  whose name is one of the elementwise binary builtins, rewrite to
 *  `fn(A, B)` and let the existing implicit-expansion path do the
 *  work. Returns the lowered IR on success, or `null` to fall through
 *  to the generic call path (which will surface a clearer error for
 *  unsupported handle targets). Custom function-handle bsxfun is
 *  deferred. */
function tryRewriteBsxfun(
  this: Lowerer,
  e: Extract<Expr, { type: "FuncCall" }>
): IRExpr | null {
  if (e.args.length !== 3) return null;
  const handleArg = e.args[0];
  if (handleArg.type !== "FuncHandle") {
    throw new UnsupportedConstruct(
      `'bsxfun' first arg must be a function-handle literal (e.g. @times); ` +
        `dynamic handle-value bsxfun is not yet supported`,
      e.span
    );
  }
  const handleName = handleArg.name;
  const knownOps = new Set([
    "plus",
    "minus",
    "times",
    "rdivide",
    "power",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "mod",
    "rem",
    "atan2",
    "hypot",
    "max",
    "min",
  ]);
  if (!knownOps.has(handleName)) {
    throw new UnsupportedConstruct(
      `'bsxfun' with handle target '@${handleName}' is not supported; ` +
        `supported targets: ${[...knownOps].sort().join(", ")}`,
      e.span
    );
  }
  const synthCall: Extract<Expr, { type: "FuncCall" }> = {
    type: "FuncCall",
    name: handleName,
    args: [e.args[1], e.args[2]],
    span: e.span,
  };
  return lowerFuncCall.call(this, synthCall);
}

/** `struct('f1', v1, 'f2', v2, ...)`. Validates that args come in
 *  (string-literal-name, value) pairs and that no field is
 *  duplicated. Each value's storage type drives the field's recorded
 *  type — typedef shape is stable across writes because storage types
 *  are widened (no `exact`, no `sign`). */
function lowerStructConstructor(
  this: Lowerer,
  e: Extract<Expr, { type: "FuncCall" }>
): IRExpr {
  if (e.args.length % 2 !== 0) {
    throw new TypeError(
      `'struct' expects an even number of args (name, value, name, value, ...)`,
      e.span
    );
  }
  const fields: { name: string; value: IRExpr }[] = [];
  const seen = new Set<string>();
  for (let i = 0; i < e.args.length; i += 2) {
    const nameExpr = e.args[i];
    if (nameExpr.type !== "String" && nameExpr.type !== "Char") {
      throw new TypeError(
        `'struct' field name (arg ${i + 1}) must be a string or char literal`,
        nameExpr.span
      );
    }
    // numbl's parser stores the literal's source text (including the
    // surrounding `'`/`"` quotes) in `value`. Strip them so the
    // recorded field name matches the user-visible name. Also
    // require a non-empty, identifier-shaped field name (no embedded
    // quotes/escapes etc).
    const fname = stripQuotes(nameExpr.value);
    if (!/^[A-Za-z][A-Za-z0-9_]*$/.test(fname)) {
      throw new TypeError(
        `'struct' field name '${fname}' is not a valid identifier`,
        nameExpr.span
      );
    }
    if (seen.has(fname)) {
      throw new TypeError(
        `'struct': duplicate field '${fname}'`,
        nameExpr.span
      );
    }
    seen.add(fname);
    const v = this.lowerExpr(e.args[i + 1]);
    this.requireValueType(v, `value for field '${fname}'`);
    // Only types that have a stable owned-or-POD C representation
    // are allowed as struct field values. Reject handles (POD but
    // their typedef matrix gets messy), void, and Unknown.
    if (v.ty.kind === "Void" || v.ty.kind === "Unknown") {
      throw new TypeError(
        `value for field '${fname}': type ${typeToString(v.ty)} is not a valid struct field type`,
        e.args[i + 1].span
      );
    }
    fields.push({ name: fname, value: v });
  }
  // Build the StructType from each value's precise type. The typedef
  // hash uses `cFieldTypeStr` (one C-type string per field), so
  // different `exact` / `sign` / tensor-shape values across
  // constructions still share one C typedef. Carrying the precise
  // type through the IR lets a subsequent `aa = s.x` read return e.g.
  // `double[1×1]:positive=1` instead of a sign-stripped form.
  const tyFields = fields.map(f => ({
    name: f.name,
    ty: f.value.ty,
  }));
  const ty = structType(tyFields);
  // `StructLit.fields` must line up with `ty.fields`; both now keep the
  // construction (insertion) order, which is what MATLAB observes for
  // field display / fieldnames. (Previously both were sorted by name.)
  return {
    kind: "StructLit",
    fields,
    ty,
    span: e.span,
  };
}

/** `cell(n)` / `cell(n, m)` / `cell(n, m, k, ...)`. Builds a `CellEmpty`
 *  IR node whose `ty` is a `CellType` whose mode depends on whether
 *  every dim is exact and the total slot count fits the exact cap:
 *    - all dims exact + total <= EXACT_ARRAY_MAX_ELEMENTS → tuple mode
 *      with one empty-double slot per index;
 *    - otherwise → uniform mode with empty-double elem.
 *
 *  Numbl reference: `interpreter/builtins/type-constructors.ts:442-469`
 *  initialises each slot as `RTV.tensor(allocFloat64Array(0), [0, 0])`. */
function lowerCellConstructor(
  this: Lowerer,
  e: Extract<Expr, { type: "FuncCall" }>
): IRExpr {
  if (e.args.length === 0) {
    // `cell()` — match numbl which treats it as cell(0) (0×0). The
    // common idiom is `cell(n)` though, so this path is rare.
    return {
      kind: "CellEmpty",
      dims: [],
      ty: cellTuple([0, 0], []),
      span: e.span,
    };
  }
  if (e.args.length > MTOC2_MAX_NDIM) {
    throw new UnsupportedConstruct(
      `'cell' accepts up to ${MTOC2_MAX_NDIM} dim arguments (got ${e.args.length})`,
      e.span
    );
  }

  // Lower each dim arg; require scalar real numeric.
  const dimExprs: IRExpr[] = e.args.map(a => {
    const v = this.lowerExpr(a);
    this.requireValueType(v, "'cell' dim argument");
    if (!isNumeric(v.ty) || !isScalar(v.ty) || v.ty.isComplex) {
      throw new TypeError(
        `'cell' dim argument must be a real scalar (got ${typeToString(v.ty)})`,
        a.span
      );
    }
    return v;
  });

  // Per-axis resolution: each axis becomes either exact (value known) or
  // dynamic. `cell(n)` expands to an `n×n` square; we mirror by feeding
  // dim 0's value to both axes when only one dim arg is supplied.
  const axesIn: { exactValue: number | null; expr: IRExpr }[] = [];
  if (dimExprs.length === 1) {
    const exactValue =
      isNumeric(dimExprs[0].ty) && typeof dimExprs[0].ty.exact === "number"
        ? dimExprs[0].ty.exact
        : null;
    axesIn.push({ exactValue, expr: dimExprs[0] });
    axesIn.push({ exactValue, expr: dimExprs[0] });
  } else {
    for (const de of dimExprs) {
      const exactValue =
        isNumeric(de.ty) && typeof de.ty.exact === "number"
          ? de.ty.exact
          : null;
      axesIn.push({ exactValue, expr: de });
    }
  }

  // Validate any exact-known dim is a non-negative integer.
  for (let i = 0; i < axesIn.length; i++) {
    const v = axesIn[i].exactValue;
    if (v !== null) {
      if (!Number.isInteger(v) || v < 0) {
        throw new TypeError(
          `'cell' dim argument ${i + 1} must be a non-negative integer (got ${v})`,
          e.span
        );
      }
    }
  }

  // Decide tuple vs uniform mode.
  const allExact = axesIn.every(a => a.exactValue !== null);
  let total: number | null = null;
  if (allExact) {
    total = 1;
    for (const a of axesIn) total *= a.exactValue as number;
  }
  const useTuple =
    allExact && total !== null && total <= EXACT_ARRAY_MAX_ELEMENTS;

  if (useTuple) {
    const shape = axesIn.map(a => a.exactValue as number);
    const slotCount = total!;
    const elements: Type[] = [];
    for (let i = 0; i < slotCount; i++) elements.push(emptyDoubleTensorType());
    return {
      kind: "CellEmpty",
      dims: axesIn.map(a => a.expr),
      ty: cellTuple(shape, elements),
      span: e.span,
    };
  }
  // Uniform mode — every axis is exact or unknown; build a DimInfo per axis.
  const dims: DimInfo[] = axesIn.map(a => {
    if (a.exactValue === null) return { kind: "unknown" };
    if (a.exactValue === 1) return DIM_ONE;
    return { kind: "exact", value: a.exactValue };
  });
  return {
    kind: "CellEmpty",
    dims: axesIn.map(a => a.expr),
    ty: cellUniform(dims, emptyDoubleTensorType()),
    span: e.span,
  };
}
