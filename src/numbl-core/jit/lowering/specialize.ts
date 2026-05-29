/**
 * User-function specialization machinery.
 *
 * `specializeUserFunction` is the single chokepoint that turns a
 * parsed `function` AST into an `IRFunc` against a specific arg-type
 * tuple, with the spec cache keyed on `(file, argTypes, nargout)`.
 * Every expression-context user-function call (bare-name, packaged
 * `pkg.foo`, instance / static class methods, method-via-arg-type,
 * handle dispatch) routes through `buildUserFunctionCall`, which
 * calls `specializeUserFunction` for the spec lookup and synthesizes
 * the matching `Call` IR node.
 *
 * Lives in its own module so the new user-defined `.js` function
 * feature has a clear integration surface: a sidecar can either
 * register a builtin (which doesn't need this path at all) or
 * synthesize a `decl: FuncStmt` and reuse the existing spec discipline.
 */

import type { Stmt } from "../parser/index.js";
import type { Span } from "../parser/index.js";
import { TypeError, UnsupportedConstruct } from "./errors.js";
import type { IRExpr, IRFunc, IRStmt } from "./ir.js";
import {
  type Type,
  canonicalizeType,
  hashType,
  sanitizeCIdent,
  UNKNOWN,
  VOID,
} from "./types.js";
import type { Lowerer } from "./lower.js";
import { cIdentForUserName } from "./lower.js";

/** Per-source-function specialization cap. Exact-value tracking
 *  shards each user function's spec cache by arg value, so a call
 *  site like `for k = 1:1000; total = total + sq(k); end` (without
 *  loop-body widening of `k`) would otherwise create 1000 specs and
 *  bloat the emitted C/JS. The cap is a safety net — well-typed
 *  programs sit far below it; pathological exact-tracking explosions
 *  abort with a clear error pointing at the offending function.
 *
 *  Read once at module load from `MTOC2_MAX_SPECS_PER_FUNCTION` so a
 *  user with a legitimate dense-specialization need can raise it
 *  without recompiling. Defaults to 256. */
const MAX_SPECS_PER_FUNCTION: number = (() => {
  // `process` doesn't exist in the browser bundle — `globalThis.process`
  // is undefined there. Guard so the module loads in both environments.
  const env =
    typeof globalThis !== "undefined" &&
    (globalThis as { process?: { env?: Record<string, string | undefined> } })
      .process?.env;
  const raw = env ? env.MTOC2_MAX_SPECS_PER_FUNCTION : undefined;
  if (!raw) return 256;
  const n = Number.parseInt(raw, 10);
  if (Number.isFinite(n) && n > 0) return n;
  return 256;
})();

type FuncStmt = Extract<Stmt, { type: "Function" }>;

/** Build a single-output `Call` IR node against a user-function AST,
 *  specializing the body on the arg-type tuple. Single chokepoint for
 *  every expression-context user-function call: bare-name calls,
 *  packaged (`pkg.foo`) calls, instance / static class methods,
 *  method-call-via-arg-type, and handle dispatch. Each caller is
 *  still responsible for the verdict-specific bookkeeping (resolver
 *  routing, prepending the receiver to args for instance methods,
 *  rejecting >=2-output methods if its dispatch path doesn't yet
 *  support truncation, …) — but the spec lookup + output-type
 *  derivation + IR construction is uniform here.
 *
 *  For 0-output declarations the resulting `Call` has `ty = VOID`,
 *  which the caller (`lowerExprStmt`) must accept as bare-statement
 *  use only. For >=1-output declarations the spec truncates to
 *  nargout=1 and the result type is the (possibly Unknown if the
 *  body didn't assign the output) first declared output.
 *
 *  The N≥2-output path uses a separate `MultiAssignCall` IR node
 *  built by `lowerMultiAssign` — this helper is single-output only. */
export function buildUserFunctionCall(
  this: Lowerer,
  decl: FuncStmt,
  callArgs: IRExpr[],
  callName: string,
  span: Span,
  opts: {
    specSource?: string;
    definingFile?: string;
  } = {}
): IRExpr {
  const argTypes = callArgs.map(a => a.ty);
  const nargout = decl.outputs.length === 0 ? 0 : 1;
  const spec = specializeUserFunction.call(
    this,
    decl,
    argTypes,
    opts.specSource,
    opts.definingFile,
    undefined,
    nargout,
    span
  );
  const ty: Type =
    decl.outputs.length === 0
      ? VOID
      : (spec.outputTypes[0] ?? { kind: "Unknown" });
  return {
    kind: "Call",
    cName: spec.cName,
    name: callName,
    args: callArgs,
    ty,
    span,
  };
}

/** Specialize a user function (or method, or anonymous-function
 *  synth) on the given arg-type tuple. The C mangling salts by the
 *  defining file so two files defining a same-named subfunction get
 *  distinct mangled names.
 *
 *  Caller is responsible for passing `definingFile` — for top-level
 *  functions resolved through the workspace, that's the resolver's
 *  verdict file; for class methods it's the class's file; for
 *  anonymous-function synth ASTs it's the file where `@(...)` was
 *  written. */
export function specializeUserFunction(
  this: Lowerer,
  decl: FuncStmt,
  argTypes: Type[],
  /** Optional override for the specialization-key source-name half.
   *  Class methods pass `<className>__<methodName>` so the mangled C
   *  name disambiguates two methods of the same source-level name
   *  on different classes. Defaults to `decl.name`. */
  specSource?: string,
  /** File the function definition lives in. Salts the spec key so
   *  cross-file homonyms get distinct C names. Defaults to the
   *  function's source span's file. */
  definingFile?: string,
  /** When set, the named output gets a synthetic first assignment to
   *  `initExpr` (an already-lowered IR expression) prepended to the
   *  body. The user's constructor body then sees the receiver
   *  initialized with the class defaults. */
  preSeedOutput?: { name: string; ty: Type; initExpr: IRExpr },
  /** Per-call-site `nargout`: the number of outputs the caller
   *  requested. Salts the spec key so two callers requesting
   *  different output counts get distinct specializations. Defaults
   *  to `decl.outputs.length` (the declared count) when the caller
   *  can't supply a more specific value (e.g. cross-file resolver
   *  paths that don't yet thread this through). Inside the body, the
   *  `nargout` identifier folds to this value via the
   *  `callFrameStack`. */
  nargout?: number,
  /** Span of the call site, used to attribute arity / output-
   *  assignment errors. Defaults to `decl.span` (the function
   *  definition) when omitted — but every translation-time call site
   *  should supply its own span so the user sees the bad call, not
   *  the definition. */
  callSiteSpan?: Span
): IRFunc {
  const errSpan = callSiteSpan ?? decl.span;
  // MATLAB allows `~` as a placeholder parameter name ("ignore this
  // argument"). It isn't a valid identifier in the target language
  // and would emit broken JS/C if we let it through to codegen. The
  // value isn't readable inside the body either (any reference to
  // `~` as an Ident is a parse error), so there's nothing to lower —
  // bail out so the caller (numbl JIT bridge or another translation
  // entry point) can route the call through its interpreter.
  for (const p of decl.params) {
    if (p === "~") {
      throw new UnsupportedConstruct(
        `function '${decl.name}' has a '~' placeholder parameter — ` +
          `cannot specialize`,
        errSpan
      );
    }
  }
  // Allow calling with FEWER arguments than declared (the standard
  // MATLAB pattern: `function y = f(a, b); if nargin < 2; b = ...;`).
  // The unfilled trailing params are left unbound in env; the body
  // must initialize them before any read (typically via an
  // `if nargin < N` arm). Calls with too MANY args still reject.
  if (argTypes.length > decl.params.length) {
    throw new TypeError(
      `function '${decl.name}' expects ${decl.params.length} arg(s), got ${argTypes.length}`,
      errSpan
    );
  }
  const source = specSource ?? decl.name;
  const file = definingFile ?? decl.span.file ?? this.currentFile;
  // Per-specialization `nargout`: defaults to the declared count so
  // resolver paths that don't yet thread the caller's request still
  // produce a working specialization (matches numbl's "max possible
  // nargout" interpretation when the call site isn't statically
  // known). Callers that DO know — `lowerMultiAssign`,
  // `lowerFuncCall`, ExprStmt drop-all — supply the precise count so
  // the spec key shards correctly.
  const effectiveNargout = nargout ?? decl.outputs.length;
  // Hash the (file, argTypes, nargout) triple together so the C name
  // salts by all three. Keep the human-readable prefix
  // (`apply__<hex>`) — the hash collapses everything that doesn't
  // matter.
  const hashInput = `${file}|${argTypes.map(canonicalizeType).join("|")}|nargout=${effectiveNargout}`;
  const key = `${sanitizeCIdent(source)}__${hashType(hashInput)}`;
  const cached = this.specializations.get(key);
  if (cached) {
    // A still-empty placeholder means we're hitting the spec while
    // its outer specializeUserFunction is still lowering the body —
    // i.e. a recursive self-call (or mutual-recursive entry to the
    // same key from a sibling specializer that's still pending).
    // Track so the outer call can re-lower the body once the final
    // outputTypes are known, swapping out the heuristic seed.
    if (cached.body.length === 0) {
      this.recursiveSpecsConsumed.add(key);
    }
    return cached;
  }

  // Cap distinct specializations per source-level function. Each
  // exact-tracked literal arg shards into its own spec, so a tight
  // value-keyed call site (e.g. unrolled `sq(1) + sq(2) + ... + sq(N)`)
  // can produce hundreds of specs and bloat the emitted code. Above
  // the cap, abort with a clear error pointing at the offending
  // function and span; users with a legitimate need can raise the cap
  // via `MTOC2_MAX_SPECS_PER_FUNCTION`. The key is `<source>|<file>`
  // so a homonymous function in two files counts independently.
  const countKey = `${source}|${file}`;
  const priorCount = this.specCountPerSource.get(countKey) ?? 0;
  if (priorCount >= MAX_SPECS_PER_FUNCTION) {
    throw new UnsupportedConstruct(
      `function '${decl.name}' has exceeded the per-function ` +
        `specialization cap (${MAX_SPECS_PER_FUNCTION}). This usually ` +
        `means exact-value tracking is sharding a call site into one ` +
        `spec per distinct input; widen the inputs (e.g. via ` +
        `'%!numbl:opaque') or raise the cap via the ` +
        `MTOC2_MAX_SPECS_PER_FUNCTION env var.`,
      errSpan
    );
  }
  this.specCountPerSource.set(countKey, priorCount + 1);

  // Per-spec output list: truncate to the caller's requested nargout.
  // A 3-output function called as `[a] = f(...)` or `x = f(...)`
  // becomes a 1-output specialization (single-output C ABI); a bare
  // `f(...)` becomes a 0-output (void) spec. The body's assignments
  // to trailing outputs are kept but unused — the nargout fold may
  // dead-code them via `if nargout >= N` branches.
  const effectiveOutputs = decl.outputs.slice(0, effectiveNargout);
  // Pre-seed each output's type with the i-th param's type (or
  // Unknown when fewer params than outputs). This is the placeholder's
  // `outputTypes` value that recursive self-calls will read while the
  // body is mid-lowering. For the common recursion shape — output
  // kind matches input kind, as in `factorial` / `fib` — the seed is
  // already correct and the body type-checks first try. For other
  // shapes the post-lowering compare below catches the mismatch and
  // re-lowers once with the actual types.
  const seedOutputs: Type[] = effectiveOutputs.map(
    (_, i) => argTypes[i] ?? UNKNOWN
  );
  // The IR function's params reflect only the args actually passed
  // — each spec compiles to its own C function whose signature
  // matches the call site's arity. Trailing declared params are
  // dropped here; the body's references to them must be gated by
  // the `nargin < N` fold.
  const effectiveParams = decl.params.slice(0, argTypes.length);
  const placeholder: IRFunc = {
    name: decl.name,
    cName: key,
    params: effectiveParams.slice(),
    cParams: effectiveParams.map(cIdentForUserName),
    paramTypes: argTypes,
    outputs: effectiveOutputs.slice(),
    cOutputs: effectiveOutputs.map(cIdentForUserName),
    outputTypes: seedOutputs,
    body: [],
    span: decl.span,
  };
  this.specializations.set(key, placeholder);

  // Save outer state. The try/finally guarantees state is restored
  // even if body lowering throws — otherwise a TypeError /
  // UnsupportedConstruct from the body would leak this function's env
  // / tempCounter / currentFile / callFrameStack to the caller.
  const savedEnv = this.env;
  const savedTempCounter = this.tempCounter;
  const savedCurrentFile = this.currentFile;
  this.currentFile = file;
  this.callFrameStack.push({
    nargin: argTypes.length,
    nargout: effectiveNargout,
  });

  // Lower the body once with the current placeholder.outputTypes.
  // Returns the freshly-built body + the actual output types read
  // from env. Resets env/tempCounter so a re-lower starts from the
  // same clean state as the first pass.
  const lowerBodyOnce = (): { body: IRStmt[]; outputTypes: Type[] } => {
    this.env = new Map();
    this.tempCounter = 0;
    // Bind params for the args that were actually passed. Trailing
    // params remain unbound — references to them inside the body
    // surface as "use of undefined variable" unless the body
    // initializes them (typically inside an `if nargin < N` arm
    // that the cond fold prunes for this specialization). The C name
    // goes through `cIdentForUserName` so a user-source
    // `function r = f(struct)` doesn't reference the C keyword
    // `struct` for reads inside the body.
    for (let i = 0; i < argTypes.length; i++) {
      const pName = decl.params[i];
      this.env.set(pName, {
        cName: cIdentForUserName(pName),
        ty: argTypes[i],
      });
    }
    // Class constructors pre-seed their output (the receiver) with
    // the default-valued class instance via an injected first stmt,
    // so the body can read `obj.x` / write `obj.x = ...` against an
    // initialized slot from the very first source statement.
    let initStmts: IRStmt[] = [];
    if (preSeedOutput !== undefined) {
      this.requireValueType(
        preSeedOutput.initExpr,
        `constructor init for '${preSeedOutput.name}'`
      );
      const initStmt = this.recordAssignment(
        preSeedOutput.name,
        preSeedOutput.initExpr,
        decl.span
      );
      initStmts = [initStmt];
    }

    const body = [...initStmts, ...this.lowerStmts(decl.body)];

    // Output types come from the final env value of each effective
    // output name. Trailing outputs the caller dropped via nargout
    // truncation aren't checked — they may legitimately be left
    // unassigned by a `if nargout >= N` body branch.
    const outputTypes: Type[] = effectiveOutputs.map(o => {
      const e = this.env.get(o);
      if (!e) {
        throw new TypeError(
          `function '${decl.name}': output '${o}' was never assigned`,
          errSpan
        );
      }
      return e.ty;
    });
    return { body, outputTypes };
  };

  try {
    let { body, outputTypes } = lowerBodyOnce();

    // If a recursive self-call consumed the placeholder's seeded
    // outputTypes during the first pass AND the actual outputs the
    // body produced differ from the seed, the recursive Call IR
    // nodes built in pass 1 carry the (now-stale) seed type. Update
    // the placeholder with the actual outputs and re-lower the body
    // so the recursive Call IR picks up the refined type. One retry
    // suffices for the common recursion shapes (`factorial`, `fib`),
    // where the output kind matches the input kind. For other
    // shapes — e.g. tensor-in / struct-out — the second pass's
    // recursive Call IR might still not match its body's final
    // output type. Rather than silently ship a third-pass-stale
    // result, raise so the user gets a clear translation-time error
    // instead of a quietly-wrong specialization.
    let seed = seedOutputs;
    if (
      this.recursiveSpecsConsumed.has(key) &&
      !sameOutputTypes(seed, outputTypes)
    ) {
      placeholder.outputTypes = outputTypes;
      seed = outputTypes;
      const second = lowerBodyOnce();
      body = second.body;
      outputTypes = second.outputTypes;
      if (!sameOutputTypes(seed, outputTypes)) {
        throw new TypeError(
          `function '${decl.name}': recursive output type did not stabilize ` +
            `after two lowering passes. First pass produced ` +
            `[${seedOutputs.map(canonicalizeType).join(", ")}], second pass ` +
            `produced [${outputTypes.map(canonicalizeType).join(", ")}]. ` +
            `This usually means the output type genuinely depends on a ` +
            `runtime value the lowerer can't resolve, or that the function ` +
            `is mutually recursive in a shape mtoc2 can't yet handle. ` +
            `Try annotating the output via an early assignment, or avoid ` +
            `recursion for this call.`,
          errSpan
        );
      }
    }

    // Reject any function-output whose final C storage can't be
    // expressed as a single C variable. Two distinct shapes trigger
    // this:
    //
    //  (a) `recordAssignment` split the output's binding mid-body
    //      because two assignments had storage-incompatible types
    //      (`s = 0; ... s = a_tensor;` at top level). The split moves
    //      the post-split writes to `_mtoc2_<o>__v<N>` but the
    //      function's `cOutputs[i]` is locked at the canonical name,
    //      so the return path reads the un-split slot — which was
    //      declared with the post-split type but written with the
    //      pre-split (incompatible) type, yielding a C compile error.
    //
    //  (b) The output's final lattice type unifies to Unknown — most
    //      commonly when an If or For-merge unifies storage-distinct
    //      types (real-tensor vs complex-tensor, scalar vs tensor in
    //      a branch the For-merge folds with envBefore, ...). The C
    //      ABI has no slot for Unknown, so the emitted signature
    //      would declare `double *_mtoc2_o<i>` and the body would
    //      write a tensor through it.
    //
    // Both shapes are correctness bugs in the C emit. Declining at the
    // specialization boundary as `UnsupportedConstruct` routes the
    // caller back to its interpreter; the JS-emit JIT also declines
    // (its dynamic-typed output happens to work in some shapes, but
    // the split case has return-value-undefined in the non-base path
    // of a recursive function, so declining is correct there too).
    for (let i = 0; i < effectiveOutputs.length; i++) {
      const o = effectiveOutputs[i];
      const entry = this.env.get(o);
      // Missing entries (output declared but never assigned) are
      // already rejected inside `lowerBodyOnce` with a TypeError, so
      // `entry` is defined here. Defensive guard:
      if (entry === undefined) continue;
      const canonicalCName = cIdentForUserName(o);
      if (entry.cName !== canonicalCName) {
        throw new UnsupportedConstruct(
          `function '${decl.name}': output '${o}' was reassigned with an ` +
            `incompatible C storage type partway through the body — the ` +
            `function output cannot be represented as a single C variable.`,
          errSpan
        );
      }
      if (outputTypes[i].kind === "Unknown") {
        throw new UnsupportedConstruct(
          `function '${decl.name}': output '${o}' resolves to an unknown ` +
            `type at function exit — usually because the body unifies two ` +
            `storage-distinct values for it (e.g. a real and a complex ` +
            `tensor, or a scalar and a tensor). Cannot represent this in ` +
            `the C/JS ABI as a single output slot.`,
          errSpan
        );
      }
    }

    const out: IRFunc = {
      ...placeholder,
      body,
      outputTypes,
    };
    this.specializations.set(key, out);
    return out;
  } catch (err) {
    // Body lowering threw — drop the placeholder so a future call
    // with the same key (e.g. after the user fixes the error and
    // re-translates against the same Lowerer instance) re-attempts
    // specialization instead of returning the empty placeholder.
    this.specializations.delete(key);
    throw err;
  } finally {
    this.env = savedEnv;
    this.tempCounter = savedTempCounter;
    this.currentFile = savedCurrentFile;
    this.callFrameStack.pop();
    // Clear the recursion-consumed marker for this key so a later
    // unrelated spec on the same Lowerer doesn't spuriously re-lower.
    this.recursiveSpecsConsumed.delete(key);
  }
}

/** True iff two output-type lists are canonically equal. Used to
 *  decide whether a recursive specialization's body needs re-lowering
 *  after the heuristic seed produced different output types than the
 *  actual body finalizes to. */
function sameOutputTypes(
  a: ReadonlyArray<Type>,
  b: ReadonlyArray<Type>
): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (canonicalizeType(a[i]) !== canonicalizeType(b[i])) return false;
  }
  return true;
}
