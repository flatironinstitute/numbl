/**
 * Builtin registry — the contract every mtoc2 builtin (and future
 * user `.mtoc2.js` function) implements.
 *
 * Each builtin can supply up to three mirrored hooks plus the
 * always-required `transfer`:
 *
 *   transfer(argTypes, nargout) — required. Given input types and
 *     the number of outputs the call site wants, return one Type
 *     per output. When every input is exact, set `exact` on the
 *     returned type(s) so the lowerer can key specialization on the
 *     value (and fold `if` conditions). Throws plain `Error` /
 *     `TypeError` / `UnsupportedConstruct` on bad input — no source
 *     span; the framework attaches the call-site span via `withSpan`.
 *
 *   emitC({argsC, argTypes, nargout, outArgsC?, useRuntime}) —
 *     optional. C codegen. Returns the full string ready to splice
 *     into the surrounding context:
 *       - nargout=1: a C expression (`(a + b)`,
 *         `mtoc2_tensor_plus_tt(a, b)`)
 *       - nargout≥2: a full call including out-pointer args
 *         (`mtoc2_sort_real_2(a, &v, &i)`). `outArgsC` holds those
 *         pre-built `&v`, `&i` strings in the order the call site
 *         requested.
 *     Activate any runtime snippets this emit-output references by
 *     calling `useRuntime(name)`.
 *
 *   emitJs({argsJs, argTypes, nargout, outTargetsJs?, useRuntime}) —
 *     optional. JS codegen. Returns a JS expression (or call
 *     sequence) ready to splice. Multi-output calls receive
 *     `outTargetsJs` pre-built as `o1`, `o2`, … the secondary
 *     outputs land in these locals via the framework's destructuring
 *     wrapper. Activates JS-side runtime snippets via `useRuntime`.
 *
 *   call({args, argTypes, nargout, ctx}) — optional. Direct JS
 *     function used by the interpreter and as a JIT bailout target.
 *     Returns one runtime value per output slot. The args shape
 *     mirrors the emit hooks' shape so dispatch on `argTypes` /
 *     `nargout` stays parallel across all three.
 *
 *   elementwise? — when true, every defined emit hook (and `call`)
 *     must also produce a correct scalar output when called with
 *     scalar argTypes and scalar argsC/argsJs/args. The framework
 *     uses this for same-shape tensor fusion (and, later, broadcast
 *     / reduction-inner / mask / stencil fusion). Absence = "do not
 *     fuse me; use the regular emit on the full tensor types."
 *
 * Each backend selector requires the matching hook to be defined on
 * every builtin in the call's transitive reach; the framework
 * computes the maximum compatible backend per call site.
 */

import type { Type } from "../lowering/types.js";
import type { RuntimeValue } from "../runtime/value.js";
import type { RuntimeContext } from "../runtime/context.js";

// ── Shared arg-shape ────────────────────────────────────────────────────

/** Common across every hook — the type-level call descriptor. */
interface CommonArgs {
  /** Per-arg type (after lowering / inference). */
  argTypes: Type[];
  /** Number of outputs requested at the call site. */
  nargout: number;
}

export interface EmitCArgs extends CommonArgs {
  /** Per-arg C expression. In fused contexts this is scalar form
   *  (e.g. `<var>.real[i]`); in the regular path it's whatever the
   *  framework's `emitExpr` produced. */
  argsC: string[];
  /** Out-pointer C expressions for multi-output calls (length =
   *  nargout when nargout≥2; otherwise undefined). The framework
   *  pre-builds these as `&v`, `&i`, … so `emit` just splices them
   *  in. */
  outArgsC?: string[];
  /** Activate a runtime C snippet on the current translation unit.
   *  See `runtime.ts` for the registry and `InlineSnippet` for the
   *  user-supplied form. */
  useRuntime(spec: string | InlineSnippet): void;
}

export interface EmitJsArgs extends CommonArgs {
  /** Per-arg JS expression. */
  argsJs: string[];
  /** Out-target JS expressions for multi-output calls. The
   *  framework pre-builds these as `o1`, `o2`, … and destructures
   *  the call's return into them. Length equals `nargout-1` when
   *  `nargout≥2` (the primary output is the expression's value).
   *  Undefined for `nargout≤1`. */
  outTargetsJs?: string[];
  /** Activate a JS-side runtime snippet. Snippets are inlined once
   *  per build at the top of the emitted module, so the call site
   *  invokes them by bare name. Mirrors the C-side `useRuntime` —
   *  same registry, paired `.h`/`.js` source files. */
  useRuntime(name: string): void;
}

export interface CallArgs extends CommonArgs {
  /** Per-arg runtime values, evaluated by the interpreter at the
   *  call site. */
  args: RuntimeValue[];
  /** Host bag — `helpers.write` for stdout, future hooks for plot /
   *  error / time. */
  ctx: RuntimeContext;
}

// ── Builtin ──────────────────────────────────────────────────────────────

export interface Builtin {
  /** Source-level name (registry key). */
  name: string;

  /** Required: output Type per output slot. Pure — safe to invoke
   *  as a probe (the framework calls this for bare-identifier 0-arg
   *  detection). */
  transfer(argTypes: Type[], nargout: number): Type[];

  /** Optional: C codegen. */
  emitC?(args: EmitCArgs): string;

  /** Optional: JS codegen. */
  emitJs?(args: EmitJsArgs): string;

  /** Optional: direct JS function — interpreter / JIT bailout. */
  call?(args: CallArgs): RuntimeValue[];

  /** Elementwise = result is computed pointwise. */
  elementwise?: boolean;
}

/** Inline-defined runtime snippet supplied by a user `.mtoc2.js`
 *  builtin's `emitC` / `emitJs`. Re-exported from `runtime.ts` — kept
 *  here for doc-locality with the `Builtin` contract. */
export interface InlineSnippet {
  name: string;
  /** C body (required for `emitC` consumers; ignored on the JS path). */
  code: string;
  /** Optional JS body (required for `emitJs` consumers). */
  jsCode?: string;
  headers?: ReadonlyArray<string>;
  deps?: ReadonlyArray<string>;
}

// ── Registry ────────────────────────────────────────────────────────────

const REGISTRY = new Map<string, Builtin>();

export function registerBuiltin(b: Builtin): void {
  // Overwrite rather than throw on duplicate: the registration list is
  // static and a real duplicate is obvious at the call site, while a throw
  // here breaks Vite HMR (the registry module instance survives reloads
  // but importers re-run their registration side-effects).
  REGISTRY.set(b.name, b);
}

export function getBuiltin(name: string): Builtin | undefined {
  return REGISTRY.get(name);
}

/** Names of every registered builtin. Drives Monaco syntax highlighting. */
export function allBuiltinNames(): readonly string[] {
  return Array.from(REGISTRY.keys());
}

// ── Backend capability probes ───────────────────────────────────────────

export type Backend = "c" | "js" | "interpreter";

export function builtinSupports(b: Builtin, backend: Backend): boolean {
  switch (backend) {
    case "c":
      return b.emitC !== undefined;
    case "js":
      return b.emitJs !== undefined;
    case "interpreter":
      return b.call !== undefined;
  }
}

// ── Internal accessors (post-backend-selection) ─────────────────────────
//
// After the framework has selected a backend and validated that every
// builtin in the call's transitive reach supports it, codegen / runtime
// dispatch sites use these to fetch the hook without the optional-chain
// dance. If the assertion fails it's a framework bug (we missed a
// backend-capability check upstream), not a user error — surface it as
// such.

export function requireEmitC(b: Builtin): (args: EmitCArgs) => string {
  if (!b.emitC) {
    throw new Error(
      `internal: builtin '${b.name}' has no emitC hook (backend selection should have rejected this earlier)`
    );
  }
  return b.emitC.bind(b);
}

export function requireEmitJs(b: Builtin): (args: EmitJsArgs) => string {
  if (!b.emitJs) {
    throw new Error(
      `internal: builtin '${b.name}' has no emitJs hook (backend selection should have rejected this earlier)`
    );
  }
  return b.emitJs.bind(b);
}

export function requireCall(b: Builtin): (args: CallArgs) => RuntimeValue[] {
  if (!b.call) {
    throw new Error(
      `internal: builtin '${b.name}' has no call hook (backend selection should have rejected this earlier)`
    );
  }
  return b.call.bind(b);
}
