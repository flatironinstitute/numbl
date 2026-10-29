/**
 * JIT runtime helpers — small C snippets inlined into the generated
 * source on demand.
 *
 * Each helper lives in its own .h file under `runtime/` so it can be
 * edited with normal C tooling (clangd, syntax highlighting). The .h
 * sources are inlined into `runtime/snippets.gen.ts` by
 * `scripts/build_runtime_snippets.ts`; this module reads them from
 * there (rather than the filesystem at load time) so the translator
 * bundles in the browser.
 *
 * Each helper is referenced by a stable name (e.g. "jit_disp_double")
 * declared, with its dependencies, in the `@snippet` header of its .h
 * file; the activator pulls dependencies in first.
 */

import {
  JS_SNIPPETS,
  SNIPPET_ALIASES,
  SNIPPET_DEFS,
} from "../builtins/runtime/snippets.gen.js";
import type { Type } from "../lowering/types.js";

/**
 * C runtime-snippet table, registered at bootstrap (Node only) rather than
 * imported statically. C-JIT (`--opt 2`) is Node-only, so its ~300 KB of C
 * source strings must not land in the browser worker bundle. The Node C-JIT
 * bootstrap (`executors/jit/compileC.node.ts`) calls `setCSnippets` with the
 * table from `snippets.c.gen.ts`; the browser never registers it and never
 * reaches C codegen, so it stays out of the browser dep graph.
 */
let cSnippets: Record<string, string> | null = null;

/** Register the C runtime-snippet table. Called once at Node bootstrap. */
export function setCSnippets(snippets: Record<string, string>): void {
  cSnippets = snippets;
}

export interface RuntimeSnippet {
  /** Standard-library headers parsed out of the source file. */
  headers: ReadonlyArray<string>;
  /** Body of the snippet (definitions only — `#include`s removed). */
  code: string;
  /** Paired JS body, when a `<basename>.js` sibling exists in the
   *  runtime directory. Populated by `loadSnippet` from
   *  `JS_SNIPPETS[<basename>.js]`. `emitJs` activates and inlines
   *  this; if absent, the snippet is C-only (the builtin that owns
   *  it has no JS path yet). */
  jsCode?: string;
  /** Other helpers (by name) this snippet depends on. The activator
   *  pulls these in first so their definitions come before this
   *  snippet's. Cycles are not supported — keep the graph acyclic. */
  deps: ReadonlyArray<string>;
}

/**
 * Parse a raw snippet source string into its `headers` and `code` parts.
 *
 * The strict `#include` pattern accepted is:
 *   optional-whitespace `#` optional-whitespace `include`
 *   whitespace `<header>` or `"header"` optional-trailing-whitespace
 *
 * Any line whose trimmed form starts with `#include` but does NOT match
 * that pattern is rejected.
 */
export function parseSnippetSource(raw: string): {
  headers: string[];
  code: string;
} {
  const headers: string[] = [];
  const bodyLines: string[] = [];
  for (const line of raw.split(/\r?\n/)) {
    const m = line.match(/^\s*#\s*include\s+(<[^>]+>|"[^"]+")\s*$/);
    if (m) {
      headers.push(m[1]);
    } else if (/^\s*#\s*include\b/.test(line)) {
      throw new Error(
        `runtime snippet: unexpected #include form: ${JSON.stringify(line)}; ` +
          `expected '#include <header>' or '#include "header"' with no trailing content`
      );
    } else {
      bodyLines.push(line);
    }
  }
  while (bodyLines.length && bodyLines[0].trim() === "") bodyLines.shift();
  while (bodyLines.length && bodyLines[bodyLines.length - 1].trim() === "")
    bodyLines.pop();
  return { headers, code: bodyLines.join("\n") + "\n" };
}

/** Parse a `.h` body out of the registered C-snippet table. Only the C
 *  codegen path (`renderRuntimeBodies` / `collectRuntimeHeaders`) reaches
 *  this, and only under Node where `setCSnippets` has run. */
function parseCSnippet(filename: string): { headers: string[]; code: string } {
  if (cSnippets === null) {
    throw new Error(
      "C runtime snippets not registered; call setCSnippets at bootstrap " +
        "(C-JIT is Node-only — see executors/jit/compileC.node.ts)"
    );
  }
  const raw = cSnippets[filename];
  if (raw === undefined) {
    throw new Error(
      `runtime snippet '${filename}' not found in snippets.c.gen.ts; ` +
        `re-run 'npm run build:snippets' after adding the .h file`
    );
  }
  return parseSnippetSource(raw);
}

function loadSnippet(
  filename: string,
  deps: ReadonlyArray<string>
): RuntimeSnippet {
  // The JS sibling (`code`'s paired `.js`) is bound eagerly so emitJs can
  // render it directly. The C `headers`/`code` are resolved lazily via
  // getters: C-JIT is Node-only, so deferring access keeps the ~300 KB C
  // table out of the browser bundle (the getters never fire there) and lets
  // this registry initialize before the Node bootstrap registers the table.
  const jsName = filename.replace(/\.h$/, ".js");
  const jsCode = JS_SNIPPETS[jsName];
  let parsed: { headers: string[]; code: string } | null = null;
  const resolveC = (): { headers: string[]; code: string } => {
    if (parsed === null) parsed = parseCSnippet(filename);
    return parsed;
  };
  const snippet: RuntimeSnippet = {
    get headers() {
      return resolveC().headers;
    },
    get code() {
      return resolveC().code;
    },
    deps,
  };
  if (jsCode !== undefined) snippet.jsCode = jsCode;
  return snippet;
}

// ── Registry ──────────────────────────────────────────────────────────

/** Snippet name → snippet definition, built from the table that
 *  `scripts/build_runtime_snippets.ts` generates from the `@snippet`
 *  header of each `.h` (see that script for the header format). The name
 *  is the C identifier the emitted code calls (e.g. `jit_disp_double`). */
const REGISTRY: ReadonlyMap<string, RuntimeSnippet> = new Map(
  SNIPPET_DEFS.map(d => [d.name, loadSnippet(d.file, d.deps)])
);

/** Alternative names (header `aliases=`) that activate a snippet. */
const ALIASES: ReadonlyMap<string, string> = new Map(
  Object.entries(SNIPPET_ALIASES)
);

/** Look up a snippet from the global registry. Throws if unregistered. */
function lookupGlobalSnippet(name: string): RuntimeSnippet {
  const s = REGISTRY.get(name);
  if (!s) {
    throw new Error(`runtime snippet '${name}' not registered`);
  }
  return s;
}

/** State-aware lookup. Consults the state's `extraSnippets` (inline
 *  snippets registered via `useRuntime({name, code, ...})`) first,
 *  then falls back to the global registry. */
function lookupSnippet(state: RuntimeState, name: string): RuntimeSnippet {
  const extra = state.extraSnippets.get(name);
  if (extra !== undefined) return extra;
  return lookupGlobalSnippet(name);
}

// ── Activation state + topological emit ────────────────────────────────

/** Inline snippet definition supplied by an emitter via
 *  `useRuntime({...})`. Once registered on a `RuntimeState`, subsequent
 *  activations of the same `name` are deduplicated.
 *
 *  `code` is the C body (required for `emitC` consumers); `jsCode`
 *  is the optional JS body (required for `emitJs` consumers but
 *  ignored on the C path). A builtin that supports both backends
 *  supplies both; one that only supports C may omit `jsCode`. */
export interface InlineSnippet {
  name: string;
  code: string;
  jsCode?: string;
  headers?: ReadonlyArray<string>;
  deps?: ReadonlyArray<string>;
}

export interface RuntimeState {
  /** Insertion-ordered set of activated snippet names. */
  active: Set<string>;
  /** Counter for `MultiAssignCall` discard-temp suffixes so adjacent
   *  multi-output call sites don't collide on `_jit_discard_<N>_<i>`
   *  names. Incremented each time a multi-output call is emitted. */
  multiAssignCallCounter: number;
  /** Inline snippets registered via `useRuntime({...})`. Looked up
   *  by name alongside the global registry. */
  extraSnippets: Map<string, RuntimeSnippet>;
  /** JS-side helper: cName list of the currently-emitting user
   *  function's outputs. Set by `emitFunction` before walking the
   *  body and cleared after so `ReturnFromFunction` can emit a
   *  return matching the function's shape (bare value / array /
   *  nothing). Undefined at top level. */
  currentFnOutputs?: ReadonlyArray<string>;
  /** JS-side: the current function's output types by cName. */
  currentFnOutputTypes?: ReadonlyMap<string, Type>;
  /** Source files named by emitted statement locations, in index order
   *  (see `locCode`). */
  locFiles: string[];
  /** JS only: `Var` reads of a local that may not be assigned yet, which
   *  the emitter guards (see `maybeUnassignedReads`). */
  maybeUnbound?: Set<object>;
  /** The enclosing loops, innermost last: a label when the loop is
   *  strip-mined (a `break` must leave both of its loops), else null. See
   *  `cancelPoll.ts`. */
  loopLabels?: (string | null)[];
  /** Counter for strip-mined loop labels. */
  stripLabelCount?: number;
  /** JS only: per user spec, the parameters its callers pass uncopied
   *  (`jsCallBorrows`). */
  callBorrows?: Map<string, boolean[]>;
  /** C only: the parameters of the function being emitted that it
   *  borrows from the host and must not free (`cEntryBorrows`). */
  borrowedParams?: ReadonlySet<string>;
}

export function newRuntimeState(): RuntimeState {
  return {
    active: new Set(),
    multiAssignCallCounter: 0,
    extraSnippets: new Map(),
    locFiles: [],
  };
}

/** Statement locations: emitted code records the source statement it is
 *  executing (`jit__loc` in C, `$loc` in JS) so a raise can be attributed
 *  to a line. The value packs a file index (into `state.locFiles`) and a
 *  character offset as `fileIdx * 2^32 + offset`; `decodeLoc` inverts it. */
export const LOC_FILE_STRIDE = 2 ** 32;

/** The location code for a source span, registering its file. */
export function locCode(
  state: RuntimeState,
  span: { file: string; start: number }
): number {
  let idx = state.locFiles.indexOf(span.file);
  if (idx < 0) {
    idx = state.locFiles.length;
    state.locFiles.push(span.file);
  }
  return idx * LOC_FILE_STRIDE + span.start;
}

/** Decode a location code against the files it was emitted with; null
 *  for "no location" (negative) or an unknown file index. */
export function decodeLoc(
  code: number,
  locFiles: ReadonlyArray<string>
): { file: string; offset: number } | null {
  if (!(code >= 0)) return null;
  const idx = Math.floor(code / LOC_FILE_STRIDE);
  const file = locFiles[idx];
  if (file === undefined) return null;
  return { file, offset: code - idx * LOC_FILE_STRIDE };
}

/** Activate a snippet by its registered name (or an alias of it). Pulls
 *  its `deps` in first so they appear before it in the final emission
 *  order; those include the snippets its paired `.js` imports. An alias
 *  activates the snippet itself, so its body is emitted once. Idempotent. */
export function useRuntimeByName(state: RuntimeState, name: string): void {
  if (!state.extraSnippets.has(name)) name = ALIASES.get(name) ?? name;
  if (state.active.has(name)) return;
  const s = lookupSnippet(state, name);
  for (const d of s.deps) useRuntimeByName(state, d);
  state.active.add(name);
}

/** Register-and-activate an inline snippet.
 *  First call registers the snippet on the state; subsequent calls
 *  (with the same name) are no-ops at the registration step and just
 *  re-activate through the global pathway. */
export function useRuntimeInline(
  state: RuntimeState,
  snippet: InlineSnippet
): void {
  if (!state.extraSnippets.has(snippet.name)) {
    const stored: RuntimeSnippet = {
      headers: snippet.headers ?? [],
      code: snippet.code,
      deps: snippet.deps ?? [],
    };
    if (snippet.jsCode !== undefined) stored.jsCode = snippet.jsCode;
    state.extraSnippets.set(snippet.name, stored);
  }
  useRuntimeByName(state, snippet.name);
}

/** Build the `useRuntime` callback passed into a `Builtin.emit` call.
 *  Accepts either a name (resolved against the global JIT runtime
 *  registry) or an `InlineSnippet` (registered on the current state).
 *  Centralized here so every emit call site activates snippets
 *  uniformly. */
export function makeEmitUseRuntime(
  state: RuntimeState
): (spec: string | InlineSnippet) => void {
  return spec => {
    if (typeof spec === "string") {
      useRuntimeByName(state, spec);
    } else {
      useRuntimeInline(state, spec);
    }
  };
}

/** All headers required by the activated snippets, in insertion order. */
export function collectRuntimeHeaders(state: RuntimeState): string[] {
  const headers = new Set<string>();
  for (const name of state.active) {
    const s = lookupSnippet(state, name);
    for (const h of s.headers) {
      // Headers are emitted verbatim as `#include ${h}`, so each must
      // already carry its delimiters (`<stdio.h>` or `"foo.h"`). A bare
      // `stdio.h` would emit `#include stdio.h` — invalid C. File-loaded
      // snippets get this for free from parseSnippetSource; inline
      // snippets declare `headers` by hand and have silently shipped the
      // unbracketed form, crashing the C compile instead of the include
      // deduping against BASE_HEADERS.
      if (!/^(<[^>]+>|"[^"]+")$/.test(h)) {
        throw new Error(
          `runtime snippet '${name}': header ${JSON.stringify(h)} must be ` +
            `bracketed (<header.h>) or quoted ("header.h")`
        );
      }
      headers.add(h);
    }
  }
  return Array.from(headers);
}

/** Concatenated bodies (definitions) of activated snippets, in
 *  activation (dep-respecting) order. */
export function renderRuntimeBodies(state: RuntimeState): string {
  if (state.active.size === 0) return "";
  const bodies: string[] = [];
  for (const name of state.active) {
    const s = lookupSnippet(state, name);
    bodies.push(s.code);
  }
  return bodies.join("\n");
}

/** JS counterpart of `renderRuntimeBodies` — concatenates the `jsCode`
 *  fields of activated snippets, in dependency order, skipping
 *  snippets that have no JS body. The result is plain JS source
 *  ready to inline at the top of the emitted module so subsequent
 *  call-site code can reference helpers by bare name.
 *
 *  Symmetric with the C side: the same `state.active` set drives
 *  both renderers — a builtin activates `jit_disp_double` once,
 *  the C path emits `disp_double.h`'s body, the JS path emits
 *  `disp_double.js`'s body. */
export function renderJsRuntimeBodies(state: RuntimeState): string {
  if (state.active.size === 0) return "";
  const bodies: string[] = [];
  for (const name of state.active) {
    const s = lookupSnippet(state, name);
    if (s.jsCode !== undefined) bodies.push(s.jsCode);
  }
  return bodies.join("\n");
}

/** Stable always-emitted header set — every translation includes these
 *  so the user-level code can rely on stdio/math without each builtin
 *  re-declaring them. Kept tiny on purpose. */
export const BASE_HEADERS: ReadonlyArray<string> = [
  "<stdio.h>",
  "<stdlib.h>",
  "<math.h>",
  "<complex.h>",
];
