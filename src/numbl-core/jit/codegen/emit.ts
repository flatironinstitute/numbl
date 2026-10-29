/**
 * Single-file C codegen for JIT IR. Emits a shared-library module:
 *   1. Standard headers + activated runtime snippets (in dep order).
 *   2. Forward declarations for user-function specializations.
 *   3. Function specialization bodies; the entry spec gets extern
 *      linkage so the host can `dlsym` it.
 *
 * Scalars compile to bare `double`. Multi-element tensors compile to
 * `jit_tensor_t` — no refcount, no COW. The codegen invariant
 * (matching mtoc's): every tensor RHS is freshly owned, every
 * ownership-transferring use of a Var read wraps in
 * `jit_tensor_copy(name)`. Pre-declaration of every owned local
 * via `jit_tensor_empty()` at function top makes the first-vs-
 * later distinction uniform (the assign helper releases the prior
 * buffer, which is NULL on first call). Scope-exit emits a
 * `jit_tensor_free` for every owned local.
 */

import type { IRExpr, IRFunc, IRStmt, IRProgram } from "../lowering/ir.js";
import { requireOwnedHelpers } from "./cHelpers.js";
import {
  cellTypedefName,
  classTypedefName,
  handleTypedefName,
  structTypedefName,
  type CellType,
  type ClassType,
  type HandleType,
  type StructType,
  type Type,
} from "../lowering/types.js";
import { forEachSubExpr, forEachTopLevelExpr } from "../lowering/walk.js";
import { callsSpec, isRecursive } from "../lowering/callGraph.js";
import { cEntryBorrows } from "../lowering/paramUse.js";
import {
  BASE_HEADERS,
  collectRuntimeHeaders,
  newRuntimeState,
  renderRuntimeBodies,
  useRuntimeByName,
  type RuntimeState,
} from "./runtime.js";
import {
  emitNamedTypedef,
  specForClass,
  specForHandle,
  specForStruct,
} from "./emitNamedTypedef.js";
import { emitCellTypedef } from "./emitCellTypedef.js";
import { emitFunction, fnParamList, fnRetType } from "./emitStmt.js";

export interface EmitOptions {
  /** cName of the entry spec to expose at module scope (no `static`
   *  qualifier so `dlsym` / koffi can resolve it after `dlopen`).
   *  `prog.functions` must contain it and every spec it transitively
   *  calls. */
  exposeSpec: string;
}

export interface EmitResult {
  source: string;
  /** Exported host-entry symbol: a wrapper around `exposeSpec` that arms
   *  the raise guard (see `emitHostEntry`). */
  hostEntry: string;
  /** Names of every runtime snippet activated during emit. */
  activatedSnippets: ReadonlyArray<string>;
  /** Files named by statement locations, by index (see `locCode`). */
  locFiles: string[];
  /** Per entry parameter: the host passes it as a pointer into its own
   *  buffer, which the entry does not free (`cEntryBorrows`). */
  borrows: boolean[];
}

export function emitProgram(prog: IRProgram, opts: EmitOptions): EmitResult {
  const state = newRuntimeState();
  const userParts: string[] = [];

  if (!prog.functions.has(opts.exposeSpec)) {
    throw new Error(
      `emitProgram: 'exposeSpec' references unknown spec '${opts.exposeSpec}'`
    );
  }

  // The raise channel and tracked allocator come first, so the allocator
  // macros cover every later snippet and the user code (see `raise.h`).
  useRuntimeByName(state, "jit_raise");

  // Struct / class / handle typedefs — one per distinct shape.
  // Emitted ahead of forward decls so user code can refer to them,
  // and in dependency-topological order so any typedef that references
  // another (a struct field of struct type, a handle capturing a
  // tensor's enclosing struct, a handle capturing another handle, ...)
  // is emitted after its dependency. Each typedef ships with its four
  // owned-kind helpers (and a `_disp` helper for structs).
  const namedTypedefs = collectNamedTypedefs(prog);
  for (const t of namedTypedefs) {
    if (t.kind === "Cell") {
      userParts.push(emitCellTypedef(t, state));
      continue;
    }
    const spec =
      t.kind === "Struct"
        ? specForStruct(t)
        : t.kind === "Class"
          ? specForClass(t)
          : specForHandle(t);
    userParts.push(emitNamedTypedef(spec, state));
  }
  if (namedTypedefs.length > 0) userParts.push("");

  // Forward declarations. Every spec is `static`; the host calls the
  // exported wrapper emitted below. The exposed spec is `noinline` so the
  // wrapper's `setjmp` does not pessimize its body.
  const storage = (cName: string): string =>
    cName === opts.exposeSpec ? "static __attribute__((noinline))" : "static";
  for (const fn of prog.functions.values()) {
    userParts.push(
      `${storage(fn.cName)} ${fnRetType(fn)} ${fn.cName}(${fnParamList(fn)});`
    );
  }
  if (prog.functions.size > 0) userParts.push("");

  // Function bodies.
  const entry = prog.functions.get(opts.exposeSpec)!;
  const borrows = cEntryBorrows(entry, isRecursive(prog, entry));
  for (const fn of prog.functions.values()) {
    const borrowed = new Set(
      fn === entry ? fn.cParams.filter((_, i) => borrows[i]) : []
    );
    userParts.push(
      emitFunction(fn, state, storage(fn.cName), callsSpec(prog, fn), borrowed)
    );
    userParts.push("");
  }
  const hostEntry = `${entry.cName}__host`;
  userParts.push(emitHostEntry(entry, hostEntry));
  userParts.push("");

  // Headers: BASE_HEADERS ∪ activated-snippet headers, deduped.
  const headers = new Set<string>(BASE_HEADERS);
  for (const h of collectRuntimeHeaders(state)) headers.add(h);

  const out: string[] = [];
  // Feature-test macro: must precede every <...> include so glibc
  // exposes POSIX.1b extensions (clock_gettime / CLOCK_MONOTONIC,
  // used by tictoc.h). Hosts that compile JIT-emitted C under a
  // strict dialect (`-std=c11`, not `-std=gnu11`) need this to see
  // the POSIX prototypes inside <time.h>.
  out.push("#define _POSIX_C_SOURCE 199309L");
  out.push("");
  for (const h of headers) out.push(`#include ${h}`);
  out.push("");
  // No FMA contraction, so results match the JS backend and the
  // interpreter bit for bit whatever compiler builds this (clang
  // contracts by default; the host also passes -ffp-contract=off). gcc
  // ignores the standard pragma (and warns under -Wall), hence the guard.
  out.push("#if defined(__clang__)");
  out.push("#pragma STDC FP_CONTRACT OFF");
  out.push("#endif");
  out.push("");

  if (state.active.size > 0) {
    out.push(renderRuntimeBodies(state));
  }

  // Release `<complex.h>`'s lower-case macros so user code can shadow
  // ordinary identifiers like `I`, `complex`, `imaginary`. The runtime
  // snippets are emitted above this point, so they've already
  // expanded any macros they used; the user code below sees plain
  // identifiers. The JIT's own complex-literal emit uses `_Complex_I`
  // (always defined by `<complex.h>` per C99 §7.3.1) so it doesn't
  // need the `I` macro either.
  out.push("#ifdef I");
  out.push("#undef I");
  out.push("#endif");
  out.push("#ifdef complex");
  out.push("#undef complex");
  out.push("#endif");
  out.push("#ifdef imaginary");
  out.push("#undef imaginary");
  out.push("#endif");
  out.push("");

  out.push(...userParts);

  // Allocator pair exported from the .so so the JIT host (koffi) can
  // allocate tensor-param buffers the emitted code frees, and free the
  // buffers of returned tensors, through the same libc malloc/free.
  out.push("/* JIT-host allocator wrappers. */");
  out.push(
    "void *jit_malloc(long n) { return malloc(n > 0 ? (size_t)n : 1); }"
  );
  out.push("void jit_free(void *p) { free(p); }");
  out.push("");
  return {
    source: out.join("\n"),
    hostEntry,
    activatedSnippets: Array.from(state.active),
    locFiles: state.locFiles,
    borrows,
  };
}

/** The exported wrapper the host calls: clears the raise status, arms the
 *  raise guard, and calls the entry spec. A `jit_raise` anywhere below
 *  `longjmp`s back here; the guard frees every block the call allocated
 *  (`jit__free_all`) and returns a zeroed value, which the host discards
 *  after reading the status. Kept separate from the spec so the spec's own
 *  recursive calls do not re-arm the guard, and so `setjmp` (which makes
 *  the compiler spill values live across it) stays out of the hot body. */
function emitHostEntry(fn: IRFunc, name: string): string {
  const ret = fnRetType(fn);
  const args = [...fn.cParams];
  if (fn.outputs.length >= 2) {
    for (let i = 0; i < fn.outputs.length; i++) args.push(`_jit_o${i}`);
  }
  const call = `${fn.cName}(${args.join(", ")})`;
  const lines: string[] = [];
  lines.push(`${ret} ${name}(${fnParamList(fn)}) {`);
  lines.push(`  jit__raise_reset();`);
  lines.push(`  if (setjmp(jit__raise_buf)) {`);
  lines.push(`    jit__free_all();`);
  if (ret === "void") {
    lines.push(`    return;`);
  } else {
    lines.push(`    ${ret} _jit_z;`);
    lines.push(`    memset(&_jit_z, 0, sizeof _jit_z);`);
    lines.push(`    return _jit_z;`);
  }
  lines.push(`  }`);
  lines.push(ret === "void" ? `  ${call};` : `  return ${call};`);
  lines.push(`}`);
  return lines.join("\n");
}

type NamedType = StructType | ClassType | HandleType | CellType;

/** Walk the program and collect every distinct named (struct / class /
 *  handle) typedef shape, returning them in dependency-topological
 *  order so any typedef that references another is emitted after its
 *  dependency. Recurses into field, property, and capture types so a
 *  transitively-used inner shape gets included even if the outer shape
 *  is the only thing the IR mentions directly. Also walks `HandleLit`
 *  capture value types — those drive the per-capture C field types
 *  even when the outer expression's `ty` is enough by itself. */
function collectNamedTypedefs(prog: IRProgram): NamedType[] {
  const seen = new Map<string, NamedType>();

  const innerTys = (t: NamedType): Type[] => {
    if (t.kind === "Struct") return t.fields.map(f => f.ty);
    if (t.kind === "Class") return t.properties.map(p => p.ty);
    if (t.kind === "Handle") return t.captures.map(c => c.ty);
    // Cell: tuple → per-slot types; uniform → single elem type.
    if (t.mode === "tuple") return t.elements ?? [];
    return t.elem !== undefined ? [t.elem] : [];
  };

  const considerNamed = (t: Type | undefined): void => {
    if (t === undefined) return;
    let key: string;
    let named: NamedType;
    if (t.kind === "Struct") {
      key = structTypedefName(t);
      named = t;
    } else if (t.kind === "Class") {
      key = classTypedefName(t);
      named = t;
    } else if (t.kind === "Handle") {
      key = handleTypedefName(t);
      named = t;
    } else if (t.kind === "Cell") {
      key = cellTypedefName(t);
      named = t;
    } else {
      return;
    }
    if (seen.has(key)) return;
    seen.set(key, named);
    for (const ity of innerTys(named)) considerNamed(ity);
  };

  const visitExpr = (e: IRExpr): void => {
    forEachSubExpr(e, sub => {
      considerNamed(sub.ty);
      if (sub.kind === "HandleLit") {
        for (const c of sub.captures) considerNamed(c.value.ty);
      }
    });
  };
  const visitStmts = (stmts: ReadonlyArray<IRStmt>): void => {
    for (const s of stmts) {
      forEachTopLevelExpr(s, visitExpr);
      if (s.kind === "Assign") considerNamed(s.ty);
      if (s.kind === "MemberStore") {
        considerNamed(s.base.ty);
        considerNamed(s.leafTy);
      }
      switch (s.kind) {
        case "If":
          visitStmts(s.thenBody);
          visitStmts(s.elseBody);
          break;
        case "While":
        case "For":
          visitStmts(s.body);
          break;
      }
    }
  };
  for (const fn of prog.functions.values()) {
    for (const ty of fn.paramTypes) considerNamed(ty);
    for (const ty of fn.outputTypes) considerNamed(ty);
    visitStmts(fn.body);
  }

  return topoSortNamedTypedefs(Array.from(seen.values()), innerTys);
}

function namedTypedefKey(t: NamedType): string {
  if (t.kind === "Struct") return structTypedefName(t);
  if (t.kind === "Class") return classTypedefName(t);
  if (t.kind === "Handle") return handleTypedefName(t);
  return cellTypedefName(t);
}

function topoSortNamedTypedefs(
  ts: NamedType[],
  innerTys: (t: NamedType) => Type[]
): NamedType[] {
  const byName = new Map<string, NamedType>();
  for (const t of ts) byName.set(namedTypedefKey(t), t);
  const visited = new Set<string>();
  const out: NamedType[] = [];
  const visit = (t: NamedType): void => {
    const name = namedTypedefKey(t);
    if (visited.has(name)) return;
    visited.add(name);
    for (const ity of innerTys(t)) {
      if (
        ity.kind === "Struct" ||
        ity.kind === "Class" ||
        ity.kind === "Handle" ||
        ity.kind === "Cell"
      ) {
        const dep = byName.get(namedTypedefKey(ity));
        if (dep) visit(dep);
      }
    }
    out.push(t);
  };
  for (const t of ts) visit(t);
  return out;
}

export function activateOwnedRuntime(t: Type, state: RuntimeState): void {
  const h = requireOwnedHelpers(t);
  if (h.isRuntime) {
    useRuntimeByName(state, h.empty);
    useRuntimeByName(state, h.assign);
    useRuntimeByName(state, h.copy);
    useRuntimeByName(state, h.free);
  }
}

/** Activate the runtime snippets needed for an owned type AND return
 *  its OwnedHelpers descriptor. Shorthand for the
 *  `activateOwnedRuntime(t, state); requireOwnedHelpers(t)` pair that
 *  appears at every owned-codegen site. */
export function activatedOwnedHelpers(t: Type, state: RuntimeState) {
  activateOwnedRuntime(t, state);
  return requireOwnedHelpers(t);
}
