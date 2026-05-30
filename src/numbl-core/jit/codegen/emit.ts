/**
 * Single-file C codegen for mtoc2 IR. Emits:
 *   1. Standard headers + activated runtime snippets (in dep order).
 *   2. Forward declarations for user-function specializations.
 *   3. Function specialization bodies.
 *   4. `main()` containing top-level statements.
 *
 * Scalars compile to bare `double`. Multi-element tensors compile to
 * `mtoc2_tensor_t` — no refcount, no COW. The codegen invariant
 * (matching mtoc's): every tensor RHS is freshly owned, every
 * ownership-transferring use of a Var read wraps in
 * `mtoc2_tensor_copy(name)`. Pre-declaration of every owned local
 * via `mtoc2_tensor_empty()` at function top makes the first-vs-
 * later distinction uniform (the assign helper releases the prior
 * buffer, which is NULL on first call). Scope-exit emits a
 * `mtoc2_tensor_free` for every owned local.
 *
 * `includeRuntime` (default true): when false, the activated runtime
 * helpers are omitted from the output (a placeholder comment is shown
 * instead) so the IDE's "runtime helpers" toggle can show user-level
 * code in isolation. The emitted C is not compilable in that mode —
 * it's a viewing aid.
 */

import type { IRExpr, IRStmt, IRProgram } from "../lowering/ir.js";
import { cTypeFor, requireOwnedHelpers } from "./cHelpers.js";
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
import {
  BASE_HEADERS,
  collectRuntimeHeaders,
  newRuntimeState,
  renderRuntimeBodies,
  useRuntimeByName,
  type RuntimeState,
} from "./runtime.js";
import { computeFutureTouches } from "./liveness.js";
import {
  emitNamedTypedef,
  specForClass,
  specForHandle,
  specForStruct,
} from "./emitNamedTypedef.js";
import { emitCellTypedef } from "./emitCellTypedef.js";
import {
  collectLocals,
  defaultInitFor,
  emitBody,
  emitFunction,
  fnParamList,
  fnRetType,
  type EmitReturnTail,
} from "./emitStmt.js";

export interface EmitOptions {
  /** Include the activated runtime helper bodies in the output.
   *  Default true. When false, headers + a placeholder stub replace
   *  them so the user sees only their generated code. */
  includeRuntime?: boolean;
  /** Max-threads OpenMP setting. See `TranslateOptions.threads`. */
  threads?: number | "auto";
  /** Workspace context. Threaded through `RuntimeState` so emit-time
   *  builtin lookups consult `.mtoc2.js` user builtins via
   *  `getUserBuiltin`. Optional — vitest unit tests that don't drive
   *  through a workspace can omit it. */
  workspace?: import("./runtime.js").WorkspaceLike;
  /** JIT-mode emit. When set, the cName of a single user-function spec
   *  to expose at module scope (no `static` qualifier so `dlsym` /
   *  koffi can resolve it after `dlopen`). Disables the `int main()`
   *  emission — the module becomes a `.so` consumers load and call
   *  into directly. Requires `prog.topLevelStmts` to be empty and
   *  `prog.functions` to contain the named spec; every spec
   *  transitively called by the entry must also be in
   *  `prog.functions` so internal calls resolve. */
  exposeSpec?: string;
}

export function emitProgram(prog: IRProgram, opts: EmitOptions = {}): string {
  const includeRuntime = opts.includeRuntime ?? true;
  const threads = opts.threads;
  const state = newRuntimeState(opts.workspace);
  const userParts: string[] = [];

  if (opts.exposeSpec !== undefined) {
    if (prog.topLevelStmts.length > 0) {
      throw new Error(
        "emitProgram: 'exposeSpec' is incompatible with non-empty 'topLevelStmts'"
      );
    }
    if (!prog.functions.has(opts.exposeSpec)) {
      throw new Error(
        `emitProgram: 'exposeSpec' references unknown spec '${opts.exposeSpec}'`
      );
    }
  }

  // Grow-bail pre-pass: scan every function for an indexed store. Done
  // up front — before any function body is emitted — so the host-entry
  // function's `setjmp` guard is emitted regardless of whether the
  // growing store sits in the entry itself or in a callee emitted later
  // in the loop below. The `setjmp` guard references `mtoc2_grow_bail_buf`
  // and needs `<setjmp.h>`, both provided by the `mtoc2_grow_bail`
  // snippet — so activate it HERE, tied to the guard decision. (Per-path
  // activation in `emitNdScalarOffset` / `emitLinearRangeSetup` covers
  // the scalar/linear-range *grow* helpers, but slice/colon stores arm
  // the guard via `bodyHasIndexStore` without touching those paths, so
  // relying on lazy activation alone left the buffer/`<setjmp.h>`
  // undeclared and the C compile failed.)
  for (const fn of prog.functions.values()) {
    if (bodyHasIndexStore(fn.body)) {
      state.usedGrowBail = true;
      useRuntimeByName(state, "mtoc2_grow_bail");
      break;
    }
  }

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

  // Forward declarations. The exposed spec (when `exposeSpec` is set)
  // gets `extern` linkage so `dlsym` / koffi can resolve it; everything
  // else stays `static`.
  for (const fn of prog.functions.values()) {
    const sc = fn.cName === opts.exposeSpec ? "" : "static ";
    userParts.push(`${sc}${fnRetType(fn)} ${fn.cName}(${fnParamList(fn)});`);
  }
  if (prog.functions.size > 0) userParts.push("");

  // Function bodies.
  for (const fn of prog.functions.values()) {
    const sc = fn.cName === opts.exposeSpec ? "" : "static";
    userParts.push(emitFunction(fn, state, sc));
    userParts.push("");
  }

  // JIT mode (`exposeSpec` set): skip the `int main()` emission. The
  // module is a `.so` the host loads + calls into directly.
  if (opts.exposeSpec !== undefined) {
    // Headers: BASE_HEADERS ∪ activated-snippet headers, deduped.
    const headers = new Set<string>(BASE_HEADERS);
    for (const h of collectRuntimeHeaders(state)) headers.add(h);

    const out: string[] = [];
    // Feature-test macro: must precede every <...> include so glibc
    // exposes POSIX.1b extensions (clock_gettime / CLOCK_MONOTONIC,
    // used by tictoc.h). Hosts that compile mtoc2-emitted C under a
    // strict dialect (`-std=c11`, not `-std=gnu11`) need this to see
    // the POSIX prototypes inside <time.h>.
    out.push("#define _POSIX_C_SOURCE 199309L");
    out.push("");
    for (const h of headers) out.push(`#include ${h}`);
    out.push("");

    if (state.active.size > 0) {
      out.push(
        includeRuntime ? renderRuntimeBodies(state) : runtimePlaceholder(state)
      );
    }

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

    // Allocator helpers exported as part of the .so so a JIT host
    // (koffi) can resolve them by name without separately loading
    // libc. `mtoc2_alloc` is used by the runtime helpers internally;
    // `mtoc2_jit_free` is the host-side counterpart for releasing
    // tensor buffers returned across the FFI. Both go through the
    // same malloc/free in libc that mtoc2's emitted code already
    // depends on transitively, so there's no allocator-mismatch
    // risk.
    out.push("/* JIT-host allocator wrapper. The .so already pulls in");
    out.push(" * libc's free transitively via <stdlib.h>; this wrapper");
    out.push(" * exposes it through the .so's own symbol table so the");
    out.push(" * host's koffi binding doesn't need to dlopen libc");
    out.push(" * separately. */");
    out.push("void mtoc2_jit_free(void *p) { free(p); }");
    out.push("");
    return out.join("\n");
  }

  // Main.
  userParts.push("int main(void) {");
  // `--threads N` (numeric, >= 2) → pin the OpenMP team size at
  // startup; subsequent `#pragma omp parallel for` regions cap at N.
  // `--threads auto` doesn't emit anything here — OpenMP picks from
  // `OMP_NUM_THREADS` / core count via its own defaults. The
  // `omp_set_num_threads` symbol comes from `<omp.h>`, which the
  // header block below pulls in conditionally on the same predicate.
  if (typeof threads === "number" && threads > 1) {
    userParts.push(`  omp_set_num_threads(${threads});`);
  }
  const mainLocals = collectLocals(prog.topLevelStmts);
  const mainOwned = mainLocals.owned;
  for (const o of mainOwned) {
    const h = activatedOwnedHelpers(o.ty, state);
    userParts.push(`  ${cTypeFor(o.ty)} ${o.cName} = ${h.empty}();`);
  }
  for (const o of mainLocals.scalars) {
    userParts.push(`  ${cTypeFor(o.ty)} ${o.cName} = ${defaultInitFor()};`);
  }
  const mainFutureTouches = computeFutureTouches(prog.topLevelStmts);
  const mainOwnedTypes = new Map<string, Type>(
    mainOwned.map(o => [o.cName, o.ty])
  );
  // Tail closure for main: scope-exit frees + `return 0;`. Called both
  // at every `ReturnFromFunction` inside the top-level body (inline
  // cleanup — no goto) and at the fall-through bottom. Activates the
  // owned `_free` runtime deps eagerly so the snippet ships even when
  // every path is an early return.
  for (const o of mainOwned) activateOwnedRuntime(o.ty, state);
  const mainReturnTail: EmitReturnTail = indent => {
    const lines: string[] = [];
    // Early-frees null the buffer, every `_free` helper is NULL-safe,
    // so duplicate frees across overlapping early-return + fall-through
    // paths are redundant but safe. A prior `nullAtScopeExit`
    // optimization tried to skip provably-NULL frees but mis-modeled
    // early returns; we don't ship it.
    for (const o of mainOwned) {
      const h = requireOwnedHelpers(o.ty);
      lines.push(`${indent}${h.free}(&${o.cName});`);
    }
    lines.push(`${indent}return 0;`);
    return lines.join("\n");
  };
  const mainBody = emitBody(
    prog.topLevelStmts,
    "  ",
    state,
    mainFutureTouches,
    mainOwnedTypes,
    mainReturnTail
  );
  if (mainBody.length > 0) userParts.push(mainBody);
  userParts.push(mainReturnTail("  "));
  userParts.push("}");
  userParts.push("");

  // Headers: BASE_HEADERS ∪ activated-snippet headers, deduped.
  // `<omp.h>` only when we actually emit a call into the OpenMP API
  // (`omp_set_num_threads(N)` for numeric `--threads N >= 2`). The
  // `_Pragma("omp ...")` lines in the elementwise macros are
  // preprocessor-handled and need no header.
  const headers = new Set<string>(BASE_HEADERS);
  for (const h of collectRuntimeHeaders(state)) headers.add(h);
  if (typeof threads === "number" && threads > 1) headers.add("<omp.h>");

  const out: string[] = [];
  // Feature-test macro: must precede every <...> include so glibc
  // exposes POSIX.1b extensions (clock_gettime / CLOCK_MONOTONIC,
  // used by tictoc.h). Hosts that compile mtoc2-emitted C under a
  // strict dialect (`-std=c11`, not `-std=gnu11`) need this to see
  // the POSIX prototypes inside <time.h>.
  out.push("#define _POSIX_C_SOURCE 199309L");
  out.push("");
  for (const h of headers) out.push(`#include ${h}`);
  out.push("");

  if (state.active.size > 0) {
    out.push(
      includeRuntime ? renderRuntimeBodies(state) : runtimePlaceholder(state)
    );
  }

  // Release `<complex.h>`'s lower-case macros so user code can shadow
  // ordinary identifiers like `I`, `complex`, `imaginary`. The runtime
  // snippets are emitted above this point, so they've already
  // expanded any macros they used; the user code below sees plain
  // identifiers. mtoc2's own complex-literal emit uses `_Complex_I`
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
  return out.join("\n");
}

function runtimePlaceholder(state: RuntimeState): string {
  if (state.active.size === 0) return "";
  const names = Array.from(state.active).join(", ");
  return `/* runtime helpers omitted (${state.active.size}): ${names} */\n`;
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
/** True if `stmts` (recursing into control-flow bodies, but NOT into
 *  nested function defs — those are separate IRFuncs) contains a store
 *  that can grow-bail at runtime — a scalar `IndexStore` or a slice
 *  `IndexSliceStore`. Drives the grow-bail `setjmp` guard decision; an
 *  unused guard (e.g. a Colon slice that can't grow) is harmless. */
function bodyHasIndexStore(stmts: ReadonlyArray<IRStmt>): boolean {
  for (const s of stmts) {
    if (s.kind === "IndexStore" || s.kind === "IndexSliceStore") return true;
    if (s.kind === "If") {
      if (bodyHasIndexStore(s.thenBody) || bodyHasIndexStore(s.elseBody)) {
        return true;
      }
    } else if (s.kind === "While" || s.kind === "For") {
      if (bodyHasIndexStore(s.body)) return true;
    }
  }
  return false;
}

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
  visitStmts(prog.topLevelStmts);

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
