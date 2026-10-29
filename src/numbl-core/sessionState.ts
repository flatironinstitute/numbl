/**
 * State that outlives a single run: what MATLAB keeps between commands at
 * the prompt. An `ExecSession` owns one and passes it to every
 * `executeCode`; a one-shot `executeCode` (CLI run/eval) gets a fresh one.
 * Each run's `Runtime` holds the session it runs in (`rt.session`), so two
 * sessions in one process do not share any of this.
 */

import type { RuntimeValue } from "./runtime/types.js";
import { Environment } from "./interpreter/types.js";
import { getCurrentRuntime } from "./runtime/refcount.js";

/** A JIT decline reason, kept for `%!numbl:assert_jit` messages (see
 *  jitDeclineDiagnostics.ts). */
export interface JitDecline {
  /** The thrown error's message (e.g. "unsupported builtin 'rand'"). */
  message: string;
  /** Error class name: "UnsupportedConstruct" | "JitTypeError". */
  kind: string;
  /** Which executor declined, e.g. "jit-loop" or "cjit-call". */
  where: string;
  /** Source position of the offending construct, when the thrown error
   *  carried one. Only used for diagnostics. */
  at?: { file: string; start: number };
}

/** State of the seedable PRNG shared by rand/randn/randi/randperm (see
 *  helpers/prng.ts). */
export class RngState {
  /** xoshiro128** state; null means unseeded (Math.random). */
  state: Uint32Array | null = null;
  /** The seed reported by `rng`. */
  seed = 0;
  /** Spare deviate cached by the polar method. */
  spare: number | null = null;
}

export class SessionState {
  /** The base workspace. Every run's base frame uses this environment, so
   *  variables (and the base workspace's `global` links) carry over, also
   *  past an error, as at the MATLAB prompt. */
  readonly baseEnv = new Environment();
  /** Global variables (`rt.$g`). */
  readonly globals: Record<string, RuntimeValue> = {};
  /** Persistent variables: "file:function" → name → value. */
  readonly persistents = new Map<string, Map<string, RuntimeValue>>();
  /** For each file whose functions hold persistents, its source when they
   *  were made: editing the file clears them (`syncPersistents`). */
  private readonly persistentSources = new Map<string, string>();
  /** Warning identifiers turned off with warning('off', id); 'all' covers
   *  identifiers without their own entry. */
  readonly warningsOff = new Set<string>();
  readonly rng = new RngState();
  /** `tic` time (performance.now() ms). */
  ticTime = 0;
  /** `spparms` values; null means the defaults. */
  spparms: Float64Array | null = null;
  /** setappdata storage: handle → name → value. */
  readonly appdata = new Map<number, Map<string, RuntimeValue>>();
  /** The most recent JIT decline (for `%!numbl:assert_jit` messages). */
  lastJitDecline: JitDecline | null = null;

  // Platform capabilities (numeric providers, a compute device, ...) are not
  // kept here: they are the session's host (host.ts, `rt.host`). State owned
  // by a module that imports the interpreter (event listeners, see
  // sessionEventRegistry) is kept by that module, keyed by session, to keep
  // this file a leaf.

  /** `clear global`: remove every global, or the named ones. */
  clearGlobals(names?: string[]): void {
    for (const name of names ?? Object.keys(this.globals)) {
      delete this.globals[name];
    }
  }

  /** `clear functions`: drop the persistents of every function, or of the
   *  functions defined in the files `file` accepts. */
  clearPersistents(file?: (fileName: string) => boolean): void {
    for (const key of [...this.persistents.keys()]) {
      const f = fileOf(key);
      if (file && !file(f)) continue;
      this.persistents.delete(key);
      this.persistentSources.delete(f);
    }
  }

  /** Clear the persistents of functions whose file changed since they were
   *  made, then record the source of the files that now hold persistents.
   *  `sourceOf` gives a file's current source, undefined when the run does
   *  not see the file (its persistents are kept). Called at the start and
   *  end of every run. */
  syncPersistents(sourceOf: (fileName: string) => string | undefined): void {
    if (this.persistents.size === 0) return;
    const changed = new Set<string>();
    for (const key of this.persistents.keys()) {
      const f = fileOf(key);
      const current = sourceOf(f);
      if (current === undefined) continue;
      const recorded = this.persistentSources.get(f);
      if (recorded === undefined) this.persistentSources.set(f, current);
      else if (recorded !== current) changed.add(f);
    }
    if (changed.size > 0) this.clearPersistents(f => changed.has(f));
  }
}

/** The file part of a persistent store key ("file:function"). */
function fileOf(key: string): string {
  return key.slice(0, key.lastIndexOf(":"));
}

/** Session state for code running without a session (helpers called
 *  outside `executeCode`, e.g. from unit tests). */
const detachedSession = new SessionState();

/** The session of the running code: the active runtime's, else a shared
 *  detached one. For helpers without a runtime handle; builtins use
 *  `ctx.rt.session`. */
export function currentSession(): SessionState {
  const rt = getCurrentRuntime() as { session?: SessionState } | null;
  return rt?.session ?? detachedSession;
}
