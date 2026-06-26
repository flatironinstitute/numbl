/**
 * Diagnostic sink for the most recent JIT decline reason.
 *
 * When a JIT executor cannot compile a unit it catches the
 * `UnsupportedConstruct` / `JitTypeError` thrown during lowering/codegen and
 * declines (returns `null`), letting the dispatcher fall through to the
 * interpreter. That message is normally lost. We stash the most recent one
 * here so a failing `%!numbl:assert_jit` can report *why* the unit declined,
 * not merely that it ran in the interpreter.
 *
 * This is a best-effort, process-global "last decline" — when an assert fires,
 * the unit it guards declined immediately beforehand, so the most recent
 * recorded reason is the relevant one. It is a diagnostic aid, not a precise
 * per-unit binding.
 *
 * Leaf module: no imports, so it can be shared by the executors
 * (`executors/jit/*`) and the interpreter (`interpreter/*`) without cycles.
 */

export interface JitDecline {
  /** The thrown error's message (e.g. "unsupported builtin 'rand'"). */
  message: string;
  /** Error class name: "UnsupportedConstruct" | "JitTypeError". */
  kind: string;
  /** Which executor declined: "jit-top-level" | "jit-call" | "jit-loop". */
  where: string;
}

let lastDecline: JitDecline | null = null;

/** Record a swallowed JIT decline reason (called from executor catch blocks). */
export function recordJitDecline(d: JitDecline): void {
  lastDecline = d;
}

/** The most recent recorded JIT decline, or null if none seen. */
export function getLastJitDecline(): JitDecline | null {
  return lastDecline;
}

/** Forget the last decline (e.g. so an unrelated stale reason isn't reported). */
export function clearJitDecline(): void {
  lastDecline = null;
}
