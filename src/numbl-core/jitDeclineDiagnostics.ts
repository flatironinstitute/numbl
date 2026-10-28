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
 * This is a best-effort "last decline" of the running session
 * (`SessionState.lastJitDecline`): when an assert fires, the unit it guards
 * declined immediately beforehand, so the most recent recorded reason is the
 * relevant one. It is a diagnostic aid, not a precise per-unit binding.
 */

import { currentSession } from "./sessionState.js";

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

/** Record a swallowed JIT decline reason (called from executor catch blocks). */
export function recordJitDecline(d: JitDecline): void {
  currentSession().lastJitDecline = d;
  if (typeof process !== "undefined" && process.env?.NUMBL_JIT_LOG) {
    const at = d.at ? ` [${d.at.file}@${d.at.start}]` : "";
    console.error(`[jit-decline] ${d.where}${at}: ${d.message}`);
  }
}

/** The most recent recorded JIT decline, or null if none seen. */
export function getLastJitDecline(): JitDecline | null {
  return currentSession().lastJitDecline;
}

/** Forget the last decline (e.g. so an unrelated stale reason isn't reported). */
export function clearJitDecline(): void {
  currentSession().lastJitDecline = null;
}
