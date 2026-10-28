/**
 * Call-graph queries over a lowered program: which specs a spec reaches.
 * Any `cName` in the IR that names a spec in the program counts as a call
 * edge.
 */

import type { IRFunc, IRProgram } from "./ir.js";

/** Visit every spec referenced from `node`, depth first. */
function visitCalls(
  prog: IRProgram,
  node: unknown,
  onSpec: (fn: IRFunc) => boolean
): void {
  if (node === null || typeof node !== "object") return;
  if (Array.isArray(node)) {
    for (const el of node) visitCalls(prog, el, onSpec);
    return;
  }
  const rec = node as Record<string, unknown>;
  const cName = rec.cName;
  if (typeof cName === "string") {
    const fn = prog.functions.get(cName);
    if (fn !== undefined && onSpec(fn)) visitCalls(prog, fn.body, onSpec);
  }
  for (const key of Object.keys(rec)) {
    if (key === "ty" || key === "span" || key === "src") continue;
    visitCalls(prog, rec[key], onSpec);
  }
}

/** Specs transitively called from `entry` (excluding it). */
export function reachableCallees(prog: IRProgram, entry: IRFunc): IRFunc[] {
  const seen = new Set<string>([entry.cName]);
  const out: IRFunc[] = [];
  visitCalls(prog, entry.body, fn => {
    if (seen.has(fn.cName)) return false;
    seen.add(fn.cName);
    out.push(fn);
    return true;
  });
  return out;
}

/** Whether `entry` can call itself, directly or through its callees. */
export function isRecursive(prog: IRProgram, entry: IRFunc): boolean {
  const seen = new Set<string>();
  let recursive = false;
  visitCalls(prog, entry.body, fn => {
    if (fn.cName === entry.cName) recursive = true;
    if (recursive || seen.has(fn.cName)) return false;
    seen.add(fn.cName);
    return true;
  });
  return recursive;
}
