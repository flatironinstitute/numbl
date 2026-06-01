/**
 * Small AST helpers consumed by both the lowerer and the interpreter.
 * Pure functions over the numbl AST shape — no dependency on the
 * lowering / runtime layers.
 */

import type { Expr, LValue } from "./index.js";

/** Walk a member-chain rooted in an `Ident` and return the dotted
 *  identifier (`pkg.fn`, `pkg.sub.fn`, `ClassName.staticMethod`).
 *  Returns null if the chain bottoms out at something that isn't a
 *  plain identifier (e.g. a function call result, a tensor literal).
 *
 *  Used to disambiguate package calls (`pkg.foo(x)` — dotted lookup)
 *  from instance-method calls (`obj.method(x)` — receiver dispatch):
 *  the qualified name is only meaningful when every segment in the
 *  chain is a plain ident.
 *
 *  This helper says nothing about whether the dotted name *resolves*
 *  to anything — that's the caller's job (consult the env / workspace
 *  / class registry). It only structurally extracts the dotted form. */
export function tryExtractDottedName(e: Expr): string | null {
  if (e.type === "Ident") return e.name;
  if (e.type === "Member") {
    const base = tryExtractDottedName(e.base);
    if (base) return `${base}.${e.name}`;
  }
  return null;
}

/** Walk a `Member`-chain Expr down to its root Ident. Returns
 *  `{ root, fields }` (fields in outermost-to-innermost order) or
 *  null when the chain ends at anything other than a bare Ident.
 *
 *  Shared by Member-rooted lvalue and index-base walkers so the
 *  same loop isn't repeated across the lowerer / interpreter. */
export function unwindMemberChain(
  e: Expr
): { root: Extract<Expr, { type: "Ident" }>; fields: string[] } | null {
  const fields: string[] = [];
  let cur: Expr = e;
  while (cur.type === "Member") {
    fields.unshift(cur.name);
    cur = cur.base;
  }
  if (cur.type !== "Ident") return null;
  return { root: cur, fields };
}

/** LValue counterpart of `unwindMemberChain`: walk a `Member`
 *  LValue chain down to its root Ident, returning the root name and
 *  the full field path (including the outer Member's `name`). Returns
 *  null when the chain ends at anything other than a bare Ident.
 *
 *  `LValue.Member` carries no span (unlike `Expr.Member`), so this
 *  helper takes the LValue shape directly. */
export function unwindMemberLvalue(
  lv: LValue
): { rootName: string; fields: string[] } | null {
  if (lv.type !== "Member") return null;
  const fields: string[] = [lv.name];
  let cur: Expr = lv.base;
  while (cur.type === "Member") {
    fields.unshift(cur.name);
    cur = cur.base;
  }
  if (cur.type !== "Ident") return null;
  return { rootName: cur.name, fields };
}
