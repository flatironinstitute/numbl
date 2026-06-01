/**
 * Driver-script prologue extractor.
 *
 * mtoc2 accepts `addpath(<literal>, ...)` only as a leading sequence
 * of top-level statements in the driver script. Each call is parsed
 * into a list of directory arguments (literal strings only) plus an
 * optional trailing `-begin` / `-end` flag.
 *
 * Two call sites:
 *  - The CLI invokes the extractor before scanning the filesystem so
 *    it knows which directories to fold into the workspace search
 *    path.
 *  - `translateProject` / the interpreter run path invoke it purely
 *    to drop the prologue statements from the driver AST before
 *    lowering / execution begins. The dirs are ignored there.
 *
 * The two calls are idempotent: same input AST → same dirs and
 * remaining body, so it doesn't matter that they parse twice.
 *
 * Any `addpath` left in the AST (anywhere outside the leading prefix)
 * still hits the `addpath` builtin's `transfer`, which errors with a
 * span-attributed message. `rmpath` / `savepath` are always rejected
 * by their own `transfer` hooks; the extractor never accepts them.
 *
 * Function-file auto-invoke: a driver whose body (after the prologue)
 * contains only function/classdef definitions and no top-level
 * statements is a "function file" — running it means calling its
 * first function with zero arguments. The extractor appends a
 * synthesized 0-arg call so every execution path (interpreter,
 * js-aot, c-aot) treats it the same way. Mirrors numbl's
 * `Interpreter.run` (interpreter.ts).
 */
import type { AbstractSyntaxTree, Stmt, Expr, Span } from "../parser/index.js";
import { UnsupportedConstruct } from "../lowering/errors.js";

export interface AddPathDir {
  /** Directory string as the user wrote it. Relative paths are NOT
   *  resolved here; callers resolve against `process.cwd()`. */
  dir: string;
  /** Whether the directory inserts at the front of the addpath block
   *  (default / `-begin`) or after the CLI `--path` dirs at the end
   *  (`-end`). */
  position: "begin" | "end";
  /** Span of the originating `addpath(...)` call. */
  span: Span;
}

export interface DriverPrologueOpts {
  /** Set to `true` when a filesystem is available to resolve dirs
   *  against (CLI). When `false` or omitted (web IDE, in-memory
   *  translate, vitest), any leading `addpath` call is itself an
   *  `UnsupportedConstruct`. */
  allowAddpath?: boolean;
}

export interface DriverPrologueResult {
  addpaths: AddPathDir[];
  /** Top-level statements after the prologue. Suitable for direct
   *  use with `Lowerer.lowerProgram` (via a shallow AST copy with
   *  this `body`) or `Interpreter.runProgram`. */
  remainingBody: Stmt[];
}

export function extractDriverPrologue(
  ast: AbstractSyntaxTree,
  opts: DriverPrologueOpts = {}
): DriverPrologueResult {
  const addpaths: AddPathDir[] = [];
  let i = 0;
  for (; i < ast.body.length; i++) {
    const s = ast.body[i];
    if (s.type !== "ExprStmt") break;
    if (s.expr.type !== "FuncCall") break;
    if (s.expr.name !== "addpath") break;
    if (!opts.allowAddpath) {
      throw new UnsupportedConstruct(
        "'addpath' requires a filesystem and is not available in this environment",
        s.span
      );
    }
    for (const d of parseAddpathArgs(s.expr.args, s.span)) {
      addpaths.push({ dir: d.dir, position: d.position, span: s.span });
    }
  }
  const remainingBody = withFunctionFileEntry(ast.body.slice(i));
  return { addpaths, remainingBody };
}

/** If `body` is a function file — only function/classdef definitions,
 *  no top-level statements — append a synthesized 0-arg call to the
 *  first function so the driver actually runs it. Otherwise return
 *  `body` unchanged. Matches numbl's `Interpreter.run`: "Function
 *  file: call the first function with 0 args". */
function withFunctionFileEntry(body: Stmt[]): Stmt[] {
  let firstFn: (Stmt & { type: "Function" }) | null = null;
  for (const s of body) {
    if (s.type === "Function") {
      if (firstFn === null) firstFn = s;
    } else if (s.type !== "ClassDef") {
      // A real top-level statement — not a function file.
      return body;
    }
  }
  if (firstFn === null) return body;
  const call: Stmt = {
    type: "ExprStmt",
    expr: {
      type: "FuncCall",
      name: firstFn.name,
      args: [],
      span: firstFn.span,
    },
    suppressed: true,
    span: firstFn.span,
  };
  return [...body, call];
}

function parseAddpathArgs(
  args: Expr[],
  callSpan: Span
): Array<{ dir: string; position: "begin" | "end" }> {
  if (args.length < 1) {
    throw new UnsupportedConstruct(
      "'addpath' requires at least 1 argument",
      callSpan
    );
  }
  let position: "begin" | "end" = "begin";
  let effective = args;
  const last = args[args.length - 1];
  const lastUnquoted = literalString(last);
  if (lastUnquoted === "-end" || lastUnquoted === "-begin") {
    position = lastUnquoted === "-end" ? "end" : "begin";
    effective = args.slice(0, -1);
  }
  if (effective.length < 1) {
    throw new UnsupportedConstruct(
      "'addpath' requires at least 1 directory argument before the flag",
      callSpan
    );
  }
  const out: Array<{ dir: string; position: "begin" | "end" }> = [];
  for (const a of effective) {
    const s = literalString(a);
    if (s === null) {
      throw new UnsupportedConstruct(
        `'addpath' arguments must be literal strings; mtoc2 resolves the search path statically and cannot evaluate a '${a.type}' expression here`,
        a.span
      );
    }
    if (s === "-end" || s === "-begin") {
      throw new UnsupportedConstruct(
        `'addpath' flag '${s}' may only appear as the last argument`,
        a.span
      );
    }
    out.push({ dir: s, position });
  }
  return out;
}

/** Unwrap a `Char` / `String` AST node into its decoded value. Returns
 *  `null` for any other expression shape — the parser keeps the
 *  surrounding `'`/`"` delimiters and the doubled-quote escape in
 *  `value`, so we strip them here the same way the lowerer does
 *  (`src/lowering/lower.ts`). */
function literalString(e: Expr): string | null {
  if (e.type === "Char") {
    const raw = e.value;
    return raw.slice(1, raw.length - 1).replaceAll("''", "'");
  }
  if (e.type === "String") {
    const raw = e.value;
    return raw.slice(1, raw.length - 1).replaceAll('""', '"');
  }
  return null;
}
