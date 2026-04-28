/**
 * c-jit-loop codegen — JIT IR → C source.
 *
 * Emits one C function per loop. Inputs are passed by value as
 * `double` args; outputs are written through a `double *out` pointer
 * (one slot per output, in order). Every variable is `double`.
 *
 * Codegen is deliberately narrow — see `whitelist.ts` for the set of
 * IR nodes accepted. Anything outside that set is a programming error
 * here (the executor's `propose()` must reject before reaching this
 * module).
 */

import { BinaryOperation, UnaryOperation } from "../../parser/types.js";
import type { JitExpr, JitStmt } from "../../jitTypes.js";

interface EmitCtx {
  readonly inputs: ReadonlySet<string>;
  readonly outputs: readonly string[];
  /** Mangled variable names — ".name" of every Var/Assign target.
   *  Used to declare locals upfront. */
  readonly locals: Set<string>;
  /** Counter for synthetic temp names (loop iterators). */
  tmpCounter: number;
}

/** Reserved C identifiers the codegen must mangle around. Tiny set —
 *  we never emit struct tags, typedef names, or C keywords beyond
 *  these as variable names. */
const C_RESERVED: ReadonlySet<string> = new Set([
  "auto",
  "break",
  "case",
  "char",
  "const",
  "continue",
  "default",
  "do",
  "double",
  "else",
  "enum",
  "extern",
  "float",
  "for",
  "goto",
  "if",
  "inline",
  "int",
  "long",
  "register",
  "restrict",
  "return",
  "short",
  "signed",
  "sizeof",
  "static",
  "struct",
  "switch",
  "typedef",
  "union",
  "unsigned",
  "void",
  "volatile",
  "while",
  // Common stdlib names we'd shadow.
  "main",
]);

function mangle(name: string): string {
  if (C_RESERVED.has(name)) return `v_${name}`;
  // MATLAB identifiers are already C-identifier-compatible (letters,
  // digits, underscores; no leading digit). Pass through.
  return name;
}

/** Emit a complete C source file. The generated function has the
 *  shape `void <fnName>(double *out, double a, double b, ...)`. */
export function generateCSource(
  fnName: string,
  inputs: readonly string[],
  outputs: readonly string[],
  body: readonly JitStmt[]
): string {
  const ctx: EmitCtx = {
    inputs: new Set(inputs),
    outputs,
    locals: new Set<string>(),
    tmpCounter: 0,
  };

  // Pass 1: collect every assigned name that isn't an input/output.
  for (const s of body) collectLocals(s, ctx);

  // Inputs are received as parameters; outputs are written to *out.
  // Locals (including ones that shadow outputs) are declared in the
  // function body. Outputs that also appear as inputs (loop-carried
  // vars passed in from the caller) are seeded from the input
  // parameter — that's how MATLAB's "outputs are pre-existing locals"
  // semantics map to a stateless C function.
  const lines: string[] = [];
  lines.push(`#include <math.h>`);
  lines.push(``);
  const params = inputs.map(n => `double ${mangle(n)}`);
  lines.push(
    `void ${fnName}(double *out${params.length > 0 ? ", " + params.join(", ") : ""}) {`
  );

  // Declare locals (and any output-only vars that aren't also inputs)
  // up front as `double <name> = 0.0`. MATLAB initializes
  // any-undefined to 0 for scalar paths.
  const localDecls = new Set<string>(ctx.locals);
  for (const o of outputs) {
    if (!ctx.inputs.has(o)) localDecls.add(o);
  }
  for (const i of inputs) localDecls.delete(i);
  if (localDecls.size > 0) {
    const sorted = [...localDecls].sort();
    lines.push(`  double ${sorted.map(n => `${mangle(n)} = 0.0`).join(", ")};`);
  }

  for (const s of body) emitStmt(s, lines, "  ", ctx);

  // Write outputs through the out-pointer in declaration order.
  for (let i = 0; i < outputs.length; i++) {
    lines.push(`  out[${i}] = ${mangle(outputs[i])};`);
  }
  lines.push(`}`);
  return lines.join("\n") + "\n";
}

function collectLocals(s: JitStmt, ctx: EmitCtx): void {
  switch (s.tag) {
    case "Assign":
      if (!ctx.inputs.has(s.name)) ctx.locals.add(s.name);
      return;
    case "For":
      ctx.locals.add(s.varName);
      for (const b of s.body) collectLocals(b, ctx);
      return;
    case "While":
      for (const b of s.body) collectLocals(b, ctx);
      return;
    case "If":
      for (const b of s.thenBody) collectLocals(b, ctx);
      for (const eb of s.elseifBlocks) {
        for (const b of eb.body) collectLocals(b, ctx);
      }
      if (s.elseBody) for (const b of s.elseBody) collectLocals(b, ctx);
      return;
    case "Break":
    case "Continue":
    case "ExprStmt":
    case "SetLoc":
      return;
  }
}

function emitStmt(
  s: JitStmt,
  lines: string[],
  indent: string,
  ctx: EmitCtx
): void {
  switch (s.tag) {
    case "Assign":
      lines.push(`${indent}${mangle(s.name)} = ${emitExpr(s.expr)};`);
      return;
    case "For": {
      // MATLAB `for v = a:b` and `for v = a:s:b`. Emit a C for-loop
      // that drives a synthetic iterator `__t` and assigns `v = __t`
      // as the first statement of the body. This matches MATLAB:
      //   - the loop variable retains the *last iterated value* after
      //     the loop terminates (not the post-increment `__t` value);
      //   - on a zero-iteration range (e.g. `1:0`), the variable is
      //     left at its prior value, not modified.
      // Mirrors the JS-JIT codegen's synthetic-temp pattern.
      const v = mangle(s.varName);
      const t = `__t${++ctx.tmpCounter}`;
      const start = emitExpr(s.start);
      const end = emitExpr(s.end);
      if (s.step === null) {
        lines.push(
          `${indent}for (double ${t} = ${start}; ${t} <= ${end}; ${t} += 1.0) {`
        );
      } else {
        // Non-unit step: support negative as well as positive. Empty
        // range when step == 0 — guard explicitly to avoid an
        // infinite loop on a degenerate input.
        const step = emitExpr(s.step);
        lines.push(
          `${indent}for (double ${t} = ${start}, __step${ctx.tmpCounter} = ${step}; __step${ctx.tmpCounter} != 0.0 && (__step${ctx.tmpCounter} > 0.0 ? ${t} <= ${end} : ${t} >= ${end}); ${t} += __step${ctx.tmpCounter}) {`
        );
      }
      lines.push(`${indent}  ${v} = ${t};`);
      for (const b of s.body) emitStmt(b, lines, indent + "  ", ctx);
      lines.push(`${indent}}`);
      return;
    }
    case "While":
      lines.push(`${indent}while (${emitExpr(s.cond)}) {`);
      for (const b of s.body) emitStmt(b, lines, indent + "  ", ctx);
      lines.push(`${indent}}`);
      return;
    case "If":
      lines.push(`${indent}if (${emitExpr(s.cond)}) {`);
      for (const b of s.thenBody) emitStmt(b, lines, indent + "  ", ctx);
      for (const eb of s.elseifBlocks) {
        lines.push(`${indent}} else if (${emitExpr(eb.cond)}) {`);
        for (const b of eb.body) emitStmt(b, lines, indent + "  ", ctx);
      }
      if (s.elseBody) {
        lines.push(`${indent}} else {`);
        for (const b of s.elseBody) emitStmt(b, lines, indent + "  ", ctx);
      }
      lines.push(`${indent}}`);
      return;
    case "Break":
      lines.push(`${indent}break;`);
      return;
    case "Continue":
      lines.push(`${indent}continue;`);
      return;
    case "ExprStmt":
      lines.push(`${indent}(void)(${emitExpr(s.expr)});`);
      return;
    case "SetLoc":
      // No-op — line tracking is JS-side only.
      return;
    default:
      throw new Error(
        `c-jit-loop codegen: unsupported stmt ${(s as { tag: string }).tag}`
      );
  }
}

function emitExpr(e: JitExpr): string {
  switch (e.tag) {
    case "NumberLiteral":
      return formatDouble(e.value);
    case "Var":
      return mangle(e.name);
    case "Binary":
      return emitBinary(e);
    case "Unary":
      return emitUnary(e);
    case "Call":
      return emitCall(e);
    default:
      throw new Error(
        `c-jit-loop codegen: unsupported expr ${(e as { tag: string }).tag}`
      );
  }
}

function emitBinary(e: JitExpr & { tag: "Binary" }): string {
  const l = emitExpr(e.left);
  const r = emitExpr(e.right);
  switch (e.op) {
    case BinaryOperation.Add:
      return `(${l} + ${r})`;
    case BinaryOperation.Sub:
      return `(${l} - ${r})`;
    case BinaryOperation.Mul:
    case BinaryOperation.ElemMul:
      return `(${l} * ${r})`;
    case BinaryOperation.Div:
    case BinaryOperation.ElemDiv:
      return `(${l} / ${r})`;
    case BinaryOperation.LeftDiv:
    case BinaryOperation.ElemLeftDiv:
      return `(${r} / ${l})`;
    case BinaryOperation.Pow:
    case BinaryOperation.ElemPow:
      return `pow(${l}, ${r})`;
    case BinaryOperation.Equal:
      return `((double)(${l} == ${r}))`;
    case BinaryOperation.NotEqual:
      return `((double)(${l} != ${r}))`;
    case BinaryOperation.Less:
      return `((double)(${l} < ${r}))`;
    case BinaryOperation.LessEqual:
      return `((double)(${l} <= ${r}))`;
    case BinaryOperation.Greater:
      return `((double)(${l} > ${r}))`;
    case BinaryOperation.GreaterEqual:
      return `((double)(${l} >= ${r}))`;
    case BinaryOperation.AndAnd:
      return `((double)((${l}) && (${r})))`;
    case BinaryOperation.OrOr:
      return `((double)((${l}) || (${r})))`;
    default:
      throw new Error(`c-jit-loop codegen: unsupported binary op ${e.op}`);
  }
}

function emitUnary(e: JitExpr & { tag: "Unary" }): string {
  const x = emitExpr(e.operand);
  switch (e.op) {
    case UnaryOperation.Plus:
      return `(+${x})`;
    case UnaryOperation.Minus:
      return `(-${x})`;
    case UnaryOperation.Not:
      return `((double)(!(${x})))`;
    default:
      throw new Error(`c-jit-loop codegen: unsupported unary op ${e.op}`);
  }
}

function emitCall(e: JitExpr & { tag: "Call" }): string {
  const args = e.args.map(emitExpr).join(", ");
  // Math.h names map 1:1 except `abs` (use `fabs` for double). Other
  // builtins outside the whitelist will already have been rejected by
  // `whitelist.ts`.
  const name = e.name === "abs" ? "fabs" : e.name;
  return `${name}(${args})`;
}

/** Emit a double literal that's unambiguous to a C compiler.
 *  - Integers get a trailing `.0` so they're treated as `double`.
 *  - Special values (NaN/Inf) use stdlib macros. */
function formatDouble(v: number): string {
  if (Number.isNaN(v)) return "((double)NAN)";
  if (v === Infinity) return "((double)INFINITY)";
  if (v === -Infinity) return "(-(double)INFINITY)";
  if (Number.isInteger(v)) return `${v}.0`;
  // JS toString gives a round-trippable form for most doubles; for
  // edge-cases that the compiler might re-parse imprecisely, use the
  // hex literal form.
  const s = String(v);
  // Guard: if the literal lacks a decimal point or exponent, append
  // `.0` (e.g. '1e10' is fine, but defensively we handle it).
  if (/[.eE]/.test(s)) return s;
  return `${s}.0`;
}
