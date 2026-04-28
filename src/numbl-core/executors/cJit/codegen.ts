/**
 * c-jit-loop codegen — JIT IR → C source.
 *
 * Emits one C function per loop. Each scalar variable is encoded
 * either as a single `double` (real) or a pair `<name>_re`, `<name>_im`
 * (complex). Inputs are passed by value (real → 1 arg, complex →
 * 2 args); outputs are written through a `double *out` pointer
 * (real → 1 slot, complex → 2 slots).
 *
 * Codegen is deliberately narrow — see `whitelist.ts` for the set of
 * IR nodes accepted. Anything outside that set is a programming error
 * here (the executor's `propose()` must reject before reaching this
 * module).
 */

import { BinaryOperation, UnaryOperation } from "../../parser/types.js";
import type { JitExpr, JitStmt, JitType } from "../../jitTypes.js";

/** Per-variable C encoding. */
export type VarEncoding = "real" | "complex";

/** Resolved (input + local) variable types. Codegen keys lookups on
 *  the variable name; both Vars and Assigns consult this map. */
export type VarTypeMap = ReadonlyMap<string, VarEncoding>;

/** Emitted scalar value. Real values are a single C expression;
 *  complex values are a (re, im) pair. */
export type Emitted =
  | { kind: "real"; expr: string }
  | { kind: "complex"; re: string; im: string };

interface EmitCtx {
  readonly varTypes: VarTypeMap;
  /** Counter for synthetic temp names (loop iterators, complex
   *  intermediates). */
  tmpCounter: number;
}

/** Reserved C identifiers the codegen must mangle around. */
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
  "main",
]);

function mangle(name: string): string {
  if (C_RESERVED.has(name)) return `v_${name}`;
  return name;
}

function paramRe(name: string): string {
  return `${mangle(name)}_re`;
}
function paramIm(name: string): string {
  return `${mangle(name)}_im`;
}

/** Walk the IR + input typings and produce a per-name encoding map.
 *  A name is "complex" iff at least one read or write of it has a
 *  complex_or_number type. Otherwise "real".
 *
 *  This mirrors the JIT's type widening: by the time the IR is
 *  produced, every Var / Assign for the same name has been unified
 *  to a single type. The walker just records that type. */
export function inferVarEncodings(
  inputs: readonly string[],
  inputTypes: readonly JitType[],
  body: readonly JitStmt[]
): Map<string, VarEncoding> {
  const map = new Map<string, VarEncoding>();
  for (let i = 0; i < inputs.length; i++) {
    map.set(inputs[i], jitTypeEncoding(inputTypes[i]));
  }
  for (const s of body) collectStmtVars(s, map);
  return map;
}

function jitTypeEncoding(t: JitType): VarEncoding {
  if (t.kind === "complex_or_number") return "complex";
  return "real";
}

function recordVar(name: string, t: JitType, map: Map<string, VarEncoding>) {
  const enc = jitTypeEncoding(t);
  const prev = map.get(name);
  if (prev === undefined) {
    map.set(name, enc);
  } else if (enc === "complex" && prev === "real") {
    map.set(name, "complex");
  }
}

function collectStmtVars(s: JitStmt, map: Map<string, VarEncoding>): void {
  switch (s.tag) {
    case "Assign":
      recordVar(s.name, s.expr.jitType, map);
      collectExprVars(s.expr, map);
      return;
    case "For":
      recordVar(s.varName, s.start.jitType, map);
      collectExprVars(s.start, map);
      if (s.step !== null) collectExprVars(s.step, map);
      collectExprVars(s.end, map);
      for (const b of s.body) collectStmtVars(b, map);
      return;
    case "While":
      collectExprVars(s.cond, map);
      for (const b of s.body) collectStmtVars(b, map);
      return;
    case "If":
      collectExprVars(s.cond, map);
      for (const b of s.thenBody) collectStmtVars(b, map);
      for (const eb of s.elseifBlocks) {
        collectExprVars(eb.cond, map);
        for (const b of eb.body) collectStmtVars(b, map);
      }
      if (s.elseBody) for (const b of s.elseBody) collectStmtVars(b, map);
      return;
    case "Break":
    case "Continue":
    case "SetLoc":
      return;
    case "ExprStmt":
      collectExprVars(s.expr, map);
      return;
  }
}

function collectExprVars(e: JitExpr, map: Map<string, VarEncoding>): void {
  switch (e.tag) {
    case "Var":
      recordVar(e.name, e.jitType, map);
      return;
    case "Binary":
      collectExprVars(e.left, map);
      collectExprVars(e.right, map);
      return;
    case "Unary":
      collectExprVars(e.operand, map);
      return;
    case "Call":
      for (const a of e.args) collectExprVars(a, map);
      return;
    case "NumberLiteral":
    case "ImagLiteral":
    case "StringLiteral":
      return;
    default:
      // Whitelist guarantees we don't see other shapes.
      return;
  }
}

/** Per-variable slot count in the marshaling ABI: real → 1, complex → 2. */
export function varSlotCount(enc: VarEncoding): number {
  return enc === "complex" ? 2 : 1;
}

/** Total number of `double` slots a variable list consumes. */
export function totalSlotCount(
  names: readonly string[],
  varTypes: VarTypeMap
): number {
  let n = 0;
  for (const name of names) n += varSlotCount(varTypes.get(name) ?? "real");
  return n;
}

/** Emit a complete C source file. */
export function generateCSource(
  fnName: string,
  inputs: readonly string[],
  outputs: readonly string[],
  body: readonly JitStmt[],
  varTypes: VarTypeMap
): string {
  const ctx: EmitCtx = { varTypes, tmpCounter: 0 };

  // Build the parameter list. Each input contributes 1 or 2 params.
  const params: string[] = [];
  for (const name of inputs) {
    if (encOf(name, varTypes) === "complex") {
      params.push(`double ${paramRe(name)}`);
      params.push(`double ${paramIm(name)}`);
    } else {
      params.push(`double ${mangle(name)}`);
    }
  }

  const lines: string[] = [];
  lines.push(`#include <math.h>`);
  lines.push(``);
  lines.push(
    `void ${fnName}(double *out${params.length > 0 ? ", " + params.join(", ") : ""}) {`
  );

  // Declare locals (every assigned name not in inputs). MATLAB
  // initializes any-undefined to 0 for scalar paths.
  const inputSet = new Set(inputs);
  const localNames: string[] = [];
  for (const [name] of varTypes) {
    if (inputSet.has(name)) continue;
    localNames.push(name);
  }
  // Outputs that aren't inputs and weren't assigned (e.g., a loop
  // variable that's live-out but never written here) — also include.
  for (const o of outputs) {
    if (!inputSet.has(o) && !localNames.includes(o)) {
      localNames.push(o);
    }
  }
  if (localNames.length > 0) {
    const sortedLocals = [...localNames].sort();
    const realDecls: string[] = [];
    const complexDecls: string[] = [];
    for (const name of sortedLocals) {
      if (encOf(name, varTypes) === "complex") {
        complexDecls.push(`${paramRe(name)} = 0.0`);
        complexDecls.push(`${paramIm(name)} = 0.0`);
      } else {
        realDecls.push(`${mangle(name)} = 0.0`);
      }
    }
    if (realDecls.length > 0) {
      lines.push(`  double ${realDecls.join(", ")};`);
    }
    if (complexDecls.length > 0) {
      lines.push(`  double ${complexDecls.join(", ")};`);
    }
  }

  for (const s of body) emitStmt(s, lines, "  ", ctx);

  // Write outputs through the out-pointer in declaration order.
  // Complex outputs occupy two consecutive slots (re, im).
  let slot = 0;
  for (const o of outputs) {
    if (encOf(o, varTypes) === "complex") {
      lines.push(`  out[${slot}] = ${paramRe(o)};`);
      lines.push(`  out[${slot + 1}] = ${paramIm(o)};`);
      slot += 2;
    } else {
      lines.push(`  out[${slot}] = ${mangle(o)};`);
      slot += 1;
    }
  }
  lines.push(`}`);
  return lines.join("\n") + "\n";
}

function encOf(name: string, varTypes: VarTypeMap): VarEncoding {
  return varTypes.get(name) ?? "real";
}

function emitStmt(
  s: JitStmt,
  lines: string[],
  indent: string,
  ctx: EmitCtx
): void {
  switch (s.tag) {
    case "Assign": {
      const rhs = emitExpr(s.expr, ctx);
      const enc = encOf(s.name, ctx.varTypes);
      if (enc === "complex") {
        const c = asComplex(rhs);
        // Emit through fresh temps to handle `z = z * z + c` aliasing
        // — without temps, `z_re = ...; z_im = ...` would read the
        // already-overwritten z_re inside z_im's RHS.
        const t = ++ctx.tmpCounter;
        lines.push(`${indent}double __re${t} = ${c.re};`);
        lines.push(`${indent}double __im${t} = ${c.im};`);
        lines.push(`${indent}${paramRe(s.name)} = __re${t};`);
        lines.push(`${indent}${paramIm(s.name)} = __im${t};`);
      } else {
        // The whitelist guarantees the assigned expr's jitType is
        // real (the type system unified it that way before lowering).
        if (rhs.kind === "complex") {
          throw new Error(
            `c-jit-loop codegen: cannot assign complex value to real var ${s.name}`
          );
        }
        lines.push(`${indent}${mangle(s.name)} = ${rhs.expr};`);
      }
      return;
    }
    case "For": {
      const v = mangle(s.varName);
      const t = `__t${++ctx.tmpCounter}`;
      const start = realExpr(emitExpr(s.start, ctx));
      const end = realExpr(emitExpr(s.end, ctx));
      if (s.step === null) {
        lines.push(
          `${indent}for (double ${t} = ${start}; ${t} <= ${end}; ${t} += 1.0) {`
        );
      } else {
        const step = realExpr(emitExpr(s.step, ctx));
        const sn = ctx.tmpCounter;
        lines.push(
          `${indent}for (double ${t} = ${start}, __step${sn} = ${step}; __step${sn} != 0.0 && (__step${sn} > 0.0 ? ${t} <= ${end} : ${t} >= ${end}); ${t} += __step${sn}) {`
        );
      }
      lines.push(`${indent}  ${v} = ${t};`);
      for (const b of s.body) emitStmt(b, lines, indent + "  ", ctx);
      lines.push(`${indent}}`);
      return;
    }
    case "While": {
      const cond = realExpr(emitExpr(s.cond, ctx));
      lines.push(`${indent}while (${cond}) {`);
      for (const b of s.body) emitStmt(b, lines, indent + "  ", ctx);
      lines.push(`${indent}}`);
      return;
    }
    case "If": {
      const cond = realExpr(emitExpr(s.cond, ctx));
      lines.push(`${indent}if (${cond}) {`);
      for (const b of s.thenBody) emitStmt(b, lines, indent + "  ", ctx);
      for (const eb of s.elseifBlocks) {
        const ec = realExpr(emitExpr(eb.cond, ctx));
        lines.push(`${indent}} else if (${ec}) {`);
        for (const b of eb.body) emitStmt(b, lines, indent + "  ", ctx);
      }
      if (s.elseBody) {
        lines.push(`${indent}} else {`);
        for (const b of s.elseBody) emitStmt(b, lines, indent + "  ", ctx);
      }
      lines.push(`${indent}}`);
      return;
    }
    case "Break":
      lines.push(`${indent}break;`);
      return;
    case "Continue":
      lines.push(`${indent}continue;`);
      return;
    case "ExprStmt": {
      const v = emitExpr(s.expr, ctx);
      // Discard result; cast to void.
      if (v.kind === "complex") {
        lines.push(`${indent}(void)(${v.re});`);
        lines.push(`${indent}(void)(${v.im});`);
      } else {
        lines.push(`${indent}(void)(${v.expr});`);
      }
      return;
    }
    case "SetLoc":
      return;
    default:
      throw new Error(
        `c-jit-loop codegen: unsupported stmt ${(s as { tag: string }).tag}`
      );
  }
}

function emitExpr(e: JitExpr, ctx: EmitCtx): Emitted {
  switch (e.tag) {
    case "NumberLiteral":
      return { kind: "real", expr: formatDouble(e.value) };
    case "ImagLiteral":
      // The unit imaginary i = 0 + 1i.
      return { kind: "complex", re: "0.0", im: "1.0" };
    case "Var": {
      const enc = encOf(e.name, ctx.varTypes);
      if (enc === "complex") {
        return { kind: "complex", re: paramRe(e.name), im: paramIm(e.name) };
      }
      return { kind: "real", expr: mangle(e.name) };
    }
    case "Binary":
      return emitBinary(e, ctx);
    case "Unary":
      return emitUnary(e, ctx);
    case "Call":
      return emitCall(e, ctx);
    default:
      throw new Error(
        `c-jit-loop codegen: unsupported expr ${(e as { tag: string }).tag}`
      );
  }
}

function emitBinary(e: JitExpr & { tag: "Binary" }, ctx: EmitCtx): Emitted {
  const l = emitExpr(e.left, ctx);
  const r = emitExpr(e.right, ctx);

  // Comparison / logical ops: the whitelist already restricted these
  // to real-typed conds; both operands are real here.
  switch (e.op) {
    case BinaryOperation.Equal:
      return {
        kind: "real",
        expr: `((double)(${realExpr(l)} == ${realExpr(r)}))`,
      };
    case BinaryOperation.NotEqual:
      return {
        kind: "real",
        expr: `((double)(${realExpr(l)} != ${realExpr(r)}))`,
      };
    case BinaryOperation.Less:
      return {
        kind: "real",
        expr: `((double)(${realExpr(l)} < ${realExpr(r)}))`,
      };
    case BinaryOperation.LessEqual:
      return {
        kind: "real",
        expr: `((double)(${realExpr(l)} <= ${realExpr(r)}))`,
      };
    case BinaryOperation.Greater:
      return {
        kind: "real",
        expr: `((double)(${realExpr(l)} > ${realExpr(r)}))`,
      };
    case BinaryOperation.GreaterEqual:
      return {
        kind: "real",
        expr: `((double)(${realExpr(l)} >= ${realExpr(r)}))`,
      };
    case BinaryOperation.AndAnd:
      return {
        kind: "real",
        expr: `((double)((${realExpr(l)}) && (${realExpr(r)})))`,
      };
    case BinaryOperation.OrOr:
      return {
        kind: "real",
        expr: `((double)((${realExpr(l)}) || (${realExpr(r)})))`,
      };
  }

  // Arithmetic ops: dispatch by whether either operand is complex.
  const isComplex =
    l.kind === "complex" ||
    r.kind === "complex" ||
    e.jitType.kind === "complex_or_number";

  if (!isComplex) {
    const lr = realExpr(l);
    const rr = realExpr(r);
    switch (e.op) {
      case BinaryOperation.Add:
        return { kind: "real", expr: `(${lr} + ${rr})` };
      case BinaryOperation.Sub:
        return { kind: "real", expr: `(${lr} - ${rr})` };
      case BinaryOperation.Mul:
      case BinaryOperation.ElemMul:
        return { kind: "real", expr: `(${lr} * ${rr})` };
      case BinaryOperation.Div:
      case BinaryOperation.ElemDiv:
        return { kind: "real", expr: `(${lr} / ${rr})` };
      case BinaryOperation.LeftDiv:
      case BinaryOperation.ElemLeftDiv:
        return { kind: "real", expr: `(${rr} / ${lr})` };
      case BinaryOperation.Pow:
      case BinaryOperation.ElemPow:
        return { kind: "real", expr: `pow(${lr}, ${rr})` };
      default:
        throw new Error(`c-jit-loop codegen: unsupported binary op ${e.op}`);
    }
  }

  // Promote operands to complex pairs.
  const lc = asComplex(l);
  const rc = asComplex(r);
  switch (e.op) {
    case BinaryOperation.Add:
      return {
        kind: "complex",
        re: `(${lc.re} + ${rc.re})`,
        im: `(${lc.im} + ${rc.im})`,
      };
    case BinaryOperation.Sub:
      return {
        kind: "complex",
        re: `(${lc.re} - ${rc.re})`,
        im: `(${lc.im} - ${rc.im})`,
      };
    case BinaryOperation.Mul:
    case BinaryOperation.ElemMul:
      // (a + bi) * (c + di) = (ac - bd) + (ad + bc) i
      return {
        kind: "complex",
        re: `(${lc.re} * ${rc.re} - ${lc.im} * ${rc.im})`,
        im: `(${lc.re} * ${rc.im} + ${lc.im} * ${rc.re})`,
      };
    case BinaryOperation.Div:
    case BinaryOperation.ElemDiv: {
      // (a + bi) / (c + di) = ((ac + bd) + (bc - ad)i) / (c^2 + d^2)
      // Hoist the denominator into a temp to keep the emitted
      // expression compact and let the compiler share the divide.
      const t = ++ctx.tmpCounter;
      // Use comma operator + statement-expr macro? Simpler: emit two
      // C exprs with shared temp via inline struct. Easier: just
      // duplicate `(c*c + d*d)` and trust the compiler to CSE it.
      const denom = `(${rc.re} * ${rc.re} + ${rc.im} * ${rc.im})`;
      void t;
      return {
        kind: "complex",
        re: `((${lc.re} * ${rc.re} + ${lc.im} * ${rc.im}) / ${denom})`,
        im: `((${lc.im} * ${rc.re} - ${lc.re} * ${rc.im}) / ${denom})`,
      };
    }
    case BinaryOperation.LeftDiv:
    case BinaryOperation.ElemLeftDiv: {
      const denom = `(${lc.re} * ${lc.re} + ${lc.im} * ${lc.im})`;
      return {
        kind: "complex",
        re: `((${rc.re} * ${lc.re} + ${rc.im} * ${lc.im}) / ${denom})`,
        im: `((${rc.im} * ${lc.re} - ${rc.re} * ${lc.im}) / ${denom})`,
      };
    }
    default:
      throw new Error(
        `c-jit-loop codegen: unsupported complex binary op ${e.op}`
      );
  }
}

function emitUnary(e: JitExpr & { tag: "Unary" }, ctx: EmitCtx): Emitted {
  const v = emitExpr(e.operand, ctx);
  switch (e.op) {
    case UnaryOperation.Plus:
      return v;
    case UnaryOperation.Minus:
      if (v.kind === "complex") {
        return { kind: "complex", re: `(-${v.re})`, im: `(-${v.im})` };
      }
      return { kind: "real", expr: `(-${v.expr})` };
    case UnaryOperation.Not:
      // The whitelist forbids Not on complex.
      return { kind: "real", expr: `((double)(!(${realExpr(v)})))` };
    default:
      throw new Error(`c-jit-loop codegen: unsupported unary op ${e.op}`);
  }
}

function emitCall(e: JitExpr & { tag: "Call" }, ctx: EmitCtx): Emitted {
  // `real`, `imag`, `conj` are projection ops that may take real or
  // complex args.
  if (e.name === "real") {
    const v = emitExpr(e.args[0], ctx);
    return v.kind === "complex"
      ? { kind: "real", expr: v.re }
      : { kind: "real", expr: v.expr };
  }
  if (e.name === "imag") {
    const v = emitExpr(e.args[0], ctx);
    return v.kind === "complex"
      ? { kind: "real", expr: v.im }
      : { kind: "real", expr: "0.0" };
  }
  if (e.name === "conj") {
    const v = emitExpr(e.args[0], ctx);
    if (v.kind === "complex") {
      return { kind: "complex", re: v.re, im: `(-${v.im})` };
    }
    return v;
  }

  // Real-only math (sin, cos, etc). Whitelist guarantees args are real.
  const args = e.args.map(a => realExpr(emitExpr(a, ctx))).join(", ");
  const name = e.name === "abs" ? "fabs" : e.name;
  return { kind: "real", expr: `${name}(${args})` };
}

/** Emit a value as a real C expression. Throws if the value is
 *  complex (caller bug — would have failed the whitelist). */
function realExpr(v: Emitted): string {
  if (v.kind === "complex") {
    throw new Error(
      "c-jit-loop codegen: expected a real value but got complex"
    );
  }
  return v.expr;
}

/** Coerce an emitted value to a complex pair. Reals become (re, 0). */
function asComplex(v: Emitted): { re: string; im: string } {
  if (v.kind === "complex") return { re: v.re, im: v.im };
  return { re: v.expr, im: "0.0" };
}

/** Emit a double literal that's unambiguous to a C compiler. */
function formatDouble(v: number): string {
  if (Number.isNaN(v)) return "((double)NAN)";
  if (v === Infinity) return "((double)INFINITY)";
  if (v === -Infinity) return "(-(double)INFINITY)";
  if (Number.isInteger(v)) return `${v}.0`;
  const s = String(v);
  if (/[.eE]/.test(s)) return s;
  return `${s}.0`;
}
