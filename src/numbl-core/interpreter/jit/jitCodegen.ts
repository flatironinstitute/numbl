/**
 * JIT IR -> JavaScript code generation.
 */

import { BinaryOperation, UnaryOperation } from "../../parser/types.js";
import {
  type JitExpr,
  type JitStmt,
  isTensorType,
  isScalarType,
} from "./jitTypes.js";

// ── JS reserved words to mangle ─────────────────────────────────────────

const JS_RESERVED = new Set([
  "abstract",
  "arguments",
  "await",
  "boolean",
  "break",
  "byte",
  "case",
  "catch",
  "char",
  "class",
  "const",
  "continue",
  "debugger",
  "default",
  "delete",
  "do",
  "double",
  "else",
  "enum",
  "eval",
  "export",
  "extends",
  "false",
  "final",
  "finally",
  "float",
  "for",
  "function",
  "goto",
  "if",
  "implements",
  "import",
  "in",
  "instanceof",
  "int",
  "interface",
  "let",
  "long",
  "native",
  "new",
  "null",
  "package",
  "private",
  "protected",
  "public",
  "return",
  "short",
  "static",
  "super",
  "switch",
  "synchronized",
  "this",
  "throw",
  "throws",
  "transient",
  "true",
  "try",
  "typeof",
  "undefined",
  "var",
  "void",
  "volatile",
  "while",
  "with",
  "yield",
]);

function mangle(name: string): string {
  if (JS_RESERVED.has(name)) return `_m$${name}`;
  return name;
}

// ── Entry point ─────────────────────────────────────────────────────────

export function generateJS(
  body: JitStmt[],
  params: string[],
  outputs: string[],
  nargout: number,
  localVars: Set<string>,
  hasTensorOps: boolean
): string {
  const lines: string[] = [];
  const indent = "  ";

  // Declare local variables (not params)
  const locals = [...localVars].filter(v => !params.includes(v));
  if (locals.length > 0) {
    lines.push(`${indent}let ${locals.map(mangle).join(", ")};`);
  }

  // Emit body
  emitStmts(lines, body, indent, hasTensorOps);

  // Return
  const effectiveOutputs = outputs.slice(0, nargout || 1);
  if (effectiveOutputs.length <= 1) {
    lines.push(`${indent}return ${mangle(effectiveOutputs[0])};`);
  } else {
    lines.push(`${indent}return [${effectiveOutputs.map(mangle).join(", ")}];`);
  }

  return lines.join("\n");
}

// ── Statement emission ──────────────────────────────────────────────────

function emitStmts(
  lines: string[],
  stmts: JitStmt[],
  indent: string,
  ht: boolean
): void {
  for (const stmt of stmts) {
    emitStmt(lines, stmt, indent, ht);
  }
}

function emitStmt(
  lines: string[],
  stmt: JitStmt,
  indent: string,
  ht: boolean
): void {
  switch (stmt.tag) {
    case "Assign":
      lines.push(`${indent}${mangle(stmt.name)} = ${emitExpr(stmt.expr, ht)};`);
      break;

    case "ExprStmt":
      lines.push(`${indent}${emitExpr(stmt.expr, ht)};`);
      break;

    case "If": {
      lines.push(`${indent}if (${emitTruthiness(stmt.cond, ht)}) {`);
      emitStmts(lines, stmt.thenBody, indent + "  ", ht);
      for (const eib of stmt.elseifBlocks) {
        lines.push(`${indent}} else if (${emitTruthiness(eib.cond, ht)}) {`);
        emitStmts(lines, eib.body, indent + "  ", ht);
      }
      if (stmt.elseBody) {
        lines.push(`${indent}} else {`);
        emitStmts(lines, stmt.elseBody, indent + "  ", ht);
      }
      lines.push(`${indent}}`);
      break;
    }

    case "For": {
      const v = mangle(stmt.varName);
      const start = emitExpr(stmt.start, ht);
      const end = emitExpr(stmt.end, ht);
      const step = stmt.step ? emitExpr(stmt.step, ht) : "1";
      // Handle both positive and negative steps
      if (stmt.step) {
        lines.push(
          `${indent}for (${v} = ${start}; ${step} > 0 ? ${v} <= ${end} : ${v} >= ${end}; ${v} += ${step}) {`
        );
      } else {
        lines.push(
          `${indent}for (${v} = ${start}; ${v} <= ${end}; ${v} += 1) {`
        );
      }
      emitStmts(lines, stmt.body, indent + "  ", ht);
      lines.push(`${indent}}`);
      break;
    }

    case "While":
      lines.push(`${indent}while (${emitTruthiness(stmt.cond, ht)}) {`);
      emitStmts(lines, stmt.body, indent + "  ", ht);
      lines.push(`${indent}}`);
      break;

    case "Break":
      lines.push(`${indent}break;`);
      break;

    case "Continue":
      lines.push(`${indent}continue;`);
      break;

    case "Return":
      // Early return uses the current output variable values
      lines.push(`${indent}return;`);
      break;
  }
}

// ── Expression emission ─────────────────────────────────────────────────

function emitExpr(expr: JitExpr, ht: boolean): string {
  switch (expr.tag) {
    case "NumberLiteral":
      return String(expr.value);

    case "Var":
      return mangle(expr.name);

    case "Binary":
      return emitBinary(expr, ht);

    case "Unary":
      return emitUnary(expr, ht);

    case "Call":
      return emitCall(expr, ht);
  }
}

function emitBinary(expr: JitExpr & { tag: "Binary" }, ht: boolean): string {
  const left = emitExpr(expr.left, ht);
  const right = emitExpr(expr.right, ht);
  const leftIsTensor = isTensorType(expr.left.jitType);
  const rightIsTensor = isTensorType(expr.right.jitType);

  // Tensor operations use helpers
  if (leftIsTensor || rightIsTensor) {
    return emitTensorBinary(expr.op, left, right);
  }

  // Scalar operations
  switch (expr.op) {
    case BinaryOperation.Add:
      return `(${left} + ${right})`;
    case BinaryOperation.Sub:
      return `(${left} - ${right})`;
    case BinaryOperation.Mul:
    case BinaryOperation.ElemMul:
      return `(${left} * ${right})`;
    case BinaryOperation.Div:
    case BinaryOperation.ElemDiv:
      return `(${left} / ${right})`;
    case BinaryOperation.Pow:
    case BinaryOperation.ElemPow:
      return `Math.pow(${left}, ${right})`;
    case BinaryOperation.Equal:
      return `((${left}) === (${right}) ? 1 : 0)`;
    case BinaryOperation.NotEqual:
      return `((${left}) !== (${right}) ? 1 : 0)`;
    case BinaryOperation.Less:
      return `((${left}) < (${right}) ? 1 : 0)`;
    case BinaryOperation.LessEqual:
      return `((${left}) <= (${right}) ? 1 : 0)`;
    case BinaryOperation.Greater:
      return `((${left}) > (${right}) ? 1 : 0)`;
    case BinaryOperation.GreaterEqual:
      return `((${left}) >= (${right}) ? 1 : 0)`;
    case BinaryOperation.AndAnd:
      return `((${left}) !== 0 && (${right}) !== 0 ? 1 : 0)`;
    case BinaryOperation.OrOr:
      return `((${left}) !== 0 || (${right}) !== 0 ? 1 : 0)`;
    default:
      return `(${left} + ${right})`; // fallback
  }
}

function emitTensorBinary(
  op: BinaryOperation,
  left: string,
  right: string
): string {
  switch (op) {
    case BinaryOperation.Add:
      return `$h.tAdd(${left}, ${right})`;
    case BinaryOperation.Sub:
      return `$h.tSub(${left}, ${right})`;
    case BinaryOperation.Mul:
    case BinaryOperation.ElemMul:
      return `$h.tMul(${left}, ${right})`;
    case BinaryOperation.Div:
    case BinaryOperation.ElemDiv:
      return `$h.tDiv(${left}, ${right})`;
    default:
      return `$h.tAdd(${left}, ${right})`; // fallback
  }
}

function emitUnary(expr: JitExpr & { tag: "Unary" }, ht: boolean): string {
  const operand = emitExpr(expr.operand, ht);

  if (isTensorType(expr.operand.jitType)) {
    switch (expr.op) {
      case UnaryOperation.Minus:
        return `$h.tNeg(${operand})`;
      case UnaryOperation.Plus:
        return operand;
      default:
        return operand;
    }
  }

  switch (expr.op) {
    case UnaryOperation.Plus:
      return `(+${operand})`;
    case UnaryOperation.Minus:
      return `(-${operand})`;
    case UnaryOperation.Not:
      return `((${operand}) !== 0 ? 0 : 1)`;
    default:
      return operand;
  }
}

function emitCall(expr: JitExpr & { tag: "Call" }, ht: boolean): string {
  const args = expr.args.map(a => emitExpr(a, ht));

  // Check if any arg is a tensor -> use tensor helper
  if (expr.args.some(a => isTensorType(a.jitType))) {
    return emitTensorCall(expr.name, args);
  }

  // Scalar math
  return emitScalarCall(expr.name, args);
}

function emitScalarCall(name: string, args: string[]): string {
  switch (name) {
    case "sin":
      return `Math.sin(${args[0]})`;
    case "cos":
      return `Math.cos(${args[0]})`;
    case "tan":
      return `Math.tan(${args[0]})`;
    case "asin":
      return `Math.asin(${args[0]})`;
    case "acos":
      return `Math.acos(${args[0]})`;
    case "atan":
      return `Math.atan(${args[0]})`;
    case "atan2":
      return `Math.atan2(${args[0]}, ${args[1]})`;
    case "sinh":
      return `Math.sinh(${args[0]})`;
    case "cosh":
      return `Math.cosh(${args[0]})`;
    case "tanh":
      return `Math.tanh(${args[0]})`;
    case "sqrt":
      return `Math.sqrt(${args[0]})`;
    case "abs":
      return `Math.abs(${args[0]})`;
    case "floor":
      return `Math.floor(${args[0]})`;
    case "ceil":
      return `Math.ceil(${args[0]})`;
    case "round":
      return `Math.round(${args[0]})`;
    case "fix":
      return `(${args[0]} | 0)`;
    case "exp":
      return `Math.exp(${args[0]})`;
    case "log":
      return `Math.log(${args[0]})`;
    case "log2":
      return `Math.log2(${args[0]})`;
    case "log10":
      return `Math.log10(${args[0]})`;
    case "sign":
      return `Math.sign(${args[0]})`;
    case "min":
      return `Math.min(${args[0]}, ${args[1]})`;
    case "max":
      return `Math.max(${args[0]}, ${args[1]})`;
    case "mod":
      return `((${args[0]} % ${args[1]}) + ${args[1]}) % ${args[1]}`;
    case "rem":
      return `(${args[0]} % ${args[1]})`;
    case "power":
      return `Math.pow(${args[0]}, ${args[1]})`;
    default:
      return `${name}(${args.join(", ")})`; // fallback
  }
}

function emitTensorCall(name: string, args: string[]): string {
  // Map to helper names: tSin, tCos, etc.
  const helperName = `t${name.charAt(0).toUpperCase()}${name.slice(1)}`;
  return `$h.${helperName}(${args.join(", ")})`;
}

// ── Truthiness ──────────────────────────────────────────────────────────

function emitTruthiness(expr: JitExpr, ht: boolean): string {
  // For scalar types, just check !== 0
  if (isScalarType(expr.jitType)) {
    return `(${emitExpr(expr, ht)}) !== 0`;
  }
  // For tensor, we'd need a helper - but conditions should be scalar
  return `(${emitExpr(expr, ht)}) !== 0`;
}
