/**
 * JIT IR -> JavaScript code generation.
 */

import { BinaryOperation, UnaryOperation } from "../../parser/types.js";
import {
  type JitExpr,
  type JitType,
  type JitStmt,
  isTensorType,
} from "./jitTypes.js";
import { getIBuiltin } from "../builtins/types.js";

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

let _tmpCounter = 0;
let _returnExpr = "undefined";

export function generateJS(
  body: JitStmt[],
  params: string[],
  outputs: string[],
  nargout: number,
  localVars: Set<string>,
  hasTensorOps: boolean
): string {
  _tmpCounter = 0;

  // Compute the return expression for early returns and the final return
  const effectiveOutputs = outputs.slice(0, nargout || 1);
  if (effectiveOutputs.length <= 1) {
    _returnExpr =
      effectiveOutputs.length > 0 ? mangle(effectiveOutputs[0]) : "undefined";
  } else {
    _returnExpr = `[${effectiveOutputs.map(mangle).join(", ")}]`;
  }
  const lines: string[] = [];
  const indent = "  ";

  // Declare local variables (not params)
  const locals = [...localVars].filter(v => !params.includes(v));
  if (locals.length > 0) {
    lines.push(`${indent}var ${locals.map(mangle).join(", ")};`);
  }

  // Emit body
  emitStmts(lines, body, indent, hasTensorOps);

  // Return
  lines.push(`${indent}return ${_returnExpr};`);

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
      const t = `$t${++_tmpCounter}`;
      const start = emitExpr(stmt.start, ht);
      const end = emitExpr(stmt.end, ht);
      const step = stmt.step ? emitExpr(stmt.step, ht) : "1";
      // Use a separate temp loop variable and assign the iterator inside
      // the body. This is important for two reasons:
      // 1. The iterator variable must retain the last value actually used
      //    in the loop body (MATLAB semantics), not the incremented value
      //    that failed the loop condition.
      // 2. This pattern appears to be faster in V8 (reason unclear).
      if (stmt.step) {
        lines.push(
          `${indent}for (var ${t} = ${start}; ${step} > 0 ? ${t} <= ${end} : ${t} >= ${end}; ${t} += ${step}) {`
        );
      } else {
        lines.push(
          `${indent}for (var ${t} = ${start}; ${t} <= ${end}; ${t} += 1) {`
        );
      }
      lines.push(`${indent}  ${v} = ${t};`);
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
      lines.push(`${indent}return ${_returnExpr};`);
      break;

    case "MultiAssign": {
      const args = stmt.args.map(a => emitExpr(a, ht));
      const nargout = stmt.names.length;
      const tmp = `$ma${++_tmpCounter}`;
      lines.push(
        `${indent}const ${tmp} = $h.ibcall(${JSON.stringify(stmt.callName)}, ${nargout}, ${args.join(", ")});`
      );
      for (let i = 0; i < stmt.names.length; i++) {
        const name = stmt.names[i];
        if (name !== null) {
          lines.push(`${indent}${mangle(name)} = ${tmp}[${i}];`);
        }
      }
      break;
    }
  }
}

// ── Expression emission ─────────────────────────────────────────────────

function isComplexType(t: JitType): boolean {
  return (
    t.kind === "complex_or_number" ||
    (t.kind === "tensor" && t.isComplex === true)
  );
}

function emitExpr(expr: JitExpr, ht: boolean): string {
  switch (expr.tag) {
    case "NumberLiteral":
      return String(expr.value);

    case "ImagLiteral":
      return `{kind:"complex_number",re:0,im:1}`;

    case "Var":
      return mangle(expr.name);

    case "Binary":
      return emitBinary(expr, ht);

    case "Unary":
      return emitUnary(expr, ht);

    case "Call":
      return emitCall(expr, ht);

    case "TensorLiteral":
      return emitTensorLiteral(expr, ht);

    case "StringLiteral":
      return JSON.stringify(expr.value);

    case "UserCall":
      return emitUserCall(expr, ht);

    case "Index":
      return emitIndex(expr, ht);
  }
}

function emitBinary(expr: JitExpr & { tag: "Binary" }, ht: boolean): string {
  const left = emitExpr(expr.left, ht);
  const right = emitExpr(expr.right, ht);
  const leftIsTensor = isTensorType(expr.left.jitType);
  const rightIsTensor = isTensorType(expr.right.jitType);
  const anyComplex =
    isComplexType(expr.left.jitType) || isComplexType(expr.right.jitType);

  // Tensor operations use helpers (handles both real and complex tensors)
  if (leftIsTensor || rightIsTensor) {
    return emitTensorBinary(expr.op, left, right);
  }

  // Complex scalar operations use helpers
  if (anyComplex) {
    return emitComplexBinary(expr.op, left, right);
  }

  // Real scalar operations
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
      throw new Error(`JIT codegen: unsupported scalar binary op ${expr.op}`);
  }
}

function emitComplexBinary(
  op: BinaryOperation,
  left: string,
  right: string
): string {
  switch (op) {
    case BinaryOperation.Add:
      return `$h.cAdd(${left}, ${right})`;
    case BinaryOperation.Sub:
      return `$h.cSub(${left}, ${right})`;
    case BinaryOperation.Mul:
    case BinaryOperation.ElemMul:
      return `$h.cMul(${left}, ${right})`;
    case BinaryOperation.Div:
    case BinaryOperation.ElemDiv:
      return `$h.cDiv(${left}, ${right})`;
    default:
      throw new Error(`JIT codegen: unsupported complex binary op ${op}`);
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
      throw new Error(`JIT codegen: unsupported tensor binary op ${op}`);
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
        throw new Error(`JIT codegen: unsupported tensor unary op ${expr.op}`);
    }
  }

  if (isComplexType(expr.operand.jitType)) {
    switch (expr.op) {
      case UnaryOperation.Minus:
        return `$h.cNeg(${operand})`;
      case UnaryOperation.Plus:
        return operand;
      default:
        throw new Error(`JIT codegen: unsupported complex unary op ${expr.op}`);
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
      throw new Error(`JIT codegen: unsupported scalar unary op ${expr.op}`);
  }
}

function emitUserCall(
  expr: JitExpr & { tag: "UserCall" },
  ht: boolean
): string {
  const args = expr.args.map(a => emitExpr(a, ht));
  return `${expr.jitName}(${args.join(", ")})`;
}

function emitTensorLiteral(
  expr: JitExpr & { tag: "TensorLiteral" },
  ht: boolean
): string {
  const { rows, nRows, nCols } = expr;
  // Column-major order: iterate columns first, then rows
  const elems: string[] = [];
  for (let c = 0; c < nCols; c++) {
    for (let r = 0; r < nRows; r++) {
      elems.push(emitExpr(rows[r][c], ht));
    }
  }
  if (expr.jitType.kind === "tensor" && expr.jitType.isComplex === true) {
    // Extract real and imag parts
    const reElems: string[] = [];
    const imElems: string[] = [];
    for (let c = 0; c < nCols; c++) {
      for (let r = 0; r < nRows; r++) {
        const e = rows[r][c];
        if (e.jitType.kind === "complex_or_number") {
          const s = emitExpr(e, ht);
          reElems.push(`$h.re(${s})`);
          imElems.push(`$h.im(${s})`);
        } else {
          reElems.push(emitExpr(e, ht));
          imElems.push("0");
        }
      }
    }
    return `$h.mkTensorC([${reElems.join(", ")}], [${imElems.join(", ")}], [${nRows}, ${nCols}])`;
  }
  return `$h.mkTensor([${elems.join(", ")}], [${nRows}, ${nCols}])`;
}

function emitIndex(expr: JitExpr & { tag: "Index" }, ht: boolean): string {
  const base = emitExpr(expr.base, ht);
  const indices = expr.indices.map(i => emitExpr(i, ht));
  if (indices.length === 1) return `$h.idx1(${base}, ${indices[0]})`;
  if (indices.length === 2)
    return `$h.idx2(${base}, ${indices[0]}, ${indices[1]})`;
  return `$h.idxN(${base}, [${indices.join(", ")}])`;
}

function emitCall(expr: JitExpr & { tag: "Call" }, ht: boolean): string {
  const args = expr.args.map(a => emitExpr(a, ht));
  // Try fast-path emission if the IBuiltin provides one
  const ib = getIBuiltin(expr.name);
  if (ib?.jitEmit) {
    const argTypes = expr.args.map(a => a.jitType);
    const fast = ib.jitEmit(args, argTypes);
    if (fast) return fast;
  }
  return `$h.ib_${expr.name}(${args.join(", ")})`;
}

// ── Truthiness ──────────────────────────────────────────────────────────

function emitTruthiness(expr: JitExpr, ht: boolean): string {
  // String/char conditions are rejected during lowering.
  // Complex values are objects, so !== 0 is always true — use $h.cTruthy.
  if (expr.jitType.kind === "complex_or_number") {
    return `$h.cTruthy(${emitExpr(expr, ht)})`;
  }
  return `(${emitExpr(expr, ht)}) !== 0`;
}
