/**
 * Numbl-like pretty-printer for the IR. Used by the C emitter to
 * annotate the generated source with readable comments — function
 * signatures with detailed type info, and a per-statement summary
 * of the lowered expression above each emitted block.
 *
 * The output is NOT a faithful reproduction of the user's source
 * (synthetic names like `_mtoc2_t1`, ANF temps, and folded branches
 * are visible). It IS a faithful render of what the IR actually says
 * after lowering — useful when debugging the translator.
 */

import type { IRExpr, IRStmt, IRFunc, IRProgram } from "../lowering/ir.js";
import { BinaryOperation, UnaryOperation } from "../parser/index.js";
import { typeToString } from "../lowering/types.js";

function binaryOpSym(op: BinaryOperation): string {
  switch (op) {
    case BinaryOperation.Add:
      return "+";
    case BinaryOperation.Sub:
      return "-";
    case BinaryOperation.Mul:
      return "*";
    case BinaryOperation.ElemMul:
      return ".*";
    case BinaryOperation.Div:
      return "/";
    case BinaryOperation.ElemDiv:
      return "./";
    case BinaryOperation.LeftDiv:
      return "\\";
    case BinaryOperation.ElemLeftDiv:
      return ".\\";
    case BinaryOperation.Pow:
      return "^";
    case BinaryOperation.ElemPow:
      return ".^";
    case BinaryOperation.Equal:
      return "==";
    case BinaryOperation.NotEqual:
      return "~=";
    case BinaryOperation.Less:
      return "<";
    case BinaryOperation.LessEqual:
      return "<=";
    case BinaryOperation.Greater:
      return ">";
    case BinaryOperation.GreaterEqual:
      return ">=";
    case BinaryOperation.OrOr:
      return "||";
    case BinaryOperation.AndAnd:
      return "&&";
    case BinaryOperation.BitOr:
      return "|";
    case BinaryOperation.BitAnd:
      return "&";
  }
}

function unaryOpSym(op: UnaryOperation): { sym: string; postfix: boolean } {
  switch (op) {
    case UnaryOperation.Plus:
      return { sym: "+", postfix: false };
    case UnaryOperation.Minus:
      return { sym: "-", postfix: false };
    case UnaryOperation.Not:
      return { sym: "~", postfix: false };
    case UnaryOperation.Transpose:
      return { sym: "'", postfix: true };
    case UnaryOperation.NonConjugateTranspose:
      return { sym: ".'", postfix: true };
  }
}

function numLitText(v: number): string {
  if (Number.isNaN(v)) return "NaN";
  if (v === Infinity) return "Inf";
  if (v === -Infinity) return "-Inf";
  if (Number.isInteger(v) && Math.abs(v) < 1e15) return v.toString();
  return v.toString();
}

function irExprToString(e: IRExpr): string {
  switch (e.kind) {
    case "NumLit":
      return numLitText(e.value);
    case "ImagLit":
      return `${numLitText(e.value)}i`;
    case "StringLit":
      return `'${e.value}'`;
    case "Var":
      return e.name;
    case "Binary":
      return `(${irExprToString(e.left)} ${binaryOpSym(e.op)} ${irExprToString(e.right)})`;
    case "Unary": {
      const { sym, postfix } = unaryOpSym(e.op);
      const inner = irExprToString(e.operand);
      return postfix ? `${inner}${sym}` : `${sym}${inner}`;
    }
    case "Call":
      return `${e.name}(${e.args.map(irExprToString).join(", ")})`;
    case "TensorBuild": {
      const [rows, cols] = e.shape;
      if (rows === 0 || cols === 0) return "[]";
      // elements is column-major: index = c*rows + r.
      const rowsOut: string[] = [];
      for (let r = 0; r < rows; r++) {
        const cells: string[] = [];
        for (let c = 0; c < cols; c++) {
          cells.push(irExprToString(e.elements[c * rows + r]));
        }
        rowsOut.push(cells.join(" "));
      }
      return `[${rowsOut.join("; ")}]`;
    }
    case "TensorConcat": {
      const rowsOut = e.cells.map(row => row.map(irExprToString).join(" "));
      return `[${rowsOut.join("; ")}]`;
    }
    case "CellLit": {
      const [rows, cols] = e.shape;
      if (rows === 0 || cols === 0) return "{}";
      const rowsOut: string[] = [];
      for (let r = 0; r < rows; r++) {
        const cells: string[] = [];
        for (let c = 0; c < cols; c++) {
          cells.push(irExprToString(e.elements[c * rows + r]));
        }
        rowsOut.push(cells.join(", "));
      }
      return `{${rowsOut.join("; ")}}`;
    }
    case "CellEmpty":
      return `cell(${e.dims.map(irExprToString).join(", ")})`;
    case "CellIndexLoad":
      return `${irExprToString(e.base)}{${e.indices.map(irExprToString).join(", ")}}`;
    case "HandleLit":
      if (e.ty.kind === "Handle") {
        if (e.captures.length === 0) return `@${e.ty.targetName}`;
        const args = e.ty.ast.params
          .slice(0, e.ty.ast.params.length - e.captures.length)
          .join(", ");
        return `@(${args}) ${e.ty.targetName}`;
      }
      return `@?`;
    case "HandleCaptureLoad":
      return `${e.base.name}.${e.captureName}`;
    case "StructLit": {
      const parts = e.fields.map(
        f => `'${f.name}', ${irExprToString(f.value)}`
      );
      return `struct(${parts.join(", ")})`;
    }
    case "MemberLoad":
      return `${irExprToString(e.base)}.${e.field}`;
    case "IndexLoad":
      return `${irExprToString(e.base)}(${e.indices.map(irExprToString).join(", ")})`;
    case "IndexSlice": {
      const slotStrs = e.index.map(slot => {
        if (slot.kind === "Colon") return ":";
        if (slot.kind === "Scalar") return irExprToString(slot.expr);
        if (slot.kind === "IndexVec") return irExprToString(slot.expr);
        if (slot.kind === "LogicalMask") return irExprToString(slot.expr);
        const stepPart =
          slot.step.kind === "NumLit" && slot.step.value === 1
            ? ""
            : `${irExprToString(slot.step)}:`;
        return `${irExprToString(slot.start)}:${stepPart}${irExprToString(slot.end)}`;
      });
      return `${irExprToString(e.base)}(${slotStrs.join(", ")})`;
    }
    case "EndRef":
      return "end";
    case "MakeRange": {
      const stepPart =
        e.step.kind === "NumLit" && e.step.value === 1
          ? ""
          : `${irExprToString(e.step)}:`;
      return `${irExprToString(e.start)}:${stepPart}${irExprToString(e.end)}`;
    }
  }
}

/** One-line summary of a statement (no trailing newline). Returns null
 *  for stmts the emitter handles structurally (compound-block bodies
 *  get their own per-stmt comments inside). */
export function irStmtHeader(s: IRStmt): string | null {
  switch (s.kind) {
    case "ExprStmt":
      return irExprToString(s.expr);
    case "Assign":
      return `${s.name} = ${irExprToString(s.expr)}`;
    case "If":
      return `if ${irExprToString(s.cond)}`;
    case "While":
      return `while ${irExprToString(s.cond)}`;
    case "For": {
      const stepPart = s.step === 1 ? "" : `${numLitText(s.step)}:`;
      return `for ${s.varName} = ${irExprToString(s.start)}:${stepPart}${irExprToString(s.end)}`;
    }
    case "ReturnFromFunction":
      return "return";
    case "Break":
      return "break";
    case "Continue":
      return "continue";
    case "TypeComment":
      // The emitter renders TypeComment as its own `/_ type ... _/`
      // lines; suppress the wrapper-comment-above-stmt that emitBody
      // would otherwise add for every IR stmt.
      return null;
    case "MemberStore": {
      const lhs = [s.base.name, ...s.fieldPath].join(".");
      return `${lhs} = ${irExprToString(s.rhs)}`;
    }
    case "MultiAssignCall": {
      const slotsTxt = s.outputs
        .map(slot => (slot.binding === null ? "~" : slot.binding.name))
        .join(", ");
      const argsTxt = s.args.map(irExprToString).join(", ");
      return `[${slotsTxt}] = ${s.name}(${argsTxt})`;
    }
    case "IndexStore": {
      const lhs =
        s.fieldPath !== undefined
          ? [s.base.name, ...s.fieldPath].join(".")
          : s.base.name;
      return `${lhs}(${s.indices.map(irExprToString).join(", ")}) = ${irExprToString(s.rhs)}`;
    }
    case "CellIndexStore": {
      return `${s.base.name}{${s.indices.map(irExprToString).join(", ")}} = ${irExprToString(s.rhs)}`;
    }
    case "IndexSliceStore": {
      const lhs =
        s.fieldPath !== undefined
          ? [s.base.name, ...s.fieldPath].join(".")
          : s.base.name;
      const slotStrs = s.index.map(slot => {
        if (slot.kind === "Colon") return ":";
        if (slot.kind === "Scalar") return irExprToString(slot.expr);
        if (slot.kind === "IndexVec") return irExprToString(slot.expr);
        if (slot.kind === "LogicalMask") return irExprToString(slot.expr);
        const stepPart =
          slot.step.kind === "NumLit" && slot.step.value === 1
            ? ""
            : `${irExprToString(slot.step)}:`;
        return `${irExprToString(slot.start)}:${stepPart}${irExprToString(slot.end)}`;
      });
      return `${lhs}(${slotStrs.join(", ")}) = ${irExprToString(s.rhs)}`;
    }
  }
}

/** Multi-line C block comment for a function specialization, listing
 *  name, mangled C identifier, per-parameter types, and output types. */
export function irFuncDocComment(fn: IRFunc): string {
  const lines: string[] = [];
  lines.push("/**");
  lines.push(` * ${fn.name} (specialized as \`${fn.cName}\`)`);
  if (fn.params.length === 0) {
    lines.push(" *");
    lines.push(" * params: (none)");
  } else {
    lines.push(" *");
    lines.push(" * params:");
    for (let i = 0; i < fn.params.length; i++) {
      lines.push(` *   ${fn.params[i]} :: ${typeToString(fn.paramTypes[i])}`);
    }
  }
  if (fn.outputs.length > 0) {
    lines.push(" *");
    lines.push(" * returns:");
    for (let i = 0; i < fn.outputs.length; i++) {
      const ty = fn.outputTypes[i];
      const tyStr = ty ? typeToString(ty) : "unknown";
      lines.push(` *   ${fn.outputs[i]} :: ${tyStr}`);
    }
  }
  lines.push(" */");
  return lines.join("\n");
}

function prettyStmtLines(s: IRStmt, indent: string): string[] {
  const step = "  ";
  switch (s.kind) {
    case "If": {
      const lines: string[] = [`${indent}if ${irExprToString(s.cond)}`];
      for (const t of s.thenBody)
        lines.push(...prettyStmtLines(t, indent + step));
      if (s.elseBody.length > 0) {
        lines.push(`${indent}else`);
        for (const e of s.elseBody)
          lines.push(...prettyStmtLines(e, indent + step));
      }
      lines.push(`${indent}end`);
      return lines;
    }
    case "While": {
      const lines: string[] = [`${indent}while ${irExprToString(s.cond)}`];
      for (const b of s.body) lines.push(...prettyStmtLines(b, indent + step));
      lines.push(`${indent}end`);
      return lines;
    }
    case "For": {
      const stepPart = s.step === 1 ? "" : `${numLitText(s.step)}:`;
      const lines: string[] = [
        `${indent}for ${s.varName} = ${irExprToString(s.start)}:${stepPart}${irExprToString(s.end)}`,
      ];
      for (const b of s.body) lines.push(...prettyStmtLines(b, indent + step));
      lines.push(`${indent}end`);
      return lines;
    }
    case "TypeComment": {
      return s.entries.map(
        e => `${indent}% type ${e.name} :: ${typeToString(e.ty)}`
      );
    }
    default: {
      const header = irStmtHeader(s);
      return header === null ? [] : [`${indent}${header}`];
    }
  }
}

/** Render the whole lowered IR as a numbl-like script with one block
 *  per function specialization plus a trailing top-level body. Used by
 *  the IDE's Internals tab in interpreter mode, where no codegen
 *  artifact exists. */
export function prettyIRProgram(prog: IRProgram): string {
  const blocks: string[] = [];
  for (const fn of prog.functions.values()) {
    const sigLhs =
      fn.outputs.length === 0
        ? ""
        : fn.outputs.length === 1
          ? `${fn.outputs[0]} = `
          : `[${fn.outputs.join(", ")}] = `;
    const lines: string[] = [];
    lines.push(irFuncDocComment(fn));
    lines.push(`function ${sigLhs}${fn.name}(${fn.params.join(", ")})`);
    for (const s of fn.body) lines.push(...prettyStmtLines(s, "  "));
    lines.push("end");
    blocks.push(lines.join("\n"));
  }
  if (prog.topLevelStmts.length > 0) {
    const lines: string[] = ["% --- top level ---"];
    for (const s of prog.topLevelStmts) lines.push(...prettyStmtLines(s, ""));
    blocks.push(lines.join("\n"));
  }
  return blocks.join("\n\n");
}
