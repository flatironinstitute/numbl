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
let _fileName: string | undefined;

/**
 * Hoisted aliases for a tensor parameter that's read (and optionally
 * written) in the loop body. Each entry maps the original variable name
 * to the local JS identifiers for its `.data`, `.data.length`, `.shape[0]`,
 * and `.shape[1]` (the latter two are only emitted if needed by the
 * dimensionality of the index ops in the body).
 *
 * If `isWriteTarget` is true the hoist was generated via a call to
 * `$h.unshare(t)` at function entry, so the per-iter store goes through
 * the hoisted `.data` alias safely (no risk of mutating shared state).
 */
interface HoistedAlias {
  data: string;
  len: string;
  d0: string;
  d1: string;
  isWriteTarget: boolean;
}

let _hoistedAliases: Map<string, HoistedAlias> = new Map();

export function generateJS(
  body: JitStmt[],
  params: string[],
  outputs: string[],
  nargout: number,
  localVars: Set<string>,
  fileName?: string,
  paramTypes?: JitType[]
): string {
  _tmpCounter = 0;
  _fileName = fileName;

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

  // Hoist loop-invariant tensor reads (and, for write-target tensors,
  // unshare once up front so the per-write body can store directly
  // through the hoisted `.data` alias without a COW check per iter).
  //
  // A param qualifies for the *read-only* hoist path if it's a real tensor
  // that the body never writes to (not in `outputs` or `localVars`).
  //
  // A param qualifies for the *write-target* hoist path if it's a real
  // tensor that IS in `outputs` (i.e. the body contains an `AssignIndex`
  // targeting it) but isn't a plain reassignment (plain reassignments show
  // up in `localVars` because `lowerAssign` adds them). Write-targets
  // reassign the local parameter to the result of `$h.unshare(param)`,
  // which is a no-op fast return when `_rc <= 1`.
  //
  // We re-walk the body to figure out which dim sizes and index arities
  // the body actually uses, so we don't emit unused aliases or unneeded
  // unshare calls.
  const outputSet = new Set(outputs);
  const localSet = localVars;
  _hoistedAliases = new Map();
  if (paramTypes && paramTypes.length === params.length) {
    const usedDims = collectMaxIndexDimsByVar(body);
    const writeTargets = collectAssignIndexBases(body);
    for (let i = 0; i < params.length; i++) {
      const name = params[i];
      const type = paramTypes[i];
      if (type.kind !== "tensor" || type.isComplex !== false) continue;

      // Plain reassignment inside the body (e.g. `t = t + 1`) — don't
      // hoist, the local would be stale after the first iteration.
      if (localSet.has(name)) continue;

      const isWriteTarget = writeTargets.has(name);
      // Output set without a write-target marker means the name shows
      // up in outputs because the analyzer flagged it, but the body
      // doesn't actually have a scalar indexed assign on it. Don't
      // hoist that — something unusual is going on.
      if (outputSet.has(name) && !isWriteTarget) continue;

      // For the read-only path, we need at least one Index read to
      // justify the hoist. Write-targets always get hoisted (even if
      // the only use is the write itself) so the store goes through
      // the hoisted alias.
      const maxRead = usedDims.get(name) ?? 0;
      const maxWrite = writeTargets.get(name) ?? 0;
      const maxDim = Math.max(maxRead, maxWrite);
      if (maxDim === 0) continue;

      const m = mangle(name);
      const dataAlias = `$${m}_data`;
      const lenAlias = `$${m}_len`;
      const d0Alias = `$${m}_d0`;
      const d1Alias = `$${m}_d1`;

      if (isWriteTarget) {
        // Unshare reassigns the param local to the un-COW'd tensor, so
        // subsequent `.data` reads (and the writeback) see the fresh
        // (or unchanged) copy we're safe to mutate.
        lines.push(`${indent}${m} = $h.unshare(${m});`);
      }

      const decls: string[] = [];
      decls.push(`${dataAlias} = ${m}.data`);
      decls.push(`${lenAlias} = ${dataAlias}.length`);
      if (maxDim >= 2) {
        decls.push(`${d0Alias} = ${m}.shape[0]`);
      }
      if (maxDim >= 3) {
        decls.push(`${d1Alias} = ${m}.shape[1]`);
      }
      lines.push(`${indent}var ${decls.join(", ")};`);
      _hoistedAliases.set(name, {
        data: dataAlias,
        len: lenAlias,
        d0: d0Alias,
        d1: d1Alias,
        isWriteTarget,
      });
    }
  }

  // Emit body
  emitStmts(lines, body, indent);

  // Return
  lines.push(`${indent}return ${_returnExpr};`);

  return lines.join("\n");
}

/**
 * Walk the JitStmt body and find, for each variable name used as the base
 * of an Index expression, the maximum number of indices it's accessed
 * with. Used by generateJS to decide how many shape dimensions to hoist
 * for each tensor parameter.
 */
function collectMaxIndexDimsByVar(body: JitStmt[]): Map<string, number> {
  const out = new Map<string, number>();
  const visit = (e: JitExpr): void => {
    switch (e.tag) {
      case "Index":
        if (e.base.tag === "Var") {
          const cur = out.get(e.base.name) ?? 0;
          if (e.indices.length > cur) out.set(e.base.name, e.indices.length);
        }
        visit(e.base);
        for (const idx of e.indices) visit(idx);
        return;
      case "Binary":
        visit(e.left);
        visit(e.right);
        return;
      case "Unary":
        visit(e.operand);
        return;
      case "Call":
      case "UserCall":
        for (const a of e.args) visit(a);
        return;
      case "TensorLiteral":
        for (const row of e.rows) for (const c of row) visit(c);
        return;
      default:
        return;
    }
  };
  const visitStmts = (stmts: JitStmt[]): void => {
    for (const s of stmts) {
      switch (s.tag) {
        case "Assign":
        case "ExprStmt":
          visit(s.expr);
          break;
        case "AssignIndex":
          // The base name doesn't contribute to the "max read dims"
          // count directly (there's no Index expr around it), but its
          // index expressions do need walking in case they themselves
          // contain Index reads.
          for (const idx of s.indices) visit(idx);
          visit(s.value);
          break;
        case "MultiAssign":
          for (const a of s.args) visit(a);
          break;
        case "If":
          visit(s.cond);
          visitStmts(s.thenBody);
          for (const eib of s.elseifBlocks) {
            visit(eib.cond);
            visitStmts(eib.body);
          }
          if (s.elseBody) visitStmts(s.elseBody);
          break;
        case "For":
          visit(s.start);
          if (s.step) visit(s.step);
          visit(s.end);
          visitStmts(s.body);
          break;
        case "While":
          visit(s.cond);
          visitStmts(s.body);
          break;
        default:
          break;
      }
    }
  };
  visitStmts(body);
  return out;
}

/**
 * Walk the JitStmt body and collect, for each tensor variable that
 * appears as the base of an `AssignIndex` statement, the max number of
 * indices used. Returns a map baseName → maxDim. Used by `generateJS`
 * to decide which params need the unshare-and-hoist write path.
 */
function collectAssignIndexBases(body: JitStmt[]): Map<string, number> {
  const out = new Map<string, number>();
  const visit = (stmts: JitStmt[]): void => {
    for (const s of stmts) {
      switch (s.tag) {
        case "AssignIndex": {
          const cur = out.get(s.baseName) ?? 0;
          if (s.indices.length > cur) out.set(s.baseName, s.indices.length);
          break;
        }
        case "If":
          visit(s.thenBody);
          for (const eib of s.elseifBlocks) visit(eib.body);
          if (s.elseBody) visit(s.elseBody);
          break;
        case "For":
        case "While":
          visit(s.body);
          break;
        default:
          break;
      }
    }
  };
  visit(body);
  return out;
}

// ── Statement emission ──────────────────────────────────────────────────

function emitStmts(lines: string[], stmts: JitStmt[], indent: string): void {
  for (const stmt of stmts) {
    emitStmt(lines, stmt, indent);
  }
}

function emitStmt(lines: string[], stmt: JitStmt, indent: string): void {
  switch (stmt.tag) {
    case "Assign":
      lines.push(`${indent}${mangle(stmt.name)} = ${emitExpr(stmt.expr)};`);
      break;

    case "AssignIndex":
      lines.push(`${indent}${emitAssignIndex(stmt)};`);
      break;

    case "ExprStmt":
      lines.push(`${indent}${emitExpr(stmt.expr)};`);
      break;

    case "If": {
      lines.push(`${indent}if (${emitTruthiness(stmt.cond)}) {`);
      emitStmts(lines, stmt.thenBody, indent + "  ");
      for (const eib of stmt.elseifBlocks) {
        lines.push(`${indent}} else if (${emitTruthiness(eib.cond)}) {`);
        emitStmts(lines, eib.body, indent + "  ");
      }
      if (stmt.elseBody) {
        lines.push(`${indent}} else {`);
        emitStmts(lines, stmt.elseBody, indent + "  ");
      }
      lines.push(`${indent}}`);
      break;
    }

    case "For": {
      const v = mangle(stmt.varName);
      const t = `$t${++_tmpCounter}`;
      const start = emitExpr(stmt.start);
      const end = emitExpr(stmt.end);
      const step = stmt.step ? emitExpr(stmt.step) : "1";
      // Use a separate temp loop variable and assign the iterator inside
      // the body. This is important for two reasons:
      // 1. The iterator variable must retain the last value actually used
      //    in the loop body (MATLAB semantics), not the incremented value
      //    that failed the loop condition.
      // 2. This pattern appears to be faster in V8 (reason unclear).
      if (stmt.step) {
        lines.push(
          `${indent}for (var ${t} = ${start}; ${step} !== 0 && (${step} > 0 ? ${t} <= ${end} : ${t} >= ${end}); ${t} += ${step}) {`
        );
      } else {
        lines.push(
          `${indent}for (var ${t} = ${start}; ${t} <= ${end}; ${t} += 1) {`
        );
      }
      lines.push(`${indent}  ${v} = ${t};`);
      emitStmts(lines, stmt.body, indent + "  ");
      lines.push(`${indent}}`);
      break;
    }

    case "While":
      lines.push(`${indent}while (${emitTruthiness(stmt.cond)}) {`);
      emitStmts(lines, stmt.body, indent + "  ");
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
      const args = stmt.args.map(a => emitExpr(a));
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

    case "SetLoc":
      if (_fileName) {
        lines.push(`${indent}$rt.$file = ${JSON.stringify(_fileName)};`);
      }
      lines.push(`${indent}$rt.$line = ${stmt.line};`);
      break;
  }
}

// ── Expression emission ─────────────────────────────────────────────────

function isComplexType(t: JitType): boolean {
  return (
    t.kind === "complex_or_number" ||
    (t.kind === "tensor" && t.isComplex === true)
  );
}

function emitExpr(expr: JitExpr): string {
  switch (expr.tag) {
    case "NumberLiteral":
      return String(expr.value);

    case "ImagLiteral":
      return `{kind:"complex_number",re:0,im:1}`;

    case "Var":
      return mangle(expr.name);

    case "Binary":
      return emitBinary(expr);

    case "Unary":
      return emitUnary(expr);

    case "Call":
      return emitCall(expr);

    case "TensorLiteral":
      return emitTensorLiteral(expr);

    case "StringLiteral":
      return JSON.stringify(expr.value);

    case "UserCall":
      return emitUserCall(expr);

    case "Index":
      return emitIndex(expr);
  }
}

function emitBinary(expr: JitExpr & { tag: "Binary" }): string {
  const left = emitExpr(expr.left);
  const right = emitExpr(expr.right);
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
    case BinaryOperation.Pow:
    case BinaryOperation.ElemPow:
      return `$h.tPow(${left}, ${right})`;
    case BinaryOperation.Equal:
      return `$h.tEq(${left}, ${right})`;
    case BinaryOperation.NotEqual:
      return `$h.tNeq(${left}, ${right})`;
    case BinaryOperation.Less:
      return `$h.tLt(${left}, ${right})`;
    case BinaryOperation.LessEqual:
      return `$h.tLe(${left}, ${right})`;
    case BinaryOperation.Greater:
      return `$h.tGt(${left}, ${right})`;
    case BinaryOperation.GreaterEqual:
      return `$h.tGe(${left}, ${right})`;
    default:
      throw new Error(`JIT codegen: unsupported tensor binary op ${op}`);
  }
}

function emitUnary(expr: JitExpr & { tag: "Unary" }): string {
  const operand = emitExpr(expr.operand);

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

function emitUserCall(expr: JitExpr & { tag: "UserCall" }): string {
  const args = expr.args.map(a => emitExpr(a));
  return `$h.callUser($rt, ${JSON.stringify(expr.name)}, ${expr.jitName}, ${args.join(", ")})`;
}

function emitTensorLiteral(expr: JitExpr & { tag: "TensorLiteral" }): string {
  const { rows, nRows, nCols } = expr;
  // Column-major order: iterate columns first, then rows
  const elems: string[] = [];
  for (let c = 0; c < nCols; c++) {
    for (let r = 0; r < nRows; r++) {
      elems.push(emitExpr(rows[r][c]));
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
          const s = emitExpr(e);
          reElems.push(`$h.re(${s})`);
          imElems.push(`$h.im(${s})`);
        } else {
          reElems.push(emitExpr(e));
          imElems.push("0");
        }
      }
    }
    return `$h.mkTensorC([${reElems.join(", ")}], [${imElems.join(", ")}], [${nRows}, ${nCols}])`;
  }
  return `$h.mkTensor([${elems.join(", ")}], [${nRows}, ${nCols}])`;
}

function emitIndex(expr: JitExpr & { tag: "Index" }): string {
  const base = emitExpr(expr.base);
  const baseType = expr.base.jitType;
  const indices = expr.indices.map(i => emitExpr(i));

  // Hoisted-base fast path: the loop generator hoisted this base's data,
  // length, and dim sizes to local aliases at function entry. This is the
  // fastest path because the per-call helper takes only scalar args (no
  // property loads on the tensor object).
  if (
    baseType.kind === "tensor" &&
    baseType.isComplex === false &&
    expr.base.tag === "Var"
  ) {
    const alias = _hoistedAliases.get(expr.base.name);
    if (alias) {
      if (indices.length === 1) {
        return `$h.idx1r_h(${alias.data}, ${alias.len}, ${indices[0]})`;
      }
      if (indices.length === 2) {
        return `$h.idx2r_h(${alias.data}, ${alias.len}, ${alias.d0}, ${indices[0]}, ${indices[1]})`;
      }
      if (indices.length === 3) {
        return `$h.idx3r_h(${alias.data}, ${alias.len}, ${alias.d0}, ${alias.d1}, ${indices[0]}, ${indices[1]}, ${indices[2]})`;
      }
    }
  }

  // Specialized fast path: real tensor with known type. The helpers skip
  // isTensor / imag / Math.round and avoid the per-call array allocation
  // that idxN otherwise needs.
  if (baseType.kind === "tensor" && baseType.isComplex === false) {
    if (indices.length === 1) return `$h.idx1r(${base}, ${indices[0]})`;
    if (indices.length === 2)
      return `$h.idx2r(${base}, ${indices[0]}, ${indices[1]})`;
    if (indices.length === 3)
      return `$h.idx3r(${base}, ${indices[0]}, ${indices[1]}, ${indices[2]})`;
  }

  if (indices.length === 1) return `$h.idx1(${base}, ${indices[0]})`;
  if (indices.length === 2)
    return `$h.idx2(${base}, ${indices[0]}, ${indices[1]})`;
  return `$h.idxN(${base}, [${indices.join(", ")}])`;
}

function emitAssignIndex(stmt: JitStmt & { tag: "AssignIndex" }): string {
  const alias = _hoistedAliases.get(stmt.baseName);
  const indices = stmt.indices.map(i => emitExpr(i));
  const value = emitExpr(stmt.value);

  // Write-target tensors are always hoisted at the top of the loop
  // function (see generateJS). If there's no alias here something is
  // wrong — lowering only emits AssignIndex for real-tensor write
  // targets, which the hoist logic always picks up.
  if (!alias) {
    throw new Error(
      `JIT codegen: AssignIndex on '${stmt.baseName}' without a hoisted alias`
    );
  }

  if (indices.length === 1) {
    return `$h.set1r_h(${alias.data}, ${alias.len}, ${indices[0]}, ${value})`;
  }
  if (indices.length === 2) {
    return `$h.set2r_h(${alias.data}, ${alias.len}, ${alias.d0}, ${indices[0]}, ${indices[1]}, ${value})`;
  }
  // 3D
  return `$h.set3r_h(${alias.data}, ${alias.len}, ${alias.d0}, ${alias.d1}, ${indices[0]}, ${indices[1]}, ${indices[2]}, ${value})`;
}

function emitCall(expr: JitExpr & { tag: "Call" }): string {
  const args = expr.args.map(a => emitExpr(a));
  // Try fast-path emission if the IBuiltin provides one
  const ib = getIBuiltin(expr.name);
  if (ib?.jitEmit) {
    const argTypes = expr.args.map(a => a.jitType);
    const fast = ib.jitEmit(args, argTypes);
    if (fast) return fast;
  }
  return `$h.ib_${expr.name}(${args.join(", ")})`;
}

// ── Truthiness / condition emission ──────────────────────────────────────
//
// emitTruthiness is called for the cond of `if` / `while` and the operands
// of `&&` / `||`. The default value-form codegen for comparisons emits
// `(a > b ? 1 : 0)` (so that "boolean" JIT values still print as 0/1
// numbers in tensor contexts). Wrapping that in `!== 0` for every if/while
// gives nested `((((a > b ? 1 : 0)) !== 0 && ...))` chains that obscure the
// expression V8 needs to inline. We recurse here so that comparison /
// logical sub-expressions emit directly as JS booleans inside conditions.

function emitTruthiness(expr: JitExpr): string {
  // String/char conditions are rejected during lowering.
  if (expr.jitType.kind === "complex_or_number") {
    return `$h.cTruthy(${emitExpr(expr)})`;
  }

  if (expr.tag === "Binary") {
    switch (expr.op) {
      case BinaryOperation.Equal:
        return `(${emitExpr(expr.left)}) === (${emitExpr(expr.right)})`;
      case BinaryOperation.NotEqual:
        return `(${emitExpr(expr.left)}) !== (${emitExpr(expr.right)})`;
      case BinaryOperation.Less:
        return `(${emitExpr(expr.left)}) < (${emitExpr(expr.right)})`;
      case BinaryOperation.LessEqual:
        return `(${emitExpr(expr.left)}) <= (${emitExpr(expr.right)})`;
      case BinaryOperation.Greater:
        return `(${emitExpr(expr.left)}) > (${emitExpr(expr.right)})`;
      case BinaryOperation.GreaterEqual:
        return `(${emitExpr(expr.left)}) >= (${emitExpr(expr.right)})`;
      case BinaryOperation.AndAnd:
        return `(${emitTruthiness(expr.left)}) && (${emitTruthiness(expr.right)})`;
      case BinaryOperation.OrOr:
        return `(${emitTruthiness(expr.left)}) || (${emitTruthiness(expr.right)})`;
      default:
        break; // fall through to value-form
    }
  }

  if (expr.tag === "Unary" && expr.op === UnaryOperation.Not) {
    return `!(${emitTruthiness(expr.operand)})`;
  }

  return `(${emitExpr(expr)}) !== 0`;
}
