/**
 * JIT IR -> JavaScript code generation.
 *
 * IR walkers for hoist-pass data collection are in jitCodegenHoist.ts.
 */

import { BinaryOperation, UnaryOperation } from "../../parser/types.js";
import {
  type JitExpr,
  type JitStmt,
  type JitType,
  isTensorType,
  isKnownInteger,
} from "./jitTypes.js";
import { getIBuiltin } from "../builtins/types.js";
import {
  type HoistedAlias,
  structFieldKey,
  structArrayElementsKey,
  collectTensorUsage,
  collectStructFieldReads,
  collectStructArrayElementReads,
} from "./jitCodegenHoist.js";

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
let _fileEmitted = false;

let _hoistedAliases: Map<string, HoistedAlias> = new Map();

/**
 * Hoisted scalar struct-field aliases: `(baseName, fieldName)` → local JS
 * identifier. At function entry we emit `var $<base>_<field> =
 * <base>.fields.get("<field>")` for each pair, and on use MemberRead
 * nodes emit the bare local identifier. Only scalar numeric fields are
 * hoisted — stage 12 doesn't support tensor-typed fields or chained
 * Member access (that's stage 13).
 */
let _hoistedStructFields: Map<string, string> = new Map();

/**
 * Hoisted struct-array element storage: `(structVarName,
 * structArrayFieldName)` → local JS identifier bound to
 * `<struct>.fields.get("<field>").elements` (the raw `RuntimeStruct[]`
 * array). At function entry we emit one `var $<struct>_<field>_elements
 * = ...` per unique pair, and on use `StructArrayMemberRead` nodes emit
 * `$<...>_elements[Math.round(i) - 1].fields.get("<leaf>")`. See stage
 * 13 lowering for the parser pattern this matches.
 */
let _hoistedStructArrayElements: Map<string, string> = new Map();

export function generateJS(
  body: JitStmt[],
  params: string[],
  outputs: string[],
  nargout: number,
  localVars: Set<string>,
  fileName?: string
): string {
  _tmpCounter = 0;
  _fileName = fileName;
  _fileEmitted = false;

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

  // Hoist real-tensor variables (params AND locals) that participate in
  // indexing. The per-tensor `.data` / `.length` / shape reads get lifted
  // to local aliases so the per-iter helper calls take only scalar args.
  //
  // We walk the JIT IR (rather than param types) so the pass naturally
  // covers four cases:
  //   * read-only param tensors          — entry-time hoist only
  //   * write-target param tensors       — entry-time hoist + unshare
  //   * tensor locals (created in body)  — declared at entry, filled by
  //                                        the per-Assign refresh path
  //   * params reassigned in the body    — entry-time hoist that's then
  //                                        refreshed by every assignment
  //
  // The per-Assign refresh (see emitHoistRefresh) re-reads `.data` etc.
  // from the (possibly new) tensor object after every plain `Assign` to a
  // hoisted name. This is what makes the chunkie grow-and-copy pattern
  // (`out_pt = zeros(N*2, 1); out_pt(1:N) = tmp_pt(1:N)`) JIT cleanly:
  // the post-`zeros` reassignment refreshes `$out_pt_data`, and the
  // subsequent slice write goes through the new buffer.
  _hoistedAliases = new Map();
  _hoistedStructFields = new Map();
  _hoistedStructArrayElements = new Map();
  const usage = collectTensorUsage(body);
  const paramSet = new Set(params);
  // Stable name order for deterministic codegen output across runs.
  const hoistNames = [...usage.keys()].sort();
  for (const name of hoistNames) {
    const u = usage.get(name)!;
    if (!u.isReal) continue;
    const isParam = paramSet.has(name);
    const isLocal = localVars.has(name);
    // A name that's neither a param nor a local shouldn't appear, but
    // skip defensively rather than emit a `var` collision.
    if (!isParam && !isLocal) continue;

    const maxDim = Math.max(u.maxReadDim, u.maxWriteDim);
    if (maxDim === 0) continue;

    const m = mangle(name);
    const dataAlias = `$${m}_data`;
    const lenAlias = `$${m}_len`;
    const d0Alias = `$${m}_d0`;
    const d1Alias = `$${m}_d1`;
    const isWriteTarget = u.maxWriteDim > 0;

    _hoistedAliases.set(name, {
      data: dataAlias,
      len: lenAlias,
      d0: d0Alias,
      d1: d1Alias,
      maxDim,
      isWriteTarget,
      isParam,
    });

    if (isParam) {
      // Initialize at function entry from the param value.
      if (isWriteTarget) {
        // Unshare reassigns the param local to the un-COW'd tensor so
        // the hoisted `.data` alias points at a buffer we own.
        lines.push(`${indent}${m} = $h.unshare(${m});`);
      }
      const decls: string[] = [];
      decls.push(`${dataAlias} = ${m}.data`);
      decls.push(`${lenAlias} = ${dataAlias}.length`);
      if (maxDim >= 2) decls.push(`${d0Alias} = ${m}.shape[0]`);
      if (maxDim >= 3) decls.push(`${d1Alias} = ${m}.shape[1]`);
      lines.push(`${indent}var ${decls.join(", ")};`);
    } else {
      // Local: declare uninitialized, the first plain Assign to this name
      // (anywhere in the body) will fill the alias via emitHoistRefresh.
      const decls: string[] = [dataAlias, lenAlias];
      if (maxDim >= 2) decls.push(d0Alias);
      if (maxDim >= 3) decls.push(d1Alias);
      lines.push(`${indent}var ${decls.join(", ")};`);
    }
  }

  // Stage 12: hoist scalar struct-field reads. Walk the IR to find every
  // `MemberRead` node, collect unique (baseName, fieldName) pairs, and
  // emit a per-pair `var $<base>_<field> = <base>.fields.get("<field>")`
  // at function entry. Later `emitExpr` case `MemberRead` substitutes
  // the alias for each use. The struct bases must be loop-invariant
  // (never reassigned inside the body); since the lowering only accepts
  // Member reads when the env type is still a struct, any post-assign
  // reads bail and we never hoist stale fields.
  const structFieldReads = collectStructFieldReads(body);
  const structFieldKeys = [...structFieldReads.keys()].sort();
  for (const key of structFieldKeys) {
    const { baseName, fieldName } = structFieldReads.get(key)!;
    const aliasName = `$${mangle(baseName)}_${fieldName}`;
    _hoistedStructFields.set(key, aliasName);
    lines.push(
      `${indent}var ${aliasName} = ${mangle(baseName)}.fields.get(${JSON.stringify(fieldName)});`
    );
  }

  // Stage 13: hoist struct-array element storage. For every unique
  // `(structVarName, structArrayFieldName)` pair found in a
  // `StructArrayMemberRead`, emit
  //   var $T_nodes_elements = T.fields.get("nodes").elements;
  // at function entry. Per-use reads pull the RuntimeStruct at index
  // `Math.round(i) - 1` and `.fields.get("leaf")` for the final field.
  // Like stage 12, we assume the struct base is loop-invariant: any
  // reassignment of `T` bails the whole lowering via the env type
  // check in the Member case.
  const structArrayReads = collectStructArrayElementReads(body);
  const structArrayKeys = [...structArrayReads.keys()].sort();
  for (const key of structArrayKeys) {
    const { structVarName, structArrayFieldName } = structArrayReads.get(key)!;
    const aliasName = `$${mangle(structVarName)}_${structArrayFieldName}_elements`;
    _hoistedStructArrayElements.set(key, aliasName);
    lines.push(
      `${indent}var ${aliasName} = ${mangle(structVarName)}.fields.get(${JSON.stringify(structArrayFieldName)}).elements;`
    );
  }

  // Emit body
  emitStmts(lines, body, indent);

  // Return
  lines.push(`${indent}return ${_returnExpr};`);

  return lines.join("\n");
}

// ── Statement emission ──────────────────────────────────────────────────

function emitStmts(lines: string[], stmts: JitStmt[], indent: string): void {
  for (const stmt of stmts) {
    emitStmt(lines, stmt, indent);
  }
}

function emitStmt(lines: string[], stmt: JitStmt, indent: string): void {
  switch (stmt.tag) {
    case "Assign": {
      // Var-to-var tensor assign: bump _rc on the aliased tensor so later
      // ops don't reuse its buffer in place (which would corrupt the other
      // binding). Non-tensor RHS skips this.
      if (stmt.expr.tag === "Var" && isTensorType(stmt.expr.jitType)) {
        lines.push(
          `${indent}${mangle(stmt.name)} = $h.shareTensor(${mangle(stmt.expr.name)});`
        );
      } else {
        // Pass the LHS name as a dest hint so the top-level RHS op (tensor
        // binary / unary / compare / builtin) can write into the previous
        // value's buffer when uniquely owned.
        lines.push(
          `${indent}${mangle(stmt.name)} = ${emitExpr(stmt.expr, stmt.name)};`
        );
      }
      // If `name` is a hoisted tensor variable, refresh its hoisted aliases
      // (`.data`, `.length`, shape) so subsequent reads/writes see the new
      // value. Without this, `out_pt = zeros(N*2, 1)` followed by
      // `out_pt(i) = v` would write to the OLD hoisted buffer.
      emitHoistRefresh(lines, stmt.name, indent);
      break;
    }

    case "AssignIndex":
      lines.push(`${indent}${emitAssignIndex(stmt)};`);
      break;

    case "AssignIndexRange":
      lines.push(`${indent}${emitAssignIndexRange(stmt)};`);
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
      lines.push(`${indent}  $rt.checkCancel();`);
      lines.push(`${indent}  ${v} = ${t};`);
      emitStmts(lines, stmt.body, indent + "  ");
      lines.push(`${indent}}`);
      break;
    }

    case "While":
      lines.push(`${indent}while (${emitTruthiness(stmt.cond)}) {`);
      lines.push(`${indent}  $rt.checkCancel();`);
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
      if (_fileName && !_fileEmitted) {
        lines.push(`${indent}$rt.$file = ${JSON.stringify(_fileName)};`);
        _fileEmitted = true;
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

// `destName` is non-undefined only when this expression is the top-level
// RHS of a plain Assign. In that case the emitter may pass the mangled
// dest local to tensor op helpers, which will reuse its `.data` buffer if
// the previous value is a rc==1 Float64 tensor of matching length.
// Sub-expressions get `undefined` — no reuse attempt for inner temps yet.
function emitExpr(expr: JitExpr, destName?: string): string {
  switch (expr.tag) {
    case "NumberLiteral":
      return String(expr.value);

    case "ImagLiteral":
      return `{kind:"complex_number",re:0,im:1}`;

    case "Var":
      return mangle(expr.name);

    case "Binary":
      return emitBinary(expr, destName);

    case "Unary":
      return emitUnary(expr, destName);

    case "Call":
      return emitCall(expr, destName);

    case "TensorLiteral":
      return emitTensorLiteral(expr);

    case "VConcatGrow":
      return `$h.vconcatGrow1r(${emitExpr(expr.base)}, ${emitExpr(expr.value)})`;

    case "MemberRead": {
      const key = structFieldKey(expr.baseName, expr.fieldName);
      const alias = _hoistedStructFields.get(key);
      if (alias) return alias;
      // Fallback: the hoist pass should have registered every MemberRead
      // (collectStructFieldReads walks the same IR). This branch only
      // fires if a future code path synthesizes a MemberRead after the
      // hoist walk completes — emit the Map lookup directly.
      return `${mangle(expr.baseName)}.fields.get(${JSON.stringify(expr.fieldName)})`;
    }

    case "StructArrayMemberRead": {
      const key = structArrayElementsKey(
        expr.structVarName,
        expr.structArrayFieldName
      );
      const elementsAlias =
        _hoistedStructArrayElements.get(key) ??
        `${mangle(expr.structVarName)}.fields.get(${JSON.stringify(expr.structArrayFieldName)}).elements`;
      const idxCode = emitExpr(expr.indexExpr);
      // Match MATLAB indexing semantics: Math.round then subtract 1
      // for 0-based JS array access. Same rounding strategy used by
      // the tensor index helpers.
      const raw = `${elementsAlias}[Math.round(${idxCode}) - 1].fields.get(${JSON.stringify(expr.leafFieldName)})`;
      // If the leaf type is a tensor, the field might hold a bare
      // scalar number at runtime (a chunkie quirk — leaf nodes with a
      // single point store `xi = 87` instead of a 1x1 tensor). Wrap
      // in asTensor so downstream tensor-read helpers always see a
      // real RuntimeTensor.
      if (expr.jitType.kind === "tensor") {
        return `$h.asTensor(${raw})`;
      }
      return raw;
    }

    case "StringLiteral":
      return JSON.stringify(expr.value);

    case "UserCall":
      return emitUserCall(expr);

    case "FuncHandleCall":
      return emitFuncHandleCall(expr);

    case "Index":
      return emitIndex(expr);
  }
}

function emitBinary(
  expr: JitExpr & { tag: "Binary" },
  destName?: string
): string {
  const left = emitExpr(expr.left);
  const right = emitExpr(expr.right);
  const leftIsTensor = isTensorType(expr.left.jitType);
  const rightIsTensor = isTensorType(expr.right.jitType);
  const anyComplex =
    isComplexType(expr.left.jitType) || isComplexType(expr.right.jitType);

  // Tensor operations use helpers (handles both real and complex tensors)
  if (leftIsTensor || rightIsTensor) {
    return emitTensorBinary(expr.op, left, right, destName);
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
  right: string,
  destName?: string
): string {
  const dest = destName !== undefined ? mangle(destName) : "undefined";
  switch (op) {
    case BinaryOperation.Add:
      return `$h.tAdd(${dest}, ${left}, ${right})`;
    case BinaryOperation.Sub:
      return `$h.tSub(${dest}, ${left}, ${right})`;
    case BinaryOperation.Mul:
    case BinaryOperation.ElemMul:
      return `$h.tMul(${dest}, ${left}, ${right})`;
    case BinaryOperation.Div:
    case BinaryOperation.ElemDiv:
      return `$h.tDiv(${dest}, ${left}, ${right})`;
    case BinaryOperation.Pow:
    case BinaryOperation.ElemPow:
      return `$h.tPow(${dest}, ${left}, ${right})`;
    case BinaryOperation.Equal:
      return `$h.tEq(${dest}, ${left}, ${right})`;
    case BinaryOperation.NotEqual:
      return `$h.tNeq(${dest}, ${left}, ${right})`;
    case BinaryOperation.Less:
      return `$h.tLt(${dest}, ${left}, ${right})`;
    case BinaryOperation.LessEqual:
      return `$h.tLe(${dest}, ${left}, ${right})`;
    case BinaryOperation.Greater:
      return `$h.tGt(${dest}, ${left}, ${right})`;
    case BinaryOperation.GreaterEqual:
      return `$h.tGe(${dest}, ${left}, ${right})`;
    default:
      throw new Error(`JIT codegen: unsupported tensor binary op ${op}`);
  }
}

function emitUnary(
  expr: JitExpr & { tag: "Unary" },
  destName?: string
): string {
  const operand = emitExpr(expr.operand);

  if (isTensorType(expr.operand.jitType)) {
    const dest = destName !== undefined ? mangle(destName) : "undefined";
    switch (expr.op) {
      case UnaryOperation.Minus:
        return `$h.tNeg(${dest}, ${operand})`;
      case UnaryOperation.Plus:
        return operand;
      case UnaryOperation.Transpose:
        return `$h.tCTranspose(${operand})`;
      case UnaryOperation.NonConjugateTranspose:
        return `$h.tTranspose(${operand})`;
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
    case UnaryOperation.Transpose:
    case UnaryOperation.NonConjugateTranspose:
      // Scalar transpose is identity
      return operand;
    default:
      throw new Error(`JIT codegen: unsupported scalar unary op ${expr.op}`);
  }
}

function emitUserCall(expr: JitExpr & { tag: "UserCall" }): string {
  const args = expr.args.map(a => emitExpr(a));
  return `$h.callUser($rt, ${JSON.stringify(expr.name)}, ${expr.jitName}, ${args.join(", ")})`;
}

function emitFuncHandleCall(expr: JitExpr & { tag: "FuncHandleCall" }): string {
  const args = expr.args.map(a => emitExpr(a));
  const expectedType = JSON.stringify(expr.jitType.kind);
  return `$h.callFuncHandle($rt, ${expr.name}, ${expectedType}, ${args.join(", ")})`;
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

  // For specialized helpers that use |0 internally, pre-round indices that
  // are not provably integer so that fractional values behave like
  // Math.round (matching the interpreter and the generic idx helpers).
  const ri = (i: number): string => {
    if (isKnownInteger(expr.indices[i].jitType)) return indices[i];
    return `Math.round(${indices[i]})`;
  };

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
        return `$h.idx1r_h(${alias.data}, ${alias.len}, ${ri(0)})`;
      }
      if (indices.length === 2) {
        return `$h.idx2r_h(${alias.data}, ${alias.len}, ${alias.d0}, ${ri(0)}, ${ri(1)})`;
      }
      if (indices.length === 3) {
        return `$h.idx3r_h(${alias.data}, ${alias.len}, ${alias.d0}, ${alias.d1}, ${ri(0)}, ${ri(1)}, ${ri(2)})`;
      }
    }
  }

  // Specialized fast path: real tensor with known type. The helpers skip
  // isTensor / imag / Math.round and avoid the per-call array allocation
  // that idxN otherwise needs.
  if (baseType.kind === "tensor" && baseType.isComplex === false) {
    if (indices.length === 1) return `$h.idx1r(${base}, ${ri(0)})`;
    if (indices.length === 2) return `$h.idx2r(${base}, ${ri(0)}, ${ri(1)})`;
    if (indices.length === 3)
      return `$h.idx3r(${base}, ${ri(0)}, ${ri(1)}, ${ri(2)})`;
  }

  // Generic helpers already use Math.round internally — no wrapping needed.
  if (indices.length === 1) return `$h.idx1(${base}, ${indices[0]})`;
  if (indices.length === 2)
    return `$h.idx2(${base}, ${indices[0]}, ${indices[1]})`;
  return `$h.idxN(${base}, [${indices.join(", ")}])`;
}

/**
 * After a plain `Assign` to a hoisted tensor variable, re-read its `.data`
 * and shape into the hoisted aliases. Called from emitStmt for the
 * `Assign` case (and only does work if the name has a hoisted alias).
 *
 * For write-target tensors, the refresh also calls `$h.unshare(name)` to
 * detach from any sharing the new RHS may have introduced (e.g. via
 * `tmp = base; ...; base(i) = v`). For fresh-from-`zeros(...)` tensors
 * unshare is a no-op fast return on `_rc <= 1`.
 */
function emitHoistRefresh(lines: string[], name: string, indent: string): void {
  const alias = _hoistedAliases.get(name);
  if (!alias) return;
  const m = mangle(name);
  if (alias.isWriteTarget) {
    lines.push(`${indent}${m} = $h.unshare(${m});`);
  }
  lines.push(`${indent}${alias.data} = ${m}.data;`);
  lines.push(`${indent}${alias.len} = ${alias.data}.length;`);
  if (alias.maxDim >= 2) {
    lines.push(`${indent}${alias.d0} = ${m}.shape[0];`);
  }
  if (alias.maxDim >= 3) {
    lines.push(`${indent}${alias.d1} = ${m}.shape[1];`);
  }
}

function emitAssignIndexRange(
  stmt: JitStmt & { tag: "AssignIndexRange" }
): string {
  const dstAlias = _hoistedAliases.get(stmt.baseName);
  const srcAlias = _hoistedAliases.get(stmt.srcBaseName);
  if (!dstAlias) {
    throw new Error(
      `JIT codegen: AssignIndexRange dst '${stmt.baseName}' without a hoisted alias`
    );
  }
  if (!srcAlias) {
    throw new Error(
      `JIT codegen: AssignIndexRange src '${stmt.srcBaseName}' without a hoisted alias`
    );
  }
  const emitRI = (e: JitExpr): string => {
    const code = emitExpr(e);
    if (isKnownInteger(e.jitType)) return code;
    return `Math.round(${code})`;
  };
  const dstStart = emitRI(stmt.dstStart);
  const dstEnd = emitRI(stmt.dstEnd);
  // stage 9: when srcStart/srcEnd are null, the source is used in its
  // entirety — substitute `1` and the source's hoisted length alias. The
  // same helper handles both forms since the check is length-based.
  const srcStart = stmt.srcStart !== null ? emitRI(stmt.srcStart) : "1";
  const srcEnd = stmt.srcEnd !== null ? emitRI(stmt.srcEnd) : srcAlias.len;
  return `$h.setRange1r_h(${dstAlias.data}, ${dstAlias.len}, ${dstStart}, ${dstEnd}, ${srcAlias.data}, ${srcAlias.len}, ${srcStart}, ${srcEnd})`;
}

function emitAssignIndex(stmt: JitStmt & { tag: "AssignIndex" }): string {
  const alias = _hoistedAliases.get(stmt.baseName);
  const indices = stmt.indices.map(i => emitExpr(i));
  const value = emitExpr(stmt.value);

  const ri = (i: number): string => {
    if (isKnownInteger(stmt.indices[i].jitType)) return indices[i];
    return `Math.round(${indices[i]})`;
  };

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
    return `$h.set1r_h(${alias.data}, ${alias.len}, ${ri(0)}, ${value})`;
  }
  if (indices.length === 2) {
    return `$h.set2r_h(${alias.data}, ${alias.len}, ${alias.d0}, ${ri(0)}, ${ri(1)}, ${value})`;
  }
  // 3D
  return `$h.set3r_h(${alias.data}, ${alias.len}, ${alias.d0}, ${alias.d1}, ${ri(0)}, ${ri(1)}, ${ri(2)}, ${value})`;
}

function emitCall(expr: JitExpr & { tag: "Call" }, destName?: string): string {
  const args = expr.args.map(a => emitExpr(a));
  // Internal helper calls (prefixed with __) go directly to $h
  if (expr.name.startsWith("__")) {
    return `$h.${expr.name}(${args.join(", ")})`;
  }
  // Try fast-path emission if the IBuiltin provides one. destName (mangled
  // LHS local) is passed so tensor-producing builtins can thread it into
  // their output-buffer-reuse slot.
  const ib = getIBuiltin(expr.name);
  if (ib?.jitEmit) {
    const argTypes = expr.args.map(a => a.jitType);
    const dest = destName !== undefined ? mangle(destName) : undefined;
    const fast = ib.jitEmit(args, argTypes, dest);
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
