/**
 * JIT IR -> JavaScript code generation.
 */

import { BinaryOperation, UnaryOperation } from "../../parser/types.js";
import {
  type JitExpr,
  type JitStmt,
  type JitType,
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
let _fileEmitted = false;

/**
 * Hoisted aliases for a tensor variable that's read (and optionally
 * written) in the loop body. Each entry maps the original variable name
 * to the local JS identifiers for its `.data`, `.data.length`, `.shape[0]`,
 * and `.shape[1]` (the latter two are only emitted if needed by the
 * dimensionality of the index ops in the body).
 *
 * If `isWriteTarget` is true the hoist sequence calls `$h.unshare(t)`
 * before reading `.data`, so the per-iter store goes through the hoisted
 * `.data` alias safely (no risk of mutating shared state).
 *
 * `maxDim` records the largest indexing arity used on this variable in
 * the body, which determines whether `.shape[0]` (≥ 2D) and `.shape[1]`
 * (3D) need to be hoisted. `isParam` distinguishes the entry-time
 * initialization (params start with a value) from locals which only get
 * their alias filled by the per-Assign refresh path.
 */
interface HoistedAlias {
  data: string;
  len: string;
  d0: string;
  d1: string;
  maxDim: number;
  isWriteTarget: boolean;
  isParam: boolean;
}

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

function structFieldKey(baseName: string, fieldName: string): string {
  return `${baseName}.${fieldName}`;
}

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

function structArrayElementsKey(
  structVarName: string,
  structArrayFieldName: string
): string {
  return `${structVarName}.${structArrayFieldName}`;
}

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

/**
 * Per-variable indexing usage collected by walking the JIT IR. Used by
 * `generateJS` to decide which tensors to hoist and how many shape
 * dimensions each one needs.
 *
 * `isReal` is true only if every Var/AssignIndex/AssignIndexRange
 * occurrence of this name carries a real (non-complex) tensor type.
 * If any single use is complex, the whole name is excluded from
 * hoisting (the codegen falls back to the per-call generic helpers).
 */
interface TensorUsage {
  maxReadDim: number;
  maxWriteDim: number;
  isReal: boolean;
}

/**
 * Walk the JIT IR collecting, for every variable that appears as the base
 * of an `Index` read, an `AssignIndex` write, or an `AssignIndexRange`
 * write, the maximum indexing arity used and whether all uses are on a
 * real tensor. The hoist pass uses this to decide which names get a
 * hoisted alias and how many shape dims it needs.
 */
function collectTensorUsage(body: JitStmt[]): Map<string, TensorUsage> {
  const out = new Map<string, TensorUsage>();
  const bump = (
    name: string,
    isRead: boolean,
    dim: number,
    isReal: boolean
  ): void => {
    let u = out.get(name);
    if (!u) {
      u = { maxReadDim: 0, maxWriteDim: 0, isReal: true };
      out.set(name, u);
    }
    if (!isReal) u.isReal = false;
    if (isRead) {
      if (dim > u.maxReadDim) u.maxReadDim = dim;
    } else {
      if (dim > u.maxWriteDim) u.maxWriteDim = dim;
    }
  };

  const visitExpr = (e: JitExpr): void => {
    switch (e.tag) {
      case "Index":
        if (e.base.tag === "Var" && e.base.jitType.kind === "tensor") {
          const real = e.base.jitType.isComplex === false;
          bump(e.base.name, true, e.indices.length, real);
        }
        visitExpr(e.base);
        for (const idx of e.indices) visitExpr(idx);
        return;
      case "Binary":
        visitExpr(e.left);
        visitExpr(e.right);
        return;
      case "Unary":
        visitExpr(e.operand);
        return;
      case "Call":
      case "UserCall":
        for (const a of e.args) visitExpr(a);
        return;
      case "TensorLiteral":
        for (const row of e.rows) for (const c of row) visitExpr(c);
        return;
      case "VConcatGrow":
        visitExpr(e.base);
        visitExpr(e.value);
        return;
      case "MemberRead":
        // No tensor-usage contribution; the struct base is not a tensor
        // index target. The struct-field hoisting is collected by a
        // separate walker (`collectStructFieldReads`).
        return;
      case "StructArrayMemberRead":
        // The struct base is not a tensor either. Recurse into the
        // index expression so that any tensor Var used inside (e.g.
        // `T.nodes(someTensor(k)).chld`) still gets its hoist.
        visitExpr(e.indexExpr);
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
          visitExpr(s.expr);
          break;
        case "AssignIndex": {
          if (s.baseType.kind === "tensor") {
            bump(
              s.baseName,
              false,
              s.indices.length,
              s.baseType.isComplex === false
            );
          }
          for (const idx of s.indices) visitExpr(idx);
          visitExpr(s.value);
          break;
        }
        case "AssignIndexRange": {
          if (s.baseType.kind === "tensor") {
            bump(s.baseName, false, 1, s.baseType.isComplex === false);
          }
          if (s.srcType.kind === "tensor") {
            bump(s.srcBaseName, true, 1, s.srcType.isComplex === false);
          }
          visitExpr(s.dstStart);
          visitExpr(s.dstEnd);
          // stage 9: srcStart/srcEnd are null for whole-tensor RHS — the
          // source's hoisted length alias is used instead at codegen time.
          if (s.srcStart) visitExpr(s.srcStart);
          if (s.srcEnd) visitExpr(s.srcEnd);
          break;
        }
        case "MultiAssign":
          for (const a of s.args) visitExpr(a);
          break;
        case "If":
          visitExpr(s.cond);
          visitStmts(s.thenBody);
          for (const eib of s.elseifBlocks) {
            visitExpr(eib.cond);
            visitStmts(eib.body);
          }
          if (s.elseBody) visitStmts(s.elseBody);
          break;
        case "For":
          visitExpr(s.start);
          if (s.step) visitExpr(s.step);
          visitExpr(s.end);
          visitStmts(s.body);
          break;
        case "While":
          visitExpr(s.cond);
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
 * Walk the JIT IR collecting all unique `(baseName, fieldName)` pairs
 * referenced by `MemberRead` nodes. The codegen hoists each pair as a
 * local alias at function entry. Keys are `"<baseName>.<fieldName>"`;
 * values carry the component parts so emission can use them directly.
 */
function collectStructFieldReads(
  body: JitStmt[]
): Map<string, { baseName: string; fieldName: string }> {
  const out = new Map<string, { baseName: string; fieldName: string }>();

  const visitExpr = (e: JitExpr): void => {
    switch (e.tag) {
      case "MemberRead": {
        const key = structFieldKey(e.baseName, e.fieldName);
        if (!out.has(key)) {
          out.set(key, { baseName: e.baseName, fieldName: e.fieldName });
        }
        return;
      }
      case "Binary":
        visitExpr(e.left);
        visitExpr(e.right);
        return;
      case "Unary":
        visitExpr(e.operand);
        return;
      case "Call":
      case "UserCall":
        for (const a of e.args) visitExpr(a);
        return;
      case "Index":
        visitExpr(e.base);
        for (const idx of e.indices) visitExpr(idx);
        return;
      case "TensorLiteral":
        for (const row of e.rows) for (const c of row) visitExpr(c);
        return;
      case "VConcatGrow":
        visitExpr(e.base);
        visitExpr(e.value);
        return;
      case "StructArrayMemberRead":
        visitExpr(e.indexExpr);
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
          visitExpr(s.expr);
          break;
        case "AssignIndex":
          for (const idx of s.indices) visitExpr(idx);
          visitExpr(s.value);
          break;
        case "AssignIndexRange":
          visitExpr(s.dstStart);
          visitExpr(s.dstEnd);
          if (s.srcStart) visitExpr(s.srcStart);
          if (s.srcEnd) visitExpr(s.srcEnd);
          break;
        case "MultiAssign":
          for (const a of s.args) visitExpr(a);
          break;
        case "If":
          visitExpr(s.cond);
          visitStmts(s.thenBody);
          for (const eib of s.elseifBlocks) {
            visitExpr(eib.cond);
            visitStmts(eib.body);
          }
          if (s.elseBody) visitStmts(s.elseBody);
          break;
        case "For":
          visitExpr(s.start);
          if (s.step) visitExpr(s.step);
          visitExpr(s.end);
          visitStmts(s.body);
          break;
        case "While":
          visitExpr(s.cond);
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
 * Walk the JIT IR collecting all unique `(structVarName,
 * structArrayFieldName)` pairs referenced by `StructArrayMemberRead`
 * nodes. The codegen hoists each pair as a local alias at function
 * entry bound to the underlying `RuntimeStruct[]` element array. Keys
 * are `"<structVarName>.<structArrayFieldName>"`.
 */
function collectStructArrayElementReads(
  body: JitStmt[]
): Map<string, { structVarName: string; structArrayFieldName: string }> {
  const out = new Map<
    string,
    { structVarName: string; structArrayFieldName: string }
  >();

  const visitExpr = (e: JitExpr): void => {
    switch (e.tag) {
      case "StructArrayMemberRead": {
        const key = structArrayElementsKey(
          e.structVarName,
          e.structArrayFieldName
        );
        if (!out.has(key)) {
          out.set(key, {
            structVarName: e.structVarName,
            structArrayFieldName: e.structArrayFieldName,
          });
        }
        visitExpr(e.indexExpr);
        return;
      }
      case "Binary":
        visitExpr(e.left);
        visitExpr(e.right);
        return;
      case "Unary":
        visitExpr(e.operand);
        return;
      case "Call":
      case "UserCall":
        for (const a of e.args) visitExpr(a);
        return;
      case "Index":
        visitExpr(e.base);
        for (const idx of e.indices) visitExpr(idx);
        return;
      case "TensorLiteral":
        for (const row of e.rows) for (const c of row) visitExpr(c);
        return;
      case "VConcatGrow":
        visitExpr(e.base);
        visitExpr(e.value);
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
          visitExpr(s.expr);
          break;
        case "AssignIndex":
          for (const idx of s.indices) visitExpr(idx);
          visitExpr(s.value);
          break;
        case "AssignIndexRange":
          visitExpr(s.dstStart);
          visitExpr(s.dstEnd);
          if (s.srcStart) visitExpr(s.srcStart);
          if (s.srcEnd) visitExpr(s.srcEnd);
          break;
        case "MultiAssign":
          for (const a of s.args) visitExpr(a);
          break;
        case "If":
          visitExpr(s.cond);
          visitStmts(s.thenBody);
          for (const eib of s.elseifBlocks) {
            visitExpr(eib.cond);
            visitStmts(eib.body);
          }
          if (s.elseBody) visitStmts(s.elseBody);
          break;
        case "For":
          visitExpr(s.start);
          if (s.step) visitExpr(s.step);
          visitExpr(s.end);
          visitStmts(s.body);
          break;
        case "While":
          visitExpr(s.cond);
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
      // If `name` is a hoisted tensor variable, refresh its hoisted aliases
      // (`.data`, `.length`, shape) so subsequent reads/writes see the new
      // value. Without this, `out_pt = zeros(N*2, 1)` followed by
      // `out_pt(i) = v` would write to the OLD hoisted buffer.
      emitHoistRefresh(lines, stmt.name, indent);
      break;

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
  const dstStart = emitExpr(stmt.dstStart);
  const dstEnd = emitExpr(stmt.dstEnd);
  // stage 9: when srcStart/srcEnd are null, the source is used in its
  // entirety — substitute `1` and the source's hoisted length alias. The
  // same helper handles both forms since the check is length-based.
  const srcStart = stmt.srcStart !== null ? emitExpr(stmt.srcStart) : "1";
  const srcEnd = stmt.srcEnd !== null ? emitExpr(stmt.srcEnd) : srcAlias.len;
  return `$h.setRange1r_h(${dstAlias.data}, ${dstAlias.len}, ${dstStart}, ${dstEnd}, ${srcAlias.data}, ${srcAlias.len}, ${srcStart}, ${srcEnd})`;
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
  // Internal helper calls (prefixed with __) go directly to $h
  if (expr.name.startsWith("__")) {
    return `$h.${expr.name}(${args.join(", ")})`;
  }
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
