/**
 * Simplified code generator for scalar-only functions.
 * Produces self-contained JS with no runtime dependency.
 */

import type { Stmt, Expr } from "../parser/types.js";
import { BinaryOperation, UnaryOperation } from "../parser/types.js";
import type { FunctionDef, LoopAnalysis } from "./types.js";

// Builtins that operate on scalars and map to JS equivalents.
// Value is the expected argument count.
const SCALAR_BUILTINS: Record<string, number> = {
  sin: 1,
  cos: 1,
  tan: 1,
  asin: 1,
  acos: 1,
  atan: 1,
  atan2: 2,
  sinh: 1,
  cosh: 1,
  tanh: 1,
  sqrt: 1,
  abs: 1,
  floor: 1,
  ceil: 1,
  round: 1,
  exp: 1,
  log: 1,
  log2: 1,
  log10: 1,
  min: 2,
  max: 2,
  mod: 2,
  rem: 2,
  sign: 1,
  fix: 1,
  power: 2,
};

// JS reserved words that need mangling
const JS_RESERVED = new Set([
  "break",
  "case",
  "catch",
  "continue",
  "debugger",
  "default",
  "delete",
  "do",
  "else",
  "finally",
  "for",
  "function",
  "if",
  "in",
  "instanceof",
  "new",
  "return",
  "switch",
  "this",
  "throw",
  "try",
  "typeof",
  "var",
  "void",
  "while",
  "with",
  "let",
  "const",
  "class",
  "export",
  "import",
  "yield",
  "static",
  "enum",
  "await",
  "super",
  "NaN",
  "Infinity",
  "undefined",
]);

function mangle(name: string): string {
  return JS_RESERVED.has(name) ? `_m$${name}` : name;
}

/**
 * Check whether a function's AST can be compiled to scalar-only JS.
 * Returns the set of local variable names if yes, or null if not compilable.
 */
export function isScalarCompilable(fn: FunctionDef): Set<string> | null {
  const locals = new Set<string>(fn.params);
  for (const out of fn.outputs) locals.add(out);

  function checkExpr(e: Expr): boolean {
    switch (e.type) {
      case "Number":
        return true;
      case "Ident":
        return locals.has(e.name);
      case "Binary":
        switch (e.op) {
          case BinaryOperation.Add:
          case BinaryOperation.Sub:
          case BinaryOperation.Mul:
          case BinaryOperation.ElemMul:
          case BinaryOperation.Div:
          case BinaryOperation.ElemDiv:
          case BinaryOperation.Pow:
          case BinaryOperation.ElemPow:
          case BinaryOperation.Equal:
          case BinaryOperation.NotEqual:
          case BinaryOperation.Less:
          case BinaryOperation.LessEqual:
          case BinaryOperation.Greater:
          case BinaryOperation.GreaterEqual:
          case BinaryOperation.AndAnd:
          case BinaryOperation.OrOr:
            return checkExpr(e.left) && checkExpr(e.right);
          default:
            return false;
        }
      case "Unary":
        switch (e.op) {
          case UnaryOperation.Plus:
          case UnaryOperation.Minus:
          case UnaryOperation.Not:
            return checkExpr(e.operand);
          default:
            return false;
        }
      case "FuncCall": {
        const expected = SCALAR_BUILTINS[e.name];
        if (expected === undefined || e.args.length !== expected) return false;
        return e.args.every(checkExpr);
      }
      case "Range":
        return (
          checkExpr(e.start) &&
          checkExpr(e.end) &&
          (e.step === null || checkExpr(e.step))
        );
      default:
        return false;
    }
  }

  function checkStmt(s: Stmt): boolean {
    switch (s.type) {
      case "Assign":
        locals.add(s.name);
        return checkExpr(s.expr);
      case "ExprStmt":
        return checkExpr(s.expr);
      case "If":
        return (
          checkExpr(s.cond) &&
          s.thenBody.every(checkStmt) &&
          s.elseifBlocks.every(
            b => checkExpr(b.cond) && b.body.every(checkStmt)
          ) &&
          (s.elseBody === null || s.elseBody.every(checkStmt))
        );
      case "While":
        return checkExpr(s.cond) && s.body.every(checkStmt);
      case "For":
        if (s.expr.type !== "Range") return false;
        locals.add(s.varName);
        return checkExpr(s.expr) && s.body.every(checkStmt);
      case "Break":
      case "Continue":
      case "Return":
        return true;
      default:
        return false;
    }
  }

  for (const s of fn.body) {
    if (!checkStmt(s)) return null;
  }
  return locals;
}

// ── Shared expression emitters ───────────────────────────────────────────

function emitExpr(e: Expr): string {
  switch (e.type) {
    case "Number":
      return e.value;
    case "Ident":
      return mangle(e.name);
    case "Binary":
      return emitBinary(e.left, e.op, e.right);
    case "Unary":
      return emitUnary(e.op, e.operand);
    case "FuncCall":
      return emitBuiltinCall(e.name, e.args);
    case "Range":
      throw new Error("Range outside for loop");
    default:
      throw new Error(`Unexpected expr type: ${e.type}`);
  }
}

function emitBinary(left: Expr, op: BinaryOperation, right: Expr): string {
  const l = emitExpr(left);
  const r = emitExpr(right);
  switch (op) {
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
    case BinaryOperation.Pow:
    case BinaryOperation.ElemPow:
      return `Math.pow(${l}, ${r})`;
    case BinaryOperation.Equal:
      return `((${l} === ${r}) ? 1 : 0)`;
    case BinaryOperation.NotEqual:
      return `((${l} !== ${r}) ? 1 : 0)`;
    case BinaryOperation.Less:
      return `((${l} < ${r}) ? 1 : 0)`;
    case BinaryOperation.LessEqual:
      return `((${l} <= ${r}) ? 1 : 0)`;
    case BinaryOperation.Greater:
      return `((${l} > ${r}) ? 1 : 0)`;
    case BinaryOperation.GreaterEqual:
      return `((${l} >= ${r}) ? 1 : 0)`;
    case BinaryOperation.AndAnd:
      return `((${l}) !== 0 ? ((${r}) !== 0 ? 1 : 0) : 0)`;
    case BinaryOperation.OrOr:
      return `((${l}) !== 0 ? 1 : ((${r}) !== 0 ? 1 : 0))`;
    default:
      throw new Error(`Unsupported binary op: ${op}`);
  }
}

function emitUnary(op: UnaryOperation, operand: Expr): string {
  const v = emitExpr(operand);
  switch (op) {
    case UnaryOperation.Plus:
      return `(+${v})`;
    case UnaryOperation.Minus:
      return `(-${v})`;
    case UnaryOperation.Not:
      return `((${v}) === 0 ? 1 : 0)`;
    default:
      throw new Error(`Unsupported unary op: ${op}`);
  }
}

function emitBuiltinCall(name: string, args: Expr[]): string {
  const a = args.map(emitExpr);
  switch (name) {
    case "sin":
    case "cos":
    case "tan":
    case "asin":
    case "acos":
    case "atan":
    case "sinh":
    case "cosh":
    case "tanh":
    case "sqrt":
    case "abs":
    case "floor":
    case "ceil":
    case "exp":
    case "log":
    case "log2":
    case "log10":
      return `Math.${name}(${a[0]})`;
    case "round":
      return `(Math.sign(${a[0]}) * Math.round(Math.abs(${a[0]})))`;
    case "atan2":
      return `Math.atan2(${a[0]}, ${a[1]})`;
    case "min":
      return `Math.min(${a[0]}, ${a[1]})`;
    case "max":
      return `Math.max(${a[0]}, ${a[1]})`;
    case "mod":
      return `((${a[0]}) - Math.floor((${a[0]}) / (${a[1]})) * (${a[1]}))`;
    case "rem":
      return `((${a[0]}) % (${a[1]}))`;
    case "sign":
      return `Math.sign(${a[0]})`;
    case "fix":
      return `Math.trunc(${a[0]})`;
    case "power":
      return `Math.pow(${a[0]}, ${a[1]})`;
    default:
      throw new Error(`Unsupported builtin: ${name}`);
  }
}

// ── Function-level JIT codegen ───────────────────────────────────────────

/**
 * Generate a self-contained JS function body for a scalar function.
 */
export function generateScalarJS(fn: FunctionDef, nargout: number): string {
  let loopCounter = 0;
  const lines: string[] = [];

  // Declare local variables (not params — those come as function arguments)
  const declared = new Set<string>(fn.params);
  for (const out of fn.outputs) declared.add(out);

  function emit(line: string) {
    lines.push(line);
  }

  function emitStmt(s: Stmt) {
    switch (s.type) {
      case "Assign": {
        const v = mangle(s.name);
        if (!declared.has(s.name)) {
          declared.add(s.name);
          emit(`let ${v} = ${emitExpr(s.expr)};`);
        } else {
          emit(`${v} = ${emitExpr(s.expr)};`);
        }
        break;
      }
      case "ExprStmt":
        emit(`${emitExpr(s.expr)};`);
        break;
      case "If": {
        emit(`if (${emitExpr(s.cond)} !== 0) {`);
        for (const st of s.thenBody) emitStmt(st);
        for (const eib of s.elseifBlocks) {
          emit(`} else if (${emitExpr(eib.cond)} !== 0) {`);
          for (const st of eib.body) emitStmt(st);
        }
        if (s.elseBody) {
          emit(`} else {`);
          for (const st of s.elseBody) emitStmt(st);
        }
        emit(`}`);
        break;
      }
      case "While":
        emit(`while (${emitExpr(s.cond)} !== 0) {`);
        for (const st of s.body) emitStmt(st);
        emit(`}`);
        break;
      case "For": {
        const range = s.expr as Extract<Expr, { type: "Range" }>;
        const id = loopCounter++;
        const v = mangle(s.varName);
        const sVar = `$_s${id}`;
        const stVar = `$_st${id}`;
        const eVar = `$_e${id}`;
        emit(`{`);
        emit(`const ${sVar} = ${emitExpr(range.start)};`);
        emit(`const ${stVar} = ${range.step ? emitExpr(range.step) : "1"};`);
        emit(`const ${eVar} = ${emitExpr(range.end)};`);
        emit(
          `if (${stVar} > 0) { for (let ${v} = ${sVar}; ${v} <= ${eVar}; ${v} += ${stVar}) {`
        );
        for (const st of s.body) emitStmt(st);
        emit(
          `} } else if (${stVar} < 0) { for (let ${v} = ${sVar}; ${v} >= ${eVar}; ${v} += ${stVar}) {`
        );
        for (const st of s.body) emitStmt(st);
        emit(`} }`);
        emit(`}`);
        break;
      }
      case "Break":
        emit("break;");
        break;
      case "Continue":
        emit("continue;");
        break;
      case "Return":
        emit(`return ${emitReturn()};`);
        break;
    }
  }

  function emitReturn(): string {
    if (nargout <= 1) {
      return mangle(fn.outputs[0]);
    }
    return "[" + fn.outputs.slice(0, nargout).map(mangle).join(", ") + "]";
  }

  // Declare output variables (initialized to 0)
  for (const out of fn.outputs) {
    if (!fn.params.includes(out)) {
      emit(`let ${mangle(out)} = 0;`);
    }
  }

  // Emit body
  for (const s of fn.body) {
    emitStmt(s);
  }

  // Final return
  emit(`return ${emitReturn()};`);

  return lines.join("\n");
}

// ── Loop-level JIT ───────────────────────────────────────────────────────

/**
 * Check if a for-loop body is scalar-compilable and collect variable info.
 * Returns a LoopAnalysis if compilable, null otherwise.
 */
export function analyzeLoopForJit(
  body: Stmt[],
  loopVar: string
): LoopAnalysis | null {
  const reads = new Set<string>();
  const writes = new Set<string>();

  function collectExpr(e: Expr): boolean {
    switch (e.type) {
      case "Number":
        return true;
      case "Ident":
        if (e.name !== loopVar) reads.add(e.name);
        return true;
      case "Binary":
        switch (e.op) {
          case BinaryOperation.Add:
          case BinaryOperation.Sub:
          case BinaryOperation.Mul:
          case BinaryOperation.ElemMul:
          case BinaryOperation.Div:
          case BinaryOperation.ElemDiv:
          case BinaryOperation.Pow:
          case BinaryOperation.ElemPow:
          case BinaryOperation.Equal:
          case BinaryOperation.NotEqual:
          case BinaryOperation.Less:
          case BinaryOperation.LessEqual:
          case BinaryOperation.Greater:
          case BinaryOperation.GreaterEqual:
          case BinaryOperation.AndAnd:
          case BinaryOperation.OrOr:
            return collectExpr(e.left) && collectExpr(e.right);
          default:
            return false;
        }
      case "Unary":
        switch (e.op) {
          case UnaryOperation.Plus:
          case UnaryOperation.Minus:
          case UnaryOperation.Not:
            return collectExpr(e.operand);
          default:
            return false;
        }
      case "FuncCall": {
        const expected = SCALAR_BUILTINS[e.name];
        if (expected === undefined || e.args.length !== expected) return false;
        return e.args.every(collectExpr);
      }
      case "Range":
        return (
          collectExpr(e.start) &&
          collectExpr(e.end) &&
          (e.step === null || collectExpr(e.step))
        );
      default:
        return false;
    }
  }

  function collectStmt(s: Stmt): boolean {
    switch (s.type) {
      case "Assign":
        writes.add(s.name);
        return collectExpr(s.expr);
      case "ExprStmt":
        return collectExpr(s.expr);
      case "If":
        return (
          collectExpr(s.cond) &&
          s.thenBody.every(collectStmt) &&
          s.elseifBlocks.every(
            b => collectExpr(b.cond) && b.body.every(collectStmt)
          ) &&
          (s.elseBody === null || s.elseBody.every(collectStmt))
        );
      case "While":
        return collectExpr(s.cond) && s.body.every(collectStmt);
      case "For":
        if (s.expr.type !== "Range") return false;
        writes.add(s.varName);
        return collectExpr(s.expr) && s.body.every(collectStmt);
      case "Break":
      case "Continue":
        return true;
      case "Return":
        return false; // can't JIT a loop containing return
      default:
        return false;
    }
  }

  for (const s of body) {
    if (!collectStmt(s)) return null;
  }

  const readArr = [...reads].sort();
  const writeArr = [...writes].sort();
  const writeOnlyArr = writeArr.filter(v => !reads.has(v));

  return {
    readVars: readArr,
    writeOnlyVars: writeOnlyArr,
    allWriteVars: writeArr,
    loopVar,
  };
}

/**
 * Generate a JS function body for a loop-level JIT.
 * Parameters: [...readVars (mangled), $_rs, $_rst, $_re]
 * Returns: null if loop didn't execute, or [allWriteVars..., loopVar] as numbers.
 */
export function generateLoopJS(
  body: Stmt[],
  loopVar: string,
  analysis: LoopAnalysis
): string {
  let loopCounter = 0;
  const lines: string[] = [];

  const declared = new Set<string>(analysis.readVars);
  const readSet = new Set(analysis.readVars);

  function emit(line: string) {
    lines.push(line);
  }

  // Declare write-only vars (not params)
  for (const v of analysis.allWriteVars) {
    if (!readSet.has(v)) {
      emit(`let ${mangle(v)};`);
      declared.add(v);
    }
  }

  // Declare loop variable
  const outerV = mangle(loopVar);
  emit(`let ${outerV};`);
  declared.add(loopVar);

  emit(`let $_ran = false;`);

  // Emit the outer for-loop using persistent loop variable
  emit(`if ($_rst > 0) { for (let $_k = $_rs; $_k <= $_re; $_k += $_rst) {`);
  emit(`$_ran = true;`);
  emit(`${outerV} = $_k;`);
  for (const s of body) emitLoopStmt(s);
  emit(
    `} } else if ($_rst < 0) { for (let $_k = $_rs; $_k >= $_re; $_k += $_rst) {`
  );
  emit(`$_ran = true;`);
  emit(`${outerV} = $_k;`);
  for (const s of body) emitLoopStmt(s);
  emit(`} }`);

  // Return null if loop didn't execute, otherwise return all outputs
  emit(`if (!$_ran) return null;`);
  const returnVars = [...analysis.allWriteVars.map(mangle), outerV];
  emit(`return [${returnVars.join(", ")}];`);

  return lines.join("\n");

  // Statement emitter with persistent loop variable handling for inner loops
  function emitLoopStmt(s: Stmt) {
    switch (s.type) {
      case "Assign": {
        const v = mangle(s.name);
        if (!declared.has(s.name)) {
          declared.add(s.name);
          emit(`let ${v} = ${emitExpr(s.expr)};`);
        } else {
          emit(`${v} = ${emitExpr(s.expr)};`);
        }
        break;
      }
      case "ExprStmt":
        emit(`${emitExpr(s.expr)};`);
        break;
      case "If": {
        emit(`if (${emitExpr(s.cond)} !== 0) {`);
        for (const st of s.thenBody) emitLoopStmt(st);
        for (const eib of s.elseifBlocks) {
          emit(`} else if (${emitExpr(eib.cond)} !== 0) {`);
          for (const st of eib.body) emitLoopStmt(st);
        }
        if (s.elseBody) {
          emit(`} else {`);
          for (const st of s.elseBody) emitLoopStmt(st);
        }
        emit(`}`);
        break;
      }
      case "While":
        emit(`while (${emitExpr(s.cond)} !== 0) {`);
        for (const st of s.body) emitLoopStmt(st);
        emit(`}`);
        break;
      case "For": {
        // Inner for-loop: use $_ki counter with persistent loop variable
        const range = s.expr as Extract<Expr, { type: "Range" }>;
        const id = loopCounter++;
        const v = mangle(s.varName);
        const sVar = `$_s${id}`;
        const stVar = `$_st${id}`;
        const eVar = `$_e${id}`;
        const kVar = `$_ki${id}`;
        emit(`{`);
        emit(`const ${sVar} = ${emitExpr(range.start)};`);
        emit(`const ${stVar} = ${range.step ? emitExpr(range.step) : "1"};`);
        emit(`const ${eVar} = ${emitExpr(range.end)};`);
        emit(
          `if (${stVar} > 0) { for (let ${kVar} = ${sVar}; ${kVar} <= ${eVar}; ${kVar} += ${stVar}) {`
        );
        emit(`${v} = ${kVar};`);
        for (const st of s.body) emitLoopStmt(st);
        emit(
          `} } else if (${stVar} < 0) { for (let ${kVar} = ${sVar}; ${kVar} >= ${eVar}; ${kVar} += ${stVar}) {`
        );
        emit(`${v} = ${kVar};`);
        for (const st of s.body) emitLoopStmt(st);
        emit(`} }`);
        emit(`}`);
        break;
      }
      case "Break":
        emit("break;");
        break;
      case "Continue":
        emit("continue;");
        break;
    }
  }
}
