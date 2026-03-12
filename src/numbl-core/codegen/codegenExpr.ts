/**
 * Expression code generation.
 *
 * Standalone functions that generate JavaScript for IR expressions.
 * Each function takes the Codegen instance as the first parameter.
 */

import {
  BinaryOperation,
  UnaryOperation,
  type IRExpr,
  type IRExprKind,
  walkExpr,
  itemTypeForExprKind,
} from "../lowering/index.js";
import { type ItemType, typeToString } from "../lowering/itemTypes.js";
import type { Codegen } from "./codegen.js";
import {
  exprContainsEndShallow,
  collectClassProperties,
  genPropertyDefaults,
} from "./codegenHelpers.js";

import type { CallSite } from "../runtime/runtimeHelpers.js";
import { resolveFunction } from "../functionResolve.js";

// ── Public entry ────────────────────────────────────────────────────────

export function genExpr(cg: Codegen, expr: IRExpr): string {
  try {
    return genExprInner(cg, expr);
  } catch (e) {
    if (e instanceof Error && e.constructor.name === "SemanticError") throw e;
    throw new Error(
      `Codegen error at ${expr.span.file}:${expr.span.start}: ${e instanceof Error ? e.message : String(e)}`
    );
  }
}

// ── Expression dispatch ─────────────────────────────────────────────────

function genExprInner(cg: Codegen, expr: IRExpr): string {
  const kind = expr.kind;
  switch (kind.type) {
    case "Number":
      return kind.value;

    case "Char":
      return `$rt.makeChar(${JSON.stringify(kind.value)})`;

    case "String":
      return `$rt.makeString(${JSON.stringify(kind.value)})`;

    case "Var":
      return cg.varRef(kind.variable.id.id);

    case "Constant":
      return `$rt.getConstant(${JSON.stringify(kind.name)})`;

    case "Unary":
      return genUnary(cg, kind);

    case "Binary":
      return genBinary(cg, kind);

    case "Tensor":
      return genTensor(cg, kind);

    case "Cell":
      return genCell(cg, kind);

    case "Index":
      return genIndex(cg, kind);

    case "IndexCell":
      return genIndexCell(cg, kind);

    case "Range":
      return genRange(cg, kind);

    case "Colon":
      return `$rt.COLON`;

    case "End":
      return cg.insideDeferredEnd ? `$end` : `$rt.END`;

    case "Member":
      return genMember(cg, kind);

    case "MemberDynamic":
      return genMemberDynamic(cg, kind);

    case "MethodCall":
      return genMethodCall(cg, kind);

    case "AnonFunc":
      return genAnonFunc(cg, kind);

    case "FuncHandle": {
      // Static method handle: @ClassName.method → wrap callClassMethod
      if (kind.name.includes(".")) {
        const dotIdx = kind.name.indexOf(".");
        const className = kind.name.substring(0, dotIdx);
        const methodName = kind.name.substring(dotIdx + 1);
        cg.ensureClassRegistered(className);
        return `$rt.makeUserFuncHandle(function($nargout) { return $rt.callClassMethod(${JSON.stringify(className)}, ${JSON.stringify(methodName)}, $nargout, Array.prototype.slice.call(arguments, 1)); })`;
      }

      // Nested function: reference directly via closure (no registry lookup)
      if (cg.nestedFunctionNames.has(kind.name)) {
        const fnRef = `$nested_function_${cg.sanitizeName(kind.name)}`;
        const nestedStub = cg.loweringCtx.getLocalFunctionStub(kind.name);
        const nestedNargin = nestedStub ? nestedStub.params.length : undefined;
        return `$rt.makeUserFuncHandle(function($nargout) { return ${fnRef}($nargout, ...Array.prototype.slice.call(arguments, 1)); }${nestedNargin !== undefined ? `, ${nestedNargin}` : ""})`;
      }

      // ── Unified resolution via resolveFunction ──────────────────────
      const fnIndex = cg.loweringCtx.functionIndex;

      // Inside a class method, if the handle name is a method of the owner
      // class, use runtime dispatch — @method should dispatch based on the
      // first argument's type at call time (not hard-wire to this class's method).
      if (cg.loweringCtx.ownerClassName) {
        const cls = cg.loweringCtx.ownerClassName;
        if (
          fnIndex.classInstanceMethods.get(cls)?.has(kind.name) ||
          fnIndex.classStaticMethods.get(cls)?.has(kind.name)
        ) {
          return `$rt.getFuncHandle(${JSON.stringify(kind.name)})`;
        }
      }

      // Class constructors: use getFuncHandle so feval dispatches through
      // the class construction path (makeUserFuncHandle loses the name).
      if (fnIndex.workspaceClasses.has(kind.name)) {
        return `$rt.getFuncHandle(${JSON.stringify(kind.name)})`;
      }

      // Resolve with empty arg types (handle will be called later)
      const handleTarget = resolveFunction(
        kind.name,
        [],
        getCallSite(cg),
        fnIndex
      );

      if (handleTarget) {
        switch (handleTarget.kind) {
          case "localFunction": {
            const jsId = cg.ensureSpecializedFunctionGenerated(kind.name, []);
            if (jsId) {
              const stub = cg.loweringCtx.getLocalFunctionStub(kind.name);
              const localNargin = stub ? stub.params.length : undefined;
              return `$rt.makeUserFuncHandle(function($nargout) { return ${jsId}($nargout, ...Array.prototype.slice.call(arguments, 1)); }${localNargin !== undefined ? `, ${localNargin}` : ""})`;
            }
            break;
          }
          case "privateFunction": {
            const jsId = cg.ensurePrivateFunctionGenerated(kind.name, []);
            if (jsId) {
              return `$rt.makeUserFuncHandle(function($nargout) { return ${jsId}($nargout, ...Array.prototype.slice.call(arguments, 1)); })`;
            }
            break;
          }
          case "workspaceFunction": {
            const jsId = cg.ensureWorkspaceFunctionGenerated(kind.name, []);
            if (jsId) {
              return `$rt.makeUserFuncHandle(function($nargout) { return ${jsId}($nargout, ...Array.prototype.slice.call(arguments, 1)); })`;
            }
            break;
          }
          default:
            // builtins, classMethod, workspaceClassConstructor → runtime dispatch
            break;
        }
      }

      return `$rt.getFuncHandle(${JSON.stringify(kind.name)})`;
    }

    case "FuncCall":
      return genFuncCall(cg, kind);

    case "MetaClass":
      return `$rt.RTV.string(${JSON.stringify(kind.name)})`;

    case "ClassInstantiation":
      return genClassInstantiation(cg, kind);

    case "SuperConstructorCall":
      return genSuperConstructorCall(cg, kind);

    case "RuntimeError":
      return `(()=>{ throw $rt.error(${JSON.stringify(kind.message)}); })()`;
  }
}

// ── Expression helpers ──────────────────────────────────────────────────

function genUnary(
  cg: Codegen,
  kind: Extract<IRExprKind, { type: "Unary" }>
): string {
  const operand = genExpr(cg, kind.operand);
  const operandType = itemTypeForExprKind(kind.operand.kind);
  if (operandType.kind === "Number") {
    switch (kind.op) {
      case UnaryOperation.Plus:
        return `(+${operand})`;
      case UnaryOperation.Minus:
        return `(-${operand})`;
      case UnaryOperation.Transpose:
        return operand;
      case UnaryOperation.NonConjugateTranspose:
        return operand;
      case UnaryOperation.Not:
        return `$rt.lg(${operand} === 0)`;
    }
  }
  switch (kind.op) {
    case UnaryOperation.Plus:
      return `$rt.uplus(${operand})`;
    case UnaryOperation.Minus:
      return `$rt.uminus(${operand})`;
    case UnaryOperation.Transpose:
      return `$rt.ctranspose(${operand})`;
    case UnaryOperation.NonConjugateTranspose:
      return `$rt.transpose(${operand})`;
    case UnaryOperation.Not:
      return `$rt.not(${operand})`;
  }
}

function genBinary(
  cg: Codegen,
  kind: Extract<IRExprKind, { type: "Binary" }>
): string {
  const left = genExpr(cg, kind.left);
  const right = genExpr(cg, kind.right);

  if (kind.op === BinaryOperation.AndAnd) {
    return `$rt.lg($rt.toBool(${left}) ? $rt.toBool(${right}) : false)`;
  }
  if (kind.op === BinaryOperation.OrOr) {
    return `$rt.lg($rt.toBool(${left}) ? true : $rt.toBool(${right}))`;
  }

  const leftType = itemTypeForExprKind(kind.left.kind);
  const rightType = itemTypeForExprKind(kind.right.kind);

  if (leftType.kind === "Number" && rightType.kind === "Number") {
    switch (kind.op) {
      case BinaryOperation.Add:
        return `(${left} + ${right})`;
      case BinaryOperation.Sub:
        return `(${left} - ${right})`;
      case BinaryOperation.Mul:
        return `(${left} * ${right})`;
      case BinaryOperation.Div:
        return `(${left} / ${right})`;
      case BinaryOperation.Pow:
        return `$rt.pow(${left}, ${right})`;
      case BinaryOperation.ElemMul:
        return `(${left} * ${right})`;
      case BinaryOperation.ElemDiv:
        return `(${left} / ${right})`;
      case BinaryOperation.ElemPow:
        return `$rt.pow(${left}, ${right})`;
      case BinaryOperation.LeftDiv:
        return `(${right} / ${left})`;
      case BinaryOperation.ElemLeftDiv:
        return `(${right} / ${left})`;
      case BinaryOperation.Equal:
        return `$rt.lg(${left} === ${right})`;
      case BinaryOperation.NotEqual:
        return `$rt.lg(${left} !== ${right})`;
      case BinaryOperation.Less:
        return `$rt.lg(${left} < ${right})`;
      case BinaryOperation.LessEqual:
        return `$rt.lg(${left} <= ${right})`;
      case BinaryOperation.Greater:
        return `$rt.lg(${left} > ${right})`;
      case BinaryOperation.GreaterEqual:
        return `$rt.lg(${left} >= ${right})`;
      case BinaryOperation.BitAnd:
        return `$rt.lg(${left} !== 0 && ${right} !== 0)`;
      case BinaryOperation.BitOr:
        return `$rt.lg(${left} !== 0 || ${right} !== 0)`;
      default:
        break;
    }
  }

  // If both types are known and neither is a class instance, binop is sync
  if (
    leftType.kind !== "Unknown" &&
    rightType.kind !== "Unknown" &&
    leftType.kind !== "ClassInstance" &&
    rightType.kind !== "ClassInstance"
  ) {
    return `$rt.binopSync(${JSON.stringify(kind.op)}, ${left}, ${right})`;
  }

  return `$rt.binop(${JSON.stringify(kind.op)}, ${left}, ${right})`;
}

function genTensor(
  cg: Codegen,
  kind: Extract<IRExprKind, { type: "Tensor" }>
): string {
  if (kind.rows.length === 0) return `$rt.emptyTensor()`;
  const genRow = (row: IRExpr[]) => {
    const elems = row.map(e => {
      const code = genExpr(cg, e);
      return isCslProducer(e) ? `...$rt.asCsl(${code})` : code;
    });
    return `$rt.horzcat([${elems.join(", ")}])`;
  };
  if (kind.rows.length === 1) {
    return genRow(kind.rows[0]);
  }
  const rowExprs = kind.rows.map(row => genRow(row));
  return `$rt.vertcat([${rowExprs.join(", ")}])`;
}

function genCell(
  cg: Codegen,
  kind: Extract<IRExprKind, { type: "Cell" }>
): string {
  const totalRows = kind.rows.length;
  const totalCols = totalRows > 0 ? kind.rows[0].length : 0;
  const allElems: string[] = [];
  for (let c = 0; c < totalCols; c++) {
    for (let r = 0; r < totalRows; r++) {
      allElems.push(genExpr(cg, kind.rows[r][c]));
    }
  }
  return `$rt.makeCell([${allElems.join(", ")}], [${totalRows}, ${totalCols}])`;
}

export function genIndexArg(cg: Codegen, expr: IRExpr): string {
  if (expr.kind.type === "End" || expr.kind.type === "Colon") {
    return genExpr(cg, expr);
  }
  if (expr.kind.type === "Range") {
    const hasCompoundEnd =
      (expr.kind.start.kind.type !== "End" &&
        exprContainsEndShallow(expr.kind.start)) ||
      (expr.kind.step !== null &&
        expr.kind.step.kind.type !== "End" &&
        exprContainsEndShallow(expr.kind.step)) ||
      (expr.kind.end.kind.type !== "End" &&
        exprContainsEndShallow(expr.kind.end));
    if (!hasCompoundEnd) return genExpr(cg, expr);
  }
  if (exprContainsEndShallow(expr)) {
    const code = cg.withCodegenContext({ insideDeferredEnd: true }, () =>
      genExpr(cg, expr)
    );
    return `(($end) => ${code})`;
  }
  return genExpr(cg, expr);
}

function genIndex(
  cg: Codegen,
  kind: Extract<IRExprKind, { type: "Index" }>
): string {
  const base = genExpr(cg, kind.base);
  const indices = kind.indices.map(i => {
    const code = genIndexArg(cg, i);
    return isCslProducer(i) ? `...$rt.asCsl(${code})` : code;
  });
  // In classdef, obj(k) inside a class method (or file-local
  // subfunction) uses built-in array indexing, not the overloaded subsref.
  // When the base type is known at compile time, pass `true`.  Otherwise,
  // if we are inside a classdef context, pass the class name so the
  // runtime can check at execution time.
  const baseType = itemTypeForExprKind(kind.base.kind);
  const ownerClassName = cg.loweringCtx.ownerClassName;
  const skipKnown =
    baseType.kind === "ClassInstance" && ownerClassName === baseType.className;
  const skipArg = skipKnown
    ? "true"
    : ownerClassName
      ? JSON.stringify(ownerClassName)
      : null;
  const nargoutArg =
    cg.nargoutOverride ??
    (kind.nargout && kind.nargout > 1 ? String(kind.nargout) : "1");
  if (skipArg) {
    return `$rt.index(${base}, [${indices.join(", ")}], ${nargoutArg}, ${skipArg})`;
  }
  if (cg.nargoutOverride || (kind.nargout && kind.nargout > 1)) {
    return `$rt.index(${base}, [${indices.join(", ")}], ${nargoutArg})`;
  }
  return `$rt.index(${base}, [${indices.join(", ")}])`;
}

function genIndexCell(
  cg: Codegen,
  kind: Extract<IRExprKind, { type: "IndexCell" }>
): string {
  const base = genExpr(cg, kind.base);
  const indices = kind.indices.map(i => genIndexArg(cg, i));
  return `$rt.indexCell(${base}, [${indices.join(", ")}])`;
}

function genRange(
  cg: Codegen,
  kind: Extract<IRExprKind, { type: "Range" }>
): string {
  const start = genExpr(cg, kind.start);
  const end = genExpr(cg, kind.end);
  if (kind.step) {
    const step = genExpr(cg, kind.step);
    return `$rt.range(${start}, ${step}, ${end})`;
  }
  return `$rt.range(${start}, 1, ${end})`;
}

export function genForRange(
  cg: Codegen,
  loopVarId: string,
  body: () => void,
  range: Extract<IRExprKind, { type: "Range" }>
): void {
  const loopVar = cg.varRef(loopVarId);
  const numericExpr = (expr: IRExpr, rawCode: string): string =>
    getNumericLiteral(expr) !== null ? rawCode : `$rt.toNumber(${rawCode})`;

  const startExpr = numericExpr(range.start, genExpr(cg, range.start));
  const endExpr = numericExpr(range.end, genExpr(cg, range.end));
  const stepLiteral = range.step ? getNumericLiteral(range.step) : 1;
  const stepExpr = range.step
    ? numericExpr(range.step, genExpr(cg, range.step))
    : "1";
  const endTemp = cg.freshTemp();
  cg.emit(`var ${endTemp} = ${endExpr};`);

  // Use a separate counter so that (a) the user variable retains the last
  // iterated value after the loop exits and (b) assignments to the loop
  // variable inside the body don't affect iteration.
  const counter = cg.freshTemp();

  if (stepLiteral !== null) {
    if (stepLiteral === 0) {
      cg.emit(`for (var ${counter} = ${startExpr}; false; ) {`);
    } else {
      const cmp = stepLiteral > 0 ? "<=" : ">=";
      const eps = Number.isInteger(stepLiteral)
        ? ""
        : stepLiteral > 0
          ? " + 1e-10"
          : " - 1e-10";
      cg.emit(
        `for (var ${counter} = ${startExpr}; ${counter} ${cmp} ${endTemp}${eps}; ${counter} += ${stepExpr}) {`
      );
    }
  } else {
    const stepTemp = cg.freshTemp();
    cg.emit(`var ${stepTemp} = ${stepExpr};`);
    cg.emit(
      `for (var ${counter} = ${startExpr}; ${stepTemp} > 0 ? ${counter} <= ${endTemp} + 1e-10 : ${stepTemp} < 0 ? ${counter} >= ${endTemp} - 1e-10 : false; ${counter} += ${stepTemp}) {`
    );
  }

  cg.pushIndent();
  cg.emit(`${loopVar} = ${counter};`);
  body();
  cg.popIndent();
  cg.emit(`}`);
}

function getNumericLiteral(expr: IRExpr): number | null {
  if (expr.kind.type === "Number") {
    const n = Number(expr.kind.value);
    return isNaN(n) ? null : n;
  }
  if (expr.kind.type === "Unary" && expr.kind.op === UnaryOperation.Minus) {
    const inner = getNumericLiteral(expr.kind.operand);
    return inner !== null ? -inner : null;
  }
  return null;
}

function genMember(
  cg: Codegen,
  kind: Extract<IRExprKind, { type: "Member" }>
): string {
  // Walk the chain of nested Member nodes to find the root expression
  const chain: string[] = [kind.name];
  let current: IRExpr = kind.base;
  while (current.kind.type === "Member") {
    chain.unshift(current.kind.name);
    current = current.kind.base;
  }
  // Check if root is a class instance with overloaded subsref
  const rootType = itemTypeForExprKind(current.kind);
  if (
    rootType.kind === "ClassInstance" &&
    hasSubsref(cg, rootType.className) &&
    cg.loweringCtx.ownerClassName !== rootType.className
  ) {
    const rootCode = genExpr(cg, current);
    return `$rt.subsrefCall(${rootCode}, ${JSON.stringify(chain)})`;
  }
  // Check if we can emit a direct field access (known stored property, no getter, no subsref)
  const baseType = itemTypeForExprKind(kind.base.kind);
  if (baseType.kind === "ClassInstance") {
    const info = cg.loweringCtx.getClassInfo(baseType.className);
    if (
      info &&
      info.propertyNames.includes(kind.name) &&
      !cg.loweringCtx.classHasMethod(baseType.className, `get.${kind.name}`) &&
      !hasSubsref(cg, baseType.className)
    ) {
      const base = genExpr(cg, kind.base);
      return `${base}.fields.get(${JSON.stringify(kind.name)})`;
    }
  }
  // Normal path
  const base = genExpr(cg, kind.base);
  return `$rt.getMember(${base}, ${JSON.stringify(kind.name)})`;
}

function genMemberDynamic(
  cg: Codegen,
  kind: Extract<IRExprKind, { type: "MemberDynamic" }>
): string {
  const base = genExpr(cg, kind.base);
  const nameExpr = genExpr(cg, kind.nameExpr);
  return `$rt.getMemberDynamic(${base}, ${nameExpr})`;
}

/** Generate dot-dispatch for unknown-type obj.method(args).
 *  Known-type cases are now lowered as FuncCall with targetClassName. */
function genMethodCall(
  cg: Codegen,
  kind: Extract<IRExprKind, { type: "MethodCall" }>
): string {
  const base = genExpr(cg, kind.base);
  const args = kind.args.map(a => {
    const code = genIndexArg(cg, a);
    return isCslProducer(a) ? `...$rt.asCsl(${code})` : code;
  });
  const allArgs = [base, ...args].join(", ");
  const nargout = cg.nargoutOverride ?? kind.nargout;
  return `$rt.methodDispatch(${JSON.stringify(kind.name)}, ${nargout}, [${allArgs}])`;
}

function genSuperConstructorCall(
  cg: Codegen,
  kind: Extract<IRExprKind, { type: "SuperConstructorCall" }>
): string {
  const args = kind.args.map(a => {
    const code = genExpr(cg, a);
    return isCslProducer(a) ? `...$rt.asCsl(${code})` : code;
  });
  const objRef = cg.varRef(kind.objVar.id.id);

  // Try compile-time specialization of the super constructor
  const classInfo = cg.loweringCtx.getClassInfo(kind.superClassName);
  if (classInfo && classInfo.constructorName) {
    const selfType: ItemType = {
      kind: "ClassInstance",
      className: kind.superClassName,
    };
    const argTypes: ItemType[] = [
      selfType,
      ...kind.args.map(a => itemTypeForExprKind(a.kind)),
    ];
    const jsId = cg.ensureClassMethodGenerated(
      kind.superClassName,
      classInfo.constructorName,
      argTypes
    );
    if (jsId) {
      return `$rt.callSuperConstructor(${objRef}, ${jsId}(1, ${objRef}, ${args.join(", ")}))`;
    }
  }
  if (kind.superClassName === "double") {
    // Built-in super class: store the data as _builtinData on the instance
    if (args.length !== 1) {
      return `(()=>{ throw $rt.error("double constructor expects exactly one argument"); })()`;
    }
    return `$rt.callSuperConstructor(${objRef}, ${args[0]})`;
  }

  // generate a runtime error
  return `(()=>{ throw $rt.error("No constructor found for superclass ${kind.superClassName}"); })()`;
}

function genAnonFunc(
  cg: Codegen,
  kind: Extract<IRExprKind, { type: "AnonFunc" }>
): string {
  const paramIdSet = new Set(kind.params.map(p => p.id.id));
  const freeVarIds = new Set<string>();
  const innerParamIds = new Set<string>();
  walkExpr(kind.body, e => {
    if (e.kind.type === "Var") freeVarIds.add(e.kind.variable.id.id);
    else if (e.kind.type === "AnonFunc")
      for (const p of e.kind.params) innerParamIds.add(p.id.id);
  });
  for (const id of innerParamIds) freeVarIds.delete(id);
  for (const pid of paramIdSet) freeVarIds.delete(pid);

  const paramVarDecls = kind.params
    .map(p => `var ${cg.varRef(p.id.id)}`)
    .join("; ");
  const paramAssigns = kind.params
    .map((p, i) => `${cg.varRef(p.id.id)} = arguments[${i + 1}]`)
    .join("; ");
  const paramsPreamble = `${paramVarDecls}; ${paramAssigns}`;

  const nParams = kind.params.length;
  return cg.withCodegenContext({ nargoutOverride: "$nargout" }, () => {
    if (freeVarIds.size === 0) {
      const body = genExpr(cg, kind.body);
      return `$rt.makeUserFuncHandle(function($nargout) { ${paramsPreamble}; return ${body}; }, ${nParams})`;
    }

    const captureIds = [...freeVarIds];
    const captureParams = captureIds.map(id => `$c_${id}`).join(", ");
    const captureArgs = captureIds
      .map(id => `$rt.share(${cg.varRef(id)})`)
      .join(", ");

    const overrideMap = new Map<string, string>();
    for (const id of freeVarIds) {
      overrideMap.set(id, `$c_${id}`);
    }
    cg.pushVarRefOverride(overrideMap);
    try {
      const body = genExpr(cg, kind.body);
      return `(function(${captureParams}) { return $rt.makeUserFuncHandle(function($nargout) { ${paramsPreamble}; return ${body}; }, ${nParams}); })(${captureArgs})`;
    } finally {
      cg.popVarRefOverride();
    }
  });
}

// ── Unknown-type dispatch helpers ────────────────────────────────────────

/** Compute arg types and check for unknowns. Returns argTypes for reuse by specialization. */
function getArgTypesAndCheckUnknown(irArgs: IRExpr[]): {
  argTypes: ItemType[];
  hasUnknown: boolean;
} {
  const argTypes = irArgs.map(a => itemTypeForExprKind(a.kind));
  const hasUnknown = argTypes.some(t => !t || t.kind === "Unknown");
  return { argTypes, hasUnknown };
}

/** Build a dispatchUnknown expression for runtime JIT dispatch. */
function emitDispatchUnknown(
  cg: Codegen,
  name: string,
  nargout: number | string,
  args: string[],
  callSite: CallSite
): string {
  const nargoutExpr = cg.nargoutOverride ?? nargout;
  return `(${`$rt.dispatchUnknown(${JSON.stringify(name)}, ${nargoutExpr}, [${args.join(", ")}], ${JSON.stringify(callSite)})`})`;
}

/** Get the call site for the current codegen context. */
function getCallSite(cg: Codegen): CallSite {
  return {
    file: cg.loweringCtx.mainFileName,
    ...(cg.loweringCtx.ownerClassName
      ? { className: cg.loweringCtx.ownerClassName }
      : {}),
    ...(cg.currentMethodName ? { methodName: cg.currentMethodName } : {}),
  };
}

/** Build the JS call expression for a specialized function. */
function buildSpecializedCall(
  cg: Codegen,
  jsId: string,
  nargout: number | string,
  args: string[]
): string {
  const n = cg.nargoutOverride ?? nargout;
  return `${jsId}(${[n, ...args].join(", ")})`;
}

/** Check if an expression produces a comma-separated list (CSL) that needs spreading.
 *  Cell curly-brace indexing can produce a CSL at runtime
 *  (e.g. X{:}, X{1:3}, X{idx}, or X{1,:} where result spans multiple cells). */
function isCslProducer(expr: IRExpr): boolean {
  return expr.kind.type === "IndexCell";
}

function genFuncCall(
  cg: Codegen,
  kind: Extract<IRExprKind, { type: "FuncCall" }>
): string {
  const args = kind.args.map(a => {
    const code = genExpr(cg, a);
    return isCslProducer(a) ? `...$rt.asCsl(${code})` : code;
  });

  // Compile-time intrinsics
  if (kind.name === "nargin" && kind.args.length === 0)
    return `(arguments.length - ${1 + cg.narginAdjust})`;
  if (kind.name === "nargout" && kind.args.length === 0) return `$nargout`;
  if (kind.name === "narginchk" && kind.args.length === 2) {
    const actualNargin = `(arguments.length - ${1 + cg.narginAdjust})`;
    return `$rt.narginchk(${actualNargin}, ${args[0]}, ${args[1]})`;
  }
  if (kind.name === "nargoutchk" && kind.args.length === 2) {
    return `$rt.nargoutchk($nargout, ${args[0]}, ${args[1]})`;
  }
  if (kind.name === "__inferred_type_str") {
    if (kind.args.length !== 1)
      throw new Error("__inferred_type_str expects exactly one argument");
    const argType = itemTypeForExprKind(kind.args[0].kind);
    const typeStr = typeToString(argType);
    return `$rt.makeString('"${typeStr}"')`;
  }
  if (kind.name === "isa") return `$rt.isa(${args[0]}, ${args[1]})`;
  if (kind.name === "exist" && kind.args.length === 2) {
    const arg0 = kind.args[0].kind;
    const arg1 = kind.args[1].kind;
    if (arg0.type === "Char" && arg1.type === "Char") {
      const varName = arg0.value.replace(/^'|'$/g, "");
      const typeArg = arg1.value.replace(/^'|'$/g, "");
      if (typeArg === "var") {
        const variable = cg.loweringCtx.allVariables.find(
          v => v.name === varName
        );
        if (variable) {
          return `(${cg.varRef(variable.id.id)} !== undefined ? 1 : 0)`;
        }
        return "0";
      }
    }
  }
  if (kind.name === "figure")
    return `$rt.plot_instr({type: "set_figure_handle", handle: ${args[0] ?? "1"}})`;
  if (kind.name === "hold")
    return `$rt.plot_instr({type: "set_hold", value: ${args[0]}})`;
  if (kind.name === "ishold") return `$rt.ishold()`;
  if (kind.name === "clf") return `$rt.plot_instr({type: "clf"})`;
  if (kind.name === "title")
    return `$rt.plot_instr({type: "set_title", text: ${args[0] ?? '""'}})`;
  if (kind.name === "xlabel")
    return `$rt.plot_instr({type: "set_xlabel", text: ${args[0] ?? '""'}})`;
  if (kind.name === "ylabel")
    return `$rt.plot_instr({type: "set_ylabel", text: ${args[0] ?? '""'}})`;
  if (kind.name === "shading")
    return `$rt.plot_instr({type: "set_shading", shading: ${args[0] ?? '"faceted"'}})`;
  if (kind.name === "close") {
    if (args.length > 0) {
      return `$rt.plot_instr({type: "close_all"})`;
    }
    return `$rt.plot_instr({type: "close"})`;
  }
  if (kind.name === "subplot")
    return `$rt.plot_instr({type: "set_subplot", rows: ${args[0] ?? "1"}, cols: ${args[1] ?? "1"}, index: ${args[2] ?? "1"}})`;
  if (kind.name === "legend") return `$rt.legend_call([${args.join(", ")}])`;
  if (kind.name === "sgtitle")
    return `$rt.plot_instr({type: "set_sgtitle", text: ${args[0] ?? '""'}})`;
  if (kind.name === "grid")
    return `$rt.plot_instr({type: "set_grid", value: ${args[0] ?? "true"}})`;

  // Nested function: call directly via closure (not through registry)
  if (cg.nestedFunctionNames.has(kind.name)) {
    return buildSpecializedCall(
      cg,
      `$nested_function_${cg.sanitizeName(kind.name)}`,
      kind.nargout,
      args
    );
  }

  // ── Direct class method resolution (static methods, super calls) ────
  if (kind.targetClassName) {
    const methodName = kind.methodName ?? kind.name;
    const nargout = cg.nargoutOverride ?? kind.nargout;

    // Ensure the class is registered
    cg.ensureClassRegistered(kind.targetClassName);

    // When called on an instance (obj.staticMethod(args)), if we can't resolve
    // at compile time, use runtime dispatch based on the instance's actual class.
    if (kind.instanceBase) {
      const argTypes: ItemType[] = kind.args.map(a =>
        itemTypeForExprKind(a.kind)
      );
      const jsId = cg.ensureClassMethodGenerated(
        kind.targetClassName,
        methodName,
        argTypes
      );
      if (jsId) {
        return buildSpecializedCall(cg, jsId, nargout, args);
      }
      // Fallback: use the instance's runtime class for dispatch
      const base = genExpr(cg, kind.instanceBase);
      return `$rt.callClassMethod($rt.getClassName(${base}), ${JSON.stringify(methodName)}, ${nargout}, [${args.join(", ")}])`;
    }

    // Direct class method call (ClassName.method or super.method)
    const argTypes: ItemType[] = kind.args.map(a =>
      itemTypeForExprKind(a.kind)
    );
    const jsId = cg.ensureClassMethodGenerated(
      kind.targetClassName,
      methodName,
      argTypes
    );
    if (jsId) {
      return buildSpecializedCall(cg, jsId, nargout, args);
    }

    // Fallback: runtime class method call
    return `$rt.callClassMethod(${JSON.stringify(kind.targetClassName)}, ${JSON.stringify(methodName)}, ${nargout}, [${args.join(", ")}])`;
  }

  // ── Unified resolution via resolveFunction ──────────────────────────
  const { argTypes, hasUnknown } = getArgTypesAndCheckUnknown(kind.args);
  const callSite = getCallSite(cg);
  const target = resolveFunction(
    kind.name,
    argTypes,
    callSite,
    cg.loweringCtx.functionIndex
  );

  if (target) {
    switch (target.kind) {
      case "localFunction": {
        // File-scoped local functions (main script, workspace file subfunctions,
        // or private file subfunctions) can never be overridden by class methods.

        // Class file local helpers: if unknown args, defer to runtime
        // (an unknown arg could be a class instance that changes dispatch)
        if (hasUnknown) {
          return emitDispatchUnknown(
            cg,
            kind.name,
            kind.nargout,
            args,
            callSite
          );
        }
        if (
          target.source.from === "main" ||
          target.source.from === "workspaceFile" ||
          target.source.from === "privateFile"
        ) {
          const jsId = cg.ensureSpecializedFunctionGenerated(
            kind.name,
            argTypes
          );
          if (jsId) return buildSpecializedCall(cg, jsId, kind.nargout, args);
        }
        // Known types — compile the local helper
        const jsId = cg.ensureSpecializedFunctionGenerated(kind.name, argTypes);
        if (jsId) return buildSpecializedCall(cg, jsId, kind.nargout, args);
        break;
      }

      case "classMethod": {
        if (hasUnknown) {
          return emitDispatchUnknown(
            cg,
            kind.name,
            kind.nargout,
            args,
            callSite
          );
        }
        // We only get here when a concrete ClassInstance arg was found
        // by the resolver, so compile the class method directly.
        const jsId = cg.ensureClassMethodGenerated(
          target.className,
          target.methodName,
          argTypes
        );
        if (jsId) return buildSpecializedCall(cg, jsId, kind.nargout, args);
        break;
      }

      case "privateFunction":
      case "workspaceFunction": {
        if (hasUnknown) {
          return emitDispatchUnknown(
            cg,
            kind.name,
            kind.nargout,
            args,
            callSite
          );
        }
        const jsId =
          target.kind === "privateFunction"
            ? cg.ensurePrivateFunctionGenerated(kind.name, argTypes)
            : cg.ensureWorkspaceFunctionGenerated(kind.name, argTypes);
        if (jsId) return buildSpecializedCall(cg, jsId, kind.nargout, args);
        break;
      }

      case "builtin": {
        // Skip native math inlining for JS user functions — they override the builtin.
        const isJsUserFunc = cg.loweringCtx.functionIndex.jsUserFunctions.has(
          kind.name
        );
        if (!isJsUserFunc) {
          const nativeMath = tryNativeMathCodegen(
            kind.name,
            kind.nargout,
            kind.args,
            args
          );
          if (nativeMath !== null) return nativeMath;
        }
        // Defer to runtime when any arg type is unknown — the argument
        // might be a class instance that overrides this builtin.
        if (hasUnknown) {
          return emitDispatchUnknown(
            cg,
            kind.name,
            kind.nargout,
            args,
            callSite
          );
        }
        return `${cg.useBuiltin(kind.name)}(${cg.nargoutOverride ?? kind.nargout}, [${args.join(", ")}])`;
      }

      case "workspaceClassConstructor":
        // Normally handled by genClassInstantiation, not genFuncCall
        break;
    }
  }

  // Function not found (or generation failed). If any arg is Unknown, defer
  // to runtime — the unknown value could be a class instance whose class
  // defines this method.
  if (hasUnknown) {
    return emitDispatchUnknown(cg, kind.name, kind.nargout, args, callSite);
  }

  // Truly unknown function — generate a runtime error
  return `(()=>{ throw $rt.error("Unknown function: ${kind.name}"); })()`;
}

function genClassInstantiation(
  cg: Codegen,
  kind: Extract<IRExprKind, { type: "ClassInstantiation" }>
): string {
  const args = kind.args.map(a => {
    const code = genExpr(cg, a);
    return isCslProducer(a) ? `...$rt.asCsl(${code})` : code;
  });
  const className = kind.className;

  const classInfo = cg.loweringCtx.getClassInfo(className);
  if (!classInfo) {
    // Unknown class — generate a throw
    return `(()=>{ throw $rt.error("Unknown class: ${className}"); })()`;
  }

  // When any argument has Unknown type, an Unknown arg might be a class
  // instance whose class has a method with the same name as this class
  // (e.g. domain(f) where f is a chebfun → should call @chebfun/domain,
  // not @domain constructor). Defer to runtime dispatch which checks class
  // methods before constructors.
  if (kind.args.length > 0) {
    const { hasUnknown } = getArgTypesAndCheckUnknown(kind.args);
    if (hasUnknown) {
      // Still register the class so the runtime can construct it if needed
      cg.ensureClassRegistered(className);
      return emitDispatchUnknown(
        cg,
        className,
        kind.nargout,
        args,
        getCallSite(cg)
      );
    }
  }

  // Ensure all class methods (including getters/setters) are registered
  cg.ensureClassRegistered(className);

  // Collect properties from the full inheritance chain
  const {
    propertyNames: allPropertyNames,
    propertyDefaults: allPropertyDefaults,
  } = collectClassProperties(cg.loweringCtx, className);
  const propsJson = JSON.stringify(allPropertyNames);
  const isHandleClass = cg.isHandleClass(classInfo);
  const defaultsArg = genPropertyDefaults(cg, className, allPropertyDefaults);

  if (classInfo.constructorName) {
    // Specialize and generate the constructor
    const selfType: ItemType = { kind: "ClassInstance", className };
    const argTypes: ItemType[] = [
      selfType,
      ...kind.args.map(a => itemTypeForExprKind(a.kind)),
    ];
    const jsId = cg.ensureClassMethodGenerated(
      className,
      classInfo.constructorName,
      argTypes
    );
    if (jsId) {
      // Create instance, then call constructor with it
      const instVar = cg.freshTemp("$inst");
      cg.emit(
        `var ${instVar} = $rt.createClassInstance(${JSON.stringify(className)}, ${propsJson}, ${defaultsArg}, ${isHandleClass});`
      );
      return `${jsId}(1, ${instVar}, ${args.join(", ")})`;
    }
  }

  // No constructor — just create instance with default properties
  return `$rt.createClassInstance(${JSON.stringify(className)}, ${propsJson}, ${defaultsArg}, ${isHandleClass})`;
}

// ── Native math code generation ─────────────────────────────────────────

/** Builtins that map to a single-arg Math.* call when the argument is Num. */
const NATIVE_MATH_1: Record<string, string> = {
  // Note: sqrt, asin, acos, log are NOT here because they can produce
  // complex results from real inputs (e.g., sqrt(-1) = 1i).
  abs: "Math.abs",
  floor: "Math.floor",
  ceil: "Math.ceil",
  fix: "Math.trunc",
  sin: "Math.sin",
  cos: "Math.cos",
  tan: "Math.tan",
  atan: "Math.atan",
  exp: "Math.exp",
  log2: "Math.log2",
  log10: "Math.log10",
  sign: "Math.sign",
};

/** Builtins that map to a two-arg Math.* call when both arguments are Num. */
const NATIVE_MATH_2: Record<string, string> = {
  // Note: max/min are NOT here because Math.max/Math.min propagate NaN,
  // but max/min ignore NaN (returning the non-NaN value).
  atan2: "Math.atan2",
  hypot: "Math.hypot",
};

function tryNativeMathCodegen(
  name: string,
  nargout: number,
  irArgs: IRExpr[],
  jsArgs: string[]
): string | null {
  // Skip native path when multiple outputs requested (e.g., [f,e]=log2(x))
  if (nargout > 1) return null;
  if (
    jsArgs.length === 1 &&
    itemTypeForExprKind(irArgs[0].kind).kind === "Number"
  ) {
    const fn = NATIVE_MATH_1[name];
    if (fn) return `${fn}(${jsArgs[0]})`;
  }
  if (
    jsArgs.length === 2 &&
    itemTypeForExprKind(irArgs[0].kind).kind === "Number" &&
    itemTypeForExprKind(irArgs[1].kind).kind === "Number"
  ) {
    const fn = NATIVE_MATH_2[name];
    if (fn) return `${fn}(${jsArgs[0]}, ${jsArgs[1]})`;
  }
  return null;
}

export function isOutputFunction(expr: IRExpr): boolean {
  if (expr.kind.type !== "FuncCall") return false;
  const outputFunctions = [
    "disp",
    "display",
    "fprintf",
    "warning",
    "assert",
    "tic",
  ];
  return outputFunctions.includes(expr.kind.name);
}

/** Check if a class defines a subsref method. */
export function hasSubsref(cg: Codegen, className: string): boolean {
  const info = cg.loweringCtx.getClassInfo(className);
  if (!info) return false;
  return info.methodNames.has("subsref");
}

/** Check if a class defines a subsasgn method. */
export function hasSubsasgn(cg: Codegen, className: string): boolean {
  const info = cg.loweringCtx.getClassInfo(className);
  if (!info) return false;
  return info.methodNames.has("subsasgn");
}
