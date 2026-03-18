/**
 * Function definition code generation.
 *
 * Handles generating JS code for function definitions, including
 * parameter handling, local variable declarations, and return capture.
 */

import { typeToString } from "../lowering/itemTypes.js";
import { IRVariable } from "../lowering/loweringTypes.js";
import { IRStmt, type IRExpr } from "../lowering/nodes.js";
import type { Codegen } from "./codegen.js";
import { collectVarIds } from "./codegenHelpers.js";

/**
 * Generate JS code for a function definition.
 *
 * The body is generated first into a capture buffer before emitting
 * the function header.
 */
export function genFunctionDef(
  cg: Codegen,
  stmt: IRStmt & { type: "Function" },
  jsId?: string
): void {
  const effectiveJsId = jsId ?? `$fn_${cg.sanitizeName(stmt.functionId)}`;

  // Save and reset globalVarRefs so mappings from other functions don't leak.
  const savedGlobalVarRefs = new Map(cg.globalVarRefs);
  cg.globalVarRefs.clear();

  return cg.withCodegenContext(
    { currentFunctionJsId: effectiveJsId, nargoutOverride: null },
    () => {
      // Collect VarIds for this function (non-recursive — excludes nested
      // Function bodies, so each nested function handles its own vars)
      const ownVarIds = new Set<string>();
      for (const p of stmt.params) ownVarIds.add(p.id.id);
      for (const o of stmt.outputs) ownVarIds.add(o.id.id);
      collectVarIds(stmt.body, ownVarIds);

      // Pre-scan for Global statements to identify var IDs that should not
      // get local `var` declarations (they are accessed via $rt.$g).
      const globalVarIds = new Set<string>();
      for (const s of stmt.body) {
        if (s.type === "Global") {
          for (const v of s.vars) globalVarIds.add(v.variable.id.id);
        }
      }

      // Determine which vars belong to a parent scope (shared via closure)
      const parentShared =
        cg.sharedVarIdStack.length > 0
          ? cg.sharedVarIdStack[cg.sharedVarIdStack.length - 1]
          : null;

      const paramIdSet = new Set(stmt.params.map(p => p.id.id));
      // Local vars = own vars − params − parent-shared − globals
      const localVarIds = [...ownVarIds]
        .filter(
          id =>
            !paramIdSet.has(id) &&
            !parentShared?.has(id) &&
            !globalVarIds.has(id)
        )
        .sort();

      const resultVarName = cg.freshTemp("$ret");

      // Detect name-value group parameters from arguments blocks
      const nameValueGroups = new Map<
        string,
        Array<{
          field: string;
          defaultValue: (typeof stmt.argumentsBlocks)[0]["entries"][0]["defaultValue"];
        }>
      >();
      for (const block of stmt.argumentsBlocks) {
        if (block.kind !== "Input" && block.kind !== "Repeating") continue;
        for (const entry of block.entries) {
          const dotIdx = entry.name.indexOf(".");
          if (dotIdx < 0) continue;
          const prefix = entry.name.slice(0, dotIdx);
          const field = entry.name.slice(dotIdx + 1);
          if (!nameValueGroups.has(prefix)) nameValueGroups.set(prefix, []);
          nameValueGroups
            .get(prefix)!
            .push({ field, defaultValue: entry.defaultValue });
        }
      }

      // Build param list
      const paramItems: string[] = [];
      for (const p of stmt.params) {
        if (nameValueGroups.has(p.name)) {
          // Name-value group: use rest parameter to collect trailing args
          paramItems.push(`...${cg.varRef(p.id.id)}`);
        } else {
          const ref =
            p.name !== "varargin"
              ? cg.varRef(p.id.id)
              : `...${cg.varRef(p.id.id)}`;
          paramItems.push(ref);
        }
      }

      cg.currentFunctionOutputs.push({
        outputs: stmt.outputs.map(o => ({ variable: o as IRVariable })),
        resultVarName,
        hasVarargout: stmt.hasVarargout,
        originalName: stmt.originalName,
        uninitializedOutputIds: new Set(),
      });

      // Build varId → IRVariable map for type annotations
      const varById = new Map<string, IRVariable>();
      for (const p of stmt.params) varById.set(p.id.id, p);
      for (const o of stmt.outputs) varById.set(o.id.id, o);

      // ── Generate body into a capture buffer ──────────────────────
      // Use a nested emit capture so body lines are collected separately
      cg.pushEmitCapture();
      cg.pushIndent();

      // Convert varargin to cell
      const varargParam = stmt.params.find(p => p.name === "varargin");
      if (varargParam) {
        const ref = cg.varRef(varargParam.id.id);
        cg.emit(`${ref} = $rt.makeCell(${ref}, [1, ${ref}.length]);`);
      }

      // Declare local variables (skip those whose JS name collides with a parameter)
      const paramJsNames = new Set(stmt.params.map(p => cg.varRef(p.id.id)));
      for (const id of localVarIds) {
        if (paramJsNames.has(cg.varRef(id))) continue;
        const v = varById.get(id);
        cg.emit(
          `var ${cg.varRef(id)};${cg.typeComment(v ? cg.typeEnv.get(v.id) : undefined)}`
        );
      }
      cg.emit(`var ${resultVarName};`);

      // Handle default values from arguments blocks
      for (const block of stmt.argumentsBlocks) {
        if (block.kind !== "Input" && block.kind !== "Repeating") continue;
        for (const entry of block.entries) {
          if (entry.name.includes(".")) continue;
          if (entry.defaultValue) {
            const param = stmt.params.find(p => p.name === entry.name);
            if (param) {
              const ref = cg.varRef(param.id.id);
              const defaultVal = cg.genExpr(entry.defaultValue);
              cg.emit(`if (${ref} === undefined) ${ref} = ${defaultVal};`);
            }
          }
        }
      }

      // Handle name-value group parameters (opts.field = default)
      for (const [prefix, fields] of nameValueGroups) {
        const param = stmt.params.find(p => p.name === prefix);
        if (!param) continue;
        const ref = cg.varRef(param.id.id);
        // ref is a rest array of trailing name-value pairs.
        // Build a struct with defaults, overridden by any passed pairs.
        const defaultEntries = fields.map(({ field, defaultValue }) => {
          const val = defaultValue ? cg.genExpr(defaultValue) : "undefined";
          return `${JSON.stringify(field)}: ${val}`;
        });
        cg.emit(
          `${ref} = $rt.buildNameValueStruct(${ref}, {${defaultEntries.join(", ")}});`
        );
      }

      // Initialize output variables.
      // When a parameter is also an output (e.g., `function obj = method(obj)`),
      // we must $rt.share() it so the function works on a COW copy, preserving
      // value semantics where modifying the output doesn't affect the
      // caller's original variable.
      const paramByName = new Map(stmt.params.map(p => [p.name, p]));
      // const outputNames = new Set(stmt.outputs.map(o => o.name));
      for (const out of stmt.outputs) {
        if (paramIdSet.has(out.id.id)) {
          // Parameter IS the output — share it for value-class COW safety
          const ref = cg.varRef(out.id.id);
          cg.emit(`${ref} = $rt.share(${ref});`);
          continue;
        }
        // Also check by name: param and output may have different IDs but same name
        const matchedParam = paramByName.get(out.name);
        if (matchedParam && matchedParam.id.id !== out.id.id) {
          // Output has same name as param but different ID — share the param
          const paramRef = cg.varRef(matchedParam.id.id);
          const outRef = cg.varRef(out.id.id);
          if (paramRef !== outRef) {
            cg.emit(`${outRef} = $rt.share(${paramRef});`);
          } else {
            cg.emit(`${outRef} = $rt.share(${outRef});`);
          }
          continue;
        }
        if (!matchedParam) {
          if (out.name === "varargout") {
            cg.emit(`${cg.varRef(out.id.id)} = $rt.makeCell([], [0, 0]);`);
          } else {
            const top =
              cg.currentFunctionOutputs[cg.currentFunctionOutputs.length - 1];
            if (top) top.uninitializedOutputIds.add(out.id.id);
          }
        }
      }

      // Persistent variables
      const persistentVars: Array<{ varId: string; name: string }> = [];
      for (const s of stmt.body) {
        if (s.type === "Persistent") {
          for (const v of s.vars) {
            persistentVars.push({ varId: v.variable.id.id, name: v.name });
          }
        }
      }

      // If this function has nested function definitions, push own VarIds
      // onto sharedVarIdStack so children know not to re-declare them
      const hasNested = stmt.body.some(s => s.type === "Function");
      const nestedNames: string[] = [];
      if (hasNested) {
        const sharedForChildren = new Set(ownVarIds);
        if (parentShared) {
          for (const id of parentShared) sharedForChildren.add(id);
        }
        cg.sharedVarIdStack.push(sharedForChildren);

        // Track nested function names for direct call/handle resolution
        for (const s of stmt.body) {
          if (s.type === "Function") {
            nestedNames.push(s.originalName);
            cg.nestedFunctionNames.add(s.originalName);
          }
        }
      }

      // Check if this function body contains any eval() calls
      const hasEvalCall = bodyContainsEvalCall(stmt.body);

      if (hasEvalCall) {
        const evalVarMap = new Map<string, string>();
        const seenNames = new Set<string>();
        for (const id of ownVarIds) {
          if (globalVarIds.has(id)) continue;
          const name = varNameFromId(id);
          if (!seenNames.has(name)) {
            seenNames.add(name);
            evalVarMap.set(name, cg.varRef(id));
          }
        }
        cg.evalVarAccessorStack.push(evalVarMap);
      }

      // Register caller accessors for external-access vars declared in this function
      const externalVars = getExternalAccessVarsForFunction(
        cg,
        stmt.originalName
      );
      // Also pre-declare any external-access vars that aren't already in ownVarIds
      for (const name of externalVars) {
        const found = [...ownVarIds].some(id => varNameFromId(id) === name);
        if (!found) {
          // Define a new variable for this external-access name
          const v = cg.loweringCtx.defineVariable(name, undefined);
          ownVarIds.add(v.id.id);
          // Also need to emit a var declaration for it
          cg.emit(`var ${cg.varRef(v.id.id)};`);
        }
      }
      if (externalVars.size > 0) {
        const entries: string[] = [];
        const seen = new Set<string>();
        for (const id of ownVarIds) {
          const name = varNameFromId(id);
          if (externalVars.has(name) && !seen.has(name)) {
            seen.add(name);
            const jsRef = cg.varRef(id);
            entries.push(
              `${JSON.stringify(name)}: [() => ${jsRef}, ($v) => { ${jsRef} = $v; }]`
            );
          }
        }
        cg.emit(`$rt.pushCallerAccessors({${entries.join(", ")}});`);
      } else {
        cg.emit(`$rt.pushCallerAccessors(null);`);
      }
      cg.emit(`$rt.pushCallFrame(${JSON.stringify(stmt.originalName)});`);
      cg.emit(`try {`);
      cg.pushIndent();

      // Generate body
      cg.genStmts(stmt.body);

      if (hasEvalCall) {
        cg.evalVarAccessorStack.pop();
      }

      // Return capture
      const fnCtx =
        cg.currentFunctionOutputs[cg.currentFunctionOutputs.length - 1];
      emitReturnCapture(cg, fnCtx);
      cg.emit(`return ${resultVarName};`);

      cg.popIndent();
      cg.emit(`} catch ($e_stack) {`);
      cg.pushIndent();
      cg.emit(`$rt.annotateError($e_stack);`);
      cg.emit(`throw $e_stack;`);
      cg.popIndent();
      cg.emit(`} finally {`);
      cg.pushIndent();
      if (persistentVars.length > 0) {
        const funcId = JSON.stringify(effectiveJsId);
        for (const v of persistentVars) {
          cg.emit(
            `$rt.setPersistent(${funcId}, ${JSON.stringify(v.name)}, ${cg.varRef(v.varId)});`
          );
        }
      }
      cg.emit(`$rt.popCallFrame();`);
      cg.emit(`$rt.popCallerAccessors();`);
      cg.popIndent();
      cg.emit(`}`);

      // Clean up nested function tracking
      if (hasNested) {
        for (const name of nestedNames) {
          cg.nestedFunctionNames.delete(name);
        }
        cg.sharedVarIdStack.pop();
      }

      cg.popIndent();

      // ── Pop body buffer ───────────────────────────────────────
      const bodyLines = cg.popEmitCaptureLines();

      // ── Emit type signature comment ────────────────────────────
      const getType = (v: IRVariable) => cg.typeEnv.get(v.id);
      const paramSig = stmt.params
        .map(p => `${p.name}: ${getType(p) ? typeToString(getType(p)!) : "?"}`)
        .join(", ");
      const outSig = stmt.outputs
        .map(o => `${o.name}: ${getType(o) ? typeToString(getType(o)!) : "?"}`)
        .join(", ");
      cg.emit(`/* [${outSig}] = ${stmt.originalName}(${paramSig}) */`);

      // ── Emit function header ───────────────────────────────────
      cg.emit(
        `function ${effectiveJsId}(${["$nargout", ...paramItems].join(", ")}) {`
      );

      // ── Emit captured body lines ───────────────────────────────
      for (const line of bodyLines) {
        cg.emitRaw(line);
      }

      cg.emit(`}`);

      cg.currentFunctionOutputs.pop();

      // Restore globalVarRefs so sibling function compilations are clean.
      cg.globalVarRefs = savedGlobalVarRefs;
    }
  );
}

/**
 * Emit JS code for capturing return values from a function.
 */
export function emitReturnCapture(
  cg: Codegen,
  fnCtx: {
    outputs: { variable: IRVariable }[];
    resultVarName: string;
    hasVarargout: boolean;
    originalName: string;
    uninitializedOutputIds: Set<string>;
  }
): void {
  // Check uninitialized outputs
  if (fnCtx.uninitializedOutputIds.size > 0) {
    for (let i = 0; i < fnCtx.outputs.length; i++) {
      const out = fnCtx.outputs[i];
      if (fnCtx.uninitializedOutputIds.has(out.variable.id.id)) {
        const varRef = cg.varRef(out.variable.id.id);
        cg.emit(
          `if ($nargout >= ${i + 1} && ${varRef} === undefined) throw $rt.error(${JSON.stringify(`Output argument '${out.variable.name}' (and maybe others) not assigned during call to '${fnCtx.originalName}'`)});`
        );
      }
    }
  }
  if (fnCtx.hasVarargout) {
    const fixedOuts = fnCtx.outputs.filter(
      o => o.variable.name !== "varargout"
    );
    const varargoutRef = cg.varRef(
      fnCtx.outputs[fnCtx.outputs.length - 1].variable.id.id
    );
    if (fixedOuts.length === 0) {
      cg.emit(
        `${fnCtx.resultVarName} = $nargout <= 1 ? ${varargoutRef}.data[0] : [...${varargoutRef}.data.slice(0, $nargout)];`
      );
    } else {
      const fixedRefs = fixedOuts
        .map(o => cg.varRef(o.variable.id.id))
        .join(", ");
      const nFixed = fixedOuts.length;
      cg.emit(
        `${fnCtx.resultVarName} = $nargout <= 1 ? ${cg.varRef(fixedOuts[0].variable.id.id)} : [${fixedRefs}, ...${varargoutRef}.data.slice(0, $nargout - ${nFixed})];`
      );
    }
  } else if (fnCtx.outputs.length === 1) {
    cg.emit(
      `${fnCtx.resultVarName} = ${cg.varRef(fnCtx.outputs[0].variable.id.id)};`
    );
  } else if (fnCtx.outputs.length > 1) {
    const firstOut = cg.varRef(fnCtx.outputs[0].variable.id.id);
    const outs = fnCtx.outputs.map(o => cg.varRef(o.variable.id.id)).join(", ");
    cg.emit(
      `${fnCtx.resultVarName} = $nargout <= 1 ? ${firstOut} : [${outs}];`
    );
  }
}

/**
 * Extract the MATLAB variable name from a VarId (format: "name_N").
 */
function varNameFromId(id: string): string {
  const lastUnderscore = id.lastIndexOf("_");
  return lastUnderscore > 0 ? id.substring(0, lastUnderscore) : id;
}

/** Check if any IR statement in the body contains a FuncCall to "eval". */
function bodyContainsEvalCall(stmts: IRStmt[]): boolean {
  for (const s of stmts) {
    if (s.type === "Function") continue;
    if (stmtContainsEvalCall(s)) return true;
  }
  return false;
}

function stmtContainsEvalCall(stmt: IRStmt): boolean {
  switch (stmt.type) {
    case "ExprStmt":
    case "Assign":
    case "MultiAssign":
    case "AssignLValue":
      return exprContainsEvalCall(stmt.expr);
    case "If":
      return (
        exprContainsEvalCall(stmt.cond) ||
        bodyContainsEvalCall(stmt.thenBody) ||
        stmt.elseifBlocks.some(
          c => exprContainsEvalCall(c.cond) || bodyContainsEvalCall(c.body)
        ) ||
        (stmt.elseBody !== null && bodyContainsEvalCall(stmt.elseBody))
      );
    case "For":
      return exprContainsEvalCall(stmt.expr) || bodyContainsEvalCall(stmt.body);
    case "While":
      return exprContainsEvalCall(stmt.cond) || bodyContainsEvalCall(stmt.body);
    case "Switch":
      return (
        exprContainsEvalCall(stmt.expr) ||
        stmt.cases.some(
          c => exprContainsEvalCall(c.value) || bodyContainsEvalCall(c.body)
        ) ||
        (stmt.otherwise !== null && bodyContainsEvalCall(stmt.otherwise))
      );
    case "TryCatch":
      return (
        bodyContainsEvalCall(stmt.tryBody) ||
        bodyContainsEvalCall(stmt.catchBody)
      );
    default:
      return false;
  }
}

function exprContainsEvalCall(expr: IRExpr): boolean {
  const k = expr.kind;
  if (k.type === "FuncCall") {
    if (k.name === "eval") return true;
    return (
      k.args.some(a => exprContainsEvalCall(a)) ||
      (k.instanceBase !== undefined &&
        k.instanceBase !== null &&
        exprContainsEvalCall(k.instanceBase))
    );
  }
  if (k.type === "Binary")
    return exprContainsEvalCall(k.left) || exprContainsEvalCall(k.right);
  if (k.type === "Unary") return exprContainsEvalCall(k.operand);
  if (k.type === "Range")
    return (
      exprContainsEvalCall(k.start) ||
      (k.step !== null && exprContainsEvalCall(k.step)) ||
      exprContainsEvalCall(k.end)
    );
  if (k.type === "Index" || k.type === "IndexCell")
    return (
      exprContainsEvalCall(k.base) ||
      k.indices.some(i => exprContainsEvalCall(i))
    );
  if (k.type === "Member") return exprContainsEvalCall(k.base);
  if (k.type === "MemberDynamic")
    return exprContainsEvalCall(k.base) || exprContainsEvalCall(k.nameExpr);
  if (k.type === "MethodCall")
    return (
      k.name === "eval" ||
      exprContainsEvalCall(k.base) ||
      k.args.some(a => exprContainsEvalCall(a))
    );
  if (k.type === "AnonFunc") return exprContainsEvalCall(k.body);
  if (k.type === "Tensor" || k.type === "Cell")
    return k.rows.some(row => row.some(e => exprContainsEvalCall(e)));
  if (k.type === "ClassInstantiation")
    return k.args.some(a => exprContainsEvalCall(a));
  if (k.type === "SuperConstructorCall")
    return k.args.some(a => exprContainsEvalCall(a));
  return false;
}

/**
 * Look up external-access variable names for a function from the
 * per-file directives stored in the workspace registry.
 */
function getExternalAccessVarsForFunction(
  cg: Codegen,
  functionName: string
): Set<string> {
  const fileName = cg.loweringCtx.mainFileName;
  const directives = cg.loweringCtx.registry.externalAccessByFile.get(fileName);
  if (!directives) return new Set();
  return directives.functionScope.get(functionName) ?? new Set();
}
