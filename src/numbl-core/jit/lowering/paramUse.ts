/**
 * How a spec uses its tensor parameters, for passing them without a copy.
 *
 * MATLAB passes arguments by value, so every call into compiled code (from
 * the host, and between specs) copies its tensor arguments. A spec that
 * never writes into a parameter's buffer, and never stores it where a
 * later write could reach it, can instead read the caller's buffer
 * directly. `paramUse` reports, per parameter, what rules that out.
 */

import type { IRExpr, IRFunc, IRStmt } from "./ir.js";
import { isMultiElement } from "./types.js";
import { forEachSubExpr, forEachTopLevelExpr } from "./walk.js";

export interface ParamUse {
  /** The parameter's buffer may be written or aliased into a value that
   *  can be: it is the base of an indexed or member store, or it is placed
   *  uncopied into a struct, cell or handle. */
  readonly mutated: boolean;
  /** The parameter's variable is assigned somewhere in the body. */
  readonly reassigned: boolean;
  /** The parameter's variable is one of the outputs. */
  readonly returned: boolean;
}

/** Expression kinds that hold their direct `Var` operands by reference. */
function aliasedOperands(e: IRExpr): IRExpr[] {
  switch (e.kind) {
    case "CellLit":
      return e.elements;
    case "StructLit":
      return e.fields.map(f => f.value);
    case "HandleLit":
      return e.captures.map(c => c.value);
    case "HandleCaptureLoad":
    case "MemberLoad":
      return [e.base];
    default:
      return [];
  }
}

export function paramUse(fn: IRFunc): ParamUse[] {
  const index = new Map<string, number>();
  fn.cParams.forEach((c, i) => index.set(c, i));
  const mutated = fn.cParams.map(() => false);
  const reassigned = fn.cParams.map(() => false);
  const mark = (flags: boolean[], cName: string): void => {
    const i = index.get(cName);
    if (i !== undefined) flags[i] = true;
  };
  const markVar = (flags: boolean[], e: IRExpr): void => {
    if (e.kind === "Var") mark(flags, e.cName);
  };
  const visitExpr = (e: IRExpr): void => {
    forEachSubExpr(e, sub => {
      for (const op of aliasedOperands(sub)) markVar(mutated, op);
    });
  };
  const visit = (stmts: ReadonlyArray<IRStmt>): void => {
    for (const s of stmts) {
      forEachTopLevelExpr(s, visitExpr);
      switch (s.kind) {
        case "Assign":
          mark(reassigned, s.cName);
          break;
        case "For":
          mark(reassigned, s.cVar);
          visit(s.body);
          break;
        case "While":
          visit(s.body);
          break;
        case "If":
          visit(s.thenBody);
          visit(s.elseBody);
          break;
        case "MultiAssignCall":
          for (const o of s.outputs) {
            if (o.binding !== null) mark(reassigned, o.binding.cName);
          }
          break;
        case "MemberStore":
        case "CellIndexStore":
          mark(mutated, s.base.cName);
          markVar(mutated, s.rhs);
          break;
        case "IndexStore":
        case "IndexSliceStore":
          mark(mutated, s.base.cName);
          break;
      }
    }
  };
  visit(fn.body);
  const outputs = new Set(fn.cOutputs);
  return fn.cParams.map((c, i) => ({
    mutated: mutated[i],
    reassigned: reassigned[i],
    returned: outputs.has(c),
  }));
}

/** Tensor parameters a caller inside compiled JS may pass uncopied: the
 *  spec neither writes them nor returns them (`y = x` copies). */
export function jsCallBorrows(fn: IRFunc): boolean[] {
  return paramUse(fn).map(
    (u, i) => isMultiElement(fn.paramTypes[i]) && !u.mutated && !u.returned
  );
}

/** Tensor parameters the host may pass to an emitted JS entry uncopied.
 *  An output may still be such a parameter, unchanged; the host copies an
 *  output that shares a borrowed buffer. */
export function jsEntryBorrows(fn: IRFunc): boolean[] {
  return paramUse(fn).map(
    (u, i) => isMultiElement(fn.paramTypes[i]) && !u.mutated
  );
}

/** Tensor parameters the host may pass to an emitted C entry as pointers
 *  into its own buffers. C frees a parameter it owns when it is
 *  reassigned or goes out of scope, and returns an output's buffer to the
 *  host to free, so such a parameter must also be neither reassigned nor
 *  returned, and the entry must not call itself (its recursive calls pass
 *  copies, which it would then have to free). The emitter leaves these
 *  parameters unfreed. */
export function cEntryBorrows(fn: IRFunc, recursive: boolean): boolean[] {
  return paramUse(fn).map(
    (u, i) =>
      !recursive &&
      isMultiElement(fn.paramTypes[i]) &&
      !u.mutated &&
      !u.reassigned &&
      !u.returned
  );
}
