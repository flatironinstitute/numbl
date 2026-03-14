/**
 * Return type inference utilities for function and method calls.
 */

import { type ItemType, IType } from "../lowering/itemTypes.js";
import { type IRExpr } from "../lowering/nodes.js";
import { itemTypeForExprKind } from "../lowering/nodeUtils.js";
import { findBuiltinBranch } from "../builtins";
import type { FunctionCandidate } from "../lowering/lowerExpr.js";
import type { LoweringContext } from "./loweringContext.js";

/**
 * Check if any argument is a CSL (comma-separated list) producer, meaning
 * it could expand to a variable number of args at runtime (e.g., varargin{:}).
 * When this is the case, we can't know the actual arg types at compile time.
 */
function hasCSLArg(args: IRExpr[]): boolean {
  for (const a of args) {
    if (a.kind.type === "IndexCell" && a.kind.indices.length === 1) {
      const idx = a.kind.indices[0];
      if (idx.kind.type === "Colon" || idx.kind.type === "Range") {
        return true;
      }
    }
  }
  return false;
}

/**
 * Extract output types from a specialized (lowered) function.
 * Shared logic for both regular functions and class methods.
 */
function extractOutputTypes(
  specialized: { outputs: { ty?: ItemType }[]; hasVarargout: boolean } | null,
  nargout: number
): ItemType {
  if (!specialized || specialized.outputs.length === 0) {
    return { kind: "Unknown" };
  }
  // When hasVarargout is true, the last output is `varargout` (a cell).
  // The caller receives varargout{1}, varargout{2}, etc. — not the cell
  // itself — so we can't trust the Cell type of varargout for those slots.
  const varargoutStart = specialized.hasVarargout
    ? specialized.outputs.length - 1
    : Infinity;

  if (nargout <= 1) {
    if (0 >= varargoutStart) {
      // The first output comes from varargout — type is unknown
      return { kind: "Unknown" };
    }
    if (specialized.outputs[0].ty) {
      return specialized.outputs[0].ty;
    }
    return { kind: "Unknown" };
  }
  if (nargout > 1 && specialized.outputs.length > 1) {
    const outputTypes = specialized.outputs
      .slice(0, nargout)
      .map((o, i) =>
        i >= varargoutStart
          ? ({ kind: "Unknown" } as ItemType)
          : (o.ty ?? ({ kind: "Unknown" } as ItemType))
      );
    if (specialized.hasVarargout) {
      while (outputTypes.length < nargout) {
        outputTypes.push({ kind: "Unknown" });
      }
    }
    return { kind: "MultipleOutputs", outputTypes };
  }
  if (0 >= varargoutStart) {
    return { kind: "Unknown" };
  }
  if (specialized.outputs[0].ty) {
    return specialized.outputs[0].ty;
  }
  return { kind: "Unknown" };
}

/**
 * Determine return type for function call candidates.
 */
export function determineReturnType(
  ctx: LoweringContext,
  name: string,
  candidates: FunctionCandidate[],
  args: IRExpr[],
  nargout: number
): ItemType {
  if (candidates.length !== 1) {
    return { kind: "Unknown" };
  }

  if (hasCSLArg(args)) {
    return { kind: "Unknown" };
  }

  const candidate = candidates[0];

  if (candidate.type === "userFunction") {
    const argTypes = args.map(a => itemTypeForExprKind(a.kind));
    const specialized =
      ctx.getOrLowerFunctionSpecialized(name, argTypes) ??
      ctx.getOrLowerWorkspaceFunctionSpecialized(name, argTypes);
    return extractOutputTypes(specialized, nargout);
  }

  // struct('field', val, ...): extract field names and value types from args
  if (candidate.type === "builtin" && name === "struct") {
    if (args.length === 0) return IType.struct();
    if (args.length % 2 === 0) {
      const fields: Record<string, ItemType> = {};
      let allCharKeys = true;
      for (let i = 0; i < args.length; i += 2) {
        const keyExpr = args[i].kind;
        if (keyExpr.type !== "Char") {
          allCharKeys = false;
          break;
        }
        const fieldName = keyExpr.value.replace(/^'|'$/g, "");
        fields[fieldName] = itemTypeForExprKind(args[i + 1].kind);
      }
      if (allCharKeys) return IType.struct(fields);
    }
    return IType.struct();
  }

  // arrayfun/cellfun: infer return type from the function argument's return type
  if (
    candidate.type === "builtin" &&
    (name === "arrayfun" || name === "cellfun")
  ) {
    if (args.length >= 2) {
      const funcArgType = itemTypeForExprKind(args[0].kind);
      if (
        funcArgType.kind === "Function" &&
        funcArgType.returns.kind !== "Unknown"
      ) {
        // Check for 'UniformOutput', false in remaining args
        let uniformOutput = true;
        for (let i = 2; i < args.length; i++) {
          const a = args[i].kind;
          if (
            a.type === "Char" &&
            a.value.replace(/^'|'$/g, "").toLowerCase() === "uniformoutput"
          ) {
            if (i + 1 < args.length) {
              const val = args[i + 1].kind;
              if (val.type === "Number" && val.value === "0") {
                uniformOutput = false;
              }
            }
          }
        }
        if (!uniformOutput) {
          return IType.cell("unknown");
        }
        const retKind = funcArgType.returns.kind;
        if (retKind === "Number" || retKind === "ComplexNumber") {
          return { kind: "Tensor", isComplex: retKind === "ComplexNumber" };
        }
        if (retKind === "Boolean") {
          return { kind: "Tensor", isLogical: true };
        }
      }
    }
    return { kind: "Unknown" };
  }

  // Builtin: try to infer from arg types.
  // But if any arg is Unknown, the call may be dispatched to a class method
  // at runtime instead of the builtin, so we can't trust the builtin's type.
  const argTypes = args.map(a => itemTypeForExprKind(a.kind));
  if (argTypes.some(t => t.kind === "Unknown")) {
    return { kind: "Unknown" };
  }
  const { branch } = findBuiltinBranch(name, argTypes, nargout);
  if (branch) {
    const result = branch.check(argTypes, nargout);
    if (result?.outputTypes?.length === 1) {
      return result.outputTypes[0];
    }
    if (result?.outputTypes && result.outputTypes.length > 1) {
      return { kind: "MultipleOutputs", outputTypes: result.outputTypes };
    }
  }
  return { kind: "Unknown" };
}

/**
 * Determine return type for a class method call by triggering specialized lowering.
 */
export function determineMethodReturnType(
  ctx: LoweringContext,
  className: string,
  methodName: string,
  argTypes: ItemType[],
  nargout: number
): ItemType {
  const specialized = ctx.getOrLowerClassMethodSpecialized(
    className,
    methodName,
    argTypes
  );
  return extractOutputTypes(specialized, nargout);
}
