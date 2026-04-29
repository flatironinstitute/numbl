/**
 * c-jit-chain codegen — combine N adjacent fusable Assigns into one
 * C kernel with a single i-loop.
 *
 * Each per-name role is decided by walking the chain in source order:
 *   - input-tensor   : first occurrence is RHS, env says it's a tensor
 *   - input-scalar   : first occurrence is RHS, env says it's a scalar
 *   - local-tensor   : first occurrence is LHS (writes the value
 *                      without reading any prior value); becomes a
 *                      per-iter `double` local
 *   - input+local    : first occurrence is RHS but the chain also
 *                      writes it — read from input pointer, then
 *                      writes update the local
 *
 * Live-outs (any LHS of any Assign in the chain) get their own `double *`
 * out parameter and a writeback at the end of the loop body.
 *
 * Today: every RHS expression must be element-wise on tensors of the
 * chain's numel (scalars broadcast). The executor's propose() rejects
 * chains where this can't be guaranteed.
 *
 * ABI:
 *   void <fnName>(
 *       long n,
 *       double *out_<name1>, double *out_<name2>, ...,   // live-outs
 *       const double *in_<nameA>, const double *in_<nameB>, ...,
 *       double <nameC>, double <nameD>, ...);
 */

import {
  BinaryOperation,
  type Expr,
  UnaryOperation,
} from "../../parser/types.js";
import type { ChainAnalysis } from "./chainPass.js";

/** Per-name role (decided at compile time using env types). */
export type NameRole =
  | { kind: "input-tensor" }
  | { kind: "input-scalar" }
  | { kind: "local-tensor"; readsFromInput: boolean };

/** Resolved per-name info carried from compile() into codegen. */
export interface ChainResolution {
  /** Role for every name referenced in the chain. */
  readonly roles: ReadonlyMap<string, NameRole>;
  /** Names whose final value is live-out (written somewhere in the
   *  chain). For Phase A: every Assign LHS is conservatively
   *  live-out. */
  readonly liveOuts: readonly string[];
  /** Tensor input names in stable order (for koffi declaration). */
  readonly tensorInputs: readonly string[];
  /** Scalar input names in stable order. */
  readonly scalarInputs: readonly string[];
}

/** Decide each name's role from the AST + a "is this var a tensor?"
 *  classifier. Returns null if any per-stmt structure is incompatible
 *  with the codegen — caller should bail. */
export function resolveChain(
  cls: ChainAnalysis,
  isTensor: (name: string) => boolean | null
): ChainResolution | null {
  const roles = new Map<string, NameRole>();
  const liveOutSet = new Set<string>();
  const liveOutsOrder: string[] = [];
  const tensorInputs: string[] = [];
  const scalarInputs: string[] = [];

  // Walk stmts in source order. For each name appearing, record its
  // first-occurrence kind (LHS or RHS) and runtime encoding.
  const writtenSoFar = new Set<string>();
  for (const a of cls.assigns) {
    // RHS reads first (so a `r = r + 1;` correctly classifies r as
    // input-then-local, not local-only).
    const rhsNames = new Set<string>();
    collectRhsIdents(a.expr, rhsNames);
    for (const name of rhsNames) {
      if (roles.has(name)) continue;
      if (writtenSoFar.has(name)) continue;
      const isT = isTensor(name);
      if (isT === null) return null;
      if (isT) {
        roles.set(name, { kind: "input-tensor" });
        tensorInputs.push(name);
      } else {
        roles.set(name, { kind: "input-scalar" });
        scalarInputs.push(name);
      }
    }

    // LHS: this name becomes a local-tensor. If it was already an
    // input, it stays as input-tensor but gets `readsFromInput: true`.
    const lhs = a.name;
    const prev = roles.get(lhs);
    if (prev) {
      if (prev.kind === "input-tensor") {
        roles.set(lhs, { kind: "local-tensor", readsFromInput: true });
      } else if (prev.kind === "input-scalar") {
        // Trying to overwrite a scalar with a tensor — would force a
        // shape change. Bail.
        return null;
      } else {
        // already local-tensor — keep as-is
      }
    } else {
      roles.set(lhs, { kind: "local-tensor", readsFromInput: false });
    }
    writtenSoFar.add(lhs);

    if (!liveOutSet.has(lhs)) {
      liveOutSet.add(lhs);
      liveOutsOrder.push(lhs);
    }
  }

  return {
    roles,
    liveOuts: liveOutsOrder,
    tensorInputs,
    scalarInputs,
  };
}

function collectRhsIdents(e: Expr, out: Set<string>): void {
  switch (e.type) {
    case "Ident":
      out.add(e.name);
      return;
    case "Number":
      return;
    case "Binary":
      collectRhsIdents(e.left, out);
      collectRhsIdents(e.right, out);
      return;
    case "Unary":
      collectRhsIdents(e.operand, out);
      return;
    case "FuncCall":
      for (const a of e.args) collectRhsIdents(a, out);
      return;
  }
}

/** Build the koffi function declaration string. */
export function buildChainDeclaration(
  fnName: string,
  res: ChainResolution
): string {
  const params: string[] = ["long n"];
  for (const name of res.liveOuts) {
    params.push(`double *out_${name}`);
  }
  for (const name of res.tensorInputs) {
    params.push(`const double *in_${name}`);
  }
  for (const name of res.scalarInputs) {
    params.push(`double ${name}`);
  }
  return `void ${fnName}(${params.join(", ")})`;
}

/** Emit the C source. */
export function generateChainCSource(
  fnName: string,
  cls: ChainAnalysis,
  res: ChainResolution
): string {
  const lines: string[] = [];
  lines.push(`#include <math.h>`);
  lines.push(``);
  // Use `restrict` on every output pointer; outputs are always
  // fresh Float64Arrays allocated by the executor, so no aliasing
  // with inputs is possible.
  const params: string[] = ["long n"];
  for (const name of res.liveOuts) {
    params.push(`double *restrict out_${name}`);
  }
  for (const name of res.tensorInputs) {
    params.push(`const double *in_${name}`);
  }
  for (const name of res.scalarInputs) {
    params.push(`double ${name}`);
  }
  lines.push(`void ${fnName}(${params.join(", ")}) {`);
  lines.push(`  #pragma omp simd`);
  lines.push(`  for (long i = 0; i < n; i++) {`);

  // Declare every local-tensor name as a `double` local. If the role
  // says `readsFromInput`, initialize from the input pointer.
  const declared = new Set<string>();
  for (const [name, role] of res.roles) {
    if (role.kind !== "local-tensor") continue;
    declared.add(name);
    if (role.readsFromInput) {
      lines.push(`    double ${name} = in_${name}[i];`);
    } else {
      lines.push(`    double ${name};`);
    }
  }

  for (const a of cls.assigns) {
    lines.push(`    ${a.name} = ${emitExpr(a.expr, res)};`);
  }

  // Writebacks for every live-out.
  for (const name of res.liveOuts) {
    lines.push(`    out_${name}[i] = ${name};`);
  }
  lines.push(`  }`);
  lines.push(`}`);
  void declared;
  return lines.join("\n") + "\n";
}

function emitExpr(e: Expr, res: ChainResolution): string {
  switch (e.type) {
    case "Number":
      return formatDouble(parseFloat(e.value));
    case "Ident": {
      const role = res.roles.get(e.name);
      if (!role) throw new Error(`chain codegen: unknown name ${e.name}`);
      if (role.kind === "input-tensor") return `in_${e.name}[i]`;
      if (role.kind === "input-scalar") return e.name;
      // local-tensor — read the local
      return e.name;
    }
    case "Binary":
      return emitBinary(e, res);
    case "Unary": {
      const x = emitExpr(e.operand, res);
      switch (e.op) {
        case UnaryOperation.Plus:
          return `(+${x})`;
        case UnaryOperation.Minus:
          return `(-${x})`;
        default:
          throw new Error(`chain codegen: unsupported unary op ${e.op}`);
      }
    }
    case "FuncCall": {
      const name = e.name === "abs" ? "fabs" : e.name;
      const arg = emitExpr(e.args[0], res);
      return `${name}(${arg})`;
    }
    default:
      throw new Error(
        `chain codegen: unsupported expr ${(e as { type: string }).type}`
      );
  }
}

function emitBinary(
  e: Expr & { type: "Binary" },
  res: ChainResolution
): string {
  const l = emitExpr(e.left, res);
  const r = emitExpr(e.right, res);
  switch (e.op) {
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
    default:
      throw new Error(`chain codegen: unsupported binary op ${e.op}`);
  }
}

function formatDouble(v: number): string {
  if (Number.isNaN(v)) return "((double)NAN)";
  if (v === Infinity) return "((double)INFINITY)";
  if (v === -Infinity) return "(-(double)INFINITY)";
  if (Number.isInteger(v)) return `${v}.0`;
  const s = String(v);
  if (/[.eE]/.test(s)) return s;
  return `${s}.0`;
}
