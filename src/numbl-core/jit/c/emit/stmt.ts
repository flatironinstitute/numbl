/**
 * Statement-level emission — the top-level dispatch for every JitStmt.
 *
 * Exports:
 *   - `emitStmts(lines, stmts, indent, ctx)` — public entry point; the
 *     outer function body is fed through this. Runs the fusion pass
 *     when `ctx.fuse` is set, otherwise emits one statement at a time.
 *   - `emitStmt` — per-statement dispatch used internally (also used
 *     inside If / For / While bodies via `emitStmts`).
 *   - `withPendingStmts` — scoped helper for nested expressions that
 *     need to hoist declarations above the calling statement.
 *
 * This module is thin on purpose: it routes to the specialized emitters
 * in [./scalar.ts](./scalar.ts), [./complexScalar.ts](./complexScalar.ts),
 * [./tensor.ts](./tensor.ts), [./assign.ts](./assign.ts),
 * [./userCall.ts](./userCall.ts), and [./fused.ts](./fused.ts).
 */
import { type JitStmt, type JitExpr, isKnownInteger } from "../../jitTypes.js";
import { findFusibleChains } from "../../fusion.js";
import {
  getTensorReductionOp,
  isComplexScalarVar,
  isTensorVar,
  mangle,
  mangleIm,
  tensorD0,
  tensorD1,
  tensorData,
  tensorLen,
  type EmitCtx,
} from "../context.js";
import { emitExpr, emitTruthiness } from "./scalar.js";
import { emitComplex } from "./complexScalar.js";
import { emitTensorAssign, emitReductionOfTensorExpr } from "./assign.js";
import { emitFusedChain } from "./fused.js";

/** Evaluate `fn` with `ctx.pendingStmts` set to `{lines, indent}` so any
 *  UserCall / RangeSliceRead nested inside can hoist its decl+call into
 *  `lines` ahead of the calling statement. Restores the prior value on
 *  exit (safe even if nested save/restore frames stack). */
export function withPendingStmts<T>(
  ctx: EmitCtx,
  lines: string[],
  indent: string,
  fn: () => T
): T {
  const prev = ctx.pendingStmts;
  ctx.pendingStmts = { lines, indent };
  try {
    return fn();
  } finally {
    ctx.pendingStmts = prev;
  }
}

export function emitStmts(
  lines: string[],
  stmts: JitStmt[],
  indent: string,
  ctx: EmitCtx
): void {
  if (!ctx.fuse) {
    for (const s of stmts) emitStmt(lines, s, indent, ctx);
    return;
  }

  const chains = findFusibleChains(
    stmts,
    ctx.cls.paramTensorNames,
    ctx.cls.tensorVars
  );

  const coveredByChain = new Map<
    number,
    { chain: ReturnType<typeof findFusibleChains>[0] }
  >();
  for (const chain of chains) {
    coveredByChain.set(chain.startIdx, { chain });
  }

  // Pre-compute complex tensor names and complex scalar vars. The fused
  // emitter dispatches on these to pick the real or complex per-element
  // path; the sets are loop-invariant over the chain iteration below.
  const dynamicOutputNames = new Set<string>();
  const complexTensorNames = new Set<string>();
  for (const [name, m] of ctx.cls.meta) {
    if (m.isDynamicOutput) dynamicOutputNames.add(name);
    if (m.isComplex) complexTensorNames.add(name);
  }

  let i = 0;
  while (i < stmts.length) {
    const entry = coveredByChain.get(i);
    if (entry) {
      emitFusedChain(
        lines,
        indent,
        entry.chain,
        ctx.cls.tensorVars,
        ctx.cls.paramTensorNames,
        ctx.cls.outputTensorNames,
        ctx.cls.localTensorNames,
        dynamicOutputNames,
        complexTensorNames,
        ctx.complexScalarVars,
        ctx.openmp
      );
      i += entry.chain.length;
    } else {
      emitStmt(lines, stmts[i], indent, ctx);
      i++;
    }
  }
}

function emitStmt(
  lines: string[],
  stmt: JitStmt,
  indent: string,
  ctx: EmitCtx
): void {
  switch (stmt.tag) {
    case "Assign": {
      if (stmt.expr.jitType.kind === "tensor" && isTensorVar(ctx, stmt.name)) {
        emitTensorAssign(lines, indent, stmt.name, stmt.expr, ctx);
        return;
      }
      if (
        stmt.expr.tag === "Call" &&
        getTensorReductionOp(stmt.expr.name) !== undefined &&
        stmt.expr.args[0]?.jitType.kind === "tensor" &&
        stmt.expr.args[0].tag !== "Var"
      ) {
        emitReductionOfTensorExpr(lines, indent, stmt.name, stmt.expr, ctx);
        return;
      }
      // Complex scalar assign: emit pair into pendingStmts then write to
      // the two paired locals.
      if (isComplexScalarVar(ctx, stmt.name)) {
        const pair = withPendingStmts(ctx, lines, indent, () =>
          emitComplex(stmt.expr, ctx)
        );
        lines.push(`${indent}${mangle(stmt.name)} = ${pair.re};`);
        lines.push(`${indent}${mangleIm(stmt.name)} = ${pair.im};`);
        return;
      }
      const rhs = withPendingStmts(ctx, lines, indent, () =>
        emitExpr(stmt.expr, ctx)
      );
      lines.push(`${indent}${mangle(stmt.name)} = ${rhs};`);
      return;
    }

    case "ExprStmt": {
      const code = withPendingStmts(ctx, lines, indent, () =>
        emitExpr(stmt.expr, ctx)
      );
      lines.push(`${indent}(void)(${code});`);
      return;
    }

    case "AssignIndex": {
      const n = stmt.indices.length;
      if (n < 1 || n > 3) {
        throw new Error(
          `C-JIT codegen: AssignIndex arity ${n} unsupported (only 1D/2D/3D)`
        );
      }
      if (!isTensorVar(ctx, stmt.baseName)) {
        throw new Error(
          `C-JIT codegen: AssignIndex base '${stmt.baseName}' is not a tensor var`
        );
      }
      ctx.needsErrorFlag = true;
      const name = stmt.baseName;
      const data = tensorData(name);
      const len = tensorLen(name);
      // Allow UserCall / RangeSliceRead in the index or value by letting
      // them prepend helper statements (decl + call) before the
      // numbl_set*r_h line.
      const { idxCodes, v } = withPendingStmts(ctx, lines, indent, () => ({
        idxCodes: stmt.indices.map(idx => {
          let s = emitExpr(idx, ctx);
          if (!isKnownInteger(idx.jitType)) s = `round(${s})`;
          return s;
        }),
        v: emitExpr(stmt.value, ctx),
      }));
      if (n === 1) {
        lines.push(
          `${indent}numbl_set1r_h(${data}, (size_t)${len}, ${idxCodes[0]}, ${v}, __err_flag);`
        );
      } else if (n === 2) {
        lines.push(
          `${indent}numbl_set2r_h(${data}, (size_t)${len}, (size_t)${tensorD0(name)}, ${idxCodes[0]}, ${idxCodes[1]}, ${v}, __err_flag);`
        );
      } else {
        lines.push(
          `${indent}numbl_set3r_h(${data}, (size_t)${len}, (size_t)${tensorD0(name)}, (size_t)${tensorD1(name)}, ${idxCodes[0]}, ${idxCodes[1]}, ${idxCodes[2]}, ${v}, __err_flag);`
        );
      }
      return;
    }

    case "AssignIndexRange": {
      if (!isTensorVar(ctx, stmt.baseName)) {
        throw new Error(
          `C-JIT codegen: AssignIndexRange base '${stmt.baseName}' is not a tensor var`
        );
      }
      if (!isTensorVar(ctx, stmt.srcBaseName)) {
        throw new Error(
          `C-JIT codegen: AssignIndexRange src '${stmt.srcBaseName}' is not a tensor var`
        );
      }
      ctx.needsErrorFlag = true;
      const dData = tensorData(stmt.baseName);
      const dLen = tensorLen(stmt.baseName);
      const sData = tensorData(stmt.srcBaseName);
      const sLen = tensorLen(stmt.srcBaseName);
      const { dStart, dEnd, srcStart, srcEnd } = withPendingStmts(
        ctx,
        lines,
        indent,
        () => {
          const emitRI = (e: JitExpr): string => {
            let code = emitExpr(e, ctx);
            if (!isKnownInteger(e.jitType)) code = `round(${code})`;
            return code;
          };
          return {
            dStart: emitRI(stmt.dstStart),
            dEnd: emitRI(stmt.dstEnd),
            srcStart: stmt.srcStart !== null ? emitRI(stmt.srcStart) : `1.0`,
            srcEnd:
              stmt.srcEnd !== null ? emitRI(stmt.srcEnd) : `(double)${sLen}`,
          };
        }
      );
      lines.push(
        `${indent}numbl_setRange1r_h(${dData}, (size_t)${dLen}, ${dStart}, ${dEnd}, ${sData}, (size_t)${sLen}, ${srcStart}, ${srcEnd}, __err_flag);`
      );
      return;
    }

    case "AssignIndexCol": {
      if (!isTensorVar(ctx, stmt.baseName)) {
        throw new Error(
          `C-JIT codegen: AssignIndexCol base '${stmt.baseName}' is not a tensor var`
        );
      }
      if (!isTensorVar(ctx, stmt.srcBaseName)) {
        throw new Error(
          `C-JIT codegen: AssignIndexCol src '${stmt.srcBaseName}' is not a tensor var`
        );
      }
      ctx.needsErrorFlag = true;
      const dData = tensorData(stmt.baseName);
      const dLen = tensorLen(stmt.baseName);
      const dRows = tensorD0(stmt.baseName);
      const sData = tensorData(stmt.srcBaseName);
      const sLen = tensorLen(stmt.srcBaseName);
      const colCode = withPendingStmts(ctx, lines, indent, () => {
        let code = emitExpr(stmt.colIndex, ctx);
        if (!isKnownInteger(stmt.colIndex.jitType)) code = `round(${code})`;
        return code;
      });
      lines.push(
        `${indent}numbl_setCol2r_h(${dData}, (size_t)${dRows}, (size_t)${dLen}, ${colCode}, ${sData}, (size_t)${sLen}, __err_flag);`
      );
      return;
    }

    case "If": {
      const condCode = withPendingStmts(ctx, lines, indent, () =>
        emitTruthiness(stmt.cond, ctx)
      );
      lines.push(`${indent}if (${condCode}) {`);
      emitStmts(lines, stmt.thenBody, indent + "  ", ctx);
      for (const eib of stmt.elseifBlocks) {
        const eibCondCode = withPendingStmts(ctx, lines, indent, () =>
          emitTruthiness(eib.cond, ctx)
        );
        lines.push(`${indent}} else if (${eibCondCode}) {`);
        emitStmts(lines, eib.body, indent + "  ", ctx);
      }
      if (stmt.elseBody) {
        lines.push(`${indent}} else {`);
        emitStmts(lines, stmt.elseBody, indent + "  ", ctx);
      }
      lines.push(`${indent}}`);
      return;
    }

    case "For": {
      const v = mangle(stmt.varName);
      const t = `__t${++ctx.tmp.n}`;
      // MATLAB evaluates start / end / step once at loop entry. Hoisting
      // once matches that; C's for header re-reads end/step each iter,
      // which is semantically neutral against a local already assigned.
      const { start, end, step } = withPendingStmts(ctx, lines, indent, () => ({
        start: emitExpr(stmt.start, ctx),
        end: emitExpr(stmt.end, ctx),
        step: stmt.step ? emitExpr(stmt.step, ctx) : "1.0",
      }));
      if (stmt.step) {
        lines.push(
          `${indent}for (double ${t} = ${start}; (${step}) != 0.0 && ((${step}) > 0.0 ? ${t} <= (${end}) : ${t} >= (${end})); ${t} += (${step})) {`
        );
      } else {
        lines.push(
          `${indent}for (double ${t} = ${start}; ${t} <= (${end}); ${t} += 1.0) {`
        );
      }
      lines.push(`${indent}  ${v} = ${t};`);
      emitStmts(lines, stmt.body, indent + "  ", ctx);
      lines.push(`${indent}}`);
      return;
    }

    case "While":
      // NO pendingStmts here: While's cond is re-evaluated every iter,
      // so a UserCall / RangeSliceRead in it can't be hoisted once —
      // those cases throw "outside statement context" and bail.
      lines.push(`${indent}while (${emitTruthiness(stmt.cond, ctx)}) {`);
      emitStmts(lines, stmt.body, indent + "  ", ctx);
      lines.push(`${indent}}`);
      return;

    case "Break":
      lines.push(`${indent}break;`);
      return;

    case "Continue":
      lines.push(`${indent}continue;`);
      return;

    case "Return":
      // No-op: return is handled by the epilogue.
      return;

    case "SetLoc":
      return;

    case "AssertCJit":
      // C-JIT codegen reached → assertion satisfied, elide.
      return;

    default:
      throw new Error(
        `C-JIT codegen: unsupported stmt ${(stmt as JitStmt).tag}`
      );
  }
}
