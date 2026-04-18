/**
 * JIT IR → pure C code generation (koffi path).
 *
 * Orchestration only: this file wires the classify / ABI / emit pieces
 * together and assembles the final C source (headers + per-callee
 * static functions + outer function).
 *
 *   classify.ts     — TensorMeta / analyzeTensorUsage, the single pass
 *                     feeding every downstream decision.
 *   abi.ts          — AbiSlot / CParamDesc / COutputDesc, buildAbiSlots.
 *                     The one schema walked by both signature and JS.
 *   emit.ts         — per-statement / per-expression C emission, reads
 *                     ctx.cls for every classification decision.
 *   codegenCtx.ts   — EmitCtx + shared name/opcode helpers.
 *
 * UserCall support: when a feasible user-defined function is called
 * from the outer body, its lowered IR is already in `generatedIRBodies`
 * (populated by `lowerUserFuncCall` in jitLower.ts). We emit each
 * reachable callee as a `static void jit_<jitName>(...)` in the same
 * .c file, in post-order so callees are defined before callers. The
 * shared `__err_flag` pointer flows from outer to every callee.
 */
import { type JitExpr, type JitStmt, type JitType } from "../jitTypes.js";
import type { GeneratedFn } from "../jitLower.js";
import { analyzeTensorUsage } from "./classify.js";
import {
  NUMBL_JIT_RT_REQUIRED_VERSION,
  spaceBeforeName,
  mangle,
  tensorD0,
  tensorD1,
  tensorData,
  tensorLen,
  scratchData,
  scratchLen,
  type EmitCtx,
} from "./codegenCtx.js";
import {
  buildAbiSlots,
  type AbiSlot,
  type CParamDesc,
  type COutputDesc,
} from "./abi.js";
import { emitStmts } from "./emit.js";

export type { AbiSlot, AbiSlotKind } from "./abi.js";
export type { CParamDesc, COutputDesc } from "./abi.js";
// Public helpers that older callers reach for — re-exported so the
// split is invisible to non-internal consumers (cFusedCodegen.ts,
// cJitInstall.ts, tests).
export {
  mangle,
  tensorData,
  tensorLen,
  tensorD0,
  tensorD1,
  formatNumberLiteral,
  C_SCALAR_TARGET,
} from "./codegenCtx.js";

export interface GenerateCResult {
  cSource: string;
  cFnName: string;
  paramDescs: CParamDesc[];
  outputDescs: COutputDesc[];
  /** The full ABI slot list in calling order:
   *    paramDescs[0].slots ++ paramDescs[1].slots ++ ...
   *    ++ outputDescs[0].slots ++ ... ++ trailer slots (ticState/errFlag).
   *  The JS wrapper walks this list to marshal values. */
  abiSlots: AbiSlot[];
  /** True when any tensor is involved (params, locals, or outputs). */
  usesTensors: boolean;
  /** koffi function signature string for declaring the C function. */
  koffiSignature: string;
  /** True when tic/toc are used — the function has an extra `double*` param. */
  needsTicState: boolean;
  /** True when any Index read was emitted — the function has an extra
   *  `double *__err_flag` trailing param. */
  needsErrorFlag: boolean;
}

/** Per-function emission output. The outer function's result is returned
 *  upstream (for koffi + JS wrapper use); callee results feed into the
 *  assembled C source only. */
interface EmitOneFnResult {
  /** The full C function definition — signature + body. */
  definition: string;
  paramDescs: CParamDesc[];
  outputDescs: COutputDesc[];
  abiSlots: AbiSlot[];
  usesTensors: boolean;
  needsTicState: boolean;
  needsErrorFlag: boolean;
  koffiSignature: string;
  cFnName: string;
}

export function generateC(
  body: JitStmt[],
  params: string[],
  outputs: string[],
  nargout: number,
  localVars: Set<string>,
  argTypes: JitType[],
  _outputType: JitType | null,
  outputTypes: JitType[],
  fnName: string,
  fuse?: boolean,
  openmp?: boolean,
  generatedIRBodies?: Map<string, GeneratedFn>
): GenerateCResult {
  if (params.length !== argTypes.length) {
    throw new Error("C-JIT codegen: params/argTypes length mismatch");
  }

  // ── Collect reachable callees in post-order (deepest first) ────────
  const reachable: string[] = [];
  if (generatedIRBodies) {
    collectReachableJitNames(body, generatedIRBodies, reachable, new Set());
  }

  // ── Emit each callee as a static function ──────────────────────────
  const calleeDefs: string[] = [];
  let anyCalleeUsesTensors = false;
  for (const jitName of reachable) {
    const callee = generatedIRBodies!.get(jitName);
    if (!callee) {
      throw new Error(
        `C-JIT codegen: UserCall '${jitName}' missing IR body in generatedIRBodies`
      );
    }
    const r = emitOneFunction(
      callee.body,
      callee.fn.params,
      callee.outputNames,
      callee.nargout,
      callee.localVars,
      callee.argTypes,
      callee.outputTypes,
      `jit_${jitName}`,
      fuse,
      openmp,
      generatedIRBodies,
      { isStatic: true, forceNeedsErrorFlag: true }
    );
    // The static callee must not need __tic_state; the outer's tic
    // state is not plumbed through to callees (not worth the ABI
    // widening for this first cut).
    if (r.needsTicState) {
      throw new Error(
        `C-JIT codegen: UserCall callee '${jitName}' uses tic/toc (unsupported)`
      );
    }
    calleeDefs.push(r.definition);
    if (r.usesTensors) anyCalleeUsesTensors = true;
  }

  // ── Emit the outer function ────────────────────────────────────────
  const outer = emitOneFunction(
    body,
    params,
    outputs,
    nargout,
    localVars,
    argTypes,
    outputTypes,
    `jit_${fnName}`,
    fuse,
    openmp,
    generatedIRBodies,
    { isStatic: false }
  );

  // ── Assemble the full C source ─────────────────────────────────────
  const usesTensors = outer.usesTensors || anyCalleeUsesTensors;
  const parts: string[] = [];
  parts.push(`/* JIT C (koffi): ${fnName}(${params.join(", ")}) */`);
  parts.push(`#include <math.h>`);
  parts.push(`#include <stdint.h>`);
  parts.push(`#include "jit_runtime.h"`);
  // Catch a stale jit_runtime.a at compile time — users get a clear
  // "rebuild the addon" message instead of a cryptic linker error.
  parts.push(
    `#if !defined(NUMBL_JIT_RT_VERSION) || NUMBL_JIT_RT_VERSION < ${NUMBL_JIT_RT_REQUIRED_VERSION}`
  );
  parts.push(
    `#error "numbl_jit_runtime too old (need version >= ${NUMBL_JIT_RT_REQUIRED_VERSION}); run \`npm run build:addon\` to rebuild"`
  );
  parts.push(`#endif`);
  if (usesTensors) {
    parts.push(`#include <stdlib.h>`);
    parts.push(`#include <string.h>`);
    parts.push(`#include "numbl_ops.h"`);
  }
  parts.push("");

  for (const def of calleeDefs) {
    parts.push(def);
    parts.push("");
  }
  parts.push(outer.definition);

  return {
    cSource: parts.join("\n"),
    cFnName: outer.cFnName,
    paramDescs: outer.paramDescs,
    outputDescs: outer.outputDescs,
    abiSlots: outer.abiSlots,
    usesTensors,
    koffiSignature: outer.koffiSignature,
    needsTicState: outer.needsTicState,
    needsErrorFlag: outer.needsErrorFlag,
  };
}

/** Walk `body` and collect every UserCall's jitName in post-order
 *  (callees before callers), deduping on revisit. The post-order makes
 *  the final C source compile cleanly without forward declarations. */
function collectReachableJitNames(
  body: JitStmt[],
  generatedIRBodies: Map<string, GeneratedFn>,
  out: string[],
  seen: Set<string>
): void {
  const visitExpr = (e: JitExpr): void => {
    switch (e.tag) {
      case "UserCall": {
        for (const a of e.args) visitExpr(a);
        if (!seen.has(e.jitName)) {
          seen.add(e.jitName);
          const callee = generatedIRBodies.get(e.jitName);
          if (callee) {
            collectReachableJitNames(callee.body, generatedIRBodies, out, seen);
            out.push(e.jitName);
          }
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
        for (const a of e.args) visitExpr(a);
        return;
      case "Index":
        visitExpr(e.base);
        for (const i of e.indices) visitExpr(i);
        return;
      case "RangeSliceRead":
        visitExpr(e.start);
        if (e.end) visitExpr(e.end);
        return;
      case "TensorLiteral":
        for (const row of e.rows) for (const cell of row) visitExpr(cell);
        return;
      case "VConcatGrow":
        visitExpr(e.base);
        visitExpr(e.value);
        return;
      default:
        return;
    }
  };
  const visitStmt = (s: JitStmt): void => {
    switch (s.tag) {
      case "Assign":
      case "ExprStmt":
        visitExpr(s.expr);
        return;
      case "AssignIndex":
        visitExpr(s.value);
        for (const i of s.indices) visitExpr(i);
        return;
      case "AssignIndexRange":
        visitExpr(s.dstStart);
        visitExpr(s.dstEnd);
        if (s.srcStart) visitExpr(s.srcStart);
        if (s.srcEnd) visitExpr(s.srcEnd);
        return;
      case "AssignIndexCol":
        visitExpr(s.colIndex);
        return;
      case "If":
        visitExpr(s.cond);
        s.thenBody.forEach(visitStmt);
        s.elseifBlocks.forEach(eb => {
          visitExpr(eb.cond);
          eb.body.forEach(visitStmt);
        });
        if (s.elseBody) s.elseBody.forEach(visitStmt);
        return;
      case "For":
        visitExpr(s.start);
        visitExpr(s.end);
        if (s.step) visitExpr(s.step);
        s.body.forEach(visitStmt);
        return;
      case "While":
        visitExpr(s.cond);
        s.body.forEach(visitStmt);
        return;
      default:
        return;
    }
  };
  body.forEach(visitStmt);
}

/** Emit a single C function (outer or static callee). Returns the full
 *  definition plus all metadata the caller might want. */
function emitOneFunction(
  body: JitStmt[],
  params: string[],
  outputs: string[],
  nargout: number,
  localVars: Set<string>,
  argTypes: JitType[],
  outputTypes: JitType[],
  cFnName: string,
  fuse: boolean | undefined,
  openmp: boolean | undefined,
  generatedIRBodies: Map<string, GeneratedFn> | undefined,
  opts: { isStatic: boolean; forceNeedsErrorFlag?: boolean }
): EmitOneFnResult {
  const effectiveOutputs = outputs.slice(0, nargout || 1);
  const effectiveOutputTypes = outputTypes.slice(0, effectiveOutputs.length);

  const cls = analyzeTensorUsage(
    body,
    params,
    argTypes,
    effectiveOutputs,
    effectiveOutputTypes
  );

  const paramOutputTensors = new Set<string>();
  const unshareTensorParams = new Set<string>();
  for (const [name, m] of cls.meta) {
    if (m.kind === "paramOutput") paramOutputTensors.add(name);
    if (m.needsUnshare) unshareTensorParams.add(name);
  }

  const paramDescs: CParamDesc[] = params.map((p, i) => {
    const kind: CParamDesc["kind"] =
      argTypes[i].kind === "tensor" ? "tensor" : "scalar";
    const desc: CParamDesc = { name: p, kind, slots: [] };
    if (kind === "tensor") {
      const d = cls.meta.get(p)?.maxIndexDim ?? 0;
      if (d >= 2) desc.ndim = d;
    }
    return desc;
  });

  const outputDescs: COutputDesc[] = effectiveOutputs.map((name, i) => {
    const t = effectiveOutputTypes[i]?.kind;
    const kind: COutputDesc["kind"] =
      t === "tensor" ? "tensor" : t === "boolean" ? "boolean" : "scalar";
    const desc: COutputDesc = { name, kind, slots: [] };
    if (kind === "tensor" && cls.meta.get(name)?.isDynamicOutput) {
      desc.dynamic = true;
    }
    return desc;
  });

  const ctx: EmitCtx = {
    cls,
    scratchCount: 0,
    tmp: { n: 0 },
    usedScratch: new Set(),
    fuse: fuse ?? false,
    needsTicState: false,
    needsErrorFlag: false,
    openmp: openmp ?? false,
  };

  const indent = "  ";
  const bodyLines: string[] = [];

  emitStmts(bodyLines, body, indent, ctx);

  // Callees always take `__err_flag` so the outer can pass it through
  // uniformly. Without this, a callee with no Index ops and no nested
  // UserCalls would have a signature mismatch with the outer's call
  // expression.
  if (opts.forceNeedsErrorFlag) ctx.needsErrorFlag = true;

  const needsShapeLocals = (name: string): boolean => {
    const m = cls.meta.get(name);
    if (!m) return false;
    if (m.hasFreshAlloc) return true;
    return m.maxIndexDim >= 2;
  };

  // ── Epilogue ────────────────────────────────────────────────────────
  const epilogueLines: string[] = [];
  for (const od of outputDescs) {
    if (od.kind === "tensor") {
      if (od.dynamic) {
        epilogueLines.push(
          `${indent}*${mangle(od.name)}_buf_out = ${tensorData(od.name)};`
        );
        epilogueLines.push(
          `${indent}*${mangle(od.name)}_out_len = ${tensorLen(od.name)};`
        );
        epilogueLines.push(
          `${indent}*${mangle(od.name)}_d0_out = ${tensorD0(od.name)};`
        );
        epilogueLines.push(
          `${indent}*${mangle(od.name)}_d1_out = ${tensorD1(od.name)};`
        );
      } else {
        epilogueLines.push(
          `${indent}*${mangle(od.name)}_out_len = ${tensorLen(od.name)};`
        );
      }
    } else {
      epilogueLines.push(
        `${indent}*${mangle(od.name)}_out = ${mangle(od.name)};`
      );
    }
  }

  for (const sIdx of ctx.usedScratch) {
    epilogueLines.push(
      `${indent}if (${scratchData(sIdx)}) free(${scratchData(sIdx)});`
    );
  }

  for (const name of cls.localTensorNames) {
    if (cls.meta.get(name)?.isDynamicOutput) continue;
    epilogueLines.push(
      `${indent}if (${tensorData(name)}) free(${tensorData(name)});`
    );
  }

  for (const p of unshareTensorParams) {
    if (cls.meta.get(p)?.isDynamicOutput) continue;
    epilogueLines.push(
      `${indent}if (${tensorData(p)}) free(${tensorData(p)});`
    );
  }

  // ── Prelude ─────────────────────────────────────────────────────────
  const paramSet = new Set(params);
  const allLocals = [...localVars].filter(v => !paramSet.has(v)).sort();
  const preludeLines: string[] = [];

  const emitParamShapeLocals = (p: string, useInSuffix: boolean): void => {
    const suf = useInSuffix ? "_in" : "";
    const d = cls.meta.get(p)?.maxIndexDim ?? 0;
    if (needsShapeLocals(p)) {
      if (d >= 2) {
        preludeLines.push(
          `${indent}int64_t ${tensorD0(p)} = ${tensorD0(p)}${suf};`
        );
      } else {
        preludeLines.push(
          `${indent}int64_t ${tensorD0(p)} = ${tensorLen(p)}${suf};`
        );
      }
      if (d >= 3) {
        preludeLines.push(
          `${indent}int64_t ${tensorD1(p)} = ${tensorD1(p)}${suf};`
        );
      } else {
        preludeLines.push(`${indent}int64_t ${tensorD1(p)} = 1;`);
      }
    } else {
      if (d >= 2) {
        preludeLines.push(
          `${indent}int64_t ${tensorD0(p)} = ${tensorD0(p)}${suf};`
        );
      }
      if (d >= 3) {
        preludeLines.push(
          `${indent}int64_t ${tensorD1(p)} = ${tensorD1(p)}${suf};`
        );
      }
    }
  };

  for (const p of paramOutputTensors) {
    if (cls.meta.get(p)?.isDynamicOutput) {
      preludeLines.push(
        `${indent}int64_t ${tensorLen(p)} = ${tensorLen(p)}_in;`
      );
      emitParamShapeLocals(p, true);
      preludeLines.push(`${indent}double *${tensorData(p)} = NULL;`);
      preludeLines.push(`${indent}if (${tensorLen(p)} > 0) {`);
      preludeLines.push(
        `${indent}  ${tensorData(p)} = (double *)malloc((size_t)${tensorLen(p)} * sizeof(double));`
      );
      preludeLines.push(
        `${indent}  memcpy(${tensorData(p)}, ${tensorData(p)}_in, (size_t)${tensorLen(p)} * sizeof(double));`
      );
      preludeLines.push(`${indent}}`);
    } else {
      preludeLines.push(
        `${indent}double *${tensorData(p)} = ${mangle(p)}_buf;`
      );
      preludeLines.push(
        `${indent}int64_t ${tensorLen(p)} = ${tensorLen(p)}_in;`
      );
      emitParamShapeLocals(p, true);
    }
  }

  for (const p of unshareTensorParams) {
    if (paramOutputTensors.has(p)) continue;
    preludeLines.push(`${indent}int64_t ${tensorLen(p)} = ${tensorLen(p)}_in;`);
    emitParamShapeLocals(p, true);
    preludeLines.push(`${indent}double *${tensorData(p)} = NULL;`);
    preludeLines.push(`${indent}if (${tensorLen(p)} > 0) {`);
    preludeLines.push(
      `${indent}  ${tensorData(p)} = (double *)malloc((size_t)${tensorLen(p)} * sizeof(double));`
    );
    preludeLines.push(
      `${indent}  memcpy(${tensorData(p)}, ${tensorData(p)}_in, (size_t)${tensorLen(p)} * sizeof(double));`
    );
    preludeLines.push(`${indent}}`);
  }

  for (const local of allLocals) {
    if (cls.tensorVars.has(local)) {
      const localMeta = cls.meta.get(local);
      const isOutput = cls.outputTensorNames.has(local);
      if (isOutput && !localMeta?.isDynamicOutput) {
        preludeLines.push(
          `${indent}double *${tensorData(local)} = ${mangle(local)}_buf;`
        );
      } else {
        preludeLines.push(`${indent}double *${tensorData(local)} = NULL;`);
      }
      preludeLines.push(`${indent}int64_t ${tensorLen(local)} = 0;`);
      if (needsShapeLocals(local)) {
        preludeLines.push(`${indent}int64_t ${tensorD0(local)} = 0;`);
        preludeLines.push(`${indent}int64_t ${tensorD1(local)} = 0;`);
      }
    } else {
      preludeLines.push(`${indent}double ${mangle(local)} = 0.0;`);
    }
  }

  for (const sIdx of ctx.usedScratch) {
    preludeLines.push(`${indent}double *${scratchData(sIdx)} = NULL;`);
    preludeLines.push(`${indent}int64_t ${scratchLen(sIdx)} = 0;`);
  }

  // ── Signature + ABI ─────────────────────────────────────────────────
  const abiSlots = buildAbiSlots(
    paramDescs,
    outputDescs,
    cls,
    paramOutputTensors,
    unshareTensorParams,
    ctx.needsTicState,
    ctx.needsErrorFlag
  );

  const sigParts = abiSlots.map(
    s => `${s.cType}${spaceBeforeName(s.cType)}${s.cName}`
  );
  const koffiParts = abiSlots.map(s => s.koffiType);
  const paramList = sigParts.length > 0 ? sigParts.join(", ") : "void";
  const prefix = opts.isStatic ? "static " : "";
  const signature = `${prefix}void ${cFnName}(${paramList})`;
  const koffiSignature = `void ${cFnName}(${koffiParts.join(", ")})`;

  const defParts: string[] = [];
  defParts.push(`${signature} {`);
  defParts.push(preludeLines.join("\n"));
  defParts.push(bodyLines.join("\n"));
  if (epilogueLines.length > 0) {
    defParts.push(epilogueLines.join("\n"));
  }
  defParts.push(`}`);

  // Suppress silly warnings we'll want if anything gets added later.
  void generatedIRBodies;

  return {
    definition: defParts.join("\n"),
    paramDescs,
    outputDescs,
    abiSlots,
    usesTensors: cls.tensorVars.size > 0,
    needsTicState: ctx.needsTicState,
    needsErrorFlag: ctx.needsErrorFlag,
    koffiSignature,
    cFnName,
  };
}
