/**
 * JIT IR → pure C code generation (koffi path).
 *
 * Orchestration only: this file wires the classify / ABI / emit pieces
 * together into a single `generateC(...)` call and assembles the final
 * C source (headers + signature + prelude + body + epilogue).
 *
 *   classify.ts     — TensorMeta / analyzeTensorUsage, the single pass
 *                     feeding every downstream decision.
 *   abi.ts          — AbiSlot / CParamDesc / COutputDesc, buildAbiSlots.
 *                     The one schema walked by both signature and JS.
 *   emit.ts         — per-statement / per-expression C emission, reads
 *                     ctx.cls for every classification decision.
 *   codegenCtx.ts   — EmitCtx + shared name/opcode helpers.
 *
 * Keep this file small: if a new iteration wants to grow codegen, put
 * the new logic in the targeted child file (emit.ts for codegen, abi.ts
 * for a new slot kind, classify.ts for a new meta flag).
 */
import { type JitStmt, type JitType } from "../jitTypes.js";
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
  openmp?: boolean
): GenerateCResult {
  if (params.length !== argTypes.length) {
    throw new Error("C-JIT codegen: params/argTypes length mismatch");
  }

  const effectiveOutputs = outputs.slice(0, nargout || 1);
  const effectiveOutputTypes = outputTypes.slice(0, effectiveOutputs.length);

  const cls = analyzeTensorUsage(
    body,
    params,
    argTypes,
    effectiveOutputs,
    effectiveOutputTypes
  );

  // Derived views used in the prelude/epilogue below.
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

  // Names for which we emit `_d0` / `_d1` mutable locals: anything that
  // either receives a fresh-alloc (shape reassignment) or is referenced
  // as a tensor with multi-index arity.
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
        // Dynamic output: transfer ownership of the C-malloc'd buffer
        // to the caller and report the runtime shape. The JS wrapper
        // reads data/d0/d1 and frees the pointer after copying into a
        // JS-owned Float64Array.
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
        // Fixed output: data already lives in the pre-allocated _buf; just
        // report the runtime length.
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

  // Emit `v_<p>_d0` / `v_<p>_d1` seeding for a tensor param. Uses the
  // `_in` suffix when the param signature is shadowed (param-output /
  // unshared).
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

  // Shadow tensor input-output params with writable locals.
  for (const p of paramOutputTensors) {
    if (cls.meta.get(p)?.isDynamicOutput) {
      // Dynamic param-output: unshare-copy `_in` into a C-owned buffer
      // and transfer the final pointer back at epilogue.
      preludeLines.push(
        `${indent}int64_t ${tensorLen(p)} = ${tensorLen(p)}_in;`
      );
      emitParamShapeLocals(p, /*useInSuffix*/ true);
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
      // Fixed param-output: data points at the seeded _buf (no copy needed).
      preludeLines.push(
        `${indent}double *${tensorData(p)} = ${mangle(p)}_buf;`
      );
      preludeLines.push(
        `${indent}int64_t ${tensorLen(p)} = ${tensorLen(p)}_in;`
      );
      emitParamShapeLocals(p, /*useInSuffix*/ true);
    }
  }

  // Unshare-at-entry for pure-input tensor params that are written.
  for (const p of unshareTensorParams) {
    if (paramOutputTensors.has(p)) continue; // already shadowed above
    preludeLines.push(`${indent}int64_t ${tensorLen(p)} = ${tensorLen(p)}_in;`);
    emitParamShapeLocals(p, /*useInSuffix*/ true);
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

  // Read-only tensor params use the signature's `v_<p>_d0` / `_d1`
  // directly — no prelude shape-local redeclaration needed. Fresh-alloc
  // and AssignIndex writes are handled by the two loops above.

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
  const cFnName = `jit_${fnName}`;
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
  const signature = `void ${cFnName}(${paramList})`;
  const koffiSignature = `void ${cFnName}(${koffiParts.join(", ")})`;

  // ── Assemble the full C source ──────────────────────────────────────
  const usesTensors = cls.tensorVars.size > 0;
  const parts: string[] = [];
  parts.push(`/* JIT C (koffi): ${fnName}(${params.join(", ")}) */`);
  parts.push(`#include <math.h>`);
  // Always include jit_runtime — the emitter may call any of its helpers
  // (mod, sign, reduce, tic/toc, idx1r) from a non-tensor scalar context.
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
  parts.push(`${signature} {`);
  parts.push(preludeLines.join("\n"));
  parts.push(bodyLines.join("\n"));
  if (epilogueLines.length > 0) {
    parts.push(epilogueLines.join("\n"));
  }
  parts.push(`}`);

  return {
    cSource: parts.join("\n"),
    cFnName,
    paramDescs,
    outputDescs,
    abiSlots,
    usesTensors,
    koffiSignature,
    needsTicState: ctx.needsTicState,
    needsErrorFlag: ctx.needsErrorFlag,
  };
}
