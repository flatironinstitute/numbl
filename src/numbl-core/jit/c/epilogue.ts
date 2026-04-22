/**
 * Function epilogue emission.
 *
 * The epilogue is the set of statements appended after the body of a
 * generated function: output-slot writes (scalar out-pointers, tensor
 * dynamic-output transfers), scratch buffer frees, and local / unshared
 * tensor frees. Called once per function by `generateC` in
 * [assemble.ts](./assemble.ts).
 *
 * The prelude is in [prelude.ts](./prelude.ts); both read from the same
 * shared state (`ClassificationResult` + `EmitCtx`).
 */
import type { ClassificationResult } from "./classify.js";
import type { COutputDesc } from "./abi.js";
import {
  mangle,
  mangleIm,
  scratchData,
  scratchDataIm,
  tensorD0,
  tensorD1,
  tensorData,
  tensorDataIm,
  tensorLen,
  type EmitCtx,
} from "./context.js";

export interface EpilogueInput {
  cls: ClassificationResult;
  ctx: EmitCtx;
  outputDescs: COutputDesc[];
  /** Pure-input tensor params that were malloc'd in the prelude's
   *  unshare path — must be freed here. */
  unshareTensorParams: Set<string>;
  /** Indent string to prepend to each emitted line. */
  indent: string;
}

export function buildEpilogue(input: EpilogueInput): string[] {
  const { cls, ctx, outputDescs, unshareTensorParams, indent } = input;
  const lines: string[] = [];

  // ── Output writes (scalar out-pointers + dynamic tensor transfers) ──
  for (const od of outputDescs) {
    if (od.kind === "tensor") {
      if (od.dynamic) {
        lines.push(
          `${indent}*${mangle(od.name)}_buf_out = ${tensorData(od.name)};`
        );
        if (od.isComplex) {
          lines.push(
            `${indent}*__im_${mangle(od.name)}_buf_out = ${tensorDataIm(od.name)};`
          );
        }
        lines.push(
          `${indent}*${mangle(od.name)}_out_len = ${tensorLen(od.name)};`
        );
        lines.push(
          `${indent}*${mangle(od.name)}_d0_out = ${tensorD0(od.name)};`
        );
        lines.push(
          `${indent}*${mangle(od.name)}_d1_out = ${tensorD1(od.name)};`
        );
      } else {
        lines.push(
          `${indent}*${mangle(od.name)}_out_len = ${tensorLen(od.name)};`
        );
      }
    } else if (od.kind === "complexScalar") {
      lines.push(`${indent}*${mangle(od.name)}_out = ${mangle(od.name)};`);
      lines.push(`${indent}*${mangleIm(od.name)}_out = ${mangleIm(od.name)};`);
    } else {
      lines.push(`${indent}*${mangle(od.name)}_out = ${mangle(od.name)};`);
    }
  }

  // ── Scratch frees ───────────────────────────────────────────────────
  for (const sIdx of ctx.usedScratch) {
    lines.push(
      `${indent}if (${scratchData(sIdx)}) free(${scratchData(sIdx)});`
    );
    if (ctx.complexScratch.has(sIdx)) {
      lines.push(
        `${indent}if (${scratchDataIm(sIdx)}) free(${scratchDataIm(sIdx)});`
      );
    }
  }

  // ── Local tensor frees (skipping dynamic outputs — ownership passed) ─
  for (const name of cls.localTensorNames) {
    if (cls.meta.get(name)?.isDynamicOutput) continue;
    lines.push(`${indent}if (${tensorData(name)}) free(${tensorData(name)});`);
    if (cls.meta.get(name)?.isComplex) {
      lines.push(
        `${indent}if (${tensorDataIm(name)}) free(${tensorDataIm(name)});`
      );
    }
  }

  // ── Unshared param frees (skipping dynamic outputs — ownership passed) ─
  for (const p of unshareTensorParams) {
    if (cls.meta.get(p)?.isDynamicOutput) continue;
    lines.push(`${indent}if (${tensorData(p)}) free(${tensorData(p)});`);
    if (cls.meta.get(p)?.isComplex) {
      lines.push(`${indent}if (${tensorDataIm(p)}) free(${tensorDataIm(p)});`);
    }
  }

  return lines;
}
