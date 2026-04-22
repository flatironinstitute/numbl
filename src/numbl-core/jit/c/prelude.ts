/**
 * Function prelude emission.
 *
 * The prelude is the set of C declarations at the top of a generated
 * function, written before any statement from the body: shadowed param
 * locals (for param-output seeding + unshare-at-entry), local tensor
 * declarations, complex-scalar imag companions, and scratch buffer
 * slots.
 *
 * Exported as `buildPrelude` — called once per function by `generateC`
 * in [jitCodegenC.ts](./jitCodegenC.ts).
 *
 * The epilogue (tensor frees, out-pointer writes) is in
 * [epilogue.ts](./epilogue.ts); both read from the same shared state
 * (`ClassificationResult` + `EmitCtx`) populated upstream.
 */
import type { JitType } from "../jitTypes.js";
import type { ClassificationResult } from "./classify.js";
import {
  mangle,
  mangleIm,
  scratchData,
  scratchDataIm,
  scratchLen,
  tensorD0,
  tensorD1,
  tensorData,
  tensorDataIm,
  tensorLen,
  type EmitCtx,
} from "./codegenCtx.js";

export interface PreludeInput {
  cls: ClassificationResult;
  ctx: EmitCtx;
  params: string[];
  argTypes: JitType[];
  /** Names with `kind === "paramOutput"` (output name reuses a param name). */
  paramOutputTensors: Set<string>;
  /** Pure-input tensor params that need an unshare-at-entry malloc+memcpy
   *  (the body writes to them, and we must not mutate the caller's buffer). */
  unshareTensorParams: Set<string>;
  /** Locals to declare — outer-scope `localVars` minus params, sorted. */
  allLocals: string[];
  /** Names carrying `complex_or_number` scalar values (paired re+im locals). */
  complexScalarVars: Set<string>;
  /** Indent string to prepend to each emitted line. */
  indent: string;
}

export function buildPrelude(input: PreludeInput): string[] {
  const {
    cls,
    ctx,
    params,
    argTypes,
    paramOutputTensors,
    unshareTensorParams,
    allLocals,
    complexScalarVars,
    indent,
  } = input;
  const lines: string[] = [];

  const needsShapeLocals = (name: string): boolean => {
    const m = cls.meta.get(name);
    if (!m) return false;
    if (m.hasFreshAlloc) return true;
    return m.maxIndexDim >= 2;
  };

  const emitParamShapeLocals = (p: string, useInSuffix: boolean): void => {
    const suf = useInSuffix ? "_in" : "";
    const d = cls.meta.get(p)?.maxIndexDim ?? 0;
    if (needsShapeLocals(p)) {
      if (d >= 2) {
        lines.push(`${indent}int64_t ${tensorD0(p)} = ${tensorD0(p)}${suf};`);
      } else {
        lines.push(`${indent}int64_t ${tensorD0(p)} = ${tensorLen(p)}${suf};`);
      }
      if (d >= 3) {
        lines.push(`${indent}int64_t ${tensorD1(p)} = ${tensorD1(p)}${suf};`);
      } else {
        lines.push(`${indent}int64_t ${tensorD1(p)} = 1;`);
      }
    } else {
      if (d >= 2) {
        lines.push(`${indent}int64_t ${tensorD0(p)} = ${tensorD0(p)}${suf};`);
      }
      if (d >= 3) {
        lines.push(`${indent}int64_t ${tensorD1(p)} = ${tensorD1(p)}${suf};`);
      }
    }
  };

  // ── Param-output tensors: shadow local + optional unshare copy ──────
  for (const p of paramOutputTensors) {
    const isComplex = !!cls.meta.get(p)?.isComplex;
    if (cls.meta.get(p)?.isDynamicOutput) {
      lines.push(`${indent}int64_t ${tensorLen(p)} = ${tensorLen(p)}_in;`);
      emitParamShapeLocals(p, true);
      lines.push(`${indent}double *${tensorData(p)} = NULL;`);
      if (isComplex) {
        lines.push(`${indent}double *${tensorDataIm(p)} = NULL;`);
      }
      lines.push(`${indent}if (${tensorLen(p)} > 0) {`);
      lines.push(
        `${indent}  ${tensorData(p)} = (double *)malloc((size_t)${tensorLen(p)} * sizeof(double));`
      );
      lines.push(
        `${indent}  memcpy(${tensorData(p)}, ${tensorData(p)}_in, (size_t)${tensorLen(p)} * sizeof(double));`
      );
      if (isComplex) {
        // Imag input may be NULL (undefined `.imag`); seed a zero
        // buffer so downstream kernels that require a non-NULL imag
        // pointer on writeable tensors still work.
        lines.push(
          `${indent}  ${tensorDataIm(p)} = (double *)calloc((size_t)${tensorLen(p)}, sizeof(double));`
        );
        lines.push(`${indent}  if (${tensorDataIm(p)}_in) {`);
        lines.push(
          `${indent}    memcpy(${tensorDataIm(p)}, ${tensorDataIm(p)}_in, (size_t)${tensorLen(p)} * sizeof(double));`
        );
        lines.push(`${indent}  }`);
      }
      lines.push(`${indent}}`);
    } else {
      lines.push(`${indent}double *${tensorData(p)} = ${mangle(p)}_buf;`);
      if (isComplex) {
        lines.push(
          `${indent}double *${tensorDataIm(p)} = __im_${mangle(p)}_buf;`
        );
      }
      lines.push(`${indent}int64_t ${tensorLen(p)} = ${tensorLen(p)}_in;`);
      emitParamShapeLocals(p, true);
    }
  }

  // ── Unshared pure-input tensor params: malloc + memcpy from caller ──
  for (const p of unshareTensorParams) {
    if (paramOutputTensors.has(p)) continue;
    const isComplex = !!cls.meta.get(p)?.isComplex;
    lines.push(`${indent}int64_t ${tensorLen(p)} = ${tensorLen(p)}_in;`);
    emitParamShapeLocals(p, true);
    lines.push(`${indent}double *${tensorData(p)} = NULL;`);
    if (isComplex) {
      lines.push(`${indent}double *${tensorDataIm(p)} = NULL;`);
    }
    lines.push(`${indent}if (${tensorLen(p)} > 0) {`);
    lines.push(
      `${indent}  ${tensorData(p)} = (double *)malloc((size_t)${tensorLen(p)} * sizeof(double));`
    );
    lines.push(
      `${indent}  memcpy(${tensorData(p)}, ${tensorData(p)}_in, (size_t)${tensorLen(p)} * sizeof(double));`
    );
    if (isComplex) {
      lines.push(
        `${indent}  ${tensorDataIm(p)} = (double *)calloc((size_t)${tensorLen(p)}, sizeof(double));`
      );
      lines.push(`${indent}  if (${tensorDataIm(p)}_in) {`);
      lines.push(
        `${indent}    memcpy(${tensorDataIm(p)}, ${tensorDataIm(p)}_in, (size_t)${tensorLen(p)} * sizeof(double));`
      );
      lines.push(`${indent}  }`);
    }
    lines.push(`${indent}}`);
  }

  // A scalar param that the body later widens to complex (e.g. `acc = 0`
  // then `acc = acc + 1i`) arrives via the real-scalar ABI (one `double`
  // slot). `complexScalarVars` tracks the effective type — we need to
  // initialize the imag companion to 0 so subsequent emitComplex reads
  // on Var(param) pick up a valid im local. Same idea for complex-param
  // cases, but those already get both slots from buildAbiSlots.
  for (let i = 0; i < params.length; i++) {
    const p = params[i];
    if (complexScalarVars.has(p) && argTypes[i].kind !== "complex_or_number") {
      lines.push(`${indent}double ${mangleIm(p)} = 0.0;`);
    }
  }

  // ── Local declarations (tensors, complex scalars, real scalars) ─────
  for (const local of allLocals) {
    if (cls.tensorVars.has(local)) {
      const localMeta = cls.meta.get(local);
      const isOutput = cls.outputTensorNames.has(local);
      const isComplex = !!localMeta?.isComplex;
      if (isOutput && !localMeta?.isDynamicOutput) {
        lines.push(
          `${indent}double *${tensorData(local)} = ${mangle(local)}_buf;`
        );
        if (isComplex) {
          lines.push(
            `${indent}double *${tensorDataIm(local)} = __im_${mangle(local)}_buf;`
          );
        }
      } else {
        lines.push(`${indent}double *${tensorData(local)} = NULL;`);
        if (isComplex) {
          lines.push(`${indent}double *${tensorDataIm(local)} = NULL;`);
        }
      }
      lines.push(`${indent}int64_t ${tensorLen(local)} = 0;`);
      if (needsShapeLocals(local)) {
        lines.push(`${indent}int64_t ${tensorD0(local)} = 0;`);
        lines.push(`${indent}int64_t ${tensorD1(local)} = 0;`);
      }
    } else if (complexScalarVars.has(local)) {
      lines.push(`${indent}double ${mangle(local)} = 0.0;`);
      lines.push(`${indent}double ${mangleIm(local)} = 0.0;`);
    } else {
      lines.push(`${indent}double ${mangle(local)} = 0.0;`);
    }
  }

  // ── Scratch buffer slots (declared once per usedScratch index) ──────
  for (const sIdx of ctx.usedScratch) {
    lines.push(`${indent}double *${scratchData(sIdx)} = NULL;`);
    if (ctx.complexScratch.has(sIdx)) {
      lines.push(`${indent}double *${scratchDataIm(sIdx)} = NULL;`);
    }
    lines.push(`${indent}int64_t ${scratchLen(sIdx)} = 0;`);
  }

  return lines;
}
