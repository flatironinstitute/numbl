/**
 * The native ABI as an explicit slot schema — one source of truth shared
 * between the C-signature builder (jitCodegenC.ts) and the JS wrapper
 * marshaller (cJitInstall.ts).
 *
 * Each `AbiSlot` carries everything either side needs: the C type for
 * the signature, the koffi type string (with `_Out_` prefix for
 * out-pointers), the identifier name in the emitted C, and a backref
 * (paramIdx / outputIdx) so the JS wrapper can locate the source value
 * or output buffer without reconstructing the param-to-slot mapping
 * itself.
 *
 * Adding a new ABI shape = add a slot kind here, emit it in
 * `buildAbiSlots`, and handle it in the JS marshaller.
 */
import type { ClassificationResult } from "./classify.js";
import {
  mangle,
  mangleIm,
  tensorD0,
  tensorD1,
  tensorData,
  tensorDataIm,
  tensorLen,
} from "./codegenCtx.js";

export type AbiSlotKind =
  // Param-sourced slots — value comes from callArgs[paramIdx].
  | "scalar"
  // Complex-scalar param slots — two per param (re + im).
  | "complexScalarRe"
  | "complexScalarIm"
  | "tensorData"
  /** Imaginary data pointer for a complex tensor param. Paired with
   *  `tensorData`. Marshaller passes the RuntimeTensor's `.imag` (a
   *  Float64Array) or NULL when `.imag === undefined`; the numbl_ops
   *  complex kernels treat NULL imag as all-zero. */
  | "tensorDataIm"
  | "tensorLen"
  | "tensorD0"
  | "tensorD1"
  // Output-allocated slots — value comes from outputBufs[outputIdx].
  | "scalarOut"
  // Complex-scalar output slots — two per output (re + im).
  | "complexScalarReOut"
  | "complexScalarImOut"
  | "fixedOutBuf"
  /** Imaginary fixed-output buffer (complex tensor output, non-dynamic). */
  | "fixedOutBufIm"
  | "fixedOutLen"
  | "dynOutBuf"
  /** Imaginary buffer pointer for a dynamic complex tensor output. C
   *  mallocs, transfers ownership via `double **`; wrapper decodes+copies
   *  into a fresh Float64Array then frees. */
  | "dynOutBufIm"
  | "dynOutLen"
  | "dynOutD0"
  | "dynOutD1"
  // Trailer slots — value comes from per-call shared buffers.
  | "ticState"
  | "errFlag";

export interface AbiSlot {
  kind: AbiSlotKind;
  /** C type string for the signature, e.g. "double", "const double *",
   *  "double **". */
  cType: string;
  /** Identifier as it appears in the C signature. */
  cName: string;
  /** koffi type string, with `_Out_` prefix where koffi must treat the
   *  pointer as an out-param. */
  koffiType: string;
  /** Index into paramDescs, for "scalar" / "tensor*" kinds. */
  paramIdx?: number;
  /** Index into outputDescs, for output-allocated kinds. */
  outputIdx?: number;
}

export interface CParamDesc {
  name: string;
  kind: "scalar" | "complexScalar" | "tensor";
  /** For tensor params: max indexing arity the body uses (1, 2, or 3).
   *  Drives the extra `_d0` / `_d1` shape args the JS wrapper must
   *  marshal. `undefined` means the tensor is only used in whole-tensor
   *  ops (legacy data/len ABI). */
  ndim?: number;
  /** True for complex tensor params. Adds an imag-data slot right after
   *  the real-data slot; the marshaller supplies the tensor's `.imag`
   *  Float64Array or NULL. Ignored for scalar kinds. */
  isComplex?: boolean;
  /** Ordered slots this param contributes to the ABI. One slot for a
   *  scalar; two for a complex scalar (re + im); two or more
   *  (data + [imag for complex] + len + optional d0/d1) for a tensor. */
  slots: AbiSlot[];
}

/** Per-output descriptor. Tells the JS wrapper how to marshal outputs. */
export interface COutputDesc {
  name: string;
  kind: "scalar" | "boolean" | "complexScalar" | "tensor";
  /** True for tensor outputs using the dynamic-output ABI: the C code
   *  malloc's the buffer and transfers ownership via `double **` and
   *  extra d0/d1 out-slots. The JS wrapper decodes the pointer, copies
   *  into a fresh Float64Array, and frees the C allocation. */
  dynamic?: boolean;
  /** True for complex tensor outputs. Fixed outputs add a paired imag
   *  Float64Array buffer; dynamic outputs add a paired imag `double **`
   *  out-pointer the caller decodes + frees after the call. */
  isComplex?: boolean;
  /** Ordered slots this output contributes to the ABI. One for scalars,
   *  two for complex scalars (reOut + imOut), two for fixed real tensor
   *  outputs (buf + lenOut), three for fixed complex (buf + bufIm +
   *  lenOut), four for dynamic real tensor outputs, five for dynamic
   *  complex (dynBuf + dynBufIm + dynLen + dynD0 + dynD1). */
  slots: AbiSlot[];
}

/** Build the ABI schema for one generated function. Mutates paramDescs /
 *  outputDescs in place by filling in `slots`; returns the complete
 *  `abiSlots` array in calling order (params, then outputs, then trailers). */
export function buildAbiSlots(
  paramDescs: CParamDesc[],
  outputDescs: COutputDesc[],
  cls: ClassificationResult,
  paramOutputTensors: Set<string>,
  unshareTensorParams: Set<string>,
  needsTicState: boolean,
  needsErrorFlag: boolean
): AbiSlot[] {
  paramDescs.forEach((pd, pi) => {
    if (pd.kind === "tensor") {
      // `_in` suffix when the param signature is shadowed by a writable
      // local (param-output / unshared). The prelude handles the copy.
      const suffix =
        paramOutputTensors.has(pd.name) || unshareTensorParams.has(pd.name)
          ? "_in"
          : "";
      const d = cls.meta.get(pd.name)?.maxIndexDim ?? 0;
      pd.slots.push({
        kind: "tensorData",
        cType: "const double *",
        cName: `${tensorData(pd.name)}${suffix}`,
        koffiType: "double *",
        paramIdx: pi,
      });
      if (pd.isComplex) {
        pd.slots.push({
          kind: "tensorDataIm",
          cType: "const double *",
          cName: `${tensorDataIm(pd.name)}${suffix}`,
          koffiType: "double *",
          paramIdx: pi,
        });
      }
      pd.slots.push({
        kind: "tensorLen",
        cType: "int64_t",
        cName: `${tensorLen(pd.name)}${suffix}`,
        koffiType: "int64_t",
        paramIdx: pi,
      });
      if (d >= 2) {
        pd.slots.push({
          kind: "tensorD0",
          cType: "int64_t",
          cName: `${tensorD0(pd.name)}${suffix}`,
          koffiType: "int64_t",
          paramIdx: pi,
        });
      }
      if (d >= 3) {
        pd.slots.push({
          kind: "tensorD1",
          cType: "int64_t",
          cName: `${tensorD1(pd.name)}${suffix}`,
          koffiType: "int64_t",
          paramIdx: pi,
        });
      }
    } else if (pd.kind === "complexScalar") {
      pd.slots.push({
        kind: "complexScalarRe",
        cType: "double",
        cName: mangle(pd.name),
        koffiType: "double",
        paramIdx: pi,
      });
      pd.slots.push({
        kind: "complexScalarIm",
        cType: "double",
        cName: mangleIm(pd.name),
        koffiType: "double",
        paramIdx: pi,
      });
    } else {
      pd.slots.push({
        kind: "scalar",
        cType: "double",
        cName: mangle(pd.name),
        koffiType: "double",
        paramIdx: pi,
      });
    }
  });

  outputDescs.forEach((od, oi) => {
    if (od.kind === "tensor") {
      if (od.dynamic) {
        // Dynamic output: C allocates, transfers ownership via `*_buf_out`.
        od.slots.push({
          kind: "dynOutBuf",
          cType: "double **",
          cName: `${mangle(od.name)}_buf_out`,
          koffiType: "_Out_ double **",
          outputIdx: oi,
        });
        if (od.isComplex) {
          od.slots.push({
            kind: "dynOutBufIm",
            cType: "double **",
            cName: `${mangle(od.name)}_buf_im_out`,
            koffiType: "_Out_ double **",
            outputIdx: oi,
          });
        }
        od.slots.push({
          kind: "dynOutLen",
          cType: "int64_t *",
          cName: `${mangle(od.name)}_out_len`,
          koffiType: "_Out_ int64_t *",
          outputIdx: oi,
        });
        od.slots.push({
          kind: "dynOutD0",
          cType: "int64_t *",
          cName: `${mangle(od.name)}_d0_out`,
          koffiType: "_Out_ int64_t *",
          outputIdx: oi,
        });
        od.slots.push({
          kind: "dynOutD1",
          cType: "int64_t *",
          cName: `${mangle(od.name)}_d1_out`,
          koffiType: "_Out_ int64_t *",
          outputIdx: oi,
        });
      } else {
        od.slots.push({
          kind: "fixedOutBuf",
          cType: "double *",
          cName: `${mangle(od.name)}_buf`,
          koffiType: "double *",
          outputIdx: oi,
        });
        if (od.isComplex) {
          od.slots.push({
            kind: "fixedOutBufIm",
            cType: "double *",
            cName: `${mangle(od.name)}_buf_im`,
            koffiType: "double *",
            outputIdx: oi,
          });
        }
        od.slots.push({
          kind: "fixedOutLen",
          cType: "int64_t *",
          cName: `${mangle(od.name)}_out_len`,
          koffiType: "int64_t *",
          outputIdx: oi,
        });
      }
    } else if (od.kind === "complexScalar") {
      od.slots.push({
        kind: "complexScalarReOut",
        cType: "double *",
        cName: `${mangle(od.name)}_out`,
        koffiType: "double *",
        outputIdx: oi,
      });
      od.slots.push({
        kind: "complexScalarImOut",
        cType: "double *",
        cName: `${mangleIm(od.name)}_out`,
        koffiType: "double *",
        outputIdx: oi,
      });
    } else {
      od.slots.push({
        kind: "scalarOut",
        cType: "double *",
        cName: `${mangle(od.name)}_out`,
        koffiType: "double *",
        outputIdx: oi,
      });
    }
  });

  const abiSlots: AbiSlot[] = [];
  for (const pd of paramDescs) abiSlots.push(...pd.slots);
  for (const od of outputDescs) abiSlots.push(...od.slots);
  if (needsTicState) {
    abiSlots.push({
      kind: "ticState",
      cType: "double *",
      cName: "__tic_state",
      koffiType: "double *",
    });
  }
  if (needsErrorFlag) {
    abiSlots.push({
      kind: "errFlag",
      cType: "double *",
      cName: "__err_flag",
      koffiType: "double *",
    });
  }
  return abiSlots;
}
