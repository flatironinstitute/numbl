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
  tensorD0,
  tensorD1,
  tensorData,
  tensorLen,
} from "./codegenCtx.js";

export type AbiSlotKind =
  // Param-sourced slots — value comes from callArgs[paramIdx].
  | "scalar"
  | "tensorData"
  | "tensorLen"
  | "tensorD0"
  | "tensorD1"
  // Output-allocated slots — value comes from outputBufs[outputIdx].
  | "scalarOut"
  | "fixedOutBuf"
  | "fixedOutLen"
  | "dynOutBuf"
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
  kind: "scalar" | "tensor";
  /** For tensor params: max indexing arity the body uses (1, 2, or 3).
   *  Drives the extra `_d0` / `_d1` shape args the JS wrapper must
   *  marshal. `undefined` means the tensor is only used in whole-tensor
   *  ops (legacy data/len ABI). */
  ndim?: number;
  /** Ordered slots this param contributes to the ABI. One slot for a
   *  scalar; two or more (data + len + optional d0/d1) for a tensor. */
  slots: AbiSlot[];
}

/** Per-output descriptor. Tells the JS wrapper how to marshal outputs. */
export interface COutputDesc {
  name: string;
  kind: "scalar" | "boolean" | "tensor";
  /** True for tensor outputs using the dynamic-output ABI: the C code
   *  malloc's the buffer and transfers ownership via `double **` and
   *  extra d0/d1 out-slots. The JS wrapper decodes the pointer, copies
   *  into a fresh Float64Array, and frees the C allocation. */
  dynamic?: boolean;
  /** Ordered slots this output contributes to the ABI. One for scalars,
   *  two for fixed tensor outputs (buf + lenOut), four for dynamic
   *  tensor outputs (dynBuf + dynLen + dynD0 + dynD1). */
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
        od.slots.push({
          kind: "fixedOutLen",
          cType: "int64_t *",
          cName: `${mangle(od.name)}_out_len`,
          koffiType: "int64_t *",
          outputIdx: oi,
        });
      }
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
