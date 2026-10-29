/**
 * C-JIT entry point: the shared pipeline (`compile.ts`) with the C target.
 * The module is ready for `cc -shared -fPIC` and `dlopen`; the host calls
 * the exported host-entry wrapper around the entry spec (`emitHostEntry`),
 * while every spec stays `static`.
 *
 * Returns a `signature` describing the C function the host declares to
 * koffi, in JIT `Type`s; the host adapter translates each to a koffi type.
 */

import { emitProgram, type EmitResult } from "./codegen/emit.js";
import {
  compileWith,
  type CompileSpecArgs,
  type CompileTarget,
} from "./compile.js";
import type { IRFunc } from "./lowering/ir.js";
import type { Type } from "./lowering/types.js";
import { C_FUSION } from "./lowering/elemwiseKernels.js";
import type { AssignedFlag } from "./lowering/assignedFlags.js";

export type { CompileSpecArgs as CompileSpecCArgs } from "./compile.js";

/** Describes a single C parameter of the JIT-emitted entry spec. */
export interface SpecCParam {
  /** Identifier the C source uses for the param. */
  readonly cName: string;
  /** JIT type — host translates to koffi type + marshals values. */
  readonly ty: Type;
}

/** Describes a single C return slot of the JIT-emitted entry spec.
 *  For `nargout === 1`: a single by-value return slot whose `cName` is
 *  unused (the function uses C `return`).
 *  For `nargout >= 2`: each output is appended as an out-pointer param
 *  named `_jit_o<i>` AFTER the user params — see `emitStmt.ts`
 *  `fnParamList` for the convention.
 *  For `nargout === 0`: no return slot. */
export interface SpecCOutput {
  readonly cName: string;
  readonly ty: Type;
}

export interface SpecCSignature {
  /** The exported host-entry wrapper around the entry spec (the symbol
   *  koffi resolves after `dlopen`; see `emitHostEntry`). */
  readonly cName: string;
  /** Param shape, in the order the C function expects. */
  readonly params: ReadonlyArray<SpecCParam>;
  /** Output shape. Convention:
   *   - 0 outputs ⇒ empty array; C return type is `void`.
   *   - 1 output ⇒ single entry; C return type is `cTypeFor(out.ty)`.
   *   - ≥2 outputs ⇒ N entries; C return type is `void`, the entries
   *     describe the out-pointer params appended to the call. */
  readonly outputs: ReadonlyArray<SpecCOutput>;
}

export interface CompileSpecCResult {
  /** Mangled entry spec name (the host calls `signature.cName`). */
  cName: string;
  /** Full C source, ready for `cc -shared -fPIC`. */
  source: string;
  /** Names of runtime snippets activated by emit. */
  activatedSnippets: ReadonlyArray<string>;
  /** Signature description so the host can build the koffi prototype. */
  signature: SpecCSignature;
  /** Synthetic entries only: outputs that may be left unassigned, each
   *  paired with a trailing flag output that is 0 when it was. The host
   *  skips writing such an output back. */
  assignedFlags: ReadonlyArray<AssignedFlag>;
  /** Files named by the statement locations a raise reports, by index
   *  (decode with `decodeLoc`). */
  locFiles: ReadonlyArray<string>;
  /** The entry spec can call itself (directly or through its callees). */
  recursive: boolean;
  /** Element-wise kernels in the entry spec (`ElemwiseKernel`). */
  kernels: number;
  /** Per parameter: the host passes this tensor as a pointer into its
   *  own buffer, which the entry neither writes, frees nor returns. */
  borrows: ReadonlyArray<boolean>;
}

/** C predeclares every local (no unassigned state) and fuses element-wise
 *  chains into single loops. */
export const cTarget: CompileTarget<EmitResult> = {
  definiteAssignment: true,
  fusion: C_FUSION,
  emit: (prog, exposeSpec) => emitProgram(prog, { exposeSpec }),
};

export function compileSpecC(args: CompileSpecArgs): CompileSpecCResult {
  const { cName, entry, assignedFlags, recursive, kernels, emitted } =
    compileWith(cTarget, args);
  return {
    cName,
    source: emitted.source,
    activatedSnippets: emitted.activatedSnippets,
    signature: { ...buildSignature(entry), cName: emitted.hostEntry },
    assignedFlags,
    locFiles: emitted.locFiles,
    recursive,
    kernels,
    borrows: emitted.borrows,
  };
}

function buildSignature(spec: IRFunc): SpecCSignature {
  const params: SpecCParam[] = spec.cParams.map((cName, i) => ({
    cName,
    ty: spec.paramTypes[i],
  }));
  const outputs: SpecCOutput[] = spec.cOutputs.map((cName, i) => ({
    cName,
    ty: spec.outputTypes[i],
  }));
  return { cName: spec.cName, params, outputs };
}
