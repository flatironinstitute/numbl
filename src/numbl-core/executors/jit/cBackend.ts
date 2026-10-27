/**
 * C backend: `compileSpecC` → `cc -shared` → koffi `dlopen`.
 *
 * Node only. Imported solely by `compileC.node.ts`, which registers it at
 * bootstrap, so the browser bundle never pulls in the C compile graph.
 * See `valueAdapterC.ts` for the marshaling protocol.
 */

import { ensureRuntimeValue } from "../../runtime/runtimeHelpers.js";
import {
  compileSpecC,
  type CompileSpecCResult,
  type NumericType,
  type SpecCSignature,
  type Type,
} from "../../jit/index.js";
import { jitTypeToCompilerType } from "./typeAdapter.js";
import { getOrCreateSession } from "./session.js";
import { compileAndLoadC, type CompiledC } from "./compileC.js";
import {
  buildCDeclaration,
  compilerTypeToCDecl,
  registerTensorStruct,
} from "./typeAdapterC.js";
import {
  bindGrowBail,
  bindHostWrite,
  makeCMarshalCtx,
  marshalInputs,
  unmarshalScalarOutput,
  unmarshalTensorOutput,
  type HostWriteBinding,
} from "./valueAdapterC.js";
import type { JitBackend } from "./backend.js";

// koffi ships no TS types for its declaration-string API.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Koffi = any;

interface CArtifact {
  readonly compiled: CompiledC;
  readonly signature: SpecCSignature;
  readonly assignedFlags: CompileSpecCResult["assignedFlags"];
  readonly paramTypes: Type[];
  // Bound once per loaded `.so`; `lib.func` lookups are too slow per call.
  readonly ctx: ReturnType<typeof makeCMarshalCtx>;
  readonly hostWrite: HostWriteBinding;
  readonly growBail: ReturnType<typeof bindGrowBail>;
}

export const cBackend: JitBackend<CArtifact> = {
  lang: "c",
  prefix: "cjit",

  // A runtime error in compiled C exits the process (see `oob.h`), so
  // nothing under a `try` may run as C.
  available: interp => !!interp.nativeBridge?.koffi && interp.tryDepth === 0,

  // Every type also needs a C-FFI representation.
  mapType(jt) {
    const t = jitTypeToCompilerType(jt);
    return t !== null && compilerTypeToCDecl(t) !== null ? t : null;
  },

  compile(req, interp) {
    const bridge = interp.nativeBridge;
    if (!bridge?.koffi) return null;
    const { workspace, lowerer } = getOrCreateSession(interp);
    const { source, signature, cName, assignedFlags } = compileSpecC({
      workspace,
      lowerer,
      funcDecl: req.funcDecl,
      argTypes: req.argTypes,
      nargout: req.nargout,
      entrySynthetic: req.synthetic,
    });
    registerTensorStruct(bridge.koffi as Koffi);
    const declaration = buildCDeclaration(signature);
    if (declaration === null) return null;
    interp.onJitCompile?.(`${req.where}:${cName}${req.describe}`, source, "c");
    const compiled = compileAndLoadC(source, declaration, bridge);
    const koffi = bridge.koffi as Koffi;
    return {
      compiled,
      signature,
      assignedFlags,
      paramTypes: signature.params.map(p => p.ty),
      ctx: makeCMarshalCtx(koffi, compiled.lib),
      hostWrite: bindHostWrite(koffi, compiled.lib),
      growBail: bindGrowBail(compiled.lib),
    };
  },

  invoke(art, args, interp, where) {
    const koffi = interp.nativeBridge?.koffi as Koffi;
    if (!koffi) return null;
    const { ctx, growBail, signature: sig } = art;
    const inputs = marshalInputs(ctx, art.paramTypes, args);
    if (inputs === null) return null;
    // Multi-output specs take one `T *_mtoc2_o<i>` out-pointer per
    // output after the params.
    const outAllocs: unknown[] = [];
    try {
      if (sig.outputs.length >= 2) {
        for (const o of sig.outputs) {
          const buf = koffi.alloc(compilerTypeToCDecl(o.ty), 1);
          outAllocs.push(buf);
          inputs.args.push(buf);
        }
      }
      growBail.reset();
      let ret: unknown;
      try {
        ret = art.hostWrite.run(
          s => interp.rt.output(s),
          () => art.compiled.fn(...inputs.args)
        );
      } catch (e) {
        // koffi rejected an argument before the call; the callee never
        // took ownership of the input buffers.
        inputs.release();
        throw e;
      }
      if (growBail.bailed()) {
        interp.onJitBail?.(
          `${where}: indexed-store array growth; falling back to interpreter`
        );
        return null;
      }
      const outs = readOutputs(koffi, ctx, ret, outAllocs, sig);
      // An output whose flag stayed 0 was never assigned: return it as
      // undefined so it isn't written back, then drop the flag slots.
      for (const { output, flag } of art.assignedFlags) {
        if (outs[flag] === 0) outs[output] = undefined;
      }
      outs.length -= art.assignedFlags.length;
      return outs.map(v => (v === undefined ? v : ensureRuntimeValue(v)));
    } finally {
      for (const buf of outAllocs) koffi.free(buf);
    }
  },
};

function isTensorType(ty: NumericType): boolean {
  return ty.dims.some(d => !(d.kind === "exact" && d.value === 1));
}

/** Pull outputs out of the koffi call. With one output `ret` is the bare
 *  return; with several, `ret` is void and each output lives in its
 *  pre-allocated `outAllocs` buffer. */
function readOutputs(
  koffi: Koffi,
  ctxC: ReturnType<typeof makeCMarshalCtx>,
  ret: unknown,
  outAllocs: unknown[],
  sig: SpecCSignature
): unknown[] {
  const nOut = sig.outputs.length;
  if (nOut === 0) return [];
  if (nOut === 1) {
    const oTy = sig.outputs[0].ty;
    if (oTy.kind !== "Numeric") return [ret];
    return [
      isTensorType(oTy)
        ? unmarshalTensorOutput(ctxC, ret, oTy)
        : unmarshalScalarOutput(ret, oTy),
    ];
  }
  return sig.outputs.map((o, i) => {
    const oTy = o.ty;
    const buf = outAllocs[i];
    if (oTy.kind !== "Numeric") return buf;
    if (isTensorType(oTy)) {
      return unmarshalTensorOutput(
        ctxC,
        koffi.decode(buf, "mtoc2_tensor_t"),
        oTy
      );
    }
    // Complex scalars ride the `mtoc2_cscalar_t` struct slot; real and
    // logical scalars ride the `double` slot (see typeAdapterC.ts).
    if (oTy.isComplex) {
      return unmarshalScalarOutput(koffi.decode(buf, "mtoc2_cscalar_t"), oTy);
    }
    return unmarshalScalarOutput(koffi.decode(buf, "double", 1)[0], oTy);
  });
}
