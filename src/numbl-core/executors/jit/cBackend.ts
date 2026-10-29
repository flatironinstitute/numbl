/**
 * C backend: `compileSpecC` → `cc -shared` → koffi `dlopen`.
 *
 * Node only. Imported solely by `cjit.node.ts`, which hands it to a host
 * (`NumblHost.cjit`), so the browser bundle never pulls in the C compile
 * graph. It also needs the host's koffi bridge (`NumblHost.nativeBridge`).
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
import { getOrCreateSession } from "./session.js";
import type { CompiledC } from "./compileC.js";
import { compileAndLoadC } from "./compileC.node.js";
import {
  buildCDeclaration,
  compilerTypeToCDecl,
  registerTensorStruct,
} from "./typeAdapterC.js";
import {
  bindCancel,
  bindHostNumeric,
  bindHostWrite,
  bindRaise,
  makeCMarshalCtx,
  marshalInputs,
  unmarshalScalarOutput,
  unmarshalTensorOutput,
  type HostWriteBinding,
} from "./valueAdapterC.js";
import { outputIsReproducible, type JitBackend } from "./backend.js";
import { cOutputSink } from "./hostHelpers.js";
import { JitRaise } from "./raise.js";
import { decodeLoc } from "../../jit/codegen/runtime.js";

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
  readonly raise: ReturnType<typeof bindRaise>;
  readonly cancel: ReturnType<typeof bindCancel>;
  readonly locFiles: ReadonlyArray<string>;
  readonly recursive: boolean;
  readonly streamsOutput: boolean;
  readonly kernels: number;
  readonly borrows: ReadonlyArray<boolean>;
}

/** `.so`s with a call in progress (see `invoke`). */
const runningLibs = new Set<string>();

export const cBackend: JitBackend<CArtifact> = {
  prefix: "cjit",
  native: true,
  strictDisplayGate: true,

  available: interp => !!interp.rt.host.nativeBridge?.koffi,

  // Every type also needs a C-FFI representation.
  accepts: t => compilerTypeToCDecl(t) !== null,

  compile(req, interp) {
    const bridge = interp.rt.host.nativeBridge;
    if (!bridge?.koffi) return null;
    const { workspace, lowerer } = getOrCreateSession(interp);
    const {
      source,
      signature,
      cName,
      assignedFlags,
      locFiles,
      recursive,
      kernels,
      activatedSnippets,
      borrows,
    } = compileSpecC({
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
    bindHostNumeric(koffi, compiled.lib);
    return {
      compiled,
      signature,
      assignedFlags,
      paramTypes: signature.params.map(p => p.ty),
      ctx: makeCMarshalCtx(koffi, compiled.lib),
      hostWrite: bindHostWrite(koffi, compiled.lib),
      raise: bindRaise(compiled.lib),
      cancel: bindCancel(compiled.lib, compiled.libPath),
      locFiles,
      recursive,
      streamsOutput: outputIsReproducible(activatedSnippets),
      kernels,
      borrows,
    };
  },

  recursive: art => art.recursive,

  streamsOutput: art => art.streamsOutput,

  kernels: art => art.kernels,

  invoke(art, args, interp) {
    const koffi = interp.rt.host.nativeBridge?.koffi as Koffi;
    if (!koffi) return null;
    const { ctx, signature: sig } = art;
    // A `.so`'s raise guard and allocator list are file-scope state, so it
    // cannot be re-entered, as it could be from an output callback (output
    // streams while the call runs). Bail instead.
    const libPath = art.compiled.libPath;
    if (runningLibs.has(libPath)) return null;
    // Point the compiled cancel polls at this runtime's flag.
    art.cancel.install(interp.rt.cancelFlag);
    const inputs = marshalInputs(ctx, art.paramTypes, args, art.borrows);
    if (inputs === null) return null;
    // Multi-output specs take one `T *_jit_o<i>` out-pointer per
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
      let ret: unknown;
      const sink = cOutputSink(interp.rt);
      runningLibs.add(libPath);
      try {
        ret = art.hostWrite.run(sink.write, () =>
          art.compiled.fn(...inputs.args)
        );
        sink.flush();
      } catch (e) {
        // koffi rejected an argument before the call; the callee never
        // took ownership of the input buffers.
        inputs.release();
        throw e;
      } finally {
        runningLibs.delete(libPath);
      }
      // A raise `longjmp`ed back to the host-entry guard, which freed the
      // call's allocations (the input buffers included) and returned a
      // zeroed value; report it instead of reading outputs.
      const st = art.raise.status();
      if (st !== null) {
        throw new JitRaise(
          st.kind === 1 ? "bail" : st.kind === 3 ? "cancel" : "error",
          st.id,
          st.msg,
          decodeLoc(st.loc, art.locFiles)
        );
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
        koffi.decode(buf, "jit_tensor_t"),
        oTy
      );
    }
    // Complex scalars ride the `jit_cscalar_t` struct slot; real and
    // logical scalars ride the `double` slot (see typeAdapterC.ts).
    if (oTy.isComplex) {
      return unmarshalScalarOutput(koffi.decode(buf, "jit_cscalar_t"), oTy);
    }
    return unmarshalScalarOutput(koffi.decode(buf, "double", 1)[0], oTy);
  });
}
