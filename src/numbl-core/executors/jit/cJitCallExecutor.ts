/**
 * cjit-call — C-backed JIT executor for user-function calls.
 *
 * Mirrors `callExecutor.ts` (the JS-emit sibling) but routes the
 * specialization's emitted code through `compileSpecC` →
 * `cc -shared -fPIC -O3` → `dlopen` via koffi instead of
 * `new Function(source)`. The dispatch, gating, and caching
 * conventions are otherwise identical, so a sibling executor can
 * register alongside the JS path at the same opt level without any
 * arbitration in the dispatcher beyond cost-based selection.
 *
 * Declines when:
 *   - `nativeBridge` / `koffi` are not available (browser).
 *   - Any param or output type fails the C-FFI feasibility check
 *     (`compilerTypeToCDecl` returns null).
 *   - mtoc2 throws `UnsupportedConstruct` / `JitTypeError` at lowering.
 */

import type { Executor, Proposal, RunResult } from "../types.js";
import type { DispatchContext } from "../context.js";
import type { LoweredStmt } from "../lowering.js";
import { jitTypeKey, type JitType } from "../../jitTypes.js";
import type { Interpreter } from "../../interpreter/interpreter.js";
import type { Stmt, Span } from "../../parser/index.js";
import type { FunctionDef } from "../../interpreter/types.js";
import type { RuntimeValue } from "../../runtime/types.js";
import { isRuntimeTensor } from "../../runtime/types.js";
import { ensureRuntimeValue } from "../../runtime/runtimeHelpers.js";
import {
  compileSpecC,
  UnsupportedConstruct,
  JitTypeError,
  type Type as CompilerType,
  type SpecCSignature,
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
  makeCMarshalCtx,
  marshalInputs,
  unmarshalScalarOutput,
  unmarshalTensorOutput,
} from "./valueAdapterC.js";

type FuncStmt = Extract<Stmt, { type: "Function" }>;

// Native compile is significantly more expensive than `new Function`,
// but cached after first hit. Per-call FFI overhead dominates for tiny
// specs; large specs win on -O3 + vector codegen. Costs are tuned so
// the dispatcher prefers C-JIT for any case where it accepts.
const COST = { compileMs: 200, perCallNs: 80, runNs: 80 };

interface CJitCallData {
  readonly fn: FunctionDef;
  readonly nargout: number;
  readonly argTypes: readonly JitType[];
  readonly compilerArgTypes: readonly CompilerType[];
  readonly args: readonly unknown[];
}

interface CompiledArtifact {
  readonly compiled: CompiledC;
  readonly signature: SpecCSignature;
}

function synthesizeFuncStmt(fd: FunctionDef): FuncStmt {
  // Same as the JS path's synthesizeFuncStmt — see callExecutor.ts.
  const fromBody = fd.body[0]?.span?.file;
  const span: Span = { file: fromBody ?? "<jit>", start: 0, end: 0 };
  return {
    type: "Function",
    name: fd.name,
    functionId: fd.name,
    params: [...fd.params],
    outputs: [...fd.outputs],
    body: fd.body,
    argumentsBlocks: fd.argumentsBlocks ?? [],
    span,
  };
}

export const cJitCallExecutor: Executor<CJitCallData, CompiledArtifact | null> =
  {
    name: "cjit-call",

    propose(
      lowered: LoweredStmt,
      ctx: DispatchContext
    ): Proposal<CJitCallData> | null {
      if (lowered.kind !== "call") return null;
      const bridge = ctx.interp.nativeBridge;
      if (!bridge || !bridge.koffi) return null;
      if (ctx.interp.loopDepth > 0) return null;
      const classification = lowered.classification;
      if (classification.nargout === 0) return null;

      const compilerArgTypes: CompilerType[] = [];
      for (const jt of classification.argTypes) {
        const mt = jitTypeToCompilerType(jt);
        if (mt === null) return null;
        // The C path is more restrictive than the JS path — every type
        // also needs a C-FFI representation. Pre-check here so the
        // dispatcher doesn't even count a proposal that compile() will
        // reject.
        if (compilerTypeToCDecl(mt) === null) return null;
        compilerArgTypes.push(mt);
      }

      return {
        data: {
          fn: classification.fn,
          nargout: classification.nargout,
          argTypes: classification.argTypes,
          compilerArgTypes,
          args: lowered.args,
        },
        cost: COST,
        bailRisk: false,
      };
    },

    cacheKey(d): string {
      return (
        d.fn.name +
        "|" +
        d.argTypes.map(jitTypeKey).join(",") +
        "|n=" +
        d.nargout
      );
    },

    compile(d, ctx: DispatchContext): CompiledArtifact | null {
      const interp = ctx.interp as Interpreter;
      const bridge = interp.nativeBridge;
      if (!bridge || !bridge.koffi) return null;
      const { workspace, lowerer } = getOrCreateSession(interp);
      let source: string;
      let signature: SpecCSignature;
      let cName: string;
      try {
        const r = compileSpecC({
          workspace,
          lowerer,
          funcDecl: synthesizeFuncStmt(d.fn),
          argTypes: d.compilerArgTypes as CompilerType[],
          nargout: d.nargout,
        });
        source = r.source;
        signature = r.signature;
        cName = r.cName;
      } catch (e) {
        if (e instanceof UnsupportedConstruct || e instanceof JitTypeError) {
          return null;
        }
        throw e;
      }
      // Register the tensor struct lazily on first compile of any
      // C-JIT spec in this process. Idempotent.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      registerTensorStruct(bridge.koffi as any);
      const declaration = buildCDeclaration(signature);
      if (declaration === null) return null;
      const typeDesc = d.argTypes.map(jitTypeKey).join(", ");
      interp.onJitCompile?.(
        `cjit-call:${cName}(${typeDesc}) -> nargout=${d.nargout}`,
        source
      );
      const compiled = compileAndLoadC(source, declaration, bridge);
      return { compiled, signature };
    },

    run(compiled, d, ctx): RunResult {
      if (compiled === null) {
        return { bail: { message: "cjit-call: codegen declined" } };
      }
      const interp = ctx.interp as Interpreter;
      const bridge = interp.nativeBridge;
      if (!bridge || !bridge.koffi) {
        return { bail: { message: "cjit-call: nativeBridge gone" } };
      }
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const ctxC = makeCMarshalCtx(bridge.koffi as any, compiled.compiled.lib);
      try {
        const sig = compiled.signature;
        const inputs = marshalInputs(
          ctxC,
          sig.params.map(p => p.ty),
          d.args.map(v => v as RuntimeValue)
        );
        if (inputs === null) {
          return {
            bail: { message: "cjit-call: input marshal declined" },
          };
        }
        // Pre-allocate out-pointer args for multi-output specs. mtoc2's
        // convention appends one `T *_mtoc2_o<i>` per output after the
        // user params.
        const nOut = sig.outputs.length;
        const outAllocs: unknown[] = [];
        if (nOut >= 2) {
          for (let i = 0; i < nOut; i++) {
            const oTy = sig.outputs[i].ty;
            const cTy = compilerTypeToCDecl(oTy);
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const buf = (bridge.koffi as any).alloc(cTy, 1);
            outAllocs.push(buf);
            inputs.args.push(buf);
          }
        }
        const ret = compiled.compiled.fn(...inputs.args);
        inputs.release();
        // Materialize outputs.
        const results = readOutputs(ctxC, ret, outAllocs, sig);
        // Result shape matches the JS path: nargout==1 ⇒ bare value;
        // nargout>=2 ⇒ array.
        if (d.nargout >= 2) {
          return { result: results.map(v => ensureRuntimeValue(v)) };
        }
        return { result: ensureRuntimeValue(results[0]) };
      } catch (e) {
        return {
          bail: {
            message: `cjit-call: runtime error: ${e instanceof Error ? e.message : String(e)}`,
          },
        };
      }
    },
  };

interface CMarshalCtxLike {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  readonly koffi: any;
  readonly free: (ptr: unknown) => void;
}

/** Pull outputs out of the koffi call. For nargout==1 ret is the
 *  bare return; for nargout>=2 ret is undefined (void return) and
 *  outputs live in the pre-allocated `outAllocs` buffers. */
function readOutputs(
  ctxC: CMarshalCtxLike,
  ret: unknown,
  outAllocs: unknown[],
  sig: SpecCSignature
): unknown[] {
  const nOut = sig.outputs.length;
  if (nOut === 0) return [];
  if (nOut === 1) {
    const oTy = sig.outputs[0].ty;
    if (oTy.kind === "Numeric") {
      if (oTy.dims.some(d => !(d.kind === "exact" && d.value === 1))) {
        return [unmarshalTensorOutput(ctxC, ret, oTy)];
      }
      return [unmarshalScalarOutput(ret, oTy)];
    }
    return [ret];
  }
  // Multi-output: decode each out-pointer slot.
  const out: unknown[] = [];
  for (let i = 0; i < nOut; i++) {
    const oTy = sig.outputs[i].ty;
    const buf = outAllocs[i];
    if (oTy.kind === "Numeric") {
      const isTensor = oTy.dims.some(
        d => !(d.kind === "exact" && d.value === 1)
      );
      if (isTensor) {
        // The buf points to a `mtoc2_tensor_t` struct. Decode it,
        // then forward to the tensor unmarshaler. We re-use the
        // struct registration done at compile() time.
        const struct = ctxC.koffi.decode(buf, "mtoc2_tensor_t");
        out.push(unmarshalTensorOutput(ctxC, struct, oTy));
        continue;
      }
      // Both `double` and `logical` scalar outputs ride the `double`
      // slot in mtoc2's C ABI — see typeAdapterC.ts.
      const val = ctxC.koffi.decode(buf, "double", 1)[0];
      out.push(unmarshalScalarOutput(val, oTy));
      continue;
    }
    out.push(buf);
  }
  // Then a final case for the tensor isRuntimeTensor signature
  // already handled above.
  void isRuntimeTensor;
  return out;
}
