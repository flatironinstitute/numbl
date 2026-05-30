/**
 * cjit-loop — C-backed JIT executor for outermost For/While
 * loops. Mirrors `loopExecutor.ts` (JS-emit sibling) but routes
 * through `compileSpecC` → `cc -shared` → `koffi`. See
 * `cJitCallExecutor.ts` for the marshaling protocol; this file just
 * adapts the loop-shape `data` to the same C-side pipeline.
 */

import type { Executor, Proposal, RunResult } from "../types.js";
import type { DispatchContext } from "../context.js";
import type { LoweredStmt } from "../lowering.js";
import { jitTypeKey, type JitType } from "../../jitTypes.js";
import type { Stmt, Span } from "../../parser/index.js";
import type { Interpreter } from "../../interpreter/interpreter.js";
import type { RuntimeValue } from "../../runtime/types.js";
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
  bindHostWrite,
  makeCMarshalCtx,
  marshalInputs,
  unmarshalScalarOutput,
  unmarshalTensorOutput,
} from "./valueAdapterC.js";

type FuncStmt = Extract<Stmt, { type: "Function" }>;

const LOOP_COST = { compileMs: 200, perCallNs: 80, runNs: 80 };

interface CJitLoopData {
  readonly loopStmt: Stmt & { type: "For" | "While" };
  readonly inputs: readonly string[];
  readonly outputs: readonly string[];
  readonly inputTypes: readonly JitType[];
  readonly compilerInputTypes: readonly CompilerType[];
  readonly currentFile: string;
  readonly cacheKey: string;
}

interface CompiledArtifact {
  readonly compiled: CompiledC;
  readonly signature: SpecCSignature;
}

function synthesizeLoopFuncStmt(
  loopStmt: Stmt & { type: "For" | "While" },
  inputs: readonly string[],
  outputs: readonly string[],
  fileName: string
): FuncStmt {
  const offset = loopStmt.span?.start ?? 0;
  const name = `$loop_${offset}`;
  const span: Span = { file: fileName, start: 0, end: 0 };
  return {
    type: "Function",
    name,
    functionId: name,
    params: [...inputs],
    outputs: [...outputs],
    body: [loopStmt],
    argumentsBlocks: [],
    span,
  };
}

export const cJitLoopExecutor: Executor<CJitLoopData, CompiledArtifact | null> =
  {
    name: "cjit-loop",

    propose(
      lowered: LoweredStmt,
      ctx: DispatchContext
    ): Proposal<CJitLoopData> | null {
      if (lowered.kind !== "loop") return null;
      const bridge = ctx.interp.nativeBridge;
      if (!bridge || !bridge.koffi) return null;
      if (ctx.interp.loopDepth > 0) return null;
      const classification = lowered.classification;
      if (classification.hasReturn) return null;

      const compilerInputTypes: CompilerType[] = [];
      for (const jt of classification.inputTypes) {
        const mt = jitTypeToCompilerType(jt);
        if (mt === null) return null;
        if (compilerTypeToCDecl(mt) === null) return null;
        compilerInputTypes.push(mt);
      }

      return {
        data: {
          loopStmt: classification.stmt,
          inputs: classification.inputs,
          outputs: classification.outputs,
          inputTypes: classification.inputTypes,
          compilerInputTypes,
          currentFile: classification.currentFile,
          cacheKey: classification.cacheKey,
        },
        cost: LOOP_COST,
        bailRisk: false,
      };
    },

    cacheKey(d): string {
      return d.cacheKey;
    },

    compile(d, ctx: DispatchContext): CompiledArtifact | null {
      const interp = ctx.interp as Interpreter;
      const bridge = interp.nativeBridge;
      if (!bridge || !bridge.koffi) return null;
      const { workspace, lowerer } = getOrCreateSession(interp);
      const funcDecl = synthesizeLoopFuncStmt(
        d.loopStmt,
        d.inputs,
        d.outputs,
        d.currentFile
      );
      const nargout = d.outputs.length;
      let source: string;
      let signature: SpecCSignature;
      let cName: string;
      try {
        const r = compileSpecC({
          workspace,
          lowerer,
          funcDecl,
          argTypes: d.compilerInputTypes as CompilerType[],
          nargout,
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
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      registerTensorStruct(bridge.koffi as any);
      const declaration = buildCDeclaration(signature);
      if (declaration === null) return null;
      const typeDesc = d.inputTypes.map(jitTypeKey).join(", ");
      interp.onJitCompile?.(
        `cjit-loop:${cName}(${typeDesc}) -> outputs=${d.outputs.length}`,
        source
      );
      const compiled = compileAndLoadC(source, declaration, bridge);
      return { compiled, signature };
    },

    run(compiled, d, ctx: DispatchContext): RunResult {
      if (compiled === null) {
        return { bail: { message: "cjit-loop: codegen declined" } };
      }
      const interp = ctx.interp as Interpreter;
      const bridge = interp.nativeBridge;
      if (!bridge || !bridge.koffi) {
        return { bail: { message: "cjit-loop: nativeBridge gone" } };
      }
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const ctxC = makeCMarshalCtx(bridge.koffi as any, compiled.compiled.lib);
      const hostWrite = bindHostWrite(
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        bridge.koffi as any,
        compiled.compiled.lib,
        (s: string) => interp.rt.output(s)
      );
      try {
        const sig = compiled.signature;
        const values: (RuntimeValue | undefined)[] = d.inputs.map(name =>
          interp.env.get(name)
        );
        const inputs = marshalInputs(
          ctxC,
          sig.params.map(p => p.ty),
          values
        );
        if (inputs === null) {
          return {
            bail: { message: "cjit-loop: input marshal declined" },
          };
        }
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
        // Write outputs back into env.
        if (nOut === 0) {
          // No live-after-loop assigns.
        } else if (nOut === 1) {
          const oTy = sig.outputs[0].ty;
          let value: unknown;
          if (oTy.kind === "Numeric") {
            const isTensor = oTy.dims.some(
              x => !(x.kind === "exact" && x.value === 1)
            );
            value = isTensor
              ? unmarshalTensorOutput(ctxC, ret, oTy)
              : unmarshalScalarOutput(ret, oTy);
          } else {
            value = ret;
          }
          interp.env.set(
            d.outputs[0],
            ensureRuntimeValue(value) as RuntimeValue
          );
        } else {
          for (let i = 0; i < nOut; i++) {
            const oTy = sig.outputs[i].ty;
            const buf = outAllocs[i];
            let value: unknown;
            if (oTy.kind === "Numeric") {
              const isTensor = oTy.dims.some(
                x => !(x.kind === "exact" && x.value === 1)
              );
              if (isTensor) {
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                const struct = (bridge.koffi as any).decode(
                  buf,
                  "mtoc2_tensor_t"
                );
                value = unmarshalTensorOutput(ctxC, struct, oTy);
              } else {
                // Logical scalar outputs ride the `double` slot in
                // mtoc2's C ABI — see typeAdapterC.ts.
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                const val = (bridge.koffi as any).decode(buf, "double", 1)[0];
                value = unmarshalScalarOutput(val, oTy);
              }
            } else {
              value = buf;
            }
            interp.env.set(
              d.outputs[i],
              ensureRuntimeValue(value) as RuntimeValue
            );
          }
        }
        return { ok: true };
      } catch (e) {
        return {
          bail: {
            message: `cjit-loop: runtime error: ${e instanceof Error ? e.message : String(e)}`,
          },
        };
      } finally {
        hostWrite.dispose();
      }
    },
  };
