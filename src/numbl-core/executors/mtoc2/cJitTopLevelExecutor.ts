/**
 * mtoc2-cjit-top-level — C-backed JIT executor for whole-script bodies.
 *
 * Mirrors `topLevelExecutor.ts` (JS-emit sibling) but routes through
 * `compileSpecC` → `cc -shared` → `koffi`. See `cJitCallExecutor.ts`
 * for the marshaling protocol. Same `isAllSuppressed` gate as the JS
 * sibling so `displayAssign`/`displayResult` semantics stay on the
 * interpreter path.
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
  Mtoc2TypeError,
  type Type as Mtoc2Type,
  type SpecCSignature,
} from "../../mtoc2/index.js";
import { jitTypeToMtoc2Type } from "./typeAdapter.js";
import { getOrCreateSession } from "./session.js";
import { compileAndLoadC, type CompiledC } from "./compileC.js";
import {
  buildCDeclaration,
  mtoc2TypeToCDecl,
  registerMtoc2TensorStruct,
} from "./typeAdapterC.js";
import {
  makeCMarshalCtx,
  marshalInputs,
  unmarshalScalarOutput,
  unmarshalTensorOutput,
} from "./valueAdapterC.js";

type FuncStmt = Extract<Stmt, { type: "Function" }>;

const TOP_LEVEL_COST = { compileMs: 200, perCallNs: 200, runNs: 200 };

interface Mtoc2CJitTopLevelData {
  readonly stmts: readonly Stmt[];
  readonly inputs: readonly string[];
  readonly outputs: readonly string[];
  readonly inputTypes: readonly JitType[];
  readonly mtoc2InputTypes: readonly Mtoc2Type[];
  readonly currentFile: string;
  readonly cacheKey: string;
}

interface CompiledArtifact {
  readonly compiled: CompiledC;
  readonly signature: SpecCSignature;
}

/** Same shape as the JS sibling's gate — decline scripts whose top
 *  level has any unsuppressed Assign/MultiAssign/ExprStmt, since
 *  those trigger numbl's display hooks. */
function isAllSuppressed(stmts: readonly Stmt[]): boolean {
  for (const s of stmts) {
    if (s.type === "ExprStmt" && !s.suppressed) return false;
    if (s.type === "Assign" && !s.suppressed) return false;
    if (s.type === "MultiAssign" && !s.suppressed) return false;
  }
  return true;
}

function synthesizeTopLevelFuncStmt(
  stmts: readonly Stmt[],
  inputs: readonly string[],
  outputs: readonly string[],
  fileName: string
): FuncStmt {
  const span: Span = { file: fileName, start: 0, end: 0 };
  return {
    type: "Function",
    name: "$top",
    functionId: "$top",
    params: [...inputs],
    outputs: [...outputs],
    body: [...stmts] as Stmt[],
    argumentsBlocks: [],
    span,
  };
}

export const mtoc2CJitTopLevelExecutor: Executor<
  Mtoc2CJitTopLevelData,
  CompiledArtifact | null
> = {
  name: "mtoc2-cjit-top-level",

  propose(
    lowered: LoweredStmt,
    ctx: DispatchContext
  ): Proposal<Mtoc2CJitTopLevelData> | null {
    if (lowered.kind !== "top-level") return null;
    const bridge = ctx.interp.nativeBridge;
    if (!bridge || !bridge.koffi) return null;
    if (ctx.interp.loopDepth > 0) return null;
    const classification = lowered.classification;
    if (classification.hasReturn) return null;
    if (!isAllSuppressed(classification.stmts)) return null;

    const mtoc2InputTypes: Mtoc2Type[] = [];
    for (const jt of classification.inputTypes) {
      const mt = jitTypeToMtoc2Type(jt);
      if (mt === null) return null;
      if (mtoc2TypeToCDecl(mt) === null) return null;
      mtoc2InputTypes.push(mt);
    }

    return {
      data: {
        stmts: classification.stmts,
        inputs: classification.inputs,
        outputs: classification.outputs,
        inputTypes: classification.inputTypes,
        mtoc2InputTypes,
        currentFile: classification.currentFile,
        cacheKey: classification.cacheKey,
      },
      cost: TOP_LEVEL_COST,
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
    const funcDecl = synthesizeTopLevelFuncStmt(
      d.stmts,
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
        argTypes: d.mtoc2InputTypes as Mtoc2Type[],
        nargout,
      });
      source = r.source;
      signature = r.signature;
      cName = r.cName;
    } catch (e) {
      if (e instanceof UnsupportedConstruct || e instanceof Mtoc2TypeError) {
        return null;
      }
      throw e;
    }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    registerMtoc2TensorStruct(bridge.koffi as any);
    const declaration = buildCDeclaration(signature);
    if (declaration === null) return null;
    const typeDesc = d.inputTypes.map(jitTypeKey).join(", ");
    interp.onJitCompile?.(
      `mtoc2-cjit-top-level:${cName}(${typeDesc}) -> outputs=${d.outputs.length}`,
      source
    );
    const compiled = compileAndLoadC(source, declaration, bridge);
    return { compiled, signature };
  },

  run(compiled, d, ctx: DispatchContext): RunResult {
    if (compiled === null) {
      return { bail: { message: "mtoc2-cjit-top-level: codegen declined" } };
    }
    const interp = ctx.interp as Interpreter;
    const bridge = interp.nativeBridge;
    if (!bridge || !bridge.koffi) {
      return { bail: { message: "mtoc2-cjit-top-level: nativeBridge gone" } };
    }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const ctxC = makeCMarshalCtx(bridge.koffi as any, compiled.compiled.lib);
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
          bail: { message: "mtoc2-cjit-top-level: input marshal declined" },
        };
      }
      const nOut = sig.outputs.length;
      const outAllocs: unknown[] = [];
      if (nOut >= 2) {
        for (let i = 0; i < nOut; i++) {
          const oTy = sig.outputs[i].ty;
          const cTy = mtoc2TypeToCDecl(oTy);
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const buf = (bridge.koffi as any).alloc(cTy, 1);
          outAllocs.push(buf);
          inputs.args.push(buf);
        }
      }
      const ret = compiled.compiled.fn(...inputs.args);
      inputs.release();
      if (nOut === 0) {
        // Suppressed top-level with no live-out assigns.
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
        interp.env.set(d.outputs[0], ensureRuntimeValue(value) as RuntimeValue);
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
          message: `mtoc2-cjit-top-level: runtime error: ${e instanceof Error ? e.message : String(e)}`,
        },
      };
    }
  },
};
