/**
 * c-jit-chain — element-wise chain executor.
 *
 *   - `propose()` filters on Synth stmts with `tag === "c-jit-chain"`.
 *     Reads each tensor input's `numel` from env and verifies they
 *     all match. Declines below `CHAIN_MIN_NUMEL`.
 *
 *   - `compile()` resolves per-name roles using runtime env types,
 *     emits the combined C kernel, builds the koffi declaration,
 *     calls compileAndLoad. Surfaces source via `interp.onCJitCompile`.
 *
 *   - `run()` gathers tensor data + scalars, allocates one
 *     `Float64Array(numel)` per live-out, invokes the kernel, writes
 *     each live-out back to env as a `RuntimeTensor` with the input
 *     tensors' shape.
 */

import type { Executor, Proposal, RunResult } from "../types.js";
import type { DispatchContext } from "../context.js";
import type { LoweredStmt, SynthLoweredStmt } from "../lowering.js";
import { RuntimeTensor, type RuntimeValue } from "../../runtime/types.js";
import type { ChainAnalysis } from "./chainPass.js";
import {
  buildChainDeclaration,
  generateChainCSource,
  resolveChain,
  type ChainResolution,
} from "./chainCodegen.js";
import { collectIdents } from "./elemwiseCodegen.js";
import { compileAndLoad, type CompiledC } from "./compile.js";
import { allocFloat64Array } from "../jsJit/helpers/alloc.js";

/** Minimum tensor numel to pay the koffi-dispatch + per-call alloc
 *  overhead. Below this, the interpreter / individual c-jit-fuse
 *  path is competitive. */
const CHAIN_MIN_NUMEL = 1024;

interface ChainCompiled {
  readonly compiled: CompiledC;
  readonly resolution: ChainResolution;
  readonly source: string;
}

const C_JIT_CHAIN_COST = { compileMs: 250, perCallNs: 200, runNs: 50 };

export const cJitChainExecutor: Executor<
  SynthLoweredStmt,
  ChainCompiled | null
> = {
  name: "c-jit-chain",

  propose(
    lowered: LoweredStmt,
    ctx: DispatchContext
  ): Proposal<SynthLoweredStmt> | null {
    if (lowered.kind !== "synth") return null;
    if (lowered.tag !== "c-jit-chain") return null;
    if (!ctx.interp.nativeBridge) return null;

    const analysis = lowered.data as ChainAnalysis;

    // Check only RHS-read idents — LHS-written names are produced
    // by the chain and don't need to be in env. Track which names
    // are RHS-read before being written; those are the chain inputs.
    let n = -1;
    const seenRhsRead = new Set<string>();
    const writtenSoFar = new Set<string>();
    for (const a of analysis.assigns) {
      if (a.type !== "Assign") continue;
      const rhsNames = new Set<string>();
      collectIdents(a.expr, rhsNames);
      for (const name of rhsNames) {
        if (writtenSoFar.has(name)) continue; // local already
        if (seenRhsRead.has(name)) continue;
        seenRhsRead.add(name);
        const v = ctx.envValue(name) as RuntimeValue | undefined;
        if (v === undefined) return null;
        if (typeof v === "number" || typeof v === "boolean") continue;
        if (typeof v !== "object" || v === null) return null;
        const t = v as RuntimeTensor;
        if (t.kind !== "tensor") return null;
        if (t.imag !== undefined) return null;
        if (!(t.data instanceof Float64Array)) return null;
        if (n === -1) n = t.data.length;
        else if (n !== t.data.length) return null;
      }
      writtenSoFar.add(a.name);
    }

    // Phase A guarantee: at least one tensor input. (A pure-scalar
    // chain has nothing to fuse — let the interpreter handle it.)
    if (n === -1) return null;
    if (n < CHAIN_MIN_NUMEL) return null;

    return {
      data: lowered,
      cost: C_JIT_CHAIN_COST,
      bailRisk: false,
    };
  },

  cacheKey(d): string {
    return (d.data as ChainAnalysis).cacheKey;
  },

  compile(d, ctx: DispatchContext): ChainCompiled | null {
    const bridge = ctx.interp.nativeBridge;
    if (!bridge) return null;

    const analysis = d.data as ChainAnalysis;

    // Resolve each name's role using the live env.
    const isTensor = (name: string): boolean | null => {
      const v = ctx.interp.env.get(name) as RuntimeValue | undefined;
      if (v === undefined) return null;
      if (typeof v === "number" || typeof v === "boolean") return false;
      if (typeof v !== "object" || v === null) return null;
      const t = v as RuntimeTensor;
      if (t.kind !== "tensor") return null;
      return true;
    };
    const resolution = resolveChain(analysis, isTensor);
    if (!resolution) return null;

    const fnName = `c_jit_chain_${resolution.liveOuts.join("_") || "void"}`;
    const source = generateChainCSource(fnName, analysis, resolution);
    const declaration = buildChainDeclaration(fnName, resolution);

    const line = ctx.interp.rt.$line ?? 0;
    ctx.interp.onCJitCompile?.(
      `chain[${analysis.assigns.length}]@${line}(out=[${resolution.liveOuts.join(",")}], t=[${resolution.tensorInputs.join(",")}], s=[${resolution.scalarInputs.join(",")}])`,
      source
    );

    let compiled: CompiledC;
    try {
      compiled = compileAndLoad(source, declaration, bridge, {
        fastMath: ctx.interp.fastMath,
      });
    } catch (e) {
      console.warn(
        `Warning: c-jit-chain compile failed; falling back. ${
          e instanceof Error ? e.message : String(e)
        }`
      );
      return null;
    }

    return { compiled, resolution, source };
  },

  run(compiled, _d, ctx: DispatchContext): RunResult {
    if (compiled === null) {
      return { bail: { message: "c-jit-chain: codegen rejected" } };
    }

    const interp = ctx.interp;
    const res = compiled.resolution;

    // Re-gather inputs from env, verify matching numel.
    const tensorArgs: Float64Array[] = [];
    let n = -1;
    let templateShape: readonly number[] = [];
    for (const name of res.tensorInputs) {
      const v = interp.env.get(name);
      if (v === undefined || typeof v !== "object" || v === null) {
        return {
          bail: { message: `c-jit-chain: input ${name} missing` },
          transient: true,
        };
      }
      const t = v as RuntimeTensor;
      if (
        t.kind !== "tensor" ||
        t.imag !== undefined ||
        !(t.data instanceof Float64Array)
      ) {
        return {
          bail: {
            message: `c-jit-chain: input ${name} not real Float64 tensor`,
          },
          transient: true,
        };
      }
      const len = t.data.length;
      if (n === -1) {
        n = len;
        templateShape = t.shape;
      } else if (n !== len) {
        return {
          bail: { message: `c-jit-chain: input ${name} shape mismatch` },
          transient: true,
        };
      }
      tensorArgs.push(t.data);
    }

    // For chains with no tensor inputs (a pure local-only chain),
    // we need numel from a live-out's role. But propose() already
    // declined if there were no tensors at all, so this branch
    // shouldn't fire — guard defensively.
    if (n === -1) {
      return {
        bail: { message: "c-jit-chain: no tensor input at run time" },
        transient: true,
      };
    }

    const scalarArgs: number[] = [];
    for (const name of res.scalarInputs) {
      const v = interp.env.get(name);
      if (typeof v === "number") scalarArgs.push(v);
      else if (typeof v === "boolean") scalarArgs.push(v ? 1 : 0);
      else {
        return {
          bail: { message: `c-jit-chain: scalar ${name} not numeric` },
          transient: true,
        };
      }
    }

    // Allocate one fresh Float64Array per live-out.
    const outs: Float64Array[] = [];
    for (let k = 0; k < res.liveOuts.length; k++) {
      outs.push(allocFloat64Array(n));
    }

    try {
      (compiled.compiled.fn as (...a: unknown[]) => unknown)(
        n,
        ...outs,
        ...tensorArgs,
        ...scalarArgs
      );
    } catch (e) {
      return {
        bail: {
          message: `c-jit-chain: native invocation failed: ${
            e instanceof Error ? e.message : String(e)
          }`,
        },
      };
    }

    // Wrap each live-out as a RuntimeTensor with the matching shape.
    for (let k = 0; k < res.liveOuts.length; k++) {
      const name = res.liveOuts[k];
      const result = new RuntimeTensor(outs[k], [...templateShape]);
      interp.env.set(name, result as RuntimeValue);
    }
    return { ok: true };
  },
};
