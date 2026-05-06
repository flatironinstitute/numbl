/**
 * c-jit-fuse — element-wise tensor op fusion.
 *
 *   - `propose()` filters on `lowered.kind === "fuse"`. Reads each
 *     tensor input's `numel` from env and verifies they all match;
 *     declines below a per-call threshold so small tensors stay on
 *     the interpreter path.
 *
 *   - `compile()` emits a single-pass C kernel, builds the koffi
 *     declaration, calls `compileAndLoad`. Surfaces the source via
 *     `interp.onCJitCompile` for `--dump-c`.
 *
 *   - `run()` gathers tensor data + scalar values, allocates an
 *     output `Float64Array(numel)`, invokes the kernel, wraps the
 *     output as a real `RuntimeTensor` whose shape mirrors the first
 *     tensor input's, and writes back.
 */

import type { Executor, Proposal, RunResult } from "../types.js";
import type { DispatchContext } from "../context.js";
import type { LoweredStmt, FuseLoweredStmt } from "../lowering.js";
import { RuntimeTensor, type RuntimeValue } from "../../runtime/types.js";
import { generateFuseCSource } from "./fuseCodegen.js";
import { compileAndLoad, type CompiledC } from "./compile.js";
import { allocFloat64Array } from "../jsJit/helpers/alloc.js";

/** Minimum tensor numel to pay the koffi-dispatch overhead. Below
 *  this, the interpreter's vectorized builtin path (Float64Array
 *  arithmetic in V8) is competitive or better. The constant is
 *  picked conservatively; tune later if benchmarks suggest. */
const FUSE_MIN_NUMEL = 1024;

interface CFuseCompiled {
  readonly compiled: CompiledC;
  readonly tensorInputs: readonly string[];
  readonly scalarInputs: readonly string[];
  readonly source: string;
}

const C_JIT_FUSE_COST = { compileMs: 150, perCallNs: 200, runNs: 100 };

export const cJitFuseExecutor: Executor<FuseLoweredStmt, CFuseCompiled | null> =
  {
    name: "c-jit-fuse",

    propose(
      lowered: LoweredStmt,
      ctx: DispatchContext
    ): Proposal<FuseLoweredStmt> | null {
      if (lowered.kind !== "fuse") return null;
      if (!ctx.interp.nativeBridge) return null;

      // All tensor inputs must be present, real Float64, and have a
      // matching numel. (Re-check at run() too — this is the propose-
      // time fast bail.)
      const cls = lowered.classification;
      let n = -1;
      for (const name of cls.tensorInputs) {
        const v = ctx.envValue(name) as RuntimeValue | undefined;
        if (v === undefined) return null;
        if (typeof v !== "object" || v === null) return null;
        const t = v as RuntimeTensor;
        if (t.kind !== "tensor") return null;
        if (t.imag !== undefined) return null;
        if (!(t.data instanceof Float64Array)) return null;
        const len = t.data.length;
        if (n === -1) n = len;
        else if (n !== len) return null;
      }
      if (n < FUSE_MIN_NUMEL) return null;

      // Scalar inputs must look like scalar numbers.
      for (const name of cls.scalarInputs) {
        const v = ctx.envValue(name);
        if (typeof v !== "number" && typeof v !== "boolean") return null;
      }

      return {
        data: lowered,
        cost: C_JIT_FUSE_COST,
        bailRisk: false,
      };
    },

    cacheKey(d): string {
      return d.classification.cacheKey;
    },

    compile(d, ctx: DispatchContext): CFuseCompiled | null {
      const bridge = ctx.interp.nativeBridge;
      if (!bridge) return null;

      const cls = d.classification;
      const fnName = `c_jit_fuse_${cls.outputName}`;
      const source = generateFuseCSource(fnName, cls);

      const params: string[] = [];
      for (let i = 0; i < cls.tensorInputs.length; i++) {
        params.push(`const double *t${i}`);
      }
      for (let i = 0; i < cls.scalarInputs.length; i++) {
        params.push(`double s${i}`);
      }
      const declaration = `void ${fnName}(double *out, long n${
        params.length > 0 ? ", " + params.join(", ") : ""
      })`;

      const line = ctx.interp.rt.$line ?? 0;
      ctx.interp.onCJitCompile?.(
        `fuse:${cls.outputName}@${line}(t=[${cls.tensorInputs.join(
          ","
        )}], s=[${cls.scalarInputs.join(",")}])`,
        source
      );

      let compiled: CompiledC;
      try {
        compiled = compileAndLoad(source, declaration, bridge, {
          fastMath: ctx.interp.fastMath,
        });
      } catch (e) {
        console.warn(
          `Warning: c-jit-fuse compile failed; falling back to interpreter. ${
            e instanceof Error ? e.message : String(e)
          }`
        );
        return null;
      }

      return {
        compiled,
        tensorInputs: cls.tensorInputs,
        scalarInputs: cls.scalarInputs,
        source,
      };
    },

    run(compiled, d, ctx: DispatchContext): RunResult {
      if (compiled === null) {
        return { bail: { message: "c-jit-fuse: codegen rejected" } };
      }

      const interp = ctx.interp;

      // Gather tensor pointers + verify shapes match (re-check —
      // env may have shifted between propose() and run()).
      const tensorArgs: Float64Array[] = [];
      let n = -1;
      let templateShape: readonly number[] = [];
      for (const name of compiled.tensorInputs) {
        const v = interp.env.get(name);
        if (v === undefined || typeof v !== "object" || v === null) {
          return {
            bail: { message: `c-jit-fuse: input ${name} missing at run time` },
            transient: true,
          };
        }
        const t = v as RuntimeTensor;
        if (t.kind !== "tensor" || t.imag !== undefined) {
          return {
            bail: { message: `c-jit-fuse: input ${name} not a real tensor` },
            transient: true,
          };
        }
        if (!(t.data instanceof Float64Array)) {
          return {
            bail: { message: `c-jit-fuse: input ${name} is not Float64` },
            transient: true,
          };
        }
        const len = t.data.length;
        if (n === -1) {
          n = len;
          templateShape = t.shape;
        } else if (n !== len) {
          return {
            bail: {
              message: `c-jit-fuse: tensor input shapes don't match (${name})`,
            },
            transient: true,
          };
        }
        tensorArgs.push(t.data);
      }

      const scalarArgs: number[] = [];
      for (const name of compiled.scalarInputs) {
        const v = interp.env.get(name);
        if (typeof v === "number") scalarArgs.push(v);
        else if (typeof v === "boolean") scalarArgs.push(v ? 1 : 0);
        else {
          return {
            bail: {
              message: `c-jit-fuse: scalar input ${name} is not a number`,
            },
            transient: true,
          };
        }
      }

      const out = allocFloat64Array(n);
      try {
        (compiled.compiled.fn as (...a: unknown[]) => unknown)(
          out,
          n,
          ...tensorArgs,
          ...scalarArgs
        );
      } catch (e) {
        return {
          bail: {
            message: `c-jit-fuse: native invocation failed: ${
              e instanceof Error ? e.message : String(e)
            }`,
          },
        };
      }

      // Wrap as RuntimeTensor preserving the input's shape.
      const result = new RuntimeTensor(out, [...templateShape]);
      interp.env.set(d.classification.outputName, result as RuntimeValue);
      return { ok: true };
    },
  };
