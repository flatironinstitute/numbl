/**
 * Installs the C-JIT backend (koffi path). Side-effect import only.
 *
 * Must be imported exactly once from a Node-only entry point (currently
 * src/cli.ts). The browser bundle never reaches this file, so the
 * Node-only dependencies stay out of the web build.
 *
 * The JS wrapper handles:
 *   - Extracting .data (Float64Array) and .data.length from RuntimeTensor args
 *   - Pre-allocating output buffers (Float64Array for tensor outputs,
 *     Float64Array(1) for scalar out-pointers)
 *   - Buffer reuse for tensor outputs (same logic as jitHelpersTensor.ts)
 *   - Wrapping results back into RuntimeTensor objects
 */

import { createRequire } from "module";
import { registerCJitBackend } from "./cJitBackend.js";
import { checkCFeasibility } from "./cFeasibility.js";
import { generateC, type COutputDesc } from "./jitCodegenC.js";
import { compileAndLoad, cJitOpenmpAvailable } from "./cCompile.js";
import { jitTypeKey } from "../jitTypes.js";
import { type RuntimeTensor } from "../../runtime/types.js";
import { uninitFloat64 } from "../../runtime/alloc.js";
import {
  getTicTime,
  setTicTime,
} from "../../interpreter/builtins/time-system.js";
import { JitBailToInterpreter } from "../js/jitHelpers.js";

// koffi is loaded lazily by cCompile.ts. We need direct access for
// decode() / free() of the dynamic-output C pointers. Mirrors the
// lazy require pattern from cCompile.ts to keep the web bundle clean.
let _koffi: typeof import("koffi") | null | undefined;
function getKoffi(): typeof import("koffi") {
  if (_koffi) return _koffi;
  const req = createRequire(import.meta.url);
  _koffi = req("koffi") as typeof import("koffi");
  return _koffi;
}

registerCJitBackend({
  tryCompile(
    interp,
    fn,
    body,
    outputNames,
    localVars,
    outputType,
    outputTypes,
    argTypes,
    nargout
  ) {
    const feas = checkCFeasibility(
      body,
      fn.params,
      argTypes,
      outputType,
      outputTypes,
      nargout
    );
    if (!feas.ok) {
      return {
        ok: false,
        kind: "infeasible",
        reason: feas.reason,
        line: feas.line,
      };
    }

    let gen;
    try {
      gen = generateC(
        body,
        fn.params,
        outputNames,
        nargout,
        localVars,
        argTypes,
        outputType,
        outputTypes,
        fn.name.replace(/[^A-Za-z0-9_]/g, "_"),
        interp.fuse,
        interp.par && cJitOpenmpAvailable()
      );
    } catch (e) {
      // Codegen throws indicate a construct the feasibility check let
      // through but the emitter can't handle — still an infeasible-IR
      // parity gap, not an env error.
      const msg = e instanceof Error ? e.message : String(e);
      return { ok: false, kind: "infeasible", reason: `codegen: ${msg}` };
    }

    const useOmp = interp.par && cJitOpenmpAvailable();
    const loaded = compileAndLoad(
      gen.cSource,
      gen.koffiSignature,
      gen.cFnName,
      interp.log,
      useOmp ? ["-fopenmp"] : undefined
    );
    if (!loaded) {
      return {
        ok: false,
        kind: "env",
        reason: "C compile/load failed (see verbose log)",
      };
    }

    // Fire --dump-c callback.
    const line = interp.rt.$line ?? 0;
    const typeDesc = argTypes.map(jitTypeKey).join(", ");
    const description = `${fn.name}@${line}(${typeDesc}) -> nargout=${nargout}`;
    interp.onCJitCompile?.(description, gen.cSource);

    const nativeFn = loaded.fn;
    const paramDescs = gen.paramDescs;
    const outputDescs = gen.outputDescs;
    const isMultiOutput = outputDescs.length > 1;

    // tic/toc support: declare get_monotonic_time, allocate shared buffer.
    let getMonotonicTime: (() => number) | undefined;
    let ticStateBuf: Float64Array | undefined;
    if (gen.needsTicState) {
      const lib = loaded.lib;
      getMonotonicTime = lib.func(
        "double numbl_monotonic_time(void)"
      ) as () => number;
      ticStateBuf = new Float64Array(1);
    }

    // Bounds-error flag: shared single-slot buffer. The C helper writes
    // 1.0 on out-of-bounds index reads; we reset before each call and
    // check after.
    const errorFlagBuf: Float64Array | undefined = gen.needsErrorFlag
      ? new Float64Array(1)
      : undefined;

    const compiledFn = (...callArgs: unknown[]): unknown => {
      // Build the koffi call arguments: extract data/len from tensors,
      // pass scalars directly, append output buffers/out-pointers.
      const koffiArgs: unknown[] = [];

      // Track the first tensor input's shape for tensor outputs.
      let firstTensorShape: number[] | undefined;
      let firstTensorLen = 0;

      for (let i = 0; i < paramDescs.length; i++) {
        const pd = paramDescs[i];
        const arg = callArgs[i];
        if (pd.kind === "tensor") {
          const t = arg as RuntimeTensor;
          const data = t.data as Float64Array;
          koffiArgs.push(data);
          koffiArgs.push(data.length);
          // Multi-index ABI: tensors indexed with arity >= 2 also
          // receive their column-major shape dims (d0 = rows, d1 = cols
          // for 3D) so the emitted C can compute 2D/3D offsets. Matches
          // the JS-JIT hoist which exposes $<name>_d0 / $<name>_d1.
          if (pd.ndim && pd.ndim >= 2) {
            const s = t.shape;
            const d0 = s.length >= 1 ? s[0] : 1;
            koffiArgs.push(d0);
            if (pd.ndim >= 3) {
              const d1 = s.length >= 2 ? s[1] : 1;
              koffiArgs.push(d1);
            }
          }
          if (!firstTensorShape) {
            firstTensorShape = t.shape;
            firstTensorLen = data.length;
          }
        } else {
          koffiArgs.push(
            typeof arg === "boolean" ? (arg ? 1 : 0) : (arg as number)
          );
        }
      }

      // Prepare output buffers.
      const outputBufs: Array<{
        desc: COutputDesc;
        buf?: Float64Array;
        lenBuf?: Float64Array;
        /** Dynamic-output: single-element array slot the C function
         *  fills via `*out = malloc(...)`. We decode the pointer post-
         *  call, copy its contents into a fresh Float64Array, and free
         *  the C allocation. */
        dynPtrSlot?: (unknown | null)[];
        dynLenSlot?: bigint[];
        dynD0Slot?: bigint[];
        dynD1Slot?: bigint[];
        /** When the output shares its name with a tensor param, this is
         *  the param's shape — used when wrapping the result so the output
         *  keeps the caller's tensor shape rather than whatever "first
         *  tensor input" it happened to see. */
        paramShape?: number[];
      }> = [];

      for (const od of outputDescs) {
        if (od.kind === "tensor") {
          if (od.dynamic) {
            // Dynamic-output ABI: pass a 4-slot set of out-params. The
            // JS wrapper decodes the pointer after the call, copies the
            // data into a fresh Float64Array, and frees the C pointer.
            const dynPtrSlot: (unknown | null)[] = [null];
            const dynLenSlot: bigint[] = [0n];
            const dynD0Slot: bigint[] = [0n];
            const dynD1Slot: bigint[] = [0n];
            koffiArgs.push(dynPtrSlot);
            koffiArgs.push(dynLenSlot);
            koffiArgs.push(dynD0Slot);
            koffiArgs.push(dynD1Slot);
            outputBufs.push({
              desc: od,
              dynPtrSlot,
              dynLenSlot,
              dynD0Slot,
              dynD1Slot,
            });
            continue;
          }
          // Fixed output: the JS wrapper preallocates the buffer.
          // If the output name matches a tensor param, MATLAB's `function
          // x = foo(x, ...)` call-by-value + local-mutation semantics
          // apply: the callee starts with a copy of caller's x and may
          // mutate it locally. Seed the output buffer with the caller's
          // data so AssignIndex writes against the expected starting
          // state. Otherwise (fresh output) allocate uninitialized — the
          // C function is expected to overwrite it via whole-tensor ops.
          const paramIdx = paramDescs.findIndex(
            p => p.kind === "tensor" && p.name === od.name
          );
          if (paramIdx >= 0) {
            const t = callArgs[paramIdx] as RuntimeTensor;
            const srcData = t.data as Float64Array;
            const buf = new Float64Array(srcData); // copy
            const lenBuf = new Float64Array(1);
            koffiArgs.push(buf);
            koffiArgs.push(lenBuf);
            outputBufs.push({ desc: od, buf, lenBuf, paramShape: t.shape });
          } else {
            const n = firstTensorLen;
            const buf = uninitFloat64(n);
            const lenBuf = new Float64Array(1);
            koffiArgs.push(buf);
            koffiArgs.push(lenBuf);
            outputBufs.push({ desc: od, buf, lenBuf });
          }
        } else {
          const buf = new Float64Array(1);
          koffiArgs.push(buf);
          outputBufs.push({ desc: od, buf });
        }
      }

      // tic/toc: convert JS ticTime to C clock domain before the call.
      let ticJsNow = 0;
      let ticCNow = 0;
      if (ticStateBuf && getMonotonicTime) {
        ticCNow = getMonotonicTime();
        ticJsNow = performance.now() / 1000;
        ticStateBuf[0] = getTicTime() / 1000 - ticJsNow + ticCNow;
        koffiArgs.push(ticStateBuf);
      }

      // Reset and append the bounds-error flag.
      if (errorFlagBuf) {
        errorFlagBuf[0] = 0;
        koffiArgs.push(errorFlagBuf);
      }

      // Call the C function.
      nativeFn(...koffiArgs);

      // tic/toc: convert C clock domain back to JS ticTime.
      if (ticStateBuf) {
        setTicTime((ticStateBuf[0] + ticJsNow - ticCNow) * 1000);
      }

      // Error-flag conventions written by the jit_runtime helpers:
      //   1.0 — OOB read → hard bounds error (matches JS-JIT's idx1r_h)
      //   2.0 — OOB write → soft-bail: interpreter re-runs with proper
      //         tensor-growth semantics (matches JS-JIT's set1r_h /
      //         setCol2r_h's col-past-end branch)
      //   3.0 — length mismatch on range / column slice write (matches
      //         JS-JIT's setRange1r_h / setCol2r_h Error message)
      if (errorFlagBuf && errorFlagBuf[0] !== 0) {
        if (errorFlagBuf[0] === 2) {
          throw new JitBailToInterpreter(
            "scalar index write requires tensor growth"
          );
        }
        if (errorFlagBuf[0] === 3) {
          throw new Error(
            "Unable to perform assignment because the indices on the left side are not compatible with the size of the right side."
          );
        }
        throw new Error("Index exceeds array bounds");
      }

      // Extract results. Tensor outputs keep the shape of the matching
      // input param when output-name == param-name (the `function x =
      // foo(x, ...)` pattern); otherwise fall back to the first tensor
      // input's shape.
      const tensorShapeFor = (ob: (typeof outputBufs)[number]): number[] =>
        ob.paramShape
          ? ob.paramShape.slice()
          : firstTensorShape
            ? firstTensorShape.slice()
            : [1, firstTensorLen];

      // Build a RuntimeTensor from a dynamic-output slot set: copy the
      // C-owned pointer contents into a fresh JS Float64Array, then free
      // the C pointer. This runs exactly once per call-per-output so the
      // extra memcpy is negligible next to the JIT / koffi overhead.
      const readDynamicTensor = (
        ob: (typeof outputBufs)[number]
      ): RuntimeTensor => {
        const ptr = ob.dynPtrSlot![0];
        const d0 = Number(ob.dynD0Slot![0]);
        const d1 = Number(ob.dynD1Slot![0]);
        const len = Number(ob.dynLenSlot![0]);
        if (!ptr || len <= 0) {
          return {
            kind: "tensor",
            data: new Float64Array(0),
            shape: [d0, d1],
            _rc: 1,
          };
        }
        const koffi = getKoffi();
        const src = koffi.decode(ptr, "double", len) as Float64Array;
        const data = new Float64Array(src);
        koffi.free(ptr);
        return {
          kind: "tensor",
          data,
          shape: [d0, d1],
          _rc: 1,
        };
      };

      if (isMultiOutput) {
        const results: unknown[] = [];
        for (const ob of outputBufs) {
          if (ob.desc.kind === "tensor") {
            if (ob.desc.dynamic) {
              results.push(readDynamicTensor(ob));
            } else {
              const tensor: RuntimeTensor = {
                kind: "tensor",
                data: ob.buf!,
                shape: tensorShapeFor(ob),
                _rc: 1,
              };
              results.push(tensor);
            }
          } else if (ob.desc.kind === "boolean") {
            results.push(ob.buf![0] !== 0);
          } else {
            results.push(ob.buf![0]);
          }
        }
        return results;
      }

      // Single output.
      const ob = outputBufs[0];
      if (!ob) return 0; // no-output function
      if (ob.desc.kind === "tensor") {
        if (ob.desc.dynamic) return readDynamicTensor(ob);
        const tensor: RuntimeTensor = {
          kind: "tensor",
          data: ob.buf!,
          shape: tensorShapeFor(ob),
          _rc: 1,
        };
        return tensor;
      }
      if (ob.desc.kind === "boolean") {
        return ob.buf![0] !== 0;
      }
      return ob.buf![0];
    };

    return { ok: true, fn: compiledFn };
  },
});
