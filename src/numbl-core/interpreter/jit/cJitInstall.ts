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

import { registerCJitBackend } from "./cJitBackend.js";
import { checkCFeasibility } from "./c/cFeasibility.js";
import { generateC, type COutputDesc } from "./c/jitCodegenC.js";
import { compileAndLoad, cJitOpenmpAvailable } from "./c/cCompile.js";
import { jitTypeKey } from "./jitTypes.js";
import { type RuntimeTensor } from "../../runtime/types.js";
import { uninitFloat64 } from "../../runtime/alloc.js";
import { getTicTime, setTicTime } from "../builtins/time-system.js";

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
      argTypes,
      outputType,
      outputTypes,
      nargout
    );
    if (!feas.ok) return null;

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
    } catch {
      return null;
    }

    const useOmp = interp.par && cJitOpenmpAvailable();
    const loaded = compileAndLoad(
      gen.cSource,
      gen.koffiSignature,
      gen.cFnName,
      interp.log,
      useOmp ? ["-fopenmp"] : undefined
    );
    if (!loaded) return null;

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
        "double get_monotonic_time(void)"
      ) as () => number;
      ticStateBuf = new Float64Array(1);
    }

    return (...callArgs: unknown[]): unknown => {
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
        buf: Float64Array;
        lenBuf?: Float64Array;
        reusedTensor?: RuntimeTensor;
      }> = [];

      for (const od of outputDescs) {
        if (od.kind === "tensor") {
          // Try to reuse the dest tensor's buffer. The dest is whatever
          // the variable currently holds in the caller's scope — which is
          // the corresponding callArg if it's a tensor param with the same
          // name, or undefined for a fresh local. For simplicity, we always
          // allocate a fresh buffer here. The C function writes into it
          // directly. We can optimize reuse later.
          const n = firstTensorLen;
          const buf = uninitFloat64(n);
          const lenBuf = new Float64Array(1);
          koffiArgs.push(buf);
          koffiArgs.push(lenBuf);
          outputBufs.push({ desc: od, buf, lenBuf });
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

      // Call the C function.
      nativeFn(...koffiArgs);

      // tic/toc: convert C clock domain back to JS ticTime.
      if (ticStateBuf) {
        setTicTime((ticStateBuf[0] + ticJsNow - ticCNow) * 1000);
      }

      // Extract results.
      if (isMultiOutput) {
        const results: unknown[] = [];
        for (const ob of outputBufs) {
          if (ob.desc.kind === "tensor") {
            const shape = firstTensorShape
              ? firstTensorShape.slice()
              : [1, firstTensorLen];
            const tensor: RuntimeTensor = {
              kind: "tensor",
              data: ob.buf,
              shape,
              _rc: 1,
            };
            results.push(tensor);
          } else if (ob.desc.kind === "boolean") {
            results.push(ob.buf[0] !== 0);
          } else {
            results.push(ob.buf[0]);
          }
        }
        return results;
      }

      // Single output.
      const ob = outputBufs[0];
      if (!ob) return 0; // no-output function
      if (ob.desc.kind === "tensor") {
        const shape = firstTensorShape
          ? firstTensorShape.slice()
          : [1, firstTensorLen];
        const tensor: RuntimeTensor = {
          kind: "tensor",
          data: ob.buf,
          shape,
          _rc: 1,
        };
        return tensor;
      }
      if (ob.desc.kind === "boolean") {
        return ob.buf[0] !== 0;
      }
      return ob.buf[0];
    };
  },
});
