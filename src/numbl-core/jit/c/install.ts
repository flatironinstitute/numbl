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
import { registerCJitBackend } from "./registry.js";
import { checkCFeasibility } from "./feasibility.js";
import { generateC, type AbiSlot, type COutputDesc } from "./assemble.js";
import { compileAndLoad, cJitOpenmpAvailable } from "./compile.js";
import { jitTypeKey } from "../jitTypes.js";
import {
  type RuntimeTensor,
  type RuntimeComplexNumber,
  isRuntimeComplexNumber,
} from "../../runtime/types.js";
import { uninitFloat64 } from "../../runtime/alloc.js";
import {
  getTicTime,
  setTicTime,
} from "../../interpreter/builtins/time-system.js";
import { JitBailToInterpreter } from "../js/jitHelpers.js";

// koffi is loaded lazily by compile.ts. We need direct access for
// decode() / free() of the dynamic-output C pointers. Mirrors the
// lazy require pattern from compile.ts to keep the web bundle clean.
let _koffi: typeof import("koffi") | null | undefined;
function getKoffi(): typeof import("koffi") {
  if (_koffi) return _koffi;
  const req = createRequire(import.meta.url);
  _koffi = req("koffi") as typeof import("koffi");
  return _koffi;
}

/** Register the `NumblDispCb` callback type with koffi exactly once.
 *  The C-JIT emits `NumblDispCb *` in the function signature; koffi needs
 *  the proto to be declared before any `lib.func()` that references it. */
let _dispCbProtoRegistered = false;
function ensureDispCbProto(): void {
  if (_dispCbProtoRegistered) return;
  _dispCbProtoRegistered = true;
  const k = getKoffi() as unknown as {
    proto: (sig: string) => unknown;
  };
  k.proto("void NumblDispCb(const char *, double, int)");
}

/** Match the JS-JIT's `formatNumber` (runtime/display.ts) so C-emitted
 *  `disp(x)` produces byte-identical output to JS-emitted `disp(x)`. */
function formatJitNumber(n: number): string {
  if (Number.isInteger(n) && Math.abs(n) < 1e15) {
    return n.toString();
  }
  const s = n.toPrecision(5);
  if (s.includes(".")) {
    return s.replace(/\.?0+$/, "") || "0";
  }
  return s;
}

/** Per-output buffer record passed between the JS wrapper's output-alloc
 *  step and the slot-marshalling + result-extraction steps. Shared with
 *  the closure-local `outputBufs` below. */
type OutputBuf = {
  desc: COutputDesc;
  buf?: Float64Array;
  /** Imaginary companion for complex scalar and complex fixed-tensor
   *  outputs. Pairs with `buf` which holds the real part. */
  imBuf?: Float64Array;
  lenBuf?: Float64Array;
  dynPtrSlot?: (unknown | null)[];
  /** Imaginary companion pointer slot for complex dynamic tensor
   *  outputs. The C callee mallocs both buffers and transfers
   *  ownership via two `double **` out-params; the wrapper decodes +
   *  frees both. */
  dynPtrImSlot?: (unknown | null)[];
  dynLenSlot?: bigint[];
  dynD0Slot?: bigint[];
  dynD1Slot?: bigint[];
  paramShape?: number[];
};

/** Given an ABI slot, return the JS value to push at the matching koffi
 *  arg position. Dispatches on `slot.kind`; param-sourced slots read
 *  from `callArgs[slot.paramIdx]`, output-sourced from
 *  `outputBufs[slot.outputIdx]`, trailers from shared buffers. */
function marshalSlot(
  slot: AbiSlot,
  callArgs: unknown[],
  outputBufs: OutputBuf[],
  ticStateBuf: Float64Array | undefined,
  errorFlagBuf: Float64Array | undefined,
  dispCb: unknown
): unknown {
  switch (slot.kind) {
    case "scalar": {
      const v = callArgs[slot.paramIdx!];
      return typeof v === "boolean" ? (v ? 1 : 0) : (v as number);
    }
    case "complexScalarRe": {
      const v = callArgs[slot.paramIdx!];
      if (
        isRuntimeComplexNumber(
          v as import("../../runtime/types.js").RuntimeValue
        )
      ) {
        return (v as RuntimeComplexNumber).re;
      }
      return typeof v === "boolean" ? (v ? 1 : 0) : (v as number);
    }
    case "complexScalarIm": {
      const v = callArgs[slot.paramIdx!];
      if (
        isRuntimeComplexNumber(
          v as import("../../runtime/types.js").RuntimeValue
        )
      ) {
        return (v as RuntimeComplexNumber).im;
      }
      return 0;
    }
    case "tensorData":
      return (callArgs[slot.paramIdx!] as RuntimeTensor).data as Float64Array;
    case "tensorDataIm": {
      // NULL when imag is absent — numbl_ops complex kernels treat that
      // as "all zero", avoiding a per-call zero malloc.
      const t = callArgs[slot.paramIdx!] as RuntimeTensor;
      return (t.imag as Float64Array | undefined) ?? null;
    }
    case "tensorLen": {
      const t = callArgs[slot.paramIdx!] as RuntimeTensor;
      return (t.data as Float64Array).length;
    }
    case "tensorD0": {
      const t = callArgs[slot.paramIdx!] as RuntimeTensor;
      return t.shape.length >= 1 ? t.shape[0] : 1;
    }
    case "tensorD1": {
      const t = callArgs[slot.paramIdx!] as RuntimeTensor;
      return t.shape.length >= 2 ? t.shape[1] : 1;
    }
    case "scalarOut":
    case "fixedOutBuf":
      return outputBufs[slot.outputIdx!].buf!;
    case "fixedOutBufIm":
      return outputBufs[slot.outputIdx!].imBuf!;
    case "complexScalarReOut":
      return outputBufs[slot.outputIdx!].buf!;
    case "complexScalarImOut":
      return outputBufs[slot.outputIdx!].imBuf!;
    case "fixedOutLen":
      return outputBufs[slot.outputIdx!].lenBuf!;
    case "dynOutBuf":
      return outputBufs[slot.outputIdx!].dynPtrSlot!;
    case "dynOutBufIm":
      return outputBufs[slot.outputIdx!].dynPtrImSlot!;
    case "dynOutLen":
      return outputBufs[slot.outputIdx!].dynLenSlot!;
    case "dynOutD0":
      return outputBufs[slot.outputIdx!].dynD0Slot!;
    case "dynOutD1":
      return outputBufs[slot.outputIdx!].dynD1Slot!;
    case "ticState":
      return ticStateBuf!;
    case "errFlag":
      return errorFlagBuf!;
    case "dispCb":
      return dispCb;
  }
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
    nargout,
    generatedIRBodies
  ) {
    const feas = checkCFeasibility(
      body,
      fn.params,
      argTypes,
      outputType,
      outputTypes,
      nargout,
      generatedIRBodies
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
        interp.par && cJitOpenmpAvailable(),
        generatedIRBodies
      );
    } catch (e) {
      // Codegen throws indicate a construct the feasibility check let
      // through but the emitter can't handle — still an infeasible-IR
      // parity gap, not an env error.
      const msg = e instanceof Error ? e.message : String(e);
      return { ok: false, kind: "infeasible", reason: `codegen: ${msg}` };
    }

    // If the generated signature references NumblDispCb, register the
    // proto globally before compileAndLoad declares the koffi function.
    if (gen.needsDispCb) ensureDispCbProto();

    // Link with -fopenmp unconditionally when the compiler supports it so
    // that numbl_ops.a (which has `#pragma omp parallel for` in bessel.c)
    // can be safely linked into any JIT module regardless of --par.
    // --par still controls whether fused loops get parallel pragmas via
    // the `interp.par && cJitOpenmpAvailable()` flag passed to generateC.
    const ompLink = cJitOpenmpAvailable();
    const loaded = compileAndLoad(
      gen.cSource,
      gen.koffiSignature,
      gen.cFnName,
      interp.log,
      ompLink ? ["-fopenmp"] : undefined
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
    const abiSlots = gen.abiSlots;
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

    // disp callback: route C-emitted disp calls back into `rt.output`.
    // Registered once per compiled-fn, captures `interp` via closure so
    // lazy-resolved `rt` picks up the current interp run. The callback
    // ptr is passed to the native fn as a trailer arg on each call.
    let dispCb: unknown = null;
    if (gen.needsDispCb) {
      const k = getKoffi() as unknown as {
        pointer: (name: string) => unknown;
        register: (fn: unknown, ptrType: unknown) => unknown;
      };
      const cb = (sPtr: string | null, num: number, kind: number): void => {
        if (kind === 0) {
          interp.rt.output((sPtr ?? "") + "\n");
        } else {
          interp.rt.output(formatJitNumber(num) + "\n");
        }
      };
      dispCb = k.register(cb, k.pointer("NumblDispCb"));
    }

    const compiledFn = (...callArgs: unknown[]): unknown => {
      // Track the first tensor input's shape so scalar-shaped outputs
      // can inherit it when no explicit output shape is known.
      let firstTensorShape: number[] | undefined;
      let firstTensorLen = 0;
      for (let i = 0; i < paramDescs.length; i++) {
        if (paramDescs[i].kind === "tensor") {
          const t = callArgs[i] as RuntimeTensor;
          firstTensorShape = t.shape;
          firstTensorLen = (t.data as Float64Array).length;
          break;
        }
      }

      // Prepare output buffers — one entry per output. Indexed by
      // `outputIdx` stored on the output-sourced ABI slots.
      const outputBufs: OutputBuf[] = outputDescs.map(od => {
        if (od.kind === "complexScalar") {
          return {
            desc: od,
            buf: new Float64Array(1),
            imBuf: new Float64Array(1),
          };
        }
        if (od.kind === "tensor") {
          if (od.dynamic) {
            const rec: OutputBuf = {
              desc: od,
              dynPtrSlot: [null],
              dynLenSlot: [0n],
              dynD0Slot: [0n],
              dynD1Slot: [0n],
            };
            if (od.isComplex) rec.dynPtrImSlot = [null];
            return rec;
          }
          // Fixed output: seed from the matching tensor param when the
          // output name is also a param (`function x = foo(x, ...)`
          // MATLAB call-by-value + local-mutation pattern); otherwise
          // allocate uninitialized and let the C code fill it via
          // whole-tensor ops. Complex outputs allocate a paired imag
          // buffer (seeded from param's `.imag` when available, else
          // zeroed).
          const paramIdx = paramDescs.findIndex(
            p => p.kind === "tensor" && p.name === od.name
          );
          if (paramIdx >= 0) {
            const t = callArgs[paramIdx] as RuntimeTensor;
            const buf = new Float64Array(t.data as Float64Array); // copy
            const lenBuf = new Float64Array(1);
            const rec: OutputBuf = {
              desc: od,
              buf,
              lenBuf,
              paramShape: t.shape,
            };
            if (od.isComplex) {
              rec.imBuf = t.imag
                ? new Float64Array(t.imag as Float64Array)
                : new Float64Array(buf.length);
            }
            return rec;
          }
          const buf = uninitFloat64(firstTensorLen);
          const lenBuf = new Float64Array(1);
          const rec: OutputBuf = { desc: od, buf, lenBuf };
          if (od.isComplex) rec.imBuf = uninitFloat64(firstTensorLen);
          return rec;
        }
        // scalar / boolean outputs: single-slot double buffer.
        return { desc: od, buf: new Float64Array(1) };
      });

      // tic/toc: convert JS ticTime to C clock domain before the call.
      let ticJsNow = 0;
      let ticCNow = 0;
      if (ticStateBuf && getMonotonicTime) {
        ticCNow = getMonotonicTime();
        ticJsNow = performance.now() / 1000;
        ticStateBuf[0] = getTicTime() / 1000 - ticJsNow + ticCNow;
      }
      if (errorFlagBuf) errorFlagBuf[0] = 0;

      // Build the koffi arg list by walking the ABI slot schedule. The
      // order of slots matches the C signature exactly; each slot's
      // `kind` tells us which JS value to push, and the paramIdx /
      // outputIdx backrefs name the source.
      const koffiArgs: unknown[] = abiSlots.map(slot =>
        marshalSlot(
          slot,
          callArgs,
          outputBufs,
          ticStateBuf,
          errorFlagBuf,
          dispCb
        )
      );

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
      // extra memcpy is negligible next to the JIT / koffi overhead. For
      // complex dynamic outputs also decodes + frees the imag buffer.
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
        const t: RuntimeTensor = {
          kind: "tensor",
          data,
          shape: [d0, d1],
          _rc: 1,
        };
        if (ob.desc.isComplex && ob.dynPtrImSlot) {
          const ptrIm = ob.dynPtrImSlot[0];
          if (ptrIm) {
            const srcIm = koffi.decode(ptrIm, "double", len) as Float64Array;
            t.imag = new Float64Array(srcIm);
            koffi.free(ptrIm);
          }
        }
        return t;
      };

      /** Finalize a fixed tensor output. For complex tensors, only
       *  attach `.imag` if it's non-zero anywhere — matches runtime
       *  convention where `imag === undefined` means "all zero". Saves
       *  downstream consumers an imag-is-all-zero branch. */
      const readFixedTensor = (ob: OutputBuf): RuntimeTensor => {
        const t: RuntimeTensor = {
          kind: "tensor",
          data: ob.buf!,
          shape: tensorShapeFor(ob),
          _rc: 1,
        };
        if (ob.desc.isComplex && ob.imBuf) {
          let anyNonZero = false;
          const im = ob.imBuf;
          for (let i = 0; i < im.length; i++) {
            if (im[i] !== 0) {
              anyNonZero = true;
              break;
            }
          }
          if (anyNonZero) t.imag = im;
        }
        return t;
      };

      /** Decode a complex scalar output as either a bare number (im=0)
       *  or a RuntimeComplexNumber. Mirrors `mkc` from
       *  interpreter/builtins/types.ts so the result's runtime shape
       *  matches what the interpreter and JS-JIT paths produce. */

      const readComplexScalar = (
        ob: OutputBuf
      ): number | RuntimeComplexNumber => {
        const re = ob.buf![0];
        const im = ob.imBuf![0];
        if (im === 0) return re;
        return { kind: "complex_number", re, im };
      };

      if (isMultiOutput) {
        const results: unknown[] = [];
        for (const ob of outputBufs) {
          if (ob.desc.kind === "tensor") {
            results.push(
              ob.desc.dynamic ? readDynamicTensor(ob) : readFixedTensor(ob)
            );
          } else if (ob.desc.kind === "boolean") {
            results.push(ob.buf![0] !== 0);
          } else if (ob.desc.kind === "complexScalar") {
            results.push(readComplexScalar(ob));
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
        return ob.desc.dynamic ? readDynamicTensor(ob) : readFixedTensor(ob);
      }
      if (ob.desc.kind === "boolean") {
        return ob.buf![0] !== 0;
      }
      if (ob.desc.kind === "complexScalar") {
        return readComplexScalar(ob);
      }
      return ob.buf![0];
    };

    return { ok: true, fn: compiledFn };
  },
});
