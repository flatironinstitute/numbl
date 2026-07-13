/**
 * Optional WASM linear-algebra accelerator bridge.
 *
 * numbl can offload heavy LAPACK/BLAS kernels to an externally-hosted
 * WebAssembly module (e.g. libFLAME/BLIS compiled to wasm, served from
 * GitHub Pages). This is the *platform-agnostic core*: it operates on an
 * already-instantiated wasm `exports` object, so it has no `fetch` /
 * `WebAssembly.instantiate` dependency and runs identically under Node and
 * the browser. The browser worker owns the fetch + instantiate step and
 * hands the exports here (see the browser loader); Node bundling can reuse
 * the same functions later.
 *
 * Two seams are installed from one instantiated module (see
 * `installWasmLapackBridge`):
 *   1. `setLapackBridge()` — covers the *interpreter* path (opt 0), which
 *      already resolves per-op via `getEffectiveBridge` and falls back to
 *      ts-lapack for any op the bridge omits.
 *   2. `globalThis.$matmulAccel` — the hook the JS-JIT matmul runtime
 *      snippet consults (opt 1, the browser IDE default), which otherwise
 *      runs a naive JS triple-loop that never touches the bridge.
 *
 * Trust model: numbl instantiates only the module's *wasm* (never remote
 * JS) with a numbl-controlled import object, and does all marshaling here,
 * so the accelerator is sandboxed — it cannot reach the DOM or network.
 *
 * v1 exposes real f64 `matmul` only. The ABI and manifest are versioned so
 * more ops (svd/qr/lu/…) can be added without breaking older endpoints.
 */

import { setLapackBridge, type LapackBridge } from "./lapack-bridge.js";

/** Bumped when the wasm ABI (export names / signatures) changes
 *  incompatibly. An endpoint manifest must advertise the same major. */
export const ACCELERATOR_ABI = 1;

/** Below this many multiply-adds (`m*n*k`) the JS loop beats the cost of
 *  marshaling into and out of the wasm heap, so the accelerator declines
 *  and the caller uses its own loop. Tuned conservatively; small matmuls
 *  dominate typical MATLAB code and should never pay the copy tax. */
export const WASM_MATMUL_MIN_WORK = 100_000;

/** Manifest served alongside the wasm (`numbl-bridge.json`). Describes
 *  which ops/precisions the endpoint provides so numbl can negotiate
 *  capabilities and fall back per-op for anything absent. */
export interface AcceleratorManifest {
  /** Wasm ABI major version. Must equal `ACCELERATOR_ABI` to load. */
  abi: number;
  /** Human-readable endpoint name (for diagnostics). */
  name?: string;
  /** Whether the module uses wasm threads (SharedArrayBuffer). v1: false. */
  threads?: boolean;
  /** Advertised operations. Only `matmul` is understood in v1. */
  ops: {
    matmul?: { precision: string[]; complex?: boolean };
  };
  /** Wasm binary filename, resolved relative to the manifest URL. */
  wasm: string;
}

/** The subset of a wasm module's exports the v1 ABI requires. A conforming
 *  module exports its linear `memory`, a bump/allocator pair, and the
 *  column-major real gemm. All pointers are byte offsets into `memory`. */
export interface WasmLapackExports {
  memory: WebAssembly.Memory;
  numbl_malloc(nbytes: number): number;
  numbl_free(ptr: number): void;
  /** C(m×n) = A(m×k) · B(k×n), all column-major f64, in `memory`. */
  numbl_matmul_f64_colmajor(
    aPtr: number,
    bPtr: number,
    cPtr: number,
    m: number,
    k: number,
    n: number
  ): void;
}

/** Signature shared by `LapackBridge.matmul` and the `$matmulAccel` hook. */
export type MatmulFn = (
  a: Float64Array,
  m: number,
  k: number,
  b: Float64Array,
  n: number
) => Float64Array;

/**
 * Validate a fetched manifest. Returns the typed manifest or throws with a
 * specific reason (so the loader can log and fall back to ts-lapack).
 */
export function validateManifest(raw: unknown): AcceleratorManifest {
  if (typeof raw !== "object" || raw === null) {
    throw new Error("accelerator manifest is not an object");
  }
  const m = raw as Record<string, unknown>;
  if (m.abi !== ACCELERATOR_ABI) {
    throw new Error(
      `accelerator manifest ABI ${String(m.abi)} != expected ${ACCELERATOR_ABI}`
    );
  }
  if (typeof m.wasm !== "string" || m.wasm.length === 0) {
    throw new Error("accelerator manifest missing 'wasm' filename");
  }
  if (typeof m.ops !== "object" || m.ops === null) {
    throw new Error("accelerator manifest missing 'ops'");
  }
  return m as unknown as AcceleratorManifest;
}

/**
 * Build a real-f64 matmul closure over an instantiated wasm module. Copies
 * the (column-major) inputs into the wasm heap, runs the kernel, and copies
 * the result out into a fresh `Float64Array`.
 *
 * Memory-growth safe: heap views are constructed *after* every `malloc`
 * (which may grow and detach `memory.buffer`) and the result view is taken
 * *after* the kernel call.
 */
export function makeWasmMatmul(exports: WasmLapackExports): MatmulFn {
  return (a, m, k, b, n) => {
    const aLen = m * k;
    const bLen = k * n;
    const cLen = m * n;
    const aPtr = exports.numbl_malloc(aLen * 8);
    const bPtr = exports.numbl_malloc(bLen * 8);
    const cPtr = exports.numbl_malloc(cLen * 8);
    if (aPtr === 0 || bPtr === 0 || cPtr === 0) {
      exports.numbl_free(cPtr);
      exports.numbl_free(bPtr);
      exports.numbl_free(aPtr);
      throw new Error("wasm matmul: allocation failed");
    }
    try {
      // Views taken after all mallocs (buffer may have grown/detached).
      const buf = exports.memory.buffer;
      new Float64Array(buf, aPtr, aLen).set(a.subarray(0, aLen));
      new Float64Array(buf, bPtr, bLen).set(b.subarray(0, bLen));
      exports.numbl_matmul_f64_colmajor(aPtr, bPtr, cPtr, m, k, n);
      // Re-view: the kernel may have grown memory too.
      const out = new Float64Array(cLen);
      out.set(new Float64Array(exports.memory.buffer, cPtr, cLen));
      return out;
    } finally {
      exports.numbl_free(cPtr);
      exports.numbl_free(bPtr);
      exports.numbl_free(aPtr);
    }
  };
}

/**
 * Assemble a (partial) `LapackBridge` from an instantiated module, exposing
 * only the ops the manifest advertises. Ops the bridge omits resolve to
 * ts-lapack via `getEffectiveBridge`.
 */
export function buildLapackBridgeFromWasm(
  exports: WasmLapackExports,
  manifest: AcceleratorManifest
): LapackBridge {
  const bridge: LapackBridge = {};
  if (
    manifest.ops.matmul?.precision?.includes("f64") &&
    typeof exports.numbl_matmul_f64_colmajor === "function"
  ) {
    bridge.matmul = makeWasmMatmul(exports);
  }
  return bridge;
}

/**
 * Install/remove the JS-JIT matmul accelerator hook (`globalThis.$matmulAccel`)
 * that `mtoc2_tensor_mtimes_real` consults. Pass `null` to uninstall.
 */
export function setMatmulAccelerator(
  fn:
    | ((
        a: Float64Array,
        m: number,
        k: number,
        b: Float64Array,
        n: number
      ) => Float64Array | null | undefined)
    | null
): void {
  if (fn) {
    (globalThis as { $matmulAccel?: unknown }).$matmulAccel = fn;
  } else {
    delete (globalThis as { $matmulAccel?: unknown }).$matmulAccel;
  }
}

/** Current matmul accelerator hook, or undefined if none is installed. */
export function getMatmulAccelerator():
  | ((
      a: Float64Array,
      m: number,
      k: number,
      b: Float64Array,
      n: number
    ) => Float64Array | null | undefined)
  | undefined {
  return (
    globalThis as {
      $matmulAccel?: (
        a: Float64Array,
        m: number,
        k: number,
        b: Float64Array,
        n: number
      ) => Float64Array | null | undefined;
    }
  ).$matmulAccel;
}

/**
 * Install both seams from one instantiated module:
 *  - registers a partial `LapackBridge` (interpreter / opt 0), and
 *  - installs the `$matmulAccel` hook (JS-JIT / opt 1) when the module
 *    provides matmul.
 *
 * The `$matmulAccel` wrapper is best-effort: it declines (returns `null`)
 * for small matmuls and on any error, so the JS loop remains the fallback.
 * The interpreter-path `bridge.matmul` is NOT wrapped — it propagates
 * errors like any other bridge (matching native-addon semantics).
 */
/** Remove both seams (e.g. when the user disables the accelerator). Leaves
 *  the interpreter and JS-JIT on ts-lapack / the JS loop. */
export function uninstallWasmLapackBridge(): void {
  setMatmulAccelerator(null);
  setLapackBridge(null as unknown as LapackBridge);
}

export function installWasmLapackBridge(
  exports: WasmLapackExports,
  manifest: AcceleratorManifest
): { matmul: boolean } {
  const bridge = buildLapackBridgeFromWasm(exports, manifest);
  setLapackBridge(bridge);
  let matmulInstalled = false;
  if (bridge.matmul) {
    const matmul = bridge.matmul;
    setMatmulAccelerator((a, m, k, b, n) => {
      if (m * n * k < WASM_MATMUL_MIN_WORK) return null; // too small to pay the copy tax
      try {
        return matmul(a, m, k, b, n);
      } catch {
        return null; // fall back to the caller's JS loop
      }
    });
    matmulInstalled = true;
  }
  return { matmul: matmulInstalled };
}
