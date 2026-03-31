import type { MatmulBackend, MatmulFn } from "./matmul-bench-core.js";
import { getTsLapackBridge } from "../numbl-core/native/ts-lapack-bridge.js";
import { getFlameBridge } from "../flame-ts/bridge.js";

const WASM_KERNELS = [
  {
    id: "wasm-xsimd",
    label: "WASM BLIS (xsimd SIMD)",
    file: "blis-dgemm.wasm",
    fn: "numbl_dgemm_f64",
  },
  {
    id: "wasm-eigen",
    label: "WASM Eigen",
    file: "eigen-dgemm.wasm",
    fn: "numbl_eigen_dgemm_f64",
  },
  {
    id: "wasm-openblas",
    label: "WASM OpenBLAS",
    file: "openblas-dgemm.wasm",
    fn: "numbl_openblas_dgemm_f64",
  },
];

function wrapWasmDgemm(
  wasm: {
    memory: WebAssembly.Memory;
    malloc: (n: number) => number;
    free: (p: number) => void;
  },
  fn: (A: number, m: number, k: number, B: number, n: number, C: number) => void
): MatmulFn {
  return (A, m, k, B, n) => {
    const pA = wasm.malloc(m * k * 8);
    const pB = wasm.malloc(k * n * 8);
    const pC = wasm.malloc(m * n * 8);
    new Float64Array(wasm.memory.buffer, pA, m * k).set(A);
    new Float64Array(wasm.memory.buffer, pB, k * n).set(B);
    fn(pA, m, k, pB, n, pC);
    const C = new Float64Array(m * n);
    C.set(new Float64Array(wasm.memory.buffer, pC, m * n));
    wasm.free(pA);
    wasm.free(pB);
    wasm.free(pC);
    return C;
  };
}

async function loadWasmBytes(file: string): Promise<ArrayBuffer | null> {
  const isNode =
    typeof process !== "undefined" &&
    typeof process.versions?.node === "string";
  if (isNode) {
    try {
      const { readFileSync } = await import("fs");
      const { dirname, join } = await import("path");
      const { fileURLToPath } = await import("url");
      const base = join(
        dirname(fileURLToPath(import.meta.url)),
        "..",
        "..",
        "public",
        "wasm-kernels"
      );
      return readFileSync(join(base, file)).buffer;
    } catch {
      return null;
    }
  } else {
    try {
      const resp = await fetch(`/wasm-kernels/${file}`);
      if (!resp.ok) return null;
      return resp.arrayBuffer();
    } catch {
      return null;
    }
  }
}

async function tryLoadWasm(
  kernel: (typeof WASM_KERNELS)[0]
): Promise<MatmulBackend | null> {
  const bytes = await loadWasmBytes(kernel.file);
  if (!bytes) return null;
  try {
    const ref: { memory?: WebAssembly.Memory } = {};
    const zeroOut = (_ptr: number, ...offsets: number[]) => {
      const mem = new Uint32Array(ref.memory!.buffer);
      for (const o of offsets) mem[o >> 2] = 0;
      return 0;
    };
    const { instance } = await WebAssembly.instantiate(bytes, {
      env: { emscripten_notify_memory_growth: () => {} },
      wasi_snapshot_preview1: {
        fd_close: () => 0,
        fd_seek: () => 0,
        fd_write: () => 0,
        fd_read: () => 0,
        fd_fdstat_get: () => 0,
        fd_prestat_get: () => -1,
        fd_prestat_dir_name: () => -1,
        environ_sizes_get: (a: number, b: number) => zeroOut(0, a, b),
        environ_get: () => 0,
        args_sizes_get: (a: number, b: number) => zeroOut(0, a, b),
        args_get: () => 0,
        proc_exit: () => {},
        clock_time_get: () => 0,
      },
    });
    ref.memory = (instance.exports as Record<string, unknown>)
      .memory as WebAssembly.Memory;
    const exports = instance.exports as Record<string, unknown>;
    const memory = exports.memory as WebAssembly.Memory;
    const malloc = exports.malloc as (n: number) => number;
    const free = exports.free as (p: number) => void;
    const fn = exports[kernel.fn] as (...args: number[]) => void;
    if (!fn) return null;
    return {
      id: kernel.id,
      label: kernel.label,
      matmul: wrapWasmDgemm({ memory, malloc, free }, fn),
    };
  } catch {
    return null;
  }
}

async function tryLoadNative(): Promise<MatmulBackend | null> {
  if (typeof process === "undefined") return null;
  try {
    const { createRequire } = await import("module");
    const { dirname, join } = await import("path");
    const { fileURLToPath } = await import("url");
    const { NATIVE_ADDON_EXPECTED_VERSION } =
      await import("../numbl-core/native/lapack-bridge.js");
    const req = createRequire(import.meta.url);
    const addonPath = join(
      dirname(fileURLToPath(import.meta.url)),
      "..",
      "..",
      "build",
      "Release",
      "numbl_addon.node"
    );
    const addon = req(addonPath);
    if (
      typeof addon.addonVersion === "function" &&
      addon.addonVersion() === NATIVE_ADDON_EXPECTED_VERSION
    ) {
      return {
        id: "native-openblas",
        label: "Native OpenBLAS (node-gyp addon)",
        matmul: addon.matmul,
      };
    }
  } catch {
    /* not available */
  }
  return null;
}

export async function discoverBackends(): Promise<MatmulBackend[]> {
  const tsLapack = getTsLapackBridge();
  const flame = getFlameBridge();

  const backends: MatmulBackend[] = [
    {
      id: "ts-lapack",
      label: "TypeScript (unblocked)",
      matmul: tsLapack.matmul!,
    },
    { id: "flame-ts", label: "FLAME TS (blocked BLIS)", matmul: flame.matmul! },
  ];

  const wasmResults = await Promise.all(WASM_KERNELS.map(tryLoadWasm));
  for (const b of wasmResults) if (b) backends.push(b);

  const native = await tryLoadNative();
  if (native) backends.push(native);

  return backends;
}
