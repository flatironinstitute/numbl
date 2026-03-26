import type { LapackBridge } from "./lapack-bridge.js";

export interface BrowserWasmTargetCapabilities {
  fft1dComplex?: boolean;
  fftAlongDim?: boolean;
  matmul?: boolean;
  inv?: boolean;
  linsolveSquare?: boolean;
  linsolveRectangular?: boolean;
}

export interface BrowserWasmManifestTarget {
  name: string;
  wasmPath: string;
  exports?: string[];
  enabledByDefault?: boolean;
  capabilities?: BrowserWasmTargetCapabilities;
}

export interface BrowserWasmManifest {
  generatedAt?: string;
  targets?: BrowserWasmManifestTarget[];
}

export type BrowserWasmKernelBridge = Pick<
  LapackBridge,
  "matmul" | "inv" | "linsolve" | "fft1dComplex" | "fftAlongDim"
>;

export const BROWSER_WASM_MANIFEST_PATH = "wasm-kernels/manifest.json";

function getImportMetaBaseUrl(): string | null {
  const env = (import.meta as ImportMeta & {
    env?: {
      BASE_URL?: string;
    };
  }).env;
  return typeof env?.BASE_URL === "string" && env.BASE_URL.length > 0
    ? env.BASE_URL
    : null;
}

export function getRuntimeLocationHref(): string | null {
  const candidate = (globalThis as { location?: { href?: string } }).location;
  return typeof candidate?.href === "string" ? candidate.href : null;
}

export function resolveRuntimeBaseHref(
  locationHref: string | null = getRuntimeLocationHref()
): string | null {
  if (locationHref === null) return null;
  const baseUrl = getImportMetaBaseUrl();
  if (baseUrl !== null) {
    return new URL(baseUrl, locationHref).toString();
  }
  return new URL("/", locationHref).toString();
}

export function resolveRuntimeAssetUrl(
  assetPath: string,
  locationHref: string | null = getRuntimeLocationHref()
): string | null {
  const baseHref = resolveRuntimeBaseHref(locationHref);
  if (baseHref === null) return null;
  return new URL(assetPath.replace(/^\/+/, ""), baseHref).toString();
}

export function createBrowserWasmImportObject(): WebAssembly.Imports {
  return {
    wasi_snapshot_preview1: {
      fd_write: () => 0,
      fd_read: () => 0,
      fd_close: () => 0,
      fd_seek: () => 0,
      fd_fdstat_get: () => 0,
      proc_exit: () => {},
      environ_sizes_get: () => 0,
      environ_get: () => 0,
      clock_time_get: () => 0,
      args_sizes_get: () => 0,
      args_get: () => 0,
    },
    env: {
      emscripten_notify_memory_growth: () => {},
    },
  };
}

function byteLengthFloat64(length: number): number {
  return length * Float64Array.BYTES_PER_ELEMENT;
}

function byteLengthInt32(length: number): number {
  return length * Int32Array.BYTES_PER_ELEMENT;
}

function outSizeAlongDim(shape: number[], dim: number, n: number): number {
  let strideDim = 1;
  for (let i = 0; i < dim; i++) strideDim *= shape[i];
  let numAbove = 1;
  for (let i = dim + 1; i < shape.length; i++) numAbove *= shape[i];
  return strideDim * n * numAbove;
}

function inferCapabilitiesFromExports(
  exportsList: readonly string[] | undefined
): BrowserWasmTargetCapabilities {
  const names = new Set(exportsList ?? []);
  return {
    fft1dComplex: names.has("numbl_fft1d_f64") || names.has("_numbl_fft1d_f64"),
    fftAlongDim:
      names.has("numbl_fft_along_dim_f64") ||
      names.has("_numbl_fft_along_dim_f64"),
    matmul: names.has("numbl_matmul_f64") || names.has("_numbl_matmul_f64"),
    inv: names.has("numbl_inv_f64") || names.has("_numbl_inv_f64"),
    linsolveSquare:
      names.has("numbl_linsolve_f64") || names.has("_numbl_linsolve_f64"),
    linsolveRectangular: false,
  };
}

export function getBrowserWasmTargetCapabilities(
  target: BrowserWasmManifestTarget
): BrowserWasmTargetCapabilities {
  return {
    ...inferCapabilitiesFromExports(target.exports),
    ...(target.capabilities ?? {}),
  };
}

export function isRelevantRuntimeBrowserWasmTarget(
  target: BrowserWasmManifestTarget
): boolean {
  const capabilities = getBrowserWasmTargetCapabilities(target);
  return Object.values(capabilities).some(Boolean);
}

export function supportsRectangularLinsolveTarget(
  target: BrowserWasmManifestTarget
): boolean {
  return getBrowserWasmTargetCapabilities(target).linsolveRectangular === true;
}

export class BrowserWasmKernel {
  readonly target: BrowserWasmManifestTarget;
  readonly memory: WebAssembly.Memory;
  readonly bridgeName: string;
  private readonly exports: Record<string, unknown>;
  private readonly mallocFn: (bytes: number) => number;
  private readonly freeFn: (ptr: number) => void;

  constructor(target: BrowserWasmManifestTarget, instance: WebAssembly.Instance) {
    this.target = target;
    this.bridgeName = target.name;
    this.exports = instance.exports as Record<string, unknown>;
    const initialize = this.exports["_initialize"];
    if (typeof initialize === "function") {
      (initialize as () => void)();
    }
    this.memory = this.requireExport<WebAssembly.Memory>(["memory"]);
    this.mallocFn = this.requireExport<(bytes: number) => number>([
      "malloc",
      "_malloc",
    ]);
    this.freeFn = this.requireExport<(ptr: number) => void>(["free", "_free"]);
  }

  hasExport(names: string[]): boolean {
    return names.some(name => typeof this.exports[name] === "function");
  }

  private requireExport<T>(names: string[]): T {
    for (const name of names) {
      const value = this.exports[name];
      if (value !== undefined) return value as T;
    }
    throw new Error(
      `${this.bridgeName}: missing export (${names.join(" or ")})`
    );
  }

  private allocBytes(bytes: number): number {
    return this.mallocFn(bytes);
  }

  private freeBytes(ptr: number): void {
    if (ptr !== 0) {
      this.freeFn(ptr);
    }
  }

  private writeF64(ptr: number, data: Float64Array): void {
    new Float64Array(this.memory.buffer, ptr, data.length).set(data);
  }

  private writeI32(ptr: number, data: Int32Array): void {
    new Int32Array(this.memory.buffer, ptr, data.length).set(data);
  }

  private readF64(ptr: number, length: number): Float64Array {
    return new Float64Array(new Float64Array(this.memory.buffer, ptr, length));
  }

  private withInputF64<T>(data: Float64Array, fn: (ptr: number) => T): T {
    const ptr = this.allocBytes(byteLengthFloat64(data.length));
    try {
      this.writeF64(ptr, data);
      return fn(ptr);
    } finally {
      this.freeBytes(ptr);
    }
  }

  private withOptionalInputF64<T>(
    data: Float64Array | null,
    fn: (ptr: number) => T
  ): T {
    if (data === null) return fn(0);
    return this.withInputF64(data, fn);
  }

  private withInputI32<T>(data: Int32Array, fn: (ptr: number) => T): T {
    const ptr = this.allocBytes(byteLengthInt32(data.length));
    try {
      this.writeI32(ptr, data);
      return fn(ptr);
    } finally {
      this.freeBytes(ptr);
    }
  }

  private callStatus(fn: (...args: number[]) => number, args: number[]): void {
    const status = fn(...args);
    if (status !== 0) {
      throw new Error(`${this.bridgeName}: wasm kernel returned status ${status}`);
    }
  }

  fft1dComplex(
    re: Float64Array,
    im: Float64Array,
    n: number,
    inverse: boolean
  ): { re: Float64Array; im: Float64Array } {
    const fn = this.requireExport<(...args: number[]) => number>([
      "numbl_fft1d_f64",
      "_numbl_fft1d_f64",
    ]);
    const outRePtr = this.allocBytes(byteLengthFloat64(n));
    const outImPtr = this.allocBytes(byteLengthFloat64(n));
    try {
      this.withInputF64(re, rePtr =>
        this.withInputF64(im, imPtr => {
          this.callStatus(fn, [
            rePtr,
            imPtr,
            n,
            inverse ? 1 : 0,
            outRePtr,
            outImPtr,
          ]);
        })
      );
      return {
        re: this.readF64(outRePtr, n),
        im: this.readF64(outImPtr, n),
      };
    } finally {
      this.freeBytes(outRePtr);
      this.freeBytes(outImPtr);
    }
  }

  fftAlongDim(
    re: Float64Array,
    im: Float64Array | null,
    shape: number[],
    dim: number,
    n: number,
    inverse: boolean
  ): { re: Float64Array; im: Float64Array } {
    const fn = this.requireExport<(...args: number[]) => number>([
      "numbl_fft_along_dim_f64",
      "_numbl_fft_along_dim_f64",
    ]);
    const shapeI32 = Int32Array.from(shape);
    const outLen = outSizeAlongDim(shape, dim, n);
    const outRePtr = this.allocBytes(byteLengthFloat64(outLen));
    const outImPtr = this.allocBytes(byteLengthFloat64(outLen));
    try {
      this.withInputF64(re, rePtr =>
        this.withOptionalInputF64(im, imPtr =>
          this.withInputI32(shapeI32, shapePtr => {
            this.callStatus(fn, [
              rePtr,
              imPtr,
              shapePtr,
              shape.length,
              dim,
              n,
              inverse ? 1 : 0,
              outRePtr,
              outImPtr,
            ]);
          })
        )
      );
      return {
        re: this.readF64(outRePtr, outLen),
        im: this.readF64(outImPtr, outLen),
      };
    } finally {
      this.freeBytes(outRePtr);
      this.freeBytes(outImPtr);
    }
  }

  matmul(
    A: Float64Array,
    m: number,
    k: number,
    B: Float64Array,
    n: number
  ): Float64Array {
    const fn = this.requireExport<(...args: number[]) => number>([
      "numbl_matmul_f64",
      "_numbl_matmul_f64",
    ]);
    const outPtr = this.allocBytes(byteLengthFloat64(m * n));
    try {
      this.withInputF64(A, aPtr =>
        this.withInputF64(B, bPtr => {
          this.callStatus(fn, [aPtr, m, k, bPtr, n, outPtr]);
        })
      );
      return this.readF64(outPtr, m * n);
    } finally {
      this.freeBytes(outPtr);
    }
  }

  inv(data: Float64Array, n: number): Float64Array {
    const fn = this.requireExport<(...args: number[]) => number>([
      "numbl_inv_f64",
      "_numbl_inv_f64",
    ]);
    const outPtr = this.allocBytes(byteLengthFloat64(n * n));
    try {
      this.withInputF64(data, dataPtr => {
        this.callStatus(fn, [dataPtr, n, outPtr]);
      });
      return this.readF64(outPtr, n * n);
    } finally {
      this.freeBytes(outPtr);
    }
  }

  linsolve(
    A: Float64Array,
    m: number,
    n: number,
    B: Float64Array,
    nrhs: number
  ): Float64Array {
    const fn = this.requireExport<(...args: number[]) => number>([
      "numbl_linsolve_f64",
      "_numbl_linsolve_f64",
    ]);
    const outPtr = this.allocBytes(byteLengthFloat64(n * nrhs));
    try {
      this.withInputF64(A, aPtr =>
        this.withInputF64(B, bPtr => {
          this.callStatus(fn, [aPtr, m, n, bPtr, nrhs, outPtr]);
        })
      );
      return this.readF64(outPtr, n * nrhs);
    } finally {
      this.freeBytes(outPtr);
    }
  }
}
