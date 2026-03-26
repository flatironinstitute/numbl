import { getLapackBridge, setLapackBridge, type LapackBridge } from "./lapack-bridge.js";
import { getTsLapackBridge } from "./ts-lapack-bridge.js";

interface BrowserWasmManifestTarget {
  name: string;
  wasmPath: string;
  exports?: string[];
}

interface BrowserWasmManifest {
  generatedAt?: string;
  targets?: BrowserWasmManifestTarget[];
}

const MANIFEST_URL = "/wasm-kernels/manifest.json";
let _loadPromise: Promise<boolean> | null = null;

function getRuntimeLocationHref(): string | null {
  const candidate = (globalThis as { location?: { href?: string } }).location;
  return typeof candidate?.href === "string" ? candidate.href : null;
}

function createImportObject(): WebAssembly.Imports {
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

class WasmKernel {
  readonly memory: WebAssembly.Memory;
  readonly bridgeName: string;
  private readonly exports: Record<string, unknown>;
  private readonly mallocFn: (bytes: number) => number;
  private readonly freeFn: (ptr: number) => void;

  constructor(target: BrowserWasmManifestTarget, instance: WebAssembly.Instance) {
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
    return new Float64Array(
      new Float64Array(this.memory.buffer, ptr, length)
    );
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

async function instantiateKernel(
  target: BrowserWasmManifestTarget
): Promise<WasmKernel> {
  const baseHref = getRuntimeLocationHref() ?? "http://localhost/";
  const url = new URL(target.wasmPath, baseHref).toString();
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${target.name}: failed to fetch ${url} (${response.status})`);
  }
  const bytes = await response.arrayBuffer();
  const { instance } = await WebAssembly.instantiate(bytes, createImportObject());
  return new WasmKernel(target, instance);
}

const FFT_KERNEL_ORDER = ["ducc0-fft", "fftw-fft"] as const;
const LINALG_KERNEL_ORDER = [
  "blas-lapack",
  "openblas-lapack",
  "flame-blas-lapack",
] as const;
const GENERAL_LINSOLVE_KERNEL_ORDER = [
  "flame-blas-lapack",
  "blas-lapack",
  "openblas-lapack",
] as const;

function pickKernel(
  kernels: Record<string, WasmKernel>,
  order: readonly string[],
  exportsToCheck: string[],
  predicate: (kernel: WasmKernel) => boolean = () => true
): WasmKernel | null {
  for (const name of order) {
    const kernel = kernels[name];
    if (!kernel) continue;
    if (!predicate(kernel)) continue;
    if (!kernel.hasExport(exportsToCheck)) continue;
    return kernel;
  }
  return null;
}

function supportsRectangularLinsolve(kernel: WasmKernel): boolean {
  return kernel.bridgeName === "flame-blas-lapack";
}

function createBrowserBridge(kernels: Record<string, WasmKernel>): LapackBridge {
  const fftKernel = pickKernel(kernels, FFT_KERNEL_ORDER, [
    "numbl_fft1d_f64",
    "_numbl_fft1d_f64",
  ]);
  const matmulKernel = pickKernel(kernels, LINALG_KERNEL_ORDER, [
    "numbl_matmul_f64",
    "_numbl_matmul_f64",
  ]);
  const invKernel = pickKernel(kernels, LINALG_KERNEL_ORDER, [
    "numbl_inv_f64",
    "_numbl_inv_f64",
  ]);
  const squareLinsolveKernel = pickKernel(kernels, LINALG_KERNEL_ORDER, [
    "numbl_linsolve_f64",
    "_numbl_linsolve_f64",
  ]);
  const generalLinsolveKernel = pickKernel(
    kernels,
    GENERAL_LINSOLVE_KERNEL_ORDER,
    ["numbl_linsolve_f64", "_numbl_linsolve_f64"],
    supportsRectangularLinsolve
  );
  const tsBridge = getTsLapackBridge();

  const bridge: LapackBridge = {
    ...tsBridge,
    bridgeName: "browser Wasm kernels",
  };

  if (fftKernel?.hasExport(["numbl_fft1d_f64", "_numbl_fft1d_f64"])) {
    bridge.fft1dComplex = (re, im, n, inverse) =>
      fftKernel.fft1dComplex(re, im, n, inverse);
  }
  if (fftKernel?.hasExport(["numbl_fft_along_dim_f64", "_numbl_fft_along_dim_f64"])) {
    bridge.fftAlongDim = (re, im, shape, dim, n, inverse) =>
      fftKernel.fftAlongDim(re, im, shape, dim, n, inverse);
  }
  if (matmulKernel !== null) {
    bridge.matmul = (A, m, k, B, n) => matmulKernel.matmul(A, m, k, B, n);
  }
  if (invKernel !== null) {
    bridge.inv = (data, n) => invKernel.inv(data, n);
  }
  if (squareLinsolveKernel !== null || generalLinsolveKernel !== null) {
    bridge.linsolve = (A, m, n, B, nrhs) => {
      const kernel =
        m === n
          ? squareLinsolveKernel ?? generalLinsolveKernel
          : generalLinsolveKernel;
      if (kernel !== null) {
        return kernel.linsolve(A, m, n, B, nrhs);
      }
      if (m !== n) {
        return tsBridge.linsolve!(A, m, n, B, nrhs);
      }
      return tsBridge.linsolve!(A, m, n, B, nrhs);
    };
  }

  return bridge;
}

async function loadManifest(): Promise<BrowserWasmManifest | null> {
  const baseHref = getRuntimeLocationHref();
  if (baseHref === null) return null;
  const url = new URL(MANIFEST_URL, baseHref).toString();
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    return null;
  }
  return (await response.json()) as BrowserWasmManifest;
}

async function configureBrowserWasmBridge(): Promise<boolean> {
  if (
    typeof fetch !== "function" ||
    typeof WebAssembly === "undefined" ||
    getRuntimeLocationHref() === null
  ) {
    return false;
  }
  if (getLapackBridge()) return true;

  try {
    const manifest = await loadManifest();
    const targets = manifest?.targets ?? [];
    const relevantTargets = targets.filter(
      target =>
        target.name === "ducc0-fft" ||
        target.name === "fftw-fft" ||
        target.name === "flame-blas-lapack" ||
        target.name === "openblas-lapack" ||
        target.name === "blas-lapack"
    );
    if (relevantTargets.length === 0) return false;

    const kernels = Object.fromEntries(
      await Promise.all(
        relevantTargets.map(async target => [target.name, await instantiateKernel(target)])
      )
    ) as Record<string, WasmKernel>;

    setLapackBridge(createBrowserBridge(kernels));
    return true;
  } catch (error) {
    console.warn(
      `Browser Wasm bridge unavailable: ${error instanceof Error ? error.message : String(error)}`
    );
    return false;
  }
}

export async function ensureBrowserWasmBridgeConfigured(): Promise<boolean> {
  if (_loadPromise === null) {
    _loadPromise = configureBrowserWasmBridge();
  }
  return _loadPromise;
}

export function resetBrowserWasmBridgeForTests(): void {
  _loadPromise = null;
  setLapackBridge(null);
}
