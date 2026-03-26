import { getLapackBridge, setLapackBridge, type LapackBridge } from "./lapack-bridge.js";
import { getTsLapackBridge } from "./ts-lapack-bridge.js";
import {
  BROWSER_WASM_MANIFEST_PATH,
  BrowserWasmKernel,
  createBrowserWasmImportObject,
  getRuntimeLocationHref,
  isRelevantRuntimeBrowserWasmTarget,
  resolveRuntimeAssetUrl,
  supportsRectangularLinsolveTarget,
  type BrowserWasmManifest,
  type BrowserWasmManifestTarget,
} from "./browser-wasm-kernel.js";

let _loadPromise: Promise<boolean> | null = null;

async function instantiateKernel(
  target: BrowserWasmManifestTarget
): Promise<BrowserWasmKernel> {
  const url = resolveRuntimeAssetUrl(target.wasmPath);
  if (url === null) {
    throw new Error(`${target.name}: runtime location is unavailable`);
  }
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`${target.name}: failed to fetch ${url} (${response.status})`);
  }
  const bytes = await response.arrayBuffer();
  const { instance } = await WebAssembly.instantiate(
    bytes,
    createBrowserWasmImportObject()
  );
  return new BrowserWasmKernel(target, instance);
}

const FFT_KERNEL_ORDER = ["ducc0-fft"] as const;
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
  kernels: Record<string, BrowserWasmKernel>,
  order: readonly string[],
  exportsToCheck: string[],
  predicate: (kernel: BrowserWasmKernel) => boolean = () => true
): BrowserWasmKernel | null {
  for (const name of order) {
    const kernel = kernels[name];
    if (!kernel) continue;
    if (!predicate(kernel)) continue;
    if (!kernel.hasExport(exportsToCheck)) continue;
    return kernel;
  }
  return null;
}

function kernelSupportsRectangularLinsolve(
  kernel: BrowserWasmKernel
): boolean {
  return supportsRectangularLinsolveTarget(kernel.target);
}

function createBrowserBridge(
  kernels: Record<string, BrowserWasmKernel>
): LapackBridge {
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
    kernelSupportsRectangularLinsolve
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
  const url = resolveRuntimeAssetUrl(BROWSER_WASM_MANIFEST_PATH);
  if (url === null) return null;
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
    const relevantTargets = targets.filter(isRelevantRuntimeBrowserWasmTarget);
    if (relevantTargets.length === 0) return false;

    const settled = await Promise.allSettled(
      relevantTargets.map(async target => [target.name, await instantiateKernel(target)] as const)
    );
    const kernels: Record<string, BrowserWasmKernel> = {};
    for (const result of settled) {
      if (result.status === "fulfilled") {
        kernels[result.value[0]] = result.value[1];
        continue;
      }
      console.warn(
        `Browser Wasm kernel unavailable: ${result.reason instanceof Error ? result.reason.message : String(result.reason)}`
      );
    }
    if (Object.keys(kernels).length === 0) {
      return false;
    }

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
