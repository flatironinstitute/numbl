/**
 * Browser/worker loader for the optional WASM linear-algebra accelerator.
 *
 * Mirrors `qhull-browser.ts`: fetch the endpoint's manifest, fetch and
 * instantiate its wasm, and install it via `installWasmLapackBridge`. numbl
 * instantiates the wasm itself (no remote JS is executed) with a
 * numbl-controlled import object, so the accelerator is sandboxed.
 *
 * Best-effort: any failure resolves to `false` (nothing installed → the
 * interpreter/JIT keep using ts-lapack / the JS loop). Idempotent per URL —
 * a repeat call for the same URL returns the in-flight/settled promise.
 *
 * `fetchImpl` / `instantiate` / `cache` are injectable so the whole
 * orchestration is unit-testable without a wasm toolchain.
 */

import {
  installWasmLapackBridge,
  uninstallWasmLapackBridge,
  validateManifest,
  type WasmLapackExports,
} from "./wasm-lapack-bridge.js";

/** Persistent byte cache (e.g. IndexedDB) for wasm binaries, keyed by URL.
 *  Optional; failures must be non-fatal. */
export interface WasmByteCache {
  get(key: string): Promise<Uint8Array | null>;
  put(key: string, bytes: Uint8Array): Promise<void>;
}

export interface LoadWasmLapackDeps {
  fetchImpl?: typeof fetch;
  /** Instantiate wasm bytes; defaults to `WebAssembly.instantiate`. */
  instantiate?: (
    bytes: BufferSource,
    imports: WebAssembly.Imports
  ) => Promise<WebAssembly.Instance>;
  cache?: WasmByteCache;
}

let currentUrl: string | null = null;
let inflight: Promise<boolean> | null = null;

/**
 * Ensure the accelerator for `url` is loaded and installed. Pass `null` (or
 * empty) to uninstall. Returns a promise for whether matmul got installed.
 */
export function ensureWasmLapackBridge(
  url: string | null,
  deps: LoadWasmLapackDeps = {}
): Promise<boolean> {
  if (!url) {
    uninstallWasmLapackBridge();
    currentUrl = null;
    inflight = null;
    return Promise.resolve(false);
  }
  if (url === currentUrl && inflight) return inflight;
  currentUrl = url;
  inflight = load(url, deps).catch(e => {
    console.warn(
      `WASM LAPACK bridge not loaded from ${url} (using ts-lapack / JS fallback):`,
      e instanceof Error ? e.message : String(e)
    );
    return false;
  });
  return inflight;
}

function ensureTrailingSlash(u: string): string {
  return u.endsWith("/") ? u : u + "/";
}

/** A defensive import object. The Phase-3 bridge is built self-contained
 *  (exports its own `memory`, imports nothing), so `{}` suffices; the extra
 *  stubs let an emscripten `-sSTANDALONE_WASM` build (which may import a few
 *  wasi/env symbols) also instantiate. Unused imports are ignored. */
function makeImportObject(): WebAssembly.Imports {
  const noop = (): void => {};
  const abort = (): never => {
    throw new Error("wasm bridge aborted");
  };
  return {
    env: {
      memory: new WebAssembly.Memory({ initial: 256, maximum: 32768 }),
      abort,
      emscripten_notify_memory_growth: noop,
    },
    wasi_snapshot_preview1: {
      proc_exit: abort,
      fd_write: () => 0,
      fd_close: () => 0,
      fd_seek: () => 0,
      environ_get: () => 0,
      environ_sizes_get: () => 0,
    },
  } as unknown as WebAssembly.Imports;
}

async function fetchWasmBytes(
  url: string,
  fetchImpl: typeof fetch,
  cache?: WasmByteCache
): Promise<Uint8Array> {
  if (cache) {
    const hit = await cache.get(url).catch(() => null);
    if (hit) return hit;
  }
  const resp = await fetchImpl(url);
  if (!resp.ok) throw new Error(`wasm HTTP ${resp.status}`);
  const bytes = new Uint8Array(await resp.arrayBuffer());
  if (cache) void cache.put(url, bytes).catch(() => {});
  return bytes;
}

async function load(
  baseUrl: string,
  deps: LoadWasmLapackDeps
): Promise<boolean> {
  const fetchImpl = deps.fetchImpl ?? fetch;
  const instantiate =
    deps.instantiate ??
    ((b, i) => WebAssembly.instantiate(b, i).then(r => r.instance));

  // Manifest is small and content changes decide which wasm to load, so
  // always fetch it fresh (no-cache); the wasm itself is byte-cached and
  // should be content-hashed in its filename for cheap invalidation.
  const manifestUrl = new URL("numbl-bridge.json", ensureTrailingSlash(baseUrl))
    .href;
  const manifestResp = await fetchImpl(manifestUrl, { cache: "no-cache" });
  if (!manifestResp.ok) {
    throw new Error(`manifest HTTP ${manifestResp.status}`);
  }
  const manifest = validateManifest(await manifestResp.json());

  const wasmUrl = new URL(manifest.wasm, manifestUrl).href;
  const bytes = await fetchWasmBytes(wasmUrl, fetchImpl, deps.cache);

  const instance = await instantiate(
    bytes as unknown as BufferSource,
    makeImportObject()
  );
  const exports = instance.exports as unknown as WasmLapackExports;
  const res = installWasmLapackBridge(exports, manifest);
  return res.matmul;
}

/** Test-only: reset the module-level idempotency cache. */
export function _resetWasmLapackLoaderForTests(): void {
  currentUrl = null;
  inflight = null;
}
