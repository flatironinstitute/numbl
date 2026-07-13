import { describe, it, expect, afterEach } from "vitest";
import { executeCode } from "../numbl-core/executeCode.js";
import { isRuntimeTensor } from "../numbl-core/runtime/types.js";
import {
  setLapackBridge,
  type LapackBridge,
} from "../numbl-core/native/lapack-bridge.js";
import {
  setMatmulAccelerator,
  getMatmulAccelerator,
  ACCELERATOR_ABI,
  type WasmLapackExports,
} from "../numbl-core/native/wasm-lapack-bridge.js";
import {
  ensureWasmLapackBridge,
  _resetWasmLapackLoaderForTests,
  type WasmByteCache,
} from "../numbl-core/native/wasm-lapack-browser.js";

function makeFakeWasm(): { exports: WasmLapackExports; calls: () => number } {
  const memory = new WebAssembly.Memory({ initial: 256 });
  let brk = 1024;
  let n = 0;
  return {
    calls: () => n,
    exports: {
      memory,
      numbl_malloc(nbytes: number) {
        const p = brk;
        brk += (nbytes + 7) & ~7;
        return p;
      },
      numbl_free() {},
      numbl_matmul_f64_colmajor(aPtr, bPtr, cPtr, m, k, nn) {
        n++;
        const A = new Float64Array(memory.buffer, aPtr, m * k);
        const B = new Float64Array(memory.buffer, bPtr, k * nn);
        const C = new Float64Array(memory.buffer, cPtr, m * nn);
        for (let j = 0; j < nn; j++)
          for (let i = 0; i < m; i++) {
            let s = 0;
            for (let p = 0; p < k; p++) s += A[i + p * m] * B[p + j * k];
            C[i + j * m] = s;
          }
      },
    },
  };
}

const MANIFEST = {
  abi: ACCELERATOR_ABI,
  name: "test-bridge",
  ops: { matmul: { precision: ["f64"] } },
  wasm: "bridge.abc123.wasm",
};

/** A fake `fetch` serving a manifest and (dummy) wasm bytes, recording URLs. */
function makeFakeFetch(opts: { manifestOk?: boolean; wasmOk?: boolean } = {}) {
  const urls: string[] = [];
  const fetchImpl = (async (input: RequestInfo | URL) => {
    const url = String(input);
    urls.push(url);
    if (url.endsWith("numbl-bridge.json")) {
      return {
        ok: opts.manifestOk !== false,
        status: opts.manifestOk === false ? 404 : 200,
        json: async () => MANIFEST,
      } as unknown as Response;
    }
    // wasm
    return {
      ok: opts.wasmOk !== false,
      status: opts.wasmOk === false ? 404 : 200,
      arrayBuffer: async () => new Uint8Array([0, 97, 115, 109]).buffer, // "\0asm"
    } as unknown as Response;
  }) as unknown as typeof fetch;
  return { fetchImpl, urls };
}

function tensorVar(
  res: ReturnType<typeof executeCode>,
  name: string
): number[] {
  const v = res.variableValues[name];
  if (!isRuntimeTensor(v)) throw new Error(`${name} not a tensor`);
  return Array.from(v.data);
}

afterEach(() => {
  setMatmulAccelerator(null);
  setLapackBridge(null as unknown as LapackBridge);
  _resetWasmLapackLoaderForTests();
});

describe("wasm-lapack-browser loader", () => {
  it("fetches manifest + wasm, instantiates, and installs the accelerator", async () => {
    const { exports } = makeFakeWasm();
    const { fetchImpl, urls } = makeFakeFetch();
    const ok = await ensureWasmLapackBridge("https://host.example/bridge/", {
      fetchImpl,
      instantiate: async () => ({ exports }) as unknown as WebAssembly.Instance,
    });
    expect(ok).toBe(true);
    expect(getMatmulAccelerator()).toBeTypeOf("function");
    // Manifest at base + filename; wasm URL resolved relative to manifest.
    expect(urls[0]).toBe("https://host.example/bridge/numbl-bridge.json");
    expect(urls[1]).toBe("https://host.example/bridge/bridge.abc123.wasm");
  });

  it("appends a trailing slash to the base URL", async () => {
    const { exports } = makeFakeWasm();
    const { fetchImpl, urls } = makeFakeFetch();
    await ensureWasmLapackBridge("https://host.example/bridge", {
      fetchImpl,
      instantiate: async () => ({ exports }) as unknown as WebAssembly.Instance,
    });
    expect(urls[0]).toBe("https://host.example/bridge/numbl-bridge.json");
  });

  it("a loaded bridge accelerates matmul through the opt-1 pipeline", async () => {
    const { exports, calls } = makeFakeWasm();
    const { fetchImpl } = makeFakeFetch();
    await ensureWasmLapackBridge("https://host.example/bridge/", {
      fetchImpl,
      instantiate: async () => ({ exports }) as unknown as WebAssembly.Instance,
    });
    const script = `A = reshape(1:3600,60,60); B = reshape(3600:-1:1,60,60); C = A*B;`;
    const c = tensorVar(executeCode(script, { optimization: "1" }), "C");
    const a = Array.from({ length: 3600 }, (_, i) => i + 1);
    const b = Array.from({ length: 3600 }, (_, i) => 3600 - i);
    // reference column-major matmul
    const ref = new Array(3600).fill(0);
    for (let j = 0; j < 60; j++)
      for (let i = 0; i < 60; i++) {
        let s = 0;
        for (let p = 0; p < 60; p++) s += a[i + p * 60] * b[p + j * 60];
        ref[i + j * 60] = s;
      }
    expect(c).toEqual(ref);
    expect(calls()).toBeGreaterThan(0);
  });

  it("is idempotent per URL (no re-fetch on repeat)", async () => {
    const { exports } = makeFakeWasm();
    const { fetchImpl, urls } = makeFakeFetch();
    const inst = async () => ({ exports }) as unknown as WebAssembly.Instance;
    const p1 = ensureWasmLapackBridge("https://host.example/b/", {
      fetchImpl,
      instantiate: inst,
    });
    const p2 = ensureWasmLapackBridge("https://host.example/b/", {
      fetchImpl,
      instantiate: inst,
    });
    expect(p1).toBe(p2);
    await p1;
    expect(urls.filter(u => u.endsWith(".json")).length).toBe(1);
  });

  it("null URL uninstalls the accelerator", async () => {
    const { exports } = makeFakeWasm();
    const { fetchImpl } = makeFakeFetch();
    await ensureWasmLapackBridge("https://host.example/b/", {
      fetchImpl,
      instantiate: async () => ({ exports }) as unknown as WebAssembly.Instance,
    });
    expect(getMatmulAccelerator()).toBeTypeOf("function");
    const ok = await ensureWasmLapackBridge(null);
    expect(ok).toBe(false);
    expect(getMatmulAccelerator()).toBeUndefined();
  });

  it("resolves false (best-effort) when the manifest 404s", async () => {
    const { fetchImpl } = makeFakeFetch({ manifestOk: false });
    const ok = await ensureWasmLapackBridge("https://host.example/b/", {
      fetchImpl,
      instantiate: async () => {
        throw new Error("should not instantiate");
      },
    });
    expect(ok).toBe(false);
    expect(getMatmulAccelerator()).toBeUndefined();
  });

  it("uses the byte cache when populated, and populates it on a miss", async () => {
    const { exports } = makeFakeWasm();
    const store = new Map<string, Uint8Array>();
    const cache: WasmByteCache = {
      get: async k => store.get(k) ?? null,
      put: async (k, v) => {
        store.set(k, v);
      },
    };
    const first = makeFakeFetch();
    await ensureWasmLapackBridge("https://host.example/b/", {
      fetchImpl: first.fetchImpl,
      instantiate: async () => ({ exports }) as unknown as WebAssembly.Instance,
      cache,
    });
    // wasm URL should now be cached
    expect(store.has("https://host.example/b/bridge.abc123.wasm")).toBe(true);

    // Second load (reset idempotency): the wasm must come from cache, so the
    // fetch layer is never asked for the .wasm URL.
    _resetWasmLapackLoaderForTests();
    const second = makeFakeFetch();
    await ensureWasmLapackBridge("https://host.example/b/", {
      fetchImpl: second.fetchImpl,
      instantiate: async () => ({ exports }) as unknown as WebAssembly.Instance,
      cache,
    });
    expect(second.urls.some(u => u.endsWith(".wasm"))).toBe(false);
    expect(second.urls.some(u => u.endsWith(".json"))).toBe(true);
  });
});
