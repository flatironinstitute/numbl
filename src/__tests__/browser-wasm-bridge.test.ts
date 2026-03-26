import { afterEach, describe, expect, it, vi } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import {
  ensureBrowserWasmBridgeConfigured,
  resetBrowserWasmBridgeForTests,
} from "../numbl-core/native/browser-wasm-bridge.js";
import { getLapackBridge } from "../numbl-core/native/lapack-bridge.js";

function align8(bytes: number): number {
  return (bytes + 7) & ~7;
}

type InstantiateResult = Awaited<ReturnType<typeof WebAssembly.instantiate>>;

function createInstantiateResult(
  exports: Record<string, unknown>
): InstantiateResult {
  return {
    instance: {
      exports,
    } as unknown as WebAssembly.Instance,
    module: {} as WebAssembly.Module,
  } as unknown as InstantiateResult;
}

function createKernelExports(
  kind: "fft" | "linalg",
  offset = 0,
  options: { rectangularLinsolve?: boolean } = {}
) {
  const memory = new WebAssembly.Memory({ initial: 2 });
  let heapOffset = 1024;

  const malloc = (bytes: number) => {
    const ptr = heapOffset;
    heapOffset += align8(bytes);
    return ptr;
  };

  const free = (_ptr: number) => {};

  const baseExports: Record<string, unknown> = {
    memory,
    malloc,
    free,
  };

  if (kind === "fft") {
    baseExports["numbl_fft1d_f64"] = (
      rePtr: number,
      imPtr: number,
      n: number,
      inverse: number,
      outRePtr: number,
      outImPtr: number
    ) => {
      const re = new Float64Array(memory.buffer, rePtr, n);
      const im = new Float64Array(memory.buffer, imPtr, n);
      const outRe = new Float64Array(memory.buffer, outRePtr, n);
      const outIm = new Float64Array(memory.buffer, outImPtr, n);
      for (let i = 0; i < n; i++) {
        outRe[i] = re[i] + (inverse ? -1 : 1);
        outIm[i] = im[i] - (inverse ? -1 : 1);
      }
      return 0;
    };

    baseExports["numbl_fft_along_dim_f64"] = (
      rePtr: number,
      imPtr: number,
      shapePtr: number,
      ndim: number,
      _dim: number,
      n: number,
      _inverse: number,
      outRePtr: number,
      outImPtr: number
    ) => {
      const shape = new Int32Array(memory.buffer, shapePtr, ndim);
      const inputLen = Array.from(shape).reduce((acc, x) => acc * x, 1);
      const outLen = (inputLen / shape[0]) * n;
      const re = new Float64Array(memory.buffer, rePtr, inputLen);
      const im =
        imPtr === 0
          ? null
          : new Float64Array(memory.buffer, imPtr, inputLen);
      const outRe = new Float64Array(memory.buffer, outRePtr, outLen);
      const outIm = new Float64Array(memory.buffer, outImPtr, outLen);
      for (let i = 0; i < outLen; i++) {
        outRe[i] = re[i % inputLen];
        outIm[i] = im ? im[i % inputLen] : 0;
      }
      return 0;
    };
  } else {
    baseExports["numbl_matmul_f64"] = (
      aPtr: number,
      m: number,
      k: number,
      bPtr: number,
      n: number,
      outPtr: number
    ) => {
      const A = new Float64Array(memory.buffer, aPtr, m * k);
      const B = new Float64Array(memory.buffer, bPtr, k * n);
      const C = new Float64Array(memory.buffer, outPtr, m * n);
      for (let col = 0; col < n; col++) {
        for (let row = 0; row < m; row++) {
          let sum = 0;
          for (let kk = 0; kk < k; kk++) {
            sum += A[row + kk * m] * B[kk + col * k];
          }
          C[row + col * m] = sum + offset;
        }
      }
      return 0;
    };

    baseExports["numbl_inv_f64"] = (
      dataPtr: number,
      n: number,
      outPtr: number
    ) => {
      const input = new Float64Array(memory.buffer, dataPtr, n * n);
      const out = new Float64Array(memory.buffer, outPtr, n * n);
      for (let i = 0; i < input.length; i++) {
        out[i] = input[i] + offset;
      }
      return 0;
    };

    baseExports["numbl_linsolve_f64"] = (
      _aPtr: number,
      m: number,
      n: number,
      bPtr: number,
      nrhs: number,
      outPtr: number
    ) => {
      if (m !== n && !options.rectangularLinsolve) {
        return -4;
      }
      const b = new Float64Array(memory.buffer, bPtr, n * nrhs);
      const out = new Float64Array(memory.buffer, outPtr, n * nrhs);
      for (let i = 0; i < b.length; i++) {
        out[i] = b[i] + offset;
      }
      return 0;
    };
  }

  return baseExports;
}

describe("browser Wasm bridge", () => {
  afterEach(() => {
    resetBrowserWasmBridgeForTests();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("returns false when the manifest is unavailable", async () => {
    vi.stubGlobal("location", { href: undefined });
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: false,
        status: 404,
      }))
    );

    const configured = await ensureBrowserWasmBridgeConfigured();
    expect(configured).toBe(false);
    expect(getLapackBridge()).toBeNull();
  });

  it("loads fft and linalg kernels and exposes bridge methods", async () => {
    vi.stubGlobal("location", { href: "https://example.test/app/" });

    const fetchMock = vi.fn(async (input: string | URL) => {
      const url = String(input);
      if (url.endsWith("/wasm-kernels/manifest.json")) {
        return {
          ok: true,
          json: async () => ({
            targets: [
              {
                name: "ducc0-fft",
                wasmPath: "/wasm-kernels/ducc0-fft.wasm",
              },
              {
                name: "blas-lapack",
                wasmPath: "/wasm-kernels/blas-lapack.wasm",
              },
            ],
          }),
        };
      }
      return {
        ok: true,
        arrayBuffer: async () => new ArrayBuffer(8),
      };
    });
    vi.stubGlobal("fetch", fetchMock);

    const instantiateSpy = vi
      .spyOn(WebAssembly, "instantiate")
      .mockResolvedValueOnce(createInstantiateResult(createKernelExports("fft")))
      .mockResolvedValueOnce(
        createInstantiateResult(createKernelExports("linalg"))
      );

    const configured = await ensureBrowserWasmBridgeConfigured();
    expect(configured).toBe(true);
    expect(instantiateSpy).toHaveBeenCalledTimes(2);

    const bridge = getLapackBridge();
    expect(bridge?.bridgeName).toBe("browser Wasm kernels");
    expect(bridge?.fft1dComplex).toBeTypeOf("function");
    expect(bridge?.matmul).toBeTypeOf("function");
    expect(bridge?.inv).toBeTypeOf("function");
    expect(bridge?.linsolve).toBeTypeOf("function");

    const fft = bridge!.fft1dComplex!(
      new Float64Array([1, 2]),
      new Float64Array([3, 4]),
      2,
      false
    );
    expect(Array.from(fft.re)).toEqual([2, 3]);
    expect(Array.from(fft.im)).toEqual([2, 3]);

    const fftAlongDim = bridge!.fftAlongDim!(
      new Float64Array([1, 2, 3, 4]),
      null,
      [2, 2],
      1,
      2,
      true
    );
    expect(Array.from(fftAlongDim.re)).toEqual([1, 2, 3, 4]);
    expect(Array.from(fftAlongDim.im)).toEqual([0, 0, 0, 0]);

    const matmul = bridge!.matmul!(
      new Float64Array([1, 2, 3, 4]),
      2,
      2,
      new Float64Array([5, 6, 7, 8]),
      2
    );
    expect(Array.from(matmul)).toEqual([23, 34, 31, 46]);

    const inv = bridge!.inv!(new Float64Array([9]), 1);
    expect(Array.from(inv)).toEqual([9]);

    const solve = bridge!.linsolve!(
      new Float64Array([10]),
      1,
      1,
      new Float64Array([11]),
      1
    );
    expect(Array.from(solve)).toEqual([11]);

    const rectangularSolve = bridge!.linsolve!(
      new Float64Array([1, 2]),
      2,
      1,
      new Float64Array([3, 4]),
      1
    );
    expect(rectangularSolve[0]).toBeCloseTo(2.2, 12);
  });

  it("routes square linsolve to blas-lapack and rectangular linsolve to flame-blas-lapack", async () => {
    vi.stubGlobal("location", { href: "https://example.test/app/" });

    const fetchMock = vi.fn(async (input: string | URL) => {
      const url = String(input);
      if (url.endsWith("/wasm-kernels/manifest.json")) {
        return {
          ok: true,
          json: async () => ({
            targets: [
              {
                name: "ducc0-fft",
                wasmPath: "/wasm-kernels/ducc0-fft.wasm",
              },
              {
                name: "blas-lapack",
                wasmPath: "/wasm-kernels/blas-lapack.wasm",
              },
              {
                name: "flame-blas-lapack",
                wasmPath: "/wasm-kernels/flame-blas-lapack.wasm",
              },
            ],
          }),
        };
      }
      return {
        ok: true,
        arrayBuffer: async () => new ArrayBuffer(8),
      };
    });
    vi.stubGlobal("fetch", fetchMock);

    vi.spyOn(WebAssembly, "instantiate")
      .mockResolvedValueOnce(createInstantiateResult(createKernelExports("fft")))
      .mockResolvedValueOnce(
        createInstantiateResult(createKernelExports("linalg", 100))
      )
      .mockResolvedValueOnce(
        createInstantiateResult(
          createKernelExports("linalg", 200, { rectangularLinsolve: true })
        )
      );

    const configured = await ensureBrowserWasmBridgeConfigured();
    expect(configured).toBe(true);

    const bridge = getLapackBridge();
    const matmul = bridge!.matmul!(
      new Float64Array([1, 2, 3, 4]),
      2,
      2,
      new Float64Array([5, 6, 7, 8]),
      2
    );

    expect(Array.from(matmul)).toEqual([123, 134, 131, 146]);

    const squareSolve = bridge!.linsolve!(
      new Float64Array([10]),
      1,
      1,
      new Float64Array([11]),
      1
    );
    expect(Array.from(squareSolve)).toEqual([111]);

    const rectangularSolve = bridge!.linsolve!(
      new Float64Array([1, 2]),
      2,
      1,
      new Float64Array([3, 4]),
      1
    );
    expect(Array.from(rectangularSolve)).toEqual([203]);
  });
});

describe("browser Wasm target manifests", () => {
  it("declares env-driven source roots and static wasm output paths", () => {
    const repoRoot = process.cwd();
    const ducc0 = JSON.parse(
      readFileSync(
        join(repoRoot, "browser-wasm", "targets", "ducc0-fft.json"),
        "utf8"
      )
    ) as {
      name: string;
      sourceRootEnv: string;
      output: string;
      exports: string[];
    };
    const flame = JSON.parse(
      readFileSync(
        join(repoRoot, "browser-wasm", "targets", "flame-blas-lapack.json"),
        "utf8"
      )
    ) as {
      name: string;
      sourceRootEnv: string;
      output: string;
      exports: string[];
    };
    const blas = JSON.parse(
      readFileSync(
        join(repoRoot, "browser-wasm", "targets", "blas-lapack.json"),
        "utf8"
      )
    ) as {
      name: string;
      sourceRootEnv: string;
      output: string;
      exports: string[];
    };

    expect(ducc0.name).toBe("ducc0-fft");
    expect(ducc0.sourceRootEnv).toBe("NUMBL_DUCC0_FFT_SRC_ROOT");
    expect(ducc0.output).toBe("public/wasm-kernels/ducc0-fft.wasm");
    expect(ducc0.exports).toEqual(
      expect.arrayContaining(["malloc", "free", "numbl_fft1d_f64"])
    );

    expect(flame.name).toBe("flame-blas-lapack");
    expect(flame.sourceRootEnv).toBe("NUMBL_FLAME_BLAS_LAPACK_SRC_ROOT");
    expect(flame.output).toBe("public/wasm-kernels/flame-blas-lapack.wasm");
    expect(flame.exports).toEqual(
      expect.arrayContaining(["malloc", "free", "numbl_inv_f64"])
    );

    expect(blas.name).toBe("blas-lapack");
    expect(blas.sourceRootEnv).toBe("NUMBL_BLAS_LAPACK_SRC_ROOT");
    expect(blas.output).toBe("public/wasm-kernels/blas-lapack.wasm");
    expect(blas.exports).toEqual(
      expect.arrayContaining(["malloc", "free", "numbl_inv_f64"])
    );
  });
});
