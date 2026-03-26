import { describe, expect, it } from "vitest";

// @ts-expect-error tests import the Node build helper directly
import { mergeRuntimeManifestTargets } from "../../scripts/build-browser-wasm.mjs";

describe("build-browser-wasm manifest handling", () => {
  it("merges successful single-target builds into the existing runtime manifest", () => {
    const merged = mergeRuntimeManifestTargets(
      [
        {
          name: "ducc0-fft",
          wasmPath: "/wasm-kernels/ducc0-fft.wasm",
          exports: ["malloc", "free", "numbl_fft1d_f64"],
        },
        {
          name: "blas-lapack",
          wasmPath: "/wasm-kernels/blas-lapack.wasm",
          exports: ["malloc", "free", "numbl_matmul_f64"],
        },
      ],
      [
        {
          name: "flame-blas-lapack",
          status: "built",
          wasmPath: "/wasm-kernels/flame-blas-lapack.wasm",
          exports: ["malloc", "free", "numbl_linsolve_f64"],
        },
      ]
    );

    expect(merged).toEqual([
      {
        name: "blas-lapack",
        wasmPath: "/wasm-kernels/blas-lapack.wasm",
        exports: ["malloc", "free", "numbl_matmul_f64"],
      },
      {
        name: "ducc0-fft",
        wasmPath: "/wasm-kernels/ducc0-fft.wasm",
        exports: ["malloc", "free", "numbl_fft1d_f64"],
      },
      {
        name: "flame-blas-lapack",
        wasmPath: "/wasm-kernels/flame-blas-lapack.wasm",
        exports: ["malloc", "free", "numbl_linsolve_f64"],
      },
    ]);
  });
});
