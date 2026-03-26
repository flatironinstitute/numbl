import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

describe("OpenBLAS linalg wrapper tree", () => {
  it("defines the narrow C ABI and builds OpenBLAS for wasm with selected LAPACKE work shims", () => {
    const repoRoot = process.cwd();
    const wrapper = readFileSync(
      join(
        repoRoot,
        "browser-wasm",
        "wrappers",
        "openblas-linalg",
        "openblas_linalg_wrapper.c"
      ),
      "utf8"
    );
    const buildScript = readFileSync(
      join(
        repoRoot,
        "browser-wasm",
        "wrappers",
        "openblas-linalg",
        "build-openblas-linalg.sh"
      ),
      "utf8"
    );
    const readme = readFileSync(
      join(
        repoRoot,
        "browser-wasm",
        "wrappers",
        "openblas-linalg",
        "README.md"
      ),
      "utf8"
    );

    expect(wrapper).toContain('#include "lapacke.h"');
    expect(wrapper).toContain("int numbl_matmul_f64");
    expect(wrapper).toContain("int numbl_inv_f64");
    expect(wrapper).toContain("int numbl_linsolve_f64");
    expect(wrapper).toContain("cblas_dgemm");
    expect(wrapper).toContain("LAPACKE_dgetrf_work");
    expect(wrapper).toContain("LAPACKE_dgetri_work");
    expect(wrapper).toContain("LAPACKE_dgesv_work");
    expect(wrapper).not.toContain("LAPACKE_dgels_work");
    expect(wrapper).toContain("if (m != n)");
    expect(wrapper).toContain("return -4;");

    expect(buildScript).toContain("TARGET=WASM128_GENERIC");
    expect(buildScript).toContain("NOFORTRAN=1");
    expect(buildScript).toContain("NUM_THREADS=1");
    expect(buildScript).toContain("USE_SIMPLE_THREADED_LEVEL3=1");
    expect(buildScript).toContain("BUILD_DOUBLE=1");
    expect(buildScript).toContain("BUILD_SINGLE=0");
    expect(buildScript).toContain("BUILD_COMPLEX=0");
    expect(buildScript).toContain("BUILD_COMPLEX16=0");
    expect(buildScript).toContain('NUMBL_BROWSER_WASM_OPT_LEVEL:--O3');
    expect(buildScript).toContain('NUMBL_BROWSER_WASM_LTO:-1');
    expect(buildScript).toContain('NUMBL_BROWSER_WASM_SIMD:-1');
    expect(buildScript).toContain('NUMBL_BROWSER_WASM_FAST_MATH:-0');
    expect(buildScript).toContain('COMMON_OPT="${compile_flags_str}"');
    expect(buildScript).toContain('LAPACK_CFLAGS="${compile_flags_str}"');
    expect(buildScript).toContain("-flto");
    expect(buildScript).toContain("-DNDEBUG");
    expect(buildScript).toContain("-fno-fast-math");
    expect(buildScript).toContain("-fno-math-errno");
    expect(buildScript).toContain("-ffp-contract=on");
    expect(buildScript).toContain("lapacke_dgetrf_work.c");
    expect(buildScript).toContain("lapacke_dgetri_work.c");
    expect(buildScript).toContain("lapacke_dgesv_work.c");
    expect(buildScript).not.toContain("lapacke_dgels_work.c");

    expect(readme).toContain("numbl_matmul_f64");
    expect(readme).toContain("numbl_inv_f64");
    expect(readme).toContain("numbl_linsolve_f64");
    expect(readme).toContain("WASM128_GENERIC");
    expect(readme).toContain("NOFORTRAN=1");
    expect(readme).toContain("`-O3 -flto -msimd128 -DNDEBUG`");
  });
});
