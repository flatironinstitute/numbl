import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

describe("libFLAME linalg wrapper tree", () => {
  it("defines the narrow C ABI and builds libFLAME with builtin BLAS/LAPACK compatibility", () => {
    const repoRoot = process.cwd();
    const wrapper = readFileSync(
      join(
        repoRoot,
        "browser-wasm",
        "wrappers",
        "libflame-linalg",
        "libflame_linalg_wrapper.c"
      ),
      "utf8"
    );
    const buildScript = readFileSync(
      join(
        repoRoot,
        "browser-wasm",
        "wrappers",
        "libflame-linalg",
        "build-libflame-linalg.sh"
      ),
      "utf8"
    );
    const manifest = readFileSync(
      join(
        repoRoot,
        "browser-wasm",
        "targets",
        "flame-blas-lapack.json"
      ),
      "utf8"
    );
    const readme = readFileSync(
      join(
        repoRoot,
        "browser-wasm",
        "wrappers",
        "libflame-linalg",
        "README.md"
      ),
      "utf8"
    );

    expect(wrapper).toContain("int numbl_matmul_f64");
    expect(wrapper).toContain("int numbl_inv_f64");
    expect(wrapper).toContain("int numbl_linsolve_f64");
    expect(wrapper).toContain("dgemm_");
    expect(wrapper).toContain("dgesv_");
    expect(wrapper).toContain("dgels_");
    expect(wrapper).toContain("out[col * n + col] = 1.0;");

    expect(buildScript).toContain("--enable-builtin-blas");
    expect(buildScript).toContain("--enable-legacy-lapack");
    expect(buildScript).toContain("--enable-lto");
    expect(buildScript).toContain("expected_host=wasm32-unknown-linux-gnu");
    expect(buildScript).toContain('--host="${expected_host}"');
    expect(buildScript).toContain('expected_archive="${build_dir}/lib/${expected_host}/libflame.a"');
    expect(buildScript).toContain("PYTHON=python3");
    expect(buildScript).toContain("FC=true");
    expect(buildScript).toContain("F77=true");
    expect(buildScript).toContain("ac_cv_prog_cc_cross=yes");
    expect(buildScript).toContain("-fno-fast-math");
    expect(buildScript).toContain("-fno-math-errno");
    expect(buildScript).toContain("-ffp-contract=on");
    expect(buildScript).toContain("sed -i 's/ -fno-semantic-interposition//g'");
    expect(buildScript).toContain("--disable-cblas-interfaces");
    expect(buildScript).toContain('NUMBL_BROWSER_WASM_OPT_LEVEL:--O3');
    expect(buildScript).toContain('NUMBL_BROWSER_WASM_LTO:-1');
    expect(buildScript).toContain('NUMBL_BROWSER_WASM_SIMD:-1');
    expect(buildScript).toContain('NUMBL_BROWSER_WASM_FAST_MATH:-0');
    expect(buildScript).toContain('build_dir=${NUMBL_BROWSER_WASM_BUILD_DIR:-"${source_root}/wasm-build-release"}');
    expect(buildScript).not.toContain("--disable-optimizations");

    expect(manifest).toContain('"name": "flame-blas-lapack"');
    expect(manifest).toContain('"enabledByDefault": false');
    expect(manifest).toContain('"sourceRootEnv": "NUMBL_FLAME_BLAS_LAPACK_SRC_ROOT"');

    expect(readme).toContain("numbl_matmul_f64");
    expect(readme).toContain("numbl_inv_f64");
    expect(readme).toContain("numbl_linsolve_f64");
    expect(readme).toContain("--enable-builtin-blas");
    expect(readme).toContain("--enable-legacy-lapack");
    expect(readme).toContain("`-O3 -flto -msimd128 -DNDEBUG`");
  });
});
