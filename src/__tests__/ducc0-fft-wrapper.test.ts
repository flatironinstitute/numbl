import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

describe("ducc0 FFT wrapper tree", () => {
  it("defines the documented C ABI and stages against an external ducc checkout", () => {
    const repoRoot = process.cwd();
    const wrapper = readFileSync(
      join(repoRoot, "browser-wasm", "wrappers", "ducc0-fft", "ducc0_fft_wrapper.cpp"),
      "utf8"
    );
    const cmake = readFileSync(
      join(repoRoot, "browser-wasm", "wrappers", "ducc0-fft", "CMakeLists.txt"),
      "utf8"
    );
    const buildScript = readFileSync(
      join(repoRoot, "browser-wasm", "wrappers", "ducc0-fft", "build-ducc0-fft.sh"),
      "utf8"
    );
    const readme = readFileSync(
      join(repoRoot, "browser-wasm", "wrappers", "ducc0-fft", "README.md"),
      "utf8"
    );

    expect(wrapper).toContain('#include "ducc0/fft/fft.h"');
    expect(wrapper).toContain("extern \"C\" int numbl_fft1d_f64");
    expect(wrapper).toContain("extern \"C\" int numbl_fft_along_dim_f64");
    expect(wrapper).toContain("ducc0::c2c");
    expect(wrapper).toContain("copyAxisInput");
    expect(wrapper).toContain("packComplex");
    expect(wrapper).toContain("unpackComplex");
    expect(wrapper).toContain("extern \"C\" int numbl_fft1d_f64");
    expect(wrapper).toContain("extern \"C\" int numbl_fft_along_dim_f64");

    expect(cmake).toContain("DUCC0_SRC_ROOT");
    expect(cmake).toContain("ducc0_fft_wrapper.cpp");
    expect(buildScript).toContain('NUMBL_BROWSER_WASM_OPT_LEVEL:--O3');
    expect(buildScript).toContain('NUMBL_BROWSER_WASM_LTO:-1');
    expect(buildScript).toContain('NUMBL_BROWSER_WASM_SIMD:-1');
    expect(buildScript).toContain('NUMBL_BROWSER_WASM_FAST_MATH:-0');
    expect(buildScript).toContain("-flto");
    expect(buildScript).toContain("-DNDEBUG");
    expect(buildScript).toContain("-fno-fast-math");
    expect(buildScript).toContain("-fno-math-errno");
    expect(buildScript).toContain("-ffp-contract=on");

    expect(readme).toContain("numbl_fft1d_f64");
    expect(readme).toContain("numbl_fft_along_dim_f64");
    expect(readme).toContain("/abs/path/to/ducc");
  });
});
