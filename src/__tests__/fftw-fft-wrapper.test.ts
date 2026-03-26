import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

describe("fftw FFT wrapper tree", () => {
  it("builds the standalone FFTW browser target with the release wasm profile", () => {
    const repoRoot = process.cwd();
    const wrapper = readFileSync(
      join(repoRoot, "browser-wasm", "wrappers", "fftw-fft", "fftw_fft_wrapper.c"),
      "utf8"
    );
    const buildScript = readFileSync(
      join(repoRoot, "browser-wasm", "wrappers", "fftw-fft", "build-fftw-fft.sh"),
      "utf8"
    );
    const readme = readFileSync(
      join(repoRoot, "browser-wasm", "wrappers", "fftw-fft", "README.md"),
      "utf8"
    );

    expect(wrapper).toContain("int numbl_fft1d_f64");
    expect(buildScript).toContain('NUMBL_BROWSER_WASM_OPT_LEVEL:--O3');
    expect(buildScript).toContain('NUMBL_BROWSER_WASM_LTO:-1');
    expect(buildScript).toContain('NUMBL_BROWSER_WASM_SIMD:-1');
    expect(buildScript).toContain('NUMBL_BROWSER_WASM_FAST_MATH:-0');
    expect(buildScript).toContain("-flto");
    expect(buildScript).toContain("-DNDEBUG");
    expect(buildScript).toContain("-fno-fast-math");
    expect(buildScript).toContain("-fno-math-errno");
    expect(buildScript).toContain("-ffp-contract=on");
    expect(readme).toContain("`-O3 -flto -msimd128 -DNDEBUG`");
  });
});
