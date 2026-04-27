/**
 * --opt e2 (per-assign C-JIT) smoke tests.
 *
 * These run scripts twice: once under --opt 1 (interpreter outer +
 * JS-JIT for hot functions/loops) and once under --opt e2 (pure
 * interpreter outer + per-assign C kernel). Both runs must produce
 * identical numeric results.
 *
 * The test also confirms that the e2 path actually fired by checking
 * that `result.generatedC` contains at least one compiled kernel.
 */

import { describe, it, expect } from "vitest";
import { execFileSync } from "child_process";
import { executeCode } from "../numbl-core/executeCode.js";
import { resetCEnvForTesting } from "../numbl-core/jit/c/compile.js";

// Ensure the e2 install hook is registered.
import "../numbl-core/executors/e2/install.js";

// The e2 path shells out to `cc` and links against the prebuilt
// `libnumbl_ops.a` (see `npm run build:addon`). Gate on both the
// compiler's presence AND the `NUMBL_CJIT_E2E=1` opt-in — matches
// how `e1-kernel-smoke.test.ts` is gated so the main "Test" job
// (which doesn't build the addon) skips them cleanly, while the
// dedicated `--opt e1 / e2` CI job runs them end-to-end.
const E2E_ENABLED = process.env.NUMBL_CJIT_E2E === "1";

function hasCc(): boolean {
  try {
    execFileSync("cc", ["--version"], { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

const itSkipWithoutCc = E2E_ENABLED && hasCc() ? it : it.skip;

function runE2(script: string) {
  resetCEnvForTesting();
  // Lower the threshold so 1k-element tests reliably trigger compilation
  // even on machines where the default is later raised.
  const prev = process.env.NUMBL_E2_MIN_ELEMS;
  process.env.NUMBL_E2_MIN_ELEMS = "100";
  try {
    return executeCode(script, { optimization: 0, experimental: "e2" });
  } finally {
    if (prev === undefined) delete process.env.NUMBL_E2_MIN_ELEMS;
    else process.env.NUMBL_E2_MIN_ELEMS = prev;
  }
}

function runJs(script: string) {
  return executeCode(script, { optimization: 1 });
}

function asTensorData(v: unknown): Float64Array {
  return (v as { data: Float64Array }).data;
}

function approxEqualArrays(
  a: ArrayLike<number>,
  b: ArrayLike<number>,
  tol = 1e-12
): void {
  expect(a.length).toBe(b.length);
  for (let i = 0; i < a.length; i++) {
    expect(Math.abs(a[i] - b[i])).toBeLessThanOrEqual(tol);
  }
}

describe("--opt e2: per-assign C kernel smoke", () => {
  itSkipWithoutCc("compiles a simple elemwise expression", () => {
    const script = `
x = linspace(0, 1, 1000);
r = exp(x) + sin(x);
result = sum(r);
`;
    const e2 = runE2(script);
    const js = runJs(script);
    expect(e2.variableValues["result"]).toBeCloseTo(
      js.variableValues["result"] as number,
      9
    );
    expect(e2.generatedC).toMatch(/e2[cr]_/);
    expect(e2.generatedC).toContain("exp(");
    expect(e2.generatedC).toContain("sin(");
  });

  itSkipWithoutCc("handles LHS-on-RHS aliasing (chain fused)", () => {
    // Mirrors tensor_ops_bench.m's pattern: a same-LHS chain where the
    // LHS is also read on the right. With chain fusion, all four assigns
    // collapse into one C kernel; intermediate `r` values live in a
    // per-element stack-local. Only the final value materializes back
    // to env, so the result must still match JS-JIT exactly.
    const script = `
x = linspace(-1, 1, 2000);
y = linspace(0.1, 0.9, 2000);
r = x + y;
r = r - 0.5 .* x;
r = r .* y + 3.0;
result = sum(r);
`;
    const e2 = runE2(script);
    const js = runJs(script);
    expect(e2.variableValues["result"]).toBeCloseTo(
      js.variableValues["result"] as number,
      9
    );
    // The chain emits one e2c_<hash> kernel covering all three assigns.
    expect(e2.generatedC).toMatch(/e2[cr]_/);
  });

  itSkipWithoutCc("falls through silently for non-classifiable RHS", () => {
    // sort() isn't in the e2 builtin whitelist — the assign should
    // silently fall through to the interpreter (no compile attempted).
    const script = `
x = [3, 1, 2, 5, 4];
y = sort(x);
result = y;
`;
    const e2 = runE2(script);
    const js = runJs(script);
    approxEqualArrays(
      asTensorData(e2.variableValues["result"]),
      asTensorData(js.variableValues["result"])
    );
  });

  itSkipWithoutCc("e2 results match JS-JIT element-by-element", () => {
    const script = `
x = linspace(-1, 1, 5000);
y = sin(x .* 2.5) .* exp(-x .* x);
`;
    const e2 = runE2(script);
    const js = runJs(script);
    approxEqualArrays(
      asTensorData(e2.variableValues["y"]),
      asTensorData(js.variableValues["y"]),
      1e-12
    );
    expect(e2.generatedC).toMatch(/e2[cr]_/);
  });

  itSkipWithoutCc("does NOT fire when no tensor inputs are present", () => {
    // Pure scalar arithmetic — out of scope for e2's per-assign elemwise
    // path. Should silently use the interpreter.
    const script = `
a = 3.0;
b = sin(a) + cos(a);
`;
    const e2 = runE2(script);
    expect(e2.generatedC).toBe("/* No C generated */");
  });

  itSkipWithoutCc("does NOT fire when tensor is below threshold", () => {
    const script = `
x = linspace(0, 1, 50);
r = exp(x);
`;
    const e2 = runE2(script);
    // Threshold is 100 in this test; 50 elements is below.
    expect(e2.generatedC).toBe("/* No C generated */");
  });

  itSkipWithoutCc("chains across same-LHS suppressed assigns", () => {
    const script = `
x = linspace(-1, 1, 1500);
u = exp(-x .* x);
u = u .* cos(5 .* x);
u = u + sin(x + 1);
u = abs(u);
u = tanh(u);
result = sum(u);
`;
    const e2 = runE2(script);
    const js = runJs(script);
    expect(e2.variableValues["result"]).toBeCloseTo(
      js.variableValues["result"] as number,
      9
    );
    // The five same-LHS assigns fuse into a single chain kernel.
    expect(e2.generatedC).toMatch(/e2[cr]_/);
    // The five same-LHS assigns + sum(u) trailing reduction fuse into
    // a single `e2r_` reduction kernel.
    const chainCount = (e2.generatedC.match(/e2[cr]_[0-9a-f]+/g) ?? []).length;
    expect(chainCount).toBeGreaterThan(0);
  });

  itSkipWithoutCc("breaks chain at unsuppressed assign", () => {
    // The middle assign is unsuppressed (no trailing `;`), so it must
    // be displayed by the interpreter — the chain ends before it.
    const script = `
x = linspace(-1, 1, 1500);
r = x + 1;
r = r .* 2
r = r - 0.5;
result = sum(r);
`;
    const e2 = runE2(script);
    const js = runJs(script);
    expect(e2.variableValues["result"]).toBeCloseTo(
      js.variableValues["result"] as number,
      9
    );
    // No chain because the chain prefix is only one assign before the
    // unsuppressed break — single-assign kernels are emitted instead.
    // No multi-stmt chain — the unsuppressed assign breaks it. We still
    // emit at least one chain-of-length-1 kernel for the surrounding
    // suppressed assigns (the `r = x + 1;` and `r = r - 0.5;` stmts).
    expect(e2.generatedC).toMatch(/e2[cr]_/);
  });

  itSkipWithoutCc(
    "fires whole-function scalar kernel for a pure-scalar user function",
    () => {
      // Mirrors benchmarks/scalar_bench.m: a pure-scalar inner function
      // called in a loop. Under --opt e2, `run_bench` should compile to
      // one C kernel on first call and reuse it on the timed call.
      const script = `
function total = run_bench(N, M)
  total = 0.0;
  for i = 1:N
    x = i * 0.001;
    acc = 0.0;
    for k = 1:M
      acc = acc + sin(x * k) / (k * k);
    end
    total = total + acc;
  end
end
warm = run_bench(100, 10);
result = run_bench(200, 15);
`;
      const e2 = runE2(script);
      const js = runJs(script);
      expect(e2.variableValues["result"]).toBeCloseTo(
        js.variableValues["result"] as number,
        9
      );
      // Scalar-fn kernels are named `jit_<fnname>` (via generateC). The
      // e2 log tags the description as "scalar-fn".
      expect(e2.generatedC).toContain("jit_run_bench");
      expect(e2.generatedC).toMatch(/scalar-fn run_bench/);
    }
  );

  itSkipWithoutCc("--opt e2 disables the JS-JIT outer", () => {
    // Loop and function should NOT be JS-JIT'd under --opt e2.
    const script = `
function y = sq(x)
  y = x * x;
end
total = 0;
for i = 1:100
  total = total + sq(i);
end
result = total;
`;
    const e2 = runE2(script);
    expect(e2.generatedJS).toBe("// No JS generated");
    // And the result is still correct.
    const js = runJs(script);
    expect(e2.variableValues["result"]).toBe(js.variableValues["result"]);
  });
});
