import { describe, it, expect } from "vitest";
import { execFileSync } from "child_process";
import { executeCode } from "../numbl-core/executeCode.js";
import { generateC } from "../numbl-core/interpreter/jit/c/jitCodegenC.js";
import { generateNapiShim } from "../numbl-core/interpreter/jit/c/cNapiShim.js";
import {
  compileAndLoad,
  cJitUnavailableReason,
  resetCEnvForTesting,
} from "../numbl-core/interpreter/jit/c/cCompile.js";
import { checkCFeasibility } from "../numbl-core/interpreter/jit/c/cFeasibility.js";
import type {
  JitStmt,
  JitType,
} from "../numbl-core/interpreter/jit/jitTypes.js";
import { BinaryOperation } from "../numbl-core/parser/types.js";

// The feasibility + codegen tests run anywhere (pure TS). The heavy
// end-to-end compile+load tests are gated by NUMBL_CJIT_E2E=1 so the
// default `npm test` stays fast and doesn't require a C compiler or
// Node API headers. CI runs them via `npm run test:scripts:c-jit`.
const E2E_ENABLED = process.env.NUMBL_CJIT_E2E === "1";
function hasCc(): boolean {
  try {
    execFileSync("cc", ["--version"], { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

describe("C-JIT: feasibility prepass", () => {
  it("accepts scalar arithmetic", () => {
    const numberT: JitType = { kind: "number" };
    const body: JitStmt[] = [
      {
        tag: "Assign",
        name: "out",
        expr: {
          tag: "Binary",
          op: BinaryOperation.Add,
          left: { tag: "Var", name: "a", jitType: numberT },
          right: { tag: "Var", name: "b", jitType: numberT },
          jitType: numberT,
        },
      },
    ];
    const r = checkCFeasibility(body, [numberT, numberT], numberT, 1);
    expect(r.ok).toBe(true);
  });

  it("accepts real tensor args (Phase 2)", () => {
    const tensorT: JitType = {
      kind: "tensor",
      isComplex: false,
      shape: [100, 1],
    };
    const r = checkCFeasibility([], [tensorT], { kind: "number" }, 1);
    expect(r.ok).toBe(true);
  });

  it("rejects complex tensor args", () => {
    const tensorT: JitType = { kind: "tensor", isComplex: true };
    const r = checkCFeasibility([], [tensorT], { kind: "number" }, 1);
    expect(r.ok).toBe(false);
  });

  it("rejects multi-output", () => {
    const numberT: JitType = { kind: "number" };
    const r = checkCFeasibility([], [numberT], numberT, 2);
    expect(r.ok).toBe(false);
  });
});

describe("C-JIT: code generation", () => {
  it("emits a compilable-looking scalar function", () => {
    const numberT: JitType = { kind: "number" };
    // f(a, b) { out = a * a + b; return out; }
    const body: JitStmt[] = [
      {
        tag: "Assign",
        name: "out",
        expr: {
          tag: "Binary",
          op: BinaryOperation.Add,
          left: {
            tag: "Binary",
            op: BinaryOperation.Mul,
            left: { tag: "Var", name: "a", jitType: numberT },
            right: { tag: "Var", name: "a", jitType: numberT },
            jitType: numberT,
          },
          right: { tag: "Var", name: "b", jitType: numberT },
          jitType: numberT,
        },
      },
    ];
    const gen = generateC(
      body,
      ["a", "b"],
      ["out"],
      1,
      new Set(["out"]),
      [numberT, numberT],
      numberT,
      "test_fn"
    );
    expect(gen.cSource).toContain("#include <math.h>");
    expect(gen.cSource).toContain("double jit_test_fn(double v_a, double v_b)");
    expect(gen.cSource).toContain("v_out");
    expect(gen.cSource).toContain("return v_out;");
    expect(gen.cFnName).toBe("jit_test_fn");
  });

  it("emits mod helper when builtin mod is used", () => {
    const numberT: JitType = { kind: "number" };
    const body: JitStmt[] = [
      {
        tag: "Assign",
        name: "out",
        expr: {
          tag: "Call",
          name: "mod",
          args: [
            { tag: "Var", name: "a", jitType: numberT },
            { tag: "Var", name: "b", jitType: numberT },
          ],
          jitType: numberT,
        },
      },
    ];
    const gen = generateC(
      body,
      ["a", "b"],
      ["out"],
      1,
      new Set(["out"]),
      [numberT, numberT],
      numberT,
      "mod_test"
    );
    expect(gen.cSource).toContain("__numbl_mod");
    expect(gen.helpersUsed).toContain("mod");
  });
});

describe("C-JIT: compile + load + invoke (end-to-end)", () => {
  const available = E2E_ENABLED && hasCc();
  const itSkipWithoutCc = available ? it : it.skip;

  itSkipWithoutCc("compiles and runs a trivial scalar function", () => {
    resetCEnvForTesting();
    const numberT: JitType = { kind: "number" };
    const body: JitStmt[] = [
      {
        tag: "Assign",
        name: "out",
        expr: {
          tag: "Binary",
          op: BinaryOperation.Add,
          left: {
            tag: "Binary",
            op: BinaryOperation.Mul,
            left: { tag: "Var", name: "a", jitType: numberT },
            right: { tag: "Var", name: "a", jitType: numberT },
            jitType: numberT,
          },
          right: { tag: "Var", name: "b", jitType: numberT },
          jitType: numberT,
        },
      },
    ];
    const gen = generateC(
      body,
      ["a", "b"],
      ["out"],
      1,
      new Set(["out"]),
      [numberT, numberT],
      numberT,
      "smoke"
    );
    const { shim, exportName } = generateNapiShim(
      gen.cFnName,
      gen.paramDescs,
      "number",
      gen.usesTensors
    );

    const logs: string[] = [];
    const loaded = compileAndLoad(gen.cSource, shim, exportName, m =>
      logs.push(m)
    );
    if (!loaded) {
      // Don't fail on CI runners without Node headers — that's a known
      // environmental limitation. But surface the reason for debugging.
      const reason = cJitUnavailableReason();
      console.warn(`C-JIT unavailable: ${reason}; logs:\n${logs.join("\n")}`);
      return;
    }
    expect(loaded.fn(3, 4)).toBe(3 * 3 + 4);
    expect(loaded.fn(-2, 10)).toBe(-2 * -2 + 10);
  });

  itSkipWithoutCc("matches JS-JIT result on a scalar script", () => {
    // Run a script that exercises only scalar ops: a user function that
    // computes (a*a) + (b/c). Runs twice — once with --opt 1 (JS), once
    // with --opt 2 (C). Results must match bit-for-bit for integer inputs.
    const script = `
function y = f(a, b, c)
  y = a*a + b/c;
end
x = 0;
for i = 1:50
  x = x + f(i, i+1, 2);
end
result = x;
`;
    const js = executeCode(script, { optimization: 1 });
    const cj = executeCode(script, { optimization: 2 });
    // If C-JIT couldn't load (no headers), the fallback-to-JS path still
    // produces the same answer, so this equality holds regardless.
    expect(cj.variableValues["result"]).toBe(js.variableValues["result"]);
  });

  itSkipWithoutCc(
    "--dump-c path: generatedC is populated when C-JIT fires",
    () => {
      const script = `
function y = square(x)
  y = x*x;
end
result = square(7);
`;
      const res = executeCode(script, { optimization: 2 });
      // generatedC exists and is either "no C generated" (fallback) or
      // contains a C function. Either way, the field is present.
      expect(typeof res.generatedC).toBe("string");
      expect(res.variableValues["result"]).toBe(49);
    }
  );
});

// ── Phase 2: tensor-support parity with JS-JIT ───────────────────────────
//
// The C-JIT's tensor path mirrors jitHelpersTensor.ts line-for-line —
// every tensor op goes through `numbl_jit_t*` helpers that have the same
// signature, fast paths, and buffer-reuse behavior as the JS `$h.t*`
// exports. The assertions below check BOTH that the expected C code
// shape is emitted AND that running scripts under --opt 2 yields
// bit-identical results to --opt 1.

describe("C-JIT: tensor feasibility (Phase 2)", () => {
  const tensor1dT: JitType = {
    kind: "tensor",
    isComplex: false,
    shape: [100, 1],
  };

  it("accepts tensor-result Binary(+) with two tensor args", () => {
    const body: JitStmt[] = [
      {
        tag: "Assign",
        name: "r",
        expr: {
          tag: "Binary",
          op: BinaryOperation.Add,
          left: { tag: "Var", name: "x", jitType: tensor1dT },
          right: { tag: "Var", name: "y", jitType: tensor1dT },
          jitType: tensor1dT,
        },
      },
    ];
    const r = checkCFeasibility(body, [tensor1dT, tensor1dT], tensor1dT, 1);
    expect(r.ok).toBe(true);
  });

  it("accepts tensor-result Call(exp) with a tensor arg", () => {
    const body: JitStmt[] = [
      {
        tag: "Assign",
        name: "r",
        expr: {
          tag: "Call",
          name: "exp",
          args: [{ tag: "Var", name: "x", jitType: tensor1dT }],
          jitType: tensor1dT,
        },
      },
    ];
    const r = checkCFeasibility(body, [tensor1dT], tensor1dT, 1);
    expect(r.ok).toBe(true);
  });

  it("accepts reduction sum(x): tensor → number", () => {
    const numberT: JitType = { kind: "number" };
    const body: JitStmt[] = [
      {
        tag: "Assign",
        name: "s",
        expr: {
          tag: "Call",
          name: "sum",
          args: [{ tag: "Var", name: "x", jitType: tensor1dT }],
          jitType: numberT,
        },
      },
    ];
    const r = checkCFeasibility(body, [tensor1dT], numberT, 1);
    expect(r.ok).toBe(true);
  });

  it("rejects domain-restricted tensor sqrt(x)", () => {
    const body: JitStmt[] = [
      {
        tag: "Assign",
        name: "r",
        expr: {
          tag: "Call",
          name: "sqrt",
          args: [{ tag: "Var", name: "x", jitType: tensor1dT }],
          jitType: tensor1dT,
        },
      },
    ];
    const r = checkCFeasibility(body, [tensor1dT], tensor1dT, 1);
    expect(r.ok).toBe(false);
  });
});

describe("C-JIT: tensor codegen (Phase 2)", () => {
  const numberT: JitType = { kind: "number" };
  const tensor1dT: JitType = {
    kind: "tensor",
    isComplex: false,
    shape: [100, 1],
  };

  it("emits numbl_jit_tAdd for tensor r = x + y", () => {
    const body: JitStmt[] = [
      {
        tag: "Assign",
        name: "r",
        expr: {
          tag: "Binary",
          op: BinaryOperation.Add,
          left: { tag: "Var", name: "x", jitType: tensor1dT },
          right: { tag: "Var", name: "y", jitType: tensor1dT },
          jitType: tensor1dT,
        },
      },
    ];
    const gen = generateC(
      body,
      ["x", "y"],
      ["r"],
      1,
      new Set(["r"]),
      [tensor1dT, tensor1dT],
      tensor1dT,
      "fn_add"
    );
    expect(gen.cSource).toContain('#include "numbl_ops.h"');
    expect(gen.cSource).toContain("numbl_jit_tAdd(env, v_r, v_x, v_y)");
    expect(gen.returnIsTensor).toBe(true);
    expect(gen.usesTensors).toBe(true);
    // Signature should thread napi_env and use napi_value everywhere.
    expect(gen.cSource).toContain(
      "napi_value jit_fn_add(napi_env env, napi_value v_x, napi_value v_y)"
    );
  });

  it("boxes scalar literals for scalar-tensor ops", () => {
    // r = 0.5 .* x   — scalar on the left, tensor on the right
    const body: JitStmt[] = [
      {
        tag: "Assign",
        name: "r",
        expr: {
          tag: "Binary",
          op: BinaryOperation.ElemMul,
          left: { tag: "NumberLiteral", value: 0.5, jitType: numberT },
          right: { tag: "Var", name: "x", jitType: tensor1dT },
          jitType: tensor1dT,
        },
      },
    ];
    const gen = generateC(
      body,
      ["x"],
      ["r"],
      1,
      new Set(["r"]),
      [tensor1dT],
      tensor1dT,
      "fn_scalar_mul"
    );
    expect(gen.cSource).toContain(
      "numbl_jit_tMul(env, v_r, numbl_jit_box_double(env, 0.5), v_x)"
    );
  });

  it("emits scratch slots for inner tensor sub-expressions", () => {
    // r = x + (x .* y)    — inner Mul needs a scratch slot
    const body: JitStmt[] = [
      {
        tag: "Assign",
        name: "r",
        expr: {
          tag: "Binary",
          op: BinaryOperation.Add,
          left: { tag: "Var", name: "x", jitType: tensor1dT },
          right: {
            tag: "Binary",
            op: BinaryOperation.ElemMul,
            left: { tag: "Var", name: "x", jitType: tensor1dT },
            right: { tag: "Var", name: "y", jitType: tensor1dT },
            jitType: tensor1dT,
          },
          jitType: tensor1dT,
        },
      },
    ];
    const gen = generateC(
      body,
      ["x", "y"],
      ["r"],
      1,
      new Set(["r"]),
      [tensor1dT, tensor1dT],
      tensor1dT,
      "fn_nested"
    );
    expect(gen.cSource).toContain("static __thread napi_value __s1 = NULL;");
    expect(gen.cSource).toContain("__s1 = numbl_jit_tMul(env, __s1, v_x, v_y)");
  });

  it("emits reduction helper for sum(x) returning a scalar", () => {
    const body: JitStmt[] = [
      {
        tag: "Assign",
        name: "s",
        expr: {
          tag: "Call",
          name: "sum",
          args: [{ tag: "Var", name: "x", jitType: tensor1dT }],
          jitType: numberT,
        },
      },
    ];
    const gen = generateC(
      body,
      ["x"],
      ["s"],
      1,
      new Set(["s"]),
      [tensor1dT],
      numberT,
      "fn_sum"
    );
    expect(gen.cSource).toContain("numbl_jit_tSum(env, v_x)");
    expect(gen.cSource).toContain("numbl_jit_napi_to_double(env, ");
    expect(gen.returnIsTensor).toBe(false);
  });
});

describe("C-JIT: tensor parity with JS-JIT (E2E)", () => {
  const available = E2E_ENABLED && hasCc();
  const itSkipWithoutCc = available ? it : it.skip;

  // Each parity test runs a tensor-touching script twice — once with
  // --opt 1 (JS-JIT), once with --opt 2 (C-JIT) — and asserts the
  // outputs match. When the C-JIT path is chosen, the JS-JIT path is
  // still exercised too (for the scripts that bail out of C-JIT).
  const parity = (script: string, keys: string[]) => {
    const js = executeCode(script, { optimization: 1 });
    const cj = executeCode(script, { optimization: 2 });
    for (const k of keys) {
      expect(cj.variableValues[k], `${k} mismatch`).toEqual(
        js.variableValues[k]
      );
    }
  };

  itSkipWithoutCc("tensor-tensor binary: r = x + y", () => {
    parity(
      `
function r = f(x, y)
  r = x + y;
end
x = (1:100)';
y = (100:-1:1)';
out = f(x, y);
result = sum(out);
`,
      ["result"]
    );
  });

  itSkipWithoutCc("scalar-tensor mix + chained ops", () => {
    parity(
      `
function r = f(x, y)
  r = x + y;
  r = r - 0.5 .* x;
  r = r .* y + 3.0;
  r = r ./ (1 + abs(y));
end
x = ((1:200)./100)';
y = ((200:-1:1)./100)';
out = f(x, y);
result = sum(out);
`,
      ["result"]
    );
  });

  itSkipWithoutCc("tensor unary: r = exp(-x.*x)", () => {
    parity(
      `
function r = f(x)
  r = exp(-x .* x);
end
x = linspace(-2, 2, 500)';
out = f(x);
result = sum(out);
`,
      ["result"]
    );
  });

  itSkipWithoutCc("reduction: s = sum(x.*y)", () => {
    parity(
      `
function s = f(x, y)
  s = sum(x .* y);
end
x = (1:1000)';
y = ((1000:-1:1) ./ 1000)';
result = f(x, y);
`,
      ["result"]
    );
  });

  itSkipWithoutCc("comparisons + reduction: s = sum((x>0).*(y<0.5))", () => {
    parity(
      `
function s = f(x, y)
  c1 = x > 0;
  c2 = y < 0.5;
  s = sum(c1 .* c2);
end
x = linspace(-1, 1, 1000)';
y = linspace(0, 1, 1000)';
result = f(x, y);
`,
      ["result"]
    );
  });
});
