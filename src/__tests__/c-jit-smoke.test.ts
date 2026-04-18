import { describe, it, expect } from "vitest";
import { execFileSync } from "child_process";
import { executeCode } from "../numbl-core/executeCode.js";
import { generateC } from "../numbl-core/jit/c/jitCodegenC.js";
import {
  compileAndLoad,
  cJitUnavailableReason,
  resetCEnvForTesting,
} from "../numbl-core/jit/c/cCompile.js";
import { checkCFeasibility } from "../numbl-core/jit/c/cFeasibility.js";
import type { JitStmt, JitType } from "../numbl-core/jit/jitTypes.js";
import { BinaryOperation } from "../numbl-core/parser/types.js";

// Register the C-JIT backend so executeCode-based tests exercise it.
import "../numbl-core/jit/c/cJitInstall.js";

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
    const r = checkCFeasibility(
      body,
      ["a", "b"],
      [numberT, numberT],
      numberT,
      [numberT],
      1
    );
    expect(r.ok).toBe(true);
  });

  it("accepts real tensor args (Phase 2)", () => {
    const tensorT: JitType = {
      kind: "tensor",
      isComplex: false,
      shape: [100, 1],
    };
    const numberT: JitType = { kind: "number" };
    const r = checkCFeasibility([], ["x"], [tensorT], numberT, [numberT], 1);
    expect(r.ok).toBe(true);
  });

  it("rejects complex tensor args", () => {
    const tensorT: JitType = { kind: "tensor", isComplex: true };
    const numberT: JitType = { kind: "number" };
    const r = checkCFeasibility([], ["x"], [tensorT], numberT, [numberT], 1);
    expect(r.ok).toBe(false);
  });

  it("accepts multi-output when all outputs are scalar/tensor", () => {
    const numberT: JitType = { kind: "number" };
    const r = checkCFeasibility(
      [],
      ["x"],
      [numberT],
      numberT,
      [numberT, numberT],
      2
    );
    expect(r.ok).toBe(true);
  });
});

describe("C-JIT: code generation (koffi)", () => {
  it("emits a compilable-looking scalar function", () => {
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
      [numberT],
      "test_fn"
    );
    expect(gen.cSource).toContain("#include <math.h>");
    // koffi path: void return, scalar out-pointer
    expect(gen.cSource).toContain(
      "void jit_test_fn(double v_a, double v_b, double *v_out_out)"
    );
    expect(gen.cSource).toContain("v_out");
    expect(gen.cSource).toContain("*v_out_out = v_out;");
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
      [numberT],
      "mod_test"
    );
    expect(gen.cSource).toContain("numbl_mod");
  });
});

describe("C-JIT: compile + load + invoke (end-to-end, koffi)", () => {
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
      [numberT],
      "smoke"
    );

    const logs: string[] = [];
    const loaded = compileAndLoad(
      gen.cSource,
      gen.koffiSignature,
      gen.cFnName,
      m => logs.push(m)
    );
    if (!loaded) {
      const reason = cJitUnavailableReason();
      console.warn(`C-JIT unavailable: ${reason}; logs:\n${logs.join("\n")}`);
      return;
    }
    // koffi path: void function with out-pointer. Call with scalar args
    // and a Float64Array(1) for the output.
    const out = new Float64Array(1);
    loaded.fn(3, 4, out);
    expect(out[0]).toBe(3 * 3 + 4);
    out[0] = 0;
    loaded.fn(-2, 10, out);
    expect(out[0]).toBe(-2 * -2 + 10);
  });

  itSkipWithoutCc("matches JS-JIT result on a scalar script", () => {
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
      expect(typeof res.generatedC).toBe("string");
      expect(res.variableValues["result"]).toBe(49);
    }
  );
});

// ── Phase 2: tensor-support parity with JS-JIT ───────────────────────────

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
    const r = checkCFeasibility(
      body,
      ["x", "y"],
      [tensor1dT, tensor1dT],
      tensor1dT,
      [tensor1dT],
      1
    );
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
    const r = checkCFeasibility(
      body,
      ["x"],
      [tensor1dT],
      tensor1dT,
      [tensor1dT],
      1
    );
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
    const r = checkCFeasibility(
      body,
      ["x"],
      [tensor1dT],
      numberT,
      [numberT],
      1
    );
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
    const r = checkCFeasibility(
      body,
      ["x"],
      [tensor1dT],
      tensor1dT,
      [tensor1dT],
      1
    );
    expect(r.ok).toBe(false);
  });
});

describe("C-JIT: tensor codegen (Phase 2, koffi)", () => {
  const numberT: JitType = { kind: "number" };
  const tensor1dT: JitType = {
    kind: "tensor",
    isComplex: false,
    shape: [100, 1],
  };

  it("emits numbl_real_binary_elemwise for tensor r = x + y", () => {
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
      [tensor1dT],
      "fn_add"
    );
    expect(gen.cSource).toContain('#include "numbl_ops.h"');
    expect(gen.cSource).toContain(
      "numbl_real_binary_elemwise(NUMBL_REAL_BIN_ADD"
    );
    expect(gen.usesTensors).toBe(true);
    // Signature should use raw double* for tensors
    expect(gen.cSource).toContain("const double *v_x_data");
    expect(gen.cSource).toContain("int64_t v_x_len");
    expect(gen.cSource).toContain("double *v_r_buf");
  });

  it("emits scalar_binary_elemwise for scalar-tensor ops", () => {
    // r = 0.5 .* x
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
      [tensor1dT],
      "fn_scalar_mul"
    );
    expect(gen.cSource).toContain(
      "numbl_real_scalar_binary_elemwise(NUMBL_REAL_BIN_MUL"
    );
  });

  it("emits scratch buffers for inner tensor sub-expressions", () => {
    // r = x + (x .* y)
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
      [tensor1dT],
      "fn_nested"
    );
    // Should have scratch buffer declarations
    expect(gen.cSource).toContain("double *__s1_data = NULL;");
    expect(gen.cSource).toContain("int64_t __s1_len = 0;");
    // Should malloc the scratch and call the binary op
    expect(gen.cSource).toContain("__s1_data = (double *)malloc(");
    expect(gen.cSource).toContain(
      "numbl_real_binary_elemwise(NUMBL_REAL_BIN_MUL"
    );
    // Should free scratch at end
    expect(gen.cSource).toContain("if (__s1_data) free(__s1_data);");
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
      [numberT],
      "fn_sum"
    );
    expect(gen.cSource).toContain("numbl_reduce_flat(NUMBL_REDUCE_SUM");
  });
});

describe("C-JIT: tensor parity with JS-JIT (E2E)", () => {
  const available = E2E_ENABLED && hasCc();
  const itSkipWithoutCc = available ? it : it.skip;

  const parity = (script: string, keys: string[], approx = false) => {
    const js = executeCode(script, { optimization: 1 });
    const cj = executeCode(script, { optimization: 2 });
    for (const k of keys) {
      if (approx && typeof cj.variableValues[k] === "number") {
        expect(cj.variableValues[k], `${k} mismatch`).toBeCloseTo(
          js.variableValues[k] as number,
          8
        );
      } else {
        expect(cj.variableValues[k], `${k} mismatch`).toEqual(
          js.variableValues[k]
        );
      }
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
      ["result"],
      true
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
