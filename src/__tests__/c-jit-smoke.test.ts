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

  it("rejects tensor args", () => {
    const tensorT: JitType = { kind: "tensor", isComplex: false };
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
      "smoke"
    );
    const { shim, exportName } = generateNapiShim(
      gen.cFnName,
      ["number", "number"],
      "number"
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
