import { describe, it, expect } from "vitest";
import { executeCode } from "../numbl-core/executeCode.js";

// A JIT'd call-shape unit: disp inside a user function, called twice. The
// unsuppressed assignments make the whole-script executor decline, so each
// call dispatches to the call executor and the second reuses the compiled
// spec.
const CODE = `
a = f(1)
b = f(2)
function r = f(x)
  disp(x * 100);
  r = x;
end
`;

describe("JIT host isolation", () => {
  it("compiled code writes to its own runtime when two runs interleave", () => {
    let inner: string[] | null = null;
    // Run B from inside run A's output callback, after A has compiled f but
    // before A's second call reuses the compiled spec.
    const a = executeCode(CODE, {
      optimization: "1",
      onOutput: () => {
        if (inner === null) {
          inner = executeCode(CODE, { optimization: "1" }).output;
        }
      },
    });
    const lines = (out: string[]) =>
      out
        .join("")
        .split("\n")
        .filter(l => /^\d+$/.test(l.trim()));
    expect(lines(a.output)).toEqual(["100", "200"]);
    expect(lines(inner!)).toEqual(["100", "200"]);
  });
});
