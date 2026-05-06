import { describe, it, expect } from "vitest";
import { executeCode } from "../numbl-core/executeCode.js";

/**
 * Refcount/pool balance tests.
 *
 * After a script ending in `clear;`, every Float64Array buffer that
 * was acquired from the pool should have been released back to it:
 *
 *   - liveSetSize === 0           (no buffers still owned by some wrapper)
 *   - attemptedAllocs === releases (every acquire matched by a release)
 *
 * `attemptedAllocs` includes cache hits, so re-acquired buffers count
 * separately on each cycle and balance still requires releases to match.
 *
 * The vitest harness routes ALL allocations (any size) through the pool
 * (see `IN_TEST` in `executors/jsJit/helpers/alloc.ts`), so even small
 * arrays like `[1 2 3]` exercise the pool's bookkeeping.
 */

interface Probe {
  attempts: number;
  releases: number;
  live: number;
}

const probe = (src: string, opt: "0" | "1" = "0"): Probe => {
  const r = executeCode(src, { optimization: opt });
  const s = r.memoryStats!;
  return {
    attempts: s.attemptedAllocs,
    releases: s.releases,
    live: s.liveSetSize,
  };
};

const expectBalanced = (src: string, opt: "0" | "1" = "0"): void => {
  const p = probe(src, opt);
  expect(p.live, `liveSetSize should be 0 (opt=${opt}); src=${src}`).toBe(0);
  expect(p.attempts).toBe(p.releases);
  expect(
    p.attempts,
    `script should have allocated buffers (opt=${opt})`
  ).toBeGreaterThan(0);
};

// ── Balanced cases (the system works) ───────────────────────────────────

describe("refcount balance: simple cases (interpreter)", () => {
  it("alias and clear: x = [1 2 3]; y = x; clear;", () => {
    expectBalanced("x = [1 2 3]; y = x; clear;");
  });

  it("rebinding: x = [1 2 3]; x = [4 5 6 7]; clear;", () => {
    expectBalanced("x = [1 2 3]; x = [4 5 6 7]; clear;");
  });

  it("zeros + clear", () => {
    expectBalanced("x = zeros(200, 1); clear;");
  });

  it("zeros + alias + clear", () => {
    expectBalanced("x = zeros(200, 1); y = x; clear;");
  });

  it("arithmetic transient + clear", () => {
    expectBalanced("a = [1 2 3]; b = a + 1; clear;");
  });

  it("transpose transient + clear", () => {
    expectBalanced("a = [1 2 3]; b = a'; clear;");
  });

  it("element-wise multiply + clear", () => {
    expectBalanced("a = [1 2 3]; b = a .* 2; clear;");
  });

  it("expression statement (no bind) drops on its own", () => {
    expectBalanced("[1 2 3]; clear;");
  });

  it("nested expression statements", () => {
    expectBalanced("[1 2 3] + [4 5 6]; [7 8 9]; clear;");
  });

  it("self-rebind", () => {
    expectBalanced("x = [1 2 3]; x = x; clear;");
  });

  it("chained add", () => {
    expectBalanced("a = [1 2 3] + [4 5 6] + [7 8 9]; clear;");
  });
});

describe("refcount balance: indexing & mutation (interpreter)", () => {
  it("scalar index assign + clear", () => {
    expectBalanced("a = [1 2 3]; a(2) = 99; clear;");
  });

  it("range index assign + clear", () => {
    expectBalanced("a = [1 2 3 4]; a(1:2) = [9 9]; clear;");
  });

  it("growing assignment + clear", () => {
    expectBalanced("a = [1 2 3]; a(5) = 9; clear;");
  });

  it("indexed read + clear", () => {
    expectBalanced("a = [1 2 3 4 5]; b = a(2:4); clear;");
  });
});

describe("refcount balance: containers (interpreter)", () => {
  it("cell construction + clear", () => {
    expectBalanced("c = {[1 2], [3 4 5]}; clear;");
  });

  it("cell element rebind + clear", () => {
    expectBalanced("c = {[1 2 3], [4 5]}; c{1} = [9 9]; clear;");
  });

  it("cell aliased then cleared", () => {
    expectBalanced("c = {[1 2 3]}; d = c; clear;");
  });

  it("struct construction + clear", () => {
    expectBalanced("s = struct('x', [1 2 3], 'y', [4 5]); clear;");
  });

  it("struct field rebind + clear", () => {
    expectBalanced("s.x = [1 2 3]; s.x = [9 9]; clear;");
  });

  it("nested struct field rebind + clear", () => {
    expectBalanced("s.a.b = [1 2 3]; s.a.b = [9]; clear;");
  });

  it("cell of cells + clear", () => {
    expectBalanced("c = {{[1 2 3]}, {[4 5]}}; clear;");
  });

  it("struct array field assign + clear", () => {
    expectBalanced("s(1).x = [1 2]; s(2).x = [3 4]; clear;");
  });
});

describe("refcount balance: loops (interpreter)", () => {
  it("for-loop produces churn + clear", () => {
    expectBalanced("for k=1:20; x = zeros(200,1); end; clear;");
  });

  it("for-loop accumulating in same var + clear", () => {
    expectBalanced("a = zeros(100, 1); for k=1:10; a = a + 1; end; clear;");
  });

  it("for-loop building cell + clear", () => {
    expectBalanced("c = {}; for k=1:5; c{end+1} = [k k+1]; end; clear;");
  });

  it("while-loop + clear", () => {
    expectBalanced(
      "k = 0; while k < 10; x = zeros(50,1); k = k+1; end; clear;"
    );
  });

  it("nested loops + clear", () => {
    expectBalanced("for i=1:5; for j=1:3; x = [i j]; end; end; clear;");
  });
});

describe("refcount balance: functions (interpreter)", () => {
  it("user function single output + clear", () => {
    expectBalanced(
      `
      function y = f(x); y = x + 1; end
      a = [1 2 3];
      b = f(a);
      clear;
      `
    );
  });

  it("user function multi-output + clear", () => {
    expectBalanced(
      `
      function [u, v] = f(x); u = x + 1; v = x - 1; end
      a = [1 2 3];
      [p, q] = f(a);
      clear;
      `
    );
  });

  it("user function ignored output + clear", () => {
    expectBalanced(
      `
      function [u, v] = f(x); u = x + 1; v = x - 1; end
      a = [1 2 3];
      [~, q] = f(a);
      clear;
      `
    );
  });

  it("user function called as statement (nargout=0) + clear", () => {
    expectBalanced(
      `
      function y = f(x); y = x + 1; end
      a = [1 2 3];
      f(a);
      clear;
      `
    );
  });

  it("recursive function + clear", () => {
    expectBalanced(
      `
      function y = f(n)
        if n == 0
          y = [0 0 0];
        else
          y = f(n-1) + 1;
        end
      end
      a = f(5);
      clear;
      `
    );
  });
});

describe("refcount balance: control flow (interpreter)", () => {
  it("if-branch + clear", () => {
    expectBalanced("if true; x = [1 2 3]; else; x = [4 5]; end; clear;");
  });

  it("else-branch + clear", () => {
    expectBalanced("if false; x = [1 2 3]; else; x = [4 5]; end; clear;");
  });

  it("try-catch (no error) + clear", () => {
    expectBalanced("try; x = [1 2 3]; catch; x = [4 5]; end; clear;");
  });

  it("try-catch (with error) + clear", () => {
    expectBalanced(
      "try; x = [1 2 3]; error('oops'); catch; y = [9 9]; end; clear;"
    );
  });
});

describe("refcount balance: clear by name (interpreter)", () => {
  it("clear specific names releases just those", () => {
    expectBalanced("x = [1 2 3]; y = [4 5]; clear x y;");
  });

  it("clear one of two then clear", () => {
    expectBalanced("x = [1 2 3]; y = [4 5]; clear x; clear;");
  });
});

describe("refcount balance: builtins that allocate", () => {
  it("complex tensor literal", () => {
    expectBalanced("x = [1+2i, 3+4i]; clear;");
  });

  it("concat horizontal", () => {
    expectBalanced("x = [[1 2], [3 4]]; clear;");
  });

  it("concat vertical", () => {
    expectBalanced("x = [[1 2]; [3 4]]; clear;");
  });

  it("linspace", () => {
    expectBalanced("x = linspace(0, 1, 100); clear;");
  });

  it("abs", () => {
    expectBalanced("x = abs([-1 2 -3]); clear;");
  });

  it("sin", () => {
    expectBalanced("x = sin([0 1 2]); clear;");
  });

  it("sort", () => {
    expectBalanced("x = sort([3 1 2]); clear;");
  });

  it("reshape", () => {
    expectBalanced("x = reshape([1 2 3 4 5 6], 2, 3); clear;");
  });

  it("squeeze", () => {
    expectBalanced("x = squeeze(zeros(1,3,1)); clear;");
  });

  it("find", () => {
    expectBalanced("x = find([0 1 0 2]); clear;");
  });

  it("logical comparison", () => {
    expectBalanced("x = [1 2 3] > 2; clear;");
  });

  it("double from logical", () => {
    expectBalanced("x = double([1 2 3] > 2); clear;");
  });

  it("sum 2D (returns tensor)", () => {
    expectBalanced("x = sum([1 2; 3 4]); clear;");
  });

  it("sum (scalar result)", () => {
    expectBalanced("x = sum([1 2 3 4]); clear;");
  });

  it("mean (scalar result)", () => {
    expectBalanced("x = mean([1 2 3 4]); clear;");
  });
});

// ── JIT path ─────────────────────────────────────────────────────────────

describe("refcount balance: JIT (optimization=1)", () => {
  it("simple alias and clear (JIT)", () => {
    expectBalanced("x = [1 2 3]; y = x; clear;", "1");
  });

  it("for-loop in JIT + clear", () => {
    expectBalanced("for k=1:50; x = zeros(200,1); end; clear;", "1");
  });

  it("arithmetic transient (JIT) + clear", () => {
    expectBalanced("a = [1 2 3]; b = a + 1; clear;", "1");
  });

  it("indexing in JIT + clear", () => {
    expectBalanced("a = [1 2 3]; a(2) = 99; clear;", "1");
  });

  it("function call in JIT + clear", () => {
    expectBalanced(
      `
      function y = f(x); y = x + 1; end
      a = [1 2 3];
      b = f(a);
      clear;
      `,
      "1"
    );
  });
});

// ── Pool reuse: cache hits balance too ──────────────────────────────────

describe("refcount balance: pool re-use", () => {
  it("repeating same-size alloc produces cache hits, all released", () => {
    const p = probe("for k=1:30; x = zeros(200,1); end; clear;");
    expect(p.live).toBe(0);
    expect(p.attempts).toBe(p.releases);
    expect(p.attempts).toBeGreaterThan(0);
  });
});

describe("refcount balance: matmul", () => {
  it("matmul (same operand)", () => {
    expectBalanced("a = [1 2; 3 4]; b = a * a; clear;");
  });

  it("matmul (different operands)", () => {
    expectBalanced("a = [1 2; 3 4]; b = [5 6; 7 8]; c = a * b; clear;");
  });

  it("dot product (1xN * Nx1)", () => {
    expectBalanced("a = [1 2 3]; b = [4 5 6]'; c = a*b; clear;");
  });

  it("matmul (JIT)", () => {
    expectBalanced("a = [1 2; 3 4]; b = a * a; clear;", "1");
  });
});

describe("refcount balance: anonymous functions", () => {
  it("identity captures-free", () => {
    expectBalanced("f = @(x) x; b = f([4 5 6]); clear;");
  });

  it("anonymous function with captures", () => {
    expectBalanced("a = [1 2 3]; f = @(x) x + a; b = f([4 5 6]); clear;");
  });

  it("anonymous function without captures", () => {
    expectBalanced("f = @(x) x + 1; b = f([4 5 6]); clear;");
  });

  it("anonymous function called as expression statement", () => {
    expectBalanced("f = @(x) x + 1; f([4 5 6]); clear;");
  });

  it("nested anonymous functions", () => {
    expectBalanced("f = @(x) @(y) x+y; g = f([1 2 3]); h = g([4 5 6]); clear;");
  });
});

describe("refcount balance: more builtins", () => {
  it("repmat", () => {
    expectBalanced("x = repmat([1 2], 2, 3); clear;");
  });

  it("min/max (scalar result)", () => {
    expectBalanced("x = max([1 2 3]); y = min([1 2 3]); clear;");
  });

  it("unique", () => {
    expectBalanced("x = unique([1 2 2 3]); clear;");
  });
});

describe("refcount balance: JIT extras", () => {
  it("anon captures (JIT)", () => {
    expectBalanced("a = [1 2 3]; f = @(x) x + a; b = f([4 5 6]); clear;", "1");
  });

  it("sum reduction (JIT)", () => {
    expectBalanced("x = sum([1 2 3 4]); clear;", "1");
  });
});

// ── Sanity: harness itself does not allocate spuriously ────────────────

describe("refcount balance: harness sanity", () => {
  it("empty clear: nothing allocated, nothing leaked", () => {
    const r = executeCode("clear;", { optimization: "0" });
    const s = r.memoryStats!;
    expect(s.liveSetSize).toBe(0);
    expect(s.attemptedAllocs).toBe(s.releases);
  });

  it("scalar-only script does not touch the pool", () => {
    const r = executeCode("x = 1 + 2; y = x * 3; clear;", {
      optimization: "0",
    });
    const s = r.memoryStats!;
    expect(s.liveSetSize).toBe(0);
  });
});
