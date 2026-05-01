/**
 * Allocator balance tests — verifies that programs which end with
 * `clear all` (and don't intentionally hold state via globals,
 * persistents, or closures) recycle every dense float buffer they
 * allocated. Concretely: `getAllocStats().allocCount ===
 * getAllocStats().disposeCount` at end of execution.
 *
 * NaN-poison and double-dispose detection in `runtime/alloc.ts` keep
 * any incorrect dispose loud — these tests assert the *positive*
 * direction (full accounting), and the existing safety nets enforce
 * the *negative* direction (no use-after-dispose).
 *
 * See `docs/developer_reference/runtime/ownership-and-dispose.md` for
 * the ownership rules being verified.
 */

import { describe, it, beforeEach } from "vitest";
import { executeCode } from "../numbl-core/executeCode.js";

function balanceFor(src: string): {
  allocs: number;
  disposes: number;
  diff: number;
} {
  // Pin to interpreter (--opt 0) for now; JIT-compiled code paths
  // don't yet implement the function-exit dispose pass.
  const result = executeCode(src, { optimization: "0" });
  const s = result.allocStats!;
  return {
    allocs: s.allocCount,
    disposes: s.disposeCount,
    diff: s.allocCount - s.disposeCount,
  };
}

function expectBalanced(src: string) {
  const r = balanceFor(src);
  if (r.diff !== 0) {
    throw new Error(
      `dispose imbalance: ${r.allocs} allocs vs ${r.disposes} disposes (diff ${r.diff})\n--- script ---\n${src}`
    );
  }
}

describe("dispose-balance: simple bindings", () => {
  beforeEach(() => {
    // Each executeCode resets via its own delta tracking, but the
    // module-level pool state may carry over and affect hit rates.
    // Balance assertions don't depend on pool state, so no reset needed.
  });

  it("vector literal then clear", () => {
    expectBalanced(`a = [1 2 3]; clear all;`);
  });

  it("scalar literal then clear", () => {
    expectBalanced(`a = 5; clear all;`);
  });

  it("nested clear by name", () => {
    expectBalanced(`a = [1 2 3]; b = [4 5 6]; clear a; clear b;`);
  });

  it("rebound variable", () => {
    expectBalanced(`a = [1 2 3]; a = [4 5 6]; clear a;`);
  });

  it("ones / zeros constructors", () => {
    expectBalanced(`a = zeros(5); b = ones(3); clear all;`);
  });

  it("range literal", () => {
    expectBalanced(`r = 1:10; clear all;`);
  });
});

describe("dispose-balance: arithmetic", () => {
  it("element-wise binary op", () => {
    expectBalanced(`a = [1 2 3]; b = [4 5 6]; c = a + b; clear all;`);
  });

  it("self-rebinding via binary op", () => {
    expectBalanced(`a = [1 2 3]; a = a + 1; clear all;`);
  });

  it("unary op", () => {
    expectBalanced(`a = [1 2 3]; b = -a; clear all;`);
  });
});

describe("dispose-balance: control flow", () => {
  it("while loop with rebinding", () => {
    expectBalanced(`
      a = [1 2 3];
      i = 0;
      while i < 3
        a = a + 1;
        i = i + 1;
      end
      clear all;
    `);
  });
});

// ── Known gaps ────────────────────────────────────────────────────────
//
// These tests describe the desired behavior but don't balance yet.
// `it.fails` passes while the test still fails; once a gap closes the
// test will report as unexpectedly-passing, prompting promotion to
// `it`. See ownership-and-dispose.md §8 for the gap list.

describe("dispose-balance: known gaps", () => {
  // GAP: chained binary ops leave intermediate buffers unrecycled
  // (`a*2`'s result feeds the `+ 1` operand and is then unreferenced —
  // the binop site doesn't dispose the LHS owned operand after consumption).
  it.fails("chain of binary ops", () => {
    expectBalanced(`a = [1 2 3]; b = a * 2 + 1 - 3; clear all;`);
  });

  // GAP: FuncCall arg passing — the caller passes an owned value
  // (TensorLit, FuncCall result, etc.) to a function. callUserFunction
  // deep-clones at entry, so the original goes unreferenced after the
  // call. We could dispose the original at the call site, but some
  // builtins (e.g. `uplus`) return their input verbatim and would
  // double-dispose. Audit / fix pass-through builtins first.
  it.fails("user function with single output", () => {
    expectBalanced(`
      a = helper([1 2 3]);
      clear all;
      function y = helper(x)
        y = x + 1;
      end
    `);
  });

  it.fails("user function with internal local", () => {
    expectBalanced(`
      a = helper([1 2 3]);
      clear all;
      function y = helper(x)
        local = [10 20 30];
        y = x + local;
      end
    `);
  });

  it.fails("user function called twice", () => {
    expectBalanced(`
      a = helper([1 2 3]);
      a = helper(a);
      clear all;
      function y = helper(x)
        y = x + 1;
      end
    `);
  });

  // GAP: nested user-function calls leak the inner call's return value
  // when the caller passes it as an arg to another user function (the
  // caller cannot safely auto-dispose owned FuncCall args because some
  // builtins like `uplus` return their input verbatim — see the
  // commit history's reverted attempt). Audit pass-through builtins.
  it.fails("nested function calls", () => {
    expectBalanced(`
      a = outer([1 2 3]);
      clear all;
      function y = outer(x)
        y = inner(x) + 1;
      end
      function y = inner(x)
        y = x * 2;
      end
    `);
  });

  // GAP: `for` loop iteration with rebinding leaves one buffer per
  // iteration unrecycled — `total = total + 1` allocates fresh, but the
  // previous iteration's buffer is the rhs operand and not yet disposed.
  it.fails("for loop accumulator", () => {
    expectBalanced(`
      total = zeros(1, 5);
      for k = 1:3
        total = total + 1;
      end
      clear all;
    `);
  });

  // GAP: same as nested-function — owned FuncCall arg (`x * 2`) inside
  // the chosen branch leaks since the binop site doesn't dispose its
  // owned operand after consumption.
  it.fails("if/else branches", () => {
    expectBalanced(`
      x = [1 2 3];
      if sum(x) > 0
        y = x * 2;
      else
        y = x * 3;
      end
      clear all;
    `);
  });
});
