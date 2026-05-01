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

  it("transpose of owned binop", () => {
    // `(a+b)'` — Binary produces an owned tensor that becomes the
    // operand of the transpose. Without dispose at evalUnary, the
    // intermediate buffer leaks per call.
    expectBalanced(`a = [1 2 3]; b = [4 5 6]; c = (a + b)'; clear all;`);
  });

  it("uminus of owned binop", () => {
    expectBalanced(`a = [1 2 3]; b = [4 5 6]; c = -(a + b); clear all;`);
  });

  it("non-conjugate transpose of owned binop", () => {
    expectBalanced(
      `a = [1+2i 3+4i]; b = [5+6i 7+8i]; c = (a + b).'; clear all;`
    );
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

describe("dispose-balance: chained binop intermediates", () => {
  // Closed by `evalBinary` disposing owned operand intermediates after
  // the binop produces its fresh result. Without that, `a*2`'s buffer
  // is fed to `+ 1` and then unreferenced — one leak per stage.
  it("chain of binary ops", () => {
    expectBalanced(`a = [1 2 3]; b = a * 2 + 1 - 3; clear all;`);
  });
});

describe("dispose-balance: indexed-assignment growth", () => {
  // `x(end+1, :) = ...` and friends rebuild the underlying tensor with
  // a fresh, larger buffer (`growTensor2D`). The old buffer was the
  // unique owner via the env binding, so it should be disposed when
  // the new tensor replaces it. Without the dispose, every growth
  // step leaks one (and one imag, if complex) buffer.

  it("scalar-grown tensor (1-d, in for-loop)", () => {
    expectBalanced(`
      x = [];
      for k = 1:5
        x(k) = k;
      end
      clear all;
    `);
  });

  it("scalar-grown tensor (2-d, end+1 row, in for-loop)", () => {
    expectBalanced(`
      x = zeros(0, 3);
      for k = 1:4
        x(end+1, :) = [k k+1 k+2];
      end
      clear all;
    `);
  });

  it("scalar-grown tensor (2-d, end+1 col, in for-loop)", () => {
    expectBalanced(`
      x = zeros(2, 0);
      for k = 1:4
        x(:, end+1) = [k; k+1];
      end
      clear all;
    `);
  });

  it("scalar-grown tensor (1-d, while loop)", () => {
    expectBalanced(`
      x = [];
      k = 1;
      while k <= 5
        x(k) = k;
        k = k + 1;
      end
      clear all;
    `);
  });

  it("auto-create-from-undefined indexed assign", () => {
    // `ders(:,1) = zeros(N,1)` when `ders` doesn't exist yet — the
    // index store synthesizes a 0×0 base, then grows it. The rhs is
    // copied out into the new buffer and is no longer referenced.
    expectBalanced(`
      ders(:, 1) = zeros(5, 1);
      clear all;
    `);
  });

  it("auto-create scalar indexed assign", () => {
    expectBalanced(`
      x(3) = 7;
      clear all;
    `);
  });

  it("indexed assignment past existing extent (no loop)", () => {
    expectBalanced(`
      x = zeros(2, 2);
      x(5, 5) = 1;
      clear all;
    `);
  });

  it("indexed assign with borrowed rhs (Ident)", () => {
    expectBalanced(`
      x = zeros(3, 3);
      y = [10 20 30];
      x(1, :) = y;
      clear all;
    `);
  });

  it("indexed assign with owned rhs (Binary)", () => {
    expectBalanced(`
      x = zeros(3, 3);
      a = [1 2 3];
      x(1, :) = a + 1;
      clear all;
    `);
  });

  it("indexed assign with owned rhs (TensorLit)", () => {
    expectBalanced(`
      x = zeros(3, 3);
      x(1, :) = [10 20 30];
      clear all;
    `);
  });

  it("indexed assign growth + binop rhs", () => {
    expectBalanced(`
      x = zeros(2, 3);
      a = [1 2 3];
      x(3, :) = a * 2;
      clear all;
    `);
  });

  it("struct field assign with owned rhs is move (no clone)", () => {
    expectBalanced(`
      s.x = [1 2 3];
      a = [4 5 6];
      s.y = a + 1;
      clear all;
    `);
  });

  it("struct field assign with borrowed rhs deep-clones", () => {
    expectBalanced(`
      a = [1 2 3];
      s.x = a;
      clear all;
    `);
  });

  it("scalar wholly-contained assign does not grow (no leak baseline)", () => {
    // Sanity check: this case never triggers growTensor2D, so it should
    // already be balanced — kept here to detect regressions in the
    // mutate-in-place path if a future change makes it accidentally
    // allocate a new tensor.
    expectBalanced(`
      x = zeros(3, 3);
      x(2, 2) = 7;
      clear all;
    `);
  });

  it("complex tensor growth via x(end+1) = i", () => {
    expectBalanced(`
      x = 1+2i;
      x(2) = 3+4i;
      x(3) = 5+6i;
      clear all;
    `);
  });
});

describe("dispose-balance: multi-output assignment", () => {
  // `[a, b] = f(...)` previously deep-cloned each output via
  // Runtime.share before assigning, even though FuncCall outputs are
  // owned and could be moved into the lvalues directly. The clones
  // leaked one buffer per output.
  it("multi-output user function (Var lvalues)", () => {
    expectBalanced(`
      [a, b] = pair([1 2 3]);
      clear all;
      function [y, z] = pair(x)
        y = x + 1;
        z = x * 2;
      end
    `);
  });

  it("multi-output sort returns vector + indices", () => {
    expectBalanced(`
      [s, i] = sort([3 1 2]);
      clear all;
    `);
  });

  it("multi-output with one Ignore", () => {
    expectBalanced(`
      [~, i] = sort([3 1 2]);
      clear all;
    `);
  });
});

describe("dispose-balance: method calls and function-handles", () => {
  // MethodCall and function-handle index paths previously used the
  // untracked `evalArgs` / `evalIndicesWithEnd` and leaked their owned
  // tensor args.
  it("function handle invocation with owned arg", () => {
    expectBalanced(`
      f = @(x) x + 1;
      r = f(1:5);
      clear all;
    `);
  });

  it("max of owned binop", () => {
    expectBalanced(`
      r = max([1 2 3] + 1);
      clear all;
    `);
  });

  it("min of owned binop (real flat reduce path)", () => {
    expectBalanced(`
      r = min([5 3 4] * 2);
      clear all;
    `);
  });

  it("max with vector dim arg (multi-dim recursion)", () => {
    expectBalanced(`
      A = reshape(1:24, [2, 3, 4]);
      r = max(A, [], [1, 3]);
      clear all;
    `);
  });
});

describe("dispose-balance: indexed read with owned index", () => {
  // `x(1:N)` — the range tensor used as the index is owned, never
  // bound to a variable, and consumed by the index implementation
  // when it copies values out into the slice. Without dispose, every
  // such read leaks the index buffer.
  it("range used as slice index", () => {
    expectBalanced(`
      x = 1:10;
      y = x(2:8);
      clear all;
    `);
  });

  it("range index used in index-assign rhs", () => {
    expectBalanced(`
      x = zeros(1, 10);
      x(2:5) = 99;
      clear all;
    `);
  });

  it("range index on tensor function call", () => {
    expectBalanced(`
      a = ones(5, 5);
      r = a(2:4, 1:3);
      clear all;
    `);
  });

  it("cell with range index", () => {
    expectBalanced(`
      c = {10, 20, 30, 40, 50};
      v = c(2:4);
      clear all;
    `);
  });
});

describe("dispose-balance: captured-name granularity", () => {
  // §6: a closure snapshot freezes the env's wrappers as of snapshot
  // time. Bindings created AFTER the snapshot are not in the snapshot
  // so disposing them on overwrite is safe.
  it("Var overwrite of post-snapshot binding (with closure)", () => {
    expectBalanced(`
      f = @(x) x + 1;
      a = [1 2 3];
      a = [4 5 6];
      clear all;
    `);
  });

  it("struct field overwrite of post-snapshot binding (with closure)", () => {
    expectBalanced(`
      f = @(x) x + 1;
      s.x = [1 2 3];
      s.x = [4 5 6];
      clear all;
    `);
  });

  it("indexed-assign growth of post-snapshot binding (with closure)", () => {
    expectBalanced(`
      f = @(x) x + 1;
      a = zeros(2, 2);
      a(3, 3) = 9;
      clear all;
    `);
  });

  it("for-loop iteration value still disposed (with closure)", () => {
    // iterVal is never bound to env.vars, so capturedNames doesn't
    // matter — the dispose should run regardless of envCaptured.
    expectBalanced(`
      f = @(x) x + 1;
      total = 0;
      for k = 1:10
        total = total + k;
      end
      clear all;
    `);
  });
});

describe("dispose-balance: cell entry overwrite", () => {
  // `c{k} = newval` rebuilds the cell with a new entry at k; the OLD
  // entry's wrapper is no longer referenced from the new cell and can
  // be recycled.
  it("scalar cell entry overwrite", () => {
    expectBalanced(`
      c = {[1 2 3], [4 5 6]};
      c{1} = [7 8 9];
      clear all;
    `);
  });

  it("loop overwriting cell entries", () => {
    expectBalanced(`
      c = {zeros(1, 3), zeros(1, 3), zeros(1, 3)};
      for k = 1:3
        c{k} = [k k+1 k+2];
      end
      clear all;
    `);
  });

  it("cell entry overwrite with binop rhs", () => {
    expectBalanced(`
      c = {[1 2], [3 4]};
      a = [10 20];
      c{1} = a + 1;
      clear all;
    `);
  });

  it("multi-output cell assign overwrites entries", () => {
    // `[out{1:N}] = func()` — every replaced entry leaks unless the
    // OLD entries are disposed at the seam. Mirrors the pattern at
    // chunkerfunc.m:216 inside its for-loop.
    expectBalanced(`
      out = {zeros(1,3), zeros(1,3)};
      [out{1:2}] = pair([10 20 30]);
      clear all;
      function [a, b] = pair(x)
        a = x;
        b = x * 2;
      end
    `);
  });

  it("multi-output cell assign in a loop", () => {
    expectBalanced(`
      out = {[], [], []};
      for k = 1:3
        [out{1:3}] = triple([k k+1]);
      end
      clear all;
      function [a, b, c] = triple(x)
        a = x;
        b = x + 1;
        c = x + 2;
      end
    `);
  });
});

describe("dispose-balance: tensor literal element disposal", () => {
  // Inside `[...]` literal: each element coming from an owned expr
  // (Range / Binary / FuncCall …) gets its values copied out by
  // horzcat/vertcat. Without disposal, every per-row element leaks one
  // buffer.
  it("row of binops", () => {
    expectBalanced(`
      a = [1 2 3];
      b = [4 5 6];
      c = [a+b, a-b, a*2];
      clear all;
    `);
  });

  it("multi-row of FuncCall results", () => {
    expectBalanced(`
      x = linspace(0, 1, 5);
      M = [cos(x); sin(x)];
      clear all;
    `);
  });

  it("vector of ranges", () => {
    expectBalanced(`
      M = [1:5; 6:10; 11:15];
      clear all;
    `);
  });
});

// ── Known gaps ────────────────────────────────────────────────────────
//
// `it.fails` passes while the test still fails; once a gap closes the
// test will report as unexpectedly-passing, prompting promotion to
// `it`. See ownership-and-dispose.md §8 for the gap list.

describe("dispose-balance: known gaps", () => {
  // (closed) FuncCall arg passing: the caller's owned arg is now
  // disposed after the call. Built into evalFuncCall via
  // evalArgsTracked. Pass-through builtins (uplus, deal, struct,
  // squeeze) clone defensively, and assignin clones before storing.
  it("user function with single output", () => {
    expectBalanced(`
      a = helper([1 2 3]);
      clear all;
      function y = helper(x)
        y = x + 1;
      end
    `);
  });

  it("user function with internal local", () => {
    expectBalanced(`
      a = helper([1 2 3]);
      clear all;
      function y = helper(x)
        local = [10 20 30];
        y = x + local;
      end
    `);
  });

  it("user function called twice", () => {
    expectBalanced(`
      a = helper([1 2 3]);
      a = helper(a);
      clear all;
      function y = helper(x)
        y = x + 1;
      end
    `);
  });

  it("nested function calls", () => {
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

  it("squeeze of FuncCall arg", () => {
    // Regression: squeeze must not share its data buffer with the
    // input — the caller may dispose the input after the call (per the
    // owned-FuncCall-args dispose pass).
    expectBalanced(`
      A = reshape(1:24, [2, 3, 4]);
      r = squeeze(max(A, [], [1, 3]));
      clear all;
    `);
  });

  it("assignin clones the stored value", () => {
    // Regression: assignin must clone — the caller may dispose the
    // arg after the call, which would NaN-poison the workspace var.
    expectBalanced(`
      ws_set();
      assert(isequal(x, [2, 5]));
      clear all;
      function ws_set()
        assignin('caller', 'x', [2, 5]);
      end
    `);
  });

  // (closed) for-loop iteration value is now disposed at loop exit.
  it("for loop accumulator", () => {
    expectBalanced(`
      total = zeros(1, 5);
      for k = 1:3
        total = total + 1;
      end
      clear all;
    `);
  });

  it("for loop iterating columns of a matrix", () => {
    // Each iteration's column is freshly allocated by forIter; without
    // disposing the previous binding, every column except the last
    // leaks one buffer.
    expectBalanced(`
      M = [1 2 3; 4 5 6; 7 8 9];
      total = zeros(3, 1);
      for c = M
        total = total + c;
      end
      clear all;
    `);
  });

  it("for loop with break", () => {
    expectBalanced(`
      acc = 0;
      for k = 1:100
        acc = acc + k;
        if k >= 5
          break;
        end
      end
      clear all;
    `);
  });

  // GAP: same as nested-function — owned FuncCall arg (`sum(x)` arg)
  // is not disposed at the call site.
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
