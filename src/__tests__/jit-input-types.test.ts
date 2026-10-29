import { describe, it, expect } from "vitest";
import {
  typeKey,
  typeOfValue,
  recordWidening,
  widenAgainst,
  type WideningHistory,
} from "../numbl-core/executors/inputTypes.js";
import { RTV } from "../numbl-core/runtime/index.js";
import { RuntimeStruct } from "../numbl-core/runtime/types.js";
import type { NumericType, Type } from "../numbl-core/jit/index.js";

const vec = (n: number) => RTV.tensor(new Float64Array(n), [1, n]);

/** Widen `values` against the types of `prev` (one input each). */
function widen(value: unknown, prevValue: unknown): Type {
  const types = [typeOfValue(value)];
  widenAgainst(types, ["x"], new Map([["x", [typeOfValue(prevValue)]]]));
  return types[0];
}

describe("executor input types", () => {
  it("types scalars by sign, not value", () => {
    expect(typeOfValue(3)).toBe(typeOfValue(7));
    const t = typeOfValue(3) as NumericType;
    expect(t.sign).toBe("positive");
    expect(t.exact).toBeUndefined();
    expect((typeOfValue(0) as NumericType).sign).toBe("nonneg");
    expect(typeOfValue(-1)).not.toBe(typeOfValue(1));
  });

  it("interns types and freezes them", () => {
    const t = typeOfValue(vec(3));
    expect(typeOfValue(vec(3))).toBe(t);
    expect(Object.isFrozen(t)).toBe(true);
    expect(typeKey(t)).toBe(typeKey(typeOfValue(vec(3))));
  });

  it("maps values no backend takes to Unknown", () => {
    expect(typeOfValue(RTV.cell([1], [1, 1])).kind).toBe("Unknown");
    const s = new RuntimeStruct(new Map([["c", RTV.cell([1], [1, 1])]]));
    expect(typeOfValue(s).kind).toBe("Unknown");
  });

  it("widens scalars through the sign lattice", () => {
    expect((widen(2, 0) as NumericType).sign).toBe("nonneg");
    expect((widen(-2, 2) as NumericType).sign).toBe("unknown");
    // Widening is stable: the widened type is reused on the next call.
    const w = widen(-2, 2);
    const again = [typeOfValue(5)];
    widenAgainst(again, ["x"], new Map([["x", [w]]]));
    expect(again[0]).toBe(w);
  });

  it("widens logical to double and real to complex", () => {
    const t = widen(true, 2) as NumericType;
    expect(t.elem).toBe("double");
    const c = widen(2, RTV.complex(1, 1)) as NumericType;
    expect(c.isComplex).toBe(true);
  });

  it("widens tensor dims but not scalar against tensor", () => {
    const t = widen(vec(3), vec(4)) as NumericType;
    expect(t.dims.map(d => d.kind)).toEqual(["exact", "unknown"]);
    expect(widen(2, vec(4))).toBe(typeOfValue(2));
    expect(widen(vec(4), 2)).toBe(typeOfValue(vec(4)));
  });

  it("converges when an input alternates between scalar and tensor", () => {
    // A `find` result: 1x1 on some calls, kx1 (varying k) on others.
    const col = (n: number) => RTV.tensor(new Float64Array(n), [n, 1]);
    let hist: WideningHistory | undefined;
    const keys = new Set<string>();
    const vals = [col(2), 3, col(5), 4, col(7), col(0), 2, col(9), col(3)];
    for (const v of vals) {
      const types = [typeOfValue(v)];
      widenAgainst(types, ["j"], hist);
      hist = recordWidening(hist, ["j"], types);
      keys.add(typeKey(types[0]));
    }
    // 2x1, 1x1, then ?x1 for every later column.
    expect(keys.size).toBe(3);
    // The steady state does not rebuild the history.
    const types = [typeOfValue(col(4))];
    widenAgainst(types, ["j"], hist);
    expect(recordWidening(hist, ["j"], types)).toBe(hist);
  });

  it("keeps an input's history when the input set changes", () => {
    let hist = recordWidening(undefined, ["a"], [typeOfValue(vec(3))]);
    hist = recordWidening(
      hist,
      ["b", "a"],
      [typeOfValue(1), typeOfValue(vec(4))]
    );
    const types = [typeOfValue(vec(5))];
    widenAgainst(types, ["a"], hist);
    expect((types[0] as NumericType).dims.map(d => d.kind)).toEqual([
      "exact",
      "unknown",
    ]);
  });

  it("widens structs field by field only with the same fields", () => {
    const s = (a: unknown, b?: unknown) =>
      new RuntimeStruct(
        new Map(
          b === undefined
            ? [["a", a as never]]
            : [
                ["a", a as never],
                ["b", b as never],
              ]
        )
      );
    const w = widen(s(-1), s(1));
    expect(w.kind).toBe("Struct");
    expect(typeKey(w)).not.toBe(typeKey(typeOfValue(s(1))));
    expect(widen(s(1, 2), s(1))).toBe(typeOfValue(s(1, 2)));
  });
});
