import { describe, it, expect } from "vitest";
import { executeCode } from "../numbl-core/executeCode.js";
import {
  builtin,
  makeBuiltin,
  getIBuiltinNargin,
} from "../numbl-core/interpreter/builtins/index.js";
import {
  checkNargin,
  checkNargout,
  numericArg,
  realNumericArg,
  realScalarArg,
} from "../numbl-core/interpreter/builtins/args.js";
import { unaryElemwise } from "../numbl-core/interpreter/builtins/types.js";
import { registerExtraBuiltinNames } from "../numbl-core/helpers/registry.js";
import { RuntimeError } from "../numbl-core/runtime/error.js";
import { RTV } from "../numbl-core/runtime/constructors.js";
import {
  RuntimeComplexNumber,
  type RuntimeValue,
} from "../numbl-core/runtime/types.js";

/** Run fn and return the RuntimeError it throws. */
function caught(fn: () => unknown): RuntimeError {
  try {
    fn();
  } catch (e) {
    expect(e).toBeInstanceOf(RuntimeError);
    return e as RuntimeError;
  }
  throw new Error("expected an error");
}

describe("builtin() metadata", () => {
  it("normalizes nargin and nargout", () => {
    const ib = makeBuiltin({ name: "t1", nargin: 2, apply: () => 0 });
    expect(ib.direct!.nargin).toEqual([2, 2]);
    expect(ib.direct!.nargout).toEqual([0, 1]);
    expect(ib.direct!.reportedNargin).toBe(2);

    const ib2 = makeBuiltin({
      name: "t2",
      nargin: [1, 3],
      nargout: 2,
      apply: () => 0,
    });
    expect(ib2.direct!.nargin).toEqual([1, 3]);
    expect(ib2.direct!.nargout).toEqual([0, 2]);
    expect(ib2.direct!.reportedNargin).toBe(3);
  });

  it("reports -1 for an unbounded nargin, as MATLAB does for zeros", () => {
    const ib = makeBuiltin({ name: "t3", apply: () => 0 });
    expect(ib.direct!.nargin).toEqual([0, Infinity]);
    expect(ib.direct!.reportedNargin).toBe(-1);
    const ib2 = makeBuiltin({
      name: "t4",
      nargin: [1, Infinity],
      apply: () => 0,
    });
    expect(ib2.direct!.reportedNargin).toBe(-1);
  });

  it("rejects invalid ranges", () => {
    expect(() =>
      makeBuiltin({ name: "bad", nargin: [2, 1], apply: () => 0 })
    ).toThrow(/invalid nargin/);
    expect(() =>
      makeBuiltin({ name: "bad", nargout: -1, apply: () => 0 })
    ).toThrow(/invalid nargout/);
  });

  it("exposes a resolve that accepts any argument types", () => {
    const ib = makeBuiltin({ name: "t5", nargin: 1, apply: a => a[0] });
    const r = ib.resolve([{ kind: "cell" }], 1);
    expect(r).not.toBeNull();
    expect(r!.outputTypes.length).toBe(1);
    expect(r!.apply([7], 1)).toBe(7);
  });
});

describe("builtin() arity checks", () => {
  const ib = makeBuiltin({
    name: "t6",
    nargin: [1, 2],
    nargout: 2,
    apply: (args, nargout) =>
      nargout > 1 ? [args[0], args[0]] : (args[0] as RuntimeValue),
  });
  const call = ib.direct!.call;

  it("passes valid calls through", () => {
    expect(call([3], 1)).toBe(3);
    expect(call([3, 4], 0)).toBe(3);
    expect(call([3], 2)).toEqual([3, 3]);
  });

  it("raises MATLAB's messages and identifiers", () => {
    let e = caught(() => call([], 1));
    expect(e.message).toBe("Not enough input arguments.");
    expect(e.identifier).toBe("MATLAB:minrhs");
    e = caught(() => call([1, 2, 3], 1));
    expect(e.message).toBe("Too many input arguments.");
    expect(e.identifier).toBe("MATLAB:maxrhs");
    e = caught(() => call([1], 3));
    expect(e.message).toBe("Too many output arguments.");
    expect(e.identifier).toBe("MATLAB:maxlhs");
  });

  it("uses M-file identifiers when mfile is set", () => {
    const mf = makeBuiltin({
      name: "t6m",
      nargin: 1,
      mfile: true,
      apply: a => a[0],
    });
    let e = caught(() => mf.direct!.call([1, 2], 1));
    expect(e.message).toBe("Too many input arguments.");
    expect(e.identifier).toBe("MATLAB:TooManyInputs");
    e = caught(() => mf.direct!.call([1], 2));
    expect(e.identifier).toBe("MATLAB:TooManyOutputs");
    e = caught(() => mf.direct!.call([], 1));
    expect(e.identifier).toBe("MATLAB:minrhs");
  });

  it("checks a minimum output count", () => {
    const e = caught(() => checkNargout(0, 1, 2));
    expect(e.message).toBe("Not enough output arguments.");
    expect(e.identifier).toBe("MATLAB:minlhs");
    expect(() => checkNargin(2, 0, Infinity)).not.toThrow();
  });
});

describe("void builtins", () => {
  let ran = 0;
  const ib = makeBuiltin({
    name: "t7",
    nargout: 0,
    apply: () => {
      ran++;
      return 5;
    },
  });

  it("discard the result and reject requested outputs", () => {
    expect(ib.direct!.call([], 0)).toBeUndefined();
    expect(ran).toBe(1);
    const e = caught(() => ib.direct!.call([], 1));
    expect(e.identifier).toBe("MATLAB:maxlhs");
    expect(ran).toBe(1);
  });

  it("report no output types through resolve", () => {
    expect(ib.resolve([], 0)!.outputTypes).toEqual([]);
  });
});

describe("builtin context", () => {
  it("gives the name, and the runtime only during execution", () => {
    let seenName = "";
    const ib = makeBuiltin({
      name: "t8",
      apply: (_a, _n, ctx) => {
        seenName = ctx.name;
        return ctx.rt ? 1 : 0;
      },
    });
    const e = caught(() => ib.direct!.call([], 1));
    expect(e.message).toBe("no active runtime");
    expect(seenName).toBe("t8");
  });
});

describe("registered builtins end to end", () => {
  builtin({
    name: "zz_api_echo",
    nargin: [1, 2],
    apply: args => args[args.length - 1],
  });
  builtin({
    name: "zz_api_say",
    nargin: 1,
    nargout: 0,
    apply: (args, _n, ctx) => {
      ctx.rt.output(`said ${String(args[0])}\n`);
    },
  });
  registerExtraBuiltinNames(["zz_api_echo", "zz_api_say"]);

  const errorOf = (stmt: string) => {
    const res = executeCode(
      `try\n${stmt};\nid = 'none'; msg = '';\ncatch e\nid = e.identifier; msg = e.message;\nend`,
      { optimization: "0" }
    );
    const v = res.variableValues;
    const text = (x: RuntimeValue) =>
      typeof x === "string" ? x : (x as { value: string }).value;
    return [text(v["id"]), text(v["msg"])];
  };

  it("checks arity at the call site", () => {
    expect(errorOf("zz_api_echo()")).toEqual([
      "MATLAB:minrhs",
      "Not enough input arguments.",
    ]);
    expect(errorOf("zz_api_echo(1,2,3)")).toEqual([
      "MATLAB:maxrhs",
      "Too many input arguments.",
    ]);
    expect(errorOf("[a,b] = zz_api_echo(1)")).toEqual([
      "MATLAB:maxlhs",
      "Too many output arguments.",
    ]);
    expect(errorOf("x = zz_api_say(1)")).toEqual([
      "MATLAB:maxlhs",
      "Too many output arguments.",
    ]);
    expect(errorOf("zz_api_echo(1, 2)")).toEqual(["none", ""]);
  });

  it("runs void builtins with the runtime and leaves ans unset", () => {
    const res = executeCode("zz_api_say(3);", { optimization: "0" });
    expect(res.output.join("")).toContain("said 3");
    expect(res.variableValues["ans"]).toBeUndefined();
  });

  it("reports nargin(@f) from the metadata", () => {
    expect(getIBuiltinNargin("zz_api_echo")).toBe(2);
    expect(getIBuiltinNargin("zz_api_say")).toBe(1);
    expect(getIBuiltinNargin("hash")).toBe(2);
    expect(errorOf("hash('md5')")).toEqual([
      "MATLAB:minrhs",
      "Not enough input arguments.",
    ]);
    const res = executeCode(
      "a = nargin(@zz_api_echo); b = nargin(@isnan); c = nargin(@and);",
      { optimization: "0" }
    );
    expect(res.variableValues["a"]).toBe(2);
    expect(res.variableValues["b"]).toBe(1);
    expect(res.variableValues["c"]).toBe(2);
  });
});

describe("argument helpers", () => {
  it("numericArg converts logical and char, rejects other types", () => {
    expect(numericArg(true)).toBe(1);
    expect(numericArg(RTV.char("a"))).toBe(97);
    const t = numericArg(RTV.char("ab"));
    expect(Array.from((t as { data: Float64Array }).data)).toEqual([97, 98]);
    const e = caught(() => numericArg(RTV.cell([], [0, 0])));
    expect(e.identifier).toBe("MATLAB:math:mustBeNumericCharOrLogical");
  });

  it("realNumericArg rejects complex input but accepts a zero imaginary part", () => {
    const e = caught(() => realNumericArg(new RuntimeComplexNumber(1, 2)));
    expect(e.identifier).toBe("MATLAB:math:mustBeReal");
    expect(realNumericArg(new RuntimeComplexNumber(4, 0))).toBe(4);
    const z = RTV.tensor(
      new Float64Array([1, 2]),
      [1, 2],
      new Float64Array([0, 0])
    );
    expect((realNumericArg(z) as { imag?: unknown }).imag).toBeUndefined();
    const c = RTV.tensor(
      new Float64Array([1, 2]),
      [1, 2],
      new Float64Array([0, 1])
    );
    expect(() => realNumericArg(c)).toThrow("Argument must be real.");
  });

  it("realScalarArg errors instead of dropping data", () => {
    expect(realScalarArg(false)).toBe(0);
    expect(realScalarArg(RTV.tensor(new Float64Array([5]), [1, 1]))).toBe(5);
    expect(() =>
      realScalarArg(RTV.tensor(new Float64Array([1, 2]), [1, 2]), "N")
    ).toThrow("N must be a real scalar.");
    const cplx1x1 = RTV.tensor(
      new Float64Array([1]),
      [1, 1],
      new Float64Array([2])
    );
    expect(() => realScalarArg(cplx1x1)).toThrow("must be a real scalar");
    expect(() => realScalarArg(RTV.string("x"))).toThrow(
      "must be a real scalar"
    );
  });

  it("unaryElemwise handles numeric, logical, and char input", () => {
    const sq = unaryElemwise(
      { realFn: x => x * x, complexFn: (re, im) => ({ re, im }) },
      "sq"
    );
    expect(sq([3])).toBe(9);
    expect(sq([true])).toBe(1);
    expect(sq([RTV.char("a")])).toBe(97 * 97);
    expect(() => sq([RTV.cell([], [0, 0])])).toThrow(
      "Invalid data type. Argument must be numeric, char, or logical."
    );
    const sqrtLike = unaryElemwise(
      {
        realFn: Math.sqrt,
        complexFn: (re, im) =>
          im === 0 && re < 0 ? { re: 0, im: Math.sqrt(-re) } : { re, im },
        maybeComplex: true,
      },
      "sqrtLike"
    );
    expect(sqrtLike([4])).toBe(2);
    expect(sqrtLike([-4])).toEqual(new RuntimeComplexNumber(0, 2));
  });
});
