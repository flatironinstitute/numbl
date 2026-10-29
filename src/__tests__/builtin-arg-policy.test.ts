import { describe, it, expect } from "vitest";
import { executeCode } from "../numbl-core/executeCode.js";
import {
  builtin,
  classMethod,
  makeBuiltin,
} from "../numbl-core/interpreter/builtins/builtin.js";
import { RTV } from "../numbl-core/runtime/constructors.js";
import { RuntimeError } from "../numbl-core/runtime/error.js";
import type { RuntimeValue } from "../numbl-core/runtime/types.js";

// Each probe returns the kind of every argument it received.
const kinds = (args: RuntimeValue[]) =>
  RTV.char(
    args.map(a => (typeof a === "object" ? a.kind : typeof a)).join(",")
  );

builtin({ name: "argPolicyDefault", apply: args => kinds(args) });
builtin({
  name: "argPolicyNative",
  accepts: { sparse: true },
  apply: args => kinds(args),
});
builtin({
  name: "argPolicyError",
  nargin: [1, 2],
  accepts: { sparse: { identifier: "probe:sparse", message: "No sparse." } },
  apply: args => kinds(args),
});

// A TypeScript-implemented class with a method named like a builtin.
const CLS = "ArgPolicyProbeClass";
builtin({
  name: "argPolicyMake",
  nargin: 0,
  apply: () => RTV.classInstance(CLS, []),
});
classMethod(CLS, {
  name: "argPolicyDefault",
  accepts: { sparse: true },
  apply: args => RTV.char("method:" + kinds(args).value),
});

const run = (code: string) =>
  executeCode(code, { optimization: "0" }, []).variableValues;
const str = (v: unknown) => (v as { value: string }).value;

const S = "S = sparse([1 0; 0 2]);";

describe("builtin argument-kind policy", () => {
  it("densifies a sparse argument for a builtin that does not declare it", () => {
    const v = run(`${S} a = argPolicyDefault(S, 3, 'x');`);
    expect(str(v["a"])).toBe("tensor,number,char");
  });

  it("passes sparse through to a builtin that accepts it", () => {
    const v = run(`${S} a = argPolicyNative(S, 3);`);
    expect(str(v["a"])).toBe("sparse_matrix,number");
  });

  it("raises the declared error, after the arity check", () => {
    const v = run(
      [
        S,
        "try, argPolicyError(S); catch e, id = e.identifier; msg = e.message; end",
        "try, argPolicyError(S, 1, 2); catch e, id2 = e.identifier; end",
      ].join("\n")
    );
    expect(str(v["id"])).toBe("probe:sparse");
    expect(str(v["msg"])).toBe("No sparse.");
    expect(str(v["id2"])).toBe("MATLAB:maxrhs");
  });

  it("applies on every call path (handle, feval, builtin)", () => {
    const v = run(
      [
        S,
        "f = @argPolicyDefault;",
        "a = f(S);",
        "b = feval('argPolicyDefault', S);",
        "c = builtin('argPolicyDefault', S);",
        "d = cellfun(@argPolicyDefault, {S}, 'UniformOutput', false);",
      ].join("\n")
    );
    expect(str(v["a"])).toBe("tensor");
    expect(str(v["b"])).toBe("tensor");
    expect(str(v["c"])).toBe("tensor");
    expect(str((v["d"] as { data: RuntimeValue[] }).data[0])).toBe("tensor");
  });

  it("lets a registered class method win before the policy applies", () => {
    const v = run(
      `${S} p = argPolicyMake(); a = argPolicyDefault(p, S); b = argPolicyDefault(S);`
    );
    // The method takes sparse itself; the builtin densifies.
    expect(str(v["a"])).toBe("method:class_instance,sparse_matrix");
    expect(str(v["b"])).toBe("tensor");
  });

  it("does not copy the argument list when nothing is converted", () => {
    let seen: RuntimeValue[] | undefined;
    const ib = makeBuiltin({
      name: "argPolicyNoCopy",
      apply: args => {
        seen = args;
        return 0;
      },
    });
    const args: RuntimeValue[] = [1, RTV.tensor(new Float64Array(2), [1, 2])];
    ib.call(args, 1);
    expect(seen).toBe(args);
  });

  it('raises a generic error for `sparse: "error"`', () => {
    const ib = makeBuiltin({
      name: "argPolicyErrDirect",
      accepts: { sparse: "error" },
      apply: () => 0,
    });
    const sp = RTV.sparseMatrix(
      1,
      1,
      new Int32Array([0]),
      new Int32Array([0, 1]),
      new Float64Array([1])
    );
    expect(() => ib.call([sp], 1)).toThrow(RuntimeError);
    try {
      ib.call([sp], 1);
    } catch (e) {
      expect((e as RuntimeError).identifier).toBe(
        "MATLAB:argPolicyErrDirect:sparseNotSupported"
      );
    }
  });
});
