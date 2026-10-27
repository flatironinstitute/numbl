/**
 * Path-management builtins. The JIT resolves the workspace search path
 * statically, so it cannot honor `addpath` / `rmpath` / `savepath`;
 * every hook throws `UnsupportedConstruct` so the scope runs on the
 * interpreter.
 */
import { UnsupportedConstruct } from "../../../lowering/errors.js";
import type { Builtin } from "../../registry.js";

const reject = (name: string, message: string): Builtin => ({
  name,
  transfer() {
    throw new UnsupportedConstruct(message);
  },
  emitC() {
    throw new UnsupportedConstruct(message);
  },
  emitJs() {
    throw new UnsupportedConstruct(message);
  },
});

export const addpath: Builtin = reject(
  "addpath",
  "'addpath' is not supported by the JIT; it resolves the " +
    "search path statically and cannot honor a runtime addpath call"
);

export const rmpath: Builtin = reject(
  "rmpath",
  "'rmpath' is not supported by the JIT; the search path is fixed " +
    "after lowering begins"
);

export const savepath: Builtin = reject(
  "savepath",
  "'savepath' is not supported by the JIT"
);
