/**
 * Path-management builtins. mtoc2 resolves the workspace search path
 * statically, so it cannot honor a runtime `addpath`/`rmpath`/`savepath`
 * call. `addpath(<literal>, ...)` is permitted only as a leading
 * sequence of top-level statements in the driver script; the prologue
 * extractor (`src/workspace/driverPrologue.ts`) strips those calls
 * before lowering begins, so they never reach `transfer`. Any other
 * use — inside a function body, in a non-driver file, or after the
 * first non-comment statement in the driver — surfaces here at the
 * call site with a span-attributed error.
 *
 * `rmpath` and `savepath` have no static interpretation and are
 * always rejected.
 *
 * Each of the three backend hooks (`emitC` / `emitJs` / `call`) is
 * stubbed alongside `transfer` so the registry-shape invariant holds
 * (every builtin implements every hook); they all throw the same
 * `UnsupportedConstruct` if reached, but `transfer` should fire first
 * in every realistic call path.
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
  call() {
    throw new UnsupportedConstruct(message);
  },
});

export const addpath: Builtin = reject(
  "addpath",
  "'addpath' is only allowed at the top of the driver script, " +
    "before any other non-comment statement; mtoc2 resolves the " +
    "search path statically and cannot honor a runtime addpath call"
);

export const rmpath: Builtin = reject(
  "rmpath",
  "'rmpath' is not supported by mtoc2; the search path is fixed " +
    "after lowering begins"
);

export const savepath: Builtin = reject(
  "savepath",
  "'savepath' is not supported by mtoc2"
);
