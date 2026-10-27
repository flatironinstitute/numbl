import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import {
  scalarDouble,
  signFromNumber,
  isNumeric,
  isScalar,
  isCell,
  type Type,
} from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";

export const length: Builtin = {
  name: "length",
  transfer(argTypes, nargout) {
    if (argTypes.length !== 1) {
      throw new TypeError(`'length' expects 1 arg(s), got ${argTypes.length}`);
    }
    if (nargout !== 1) {
      throw new UnsupportedConstruct(
        `'length' does not support multi-output (nargout=${nargout})`
      );
    }
    const t: Type = argTypes[0];
    // Cells take the same max-of-shape semantics. A cell of unknown
    // shape would need a per-shape branch in codegen (deferred).
    if (isCell(t)) {
      if (t.shape !== undefined) {
        const v = t.shape.some(s => s === 0)
          ? 0
          : t.shape.reduce((a, b) => Math.max(a, b), 0);
        return [scalarDouble(signFromNumber(v), v)];
      }
      return [scalarDouble("nonneg")];
    }
    if (t.kind === "Unknown") {
      // An Unknown-typed value (e.g. `fieldnames(s)`) can't be
      // materialized by codegen, so this never reaches an emit hook.
      return [scalarDouble("nonneg")];
    }
    if (!isNumeric(t)) {
      throw new TypeError(`'length' arg must be numeric (got ${t.kind})`);
    }
    if (t.shape === undefined) {
      return [scalarDouble("nonneg")];
    }
    let v = 0;
    if (t.shape.some(s => s === 0)) v = 0;
    else v = t.shape.reduce((a, b) => Math.max(a, b), 0);
    return [scalarDouble(signFromNumber(v), v)];
  },
  emitC({ argsC, argTypes, useRuntime }) {
    useRuntime("mtoc2_length");
    const t = argTypes[0];
    if (isNumeric(t) && isScalar(t)) {
      return `1.0`;
    }
    return `mtoc2_length(${argsC[0]})`;
  },
  emitJs({ argsJs, argTypes }) {
    const t = argTypes[0];
    if (isNumeric(t) && isScalar(t)) return `1`;
    // MATLAB length: max(shape) when any element, else 0.
    return (
      `(${argsJs[0]}.shape.some(s=>s===0) ? 0 : ` +
      `${argsJs[0]}.shape.reduce((a,b)=>a>b?a:b, 0))`
    );
  },
};
