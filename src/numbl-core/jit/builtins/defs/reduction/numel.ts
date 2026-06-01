import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import {
  scalarDouble,
  shapeNumel,
  signFromNumber,
  isNumeric,
  isScalar,
} from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import { isTensor } from "../../../runtime/value.js";

export const numel: Builtin = {
  name: "numel",
  transfer(argTypes, nargout) {
    if (argTypes.length !== 1) {
      throw new TypeError(`'numel' expects 1 arg(s), got ${argTypes.length}`);
    }
    if (nargout !== 1) {
      throw new UnsupportedConstruct(
        `'numel' does not support multi-output (nargout=${nargout})`
      );
    }
    const t = argTypes[0];
    if (!isNumeric(t)) {
      throw new TypeError(`'numel' arg must be numeric (got ${t.kind})`);
    }
    if (t.shape === undefined) {
      return [scalarDouble("nonneg")];
    }
    const v = shapeNumel(t.shape);
    return [scalarDouble(signFromNumber(v), v)];
  },
  emitC({ argsC, argTypes, useRuntime }) {
    useRuntime("mtoc2_numel");
    const t = argTypes[0];
    if (isNumeric(t) && isScalar(t)) {
      return `1.0`;
    }
    return `mtoc2_numel(${argsC[0]})`;
  },
  emitJs({ argsJs, argTypes }) {
    const t = argTypes[0];
    if (isNumeric(t) && isScalar(t)) return `1`;
    // For a tensor, the JS representation carries `.shape`; multiply it.
    return `${argsJs[0]}.shape.reduce((a,b)=>a*b, 1)`;
  },
  call({ args }) {
    const v = args[0];
    if (typeof v === "number" || typeof v === "boolean") return [1];
    if (isTensor(v)) return [v.shape.reduce((a, b) => a * b, 1)];
    if (typeof v === "string") return [v.length];
    throw new TypeError(
      `'numel' got an unsupported runtime value (typeof = ${typeof v})`
    );
  },
};
