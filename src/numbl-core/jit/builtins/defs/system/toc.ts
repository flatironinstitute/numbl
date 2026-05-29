/**
 * toc — read the elapsed wall-clock seconds since `tic`.
 */

import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import { isScalarRealNumeric, scalarDouble } from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import { mtoc2_toc, mtoc2_toc_handle } from "../../runtime/snippets.gen.js";

export const toc: Builtin = {
  name: "toc",
  transfer(argTypes, nargout) {
    if (argTypes.length > 1) {
      throw new TypeError(`'toc' expects 0..1 arg(s), got ${argTypes.length}`);
    }
    if (nargout !== 1) {
      throw new UnsupportedConstruct(
        `'toc' does not support multi-output (nargout=${nargout})`
      );
    }
    if (argTypes.length === 1 && !isScalarRealNumeric(argTypes[0])) {
      throw new TypeError(
        `'toc' tic-handle argument must be a scalar real numeric ` +
          `(the value returned by 'tic'); got ${argTypes[0].kind}`
      );
    }
    return [scalarDouble("nonneg")];
  },
  emitC({ argsC, useRuntime }) {
    useRuntime("mtoc2_tic_toc");
    if (argsC.length === 1) return `mtoc2_toc_handle(${argsC[0]})`;
    return "mtoc2_toc()";
  },
  emitJs({ argsJs, useRuntime }) {
    useRuntime("mtoc2_tic_toc");
    if (argsJs.length === 1) return `mtoc2_toc_handle(${argsJs[0]})`;
    return "mtoc2_toc()";
  },
  call({ args }) {
    if (args.length === 1) {
      const s = typeof args[0] === "number" ? args[0] : Number(args[0]);
      return [mtoc2_toc_handle(s)];
    }
    return [mtoc2_toc()];
  },
};
