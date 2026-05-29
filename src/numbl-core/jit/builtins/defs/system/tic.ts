/**
 * tic — start the wall-clock stopwatch.
 *
 * Returns the start time in seconds (matches numbl: `performance.now()
 * / 1000`).
 */

import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import { scalarDouble } from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import { mtoc2_tic } from "../../runtime/snippets.gen.js";

export const tic: Builtin = {
  name: "tic",
  transfer(argTypes, nargout) {
    if (argTypes.length !== 0) {
      throw new TypeError(`'tic' expects 0 arg(s), got ${argTypes.length}`);
    }
    if (nargout !== 1) {
      throw new UnsupportedConstruct(
        `'tic' does not support multi-output (nargout=${nargout})`
      );
    }
    return [scalarDouble("positive")];
  },
  emitC({ useRuntime }) {
    useRuntime("mtoc2_tic_toc");
    return "mtoc2_tic()";
  },
  emitJs({ useRuntime }) {
    useRuntime("mtoc2_tic_toc");
    return "mtoc2_tic()";
  },
  call() {
    return [mtoc2_tic()];
  },
};
