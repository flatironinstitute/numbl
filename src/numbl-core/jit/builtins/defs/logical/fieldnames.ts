/**
 * `fieldnames(s)` — returns an N×1 cell of char-array field names
 * for a struct or class instance. Matches numbl's
 * `introspection.ts:fieldnamesApply` semantics, with the result's
 * cell-element type being `char` (matches MATLAB and the
 * interpreter, which also emits chars).
 *
 * Interpreter-only for now; the JIT would need to materialize
 * a heterogeneous-len cell of chars which is a different feature
 * class. The `emitC` / `emitJs` hooks exist so the registry-shape
 * invariant holds but they raise `UnsupportedConstruct` rather than
 * letting the framework's generic "no emitC hook" surface.
 */

import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import { isStruct, isClass, UNKNOWN } from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";

export const fieldnames: Builtin = {
  name: "fieldnames",
  transfer(argTypes, nargout) {
    if (argTypes.length !== 1) {
      throw new TypeError(
        `'fieldnames' expects 1 arg(s), got ${argTypes.length}`
      );
    }
    if (nargout !== 1) {
      throw new UnsupportedConstruct(
        `'fieldnames' does not support multi-output (nargout=${nargout})`
      );
    }
    const t = argTypes[0];
    if (!isStruct(t) && !isClass(t)) {
      throw new TypeError(
        `'fieldnames' argument must be a struct or class instance`
      );
    }
    // Result is a runtime-length cell of chars. The JIT's cell
    // lattice doesn't have a "row-vector cell of variable-length
    // chars" shape today, so return Unknown; codegen of any
    // downstream use declines.
    return [UNKNOWN];
  },
  emitC() {
    throw new UnsupportedConstruct(
      `'fieldnames' is interpreter-only; the JIT can't materialize a runtime-length cell of chars`
    );
  },
  emitJs() {
    throw new UnsupportedConstruct(
      `'fieldnames' is interpreter-only; the JIT can't materialize a runtime-length cell of chars`
    );
  },
};
