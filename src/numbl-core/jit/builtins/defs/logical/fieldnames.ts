/**
 * `fieldnames(s)` — returns an N×1 cell of char-array field names
 * for a struct or class instance. Matches numbl's
 * `introspection.ts:fieldnamesApply` semantics, with the result's
 * cell-element type being `char` (matches MATLAB; numbl wraps as
 * string but mtoc2 follows MATLAB by emitting chars).
 *
 * Interpreter-only for now; the AOT path would need to materialize
 * a heterogeneous-len cell of chars which is a different feature
 * class. The `emitC` / `emitJs` hooks exist so the registry-shape
 * invariant holds but they raise `UnsupportedConstruct` rather than
 * letting the framework's generic "no emitC hook" surface.
 */

import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import { isStruct, isClass, UNKNOWN } from "../../../lowering/types.js";
import {
  isTensor,
  isChar as isCharRV,
  makeChar,
  type RuntimeValue,
} from "../../../runtime/value.js";
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
    // Result is a runtime-length cell of chars. mtoc2's AOT cell
    // lattice doesn't have a "row-vector cell of variable-length
    // chars" shape today, so AOT lowering of any downstream use
    // would fail. The interpreter just needs `transfer` not to
    // throw — return Unknown so the interpreter's
    // post-`transfer` `call` hook runs and produces a concrete
    // cell. AOT (c-aot / js-aot) consumers will hit downstream
    // errors when they try to materialize an Unknown result.
    return [UNKNOWN];
  },
  emitC() {
    throw new UnsupportedConstruct(
      `'fieldnames' is interpreter-only; the c-aot backend can't materialize a runtime-length cell of chars`
    );
  },
  emitJs() {
    throw new UnsupportedConstruct(
      `'fieldnames' is interpreter-only; the js-aot backend can't materialize a runtime-length cell of chars`
    );
  },
  call({ args }) {
    const v = args[0];
    if (v === null || typeof v !== "object") {
      throw new Error(`fieldnames: argument must be a struct or class`);
    }
    const rv = v as RuntimeValue;
    if (isTensor(rv) || isCharRV(rv)) {
      throw new Error(`fieldnames: argument must be a struct or class`);
    }
    const tagObj = v as {
      mtoc2Tag?: string;
      mtoc2Class?: string;
      mtoc2Handle?: boolean;
    };
    if (
      tagObj.mtoc2Tag === "cell" ||
      tagObj.mtoc2Handle === true ||
      tagObj.mtoc2Tag === "tensor" ||
      tagObj.mtoc2Tag === "char"
    ) {
      throw new Error(`fieldnames: argument must be a struct or class`);
    }
    // Enumerable own keys, excluding internal tags (mtoc2Class).
    const keys = Object.keys(v as Record<string, unknown>);
    const data: RuntimeValue[] = keys.map(k => makeChar(k));
    return [
      {
        mtoc2Tag: "cell",
        shape: [keys.length, 1],
        data,
      } as unknown as RuntimeValue,
    ];
  },
};
