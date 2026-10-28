import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import { isText, typeToString, type Type } from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import {
  emitFormatSlot,
  emitFormatSlotArray,
  emitTextView,
  validateFormatArgs,
} from "./_format_args.js";

export const MATLAB_ID_REGEX = /^[A-Za-z]\w*(:[A-Za-z]\w*)+$/;

export const errorBuiltin: Builtin = {
  name: "error",
  transfer(argTypes, nargout) {
    if (argTypes.length === 0) {
      throw new TypeError(`'error' requires at least 1 arg`);
    }
    if (nargout !== 1) {
      throw new UnsupportedConstruct(
        `'error' does not support multi-output (nargout=${nargout})`
      );
    }
    const first = argTypes[0];
    if (!isText(first)) {
      throw new TypeError(
        `'error' first arg must be char or string (got ${typeToString(first)})`
      );
    }
    const firstExact =
      first.kind === "Char" || first.kind === "String"
        ? first.exact
        : undefined;
    if (firstExact === undefined) {
      throw new UnsupportedConstruct(
        `'error' first arg must be a literal text value in v1 (so the ` +
          `id-vs-format decision can be resolved at compile time)`
      );
    }
    let fmtIdx = 0;
    if (argTypes.length >= 2 && MATLAB_ID_REGEX.test(firstExact)) {
      fmtIdx = 1;
    }
    const fmt = argTypes[fmtIdx];
    if (!isText(fmt)) {
      throw new TypeError(
        `'error' format arg must be char or string (got ${typeToString(fmt)})`
      );
    }
    validateFormatArgs("error", argTypes, fmtIdx + 1);
    return [{ kind: "Void" }];
  },
  emitC({ argsC, argTypes, useRuntime }) {
    useRuntime("jit_error_fmt");
    const { fmtIdx, id } = errorForm(argTypes);
    const view = emitTextView(argsC[fmtIdx], argTypes[fmtIdx]);
    const idC = JSON.stringify(id);
    // Mirrors the interpreter: only a lone `error(msg)` is taken
    // literally; the id form and any form with arguments format.
    if (argTypes.length === 1) {
      return `jit_error_text(${idC}, ${view})`;
    }
    const slots: string[] = [];
    for (let i = fmtIdx + 1; i < argTypes.length; i++) {
      slots.push(emitFormatSlot("error", argsC[i], argTypes[i], i));
    }
    return `jit_error_fmt(${idC}, ${view}, ${slots.length}, ${emitFormatSlotArray(slots)})`;
  },
  emitJs({ argsJs, argTypes, useRuntime }) {
    useRuntime("jit_error_fmt");
    const { fmtIdx, id } = errorForm(argTypes);
    const idJs = JSON.stringify(id);
    if (argTypes.length === 1) {
      return `jit_error_text(${idJs}, ${argsJs[0]})`;
    }
    return `jit_error_fmt(${idJs}, ${argsJs.slice(fmtIdx).join(", ")})`;
  },
};

/** The interpreter's `error` argument forms: `error(id, msg, ...)` when
 *  the (literal) first argument is a MATLAB identifier and more arguments
 *  follow, else `error(msg, ...)`. */
function errorForm(argTypes: ReadonlyArray<Type>): {
  fmtIdx: number;
  id: string;
} {
  const first = argTypes[0];
  const firstExact =
    first.kind === "Char" || first.kind === "String" ? first.exact : undefined;
  if (
    argTypes.length >= 2 &&
    firstExact !== undefined &&
    MATLAB_ID_REGEX.test(firstExact)
  ) {
    return { fmtIdx: 1, id: firstExact };
  }
  return { fmtIdx: 0, id: "" };
}
