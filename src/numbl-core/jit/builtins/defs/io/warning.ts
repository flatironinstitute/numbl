/**
 * `warning(msg)`, `warning(fmt, args...)` and `warning(id, fmt, args...)`.
 * Mirrors the interpreter's `warning` (interpreter/builtins/output.ts): a
 * lone `warning(msg)` is taken literally, the id form and any form with
 * arguments format.
 *
 * Whether a warning is shown depends on the runtime warning state, so the
 * emitted code hands the identifier and message to the host
 * (`jit_warning_*` snippets), which checks the state and writes the
 * `Warning: ...` line. The state-changing and query forms
 * (`warning('off', id)`, `warning('on')`, `warning('query', id)`, the
 * struct-restore form and `warning()`) decline to the interpreter.
 */

import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import { isText, typeToString, type Type } from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import {
  emitFormatSlot,
  emitFormatSlotArray,
  emitTextView,
  validateFormatArgs,
} from "./_format_args.js";

/** First arguments that make `warning` a state set or query. */
const STATE_MODES = new Set(["on", "off", "query"]);

/** The interpreter's `warning` argument forms. Its id detection is looser
 *  than `error`'s: any colon in a first argument followed by more arguments
 *  makes it the identifier. `id` is "" for none. */
function warningForm(argTypes: ReadonlyArray<Type>): {
  fmtIdx: number;
  id: string;
} {
  const first = argTypes[0];
  const firstExact =
    first.kind === "Char" || first.kind === "String" ? first.exact : undefined;
  if (
    argTypes.length >= 2 &&
    firstExact !== undefined &&
    firstExact.includes(":")
  ) {
    return { fmtIdx: 1, id: firstExact };
  }
  return { fmtIdx: 0, id: "" };
}

export const warningBuiltin: Builtin = {
  name: "warning",
  transfer(argTypes, nargout) {
    if (argTypes.length === 0) {
      throw new UnsupportedConstruct(
        `'warning' with no args (a state query) is not supported`
      );
    }
    if (nargout !== 1) {
      throw new UnsupportedConstruct(
        `'warning' does not support multi-output (nargout=${nargout})`
      );
    }
    const first = argTypes[0];
    if (!isText(first)) {
      throw new TypeError(
        `'warning' first arg must be char or string (got ${typeToString(first)})`
      );
    }
    const firstExact =
      first.kind === "Char" || first.kind === "String"
        ? first.exact
        : undefined;
    if (firstExact === undefined) {
      throw new UnsupportedConstruct(
        `'warning' first arg must be a literal text value (so the ` +
          `id-vs-format decision can be resolved at compile time)`
      );
    }
    if (argTypes.length <= 2 && STATE_MODES.has(firstExact)) {
      throw new UnsupportedConstruct(
        `warning('${firstExact}', ...) changes or queries the warning ` +
          `state, which is not supported`
      );
    }
    const { fmtIdx } = warningForm(argTypes);
    const fmt = argTypes[fmtIdx];
    if (!isText(fmt)) {
      throw new TypeError(
        `'warning' format arg must be char or string (got ${typeToString(fmt)})`
      );
    }
    validateFormatArgs("warning", argTypes, fmtIdx + 1);
    return [{ kind: "Void" }];
  },
  emitC({ argsC, argTypes, useRuntime }) {
    useRuntime("jit_warning_fmt");
    const { fmtIdx, id } = warningForm(argTypes);
    const view = emitTextView(argsC[fmtIdx], argTypes[fmtIdx]);
    const idC = JSON.stringify(id);
    if (argTypes.length === 1) {
      return `jit_warning_text(${idC}, ${view})`;
    }
    const slots: string[] = [];
    for (let i = fmtIdx + 1; i < argTypes.length; i++) {
      slots.push(emitFormatSlot("warning", argsC[i], argTypes[i], i));
    }
    return `jit_warning_fmt(${idC}, ${view}, ${slots.length}, ${emitFormatSlotArray(slots)})`;
  },
  emitJs({ argsJs, argTypes, useRuntime }) {
    useRuntime("jit_warning_fmt");
    const { fmtIdx, id } = warningForm(argTypes);
    const idJs = JSON.stringify(id);
    if (argTypes.length === 1) {
      return `jit_warning_text(${idJs}, ${argsJs[0]})`;
    }
    return `jit_warning_fmt(${idJs}, ${argsJs.slice(fmtIdx).join(", ")})`;
  },
};
