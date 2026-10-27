import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import { isNumeric, isText, typeToString } from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import {
  emitFormatSlot,
  emitFormatSlotArray,
  emitTextView,
  validateFormatArgs,
} from "./_format_args.js";

export const fprintf: Builtin = {
  name: "fprintf",
  transfer(argTypes, nargout) {
    if (argTypes.length === 0) {
      throw new TypeError(`'fprintf' requires at least a format arg`);
    }
    if (nargout !== 1) {
      throw new UnsupportedConstruct(
        `'fprintf' does not support multi-output (nargout=${nargout})`
      );
    }
    const fmt = argTypes[0];
    if (isNumeric(fmt)) {
      throw new UnsupportedConstruct(
        `'fprintf' with a leading fid (e.g. fprintf(1, fmt, ...)) is not ` +
          `yet supported — pass the format as the first arg`
      );
    }
    if (!isText(fmt)) {
      throw new TypeError(
        `'fprintf' first arg must be a char or string format (got ${typeToString(fmt)})`
      );
    }
    validateFormatArgs("fprintf", argTypes, 1);
    return [{ kind: "Void" }];
  },
  emitC({ argsC, argTypes, useRuntime }) {
    useRuntime("mtoc2_fprintf");
    const fmtView = emitTextView(argsC[0], argTypes[0]);
    const slots: string[] = [];
    for (let i = 1; i < argTypes.length; i++) {
      slots.push(emitFormatSlot("fprintf", argsC[i], argTypes[i], i));
    }
    return `mtoc2_fprintf(stdout, ${fmtView}, ${slots.length}, ${emitFormatSlotArray(slots)})`;
  },
  emitJs({ argsJs, useRuntime }) {
    // Different call shape from C side — JS uses variadic args
    // directly. Format text passes through as a JS string. Tensor/
    // char/string args are passed by value; the format engine
    // flattens tensors column-major internally.
    useRuntime("mtoc2_fprintf");
    return `mtoc2_fprintf(${argsJs.join(", ")})`;
  },
};
