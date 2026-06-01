import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import { isText, typeToString } from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import {
  emitFormatSlot,
  emitFormatSlotArray,
  emitTextView,
  validateFormatArgs,
} from "./_format_args.js";
import {
  mtoc2_error_fmt as jsErrorFmt,
  mtoc2_error_fmt_id as jsErrorFmtId,
} from "../../runtime/snippets.gen.js";

const MATLAB_ID_REGEX = /^[A-Za-z]\w*(:[A-Za-z]\w*)+$/;

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
    useRuntime("mtoc2_error_fmt");
    const first = argTypes[0];
    const firstExact =
      first.kind === "Char" || first.kind === "String"
        ? first.exact
        : undefined;
    let fmtIdx = 0;
    if (
      argTypes.length >= 2 &&
      firstExact !== undefined &&
      MATLAB_ID_REGEX.test(firstExact)
    ) {
      fmtIdx = 1;
    }
    const fmtView = emitTextView(argsC[fmtIdx], argTypes[fmtIdx]);
    const slots: string[] = [];
    for (let i = fmtIdx + 1; i < argTypes.length; i++) {
      slots.push(emitFormatSlot("error", argsC[i], argTypes[i], i));
    }
    return `mtoc2_error_fmt(${fmtView}, ${slots.length}, ${emitFormatSlotArray(slots)})`;
  },
  emitJs({ argsJs, argTypes, useRuntime }) {
    const first = argTypes[0];
    const firstExact =
      first.kind === "Char" || first.kind === "String"
        ? first.exact
        : undefined;
    let fmtIdx = 0;
    if (
      argTypes.length >= 2 &&
      firstExact !== undefined &&
      MATLAB_ID_REGEX.test(firstExact)
    ) {
      fmtIdx = 1;
    }
    if (fmtIdx === 1 && firstExact !== undefined) {
      useRuntime("mtoc2_error_fmt_id");
      return `mtoc2_error_fmt_id(${JSON.stringify(firstExact)}, ${argsJs.slice(fmtIdx).join(", ")})`;
    }
    useRuntime("mtoc2_error_fmt");
    return `mtoc2_error_fmt(${argsJs.slice(fmtIdx).join(", ")})`;
  },
  call({ args, argTypes }) {
    const first = argTypes[0];
    const firstExact =
      first.kind === "Char" || first.kind === "String"
        ? first.exact
        : undefined;
    let fmtIdx = 0;
    if (
      argTypes.length >= 2 &&
      firstExact !== undefined &&
      MATLAB_ID_REGEX.test(firstExact)
    ) {
      fmtIdx = 1;
    }
    const unwrapped = args.slice(fmtIdx).map(unwrapFmtArg);
    const fmt = unwrapped[0] as string;
    if (fmtIdx === 1 && firstExact !== undefined) {
      // Identifier-bearing form `error('id:bad', fmt, args...)` —
      // thread the identifier so a downstream `try/catch ME` arm
      // sees it on `ME.identifier`.
      jsErrorFmtId(firstExact, fmt, ...unwrapped.slice(1));
    } else {
      jsErrorFmt(fmt, ...unwrapped.slice(1));
    }
    return [];
  },
};

function unwrapFmtArg(v: unknown): unknown {
  if (typeof v === "object" && v !== null) {
    const o = v as { mtoc2Tag?: string; value?: string };
    if (o.mtoc2Tag === "char" && typeof o.value === "string") return o.value;
  }
  return v;
}
