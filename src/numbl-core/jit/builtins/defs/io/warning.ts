/**
 * `warning(fmt, args...)` and `warning('id:topic', fmt, args...)` —
 * print `Warning: <message>\n` to stdout. Matches numbl
 * (`numbl-core/runtime/specialBuiltins.ts:190`).
 *
 * Unlike `error`, `warning` doesn't throw and has no `try/catch`
 * interaction; the identifier-bearing form is accepted for
 * source-level compatibility but the id is dropped (numbl wires it
 * into a `lastwarn` state mtoc2 doesn't expose).
 *
 * The state-query form `warning('on'/'off', id)` and the no-arg
 * call `warning()` are deferred — chunkie doesn't use them.
 */

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
  mtoc2_warning_fmt as jsWarningFmt,
  mtoc2_warning_fmt_id as jsWarningFmtId,
} from "../../runtime/snippets.gen.js";

/** Numbl's id-detection for `warning` is intentionally looser than for
 *  `error`: any colon in the first text arg flips it to "first arg is
 *  the identifier" mode, even something like `'count: %d'`. We mirror
 *  byte-for-byte (see numbl `runtime/specialBuiltins.ts:215`) — it's
 *  arguably a numbl design quirk, but the cross-runner contract is
 *  exact match, and users writing colon-bearing formats just get
 *  numbl's behavior. */
function warningFirstArgIsId(s: string): boolean {
  return s.includes(":");
}

export const warningBuiltin: Builtin = {
  name: "warning",
  transfer(argTypes, nargout) {
    if (argTypes.length === 0) {
      throw new UnsupportedConstruct(
        `'warning' with no args (numbl uses it for state queries) is not yet supported`
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
        `'warning' first arg must be a literal text value in v1 (so the ` +
          `id-vs-format decision can be resolved at compile time)`
      );
    }
    let fmtIdx = 0;
    if (argTypes.length >= 2 && warningFirstArgIsId(firstExact)) {
      fmtIdx = 1;
    }
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
    useRuntime("mtoc2_warning_fmt");
    const first = argTypes[0];
    const firstExact =
      first.kind === "Char" || first.kind === "String"
        ? first.exact
        : undefined;
    let fmtIdx = 0;
    if (
      argTypes.length >= 2 &&
      firstExact !== undefined &&
      warningFirstArgIsId(firstExact)
    ) {
      fmtIdx = 1;
    }
    const fmtView = emitTextView(argsC[fmtIdx], argTypes[fmtIdx]);
    const slots: string[] = [];
    for (let i = fmtIdx + 1; i < argTypes.length; i++) {
      slots.push(emitFormatSlot("warning", argsC[i], argTypes[i], i));
    }
    return `mtoc2_warning_fmt(${fmtView}, ${slots.length}, ${emitFormatSlotArray(slots)})`;
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
      warningFirstArgIsId(firstExact)
    ) {
      fmtIdx = 1;
    }
    if (fmtIdx === 1 && firstExact !== undefined) {
      useRuntime("mtoc2_warning_fmt_id");
      return `mtoc2_warning_fmt_id(${JSON.stringify(firstExact)}, ${argsJs.slice(fmtIdx).join(", ")})`;
    }
    useRuntime("mtoc2_warning_fmt");
    return `mtoc2_warning_fmt(${argsJs.slice(fmtIdx).join(", ")})`;
  },
  call({ args, argTypes, ctx }) {
    const first = argTypes[0];
    const firstExact =
      first.kind === "Char" || first.kind === "String"
        ? first.exact
        : undefined;
    let fmtIdx = 0;
    if (
      argTypes.length >= 2 &&
      firstExact !== undefined &&
      warningFirstArgIsId(firstExact)
    ) {
      fmtIdx = 1;
    }
    const unwrapped = args.slice(fmtIdx).map(unwrapFmtArg);
    const fmt = unwrapped[0] as string;
    globalThis.$write = ctx.helpers.write;
    if (fmtIdx === 1 && firstExact !== undefined) {
      jsWarningFmtId(firstExact, fmt, ...unwrapped.slice(1));
    } else {
      jsWarningFmt(fmt, ...unwrapped.slice(1));
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
