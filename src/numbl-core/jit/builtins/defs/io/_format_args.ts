import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import {
  isMultiElement,
  isNumeric,
  isScalarRealNumeric,
  isText,
  typeToString,
  type Type,
} from "../../../lowering/types.js";

/** Validate that args at positions [from..) are legal `mtoc2_fprintf_arg_t`
 *  slots — scalar real numeric, text, or real multi-element tensor.
 *  Throws TypeError / UnsupportedConstruct (no span — framework attaches
 *  via `withSpan`). Shared by `fprintf`, `error`, future `sprintf`. */
export function validateFormatArgs(
  builtinName: string,
  argTypes: ReadonlyArray<Type>,
  from: number
): void {
  for (let i = from; i < argTypes.length; i++) {
    const a = argTypes[i];
    if (isText(a)) continue;
    if (isNumeric(a)) {
      if (a.isComplex) {
        throw new UnsupportedConstruct(
          `'${builtinName}' on a complex arg is not yet supported`
        );
      }
      if (a.elem !== "double" && a.elem !== "logical") {
        throw new UnsupportedConstruct(
          `'${builtinName}' on a '${a.elem}' arg is not yet supported`
        );
      }
      continue;
    }
    throw new TypeError(
      `'${builtinName}' arg ${i + 1} must be numeric or text (got ${typeToString(a)})`
    );
  }
}

/** `mtoc2_fprintf_arg_t.kind` tag values. Must stay in lockstep with
 *  the `MTOC2_FA_*` enum in `src/codegen/runtime/format_engine.h`. */
const MTOC2_FA_DOUBLE = "MTOC2_FA_DOUBLE";
const MTOC2_FA_TEXT = "MTOC2_FA_TEXT";
const MTOC2_FA_TENSOR = "MTOC2_FA_TENSOR";

/** Emit one `mtoc2_fprintf_arg_t` slot for `argsC[i]` typed `argTypes[i]`.
 *  Used to build the compound-literal slot array for fprintf / error /
 *  sprintf call sites. ANF in the lowerer guarantees multi-element
 *  tensor args are bare Var lvalues (so `&c` is well-defined). */
export function emitFormatSlot(
  builtinName: string,
  c: string,
  t: Type,
  argIndex: number
): string {
  if (t.kind === "String") {
    return `{.kind = ${MTOC2_FA_TEXT}, .u = {.t = mtoc2_text_from_string(${c})}}`;
  }
  if (t.kind === "Char") {
    return `{.kind = ${MTOC2_FA_TEXT}, .u = {.t = mtoc2_text_from_char_tensor(${c})}}`;
  }
  if (isScalarRealNumeric(t)) {
    return `{.kind = ${MTOC2_FA_DOUBLE}, .u = {.d = (double)(${c})}}`;
  }
  if (isNumeric(t) && isMultiElement(t)) {
    return `{.kind = ${MTOC2_FA_TENSOR}, .u = {.tensor = &${c}}}`;
  }
  throw new Error(
    `internal: '${builtinName}' arg ${argIndex + 1} reached codegen with unsupported type ${t.kind}`
  );
}

/** Build the `(const mtoc2_fprintf_arg_t[]){ ... }` compound literal
 *  from a slice of slots (or the `{ {0} }` empty form C99 needs to
 *  avoid the zero-length-array GNU extension). */
export function emitFormatSlotArray(slots: ReadonlyArray<string>): string {
  if (slots.length === 0) return `(const mtoc2_fprintf_arg_t[]){ {0} }`;
  return `(const mtoc2_fprintf_arg_t[]){ ${slots.join(", ")} }`;
}

/** Adapter expression for a text-typed format arg → `mtoc2_text_view_t`. */
export function emitTextView(c: string, t: Type): string {
  if (t.kind === "String") return `mtoc2_text_from_string(${c})`;
  if (t.kind === "Char") return `mtoc2_text_from_char_tensor(${c})`;
  throw new Error(`internal: emitTextView called on non-text type ${t.kind}`);
}
