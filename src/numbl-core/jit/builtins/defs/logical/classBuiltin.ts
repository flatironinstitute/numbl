/**
 * `class(x)` — return the static type name as a char (single-quoted)
 * value.
 *
 * Always static in mtoc2: the type lattice knows every value's kind
 * (numeric/double, numeric/logical, char, string, struct, class
 * instance, cell, function-handle). The transfer attaches the
 * resolved name to the `CharType`'s `exact` slot so subsequent
 * `strcmp` / `if class(x) == 'double'` patterns see a folded value.
 *
 * Matches numbl `introspection.ts`:
 *   - number / complex-number / sparse / real-numeric tensor → "double"
 *   - boolean / logical-flagged tensor → "logical"
 *   - char → "char", string → "string"
 *   - struct / struct-array → "struct"
 *   - cell → "cell"
 *   - class instance → its `className`
 *   - function handle → "function_handle"
 */

import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import type { Type } from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import {
  isChar as isRtChar,
  isComplexValue,
  isHandleValue,
  isTensor,
  makeChar,
  type RuntimeValue,
} from "../../../runtime/value.js";

function classNameForType(t: Type): string {
  switch (t.kind) {
    case "Numeric":
      // Logical lane → "logical"; everything else (double or complex)
      // → "double". Matches numbl's classifier.
      return t.elem === "logical" ? "logical" : "double";
    case "Char":
      return "char";
    case "String":
      return "string";
    case "Struct":
      return "struct";
    case "Class":
      return t.className;
    case "Cell":
      return "cell";
    case "Handle":
      return "function_handle";
    case "Unknown":
    case "Void":
      throw new UnsupportedConstruct(
        `'class' on a ${t.kind === "Void" ? "void" : "unknown"} value is not supported`
      );
  }
}

/** Inline ASCII C-string literal builder. The names we emit
 *  (`"double"`, `"logical"`, `"char"`, …, plus user class names that
 *  must already be C identifiers per the parser) contain no
 *  characters that need C-level escaping. */
function asciiCLit(s: string): { lit: string; byteLen: number } {
  let out = '"';
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    if (c === 0x5c) out += "\\\\";
    else if (c === 0x22) out += '\\"';
    else if (c >= 0x20 && c < 0x7f) out += String.fromCharCode(c);
    else {
      // Defensive fallback — user class names that snuck through the
      // identifier check with non-ASCII bytes would otherwise emit
      // broken C.
      out += `\\x${c.toString(16).padStart(2, "0")}`;
    }
  }
  out += '"';
  return { lit: out, byteLen: new TextEncoder().encode(s).length };
}

export const classBuiltin: Builtin = {
  name: "class",
  transfer(argTypes, nargout) {
    if (argTypes.length !== 1) {
      throw new TypeError(`'class' expects 1 arg(s), got ${argTypes.length}`);
    }
    if (nargout !== 1) {
      throw new UnsupportedConstruct(
        `'class' does not support multi-output (nargout=${nargout})`
      );
    }
    const name = classNameForType(argTypes[0]);
    return [{ kind: "Char", exact: name }];
  },
  emitC({ argTypes, useRuntime }) {
    const name = classNameForType(argTypes[0]);
    useRuntime("mtoc2_char_tensor_from_literal");
    const { lit, byteLen } = asciiCLit(name);
    return `mtoc2_char_tensor_from_literal(${lit}, ${byteLen})`;
  },
  emitJs({ argTypes }) {
    const name = classNameForType(argTypes[0]);
    return `({mtoc2Tag:"char", value:${JSON.stringify(name)}})`;
  },
  call({ args, argTypes }) {
    // Check the runtime value for a function handle first — the
    // interpreter's `inferTypeFromValue` does not yet detect handles
    // (it falls them through to struct-like), which would otherwise
    // misclassify `class(@foo)` as "struct".
    if (isHandleValue(args[0])) return [makeChar("function_handle")];
    const t = argTypes[0];
    if (t.kind !== "Unknown" && t.kind !== "Void") {
      return [makeChar(classNameForType(t))];
    }
    // Fallback runtime classifier — exercised only when the static
    // type was lost (shouldn't happen via `inferTypeFromValue`, but
    // keep parity with numbl for safety).
    const v: RuntimeValue = args[0];
    if (typeof v === "number") return [makeChar("double")];
    if (typeof v === "boolean") return [makeChar("logical")];
    if (typeof v === "string") return [makeChar("string")];
    if (isRtChar(v)) return [makeChar("char")];
    if (isComplexValue(v)) return [makeChar("double")];
    if (isTensor(v)) {
      return [makeChar(v.isLogical ? "logical" : "double")];
    }
    if (isHandleValue(v)) return [makeChar("function_handle")];
    if (v && typeof v === "object") {
      if ((v as { mtoc2Tag?: string }).mtoc2Tag === "cell") {
        return [makeChar("cell")];
      }
      const cn = (v as { mtoc2Class?: string }).mtoc2Class;
      if (cn !== undefined) return [makeChar(cn)];
      return [makeChar("struct")];
    }
    return [makeChar("unknown")];
  },
};
