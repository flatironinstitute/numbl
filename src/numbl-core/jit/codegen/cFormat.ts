/**
 * Pure C-source formatting utilities used by the codegen modules. No
 * runtime-state threading, no IR dependencies — string in, string out.
 * Lives separately so the emit-* modules can import them without
 * cycling through emit.ts.
 */

import type { Type } from "../lowering/types.js";

/** Encode `s` as a C string literal `"..."` whose bytes are the UTF-8
 *  encoding of `s`. Returns `{ lit, byteLen }` where `byteLen` is the
 *  encoded byte count (excluding the trailing NUL). Non-printable bytes
 *  use `\xHH`; backslash/quote/standard whitespace use the C escapes.
 *
 *  Why hex over `\uHHHH`: narrow C string literals don't accept
 *  universal-character names; we need byte-level escapes to keep the
 *  emitted code portable. */
export function cStringLiteral(s: string): { lit: string; byteLen: number } {
  const bytes = new TextEncoder().encode(s);
  let out = '"';
  for (let i = 0; i < bytes.length; i++) {
    const b = bytes[i];
    if (b === 0x5c) {
      out += "\\\\";
    } else if (b === 0x22) {
      out += '\\"';
    } else if (b === 0x0a) {
      out += "\\n";
    } else if (b === 0x0d) {
      out += "\\r";
    } else if (b === 0x09) {
      out += "\\t";
    } else if (b >= 0x20 && b < 0x7f) {
      out += String.fromCharCode(b);
    } else {
      // \xHH (lowercase). Wrap in `""` separator if a hex digit follows
      // so the runtime hex escape doesn't gobble up valid following
      // characters (e.g. "\x0a1" must be "\x0a" "1", not "\x0a1").
      const hex = b.toString(16).padStart(2, "0");
      const next = i + 1 < bytes.length ? bytes[i + 1] : 0;
      const nextIsHex =
        (next >= 0x30 && next <= 0x39) ||
        (next >= 0x41 && next <= 0x46) ||
        (next >= 0x61 && next <= 0x66);
      out += `\\x${hex}`;
      if (nextIsHex) out += '" "';
    }
  }
  out += '"';
  return { lit: out, byteLen: bytes.length };
}

/** Format a Span as a quoted "<file>:<offset>" string literal for
 *  passing to a runtime helper. The file path is JSON-escaped (the
 *  emitter already requires a single C-string-safe form), and the
 *  offset is the byte offset of the violating expression — matches
 *  the format used by translate-time `UnsupportedConstruct` errors,
 *  so the user sees a familiar location for runtime OOB too. */
export function locStringOf(span: { file: string; start: number }): string {
  return `${JSON.stringify(`${span.file}:offset ${span.start}`)}`;
}

/** Build the C expression for `numel(<cName>)` as the product of its
 *  per-axis `.dims[i]` slots. Returns `"0L"` for an empty / non-numeric
 *  type so the emitted C still parses on degenerate inputs. */
export function dimsProductExpr(cName: string, ty: Type): string {
  if (ty.kind !== "Numeric" || ty.dims.length === 0) return "0L";
  const parts: string[] = [];
  for (let i = 0; i < ty.dims.length; i++) parts.push(`${cName}.dims[${i}]`);
  return parts.join(" * ");
}

/** Compute the column-major linear offset for an N-D access given
 *  per-axis source terms and a stride source. */
export function formatNdOffset(
  terms: ReadonlyArray<string>,
  stride: (axisIndex: number) => string
): string {
  if (terms.length === 0) return "0";
  const out: string[] = [];
  for (let i = 0; i < terms.length; i++) {
    if (i === 0) {
      out.push(terms[i]);
    } else {
      const strideParts: string[] = [];
      for (let j = 0; j < i; j++) strideParts.push(stride(j));
      out.push(`${terms[i]} * ${strideParts.join(" * ")}`);
    }
  }
  return out.join(" + ");
}
