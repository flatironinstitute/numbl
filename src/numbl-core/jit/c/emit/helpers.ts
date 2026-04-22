/**
 * Small shared helpers used across the emit/ modules.
 *
 * These are one-liners and trivial utilities that don't belong in any
 * one topic file — keeping them here avoids either duplicating them
 * across topic files or inflating a topic file with unrelated detail.
 */
import type { JitExpr } from "../../jitTypes.js";
import type { TensorMeta } from "../classify.js";
import type { EmitCtx } from "../codegenCtx.js";

export function isTensorExpr(expr: JitExpr): boolean {
  return expr.jitType.kind === "tensor";
}

export function isComplexExpr(expr: JitExpr): boolean {
  return expr.jitType.kind === "complex_or_number";
}

/** Pair of C expressions holding the real and imaginary parts of a
 *  complex scalar value. Produced by `emitComplex`. */
export interface ComplexPair {
  re: string;
  im: string;
}

/** Complex tensor expression result: data + dataIm + len in C. For a
 *  Var whose JitType is a real tensor, `dataIm` is the literal string
 *  `"NULL"` — the numbl_ops complex kernels treat that as "all zero",
 *  so a real tensor flowing into a complex op doesn't need a zero
 *  buffer. */
export interface ComplexTensorResult {
  data: string;
  dataIm: string;
  len: string;
}

/** Widen a real scalar C expression to a complex pair (im = 0). */
export function widenRealToComplex(realCode: string): ComplexPair {
  return { re: realCode, im: "0.0" };
}

/** Escape a JS string into a C string literal (double-quoted, with C
 *  escapes for backslash, double-quote, and common control chars;
 *  non-ASCII bytes encoded as `\xNN` octets of their UTF-8 encoding). */
export function cStringLiteral(s: string): string {
  const bytes = Buffer.from(s, "utf-8");
  let out = '"';
  for (let i = 0; i < bytes.length; i++) {
    const b = bytes[i];
    if (b === 0x5c /* \ */) out += "\\\\";
    else if (b === 0x22 /* " */) out += '\\"';
    else if (b === 0x0a) out += "\\n";
    else if (b === 0x0d) out += "\\r";
    else if (b === 0x09) out += "\\t";
    else if (b >= 0x20 && b < 0x7f) out += String.fromCharCode(b);
    else out += "\\x" + b.toString(16).padStart(2, "0");
  }
  out += '"';
  return out;
}

/** Resolve a tensor name's meta or throw — the tensor-creation emit
 *  helpers depend on the name being classified (with `hasFreshAlloc`)
 *  for the d0/d1 locals they write to actually exist at runtime.
 *  Failing loudly here beats emitting C that references undeclared
 *  identifiers. */
export function requireFreshAllocMeta(
  ctx: EmitCtx,
  destName: string,
  site: string
): TensorMeta {
  const m = ctx.cls.meta.get(destName);
  if (!m) {
    throw new Error(
      `C-JIT codegen: ${site}: dest '${destName}' has no TensorMeta (not classified as a tensor)`
    );
  }
  if (!m.hasFreshAlloc) {
    throw new Error(
      `C-JIT codegen: ${site}: dest '${destName}' is not hasFreshAlloc — shape locals (_d0/_d1) wouldn't exist`
    );
  }
  return m;
}
