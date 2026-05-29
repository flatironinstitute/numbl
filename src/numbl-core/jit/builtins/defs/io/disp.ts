import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import { emitDispCallC } from "../../../codegen/cHelpers.js";
import {
  isCell,
  isScalarRealNumeric,
  isNumeric,
  isScalar,
  isText,
} from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import { isChar, isComplexValue, isTensor } from "../../../runtime/value.js";
import {
  mtoc2_disp_complex,
  mtoc2_disp_double,
  mtoc2_disp_struct,
  mtoc2_disp_tensor,
  mtoc2_disp_tensor_complex,
  mtoc2_format_cell,
} from "../../runtime/snippets.gen.js";

export const disp: Builtin = {
  name: "disp",
  transfer(argTypes, nargout) {
    if (argTypes.length !== 1) {
      throw new TypeError(`'disp' expects 1 arg(s), got ${argTypes.length}`);
    }
    if (nargout !== 1) {
      throw new UnsupportedConstruct(
        `'disp' does not support multi-output (nargout=${nargout})`
      );
    }
    const t = argTypes[0];
    if (isScalarRealNumeric(t)) {
      return [{ kind: "Unknown" }];
    }
    if (isNumeric(t) && isScalar(t) && t.isComplex && t.elem === "double") {
      return [{ kind: "Unknown" }];
    }
    if (
      isNumeric(t) &&
      !t.isComplex &&
      (t.elem === "double" || t.elem === "logical")
    ) {
      return [{ kind: "Unknown" }];
    }
    if (isNumeric(t) && t.isComplex && !isScalar(t) && t.elem === "double") {
      return [{ kind: "Unknown" }];
    }
    if (isText(t)) {
      return [{ kind: "Unknown" }];
    }
    if (t.kind === "Struct") {
      return [{ kind: "Unknown" }];
    }
    if (isCell(t)) {
      return [{ kind: "Unknown" }];
    }
    throw new TypeError(
      `'disp' arg must be a scalar numeric, a real or complex tensor, text, a struct, or a cell ` +
        `(got ${t.kind})`
    );
  },
  emitC({ argsC, argTypes, useRuntime }) {
    const t = argTypes[0];
    const call = emitDispCallC(t, argsC[0], useRuntime);
    if (call === null) {
      // `transfer` already vetted the kinds disp accepts, so a null
      // here means a new accepted kind was added to `transfer`
      // without a paired emit dispatch — fail loud rather than
      // emitting a stray `0` and confusing the C compiler.
      throw new UnsupportedConstruct(
        `internal: 'disp' has no c-aot emit for type '${t.kind}'`
      );
    }
    return call;
  },
  emitJs({ argsJs, argTypes, useRuntime }) {
    const t = argTypes[0];
    if (isNumeric(t) && t.isComplex && isScalar(t)) {
      useRuntime("mtoc2_disp_complex");
      return `mtoc2_disp_complex(${argsJs[0]})`;
    }
    if (isNumeric(t) && t.isComplex && !isScalar(t)) {
      useRuntime("mtoc2_disp_tensor_complex");
      return `mtoc2_disp_tensor_complex(${argsJs[0]})`;
    }
    if (isNumeric(t) && !t.isComplex && isScalar(t)) {
      useRuntime("mtoc2_disp_double");
      // Scalar logical is a JS boolean; coerce to 0/1 before
      // formatting (MATLAB / numbl disp logical as "0" / "1").
      const arg = t.elem === "logical" ? `(${argsJs[0]} ? 1 : 0)` : argsJs[0];
      return `mtoc2_disp_double(${arg})`;
    }
    if (isNumeric(t) && !t.isComplex && !isScalar(t)) {
      useRuntime("mtoc2_disp_tensor");
      return `mtoc2_disp_tensor(${argsJs[0]})`;
    }
    if (t.kind === "String") {
      // String runtime value is a JS string; print + newline.
      return `($write(${argsJs[0]} + "\\n"))`;
    }
    if (t.kind === "Char") {
      // Char runtime value is `{mtoc2Tag:"char",value}`; print value + newline.
      return `($write(${argsJs[0]}.value + "\\n"))`;
    }
    if (t.kind === "Struct") {
      useRuntime("mtoc2_disp_struct");
      return `mtoc2_disp_struct(${argsJs[0]})`;
    }
    if (isCell(t)) {
      // Format via `mtoc2_format_cell` and print with a trailing
      // newline, matching numbl's `disp(c) = displayValue(c) + '\n'`.
      useRuntime("mtoc2_format_cell");
      return `($write(mtoc2_format_cell(${argsJs[0]}) + "\\n"))`;
    }
    throw new UnsupportedConstruct(
      `'disp' emitJs for arg kind '${t.kind}' is not yet wired`
    );
  },
  call({ args, ctx }) {
    const v = args[0];
    globalThis.$write = ctx.helpers.write;
    if (typeof v === "number") mtoc2_disp_double(v);
    else if (typeof v === "boolean") mtoc2_disp_double(v ? 1 : 0);
    else if (typeof v === "string") ctx.helpers.write(v + "\n");
    else if (isChar(v)) ctx.helpers.write(v.value + "\n");
    else if (isComplexValue(v)) mtoc2_disp_complex(v);
    else if (isTensor(v)) {
      if (v.imag !== undefined) {
        mtoc2_disp_tensor_complex(v);
      } else {
        mtoc2_disp_tensor(v);
      }
    } else if (
      v &&
      typeof v === "object" &&
      (v as { mtoc2Tag?: string }).mtoc2Tag === "cell"
    ) {
      ctx.helpers.write(
        mtoc2_format_cell(v as Parameters<typeof mtoc2_format_cell>[0]) + "\n"
      );
    } else if (v && typeof v === "object") {
      mtoc2_disp_struct(v as Record<string, unknown>);
    } else {
      throw new UnsupportedConstruct(
        `'disp' 'call' got an unsupported value shape (got ${typeof v})`
      );
    }
    return [];
  },
};
