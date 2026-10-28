/**
 * `assert(cond)` / `assert(cond, msg)` / `assert(cond, [id,] fmt, args...)`:
 * runtime + compile-time assertion. A failure raises the interpreter's
 * error (see `assertForm`).
 */

import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import {
  isNumeric,
  isScalar,
  isText,
  typeToString,
  type Type,
} from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import { exactDouble } from "../_shared.js";
import {
  emitFormatSlot,
  emitFormatSlotArray,
  emitTextView,
  validateFormatArgs,
} from "../io/_format_args.js";
import { MATLAB_ID_REGEX } from "../io/error.js";

export const assert: Builtin = {
  name: "assert",
  transfer(argTypes, nargout) {
    if (argTypes.length < 1) {
      throw new TypeError(`'assert' requires at least 1 arg`);
    }
    if (nargout !== 1) {
      throw new UnsupportedConstruct(
        `'assert' does not support multi-output (nargout=${nargout})`
      );
    }
    const cond = argTypes[0];
    if (!isNumeric(cond)) {
      throw new TypeError(
        `'assert' first arg must be numeric (got ${typeToString(cond)})`
      );
    }
    if (cond.isComplex) {
      throw new TypeError(`'assert' on complex cond is not supported`);
    }
    if (!isScalar(cond)) {
      throw new UnsupportedConstruct(
        `'assert' on a tensor cond is not yet supported by the JIT ` +
          `(numbl tests every element non-zero and non-NaN); pass a scalar predicate instead`
      );
    }
    if (argTypes.length >= 2) {
      const m = argTypes[1];
      if (!isText(m)) {
        throw new UnsupportedConstruct(
          `'assert' message must be char or string (got ${typeToString(m)})`
        );
      }
      validateFormatArgs("assert", argTypes, 2);
      if (argTypes.length >= 3 && textExact(m) === undefined) {
        throw new UnsupportedConstruct(
          `'assert' with format arguments needs a literal message (the ` +
            `identifier-vs-format decision is made at compile time)`
        );
      }
    }

    const v = exactDouble(cond);
    if (v !== undefined) {
      const truthy = v !== 0 && !Number.isNaN(v);
      if (!truthy) {
        const m = argTypes.length >= 2 ? argTypes[1] : undefined;
        const msgPart =
          m && (m.kind === "String" || m.kind === "Char")
            ? `: ${m.exact ?? "(opaque)"}`
            : "";
        throw new TypeError(`'assert' is statically false${msgPart}`);
      }
    }

    // assert has no return value; typing it Void lets a bare
    // `assert(...)` statement pass compileSpec's non-void-bare-expr
    // check and JIT (matching fprintf / error / warning).
    return [{ kind: "Void" }];
  },
  emitC({ argsC, argTypes, useRuntime }) {
    const cond = argTypes[0];
    if (isNumeric(cond) && exactDouble(cond) !== undefined) {
      return `((void)0)`;
    }
    const condC = `(double)(${argsC[0]})`;
    const form = assertForm(argTypes);
    const idC = JSON.stringify(form.id);
    if (form.kind === "literal") {
      useRuntime("jit_assert_scalar");
      return `jit_assert_scalar(${condC}, ${idC}, ${JSON.stringify(form.msg)})`;
    }
    useRuntime("jit_assert_scalar_fmt");
    const view = emitTextView(argsC[form.msgIdx], argTypes[form.msgIdx]);
    if (form.kind === "text") {
      return `jit_assert_text(${condC}, ${idC}, ${view})`;
    }
    const slots: string[] = [];
    for (let i = form.msgIdx + 1; i < argTypes.length; i++) {
      slots.push(emitFormatSlot("assert", argsC[i], argTypes[i], i));
    }
    return `jit_assert_scalar_fmt(${condC}, ${idC}, ${view}, ${slots.length}, ${emitFormatSlotArray(slots)})`;
  },
  emitJs({ argsJs, argTypes, useRuntime }) {
    useRuntime("jit_assert_scalar_fmt");
    const cond = argTypes[0];
    if (isNumeric(cond) && exactDouble(cond) !== undefined) {
      return `undefined`;
    }
    const form = assertForm(argTypes);
    const idJs = JSON.stringify(form.id);
    if (form.kind === "literal") {
      return `jit_assert_text(${argsJs[0]}, ${idJs}, ${JSON.stringify(form.msg)})`;
    }
    if (form.kind === "text") {
      return `jit_assert_text(${argsJs[0]}, ${idJs}, ${argsJs[form.msgIdx]})`;
    }
    return `jit_assert_scalar_fmt(${argsJs[0]}, ${idJs}, ${argsJs.slice(form.msgIdx).join(", ")})`;
  },
};

function textExact(t: Type): string | undefined {
  return t.kind === "Char" || t.kind === "String" ? t.exact : undefined;
}

/** The interpreter's (and MATLAB's) `assert` forms:
 *    assert(c)                  "Assertion failed.", MATLAB:assertion:failed
 *    assert(c, msg)             msg taken literally
 *    assert(c, id, fmt, ...)    formatted, identifier id (literal id)
 *    assert(c, fmt, ...)        formatted */
function assertForm(
  argTypes: ReadonlyArray<Type>
):
  | { kind: "literal"; id: string; msg: string }
  | { kind: "text" | "fmt"; id: string; msgIdx: number } {
  if (argTypes.length === 1) {
    return {
      kind: "literal",
      id: "MATLAB:assertion:failed",
      msg: "Assertion failed.",
    };
  }
  if (argTypes.length === 2) {
    const exact = textExact(argTypes[1]);
    if (exact !== undefined) return { kind: "literal", id: "", msg: exact };
    return { kind: "text", id: "", msgIdx: 1 };
  }
  const first = textExact(argTypes[1]);
  if (first !== undefined && MATLAB_ID_REGEX.test(first)) {
    return { kind: "fmt", id: first, msgIdx: 2 };
  }
  return { kind: "fmt", id: "", msgIdx: 1 };
}
