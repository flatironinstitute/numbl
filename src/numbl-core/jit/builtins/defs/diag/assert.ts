/**
 * `assert(cond)` / `assert(cond, msg)` / `assert(cond, fmt, args...)` —
 * runtime + compile-time assertion.
 */

import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import {
  isNumeric,
  isScalar,
  isText,
  typeToString,
} from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import { exactDouble } from "../_shared.js";
import {
  emitFormatSlot,
  emitFormatSlotArray,
  emitTextView,
  validateFormatArgs,
} from "../io/_format_args.js";

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
    useRuntime("mtoc2_assert_scalar");
    useRuntime("mtoc2_assert_scalar_fmt");
    const cond = argTypes[0];
    if (isNumeric(cond) && exactDouble(cond) !== undefined) {
      return `((void)0)`;
    }
    const condC = `(double)(${argsC[0]})`;
    if (argTypes.length < 2) {
      return `mtoc2_assert_scalar(${condC}, (const char *)0)`;
    }
    if (argTypes.length === 2) {
      const m = argTypes[1];
      if ((m.kind === "Char" || m.kind === "String") && m.exact !== undefined) {
        return `mtoc2_assert_scalar(${condC}, ${JSON.stringify(m.exact)})`;
      }
    }
    const fmtView = emitTextView(argsC[1], argTypes[1]);
    const slots: string[] = [];
    for (let i = 2; i < argTypes.length; i++) {
      slots.push(emitFormatSlot("assert", argsC[i], argTypes[i], i));
    }
    return `mtoc2_assert_scalar_fmt(${condC}, ${fmtView}, ${slots.length}, ${emitFormatSlotArray(slots)})`;
  },
  emitJs({ argsJs, argTypes, useRuntime }) {
    useRuntime("mtoc2_assert_scalar_fmt");
    const cond = argTypes[0];
    if (isNumeric(cond) && exactDouble(cond) !== undefined) {
      return `undefined`;
    }
    if (argTypes.length < 2) {
      return `mtoc2_assert_scalar_fmt(${argsJs[0]}, "assertion failed")`;
    }
    return `mtoc2_assert_scalar_fmt(${argsJs.join(", ")})`;
  },
};
