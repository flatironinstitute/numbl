/**
 * `strcmp(a, b)` / `strcmpi(a, b)` — text equality.
 *
 * Numbl semantics (`numbl-core/interpreter/builtins/strings.ts`):
 *   - Both args text (Char / String) → byte-equality (or ASCII
 *     case-folded equality for `strcmpi`). Length mismatch is not an
 *     error; the inputs simply compare unequal.
 *   - Either arg non-text → `false`. Numbl gates the comparison
 *     behind `isText(a) && isText(b)`; non-text just yields `0`
 *     silently. We mirror that to keep cross-runner output identical.
 *   - Cell-of-text vectorization (`strcmp({'a','b'}, 'a')`) is in
 *     numbl but cells aren't yet a feature class in mtoc2 — that
 *     overload will fall out naturally when cells land.
 *
 * Folds to a literal when both args are exact text. Otherwise emits
 * the runtime `mtoc2_strcmp` / `mtoc2_strcmpi` helper which compares
 * byte-for-byte on text views.
 */

import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import { isText, scalarLogical, type Type } from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import { emitTextView } from "../io/_format_args.js";
import {
  mtoc2_strcmp as jsStrcmp,
  mtoc2_strcmpi as jsStrcmpi,
} from "../../runtime/snippets.gen.js";
import { isChar as isRuntimeChar } from "../../../runtime/value.js";
import type { RuntimeValue } from "../../../runtime/value.js";

type CmpMode = "exact" | "case-fold";

function staticAnswer(a: Type, b: Type, mode: CmpMode): boolean | undefined {
  if (!isText(a) || !isText(b)) return false;
  if (a.kind !== "Char" && a.kind !== "String") return undefined;
  if (b.kind !== "Char" && b.kind !== "String") return undefined;
  if (a.exact === undefined || b.exact === undefined) return undefined;
  if (mode === "case-fold") {
    return a.exact.toLowerCase() === b.exact.toLowerCase();
  }
  return a.exact === b.exact;
}

function checkArity(name: string, argTypes: Type[], nargout: number): void {
  if (argTypes.length !== 2) {
    throw new TypeError(`'${name}' expects 2 arg(s), got ${argTypes.length}`);
  }
  if (nargout !== 1) {
    throw new UnsupportedConstruct(
      `'${name}' does not support multi-output (nargout=${nargout})`
    );
  }
}

function runtimeText(v: RuntimeValue): string | null {
  if (typeof v === "string") return v;
  if (isRuntimeChar(v)) return v.value;
  return null;
}

function makeBuiltin(name: string, mode: CmpMode, helperName: string): Builtin {
  const jsFn = mode === "case-fold" ? jsStrcmpi : jsStrcmp;
  return {
    name,
    transfer(argTypes, nargout) {
      checkArity(name, argTypes, nargout);
      const v = staticAnswer(argTypes[0], argTypes[1], mode);
      return [v === undefined ? scalarLogical() : scalarLogical(v)];
    },
    emitC({ argsC, argTypes, useRuntime }) {
      const v = staticAnswer(argTypes[0], argTypes[1], mode);
      if (v !== undefined) return v ? `1.0` : `0.0`;
      useRuntime(helperName);
      const a = emitTextView(argsC[0], argTypes[0]);
      const b = emitTextView(argsC[1], argTypes[1]);
      return `${helperName}(${a}, ${b})`;
    },
    emitJs({ argsJs, argTypes, useRuntime }) {
      const v = staticAnswer(argTypes[0], argTypes[1], mode);
      if (v !== undefined) return v ? `true` : `false`;
      useRuntime(helperName);
      // Helper returns 0/1 (matches C signature). Wrap to bool so
      // the spec output round-trips as a JS boolean.
      return `(${helperName}(${argsJs[0]}, ${argsJs[1]}) !== 0)`;
    },
    call({ args }) {
      const a = runtimeText(args[0]);
      const b = runtimeText(args[1]);
      if (a === null || b === null) return [false];
      return [(jsFn(a, b) as number) !== 0];
    },
  };
}

export const strcmp = makeBuiltin("strcmp", "exact", "mtoc2_strcmp");
export const strcmpi = makeBuiltin("strcmpi", "case-fold", "mtoc2_strcmpi");
