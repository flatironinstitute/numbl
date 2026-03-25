/**
 * Validation builtins: mustBe* family.
 */

import type { RuntimeValue } from "../../runtime/types.js";
import {
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeTensor,
  isRuntimeCell,
  isRuntimeChar,
  isRuntimeString,
} from "../../runtime/types.js";
import { toNumber, RuntimeError } from "../../runtime/index.js";
import { defineBuiltin } from "./types.js";
import type { JitType } from "../jit/jitTypes.js";

function isNumeric(v: RuntimeValue): boolean {
  return isRuntimeNumber(v) || isRuntimeTensor(v) || isRuntimeLogical(v);
}

function numericElements(v: RuntimeValue): number[] {
  if (isRuntimeNumber(v)) return [v as number];
  if (isRuntimeLogical(v)) return [v ? 1 : 0];
  if (isRuntimeTensor(v)) return Array.from(v.data);
  return [];
}

function numelVal(v: RuntimeValue): number {
  if (isRuntimeNumber(v) || isRuntimeLogical(v)) return 1;
  if (isRuntimeTensor(v)) return v.data.length;
  if (isRuntimeChar(v)) return v.value.length;
  if (isRuntimeString(v)) return 1;
  if (isRuntimeCell(v)) return v.data.length;
  return 0;
}

function validatorMatch(argTypes: JitType[]): JitType[] | null {
  if (argTypes.length < 1) return null;
  return [{ kind: "number" }];
}

function makeElemValidator(
  name: string,
  predicate: (n: number) => boolean,
  message: string
): void {
  defineBuiltin({
    name,
    cases: [
      {
        match: validatorMatch,
        apply: args => {
          const v = args[0];
          if (!v) throw new RuntimeError(`${name}: missing argument`);
          if (!isNumeric(v)) throw new RuntimeError(message);
          for (const n of numericElements(v)) {
            if (!predicate(n)) throw new RuntimeError(message);
          }
          return 0;
        },
      },
    ],
  });
}

defineBuiltin({
  name: "mustBeNumeric",
  cases: [
    {
      match: validatorMatch,
      apply: args => {
        const v = args[0];
        if (!v) throw new RuntimeError("mustBeNumeric: missing argument");
        if (!isNumeric(v)) throw new RuntimeError("Value must be numeric.");
        return 0;
      },
    },
  ],
});

makeElemValidator(
  "mustBeInteger",
  n => Number.isFinite(n) && Math.floor(n) === n,
  "Value must be integer."
);
makeElemValidator("mustBePositive", n => n > 0, "Value must be positive.");
makeElemValidator(
  "mustBeNonnegative",
  n => n >= 0,
  "Value must be nonnegative."
);
makeElemValidator("mustBeNonzero", n => n !== 0, "Value must be nonzero.");
makeElemValidator(
  "mustBeFinite",
  n => Number.isFinite(n),
  "Value must be finite."
);

defineBuiltin({
  name: "mustBeNonempty",
  cases: [
    {
      match: validatorMatch,
      apply: args => {
        const v = args[0];
        if (!v) throw new RuntimeError("mustBeNonempty: missing argument");
        if (numelVal(v) === 0)
          throw new RuntimeError("Value must be nonempty.");
        return 0;
      },
    },
  ],
});

defineBuiltin({
  name: "mustBeScalarOrEmpty",
  cases: [
    {
      match: validatorMatch,
      apply: args => {
        const v = args[0];
        if (!v) throw new RuntimeError("mustBeScalarOrEmpty: missing argument");
        const n = numelVal(v);
        if (n !== 0 && n !== 1)
          throw new RuntimeError("Value must be scalar or empty.");
        return 0;
      },
    },
  ],
});

defineBuiltin({
  name: "mustBeVector",
  cases: [
    {
      match: validatorMatch,
      apply: args => {
        const v = args[0];
        if (!v) throw new RuntimeError("mustBeVector: missing argument");
        if (isRuntimeTensor(v)) {
          const isVec =
            v.shape.length <= 2 &&
            (v.shape[0] === 1 || v.shape[1] === 1 || v.data.length === 0);
          if (!isVec) throw new RuntimeError("Value must be a vector.");
        } else if (!isRuntimeNumber(v) && !isRuntimeLogical(v)) {
          throw new RuntimeError("Value must be a vector.");
        }
        return 0;
      },
    },
  ],
});

defineBuiltin({
  name: "mustBeMember",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length < 2) return null;
        return [{ kind: "number" }];
      },
      apply: args => {
        const v = args[0];
        const set = args[1];
        if (!v || !set)
          throw new RuntimeError("mustBeMember: requires 2 arguments");
        const vElems = isRuntimeString(v) ? [v] : numericElements(v);
        const setElems = isRuntimeCell(set)
          ? set.data
          : isRuntimeString(set)
            ? [set]
            : numericElements(set);
        for (const elem of vElems) {
          const found = (setElems as unknown[]).some(s =>
            typeof s === "object" &&
            s !== null &&
            "kind" in s &&
            isRuntimeString(s as RuntimeValue)
              ? (s as { kind: string; value: string }).value === String(elem)
              : s === elem
          );
          if (!found)
            throw new RuntimeError(
              "Value must be a member of the specified set."
            );
        }
        return 0;
      },
    },
  ],
});

defineBuiltin({
  name: "mustBeInRange",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length < 1) return null;
        return [{ kind: "number" }];
      },
      apply: args => {
        const v = args[0];
        const lower = args[1] ? toNumber(args[1]) : -Infinity;
        const upper = args[2] ? toNumber(args[2]) : Infinity;
        if (!v) throw new RuntimeError("mustBeInRange: missing argument");
        if (!isNumeric(v)) throw new RuntimeError("Value must be in range.");
        for (const n of numericElements(v)) {
          if (n < lower || n > upper) {
            throw new RuntimeError(
              `Value must be in range [${lower}, ${upper}].`
            );
          }
        }
        return 0;
      },
    },
  ],
});
