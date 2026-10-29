/**
 * Binary scalar builtins: atan2, min, max, mod, rem, power.
 */

import {
  isRuntimeChar,
  isRuntimeComplexNumber,
  isRuntimeString,
  isRuntimeTensor,
  type RuntimeValue,
} from "../../runtime/types.js";
import { RuntimeError } from "../../runtime/error.js";
import { RTV } from "../../runtime/constructors.js";
import { minMaxImpl } from "../../helpers/reduction/min-max.js";
import { binop } from "../../runtime/runtimeOperators.js";
import { BinaryOperation } from "../../parser/index.js";
import { builtin } from "./builtin.js";
import { builtinError, mustBeRealError, numericArg } from "./args.js";
import { realDenseArg, realFloatArg } from "./math-args.js";
import { applyBinaryElemwise } from "./types.js";
import { OpBin } from "../../ops/index.js";

// ── atan2 ────────────────────────────────────────────────────────────────

builtin({
  name: "atan2",
  singleOutputs: true,
  nargin: 2,
  apply: args =>
    applyBinaryElemwise(
      [
        realFloatArg(args[0], mustBeRealError),
        realFloatArg(args[1], mustBeRealError),
      ],
      OpBin.ATAN2
    ),
});

// ── min / max ───────────────────────────────────────────────────────────

const isText = (v: RuntimeValue) => isRuntimeChar(v) || isRuntimeString(v);

function minMaxBuiltin(mode: "min" | "max"): void {
  const initVal = mode === "min" ? Infinity : -Infinity;
  const cmpFn =
    mode === "min"
      ? (a: number, b: number) => a < b
      : (a: number, b: number) => a > b;
  const impl = (args: RuntimeValue[], nargout: number) =>
    minMaxImpl(mode, args, nargout, initVal, cmpFn);

  // Logical values stay logical (min(true, false) is logical); char
  // becomes its character codes.
  const operand = (v: RuntimeValue, first: boolean): RuntimeValue => {
    if (typeof v === "boolean") return v;
    if (
      first &&
      !(
        typeof v === "number" ||
        isRuntimeTensor(v) ||
        isRuntimeComplexNumber(v) ||
        isRuntimeChar(v)
      )
    )
      throw builtinError(
        `MATLAB:${mode}:wrongInput`,
        "First input array is an invalid data type."
      );
    return numericArg(v);
  };

  builtin({
    name: mode,
    singleOutputs: [1],
    // min(A), min(A, B), min(A, [], dim), min(A, [], dim, flag)
    nargin: [1, 4],
    nargout: 2,
    apply: (args, nargout) => {
      const a = operand(args[0], true);
      if (args.length === 1) return impl([a], nargout);
      if (args.length === 2) {
        if (isText(args[1]))
          throw builtinError(
            `MATLAB:${mode}:wrongSecondInputText`,
            `Invalid second argument. To specify an additional argument, use ${mode}(A,[],...).`
          );
        if (nargout > 1)
          throw builtinError(
            `MATLAB:${mode}:TwoInTwoOutCaseNotSupported`,
            "Two output arguments are not supported for two input arrays."
          );
        return impl([a, operand(args[1], false)], nargout);
      }
      const b = args[1];
      if (!(isRuntimeTensor(b) && b.data.length === 0))
        throw builtinError(
          `MATLAB:${mode}:caseNotSupported`,
          "Specifying the operating dimension is not supported for two input arrays."
        );
      // Trailing flags. 'omitnan' is the default.
      const rest = args.slice(2);
      while (rest.length > 0 && isText(rest[rest.length - 1])) {
        const flag = String(
          isRuntimeChar(rest[rest.length - 1])
            ? (rest[rest.length - 1] as { value: string }).value
            : rest[rest.length - 1]
        ).toLowerCase();
        if (flag === "all") break;
        if (flag !== "omitnan")
          throw new RuntimeError(
            `${mode}: the '${flag}' option is not supported`
          );
        rest.pop();
      }
      if (rest.length > 1) throw new RuntimeError(`${mode}: invalid arguments`);
      if (rest.length === 0) return impl([a], nargout);
      if (isRuntimeComplexNumber(a)) return nargout > 1 ? [a, 1] : a;
      return impl(
        [a, RTV.tensor(new Float64Array(0), [0, 0]), rest[0]],
        nargout
      );
    },
  });
}

minMaxBuiltin("min");
minMaxBuiltin("max");

// ── mod / rem ────────────────────────────────────────────────────────────

function realBinary(name: string, op: number): void {
  builtin({
    name,
    singleOutputs: true,
    nargin: 2,
    apply: args =>
      applyBinaryElemwise(
        [
          realDenseArg(args[0], mustBeRealError),
          realDenseArg(args[1], mustBeRealError),
        ],
        op
      ),
  });
}

realBinary("mod", OpBin.MOD);
realBinary("rem", OpBin.REM);

// ── power ────────────────────────────────────────────────────────────────
// power(a, b) is the functional form of a.^b, including complex results
// for negative bases with fractional exponents.

builtin({
  name: "power",
  nargin: 2,
  apply: args => {
    numericArg(args[0]);
    numericArg(args[1]);
    return binop(BinaryOperation.ElemPow, args[0], args[1]) as RuntimeValue;
  },
});
