/**
 * Zero-arg numeric constants: `pi`, `eps`, `Inf`/`inf`, `NaN`/`nan`.
 *
 * MATLAB resolves these as function names. Numbl-equivalent: the bare-
 * name read goes through `lowerIdent`, which forwards to the matching
 * 0-arity builtin when the user hasn't shadowed it. A `pi()` paren-form
 * call works for free (standard builtin-call path).
 *
 * `e` is intentionally absent — MATLAB and numbl don't define it
 * either (use `exp(1)`).
 *
 * `nan` / `NaN` / `Inf` / `inf` also accept the MATLAB shape-constructor
 * form (`nan(3)`, `Inf(2, 3)`, `nan(m, n, p)`) — see `fillConstBuiltin`
 * below. `pi` / `eps` stay strictly 0-arity; passing args errors.
 */
import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import { scalarDouble } from "../../../lowering/types.js";
import {
  type Builtin,
  requireEmitC,
  requireEmitJs,
  requireCall,
} from "../../registry.js";
import { defineShapeConstructor } from "../shape/_construct.js";

function constBuiltin(
  name: string,
  value: number,
  sign: "positive" | "unknown",
  cLiteral: string
): Builtin {
  // JS literal form. `Infinity` and `NaN` are JS keywords with the same
  // numeric value as their C counterparts; the C literal happens to be
  // the same for most numeric constants except Inf/NaN where C uses
  // <math.h> macros.
  const jsLiteral = Number.isFinite(value)
    ? String(value)
    : value > 0
      ? "Infinity"
      : value < 0
        ? "-Infinity"
        : "NaN";
  return {
    name,
    transfer(argTypes, nargout) {
      if (argTypes.length !== 0) {
        throw new TypeError(
          `'${name}' expects 0 arg(s), got ${argTypes.length}`
        );
      }
      if (nargout !== 1) {
        throw new UnsupportedConstruct(
          `'${name}' does not support multi-output (nargout=${nargout})`
        );
      }
      return [scalarDouble(sign, value)];
    },
    emitC() {
      return cLiteral;
    },
    emitJs() {
      return jsLiteral;
    },
    call() {
      return [value];
    },
    // Per-slot is the same literal — no loop-dependent context. Marking
    // elementwise lets the fused emitter accept Assigns whose RHS contains
    // `pi`, `eps`, `Inf`, `NaN`.
    elementwise: true,
  };
}

/** Build a "constant + shape constructor" builtin for `nan` / `NaN` /
 *  `Inf` / `inf`. The 0-arg call is the scalar constant (same as
 *  `constBuiltin`); the >=1-arg call routes through the shared shape
 *  constructor with `cLiteral` as the C-side fill value. */
function fillConstBuiltin(
  name: string,
  value: number,
  sign: "positive" | "unknown",
  cLiteral: string
): Builtin {
  const shape = defineShapeConstructor(
    name,
    value,
    "mtoc2_tensor_fill_nd",
    "mtoc2_tensor_fill_square",
    { minArgs: 0, cFillValue: cLiteral }
  );
  return {
    name,
    transfer(argTypes, nargout) {
      if (argTypes.length === 0) {
        if (nargout !== 1) {
          throw new UnsupportedConstruct(
            `'${name}' does not support multi-output (nargout=${nargout})`
          );
        }
        return [scalarDouble(sign, value)];
      }
      return shape.transfer(argTypes, nargout);
    },
    emitC(args) {
      if (args.argTypes.length === 0) return cLiteral;
      return requireEmitC(shape)(args);
    },
    emitJs(args) {
      if (args.argTypes.length === 0) {
        return Number.isFinite(value)
          ? String(value)
          : value > 0
            ? "Infinity"
            : "NaN";
      }
      return requireEmitJs(shape)(args);
    },
    call(args) {
      if (args.argTypes.length === 0) return [value];
      return requireCall(shape)(args);
    },
    // Per-slot is only invoked by the fused emitter for elementwise ops,
    // where the constant appears as a scalar slot. The shape-constructor
    // (>=1 args) is a tensor producer — A-normalization hoists it out
    // before the fused emitter sees the surrounding expression, so the
    // 0-arg branch is the only one fusion ever asks for.
    elementwise: true,
  };
}

export const pi = constBuiltin("pi", Math.PI, "positive", "3.141592653589793");
export const eps = constBuiltin(
  "eps",
  Number.EPSILON,
  "positive",
  "2.220446049250313e-16"
);

/* `INFINITY` and `NAN` come from <math.h> (in BASE_HEADERS). Using the
 * macros (rather than `1.0/0.0` etc.) lets the compiler avoid div-by-
 * zero warnings and matches what numbl prints via its formatter. */
export const Inf = fillConstBuiltin("Inf", Infinity, "positive", "INFINITY");
export const inf = fillConstBuiltin("inf", Infinity, "positive", "INFINITY");
export const NaNBuiltin = fillConstBuiltin("NaN", NaN, "unknown", "NAN");
export const nan = fillConstBuiltin("nan", NaN, "unknown", "NAN");
