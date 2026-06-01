/**
 * Static type / shape query builtins, each returning a logical (or, for
 * `ndims`, a double) scalar. mtoc2 always knows an argument's rank and
 * element kind at type time, so the kind predicates and `ndims` fold to
 * a literal. `isrow` / `iscolumn` fold when the relevant dim is known
 * and otherwise emit a runtime check against the tensor's `ndim` /
 * `dims`.
 *
 * Mirrors numbl's introspection builtins
 * (interpreter/builtins/introspection.ts).
 */
import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import {
  scalarDouble,
  scalarLogical,
  signFromNumber,
  isNumeric,
  isChar,
  isString,
  isDimOne,
  type Type,
} from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import {
  isTensor,
  isChar as isRtChar,
  isComplexValue,
  type RuntimeValue,
} from "../../../runtime/value.js";

function checkUnary(name: string, argTypes: Type[], nargout: number): void {
  if (argTypes.length !== 1) {
    throw new TypeError(`'${name}' expects 1 arg(s), got ${argTypes.length}`);
  }
  if (nargout > 1) {
    throw new UnsupportedConstruct(
      `'${name}' does not support multi-output (nargout=${nargout})`
    );
  }
}

/** Runtime shape of an interpreter value (numbl's getShape rule). */
function rtShape(v: RuntimeValue): number[] {
  if (isTensor(v)) return v.shape;
  if (isRtChar(v)) return [1, v.value.length];
  return [1, 1]; // scalar number / boolean / complex / string / struct
}

// ── kind predicates (always static) ──────────────────────────────────

/** Build a predicate that depends only on the argument's kind, never
 *  its value — so it folds to a literal in every backend. */
function kindPredicate(
  name: string,
  ofType: (t: Type) => boolean,
  ofValue: (v: RuntimeValue) => boolean
): Builtin {
  return {
    name,
    transfer(argTypes, nargout) {
      checkUnary(name, argTypes, nargout);
      return [scalarLogical(ofType(argTypes[0]))];
    },
    emitC({ argTypes }) {
      return ofType(argTypes[0]) ? "1.0" : "0.0";
    },
    emitJs({ argTypes }) {
      return ofType(argTypes[0]) ? "true" : "false";
    },
    call({ args }) {
      return [ofValue(args[0])];
    },
  };
}

// `islogical` / `isnumeric` / `isfloat`: scalar logical is a JS
// `boolean` at runtime, and logical tensors carry an `isLogical: true`
// flag — so we can tell them apart from doubles. Mirrors numbl.

export const islogical = kindPredicate(
  "islogical",
  t => isNumeric(t) && t.elem === "logical",
  v => {
    if (typeof v === "boolean") return true;
    if (isTensor(v) && v.isLogical) return true;
    return false;
  }
);

// `isnumeric` is defined in its own file (`isnumeric.ts`).

export const isfloat = kindPredicate(
  "isfloat",
  // Floating-point: only double (mtoc2 has no single-precision lane).
  t => isNumeric(t) && t.elem === "double",
  v => {
    if (typeof v === "number") return true;
    if (typeof v === "boolean") return false;
    if (isComplexValue(v)) return true;
    if (isTensor(v)) return !v.isLogical;
    return false;
  }
);

export const ischarBuiltin = kindPredicate(
  "ischar",
  t => isChar(t),
  v => isRtChar(v)
);

export const isstring = kindPredicate(
  "isstring",
  t => isString(t),
  v => typeof v === "string"
);

export const isinteger = kindPredicate(
  "isinteger",
  () => false,
  () => false
);

// ── rank queries (static from the dims lattice) ──────────────────────

function typeRank(t: Type): number {
  return isNumeric(t) ? t.dims.length : 2;
}

export const ndims: Builtin = {
  name: "ndims",
  transfer(argTypes, nargout) {
    checkUnary("ndims", argTypes, nargout);
    const r = Math.max(2, typeRank(argTypes[0]));
    return [scalarDouble(signFromNumber(r), r)];
  },
  emitC({ argTypes }) {
    return `${Math.max(2, typeRank(argTypes[0])).toFixed(1)}`;
  },
  emitJs({ argTypes }) {
    return `${Math.max(2, typeRank(argTypes[0]))}`;
  },
  call({ args }) {
    return [Math.max(2, rtShape(args[0]).length)];
  },
};

export const ismatrix: Builtin = {
  name: "ismatrix",
  transfer(argTypes, nargout) {
    checkUnary("ismatrix", argTypes, nargout);
    return [scalarLogical(typeRank(argTypes[0]) <= 2)];
  },
  emitC({ argTypes }) {
    return typeRank(argTypes[0]) <= 2 ? "1.0" : "0.0";
  },
  emitJs({ argTypes }) {
    return typeRank(argTypes[0]) <= 2 ? "true" : "false";
  },
  call({ args }) {
    return [rtShape(args[0]).length <= 2];
  },
};

// ── isrow / iscolumn (fold when known, else runtime) ─────────────────

/** Build `isrow` / `iscolumn`: a 2-D value whose `axis`-th dim is 1.
 *  Scalars are both. Tensors with a statically-1 axis fold; otherwise
 *  the check reads the tensor's runtime `ndim` / `dims`. */
function rowColPredicate(name: string, axis: 0 | 1): Builtin {
  const otherDescribed = (t: Type): boolean | "runtime" => {
    if (!isNumeric(t)) return false; // char/string/struct → not a row/col vector here
    const dims = t.dims;
    if (dims.length !== 2) return false;
    if (isDimOne(dims[axis])) return true;
    // The axis dim is statically known and not 1 → definitely false.
    if (dims[axis].kind === "exact") return false;
    return "runtime";
  };
  return {
    name,
    transfer(argTypes, nargout) {
      checkUnary(name, argTypes, nargout);
      const v = otherDescribed(argTypes[0]);
      if (v === "runtime") return [scalarLogical()];
      return [scalarLogical(v)];
    },
    emitC({ argTypes, argsC }) {
      const v = otherDescribed(argTypes[0]);
      if (v === true) return "1.0";
      if (v === false) return "0.0";
      return `((${argsC[0]}).ndim == 2 && (${argsC[0]}).dims[${axis}] == 1)`;
    },
    emitJs({ argTypes, argsJs }) {
      const v = otherDescribed(argTypes[0]);
      if (v === true) return "true";
      if (v === false) return "false";
      return `(${argsJs[0]}.shape.length === 2 && ${argsJs[0]}.shape[${axis}] === 1)`;
    },
    call({ args }) {
      const s = rtShape(args[0]);
      return [s.length === 2 && s[axis] === 1];
    },
  };
}

export const isrow = rowColPredicate("isrow", 0);
export const iscolumn = rowColPredicate("iscolumn", 1);
