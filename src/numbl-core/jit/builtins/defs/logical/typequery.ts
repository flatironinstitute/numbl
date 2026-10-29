/**
 * Static type / shape query builtins, each returning a logical (or, for
 * `ndims`, a double) scalar. The JIT always knows an argument's rank and
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

// ── kind predicates (always static) ──────────────────────────────────

/** Build a predicate that depends only on the argument's kind, never
 *  its value — so it folds to a literal in every backend. */
function kindPredicate(name: string, ofType: (t: Type) => boolean): Builtin {
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
  };
}

export const islogical = kindPredicate(
  "islogical",
  t => isNumeric(t) && t.elem === "logical"
);

// `isnumeric` is defined in its own file (`isnumeric.ts`).

export const isfloat = kindPredicate(
  "isfloat",
  // Floating-point: only double (the JIT has no single-precision lane).
  t => isNumeric(t) && t.elem === "double"
);

export const ischarBuiltin = kindPredicate("ischar", t => isChar(t));

export const isstring = kindPredicate("isstring", t => isString(t));

export const isinteger = kindPredicate("isinteger", () => false);

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
};

// ── isrow / iscolumn (fold when known, else runtime) ─────────────────

/** Build `isrow` / `iscolumn`: a 2-D value whose `axis`-th dim is 1.
 *  Scalars are both. Tensors with a statically-1 axis fold; otherwise
 *  the check reads the tensor's runtime `ndim` / `dims`. */
function rowColPredicate(name: string, axis: 0 | 1): Builtin {
  const otherDescribed = (t: Type): boolean | "runtime" => {
    // A string and a struct reach the JIT only as 1x1 values; a char is a
    // row, of a length known only when it is exact.
    if (isString(t) || t.kind === "Struct") return true;
    if (isChar(t)) {
      if (axis === 0) return true;
      if (t.exact !== undefined) return t.exact.length === 1;
      throw new UnsupportedConstruct(`'${name}' of a char of unknown length`);
    }
    if (!isNumeric(t)) return false;
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
  };
}

export const isrow = rowColPredicate("isrow", 0);
export const iscolumn = rowColPredicate("iscolumn", 1);
