/**
 * Case tables for the JIT builtin conformance test
 * (`jit-builtin-conformance.test.ts`). Every name in the JIT builtin
 * registry needs a table here, or an `ALLOWLIST` entry saying why it
 * cannot be checked this way.
 *
 * Element-wise builtins get generated tables (`familyCases`): the
 * registry's `elementwise` flag plus a probe of `transfer` picks the
 * nullary (constant), unary and binary families, each a fixed sweep over
 * the shared input pool. `TABLES` adds the cases specific to one builtin,
 * and holds the whole table of every other builtin.
 */

import type { Builtin } from "../../numbl-core/jit/builtins/registry.js";
import { plotBuiltins } from "../../numbl-core/jit/builtins/defs/plot/dispatch.js";
import { scalarDouble } from "../../numbl-core/jit/lowering/types.js";
import type { Arg, Case, DivergenceKind } from "./harness.js";

// ── Input pool ─────────────────────────────────────────────────────────

/** Named MATLAB expressions for inputs. */
export const V = {
  pos: "2.5",
  neg: "-1.5",
  zero: "0",
  nzero: "-0",
  nan: "NaN",
  inf: "Inf",
  ninf: "-Inf",
  int: "3",
  row: "[1 -2 3.5 0 -0 NaN Inf -Inf]",
  row3: "[3 -1 2]",
  col: "[0.5; -3; 2]",
  mat: "[1 -2 3; 4 NaN -6; -0 8 Inf; 0.25 -1 2]",
  mat2: "[2 2 -1; 0.5 3 NaN; 1 -0 4; -7 1 0]",
  sq: "[2 -1; 0.5 3]",
  t3: "reshape([1 -2 3 0.5 -0 6 NaN 8 -9 10 11 -12], 2, 3, 2)",
  e00: "[]",
  e03: "zeros(0, 3)",
  e10: "zeros(1, 0)",
  cs: "complex(1.5, -2)",
  cz: "complex(2, 0)",
  crow: "[1+2i, -3i, 4, NaN+1i, -2-0.5i]",
  cmat: "[1+1i 2; -3i 4-0.5i]",
  ls: "true",
  lrow: "[true false true]",
  lmat: "logical([1 0; 0 1; 1 1])",
  chr: "'abc'",
  str: '"abc"',
  st: "struct('a', 1, 'b', [1 2])",
} as const;

/** Every element value pair of `SPECIALS` as two 1x100 rows. */
const SPECIALS = "[-2.5 -1 -0 0 0.5 1 3 NaN Inf -Inf]";
const PAIR_A = `repelem(${SPECIALS}, 10)`;
const PAIR_B = `repmat(${SPECIALS}, 1, 10)`;

const SCALARS = [V.pos, V.neg, V.zero, V.nzero, V.nan, V.inf, V.ninf];
const TENSORS = [V.row, V.col, V.mat, V.t3, V.e00, V.e03, V.e10];
const COMPLEX = [V.cs, V.cz, V.crow, V.cmat];
const LOGICAL = [V.ls, V.lrow, V.lmat];
/** Numeric inputs of every kind. */
const NUMERIC = [...SCALARS, ...TENSORS, ...COMPLEX, ...LOGICAL];
/** Everything, for type queries. */
const ANY = [...NUMERIC, V.chr, V.str, V.st];

const lit = (s: string): Arg => ({ lit: s });
/** One case per input, each called as `name(x, ...rest)`. */
const each = (xs: readonly Arg[], ...rest: Arg[]): Case[] =>
  xs.map(x => ({ args: [x, ...rest] }));
/** A statement-form case: run `stmt`, return 0. */
const stmt = (body: string, ...args: Arg[]): Case => ({
  args,
  body: `${body}; y1 = 0;`,
});

// ── Generated families ─────────────────────────────────────────────────

const UNARY: Case[] = [
  ...each(NUMERIC),
  // Exact (folded) scalars.
  ...SCALARS.map(s => ({ args: [lit(s)] })),
  { args: [lit("-4")] },
  { args: [lit(V.cs)] },
];

const BINARY_PAIRS: [string, string][] = [
  [V.pos, V.neg],
  [V.nzero, V.zero],
  [V.neg, V.nzero],
  [V.nan, V.pos],
  [V.inf, V.ninf],
  [V.zero, V.zero],
  [V.inf, V.inf],
  [V.int, V.neg],
  [PAIR_A, PAIR_B],
  [V.row, V.pos],
  [V.neg, V.row],
  [V.mat, V.mat2],
  [V.col, V.row3],
  [V.mat, V.row3],
  [V.mat, "[1; -2; 0; 3]"],
  [V.t3, V.row3],
  [V.t3, "[2; -1]"],
  [V.e03, V.row3],
  [V.e00, V.pos],
  [V.e00, V.e00],
  [V.cs, V.pos],
  [V.neg, V.cs],
  [V.crow, V.pos],
  [V.crow, "[2, 1i, -1, 0, 3]"],
  [V.cs, V.crow],
  [V.cmat, V.sq],
  [V.cz, V.neg],
  [V.lrow, "[false false true]"],
  [V.lrow, V.pos],
  [V.ls, V.lrow],
  // Incompatible sizes: an error in every mode.
  [V.row3, "[1 2]"],
];

const BINARY: Case[] = [
  ...BINARY_PAIRS.map(([a, b]) => ({ args: [a, b] })),
  // Exact operands.
  { args: [lit(V.pos), lit(V.neg)] },
  { args: [lit("-8"), lit("1/3")] },
  { args: [lit(V.nzero), lit(V.zero)] },
  { args: [V.row, lit("2")] },
  { args: [lit("2"), V.mat] },
];

const NULLARY: Case[] = [
  { args: [] },
  { args: [lit("3")] },
  { args: [lit("2"), lit("3")] },
  { args: [lit("2"), lit("3"), lit("2")] },
  { args: [lit("0"), lit("3")] },
  { args: ["[2 3]"] },
  { args: [V.int] },
  { args: [V.int, lit("2")] },
];

const probe = (b: Builtin, args: ReturnType<typeof scalarDouble>[]) => {
  try {
    b.transfer(args, 1);
    return true;
  } catch {
    return false;
  }
};

/** The generated table of an element-wise builtin: one sweep per arity
 *  its `transfer` accepts. */
export function familyCases(b: Builtin): Case[] {
  if (!b.elementwise) return [];
  const s = scalarDouble("positive");
  const out: Case[] = [];
  // A constant's arguments are sizes (`NaN(2, 3)`), so it gets only the
  // nullary sweep.
  if (probe(b, [])) return NULLARY;
  if (probe(b, [s])) out.push(...UNARY);
  if (probe(b, [s, s])) out.push(...BINARY);
  if (out.length === 0) throw new Error(`no arity probe fits '${b.name}'`);
  return out;
}

// ── Shared non-element-wise tables ─────────────────────────────────────

const REDUCE_IN = [
  V.pos,
  V.nan,
  V.row,
  V.col,
  V.mat,
  V.t3,
  V.e00,
  V.e03,
  V.e10,
  V.crow,
  V.cmat,
  V.lrow,
  V.lmat,
];

/** `f(x)`, `f(x, dim)` for dims 1..3, `f(x, 'all')`. */
const REDUCE: Case[] = [
  ...each(REDUCE_IN),
  ...each(REDUCE_IN, lit("1")),
  ...each(REDUCE_IN, lit("2")),
  ...each([V.mat, V.t3, V.row], lit("3")),
  ...each([V.mat, V.t3, V.e03], lit("'all'")),
  { args: [V.mat, V.int] },
];

const NANFLAG: Case[] = [
  ...each([V.row, V.mat, V.t3, V.e03], lit("'omitnan'")),
  ...each([V.row, V.mat], lit("'includenan'")),
  ...each([V.mat, V.crow], lit("2"), lit("'omitnan'")),
];

const CUMULATIVE: Case[] = [
  ...each(REDUCE_IN),
  ...each(REDUCE_IN, lit("1")),
  ...each([V.row, V.mat, V.t3, V.crow, V.e03, V.lmat], lit("2")),
  ...each([V.mat, V.t3], lit("3")),
  { args: [V.row, lit("'reverse'")] },
];

const VARIANCE: Case[] = [
  ...each(REDUCE_IN),
  ...each([V.row, V.mat, V.t3, V.crow, V.e03], lit("0")),
  ...each([V.row, V.mat, V.t3, V.crow], lit("1")),
  ...each([V.mat, V.t3, V.cmat], lit("0"), lit("1")),
  ...each([V.mat, V.t3, V.row], lit("0"), lit("2")),
  ...each([V.mat, V.t3], lit("1"), lit("3")),
  { args: [V.row3, lit("[]"), lit("2")] },
  { args: [V.pos, lit("0")] },
];

const MINMAX: Case[] = [
  ...each(REDUCE_IN),
  ...each(REDUCE_IN, lit("[]"), lit("1")),
  ...each([V.row, V.mat, V.t3, V.crow, V.e03, V.lmat], lit("[]"), lit("2")),
  ...each([V.mat, V.t3], lit("[]"), lit("3")),
  ...each([V.mat, V.t3], lit("[]"), lit("'all'")),
  ...[V.row, V.mat, V.t3, V.crow, V.e03, V.lrow, "[NaN NaN 1]"].map(x => ({
    args: [x],
    nargout: 2,
  })),
  ...[V.mat, V.t3, V.cmat].map(x => ({
    args: [x, lit("[]"), lit("2")],
    nargout: 2,
  })),
  // Two-argument (element-wise) form.
  ...BINARY_PAIRS.map(([a, b]) => ({ args: [a, b] })),
  { args: [V.row, lit("0")] },
  { args: [lit("3"), lit("-1")] },
];

const PREDICATE_IN = ANY;

const SHAPE_ARGS: Case[] = [
  { args: [] },
  { args: [lit("3")] },
  { args: [lit("2"), lit("3")] },
  { args: [lit("2"), lit("3"), lit("2")] },
  { args: [lit("0"), lit("3")] },
  { args: [lit("-1")] },
  { args: ["[2 3]"] },
  { args: [lit("[2 3]")] },
  { args: [V.int] },
  { args: [V.int, lit("2")] },
  { args: [lit("2"), V.int] },
  { args: [lit("2.5")] },
];

const FLIP_IN = [
  V.pos,
  V.row,
  V.col,
  V.mat,
  V.t3,
  V.e00,
  V.e03,
  V.crow,
  V.cmat,
  V.lmat,
];

const TRIANGULAR: Case[] = [
  ...each([V.mat, V.sq, V.row, V.col, V.e00, V.cmat, V.lmat, V.t3]),
  ...each([V.mat, V.cmat], lit("1")),
  ...each([V.mat, V.sq], lit("-1")),
  ...each([V.mat], lit("5")),
  ...each([V.mat], lit("-4")),
  { args: [V.mat, V.int] },
];

const FMT = (f: string) => lit(`'${f}'`);
const FORMATS: Arg[][] = [
  [FMT("hello\\n")],
  [FMT("%g\\n"), V.pos],
  [FMT("%g|"), V.row],
  [FMT("%d %d\\n"), V.mat],
  [FMT("%5.2f;"), V.col],
  [FMT("%e\\n"), lit("pi")],
  [FMT("%d\\n"), V.nan],
  [FMT("%d %i\\n"), V.inf, V.ninf],
  [FMT("%d\\n"), V.pos],
  [FMT("%x %o\\n"), lit("255"), lit("8")],
  [FMT("%s|%s\\n"), V.chr, lit("'xy'")],
  [FMT("%c%c\\n"), lit("65"), lit("66")],
  [FMT("[%8.3g]\\n"), lit("-0.000123456")],
  [FMT("%+.3e\\n"), V.nzero],
  [FMT("%g %g\\n"), V.lrow],
  [FMT("%-6d|\\n"), V.int],
  [FMT("%%\\n")],
  [FMT("%g\\n"), V.e00],
  [FMT("%g\\n"), V.cs],
];

// ── Per-builtin tables ─────────────────────────────────────────────────

/** Cases per builtin; for an element-wise builtin these are added to its
 *  generated family table. */
export const TABLES: Record<string, Case[]> = {
  // Arithmetic with matrix semantics.
  mtimes: [
    { args: ["[1 2 3; 4 5 6]", "[1 -1; 0 2; 3 0.5]"] },
    { args: [V.pos, V.mat] },
    { args: [V.mat, V.neg] },
    { args: [V.row3, "[1; 2; 3]"] },
    { args: ["[1; 2; 3]", V.row3] },
    { args: [V.e03, "[1 2; 3 4; 5 6]"] },
    { args: [V.e10, "zeros(0, 2)"] },
    { args: [V.cmat, V.sq] },
    { args: [V.cs, V.cmat] },
    { args: [V.mat, V.mat2] },
    { args: [V.lmat, "[1 2; 3 4]"] },
    { args: [V.pos, V.neg] },
  ],
  mrdivide: [
    { args: [V.mat, V.pos] },
    { args: [V.pos, V.neg] },
    { args: [V.sq, "[1 2; 3 -1]"] },
    { args: [V.row3, "[2 0 1; 1 1 0; 0 3 1]"] },
    { args: [V.cmat, V.cs] },
    { args: [V.row, V.zero] },
    { args: [V.pos, V.sq] },
  ],
  mpower: [
    { args: [V.sq, lit("2")] },
    { args: [V.sq, lit("0")] },
    { args: [V.sq, lit("3")] },
    { args: [V.sq, lit("-1")] },
    { args: [V.sq, V.int] },
    { args: [V.pos, V.neg] },
    { args: [V.neg, lit("0.5")] },
    { args: [V.cs, lit("2")] },
    { args: [V.cmat, lit("2")] },
    { args: [V.mat, lit("2")] },
    { args: [V.e00, lit("2")] },
  ],
  power: [{ args: [V.neg, lit("0.5")] }, { args: [V.row, lit("0.5")] }],
  log2: [V.pos, V.row, V.mat, V.zero, V.nan, V.inf, V.ninf, V.e03].map(x => ({
    args: [x],
    nargout: 2,
  })),
  angle: UNARY,
  // Comparison and logic.
  not: each(NUMERIC),
  oror: [
    ...[
      [V.pos, V.zero],
      [V.zero, V.zero],
      [V.ls, V.zero],
      [V.nzero, V.neg],
      [V.nan, V.pos],
      [V.zero, V.nan],
      [V.row3, V.pos],
      [V.e00, V.pos],
    ].map(([a, b]) => ({ args: [a, b], body: "y1 = $1 || $2;" })),
  ],
  andand: [
    ...[
      [V.pos, V.zero],
      [V.pos, V.neg],
      [V.ls, V.ls],
      [V.zero, V.nan],
      [V.nan, V.pos],
      [V.row3, V.pos],
      [V.e00, V.pos],
    ].map(([a, b]) => ({ args: [a, b], body: "y1 = $1 && $2;" })),
  ],
  or: BINARY_PAIRS.map(([a, b]) => ({ args: [a, b] })),
  and: BINARY_PAIRS.map(([a, b]) => ({ args: [a, b] })),
  xor: BINARY_PAIRS.map(([a, b]) => ({ args: [a, b] })),
  // Reductions.
  sum: [...REDUCE, ...NANFLAG],
  prod: [...REDUCE, ...NANFLAG],
  mean: [...REDUCE, ...NANFLAG],
  any: REDUCE,
  all: REDUCE,
  min: MINMAX,
  max: MINMAX,
  var: VARIANCE,
  std: VARIANCE,
  cumsum: [
    ...CUMULATIVE,
    // A bracketed single output.
    { args: [V.mat], body: "[y1] = cumsum($1);" },
  ],
  cumprod: CUMULATIVE,
  cummax: [
    ...CUMULATIVE,
    ...[V.row, V.mat].map(x => ({ args: [x], nargout: 2 })),
  ],
  cummin: [
    ...CUMULATIVE,
    ...[V.row, V.mat].map(x => ({ args: [x], nargout: 2 })),
  ],
  diff: [
    ...each(REDUCE_IN),
    ...each([V.row, V.mat, V.t3, V.crow], lit("2")),
    ...each([V.row, V.mat, V.t3, V.cmat], lit("1"), lit("1")),
    ...each([V.mat, V.t3, V.row], lit("1"), lit("2")),
    ...each([V.t3], lit("1"), lit("3")),
    ...each([V.mat], lit("2"), lit("1")),
    ...each([V.row3], lit("5")),
    ...each([V.row], lit("0")),
  ],
  norm: [
    ...each([
      V.pos,
      V.neg,
      V.row,
      V.col,
      V.mat,
      V.sq,
      V.crow,
      V.cmat,
      V.e00,
      V.e03,
      V.lrow,
    ]),
    ...each([V.row, V.col, V.crow], lit("1")),
    ...each([V.row, V.col, V.crow], lit("Inf")),
    ...each([V.row, V.col], lit("-Inf")),
    ...each([V.row3, V.col], lit("3")),
    ...each([V.mat, V.sq, V.cmat], lit("1")),
    ...each([V.mat, V.sq], lit("Inf")),
    ...each([V.mat, V.sq, V.cmat, V.row], lit("'fro'")),
    ...each([V.sq], lit("2")),
  ],
  dot: [
    { args: [V.row, V.row] },
    { args: [V.row3, "[1 0 -2]"] },
    { args: [V.col, "[1; 2; 3]"] },
    { args: [V.row3, "[1; 2; 3]"] },
    { args: [V.mat, V.mat2] },
    { args: [V.mat, V.mat2, lit("2")] },
    { args: [V.mat, V.mat2, lit("1")] },
    { args: [V.crow, V.crow] },
    { args: [V.crow, "[1 2 3 4 5]"] },
    { args: [V.e10, V.e10] },
    { args: [V.pos, V.neg] },
    { args: [V.row3, "[1 2]"] },
  ],
  uniquetol: [
    ...each([
      "[1 1.0000001 2 3 2.9999999 NaN -0 0]",
      V.row3,
      V.col,
      V.mat,
      V.e00,
      V.pos,
    ]),
    ...each(["[1 1.05 2 3 2.95]", V.row], lit("0.1")),
    { args: ["[1 1.05 2 3 2.95]", "0.1"] },
  ],
  besselh: [
    { args: [lit("0"), lit("1"), V.pos] },
    { args: [lit("1"), lit("1"), V.pos] },
    { args: [lit("0"), lit("1"), "[0.5 1 2 10]"] },
    { args: [lit("1"), lit("1"), "[0.5 1 2 10]"] },
    { args: [lit("0"), lit("1"), V.zero] },
    { args: [lit("1"), lit("1"), V.neg] },
    { args: [lit("0"), lit("1"), V.inf] },
    { args: [lit("0"), lit("1"), V.e10] },
    { args: [lit("0"), V.pos] },
    { args: [lit("1"), V.pos] },
    { args: [lit("0"), lit("2"), V.pos] },
    { args: [lit("1"), lit("1"), V.cs] },
    { args: [lit("0"), "[0.5 1 2 10]"] },
    { args: [lit("2"), "[0.5 1 2 10]"] },
    { args: [lit("0.5"), V.pos] },
    { args: [V.int, V.pos] },
    { args: [lit("0"), V.zero] },
    { args: [lit("0"), V.neg] },
  ],
  linspace: [
    { args: [lit("0"), lit("1")] },
    { args: [lit("0"), lit("1"), lit("5")] },
    { args: [V.neg, V.pos, lit("4")] },
    { args: [V.pos, V.neg, V.int] },
    { args: [lit("0"), lit("1"), lit("1")] },
    { args: [lit("0"), lit("1"), lit("0")] },
    { args: [lit("1"), lit("1"), lit("3")] },
    { args: [V.zero, V.inf, lit("3")] },
    { args: [lit("0"), lit("1"), lit("2.5")] },
    { args: [V.cs, V.pos, lit("3")] },
  ],
  logspace: [
    { args: [lit("0"), lit("2")] },
    { args: [lit("0"), lit("2"), lit("3")] },
    { args: [V.neg, V.pos, lit("4")] },
    { args: [lit("1"), lit("0"), lit("1")] },
    { args: [lit("0"), lit("pi"), lit("3")] },
    { args: [lit("0"), lit("1"), lit("2.5")] },
  ],
  rand: [
    ...SHAPE_ARGS.filter(
      c => !c.args.some(a => typeof a !== "string" && a.lit === "-1")
    ).map(c => ({
      ...c,
      pre: "rng(3)",
    })),
  ],
  // Constructors and shape.
  zeros: [
    ...SHAPE_ARGS,
    { args: [lit("2"), lit("'logical'")] },
    { args: [lit("2"), lit("3"), lit("'like'"), V.cs] },
  ],
  ones: [...SHAPE_ARGS, { args: [lit("2"), lit("'like'"), V.ls] }],
  eye: SHAPE_ARGS,
  reshape: [
    { args: [V.row, lit("2"), lit("4")] },
    { args: [V.row, "[4 2]"] },
    { args: [V.row, lit("[]"), lit("2")] },
    { args: [V.row, lit("2"), lit("[]")] },
    { args: [V.mat, lit("3"), lit("4")] },
    { args: [V.mat, lit("2"), lit("3"), lit("2")] },
    { args: [V.t3, lit("4"), lit("3")] },
    { args: [V.e03, lit("3"), lit("0")] },
    { args: [V.crow, lit("5"), lit("1")] },
    { args: [V.lmat, lit("2"), lit("3")] },
    { args: [V.row, V.int, lit("3")] },
    { args: [V.row, lit("2"), lit("4.5")] },
  ],
  transpose: [
    ...each(NUMERIC),
    { args: [V.cmat], body: "y1 = $1';" },
    { args: [V.crow], body: "y1 = $1.';" },
  ],
  size: [
    ...each(ANY),
    ...each([V.mat, V.t3, V.row, V.e03], lit("1")),
    ...each([V.mat, V.t3, V.e03], lit("2")),
    ...each([V.mat, V.t3], lit("3")),
    { args: [V.mat, V.int] },
    ...[V.mat, V.t3, V.e03, V.pos].map(x => ({ args: [x], nargout: 2 })),
    ...[V.mat, V.t3].map(x => ({ args: [x], nargout: 3 })),
    { args: [V.mat, lit("[1 2]")] },
    { args: [V.t3], body: "[y1] = size($1);" },
  ],
  length: each(ANY),
  numel: each(ANY),
  ndims: each(ANY),
  flipud: each(FLIP_IN),
  fliplr: each(FLIP_IN),
  flip: [
    ...each(FLIP_IN),
    ...each([V.mat, V.t3, V.row, V.crow], lit("1")),
    ...each([V.mat, V.t3, V.col], lit("2")),
    ...each([V.mat, V.t3], lit("3")),
  ],
  sort: [
    ...each([...REDUCE_IN, "[3 -0 0 NaN -Inf 2 NaN 0 -0]"]),
    ...each([V.mat, V.t3, V.row], lit("2")),
    ...each([V.mat, V.t3], lit("1")),
    ...each([V.row, V.mat, V.crow], lit("'descend'")),
    ...each([V.mat], lit("2"), lit("'descend'")),
    ...each([V.row, V.mat], lit("'ascend'")),
    ...[V.row, V.mat, V.crow, "[2 1 2 1 NaN 0 -0]", V.lrow].map(x => ({
      args: [x],
      nargout: 2,
    })),
    { args: [V.mat, lit("2"), lit("'descend'")], nargout: 2 },
    { args: [V.row], body: "[y1] = sort($1);" },
    { args: [V.row], body: "[~] = sort($1); y1 = 0;" },
  ],
  meshgrid: [
    ...each([V.row3, V.col, V.e10, V.crow]),
    { args: [V.row3, "[10 20]"] },
    { args: [V.row3, "[10 20]"], nargout: 2 },
    { args: [V.row3], nargout: 2 },
    { args: ["1:2", "[5 6 7]", "[0 1]"], nargout: 3 },
    { args: [V.row3, V.row3, V.row3] },
  ],
  repmat: [
    { args: [V.row3, lit("2"), lit("3")] },
    { args: [V.row3, "[2 3]"] },
    { args: [V.mat, lit("2")] },
    { args: [V.pos, lit("2"), lit("2")] },
    { args: [V.cs, lit("2"), lit("2")] },
    { args: [V.crow, lit("1"), lit("2")] },
    { args: [V.lrow, lit("2"), lit("1")] },
    { args: [V.e03, lit("2"), lit("1")] },
    { args: [V.row3, lit("0"), lit("2")] },
    { args: [V.mat, lit("1"), lit("1"), lit("2")] },
    { args: [V.row3, V.int, lit("1")] },
    { args: [V.row3, lit("1.5"), lit("1")] },
  ],
  diag: [
    ...each([
      V.row3,
      V.col,
      V.mat,
      V.sq,
      V.cmat,
      V.crow,
      V.e00,
      V.pos,
      V.lrow,
      V.e10,
    ]),
    ...each([V.row3, V.mat], lit("1")),
    ...each([V.row3, V.mat], lit("-1")),
    ...each([V.mat], lit("5")),
    { args: [V.row3, V.int] },
  ],
  triu: TRIANGULAR,
  tril: TRIANGULAR,
  cat: [
    { args: [lit("1"), V.row3, "[4 5 6]"] },
    { args: [lit("2"), V.row3, "[4 5]"] },
    { args: [lit("3"), V.sq, "[5 6; 7 8]"] },
    { args: [lit("1"), V.mat, V.e00] },
    { args: [lit("2"), V.e00, V.row3] },
    { args: [lit("1"), V.crow, "[1 2 3 4 5]"] },
    { args: [lit("2"), V.lrow, V.lrow] },
    { args: [lit("2"), V.lrow, V.row3] },
    { args: [lit("1"), V.pos, V.neg, V.nan] },
    { args: [lit("1"), V.row3, "[1 2]"] },
    { args: [lit("4"), V.pos, V.neg] },
    { args: [lit("2"), V.e03, "zeros(0, 2)"] },
  ],
  // Constants with argument forms beyond the generated nullary sweep.
  eps: each([
    V.pos,
    V.zero,
    V.inf,
    V.nan,
    V.row,
    lit("1e300"),
    lit("'double'"),
  ]),
  // Type queries and predicates.
  isfield: [
    { args: [V.st, lit("'a'")] },
    { args: [V.st, lit("'z'")] },
    { args: [V.pos, lit("'a'")] },
    { args: [V.st, lit("{'a', 'z', 'b'}")] },
    { args: [V.st, V.chr] },
  ],
  isscalar: each(PREDICATE_IN),
  iscell: each(PREDICATE_IN),
  isstruct: each(PREDICATE_IN),
  isnumeric: each(PREDICATE_IN),
  isreal: each(PREDICATE_IN),
  isempty: each(PREDICATE_IN),
  ischar: each(PREDICATE_IN),
  isstring: each(PREDICATE_IN),
  isinteger: each(PREDICATE_IN),
  islogical: each(PREDICATE_IN),
  isfloat: each(PREDICATE_IN),
  ismatrix: each(PREDICATE_IN),
  isrow: each(PREDICATE_IN),
  iscolumn: each(PREDICATE_IN),
  class: each(PREDICATE_IN),
  isa: [
    ...each([V.pos, V.row, V.cs, V.ls, V.chr, V.st], lit("'double'")),
    ...each([V.pos, V.ls, V.chr], lit("'numeric'")),
    ...each([V.pos, V.ls], lit("'float'")),
    ...each([V.ls, V.lrow, V.pos], lit("'logical'")),
    ...each([V.chr, V.str], lit("'char'")),
    ...each([V.st], lit("'struct'")),
    ...each([V.pos], lit("'integer'")),
  ],
  // A cell result cannot leave a kernel, so the cases read it.
  fieldnames: [V.st, "struct()", V.pos].flatMap(x => [
    { args: [x], body: "c = fieldnames($1); y1 = numel(c);" },
    { args: [x], body: "c = fieldnames($1); y1 = c{end};" },
  ]),
  isequal: [
    { args: [V.row, V.row] },
    { args: [V.row3, "[3 -1 2]"] },
    { args: [V.row3, "[3; -1; 2]"] },
    { args: [V.nan, V.nan] },
    { args: [V.zero, V.nzero] },
    { args: [V.mat, V.mat2] },
    { args: [V.pos, "2.5"] },
    { args: [V.cs, V.cs] },
    { args: [V.cz, "2"] },
    { args: [V.lrow, "[1 0 1]"] },
    { args: [V.e00, V.e03] },
    { args: [V.e03, "zeros(0, 3)"] },
    { args: [V.chr, lit("'abc'")] },
    { args: [V.chr, V.str] },
    { args: [V.st, V.st] },
    { args: [V.row3, V.row3, "[3 -1 2]"] },
    { args: [V.pos, V.chr] },
  ],
  logical: each([
    V.pos,
    V.zero,
    V.nzero,
    V.lrow,
    "[1 0 2 -0]",
    V.mat2,
    V.nan,
    V.cs,
    V.e00,
    V.e03,
    V.chr,
  ]),
  double: each([...NUMERIC, V.chr]),
  strcmp: [
    { args: [V.chr, lit("'abc'")] },
    { args: [V.chr, lit("'ABC'")] },
    { args: [V.chr, lit("'ab'")] },
    { args: [V.chr, V.str] },
    { args: [V.str, lit('"abd"')] },
    { args: [V.pos, lit("'a'")] },
    { args: [V.chr, V.chr] },
    { args: [lit("''"), "''"] },
  ],
  strcmpi: [
    { args: [V.chr, lit("'ABC'")] },
    { args: [V.chr, lit("'abd'")] },
    { args: [V.str, lit("'AbC'")] },
    { args: [V.pos, lit("'a'")] },
  ],
  // System.
  deal: [
    { args: [V.pos] },
    { args: [V.row3] },
    { args: [V.pos], nargout: 2 },
    { args: [V.pos, V.crow], nargout: 2 },
    { args: [V.st, V.chr], nargout: 2 },
    { args: [V.pos, V.neg], nargout: 1 },
  ],
  feval: [
    { args: [lit("'sin'"), V.pos] },
    { args: [lit("@sin"), V.row] },
    { args: [lit("'max'"), V.mat] },
    { args: [lit("'plus'"), V.row, V.pos] },
    { args: [lit("@(x) x.^2 + 1"), V.row3] },
    { args: [lit("'size'"), V.mat], nargout: 2 },
  ],
  tic: [
    { args: [], body: "t = tic; y1 = toc(t) >= 0;" },
    { args: [], body: "tic; y1 = toc >= 0;" },
  ],
  toc: [
    { args: [], body: "tic; y1 = toc < 100;" },
    { args: [], body: "t = tic; y1 = toc(t) < 100;" },
  ],
  // Output and errors.
  disp: [
    ...ANY,
    "1e10",
    "[1 1e6 -3]",
    "[0.001 1000]",
    "-0.5",
    "1:20",
    "struct()",
    "{}",
  ].map(x => stmt("disp($1)", x)),
  fprintf: FORMATS.map(args =>
    stmt(`fprintf(${args.map((_, k) => `$${k + 1}`).join(", ")})`, ...args)
  ),
  sprintf: FORMATS.map(args => ({ args })),
  error: [
    stmt("error($1)", lit("'plain message'")),
    stmt("error($1, $2)", lit("'value %d too big'"), V.int),
    stmt(
      "error($1, $2, $3)",
      lit("'pkg:ident'"),
      lit("'got %g and %s'"),
      V.pos,
      lit("'x'")
    ),
    stmt("error($1, $2)", lit("'pkg:ident'"), lit("'no format'")),
    stmt("error($1)", lit("'pkg:ident'")),
    stmt("error($1)", V.chr),
    stmt("if $1 > 0, error('positive'); end", V.neg),
  ],
  warning: [
    stmt("warning($1)", lit("'plain warning'")),
    stmt("warning($1, $2)", lit("'value %d'"), V.int),
    stmt("warning($1, $2, $3)", lit("'pkg:w'"), lit("'id %g'"), V.pos),
    stmt("warning($1, $2)", lit("'off'"), lit("'pkg:w'")),
  ],
  assert: [
    stmt("assert($1)", V.ls),
    stmt("assert($1)", V.pos),
    stmt("assert($1)", V.lrow),
    stmt("assert($1)", "[true false]"),
    stmt("assert($1)", V.zero),
    stmt("assert($1)", V.e00),
    stmt("assert($1, $2)", V.zero, lit("'boom'")),
    stmt("assert($1, $2, $3)", V.zero, lit("'bad %d'"), V.int),
    stmt("assert($1, $2, $3)", V.zero, lit("'pkg:a'"), lit("'msg'")),
    stmt("assert($1)", V.nan),
  ],
};

// ── Allowlist ──────────────────────────────────────────────────────────

/** Builtins whose table compiles nothing at `--opt 1`, with the reason.
 *  Any other table must compile at least one case there. */
export const NO_JS_JIT: Record<string, string> = {
  besselh: "no JS-JIT emitter yet (needs a JS j0/j1/y0/y1); checked at --opt 2",
  fieldnames: "every form declines: the JIT has no type for its cell result",
};

/** Registered JIT builtins that this harness does not check, each with
 *  the reason. */
export const ALLOWLIST: Record<string, string> = {
  ...Object.fromEntries(
    plotBuiltins.map(b => [
      b.name,
      "graphics: acts on figure state through the plot host hook, not a value",
    ])
  ),
  addpath: "changes the session search path (a host side effect)",
  rmpath: "changes the session search path (a host side effect)",
  savepath: "writes the search path to disk",
};

export type KnownDivergence = DivergenceKind & {
  /** Unique; shown in reports. */
  reason: string;
  opts: ("1" | "2")[];
  names: string[];
  /** A tolerance that a given platform may not need (its C library may
   *  agree with V8 bit for bit), so it is exempt from the check that
   *  every entry still matches something. */
  optional?: boolean;
};

/** Known ways the JIT differs from the interpreter, each scoped to JIT
 *  levels and builtins. An entry (or a case id of a `case` entry) that
 *  no longer matches anything fails the test, so the list stays current.
 *  Unless noted, the interpreter agrees with MATLAB. */
export const KNOWN_DIVERGENCES: KnownDivergence[] = [
  {
    kind: "zero-imag",
    reason:
      "a complex result whose imaginary parts are all zero stays complex " +
      "(e.g. exp(complex(2, 0)), log(NaN), sqrt(zeros(0, 3)))",
    opts: ["1", "2"],
    names: [
      ...["atan", "besselh", "ceil", "conj", "cos", "cosh", "double", "exp"],
      "fix",
      ...["floor", "log", "log10", "log2", "max", "minus", "mrdivide"],
      ...["mtimes", "plus", "power", "rdivide", "round", "sign", "sin"],
      ...["sinh", "sqrt", "tan", "tanh", "times"],
    ],
  },
  {
    kind: "logical-class",
    reason: "a logical input comes back double (e.g. flip, reshape, max)",
    opts: ["1", "2"],
    names: [
      ...["cat", "cummax", "cummin", "diag", "flip", "fliplr", "flipud"],
      ...["max", "min", "repmat", "reshape"],
    ],
  },
  {
    kind: "ulp",
    ulps: 4,
    reason:
      "last-bit differences between V8's Math and the C library (exp, log, " +
      "pow, trigonometric and hyperbolic functions, hypot, complex " +
      "elementary functions); C libraries differ between machines",
    opts: ["2"],
    optional: true,
    names: [
      ...["abs", "angle", "asinh", "atan", "atan2", "cos", "cosh", "exp"],
      ...["feval", "hypot", "log", "log10", "log2", "logspace", "mpower"],
      ...["power", "sign", "sin", "sinh", "sqrt", "tan", "tanh"],
    ],
  },
  {
    kind: "ulp",
    ulps: 4,
    // p = 1, 2, Inf and -Inf of a real vector use only fabs, sums and
    // sqrt, so they stay exact.
    reason: "C-JIT norm(v, p) for a general p calls the C library's pow",
    opts: ["2"],
    optional: true,
    names: ["norm"],
    ids: ["norm([3 -1 2], 3)", "norm([0.5; -3; 2], 3)"],
  },
  {
    kind: "ulp",
    ulps: 0,
    reason: "C fmin/fmax return either zero of a -0/0 tie",
    opts: ["2"],
    names: ["max", "min"],
  },
  {
    kind: "ulp",
    ulps: 4096,
    reason:
      "complex elementary functions and log2/log10 use other formulas " +
      "than the interpreter's",
    opts: ["1", "2"],
    names: ["atan", "cos", "log10", "log2", "sin", "sqrt", "tan", "tanh"],
  },
  {
    kind: "ulp",
    ulps: 64,
    reason:
      "C-JIT besselh calls the C library's j0/j1/y0/y1; the interpreter " +
      "has its own implementation",
    opts: ["2"],
    names: ["besselh"],
  },
  {
    kind: "case",
    reason:
      "besselh at 0, Inf and a negative x: C-JIT and the interpreter " +
      "disagree with each other and with MATLAB (besselh(0, 1, 0) is " +
      "1 - Inf*i in MATLAB)",
    opts: ["2"],
    names: ["besselh"],
    ids: ["besselh(0, 1, 0)", "besselh(1, 1, -1.5)", "besselh(0, 1, Inf)"],
  },
  {
    kind: "case",
    reason: "logical(x) does not raise for NaN",
    opts: ["1", "2"],
    names: ["logical"],
    ids: [
      "logical(NaN)",
      "logical([1 -2 3.5 0 -0 NaN Inf -Inf])",
      "logical([1 -2 3; 4 NaN -6; -0 8 Inf; 0.25 -1 2])",
      "logical([2 2 -1; 0.5 3 NaN; 1 -0 4; -7 1 0])",
      "logical(reshape([1 -2 3 0.5 -0 6 NaN 8 -9 10 11 -12], 2, 3, 2))",
    ],
  },
  {
    kind: "case",
    reason: "any(z, dim) of a complex array counts NaN + 1i as nonzero",
    opts: ["1", "2"],
    names: ["any"],
    ids: ["any([1+2i, -3i, 4, NaN+1i, -2-0.5i], 1)"],
  },
  {
    kind: "case",
    reason: "cat(2, zeros(0, 3), zeros(0, 2)) is 0x0, not 0x5",
    opts: ["1", "2"],
    names: ["cat"],
    ids: ["cat(2, zeros(0, 3), zeros(0, 2))"],
  },
  {
    kind: "case",
    reason: "norm(v, Inf) and norm(v, -Inf) skip NaN instead of returning NaN",
    opts: ["1", "2"],
    names: ["norm"],
    ids: [
      "norm([1 -2 3.5 0 -0 NaN Inf -Inf], Inf)",
      "norm([1 -2 3.5 0 -0 NaN Inf -Inf], -Inf)",
      "norm([1+2i, -3i, 4, NaN+1i, -2-0.5i], Inf)",
    ],
  },
  {
    kind: "case",
    reason: "complex ./ 0: (NaN + 1i) / 0 has real part 0, not NaN",
    opts: ["1", "2"],
    names: ["rdivide"],
    ids: ["rdivide([1+2i, -3i, 4, NaN+1i, -2-0.5i], [2, 1i, -1, 0, 3])"],
  },
  {
    kind: "case",
    reason:
      "disp(struct()) prints nothing (the interpreter prints a blank line, " +
      "MATLAB 'struct with no fields.')",
    opts: ["1"],
    names: ["disp"],
    ids: ["disp(struct()); y1 = 0;"],
  },
  {
    kind: "case",
    reason:
      "C-JIT emits C that does not compile for diag(zeros(1, 0)) (the " +
      "interpreter returns NaN, MATLAB a 0x0 empty)",
    opts: ["2"],
    names: ["diag"],
    ids: ["diag(zeros(1, 0))"],
  },
  {
    kind: "case",
    reason:
      "C-JIT formats -0 as '0' under %g and '+0.000e+00' under %+.3e " +
      "(MATLAB: '-0' and '-0.000e+00'; the interpreter: '-0' and '0.000e+00')",
    opts: ["2"],
    names: ["fprintf"],
    ids: [
      "fprintf('%g|', [1 -2 3.5 0 -0 NaN Inf -Inf]); y1 = 0;",
      "fprintf('%+.3e\\n', -0); y1 = 0;",
    ],
  },
  {
    kind: "case",
    reason:
      "C-JIT mean of a complex NaN + 1i element makes the imaginary part NaN",
    opts: ["2"],
    names: ["mean"],
    ids: [
      "mean([1+2i, -3i, 4, NaN+1i, -2-0.5i])",
      "mean([1+2i, -3i, 4, NaN+1i, -2-0.5i], 1)",
      "mean([1+2i, -3i, 4, NaN+1i, -2-0.5i], 2)",
    ],
  },
];
