/**
 * Builtins package. Each builtin lives in its own file and is registered
 * here via `registerBuiltin`. Public API:
 *
 *  - `getBuiltin(name)` / `allBuiltinNames()` for runtime lookup.
 *  - `binaryOpBuiltin(op)` / `unaryOpBuiltin(op)` to map AST operator
 *    enums to builtin names.
 *
 *  See `registry.ts` for the `Builtin` interface and the registry itself.
 */

import type { Span } from "../parser/index.js";
import { BinaryOperation, UnaryOperation } from "../parser/index.js";
import { UnsupportedConstruct } from "../lowering/errors.js";
import { registerBuiltin } from "./registry.js";

import { plus } from "./defs/arithmetic/plus.js";
import { minus } from "./defs/arithmetic/minus.js";
import { times } from "./defs/arithmetic/times.js";
import { rdivide } from "./defs/arithmetic/rdivide.js";
import { mtimes } from "./defs/arithmetic/mtimes.js";
import { mrdivide } from "./defs/arithmetic/mrdivide.js";
import { uminus } from "./defs/arithmetic/uminus.js";
import { power } from "./defs/arithmetic/power.js";
import { mpower } from "./defs/arithmetic/mpower.js";
import { eq } from "./defs/compare/eq.js";
import { ne } from "./defs/compare/ne.js";
import { lt } from "./defs/compare/lt.js";
import { le } from "./defs/compare/le.js";
import { gt } from "./defs/compare/gt.js";
import { ge } from "./defs/compare/ge.js";
import { disp } from "./defs/io/disp.js";
import { errorBuiltin } from "./defs/io/error.js";
import { warningBuiltin } from "./defs/io/warning.js";
import { fprintf } from "./defs/io/fprintf.js";
import { sprintfBuiltin } from "./defs/io/sprintf.js";
import { length } from "./defs/reduction/length.js";
import { numel } from "./defs/reduction/numel.js";
import { sum } from "./defs/reduction/sum.js";
import { prod } from "./defs/reduction/prod.js";
import { mean } from "./defs/reduction/mean.js";
import { min } from "./defs/reduction/min.js";
import { max } from "./defs/reduction/max.js";
import { any } from "./defs/reduction/any.js";
import { all } from "./defs/reduction/all.js";
import { zeros } from "./defs/shape/zeros.js";
import { ones } from "./defs/shape/ones.js";
import { eye } from "./defs/shape/eye.js";
import { reshape } from "./defs/shape/reshape.js";
import { transpose } from "./defs/shape/transpose.js";
import { size } from "./defs/shape/size.js";
import { flipud, fliplr, flip } from "./defs/shape/flip.js";
import { sort } from "./defs/shape/sort.js";
import { meshgrid } from "./defs/shape/meshgrid.js";
import { assert as assertBuiltin } from "./defs/diag/assert.js";
import { tic } from "./defs/system/tic.js";
import { toc } from "./defs/system/toc.js";
import { addpath, rmpath, savepath } from "./defs/system/pathBuiltins.js";
import { cos } from "./defs/math/cos.js";
import { sin } from "./defs/math/sin.js";
import { tan } from "./defs/math/tan.js";
import { atan } from "./defs/math/atan.js";
import { exp } from "./defs/math/exp.js";
import { abs } from "./defs/math/abs.js";
import { signBuiltin } from "./defs/math/sign.js";
import { floor } from "./defs/math/floor.js";
import { ceil } from "./defs/math/ceil.js";
import { round } from "./defs/math/round.js";
import { fix } from "./defs/math/fix.js";
import { sqrt } from "./defs/math/sqrt.js";
import { norm } from "./defs/math/norm.js";
import { dot } from "./defs/math/dot.js";
import { uniquetol } from "./defs/math/uniquetol.js";
import { log } from "./defs/math/log.js";
import { log2 } from "./defs/math/log2.js";
import { log10 } from "./defs/math/log10.js";
import { mod } from "./defs/math/mod.js";
import { rem } from "./defs/math/rem.js";
import { atan2 } from "./defs/math/atan2.js";
import { hypot } from "./defs/math/hypot.js";
import { besselh } from "./defs/math/besselh.js";
import { linspace } from "./defs/math/linspace.js";
import { logspace } from "./defs/math/logspace.js";
import { real } from "./defs/math/real.js";
import { imag } from "./defs/math/imag.js";
import { conj } from "./defs/math/conj.js";
import { angle } from "./defs/math/angle.js";
import { pi, eps, Inf, inf, NaNBuiltin, nan } from "./defs/math/constants.js";
import { notBuiltin } from "./defs/logical/not.js";
import { oror } from "./defs/logical/oror.js";
import { andand } from "./defs/logical/andand.js";
import { orBuiltin } from "./defs/logical/or.js";
import { andBuiltin } from "./defs/logical/and.js";
import { xorBuiltin } from "./defs/logical/xor.js";
import { isfield } from "./defs/logical/isfield.js";
import { isscalar } from "./defs/logical/isscalar.js";
import { iscell } from "./defs/logical/iscell.js";
import { isstruct } from "./defs/logical/isstruct.js";
import { isa } from "./defs/logical/isa.js";
import { fieldnames } from "./defs/logical/fieldnames.js";
import { isnumeric } from "./defs/logical/isnumeric.js";
import { trueBuiltin, falseBuiltin } from "./defs/logical/boolconst.js";
import { isequal } from "./defs/logical/isequal.js";
import { isreal } from "./defs/logical/isreal.js";
import { isempty } from "./defs/logical/isempty.js";
import { isnan } from "./defs/logical/isnan.js";
import { logical } from "./defs/logical/logical.js";
import { isinf, isfinite } from "./defs/logical/isinf.js";
import { classBuiltin } from "./defs/logical/classBuiltin.js";
import { cumsum } from "./defs/reduction/cumsum.js";
import { cumprod } from "./defs/reduction/cumprod.js";
import { repmat } from "./defs/shape/repmat.js";
import { diag } from "./defs/shape/diag.js";
import { triu } from "./defs/shape/triu.js";
import { tril } from "./defs/shape/tril.js";
import { cat } from "./defs/shape/cat.js";
import { doubleBuiltin } from "./defs/math/double.js";
import { deal } from "./defs/system/deal.js";
import { feval } from "./defs/system/feval.js";
import {
  ischarBuiltin,
  isstring,
  isinteger,
  islogical,
  isfloat,
  ndims,
  ismatrix,
  isrow,
  iscolumn,
} from "./defs/logical/typequery.js";
import { strcmp, strcmpi } from "./defs/logical/strcmp.js";
import { plotBuiltins } from "./defs/plot/dispatch.js";

for (const b of [
  ...plotBuiltins,
  plus,
  minus,
  times,
  rdivide,
  mtimes,
  mrdivide,
  uminus,
  power,
  mpower,
  eq,
  ne,
  lt,
  le,
  gt,
  ge,
  disp,
  errorBuiltin,
  warningBuiltin,
  fprintf,
  sprintfBuiltin,
  length,
  numel,
  sum,
  prod,
  mean,
  min,
  max,
  any,
  all,
  zeros,
  ones,
  eye,
  reshape,
  transpose,
  size,
  flipud,
  fliplr,
  flip,
  sort,
  meshgrid,
  assertBuiltin,
  tic,
  toc,
  addpath,
  rmpath,
  savepath,
  cos,
  sin,
  tan,
  atan,
  exp,
  abs,
  signBuiltin,
  floor,
  ceil,
  round,
  fix,
  sqrt,
  norm,
  dot,
  uniquetol,
  log,
  log2,
  log10,
  mod,
  rem,
  atan2,
  hypot,
  besselh,
  linspace,
  logspace,
  real,
  imag,
  conj,
  angle,
  pi,
  eps,
  Inf,
  inf,
  NaNBuiltin,
  nan,
  notBuiltin,
  oror,
  andand,
  orBuiltin,
  andBuiltin,
  xorBuiltin,
  isfield,
  isscalar,
  iscell,
  isstruct,
  isa,
  fieldnames,
  isnumeric,
  trueBuiltin,
  falseBuiltin,
  isequal,
  isreal,
  isempty,
  isnan,
  logical,
  isinf,
  isfinite,
  ischarBuiltin,
  isstring,
  isinteger,
  islogical,
  isfloat,
  ndims,
  ismatrix,
  isrow,
  iscolumn,
  classBuiltin,
  cumsum,
  cumprod,
  repmat,
  diag,
  triu,
  tril,
  cat,
  doubleBuiltin,
  deal,
  feval,
  strcmp,
  strcmpi,
]) {
  registerBuiltin(b);
}

export type { Builtin } from "./registry.js";
export { registerBuiltin, getBuiltin, allBuiltinNames } from "./registry.js";

/** Source-level surface form for a binary op. Used to phrase
 *  user-facing rejection messages so the unsupported construct is
 *  identified by what the user typed (`^`, `||`, etc.), not by the
 *  AST enum tag. */
function binaryOpSurface(op: BinaryOperation): string {
  switch (op) {
    case BinaryOperation.Pow:
      return "^";
    case BinaryOperation.ElemPow:
      return ".^";
    case BinaryOperation.LeftDiv:
      return "\\";
    case BinaryOperation.ElemLeftDiv:
      return ".\\";
    case BinaryOperation.OrOr:
      return "||";
    case BinaryOperation.AndAnd:
      return "&&";
    case BinaryOperation.BitOr:
      return "|";
    case BinaryOperation.BitAnd:
      return "&";
    default:
      return op;
  }
}

function unaryOpSurface(op: UnaryOperation): string {
  switch (op) {
    case UnaryOperation.Not:
      return "~";
    case UnaryOperation.Transpose:
      return "'";
    case UnaryOperation.NonConjugateTranspose:
      return ".'";
    default:
      return op;
  }
}

export function binaryOpBuiltin(op: BinaryOperation, span: Span): string {
  switch (op) {
    case BinaryOperation.Add:
      return "plus";
    case BinaryOperation.Sub:
      return "minus";
    case BinaryOperation.Mul:
      return "mtimes";
    case BinaryOperation.ElemMul:
      return "times";
    case BinaryOperation.Div:
      return "mrdivide";
    case BinaryOperation.ElemDiv:
      return "rdivide";
    case BinaryOperation.Equal:
      return "eq";
    case BinaryOperation.NotEqual:
      return "ne";
    case BinaryOperation.Less:
      return "lt";
    case BinaryOperation.LessEqual:
      return "le";
    case BinaryOperation.Greater:
      return "gt";
    case BinaryOperation.GreaterEqual:
      return "ge";
    case BinaryOperation.OrOr:
      return "oror";
    case BinaryOperation.AndAnd:
      return "andand";
    // Eager elementwise `|` / `&`. The `or` / `and` builtins are
    // scalar-only (see `_shortcircuit.ts`); for scalar operands `a | b`
    // and `a & b` are identical to `||` / `&&` (both operands are
    // already evaluated by the JIT before the call, so the C-level
    // short-circuit is unobservable). Tensor operands are rejected at
    // the builtin's transfer and fall back to the interpreter.
    case BinaryOperation.BitOr:
      return "or";
    case BinaryOperation.BitAnd:
      return "and";
    case BinaryOperation.Pow:
      return "mpower";
    case BinaryOperation.ElemPow:
      return "power";
    default:
      // The remaining BinaryOperation cases (LeftDiv / ElemLeftDiv) are
      // valid numbl operators that mtoc2 doesn't yet support. Surface
      // as a span-attributed UnsupportedConstruct so the CLI prints the
      // source location of the offending operator instead of an internal
      // "unmapped op" error.
      throw new UnsupportedConstruct(
        `binary operator '${binaryOpSurface(op)}' is not yet supported`,
        span
      );
  }
}

export function unaryOpBuiltin(op: UnaryOperation, span: Span): string {
  switch (op) {
    case UnaryOperation.Minus:
      return "uminus";
    case UnaryOperation.Plus:
      return "uplus";
    case UnaryOperation.Transpose:
    case UnaryOperation.NonConjugateTranspose:
      // For real-typed inputs (mtoc2 v1) `.'` and `'` are identical.
      // When complex lands, `'` will route to a separate `ctranspose`.
      return "transpose";
    case UnaryOperation.Not:
      return "not";
    default:
      throw new UnsupportedConstruct(
        `unary operator '${unaryOpSurface(op)}' is not yet supported`,
        span
      );
  }
}
