/**
 * PRNG builtins: rand, randn, randi, randperm, rng.
 */

import {
  isRuntimeString,
  isRuntimeChar,
  isRuntimeComplexNumber,
  isRuntimeTensor,
  isRuntimeStruct,
} from "../../runtime/types.js";
import type { RuntimeValue, RuntimeStruct } from "../../runtime/types.js";
import { RTV, RuntimeError, numel } from "../../runtime/index.js";
import { toString } from "../../runtime/convert.js";
import {
  rngRandom,
  boxMullerRandom,
  fillRandn,
  fillRand,
  seedRng,
  setRngShuffle,
  setRngSeed,
  getRngStateStruct,
  restoreRngState,
} from "../../helpers/prng.js";
import { allocFloat64Array } from "../../runtime/alloc.js";
import { builtin } from "./builtin.js";
import { builtinError, realNumericArg, realScalarArg } from "./args.js";
import { stripClassNameArgs } from "./array-construction.js";

const isText = (v: RuntimeValue) => isRuntimeChar(v) || isRuntimeString(v);

// ── Size arguments ──────────────────────────────────────────────────────

/** Check a trailing class-name argument (rand(n, 'single'), randi(10,
 *  'int32')): numbl only has double, so the class is accepted and
 *  ignored. */
function checkClassName(name: string, args: RuntimeValue[]): void {
  const last = args[args.length - 1];
  if (args.length === 0 || !isText(last)) return;
  for (const a of args)
    if (isText(a) && toString(a).toLowerCase() === "like") return;
  const cls = toString(last).toLowerCase();
  const ok =
    cls === "double" ||
    cls === "single" ||
    (name === "randi" && /^u?int(8|16|32|64)$/.test(cls));
  if (!ok)
    throw builtinError(
      `MATLAB:${name}:invalidOutputType`,
      `Class name must be a class that supports ${name.toUpperCase()}, such as "double" or "single".`
    );
}

/** The dimensions of rand(n), rand(m, n, ...), rand([m n ...]). */
function sizeArgs(name: string, args: RuntimeValue[]): number[] {
  const values: number[] = [];
  for (const a of args) {
    if (isRuntimeComplexNumber(a) || (isRuntimeTensor(a) && a.imag))
      throw builtinError(
        "MATLAB:obsoleteComplexInput",
        "Size inputs must be real."
      );
    if (
      !(typeof a === "number" || typeof a === "boolean" || isRuntimeTensor(a))
    )
      throw builtinError(
        `MATLAB:${name}:inputType`,
        "Size inputs must be numeric."
      );
    const v = realNumericArg(a);
    if (typeof v === "number") values.push(v);
    else if (isRuntimeTensor(v)) {
      if (args.length > 1 && v.data.length !== 1)
        throw new RuntimeError(`${name}: size inputs must be scalar`);
      for (let i = 0; i < v.data.length; i++) values.push(v.data[i]);
    } else values.push(realScalarArg(v));
  }
  const shape = values.map(x => Math.max(0, Math.round(x)));
  if (shape.length === 1) shape.push(shape[0]);
  return shape;
}

// ── rng ─────────────────────────────────────────────────────────────────

builtin({
  name: "rng",
  // rng, rng(seed), rng(seed, generator), rng('shuffle'|'default'), rng(s)
  nargin: [0, 2],
  mfile: true,
  apply: (args, nargout) => {
    const prevState = getRngStateStruct();
    if (args.length === 0) return prevState;

    const a = args[0];
    if (isRuntimeStruct(a)) {
      restoreRngState(a as RuntimeStruct);
    } else if (isText(a)) {
      const s = toString(a).toLowerCase();
      if (s === "shuffle") {
        setRngShuffle();
      } else if (s === "default" || s === "twister") {
        seedRng(0);
        setRngSeed(0);
      } else {
        throw builtinError(
          "MATLAB:rng:unknownRNGType",
          `'${toString(a)}' is not a valid generator type.`
        );
      }
    } else if (
      typeof a === "number" ||
      typeof a === "boolean" ||
      isRuntimeTensor(a) ||
      isRuntimeComplexNumber(a)
    ) {
      const badSeed = () =>
        builtinError(
          "MATLAB:rng:badSeed",
          "SD must be a nonnegative integer seed less than 2^32."
        );
      let seed: number;
      try {
        seed = realScalarArg(a);
      } catch {
        throw badSeed();
      }
      if (!Number.isInteger(seed) || seed < 0 || seed >= 2 ** 32)
        throw badSeed();
      // The optional generator name (args[1]) is accepted and ignored.
      seedRng(seed);
      setRngSeed(seed);
    } else {
      throw builtinError(
        "MATLAB:rng:badFirstOpt",
        "First input must be a nonnegative integer seed less than 2^32, 'shuffle', 'default', a valid generator name, or generator settings captured previously using S = RNG."
      );
    }

    // Called as a statement, rng sets no ans.
    if (nargout >= 1) return prevState;
  },
});

// ── rand / randn ────────────────────────────────────────────────────────

function registerRandBuiltin(
  name: string,
  gen: () => number,
  bulkFill?: (data: Float64Array) => void
): void {
  builtin({
    name,
    apply: rawArgs => {
      // rand('seed', v) — legacy seeding
      if (rawArgs.length === 2 && isText(rawArgs[0])) {
        const s = toString(rawArgs[0]);
        if (s !== "seed")
          throw new RuntimeError(
            `${name}: only 'seed' string option is supported`
          );
        seedRng(Math.round(realScalarArg(rawArgs[1])));
        return RTV.num(0);
      }
      // rand(), rand(n), rand(m,n), rand([m,n]), with an optional trailing
      // class name (rand(n,'single')) that numbl ignores.
      checkClassName(name, rawArgs);
      const args = stripClassNameArgs(rawArgs);
      if (args.length === 0) return RTV.num(gen());
      const shape = sizeArgs(name, args);
      const n = numel(shape);
      const data = allocFloat64Array(n);
      if (bulkFill) {
        bulkFill(data);
      } else {
        for (let i = 0; i < n; i++) data[i] = gen();
      }
      return RTV.tensor(data, shape);
    },
  });
}

registerRandBuiltin("rand", rngRandom, fillRand);
registerRandBuiltin("randn", boxMullerRandom, fillRandn);

// ── randi ───────────────────────────────────────────────────────────────

const randiLimitsError = (id = "MATLAB:randi:invalidLimits") =>
  builtinError(
    id,
    "First input must be a positive scalar integer value IMAX, or two integer values [IMIN IMAX] with IMIN less than or equal to IMAX."
  );

builtin({
  name: "randi",
  nargin: [1, Infinity],
  apply: args => {
    const first = args[0];
    if (
      !(
        typeof first === "number" ||
        typeof first === "boolean" ||
        isRuntimeTensor(first)
      ) ||
      (isRuntimeTensor(first) && first.imag)
    )
      throw randiLimitsError();
    let imin = 1;
    let imax: number;
    if (isRuntimeTensor(first) && first.data.length === 2) {
      imin = Math.round(first.data[0]);
      imax = Math.round(first.data[1]);
    } else {
      imax = Math.round(realScalarArg(first));
    }
    if (imin > imax)
      throw randiLimitsError("MATLAB:RandStream:randi:invalidLimits");
    const range = imax - imin + 1;
    const rest = args.slice(1);
    checkClassName("randi", rest);
    const shapeArgs = stripClassNameArgs(rest);
    if (shapeArgs.length === 0)
      return RTV.num(Math.floor(rngRandom() * range) + imin);
    const shape = sizeArgs("randi", shapeArgs);
    const n = numel(shape);
    const data = allocFloat64Array(n);
    fillRand(data);
    for (let i = 0; i < n; i++) data[i] = Math.floor(data[i] * range) + imin;
    return RTV.tensor(data, shape);
  },
});

// ── randperm ────────────────────────────────────────────────────────────

builtin({
  name: "randperm",
  nargin: [1, 2],
  apply: args => {
    const scalar = (v: RuntimeValue) => {
      if (isRuntimeComplexNumber(v) || (isRuntimeTensor(v) && v.imag))
        throw builtinError(
          "MATLAB:obsoleteComplexInput",
          "Size inputs must be real."
        );
      return Math.round(realScalarArg(v));
    };
    const n = scalar(args[0]);
    const k = args.length === 2 ? scalar(args[1]) : n;
    if (k > n)
      throw builtinError(
        "MATLAB:randperm:inputKTooLarge",
        "K must be less than or equal to N."
      );
    const perm = allocFloat64Array(n);
    for (let i = 0; i < n; i++) perm[i] = i + 1;
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rngRandom() * (i + 1));
      const tmp = perm[i];
      perm[i] = perm[j];
      perm[j] = tmp;
    }
    if (k === n) return RTV.tensor(perm, [1, n]);
    return RTV.tensor(perm.slice(0, k), [1, k]);
  },
});
