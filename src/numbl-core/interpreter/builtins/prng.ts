/**
 * PRNG builtins: rand, randn, randi, randperm, rng.
 */

import {
  FloatXArray,
  isRuntimeString,
  isRuntimeChar,
  isRuntimeTensor,
  isRuntimeStruct,
} from "../../runtime/types.js";
import type { RuntimeValue, RuntimeStruct } from "../../runtime/types.js";
import { RTV, RuntimeError, numel } from "../../runtime/index.js";
import { toNumber, toString } from "../../runtime/convert.js";
import { defineBuiltin, registerIBuiltin } from "./types.js";
import {
  rngRandom,
  boxMullerRandom,
  seedRng,
  setRngShuffle,
  setRngSeed,
  getRngStateStruct,
  restoreRngState,
} from "../../builtins/prng.js";

// ── Shape parsing (local, mirrors builtins/shape-utils.ts) ──────────────

function parseShapeArgs(args: RuntimeValue[]): number[] {
  if (args.length === 1 && isRuntimeTensor(args[0])) {
    const t = args[0];
    const shape: number[] = [];
    for (let i = 0; i < t.data.length; i++)
      shape.push(Math.max(0, Math.round(t.data[i])));
    return shape;
  }
  return args.map(a => Math.max(0, Math.round(toNumber(a))));
}

// ── rng ─────────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "rng",
  resolve: () => {
    return {
      outputTypes: [{ kind: "struct", fields: {} }],
      apply: (args, nargout) => {
        const prevState = getRngStateStruct();
        if (args.length === 0) return prevState;

        const a = args[0];
        if (isRuntimeStruct(a)) {
          restoreRngState(a as RuntimeStruct);
        } else if (isRuntimeString(a) || isRuntimeChar(a)) {
          const s = toString(a).toLowerCase();
          if (s === "shuffle") {
            setRngShuffle();
          } else if (s === "default" || s === "twister") {
            seedRng(0);
            setRngSeed(0);
          } else {
            throw new RuntimeError(`rng: unknown option '${toString(a)}'`);
          }
        } else {
          const seed = Math.round(toNumber(a));
          // Accept optional generator name as second arg
          seedRng(seed);
          setRngSeed(seed);
        }

        if (nargout >= 1) return prevState;
        return RTV.num(0);
      },
    };
  },
});

// ── rand / randn ────────────────────────────────────────────────────────

function registerRandBuiltin(name: string, gen: () => number): void {
  defineBuiltin({
    name,
    cases: [
      // rand('seed', v) — legacy seeding
      {
        match: argTypes => {
          if (
            argTypes.length === 2 &&
            (argTypes[0].kind === "string" || argTypes[0].kind === "char")
          )
            return [{ kind: "number" }];
          return null;
        },
        apply: args => {
          const s = toString(args[0]);
          if (s !== "seed")
            throw new RuntimeError(
              `${name}: only 'seed' string option is supported`
            );
          seedRng(Math.round(toNumber(args[1])));
          return RTV.num(0);
        },
      },
      // rand(), rand(n), rand(m,n), rand([m,n])
      {
        match: argTypes => {
          if (argTypes.length === 0) return [{ kind: "number" }];
          for (const a of argTypes) {
            if (
              a.kind !== "number" &&
              a.kind !== "boolean" &&
              a.kind !== "tensor"
            )
              return null;
          }
          if (argTypes.length === 1 && argTypes[0].kind === "number")
            return [{ kind: "tensor", isComplex: false }];
          return [{ kind: "tensor", isComplex: false }];
        },
        apply: args => {
          if (args.length === 0) return RTV.num(gen());
          const shape = parseShapeArgs(args);
          if (shape.length === 1) shape.push(shape[0]);
          const n = numel(shape);
          const data = new FloatXArray(n);
          for (let i = 0; i < n; i++) data[i] = gen();
          return RTV.tensor(data, shape);
        },
      },
    ],
  });
}

registerRandBuiltin("rand", rngRandom);
registerRandBuiltin("randn", boxMullerRandom);

// ── randi ───────────────────────────────────────────────────────────────

defineBuiltin({
  name: "randi",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length === 0) return null;
        // First arg must be numeric
        const a = argTypes[0];
        if (a.kind !== "number" && a.kind !== "boolean" && a.kind !== "tensor")
          return null;
        if (argTypes.length === 1) return [{ kind: "number" }];
        return [{ kind: "tensor", isComplex: false }];
      },
      apply: args => {
        if (args.length === 0)
          throw new RuntimeError("randi requires at least 1 argument");
        let imin = 1;
        let imax: number;
        const first = args[0];
        if (isRuntimeTensor(first) && first.data.length === 2) {
          imin = Math.round(first.data[0]);
          imax = Math.round(first.data[1]);
        } else {
          imax = Math.round(toNumber(first));
        }
        if (imin > imax)
          throw new RuntimeError("randi: range must satisfy IMIN <= IMAX");
        const range = imax - imin + 1;
        const shapeArgs = args.slice(1);
        if (shapeArgs.length === 0)
          return RTV.num(Math.floor(rngRandom() * range) + imin);
        const shape = parseShapeArgs(shapeArgs);
        if (shape.length === 1) shape.push(shape[0]);
        const n = numel(shape);
        const data = new FloatXArray(n);
        for (let i = 0; i < n; i++)
          data[i] = Math.floor(rngRandom() * range) + imin;
        return RTV.tensor(data, shape);
      },
    },
  ],
});

// ── randperm ────────────────────────────────────────────────────────────

defineBuiltin({
  name: "randperm",
  cases: [
    {
      match: argTypes => {
        if (argTypes.length < 1 || argTypes.length > 2) return null;
        return [{ kind: "tensor", isComplex: false }];
      },
      apply: args => {
        const n = Math.round(toNumber(args[0]));
        const k = args.length === 2 ? Math.round(toNumber(args[1])) : n;
        if (k > n)
          throw new RuntimeError("randperm: K must be less than or equal to N");
        const perm = new FloatXArray(n);
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
    },
  ],
});
