/**
 * Seedable PRNG (xoshiro128**) for random number generation
 */

import {
  RuntimeValue,
  RTV,
  toNumber,
  RuntimeError,
  numel,
} from "../runtime/index.js";
import { rstr } from "../runtime/runtime.js";
import {
  FloatXArray,
  isRuntimeString,
  isRuntimeChar,
  isRuntimeTensor,
  isRuntimeStruct,
} from "../runtime/types.js";
import {
  register,
  builtinSingle,
  realArrayConstructorCheck,
} from "./registry.js";
import { parseShapeArgs } from "./shape-utils.js";

// ── Seedable PRNG (xoshiro128**) ────────────────────────────────────────

let _rngState: Uint32Array | null = null; // null = use Math.random()
let _rngSeed: number = 0; // track current seed for rng() output

export function setRngShuffle(): void {
  _rngState = null;
  _rngSeed = 0;
}

export function setRngSeed(seed: number): void {
  _rngSeed = seed;
}

function splitmix32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x9e3779b9) | 0;
    let z = s;
    z = Math.imul(z ^ (z >>> 16), 0x85ebca6b);
    z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35);
    return (z ^ (z >>> 16)) >>> 0;
  };
}

export function seedRng(seed: number): void {
  const sm = splitmix32(seed);
  _rngState = new Uint32Array([sm(), sm(), sm(), sm()]);
  // Ensure state is non-zero
  if (
    _rngState[0] === 0 &&
    _rngState[1] === 0 &&
    _rngState[2] === 0 &&
    _rngState[3] === 0
  ) {
    _rngState[0] = 1;
  }
}

function xoshiro128ss(): number {
  const s = _rngState!;
  const result = Math.imul(rotl(Math.imul(s[1], 5), 7), 9) >>> 0;
  const t = s[1] << 9;
  s[2] ^= s[0];
  s[3] ^= s[1];
  s[1] ^= s[2];
  s[0] ^= s[3];
  s[2] ^= t;
  s[3] = rotl(s[3], 11);
  return result;
}

function rotl(x: number, k: number): number {
  return ((x << k) | (x >>> (32 - k))) >>> 0;
}

/** Return a random float in [0, 1) using seeded or unseeded PRNG */
export function rngRandom(): number {
  if (_rngState === null) return Math.random();
  return (xoshiro128ss() >>> 0) / 0x100000000;
}

export function boxMullerRandom(): number {
  let u = 0,
    v = 0;
  while (u === 0) u = rngRandom();
  while (v === 0) v = rngRandom();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

/** Return the current RNG state as a struct {Type, Seed, State} */
export function getRngStateStruct(): RuntimeValue {
  const stateArray = _rngState
    ? RTV.tensor(new FloatXArray(Array.from(_rngState).map(v => v)), [4, 1])
    : RTV.tensor(new FloatXArray(0), [0, 1]);
  return RTV.struct({
    Type: RTV.char("twister"),
    Seed: RTV.num(_rngSeed),
    State: stateArray,
  });
}

/** Restore RNG state from a struct previously returned by rng() */
export function restoreRngState(s: {
  kind: "struct";
  fields: Map<string, RuntimeValue>;
}): void {
  const seedField = s.fields.get("Seed");
  const stateField = s.fields.get("State");
  if (seedField === undefined || stateField === undefined) {
    throw new RuntimeError("rng: invalid state structure");
  }
  _rngSeed = Math.round(toNumber(seedField));
  if (isRuntimeTensor(stateField) && stateField.data.length === 4) {
    _rngState = new Uint32Array(4);
    for (let i = 0; i < 4; i++) _rngState[i] = stateField.data[i] >>> 0;
  } else {
    // Empty state means unseeded
    _rngState = null;
  }
}

export function registerPrngFunctions(): void {
  register(
    "rng",
    builtinSingle((args, nargout) => {
      // Build the "previous state" struct to return if nargout >= 1
      const prevState = getRngStateStruct();

      if (args.length === 0) {
        // rng() or t = rng — just return current state
        return prevState;
      }

      const a = args[0];

      if (isRuntimeStruct(a)) {
        // rng(s) — restore state from struct
        restoreRngState(a);
      } else if (isRuntimeString(a) || isRuntimeChar(a)) {
        const s = rstr(a).toLowerCase();
        if (s === "shuffle") {
          _rngState = null;
          _rngSeed = 0;
        } else if (s === "default") {
          seedRng(0);
          _rngSeed = 0;
        } else if (s === "twister") {
          // rng('twister') equivalent to rng(0, 'twister')
          seedRng(0);
          _rngSeed = 0;
        } else {
          throw new RuntimeError(`rng: unknown option '${rstr(a)}'`);
        }
      } else {
        const seed = Math.round(toNumber(a));
        // Second arg is optional generator name — accept and ignore (we only have one PRNG)
        if (args.length >= 2) {
          const gen = args[1];
          if (isRuntimeString(gen) || isRuntimeChar(gen)) {
            // Accept known generator names
            const gname = rstr(gen).toLowerCase();
            if (
              gname !== "twister" &&
              gname !== "simdtwister" &&
              gname !== "combrecursive" &&
              gname !== "philox" &&
              gname !== "threefry" &&
              gname !== "multfibonacci" &&
              gname !== "v5uniform" &&
              gname !== "v5normal" &&
              gname !== "v4"
            ) {
              throw new RuntimeError(`rng: unknown generator '${rstr(gen)}'`);
            }
          }
        }
        seedRng(seed);
        _rngSeed = seed;
      }

      if (nargout >= 1) {
        return prevState;
      }
      return RTV.num(0);
    })
  );

  // Shared helper for rand/randn: legacy 'seed' branch + array constructor
  function registerRandFn(name: string, gen: () => number): void {
    register(name, [
      {
        check: argTypes =>
          argTypes.length === 2 &&
          (argTypes[0].kind === "Char" || argTypes[0].kind === "String") &&
          (argTypes[1].kind === "Number" || argTypes[1].kind === "Unknown")
            ? { outputTypes: [{ kind: "Number" }] }
            : null,
        apply: args => {
          const s = args[0];
          if ((!isRuntimeString(s) && !isRuntimeChar(s)) || rstr(s) !== "seed")
            throw new RuntimeError(
              `${name}: only 'seed' string option is supported`
            );
          seedRng(Math.round(toNumber(args[1])));
          return RTV.num(0);
        },
      },
      {
        check: realArrayConstructorCheck,
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
    ]);
  }

  registerRandFn("rand", rngRandom);
  registerRandFn("randn", boxMullerRandom);

  // randi(imax), randi(imax, n), randi(imax, m, n, ...), randi([imin imax], ...)
  register(
    "randi",
    builtinSingle(args => {
      if (args.length === 0)
        throw new RuntimeError("randi requires at least 1 argument");

      // Parse range: first arg is either imax (scalar) or [imin, imax] (vector)
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

      if (shapeArgs.length === 0) {
        // randi(imax) — scalar
        return RTV.num(Math.floor(rngRandom() * range) + imin);
      }

      const shape = parseShapeArgs(shapeArgs);
      if (shape.length === 1) shape.push(shape[0]);
      const n = numel(shape);
      const data = new FloatXArray(n);
      for (let i = 0; i < n; i++) {
        data[i] = Math.floor(rngRandom() * range) + imin;
      }
      return RTV.tensor(data, shape);
    })
  );

  register(
    "randperm",
    builtinSingle(args => {
      if (args.length < 1 || args.length > 2)
        throw new RuntimeError("randperm requires 1 or 2 arguments");
      const n = Math.round(toNumber(args[0]));
      const k = args.length === 2 ? Math.round(toNumber(args[1])) : n;
      if (k > n)
        throw new RuntimeError("randperm: K must be less than or equal to N");
      // Fisher-Yates shuffle on 1:n, then take first k
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
    })
  );
}
