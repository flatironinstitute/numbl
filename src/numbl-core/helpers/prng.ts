/**
 * Seedable PRNG (xoshiro128**) for random number generation
 */

import { RuntimeValue, RTV, toNumber, RuntimeError } from "../runtime/index.js";
import { FloatXArray, isRuntimeTensor } from "../runtime/types.js";

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
