/**
 * Seedable PRNG (xoshiro128**) for random number generation
 */

import { RuntimeValue, RTV, toNumber, RuntimeError } from "../runtime/index.js";
import {
  FloatXArray,
  type FloatXArrayType,
  isRuntimeTensor,
} from "../runtime/types.js";
import { getLapackBridge } from "../native/lapack-bridge.js";

// ── Seedable PRNG (xoshiro128**) ────────────────────────────────────────

let _rngState: Uint32Array | null = null; // null = use Math.random()
let _rngSeed: number = 0; // track current seed for rng() output
let _bmSpare: number | null = null; // cached spare from Box-Muller

export function setRngShuffle(): void {
  _rngState = null;
  _rngSeed = 0;
  _bmSpare = null;
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
  _bmSpare = null;
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

// Marsaglia polar method — generates pairs without trig functions.
// Uses rejection sampling (~78.5% acceptance) but avoids cos/sin entirely.

export function boxMullerRandom(): number {
  if (_bmSpare !== null) {
    const s = _bmSpare;
    _bmSpare = null;
    return s;
  }
  let u: number, v: number, s: number;
  do {
    u = 2.0 * rngRandom() - 1.0;
    v = 2.0 * rngRandom() - 1.0;
    s = u * u + v * v;
  } while (s >= 1.0 || s === 0.0);
  const mul = Math.sqrt((-2.0 * Math.log(s)) / s);
  _bmSpare = v * mul;
  return u * mul;
}

/** Fill a typed array with normal random values (bulk polar method) */
export function fillRandn(data: FloatXArrayType): void {
  // Native fast path: seeded PRNG + addon available
  if (_rngState !== null) {
    const bridge = getLapackBridge();
    if (bridge?.fillRandn) {
      const r = bridge.fillRandn(
        _rngState,
        data.length,
        _bmSpare ?? 0,
        _bmSpare !== null
      );
      // State was mutated in-place by the addon. Update spare.
      _bmSpare = r.hasSpare ? r.spare : null;
      // Copy result into caller's typed array
      if (data instanceof Float64Array) {
        data.set(r.data);
      } else {
        for (let i = 0; i < data.length; i++) data[i] = r.data[i];
      }
      return;
    }
  }

  // JS fallback
  const n = data.length;
  let i = 0;
  if (_bmSpare !== null && i < n) {
    data[i++] = _bmSpare;
    _bmSpare = null;
  }
  for (; i + 1 < n; i += 2) {
    let u: number, v: number, s: number;
    do {
      u = 2.0 * rngRandom() - 1.0;
      v = 2.0 * rngRandom() - 1.0;
      s = u * u + v * v;
    } while (s >= 1.0 || s === 0.0);
    const mul = Math.sqrt((-2.0 * Math.log(s)) / s);
    data[i] = u * mul;
    data[i + 1] = v * mul;
  }
  if (i < n) {
    data[i] = boxMullerRandom();
  }
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
