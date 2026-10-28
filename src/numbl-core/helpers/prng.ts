/**
 * Seedable PRNG (xoshiro128**) for random number generation
 */

import { RuntimeValue, RTV, toNumber, RuntimeError } from "../runtime/index.js";
import { isRuntimeTensor } from "../runtime/types.js";
import { getLapackBridge } from "../native/lapack-bridge.js";
import { allocFloat64Array } from "../runtime/alloc.js";
import { currentSession, type RngState } from "../sessionState.js";

// ── Seedable PRNG (xoshiro128**) ────────────────────────────────────────
//
// The generator state is per session (`SessionState.rng`), so `rng` in one
// session does not move another's stream, and the state carries across a
// session's runs.

const rng = (): RngState => currentSession().rng;

export function setRngShuffle(): void {
  const r = rng();
  r.state = null;
  r.seed = 0;
  r.spare = null;
}

export function setRngSeed(seed: number): void {
  rng().seed = seed;
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
  const r = rng();
  r.spare = null;
  const sm = splitmix32(seed);
  const state = new Uint32Array([sm(), sm(), sm(), sm()]);
  // Ensure state is non-zero
  if (state[0] === 0 && state[1] === 0 && state[2] === 0 && state[3] === 0) {
    state[0] = 1;
  }
  r.state = state;
}

function xoshiro128ss(s: Uint32Array): number {
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

/** Opaque copy of the full PRNG state (generator, seed, Box-Muller spare),
 *  shared by rand/randn/randi. */
export interface RngSnapshot {
  readonly state: Uint32Array | null;
  readonly seed: number;
  readonly spare: number | null;
}

/** Copy the PRNG state, e.g. before a JIT attempt that may bail and be
 *  re-run by the interpreter. The unseeded (`Math.random`) stream cannot be
 *  rewound; only seeded state is restored exactly. */
export function snapshotRng(): RngSnapshot {
  const r = rng();
  return { state: r.state && r.state.slice(), seed: r.seed, spare: r.spare };
}

/** Restore a state taken by `snapshotRng`. */
export function restoreRng(s: RngSnapshot): void {
  const r = rng();
  r.state = s.state && s.state.slice();
  r.seed = s.seed;
  r.spare = s.spare;
}

/** A random float in [0, 1) from `r`. */
function nextUniform(r: RngState): number {
  if (r.state === null) return Math.random();
  return (xoshiro128ss(r.state) >>> 0) / 0x100000000;
}

/** Return a random float in [0, 1) using seeded or unseeded PRNG */
export function rngRandom(): number {
  return nextUniform(rng());
}

/** Fill a typed array with uniform values in [0, 1). */
export function fillRand(data: Float64Array): void {
  const r = rng();
  for (let i = 0; i < data.length; i++) data[i] = nextUniform(r);
}

// Marsaglia polar method — generates pairs without trig functions.
// Uses rejection sampling (~78.5% acceptance) but avoids cos/sin entirely.

export function boxMullerRandom(): number {
  const r = rng();
  if (r.spare !== null) {
    const s = r.spare;
    r.spare = null;
    return s;
  }
  let u: number, v: number, s: number;
  do {
    u = 2.0 * nextUniform(r) - 1.0;
    v = 2.0 * nextUniform(r) - 1.0;
    s = u * u + v * v;
  } while (s >= 1.0 || s === 0.0);
  const mul = Math.sqrt((-2.0 * Math.log(s)) / s);
  r.spare = v * mul;
  return u * mul;
}

/** Fill a typed array with normal random values (bulk polar method) */
export function fillRandn(data: Float64Array): void {
  const r = rng();
  // Native fast path: seeded PRNG + addon available
  if (r.state !== null) {
    const bridge = getLapackBridge();
    if (bridge.fillRandn) {
      const res = bridge.fillRandn(
        r.state,
        data.length,
        r.spare ?? 0,
        r.spare !== null
      );
      // State was mutated in-place by the addon. Update spare.
      r.spare = res.hasSpare ? res.spare : null;
      // Copy result into caller's typed array
      data.set(res.data);
      return;
    }
  }

  // JS fallback
  const n = data.length;
  let i = 0;
  if (r.spare !== null && i < n) {
    data[i++] = r.spare;
    r.spare = null;
  }
  for (; i + 1 < n; i += 2) {
    let u: number, v: number, s: number;
    do {
      u = 2.0 * nextUniform(r) - 1.0;
      v = 2.0 * nextUniform(r) - 1.0;
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
  const r = rng();
  const stateArray = r.state
    ? RTV.tensor(allocFloat64Array(Array.from(r.state)), [4, 1])
    : RTV.tensor(allocFloat64Array(0), [0, 1]);
  return RTV.struct({
    Type: RTV.char("twister"),
    Seed: RTV.num(r.seed),
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
  const r = rng();
  r.seed = Math.round(toNumber(seedField));
  if (isRuntimeTensor(stateField) && stateField.data.length === 4) {
    const state = new Uint32Array(4);
    for (let i = 0; i < 4; i++) state[i] = stateField.data[i] >>> 0;
    r.state = state;
  } else {
    // Empty state means unseeded
    r.state = null;
  }
}
