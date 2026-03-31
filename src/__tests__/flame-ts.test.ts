import { describe, it, expect } from "vitest";
import { getTsLapackBridge } from "../numbl-core/native/ts-lapack-bridge.js";
import { getFlameBridge } from "../flame-ts/bridge.js";

const ts = getTsLapackBridge();
const flame = getFlameBridge();

function splitmix32(seed: number): () => number {
  return () => {
    seed |= 0;
    seed = (seed + 0x9e3779b9) | 0;
    let t = seed ^ (seed >>> 16);
    t = Math.imul(t, 0x21f0aaad);
    t = t ^ (t >>> 15);
    t = Math.imul(t, 0x735a2d97);
    t = t ^ (t >>> 15);
    return (t >>> 0) / 4294967296;
  };
}

function randomMatrix(rows: number, cols: number, seed: number): Float64Array {
  const rng = splitmix32(seed);
  const data = new Float64Array(rows * cols);
  for (let i = 0; i < data.length; i++) data[i] = rng() * 2 - 1;
  return data;
}

function randomSPD(n: number, seed: number): Float64Array {
  const m = randomMatrix(n, n, seed);
  const out = new Float64Array(n * n);
  for (let j = 0; j < n; j++)
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += m[k + i * n] * m[k + j * n];
      out[i + j * n] = s;
    }
  for (let i = 0; i < n; i++) out[i + i * n] += n;
  return out;
}

function wellConditioned(n: number, seed: number): Float64Array {
  const data = randomMatrix(n, n, seed);
  for (let i = 0; i < n; i++) data[i + i * n] += n;
  return data;
}

function assertClose(
  actual: Float64Array,
  expected: Float64Array,
  tol: number,
  label: string
) {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < actual.length; i++) {
    const scale = Math.max(1, Math.abs(expected[i]));
    expect(
      Math.abs(actual[i] - expected[i]) / scale,
      `${label} index ${i}: got ${actual[i]}, expected ${expected[i]}`
    ).toBeLessThan(tol);
  }
}

const SIZES = [16, 64, 128];
const TOL = 1e-8;

describe("flame-ts correctness vs ts-lapack", () => {
  describe("matmul", () => {
    for (const n of SIZES) {
      it(`${n}x${n}`, () => {
        const A = randomMatrix(n, n, n * 7);
        const B = randomMatrix(n, n, n * 11);
        const ref = ts.matmul!(A, n, n, B, n);
        const res = flame.matmul!(A, n, n, B, n);
        assertClose(res, ref, TOL, "matmul");
      });
    }

    it("non-square 100x80 * 80x60", () => {
      const m = 100,
        k = 80,
        n = 60;
      const A = randomMatrix(m, k, 42);
      const B = randomMatrix(k, n, 43);
      const ref = ts.matmul!(A, m, k, B, n);
      const res = flame.matmul!(A, m, k, B, n);
      assertClose(res, ref, TOL, "matmul non-square");
    });
  });

  describe("inv", () => {
    for (const n of SIZES) {
      it(`${n}x${n}`, () => {
        const A = wellConditioned(n, n * 13);
        const ref = ts.inv(A, n);
        const res = flame.inv(A, n);
        assertClose(res, ref, TOL, "inv");
      });
    }
  });

  describe("lu", () => {
    for (const n of SIZES) {
      it(`${n}x${n}`, () => {
        const A = randomMatrix(n, n, n * 17);
        const ref = ts.lu!(A, n, n);
        const res = flame.lu!(A, n, n);
        // LU factors may differ due to different pivot choices,
        // so verify P*A = L*U instead of comparing factors directly
        // For simplicity, verify that the factored matrix reconstructs A
        // Apply ref pivots to A and check L*U matches
        const refA = new Float64Array(A);
        for (let i = 0; i < n; i++) {
          const pi = ref.ipiv[i] - 1;
          if (pi !== i) {
            for (let c = 0; c < n; c++) {
              const tmp = refA[i + c * n];
              refA[i + c * n] = refA[pi + c * n];
              refA[pi + c * n] = tmp;
            }
          }
        }

        const resA = new Float64Array(A);
        for (let i = 0; i < n; i++) {
          const pi = res.ipiv[i] - 1;
          if (pi !== i) {
            for (let c = 0; c < n; c++) {
              const tmp = resA[i + c * n];
              resA[i + c * n] = resA[pi + c * n];
              resA[pi + c * n] = tmp;
            }
          }
        }

        // Reconstruct L*U from res and check it matches P*A
        const LU = new Float64Array(n * n);
        for (let j = 0; j < n; j++) {
          for (let i = 0; i < n; i++) {
            let s = 0;
            const kmax = Math.min(i, j);
            for (let k = 0; k <= kmax; k++) {
              const l_ik = k === i ? 1 : k < i ? res.LU[i + k * n] : 0;
              const u_kj = k <= j ? res.LU[k + j * n] : 0;
              s += l_ik * u_kj;
            }
            LU[i + j * n] = s;
          }
        }
        assertClose(LU, resA, TOL, "lu reconstruction");
      });
    }
  });

  describe("linsolve", () => {
    for (const n of SIZES) {
      it(`${n}x${n} nrhs=1`, () => {
        const A = wellConditioned(n, n * 19);
        const B = randomMatrix(n, 1, n * 23);
        const ref = ts.linsolve!(A, n, n, B, 1);
        const res = flame.linsolve!(A, n, n, B, 1);
        assertClose(res, ref, TOL, "linsolve");
      });
    }
  });

  describe("chol", () => {
    for (const n of SIZES) {
      it(`${n}x${n} upper`, () => {
        const A = randomSPD(n, n * 29);
        const ref = ts.chol!(A, n, true);
        const res = flame.chol!(A, n, true);
        expect(res.info).toBe(0);
        assertClose(res.R, ref.R, TOL, "chol upper");
      });

      it(`${n}x${n} lower`, () => {
        const A = randomSPD(n, n * 31);
        const ref = ts.chol!(A, n, false);
        const res = flame.chol!(A, n, false);
        expect(res.info).toBe(0);
        assertClose(res.R, ref.R, TOL, "chol lower");
      });
    }
  });

  describe("qr (fallback to ts-lapack)", () => {
    it("128x128", () => {
      const n = 128;
      const A = randomMatrix(n, n, 37);
      const ref = ts.qr!(A, n, n, true, true);
      const res = flame.qr!(A, n, n, true, true);
      assertClose(res.Q, ref.Q, TOL, "qr Q");
      assertClose(res.R, ref.R, TOL, "qr R");
    });
  });

  describe("eig (fallback to ts-lapack)", () => {
    it("64x64", () => {
      const n = 64;
      const A = randomMatrix(n, n, 41);
      const ref = ts.eig!(A, n, false, true, true);
      const res = flame.eig!(A, n, false, true, true);
      assertClose(res.wr, ref.wr, TOL, "eig wr");
      assertClose(res.wi, ref.wi, TOL, "eig wi");
    });
  });

  describe("svd (fallback to ts-lapack)", () => {
    it("64x64", () => {
      const n = 64;
      const A = randomMatrix(n, n, 43);
      const ref = ts.svd!(A, n, n, true, true);
      const res = flame.svd!(A, n, n, true, true);
      assertClose(res.S!, ref.S!, TOL, "svd S");
    });
  });
});
