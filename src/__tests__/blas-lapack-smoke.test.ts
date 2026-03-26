import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

function createImportObject(): WebAssembly.Imports {
  return {
    wasi_snapshot_preview1: {
      fd_write: () => 0,
      fd_read: () => 0,
      fd_close: () => 0,
      fd_seek: () => 0,
      fd_fdstat_get: () => 0,
      proc_exit: () => {},
      environ_sizes_get: () => 0,
      environ_get: () => 0,
      clock_time_get: () => 0,
      args_sizes_get: () => 0,
      args_get: () => 0,
    },
    env: {
      emscripten_notify_memory_growth: () => {},
    },
  };
}

function align8(bytes: number): number {
  return (bytes + 7) & ~7;
}

describe("blas-lapack wasm smoke", () => {
  const repoRoot = process.cwd();
  const wasmPath = join(repoRoot, "public", "wasm-kernels", "blas-lapack.wasm");
  const maybeIt = existsSync(wasmPath) ? it : it.skip;

  maybeIt("instantiates and executes small linalg kernels", async () => {
    const bytes = readFileSync(wasmPath);
    const { instance } = await WebAssembly.instantiate(
      bytes,
      createImportObject()
    );
    const exports = instance.exports as Record<string, unknown>;
    const initialize = exports._initialize as (() => void) | undefined;
    if (typeof initialize === "function") {
      initialize();
    }
    const memory = exports.memory as WebAssembly.Memory;
    const malloc = exports.malloc as (bytes: number) => number;
    const free = exports.free as (ptr: number) => void;
    const matmul = exports.numbl_matmul_f64 as (...args: number[]) => number;
    const inv = exports.numbl_inv_f64 as (...args: number[]) => number;
    const linsolve = exports.numbl_linsolve_f64 as (...args: number[]) => number;

    const writeF64 = (values: number[]) => {
      const bytesNeeded = align8(values.length * Float64Array.BYTES_PER_ELEMENT);
      const ptr = malloc(bytesNeeded);
      new Float64Array(memory.buffer, ptr, values.length).set(values);
      return ptr;
    };

    const readF64 = (ptr: number, length: number) =>
      Array.from(new Float64Array(memory.buffer, ptr, length));

    const closeTo = (actual: number[], expected: number[]) => {
      expect(actual.length).toBe(expected.length);
      for (let i = 0; i < actual.length; i++) {
        expect(actual[i]).toBeCloseTo(expected[i], 8);
      }
    };

    const aPtr = writeF64([1, 2, 3, 4]);
    const bPtr = writeF64([5, 6, 7, 8]);
    const cPtr = malloc(align8(4 * Float64Array.BYTES_PER_ELEMENT));
    expect(matmul(aPtr, 2, 2, bPtr, 2, cPtr)).toBe(0);
    expect(readF64(cPtr, 4)).toEqual([23, 34, 31, 46]);

    const invInputPtr = writeF64([4, 2, 7, 6]);
    const invOutPtr = malloc(align8(4 * Float64Array.BYTES_PER_ELEMENT));
    expect(inv(invInputPtr, 2, invOutPtr)).toBe(0);
    closeTo(readF64(invOutPtr, 4), [0.6, -0.2, -0.7, 0.4]);

    const solveAPtr = writeF64([4, 2, 7, 6]);
    const solveBPtr = writeF64([1, 0]);
    const solveOutPtr = malloc(align8(2 * Float64Array.BYTES_PER_ELEMENT));
    expect(linsolve(solveAPtr, 2, 2, solveBPtr, 1, solveOutPtr)).toBe(0);
    closeTo(readF64(solveOutPtr, 2), [0.6, -0.2]);

    for (const ptr of [
      aPtr,
      bPtr,
      cPtr,
      invInputPtr,
      invOutPtr,
      solveAPtr,
      solveBPtr,
      solveOutPtr,
    ]) {
      free(ptr);
    }
  });
});
