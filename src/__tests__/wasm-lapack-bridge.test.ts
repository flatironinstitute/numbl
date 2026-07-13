import { describe, it, expect, afterEach } from "vitest";
import { executeCode } from "../numbl-core/executeCode.js";
import { isRuntimeTensor } from "../numbl-core/runtime/types.js";
import {
  setLapackBridge,
  type LapackBridge,
} from "../numbl-core/native/lapack-bridge.js";
import {
  makeWasmMatmul,
  buildLapackBridgeFromWasm,
  installWasmLapackBridge,
  setMatmulAccelerator,
  validateManifest,
  ACCELERATOR_ABI,
  WASM_MATMUL_MIN_WORK,
  type WasmLapackExports,
} from "../numbl-core/native/wasm-lapack-bridge.js";

// A JS-backed stand-in for an instantiated wasm module implementing the v1
// ABI over a real WebAssembly.Memory. Exercises the exact marshaling code
// (malloc offsets, HEAPF64 copy in/out) that will run against the real
// libFLAME/BLIS wasm in Phase 3, without needing a wasm toolchain here.
function makeFakeWasm(): { exports: WasmLapackExports; calls: () => number } {
  const memory = new WebAssembly.Memory({ initial: 256 }); // 16 MiB
  let brk = 1024; // leave 0 as the "null" pointer
  let matmulCalls = 0;
  const exports: WasmLapackExports = {
    memory,
    numbl_malloc(nbytes: number): number {
      const p = brk;
      brk += (nbytes + 7) & ~7; // 8-byte align
      return p;
    },
    numbl_free(): void {
      /* bump allocator — no-op */
    },
    numbl_matmul_f64_colmajor(aPtr, bPtr, cPtr, m, k, n): void {
      matmulCalls++;
      const A = new Float64Array(memory.buffer, aPtr, m * k);
      const B = new Float64Array(memory.buffer, bPtr, k * n);
      const C = new Float64Array(memory.buffer, cPtr, m * n);
      for (let j = 0; j < n; j++) {
        for (let i = 0; i < m; i++) {
          let s = 0;
          for (let p = 0; p < k; p++) s += A[i + p * m] * B[p + j * k];
          C[i + j * m] = s;
        }
      }
    },
  };
  return { exports, calls: () => matmulCalls };
}

function matmulManifest(): {
  abi: number;
  ops: { matmul: { precision: string[] } };
  wasm: string;
} {
  return {
    abi: ACCELERATOR_ABI,
    ops: { matmul: { precision: ["f64"] } },
    wasm: "b.wasm",
  };
}

/** Column-major reference matmul for cross-checking. */
function refMatmul(
  a: number[],
  m: number,
  k: number,
  b: number[],
  n: number
): number[] {
  const c = new Array(m * n).fill(0);
  for (let j = 0; j < n; j++)
    for (let i = 0; i < m; i++) {
      let s = 0;
      for (let p = 0; p < k; p++) s += a[i + p * m] * b[p + j * k];
      c[i + j * m] = s;
    }
  return c;
}

function tensorVar(
  result: ReturnType<typeof executeCode>,
  name: string
): number[] {
  const v = result.variableValues[name];
  if (!isRuntimeTensor(v)) throw new Error(`${name} is not a tensor`);
  return Array.from(v.data);
}

afterEach(() => {
  // Reset both seams so tests don't bleed into each other or other files.
  setMatmulAccelerator(null);
  setLapackBridge(null as unknown as LapackBridge);
});

describe("wasm-lapack-bridge: manifest validation", () => {
  it("accepts a well-formed matmul manifest", () => {
    expect(() => validateManifest(matmulManifest())).not.toThrow();
  });
  it("rejects an ABI mismatch", () => {
    expect(() =>
      validateManifest({ ...matmulManifest(), abi: ACCELERATOR_ABI + 1 })
    ).toThrow(/ABI/);
  });
  it("rejects a missing wasm filename", () => {
    expect(() => validateManifest({ abi: ACCELERATOR_ABI, ops: {} })).toThrow(
      /wasm/
    );
  });
  it("rejects a non-object", () => {
    expect(() => validateManifest(null)).toThrow();
  });
});

describe("wasm-lapack-bridge: marshaling (makeWasmMatmul)", () => {
  it("copies column-major data in/out and returns A*B", () => {
    const { exports } = makeFakeWasm();
    const matmul = makeWasmMatmul(exports);
    // A: 2×3, B: 3×2 (column-major)
    const a = [1, 4, 2, 5, 3, 6]; // rows [1 2 3; 4 5 6]
    const b = [7, 9, 11, 8, 10, 12]; // rows [7 8; 9 10; 11 12]
    const out = matmul(new Float64Array(a), 2, 3, new Float64Array(b), 2);
    expect(Array.from(out)).toEqual(refMatmul(a, 2, 3, b, 2));
  });

  it("only reads m*k / k*n elements from over-long inputs", () => {
    const { exports } = makeFakeWasm();
    const matmul = makeWasmMatmul(exports);
    const a = new Float64Array([1, 2, 3, 4, /* garbage */ 999, 999]);
    const b = new Float64Array([5, 6, 7, 8, /* garbage */ 999]);
    const out = matmul(a, 2, 2, b, 2);
    expect(Array.from(out)).toEqual(
      refMatmul([1, 2, 3, 4], 2, 2, [5, 6, 7, 8], 2)
    );
  });

  it("buildLapackBridgeFromWasm exposes matmul only when advertised", () => {
    const { exports } = makeFakeWasm();
    expect(
      buildLapackBridgeFromWasm(exports, matmulManifest()).matmul
    ).toBeDefined();
    const noMatmul = buildLapackBridgeFromWasm(exports, {
      abi: ACCELERATOR_ABI,
      ops: {},
      wasm: "b.wasm",
    });
    expect(noMatmul.matmul).toBeUndefined();
  });
});

describe("wasm-lapack-bridge: pipeline seams", () => {
  // Build A (60×60), B (60×60) deterministically inside the script and
  // multiply. m*n*k = 216000 > WASM_MATMUL_MIN_WORK, so the accelerator is
  // not declined for size.
  const script = `A = reshape(1:3600, 60, 60); B = reshape(3600:-1:1, 60, 60); C = A*B;`;

  it("opt 1 (JS-JIT) routes matmul through $matmulAccel", () => {
    // Sentinel accelerator: returns all-7s. If (and only if) the JIT path
    // consulted the hook, C comes back all 7s.
    setMatmulAccelerator((_a, m, _k, _b, n) => new Float64Array(m * n).fill(7));
    const c = tensorVar(executeCode(script, { optimization: "1" }), "C");
    expect(c.length).toBe(3600);
    expect(c.every(x => x === 7)).toBe(true);
  });

  it("opt 0 (interpreter) routes matmul through the LapackBridge", () => {
    // Sentinel bridge: returns all-13s via getEffectiveBridge.
    setLapackBridge({
      matmul: (_a, m, _k, _b, n) => new Float64Array(m * n).fill(13),
    });
    const c = tensorVar(executeCode(script, { optimization: "0" }), "C");
    expect(c.length).toBe(3600);
    expect(c.every(x => x === 13)).toBe(true);
  });

  it("produces the correct product through opt 1 using the real (fake) wasm", () => {
    const { exports, calls } = makeFakeWasm();
    installWasmLapackBridge(exports, matmulManifest());
    const res = executeCode(script, { optimization: "1" });
    const c = tensorVar(res, "C");
    // Cross-check against a JS reference of the same script data.
    const a = Array.from({ length: 3600 }, (_, i) => i + 1);
    const b = Array.from({ length: 3600 }, (_, i) => 3600 - i);
    expect(c).toEqual(refMatmul(a, 60, 60, b, 60));
    expect(calls()).toBeGreaterThan(0); // wasm kernel actually ran
  });

  it("declines small matmuls (below the work threshold) and still returns correct results", () => {
    const { exports, calls } = makeFakeWasm();
    installWasmLapackBridge(exports, matmulManifest());
    // 3×3 * 3×3 → 27 mults, far below WASM_MATMUL_MIN_WORK.
    expect(3 * 3 * 3).toBeLessThan(WASM_MATMUL_MIN_WORK);
    const small = `A = reshape(1:9, 3, 3); B = reshape(9:-1:1, 3, 3); C = A*B;`;
    const c = tensorVar(executeCode(small, { optimization: "1" }), "C");
    const a = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    const b = [9, 8, 7, 6, 5, 4, 3, 2, 1];
    expect(c).toEqual(refMatmul(a, 3, 3, b, 3));
    expect(calls()).toBe(0); // declined by the size threshold → JS loop used
  });
});
