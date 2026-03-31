/**
 * Allocation benchmark scenarios.
 *
 * Measures allocation/deallocation throughput for Float64Array under
 * different workload patterns: small churn, medium churn, large blocks,
 * interleaved lifecycles, and burst-then-free.
 */

export interface AllocBenchBackend {
  id: string;
  label: string;
  alloc: (n: number) => Float64Array;
  free: (arr: Float64Array) => void;
}

export interface AllocBenchScenario {
  id: string;
  label: string;
  defaultWarmup: number;
  defaultIterations: number;
  /** The benchmark function. Must consume results to prevent DCE. */
  execute: (backend: AllocBenchBackend) => number;
}

export const QUICK_ALLOC_BENCH_SCENARIO_IDS = new Set([
  "small-churn",
  "medium-churn",
  "interleaved",
]);

/** Prevent dead-code elimination by consuming a Float64Array. */
function consume(arr: Float64Array): number {
  if (arr.length === 0) return 0;
  return arr[0] + arr[arr.length - 1];
}

function createSmallChurnScenario(): AllocBenchScenario {
  const sizes = [1, 4, 9, 16, 25, 36, 64, 100, 144, 196, 256];
  return {
    id: "small-churn",
    label: "1000 alloc/free cycles, 1–256 elements",
    defaultWarmup: 4,
    defaultIterations: 80,
    execute: backend => {
      let sink = 0;
      for (let i = 0; i < 1000; i++) {
        const n = sizes[i % sizes.length];
        const arr = backend.alloc(n);
        arr[0] = i;
        sink += consume(arr);
        backend.free(arr);
      }
      return sink;
    },
  };
}

function createMediumChurnScenario(): AllocBenchScenario {
  const sizes = [4096, 8192, 16384, 32768, 65536];
  return {
    id: "medium-churn",
    label: "100 alloc/free cycles, 4K–64K elements",
    defaultWarmup: 3,
    defaultIterations: 40,
    execute: backend => {
      let sink = 0;
      for (let i = 0; i < 100; i++) {
        const n = sizes[i % sizes.length];
        const arr = backend.alloc(n);
        arr[0] = i;
        sink += consume(arr);
        backend.free(arr);
      }
      return sink;
    },
  };
}

function createLargeAllocScenario(): AllocBenchScenario {
  return {
    id: "large-alloc",
    label: "10 alloc/free cycles, 1M elements",
    defaultWarmup: 2,
    defaultIterations: 20,
    execute: backend => {
      let sink = 0;
      for (let i = 0; i < 10; i++) {
        const arr = backend.alloc(1_000_000);
        arr[0] = i;
        sink += consume(arr);
        backend.free(arr);
      }
      return sink;
    },
  };
}

function createInterleavedScenario(): AllocBenchScenario {
  const sizes = [64, 256, 1024, 4096, 16384];
  return {
    id: "interleaved",
    label: "alloc 100, free 50, alloc 50, free all",
    defaultWarmup: 3,
    defaultIterations: 30,
    execute: backend => {
      let sink = 0;
      const live: Float64Array[] = [];
      // Allocate 100
      for (let i = 0; i < 100; i++) {
        const arr = backend.alloc(sizes[i % sizes.length]);
        arr[0] = i;
        sink += consume(arr);
        live.push(arr);
      }
      // Free oldest 50
      for (let i = 0; i < 50; i++) {
        backend.free(live[i]);
      }
      // Allocate 50 more
      for (let i = 0; i < 50; i++) {
        const arr = backend.alloc(sizes[i % sizes.length]);
        arr[0] = i + 100;
        sink += consume(arr);
        live[i] = arr;
      }
      // Free all
      for (let i = 0; i < 100; i++) {
        backend.free(live[i]);
      }
      return sink;
    },
  };
}

function createBurstThenFreeScenario(): AllocBenchScenario {
  return {
    id: "burst-then-free",
    label: "alloc 1000 small, then free all",
    defaultWarmup: 3,
    defaultIterations: 40,
    execute: backend => {
      let sink = 0;
      const blocks: Float64Array[] = new Array(1000);
      for (let i = 0; i < 1000; i++) {
        blocks[i] = backend.alloc(64);
        blocks[i][0] = i;
      }
      for (let i = 0; i < 1000; i++) {
        sink += consume(blocks[i]);
        backend.free(blocks[i]);
      }
      return sink;
    },
  };
}

function createMatmulChainScenario(): AllocBenchScenario {
  const n = 64;
  return {
    id: "matmul-chain",
    label: "simulate A*B*C*D intermediates (64x64)",
    defaultWarmup: 3,
    defaultIterations: 30,
    execute: backend => {
      let sink = 0;
      // Simulate 4-matrix chain: allocate intermediates, free old ones
      let current = backend.alloc(n * n);
      current[0] = 1;
      for (let step = 0; step < 3; step++) {
        const next = backend.alloc(n * n);
        next[0] = current[0] + 1;
        sink += consume(next);
        backend.free(current);
        current = next;
      }
      sink += consume(current);
      backend.free(current);
      return sink;
    },
  };
}

export function buildAllocBenchScenarios(): AllocBenchScenario[] {
  return [
    createSmallChurnScenario(),
    createMediumChurnScenario(),
    createLargeAllocScenario(),
    createInterleavedScenario(),
    createBurstThenFreeScenario(),
    createMatmulChainScenario(),
  ];
}

export function createJsAllocBackend(): AllocBenchBackend {
  return {
    id: "js",
    label: "JS Float64Array (baseline)",
    alloc: (n: number) => new Float64Array(n),
    free: () => {},
  };
}
