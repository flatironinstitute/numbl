import { getTsLapackBridge } from "../numbl-core/native/ts-lapack-bridge.js";

export type BenchCapability =
  | "matmul"
  | "inv"
  | "linsolve"
  | "fft1dComplex"
  | "fftAlongDim";

export type ComplexArray = {
  re: Float64Array;
  im: Float64Array;
};

export type BenchValue = Float64Array | ComplexArray;

export type BenchKernelBridge = {
  matmul?: (
    A: Float64Array,
    m: number,
    k: number,
    B: Float64Array,
    n: number
  ) => Float64Array;
  inv?: (data: Float64Array, n: number) => Float64Array;
  linsolve?: (
    A: Float64Array,
    m: number,
    n: number,
    B: Float64Array,
    nrhs: number
  ) => Float64Array;
  fft1dComplex?: (
    re: Float64Array,
    im: Float64Array,
    n: number,
    inverse: boolean
  ) => ComplexArray;
  fftAlongDim?: (
    re: Float64Array,
    im: Float64Array | null,
    shape: number[],
    dim: number,
    n: number,
    inverse: boolean
  ) => ComplexArray;
};

export interface BackendBenchScenario {
  id: string;
  label: string;
  capability: BenchCapability;
  defaultWarmup: number;
  defaultIterations: number;
  tolerance: number;
  execute: (backend: BenchKernelBridge) => BenchValue;
}

export const QUICK_BACKEND_BENCH_SCENARIO_IDS = new Set([
  "matmul-128",
  "inv-64",
  "linsolve-128",
  "fft1d-4096",
  "fftAlongDim-1024x64-d0",
]);

const F64_EPSILON = Number.EPSILON;
const tsBridge = getTsLapackBridge();

function scaledMachineTolerance(scale: number): number {
  return F64_EPSILON * scale;
}

function matrixOneNorm(data: Float64Array, m: number, n: number): number {
  let maxColumnSum = 0;
  for (let col = 0; col < n; col++) {
    let sum = 0;
    for (let row = 0; row < m; row++) {
      sum += Math.abs(data[row + col * m]);
    }
    if (sum > maxColumnSum) {
      maxColumnSum = sum;
    }
  }
  return maxColumnSum;
}

function estimateSquareConditionNumber(data: Float64Array, n: number): number {
  try {
    const inverse = tsBridge.inv(data, n);
    return matrixOneNorm(data, n, n) * matrixOneNorm(inverse, n, n);
  } catch {
    return 1;
  }
}

function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeDenseMatrix(m: number, n: number, seed: number): Float64Array {
  const rand = mulberry32(seed);
  const out = new Float64Array(m * n);
  for (let col = 0; col < n; col++) {
    for (let row = 0; row < m; row++) {
      const idx = row + col * m;
      out[idx] =
        Math.sin((row + 1) * 0.31) +
        Math.cos((col + 1) * 0.17) +
        (rand() - 0.5) * 0.2;
    }
  }
  return out;
}

function makeDiagonallyDominantSquare(n: number, seed: number): Float64Array {
  const rand = mulberry32(seed);
  const out = new Float64Array(n * n);
  for (let col = 0; col < n; col++) {
    for (let row = 0; row < n; row++) {
      const idx = row + col * n;
      const noise = (rand() - 0.5) * 0.25;
      out[idx] = row === col ? n + 1 + noise : noise;
    }
  }
  return out;
}

function makeComplexSignal(length: number, seed: number): ComplexArray {
  const rand = mulberry32(seed);
  const re = new Float64Array(length);
  const im = new Float64Array(length);
  for (let i = 0; i < length; i++) {
    const x = (2 * Math.PI * i) / length;
    re[i] = Math.sin(3 * x) + 0.25 * Math.cos(11 * x) + (rand() - 0.5) * 0.01;
    im[i] = Math.cos(5 * x) - 0.2 * Math.sin(7 * x) + (rand() - 0.5) * 0.01;
  }
  return { re, im };
}

function makeComplexTensor(shape: number[], seed: number): ComplexArray {
  const length = shape.reduce((acc, dim) => acc * dim, 1);
  return makeComplexSignal(length, seed);
}

function createMatmulScenario(
  m: number,
  k: number,
  n: number,
  warmup: number,
  iterations: number
): BackendBenchScenario {
  const A = makeDenseMatrix(m, k, m * 101 + k * 17 + n * 7);
  const B = makeDenseMatrix(k, n, m * 37 + k * 13 + n * 29);
  return {
    id: `matmul-${m}`,
    label: `matmul ${m}x${k} * ${k}x${n}`,
    capability: "matmul",
    defaultWarmup: warmup,
    defaultIterations: iterations,
    tolerance: scaledMachineTolerance(Math.max(2048, m * k * 2)),
    execute: backend => {
      if (typeof backend.matmul !== "function") {
        throw new Error("backend does not support matmul");
      }
      return backend.matmul(A, m, k, B, n);
    },
  };
}

function createInvScenario(
  n: number,
  warmup: number,
  iterations: number
): BackendBenchScenario {
  const A = makeDiagonallyDominantSquare(n, n * 97 + 11);
  const condition = estimateSquareConditionNumber(A, n);
  return {
    id: `inv-${n}`,
    label: `inv ${n}x${n}`,
    capability: "inv",
    defaultWarmup: warmup,
    defaultIterations: iterations,
    tolerance: scaledMachineTolerance(
      Math.max(4096, Math.ceil(condition * n * 256))
    ),
    execute: backend => {
      if (typeof backend.inv !== "function") {
        throw new Error("backend does not support inv");
      }
      return backend.inv(A, n);
    },
  };
}

function createLinsolveScenario(
  n: number,
  nrhs: number,
  warmup: number,
  iterations: number
): BackendBenchScenario {
  const A = makeDiagonallyDominantSquare(n, n * 71 + nrhs * 13);
  const B = makeDenseMatrix(n, nrhs, n * 53 + nrhs * 19);
  const condition = estimateSquareConditionNumber(A, n);
  return {
    id: `linsolve-${n}`,
    label: `linsolve ${n}x${n} rhs=${nrhs}`,
    capability: "linsolve",
    defaultWarmup: warmup,
    defaultIterations: iterations,
    tolerance: scaledMachineTolerance(
      Math.max(4096, Math.ceil(condition * n * nrhs * 256))
    ),
    execute: backend => {
      if (typeof backend.linsolve !== "function") {
        throw new Error("backend does not support linsolve");
      }
      return backend.linsolve(A, n, n, B, nrhs);
    },
  };
}

function createFft1dScenario(
  n: number,
  warmup: number,
  iterations: number
): BackendBenchScenario {
  const signal = makeComplexSignal(n, n * 19 + 5);
  return {
    id: `fft1d-${n}`,
    label: `fft1d complex n=${n}`,
    capability: "fft1dComplex",
    defaultWarmup: warmup,
    defaultIterations: iterations,
    tolerance: scaledMachineTolerance(Math.max(4096, n * 1024)),
    execute: backend => {
      if (typeof backend.fft1dComplex !== "function") {
        throw new Error("backend does not support fft1dComplex");
      }
      return backend.fft1dComplex(signal.re, signal.im, n, false);
    },
  };
}

function createFftAlongDimScenario(
  shape: number[],
  dim: number,
  n: number,
  warmup: number,
  iterations: number
): BackendBenchScenario {
  const tensor = makeComplexTensor(
    shape,
    shape.reduce((acc, value) => acc * 31 + value, 17)
  );
  const shapeLabel = `${shape.join("x")}-d${dim}`;
  return {
    id: `fftAlongDim-${shapeLabel}`,
    label: `fftAlongDim shape=${shape.join("x")} dim=${dim} n=${n}`,
    capability: "fftAlongDim",
    defaultWarmup: warmup,
    defaultIterations: iterations,
    tolerance: scaledMachineTolerance(
      Math.max(4096, shape.reduce((acc, value) => acc * value, 1) * 64)
    ),
    execute: backend => {
      if (typeof backend.fftAlongDim !== "function") {
        throw new Error("backend does not support fftAlongDim");
      }
      return backend.fftAlongDim(tensor.re, tensor.im, shape, dim, n, false);
    },
  };
}

export function buildBackendBenchScenarios(): BackendBenchScenario[] {
  return [
    createMatmulScenario(128, 128, 128, 4, 60),
    createMatmulScenario(256, 256, 256, 3, 18),
    createMatmulScenario(512, 512, 512, 2, 5),
    createInvScenario(64, 3, 24),
    createInvScenario(128, 2, 10),
    createInvScenario(256, 1, 4),
    createLinsolveScenario(128, 4, 3, 20),
    createLinsolveScenario(256, 4, 2, 8),
    createFft1dScenario(4096, 4, 40),
    createFft1dScenario(16384, 2, 12),
    createFftAlongDimScenario([1024, 64], 0, 1024, 2, 8),
  ];
}

export function percentile(sortedValues: number[], fraction: number): number {
  if (sortedValues.length === 0) return 0;
  const index = Math.min(
    sortedValues.length - 1,
    Math.max(0, Math.ceil(sortedValues.length * fraction) - 1)
  );
  return sortedValues[index];
}

export function percentileFromSamples(
  samples: readonly number[] | undefined,
  fraction: number
): number | undefined {
  if (!samples || samples.length === 0) return undefined;
  return percentile([...samples].sort((a, b) => a - b), fraction);
}

function isComplexValue(value: BenchValue): value is ComplexArray {
  return (
    typeof value === "object" &&
    "re" in value &&
    value.re instanceof Float64Array &&
    "im" in value &&
    value.im instanceof Float64Array
  );
}

function digestFloat64Array(values: Float64Array): string {
  if (values.length === 0) return "len=0";
  const sample = [0, Math.floor(values.length / 2), values.length - 1]
    .filter((index, i, all) => all.indexOf(index) === i)
    .map(index => values[index].toFixed(6))
    .join(", ");
  let checksum = 0;
  for (
    let i = 0;
    i < values.length;
    i += Math.max(1, Math.floor(values.length / 16))
  ) {
    checksum += values[i];
  }
  return `len=${values.length} sample=[${sample}] checksum=${checksum.toFixed(6)}`;
}

export function digestBenchValue(value: BenchValue): string {
  if (isComplexValue(value)) {
    return [
      `re:${digestFloat64Array(value.re)}`,
      `im:${digestFloat64Array(value.im)}`,
    ].join(" | ");
  }
  return digestFloat64Array(value);
}

function consumeFloat64Array(values: Float64Array): number {
  if (values.length === 0) return 0;
  const mid = Math.floor(values.length / 2);
  return values[0] + values[mid] + values[values.length - 1];
}

export function consumeBenchValue(value: BenchValue): number {
  if (isComplexValue(value)) {
    return consumeFloat64Array(value.re) + consumeFloat64Array(value.im);
  }
  return consumeFloat64Array(value);
}

function compareFloat64Arrays(
  reference: Float64Array,
  actual: Float64Array,
  tolerance: number
): number {
  if (reference.length !== actual.length) {
    throw new Error(
      `benchmark output length mismatch (${reference.length} vs ${actual.length})`
    );
  }
  let maxAbsDiff = 0;
  for (let i = 0; i < reference.length; i++) {
    const absDiff = Math.abs(reference[i] - actual[i]);
    if (absDiff > maxAbsDiff) maxAbsDiff = absDiff;
    const scale = Math.max(1, Math.abs(reference[i]), Math.abs(actual[i]));
    if (absDiff > tolerance * scale) {
      throw new Error(
        `benchmark validation failed at index ${i}: diff=${absDiff} tolerance=${tolerance}`
      );
    }
  }
  return maxAbsDiff;
}

export function compareBenchValues(
  reference: BenchValue,
  actual: BenchValue,
  tolerance: number
): number {
  if (isComplexValue(reference) !== isComplexValue(actual)) {
    throw new Error("benchmark output shape mismatch");
  }
  if (isComplexValue(reference) && isComplexValue(actual)) {
    return Math.max(
      compareFloat64Arrays(reference.re, actual.re, tolerance),
      compareFloat64Arrays(reference.im, actual.im, tolerance)
    );
  }
  if (reference instanceof Float64Array && actual instanceof Float64Array) {
    return compareFloat64Arrays(reference, actual, tolerance);
  }
  throw new Error("unsupported benchmark value type");
}
