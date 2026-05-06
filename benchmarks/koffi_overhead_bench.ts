/*
 * koffi_overhead_bench.ts — measure koffi call overhead and JS↔C
 * crossover for a fused `y = exp(1 + sqrt(x))` tensor kernel.
 *
 * For each tensor size N, times four paths:
 *   - JS:         a plain JS loop over Float64Array — this is the
 *                 ceiling numbl's JS-JIT lowers to for this kernel.
 *   - C-fused:    one koffi call into the fused C loop.
 *   - koffi-noop: a koffi call into an empty C function (no args).
 *   - koffi-args: a koffi call that passes (n, x, y) but does no work.
 *
 * x and y live in JS-owned Float64Arrays; koffi binds them directly as
 * `double *` (no copy) — the same zero-copy contract numbl's C-JIT uses
 * for tensor params.
 *
 * Usage:
 *   npx tsx benchmarks/koffi_overhead_bench.ts
 *
 * Optional flags:
 *   --sizes=1,10,100,1000,...     override default N sweep
 *   --target-ms=300                target per-measurement wall time
 *   --no-fast-math                 drop -ffast-math (keeps sqrt/exp scalar)
 */

import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { allocFloat64Array } from "../src/numbl-core/executors/jsJit/helpers/alloc";

const require = createRequire(import.meta.url);
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const koffi = require("koffi") as any;

// ── CLI parsing ────────────────────────────────────────────────────────

function parseArgs(): {
  sizes: number[];
  targetMs: number;
  fastMath: boolean;
} {
  const defaults = {
    sizes: [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000],
    targetMs: 300,
    fastMath: true,
  };
  const out = { ...defaults };
  for (const raw of process.argv.slice(2)) {
    if (raw.startsWith("--sizes=")) {
      out.sizes = raw
        .slice("--sizes=".length)
        .split(",")
        .map(s => Number(s))
        .filter(n => Number.isFinite(n) && n > 0);
    } else if (raw.startsWith("--target-ms=")) {
      out.targetMs = Number(raw.slice("--target-ms=".length));
    } else if (raw === "--no-fast-math") {
      out.fastMath = false;
    } else {
      console.error(`unknown arg: ${raw}`);
      process.exit(2);
    }
  }
  return out;
}

// ── Compile the C kernel ───────────────────────────────────────────────

function compileKernel(fastMath: boolean): string {
  const here = dirname(fileURLToPath(import.meta.url));
  const srcPath = join(here, "koffi_overhead_bench.c");
  const src = readFileSync(srcPath, "utf-8");
  const cc = process.env.NUMBL_CC || "cc";

  const flags = [
    "-O2",
    "-fPIC",
    "-shared",
    "-std=c11",
    "-march=native",
    "-fopenmp-simd",
    "-fno-math-errno",
  ];
  if (fastMath) flags.push("-ffast-math");

  // Cache by (source, cc, flags) hash so repeated runs skip the recompile.
  let ccVersion = "";
  try {
    ccVersion = execFileSync(cc, ["--version"], {
      stdio: ["ignore", "pipe", "ignore"],
      timeout: 5_000,
    })
      .toString()
      .split("\n")[0]
      .trim();
  } catch {
    console.error(`error: C compiler '${cc}' not found (set NUMBL_CC)`);
    process.exit(1);
  }

  const h = createHash("sha256");
  h.update(src);
  h.update(cc);
  h.update(ccVersion);
  h.update(flags.join(" "));
  const hash = h.digest("hex").slice(0, 16);

  const cacheDir = join(tmpdir(), "numbl-koffi-overhead-bench");
  mkdirSync(cacheDir, { recursive: true });
  const soPath = join(cacheDir, `${hash}.so`);
  if (existsSync(soPath)) return soPath;

  const buildDir = mkdtempSync(join(tmpdir(), "numbl-koffi-bench-"));
  const buildSrc = join(buildDir, "src.c");
  const buildOut = join(buildDir, "out.so");
  writeFileSync(buildSrc, src);

  try {
    execFileSync(cc, [...flags, "-o", buildOut, buildSrc, "-lm"], {
      stdio: ["ignore", "ignore", "pipe"],
      timeout: 60_000,
    });
  } catch (e) {
    const err = e as { stderr?: Buffer; message: string };
    const msg = err.stderr ? err.stderr.toString() : err.message;
    console.error(`cc failed:\n${msg}`);
    process.exit(1);
  }

  writeFileSync(soPath, readFileSync(buildOut));
  console.error(`compiled: ${cc} ${flags.join(" ")} -> ${soPath}`);
  return soPath;
}

// ── Timing helpers ─────────────────────────────────────────────────────

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >>> 1;
  return s.length % 2 ? s[m] : 0.5 * (s[m - 1] + s[m]);
}

/**
 * Run `fn()` enough times to fill roughly `targetMs` of wall-clock work,
 * returning the per-call time in seconds (median of 5 outer samples).
 * A quick pilot sets `reps` so total time stays predictable.
 */
function timePerCall(
  fn: () => void,
  targetMs: number,
  maxReps = 20_000_000
): { perCallSec: number; reps: number } {
  // Warm-up + pilot: 10 calls to get a ballpark rate.
  for (let i = 0; i < 10; i++) fn();
  let t0 = performance.now();
  let pilotReps = 10;
  for (let i = 0; i < pilotReps; i++) fn();
  let dt = (performance.now() - t0) / 1000;
  // Ensure the pilot itself took at least 2 ms so the rate estimate is stable.
  while (dt < 0.002 && pilotReps < maxReps) {
    pilotReps *= 4;
    t0 = performance.now();
    for (let i = 0; i < pilotReps; i++) fn();
    dt = (performance.now() - t0) / 1000;
  }
  const perCall = dt / pilotReps;
  let reps = Math.max(1, Math.ceil(targetMs / 1000 / perCall));
  if (reps > maxReps) reps = maxReps;

  const samples: number[] = [];
  for (let s = 0; s < 5; s++) {
    const start = performance.now();
    for (let i = 0; i < reps; i++) fn();
    const elapsed = (performance.now() - start) / 1000;
    samples.push(elapsed / reps);
  }
  return { perCallSec: median(samples), reps };
}

function fmtTime(sec: number): string {
  if (sec < 1e-6) return `${(sec * 1e9).toFixed(1)} ns`;
  if (sec < 1e-3) return `${(sec * 1e6).toFixed(2)} µs`;
  if (sec < 1) return `${(sec * 1e3).toFixed(2)} ms`;
  return `${sec.toFixed(3)} s`;
}

// ── Main ───────────────────────────────────────────────────────────────

function main(): void {
  const { sizes, targetMs, fastMath } = parseArgs();

  const soPath = compileKernel(fastMath);
  const lib = koffi.load(soPath);
  const fused = lib.func("void numbl_bench_fused(int64_t, double *, double *)");
  const noop = lib.func("void numbl_bench_noop(void)");
  const noopArgs = lib.func(
    "void numbl_bench_noop_args(int64_t, double *, double *)"
  );

  // Parity check at a moderate N: JS and C should agree to ~FP rounding.
  {
    const N = 10_000;
    const x = allocFloat64Array(N);
    for (let i = 0; i < N; i++) x[i] = (i + 1) / N; // (0, 1]
    const yJs = allocFloat64Array(N);
    const yC = allocFloat64Array(N);
    for (let i = 0; i < N; i++) yJs[i] = Math.exp(1 + Math.sqrt(x[i]));
    fused(N, x, yC);
    let maxAbs = 0;
    for (let i = 0; i < N; i++) {
      const d = Math.abs(yJs[i] - yC[i]);
      if (d > maxAbs) maxAbs = d;
    }
    console.log(
      `parity check (N=${N}): max|JS-C| = ${maxAbs.toExponential(2)}${fastMath ? "   (-ffast-math: small diffs expected)" : ""}`
    );
  }

  // Measure koffi call overhead once — N-independent for noop paths.
  const xDummy = allocFloat64Array(1);
  const yDummy = allocFloat64Array(1);
  const noopOnce = timePerCall(() => noop(), targetMs);
  const noopArgsOnce = timePerCall(() => noopArgs(1, xDummy, yDummy), targetMs);
  console.log(
    `koffi call overhead:   noop() = ${fmtTime(noopOnce.perCallSec)}, ` +
      `noop(n,x,y) = ${fmtTime(noopArgsOnce.perCallSec)}`
  );
  console.log("");

  // Per-N sweep.
  const header =
    "N".padStart(10) +
    "  " +
    "JS/call".padStart(12) +
    "  " +
    "C/call".padStart(12) +
    "  " +
    "speedup".padStart(8) +
    "  " +
    "JS/elem".padStart(10) +
    "  " +
    "C/elem".padStart(10) +
    "  " +
    "C−noop".padStart(10);
  console.log(header);
  console.log("-".repeat(header.length));

  for (const N of sizes) {
    const x = allocFloat64Array(N);
    for (let i = 0; i < N; i++) x[i] = (i + 1) / N;
    const y = allocFloat64Array(N);

    const jsFn = (): void => {
      for (let i = 0; i < N; i++) y[i] = Math.exp(1 + Math.sqrt(x[i]));
    };
    const cFn = (): void => {
      fused(N, x, y);
    };

    const jsT = timePerCall(jsFn, targetMs);
    const cT = timePerCall(cFn, targetMs);

    const speedup = jsT.perCallSec / cT.perCallSec;
    const jsPerElem = jsT.perCallSec / N;
    const cPerElem = cT.perCallSec / N;
    const cMinusNoop = cT.perCallSec - noopArgsOnce.perCallSec;

    console.log(
      N.toString().padStart(10) +
        "  " +
        fmtTime(jsT.perCallSec).padStart(12) +
        "  " +
        fmtTime(cT.perCallSec).padStart(12) +
        "  " +
        (speedup >= 1
          ? `${speedup.toFixed(2)}x`
          : `${speedup.toFixed(3)}x`
        ).padStart(8) +
        "  " +
        fmtTime(jsPerElem).padStart(10) +
        "  " +
        fmtTime(cPerElem).padStart(10) +
        "  " +
        fmtTime(Math.max(0, cMinusNoop)).padStart(10)
    );
  }

  // Rough break-even: at what N does JS/call ≈ C/call?
  // Using a linear model: time_c(N) ≈ a + b*N (a = koffi-args overhead,
  // b = per-element C cost at large N); time_js(N) ≈ c*N.
  // Estimate b, c from the largest N we swept; a from noop-args.
  const largestN = sizes[sizes.length - 1];
  {
    const x = allocFloat64Array(largestN);
    for (let i = 0; i < largestN; i++) x[i] = (i + 1) / largestN;
    const y = allocFloat64Array(largestN);
    const jsT = timePerCall(() => {
      for (let i = 0; i < largestN; i++) y[i] = Math.exp(1 + Math.sqrt(x[i]));
    }, targetMs);
    const cT = timePerCall(() => fused(largestN, x, y), targetMs);
    const a = noopArgsOnce.perCallSec;
    const b = Math.max(0, (cT.perCallSec - a) / largestN);
    const c = jsT.perCallSec / largestN;
    const breakEven = c > b ? a / (c - b) : Infinity;
    console.log("");
    console.log(
      `fit @ N=${largestN}:  C ≈ ${fmtTime(a)} + ${fmtTime(b)}·N,  ` +
        `JS ≈ ${fmtTime(c)}·N`
    );
    console.log(
      `break-even N (C-fused beats JS):  ~${Number.isFinite(breakEven) ? Math.round(breakEven).toLocaleString() : "never (JS faster per-elem)"}`
    );
  }
}

main();
