#!/usr/bin/env tsx
/**
 * Benchmark runner for CI.
 *
 * Runs every .m benchmark under each mode in MODES, parses per-kernel
 * timings out of stdout, takes the median across N_RUNS invocations,
 * and emits a JSON array of {name, unit, value} triples suitable for
 * benchmark-action/github-action-benchmark's `customSmallerIsBetter`
 * tool.
 *
 * Usage: tsx benchmarks/run_benchmarks.ts <output-json-path>
 *
 * Env:
 *   NUMBL_PAR_CC    — compiler for the `--opt e1 --par` mode (e.g.
 *                     `gcc-14` on macOS, since Apple clang ships without
 *                     OpenMP threading). If unset, `--par` uses whatever
 *                     `cc` resolves to.
 *   N_RUNS          — number of runs per (benchmark, mode) pair. Default 3.
 */

import { execFileSync } from "node:child_process";
import { writeFileSync } from "node:fs";
import { arch, cpus, platform, release, totalmem } from "node:os";

interface Metric {
  name: string;
  unit: string;
  value: number;
  /** Stamped on every metric so chart tooltips show runner context
   *  (CPU model, OS, compiler, etc.). Invaluable later when a runner-image
   *  upgrade shifts the baseline — the diff point carries its own
   *  environment fingerprint. */
  extra?: string;
}

const N_RUNS = parseInt(process.env.N_RUNS ?? "", 10) || 3;

// ── System info stamp (attached to every metric's `extra` field) ─────

function collectSystemInfo(): string {
  const lines: string[] = [];
  lines.push(`OS: ${platform()} ${release()} (${arch()})`);
  const cpuList = cpus();
  if (cpuList.length > 0) {
    lines.push(`CPU: ${cpuList[0].model.trim()} (${cpuList.length} cores)`);
  }
  lines.push(`RAM: ${Math.round(totalmem() / 1024 ** 3)} GB`);
  lines.push(`Node: ${process.version}`);
  try {
    const ccVer = execFileSync("cc", ["--version"], {
      encoding: "utf-8",
      stdio: ["ignore", "pipe", "ignore"],
      timeout: 5_000,
    })
      .split("\n")[0]
      .trim();
    lines.push(`cc: ${ccVer}`);
  } catch {
    // cc not on PATH — fine, just skip
  }
  if (process.env.NUMBL_PAR_CC) {
    lines.push(`--par cc: ${process.env.NUMBL_PAR_CC}`);
  }
  // GitHub Actions sets these on hosted runners; useful for distinguishing
  // runner-image upgrades that shift baselines invisibly.
  const runnerBits = [
    process.env.RUNNER_OS,
    process.env.ImageOS,
    process.env.ImageVersion,
  ]
    .filter(Boolean)
    .join(" ");
  if (runnerBits) lines.push(`Runner: ${runnerBits}`);
  return lines.join("\n");
}

const SYSTEM_INFO = collectSystemInfo();
process.stderr.write(`\n[run_benchmarks] system info:\n${SYSTEM_INFO}\n`);

// ── Per-benchmark output parsers ──────────────────────────────────────

interface BenchSpec {
  file: string;
  parse: (output: string) => Record<string, number>;
}

function parseKernelLines(
  out: string,
  labels: Record<string, string>
): Record<string, number> {
  const result: Record<string, number> = {};
  const totalMatch = out.match(/elapsed = ([\d.]+) s/);
  if (totalMatch) result.total = Number(totalMatch[1]);
  for (const [label, metric] of Object.entries(labels)) {
    const escaped = label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const m = out.match(new RegExp(`${escaped}\\s+([\\d.]+)\\s*s`));
    if (m) result[metric] = Number(m[1]);
  }
  return result;
}

function parseComplexTensorOutput(out: string): Record<string, number> {
  const result: Record<string, number> = {};
  const totalMatch = out.match(/elapsed = ([\d.]+) s/);
  if (totalMatch) result.total = Number(totalMatch[1]);
  // Lines like: "1. Mandelbrot z.*z+c:         0.053 s   (fused)"
  const lineRe = /^\s*(\d+)\..*?:\s+([\d.]+)\s*s/gm;
  let m: RegExpExecArray | null;
  const names = [
    "k1_mandelbrot",
    "k2_tensor_chain",
    "k3_conj_chain",
    "k4_widening",
    "k5_divide",
    "k6_abs_reduce",
  ];
  while ((m = lineRe.exec(out)) !== null) {
    const idx = Number(m[1]) - 1;
    if (idx >= 0 && idx < names.length) {
      result[names[idx]] = Number(m[2]);
    }
  }
  return result;
}

function parseChunkieOutput(out: string): Record<string, number> {
  // The benchmark prints a BENCH:-tagged phase line for warmup AND for the
  // timed hot run. Taking the last value per phase gives us the hot timings;
  // `phase=execution` only appears once (the hot total).
  const result: Record<string, number> = {};
  const phaseRe = /BENCH: phase=(\w+) t=([\d.]+)/g;
  let m: RegExpExecArray | null;
  while ((m = phaseRe.exec(out)) !== null) {
    result[m[1]] = Number(m[2]);
  }
  // Drop chunkie_load and warmup from reported metrics — they're setup cost.
  delete result["chunkie_load"];
  delete result["warmup"];
  return result;
}

const BENCHMARKS: BenchSpec[] = [
  {
    file: "benchmarks/scalar_bench.m",
    parse: out => {
      const m = out.match(/elapsed = ([\d.]+) s/);
      return m ? { elapsed: Number(m[1]) } : {};
    },
  },
  {
    file: "benchmarks/complex_scalar_bench.m",
    parse: out => {
      const m = out.match(/elapsed = ([\d.]+) s/);
      return m ? { elapsed: Number(m[1]) } : {};
    },
  },
  {
    file: "benchmarks/tensor_ops_bench.m",
    parse: out =>
      parseKernelLines(out, {
        "Real binary elemwise:": "Binary",
        "Real unary elemwise:": "Unary",
        "Comparisons + reduce:": "CmpRed",
        "Reductions:": "Reduce",
        "Chained pipeline:": "Chain",
      }),
  },
  {
    file: "benchmarks/tensor_ops_bench2.m",
    parse: out =>
      parseKernelLines(out, {
        "Single-expr Gaussian:": "Gauss",
        "Single-expr nested:": "Nested",
        "Inline reduction:": "InlRed",
        "Inline accum reduction:": "AccRed",
        "Binary builtins:": "BinOps",
        "Clamp + distance:": "Clamp",
      }),
  },
  {
    file: "benchmarks/complex_tensor_bench.m",
    parse: parseComplexTensorOutput,
  },
  {
    file: "benchmarks/chunkie_helmholtz_starfish.m",
    parse: parseChunkieOutput,
  },
];

// ── Modes ─────────────────────────────────────────────────────────────

interface ModeSpec {
  label: string;
  slug: string; // file-safe label used in the metric name
  args: string[];
  env?: NodeJS.ProcessEnv;
}

const MODES: ModeSpec[] = [
  { label: "--opt 1", slug: "opt-1", args: ["--opt", "1"] },
  { label: "--opt e1", slug: "opt-e1", args: ["--opt", "e1"] },
  {
    label: "--opt e1 --par",
    slug: "opt-e1-par",
    args: ["--opt", "e1", "--par"],
    env: process.env.NUMBL_PAR_CC
      ? { NUMBL_CC: process.env.NUMBL_PAR_CC }
      : undefined,
  },
];

// ── Helpers ───────────────────────────────────────────────────────────

function median(xs: number[]): number {
  const sorted = [...xs].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function runNumbl(
  benchFile: string,
  args: string[],
  env: NodeJS.ProcessEnv
): string {
  try {
    return execFileSync(
      "npx",
      ["tsx", "src/cli.ts", "run", benchFile, ...args],
      {
        encoding: "utf-8",
        stdio: ["ignore", "pipe", "inherit"],
        env: { ...process.env, ...env },
        maxBuffer: 64 * 1024 * 1024,
      }
    );
  } catch (e) {
    const err = e as { stdout?: Buffer | string; message: string };
    process.stderr.write(
      `\n[run_benchmarks] command failed for ${benchFile}: ${err.message}\n`
    );
    return typeof err.stdout === "string"
      ? err.stdout
      : (err.stdout?.toString("utf-8") ?? "");
  }
}

// ── Main ──────────────────────────────────────────────────────────────

const outPath = process.argv[2];
if (!outPath) {
  process.stderr.write(
    "usage: tsx benchmarks/run_benchmarks.ts <output-json-path>\n"
  );
  process.exit(1);
}

const metrics: Metric[] = [];

for (const bench of BENCHMARKS) {
  const benchSlug = bench.file.replace(/^benchmarks\//, "").replace(/\.m$/, "");

  for (const mode of MODES) {
    process.stderr.write(`\n=== ${bench.file}  [${mode.label}] ===\n`);

    const runs: Record<string, number>[] = [];
    for (let i = 0; i < N_RUNS; i++) {
      const out = runNumbl(bench.file, mode.args, mode.env ?? {});
      const parsed = bench.parse(out);
      process.stderr.write(
        `  run ${i + 1}: ${
          Object.keys(parsed).length === 0
            ? "(no metrics parsed — check output)"
            : JSON.stringify(parsed)
        }\n`
      );
      runs.push(parsed);
    }

    // Collect all keys seen across runs, take median where present.
    const allKeys = new Set<string>();
    for (const r of runs) for (const k of Object.keys(r)) allKeys.add(k);

    for (const key of [...allKeys].sort()) {
      const vals = runs
        .map(r => r[key])
        .filter((v): v is number => v !== undefined && Number.isFinite(v));
      if (vals.length < N_RUNS) {
        process.stderr.write(
          `  WARN ${benchSlug} ${mode.slug} ${key}: only ${vals.length}/${N_RUNS} runs reported a value\n`
        );
      }
      if (vals.length === 0) continue;
      metrics.push({
        name: `${benchSlug} / ${key} / ${mode.slug}`,
        unit: "s",
        value: median(vals),
        extra: `median of ${vals.length}/${N_RUNS} runs: [${vals.map(v => v.toFixed(4)).join(", ")}]\n${SYSTEM_INFO}`,
      });
    }
  }
}

writeFileSync(outPath, JSON.stringify(metrics, null, 2) + "\n");
process.stderr.write(
  `\n[run_benchmarks] wrote ${metrics.length} metrics to ${outPath}\n`
);
