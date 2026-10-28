#!/usr/bin/env tsx
/**
 * numbl benchmark suite: cases x sizes x execution modes.
 *
 *   npm run bench                       full matrix
 *   npm run bench -- --quick            small subset (about 1-2 minutes)
 *   npm run bench -- --case matmul,fft --mode opt0,opt1 --size 256,600
 *   npm run bench -- compare base.json new.json [--threshold 0.15]
 *   npm run bench -- report results.json      regenerate the .md summary
 *   npm run bench -- --list                   list cases and modes
 *
 * Each mode runs in its own numbl process, which runs the selected cases one
 * after another, each sweeping its sizes in increasing order (if the process
 * dies, the remaining cases are rerun in a new one). Timing happens inside
 * numbl (tic/toc around a batch of kernel calls), so process start-up
 * (measured separately with a trivial script) and JIT compile time (the
 * first call) are not part of the reported per-call time. See
 * docs/developer_reference/testing.md.
 */

import { execFileSync, spawn } from "node:child_process";
import {
  closeSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  openSync,
  readFileSync,
  rmSync,
  statSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { createRequire } from "node:module";
import {
  cpus,
  homedir,
  loadavg,
  platform,
  release,
  tmpdir,
  totalmem,
} from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { CASES, type BenchCase } from "./cases.js";
import {
  compareResults,
  renderMarkdown,
  type BenchResult,
  type Entry,
} from "./report.js";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const require = createRequire(import.meta.url);

// ── Modes ────────────────────────────────────────────────────────────────

interface Mode {
  name: string;
  opt: 0 | 1 | 2;
  native: boolean;
}

const ALL_MODES: Mode[] = [
  { name: "opt0", opt: 0, native: true },
  { name: "opt1", opt: 1, native: true },
  { name: "opt2", opt: 2, native: true },
  { name: "opt0-nonative", opt: 0, native: false },
  { name: "opt1-nonative", opt: 1, native: false },
  { name: "opt2-nonative", opt: 2, native: false },
];
const QUICK_MODES = ["opt0", "opt1", "opt2"];

// ── Options ──────────────────────────────────────────────────────────────

interface RunOptions {
  quick: boolean;
  cases?: string[];
  sizes?: number[];
  modes?: string[];
  reps: number;
  out?: string;
  cli?: string;
  warmCJit: boolean;
  /** Target duration of one timed batch of kernel calls (s). */
  minBatch: number;
  /** Time budget for the timed batches at one size (s). */
  budget: number;
  /** Skip the remaining sizes once a call is predicted to exceed this (s). */
  maxCall: number;
}

function usage(): never {
  console.log(`Usage:
  npm run bench -- [options]              run the benchmark matrix
  npm run bench -- compare <base.json> <new.json> [--threshold 0.15]
  npm run bench -- report <result.json>   regenerate the markdown summary
  npm run bench -- --list                 list cases and modes

Options:
  --quick             small subset of cases, sizes and modes (3 reps)
  --case <a,b,...>    cases to run (names, * wildcards allowed)
  --size <n,...>      only these sizes (e.g. 1e3,1e6,2^20)
  --mode <m,...>      modes to run (${ALL_MODES.map(m => m.name).join(", ")})
  --reps <k>          timed batches per size (default 5); the reported
                      time is the median
  --out <file.json>   output path (default
                      benchmarks/results/local/<date>-<commit>.json); the
                      markdown summary is written next to it
  --cli <file.js>     run a built CLI (e.g. dist-cli/cli.js) instead of
                      src/cli.ts under tsx
  --warm-cjit         use the user's C-JIT cache (default: an empty cache,
                      so the first call at --opt 2 includes cc)
  --max-call <s>      skip larger sizes once a call is predicted to take
                      longer than this (default 3)`);
  process.exit(0);
}

function parseSize(s: string): number {
  const pow = s.match(/^(\d+)\^(\d+)$/);
  const v = pow ? Number(pow[1]) ** Number(pow[2]) : Number(s);
  if (!Number.isFinite(v) || v <= 0) throw new Error(`bad size: ${s}`);
  return v;
}

function parseRunArgs(argv: string[]): RunOptions {
  const o: RunOptions = {
    quick: false,
    reps: 0,
    warmCJit: false,
    minBatch: 0.02,
    budget: 2,
    maxCall: 3,
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => {
      const v = argv[++i];
      if (v === undefined) throw new Error(`${a} needs a value`);
      return v;
    };
    if (a === "--quick") o.quick = true;
    else if (a === "--case") o.cases = next().split(",");
    else if (a === "--size") o.sizes = next().split(",").map(parseSize);
    else if (a === "--mode") o.modes = next().split(",");
    else if (a === "--reps") o.reps = parseInt(next(), 10);
    else if (a === "--out") o.out = resolve(next());
    else if (a === "--cli") o.cli = resolve(next());
    else if (a === "--warm-cjit") o.warmCJit = true;
    else if (a === "--max-call") o.maxCall = Number(next());
    else if (a === "--help" || a === "-h") usage();
    else throw new Error(`unknown option: ${a}`);
  }
  if (!o.reps) o.reps = o.quick ? 3 : 5;
  if (o.quick) o.minBatch = 0.01;
  return o;
}

// ── Machine info ─────────────────────────────────────────────────────────

function tryExec(cmd: string, args: string[]): string | undefined {
  try {
    return execFileSync(cmd, args, {
      cwd: ROOT,
      encoding: "utf-8",
      stdio: ["ignore", "pipe", "ignore"],
      timeout: 60_000,
    }).trim();
  } catch {
    return undefined;
  }
}

function numblCommand(o: RunOptions, args: string[]): [string, string[]] {
  if (o.cli) return [process.execPath, [o.cli, ...args]];
  const tsxCli = require.resolve("tsx/cli");
  return [process.execPath, [tsxCli, join(ROOT, "src", "cli.ts"), ...args]];
}

function collectMachine(o: RunOptions): BenchResult["machine"] {
  const cpuList = cpus();
  const [cmd, args] = numblCommand(o, ["info"]);
  let info: { version?: string; nativeAddon?: boolean } = {};
  try {
    info = JSON.parse(tryExec(cmd, args) ?? "{}");
  } catch {
    // leave empty
  }
  let koffi = false;
  try {
    require.resolve("koffi");
    koffi = true;
  } catch {
    // koffi missing: no --opt 2
  }
  return {
    cpu: cpuList[0]?.model.trim() ?? "unknown",
    cores: cpuList.length,
    memGB: Math.round(totalmem() / 1024 ** 3),
    os: `${platform()} ${release()}`,
    node: process.version,
    numblVersion: info.version ?? "unknown",
    commit: tryExec("git", ["rev-parse", "--short", "HEAD"]) ?? "unknown",
    dirty: !!tryExec("git", ["status", "--porcelain"]),
    nativeAddon: !!info.nativeAddon,
    cc: tryExec(process.env.NUMBL_CC || "cc", ["--version"])?.split("\n")[0],
    koffi,
    env: Object.fromEntries(
      ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMBL_CC"]
        .filter(k => process.env[k] !== undefined)
        .map(k => [k, process.env[k] as string])
    ),
  };
}

// ── MATLAB code generation ───────────────────────────────────────────────

const indent = (s: string, n: number) =>
  s
    .split("\n")
    .filter(l => l.trim())
    .map(l => " ".repeat(n) + l)
    .join("\n");

/** The kernel runs the body once before the batch loop so that `y` is
 *  assigned on every path (otherwise the C-JIT declines the function). */
function genKernel(c: BenchCase): string {
  return `function [bench_t, y] = bk_${c.name}(bench_inner, ${c.inputs.join(", ")})
bench_t0 = tic;
${indent(c.body, 0)}
for bench_it = 2:bench_inner
${indent(c.body, 2)}
end
bench_t = toc(bench_t0);
end
`;
}

function genDriver(c: BenchCase, sizes: number[], o: RunOptions): string {
  const exp = c.costExponent ?? 1;
  const growth = sizes.map((s, i) =>
    i + 1 < sizes.length ? (sizes[i + 1] / s) ** exp : 1
  );
  const call = (inner: string) =>
    `bk_${c.name}(${inner}, ${c.inputs.join(", ")})`;
  const tag = `${c.name} %.17g`;
  return `function bench_${c.name}()
bench_sizes = [${sizes.join(" ")}];
bench_growth = [${growth.map(g => g.toPrecision(6)).join(" ")}];
bench_reps = ${o.reps}; bench_minbatch = ${o.minBatch};
bench_budget = ${o.budget}; bench_maxcall = ${o.maxCall};
bench_skip = false;
for bench_si = 1:numel(bench_sizes)
  n = bench_sizes(bench_si);
  if bench_skip
    fprintf('BENCH_SKIP ${tag}\\n', n);
    continue;
  end
  try
    rng(0);
${indent(c.setup, 4)}
    bench_t0 = tic;
    [bench_t, y] = ${call("1")};
    bench_first = toc(bench_t0);
    [bench_t2, y] = ${call("1")};
    bench_inner = max(1, min(1e6, ceil(bench_minbatch / max(bench_t2, 1e-9))));
    bench_nreps = bench_reps;
    if bench_t2 * bench_inner * bench_nreps > bench_budget
      bench_nreps = max(1, floor(bench_budget / (bench_t2 * bench_inner)));
    end
    bench_ts = zeros(1, bench_nreps);
    for bench_r = 1:bench_nreps
      [bench_t, y] = ${call("bench_inner")};
      bench_ts(bench_r) = bench_t / bench_inner;
    end
    bench_chk = sum(abs(double(y(:))));
    fprintf('BENCH_RESULT ${tag} %.9g %.9g %.9g %d %d %.17g\\n', n, bench_first, ...
      median(bench_ts), min(bench_ts), bench_inner, bench_nreps, bench_chk);
    if max(bench_t2, median(bench_ts)) * bench_growth(bench_si) > bench_maxcall
      bench_skip = true;
    end
  catch bench_err
    fprintf('BENCH_ERROR ${tag} %s\\n', n, strrep(bench_err.message, newline, ' '));
  end
end
end
`;
}

function genMain(cases: BenchCase[]): string {
  const lines = cases.map(
    c =>
      `fprintf('BENCH_CASE ${c.name}\\n'); bench_t0 = tic; bench_${c.name}(); ` +
      `fprintf('BENCH_DONE ${c.name} %.3f\\n', toc(bench_t0));`
  );
  return `function bench_main()\n${lines.join("\n")}\nend\n`;
}

// ── Process execution ────────────────────────────────────────────────────

interface ProcResult {
  /** Interleaved stdout and stderr. */
  output: string;
  code: number | null;
  wallMs: number;
}

/** Run a numbl script. stdout and stderr go to one file so that JIT
 *  diagnostics on stderr can be attributed to the case running at the
 *  time. `onLine` sees complete lines as they arrive. The process is
 *  killed after `idleMs` without output. */
function runNumbl(
  o: RunOptions,
  mode: Mode,
  file: string,
  idleMs: number,
  onLine?: (line: string) => void
): Promise<ProcResult> {
  const args = ["run", file, "--opt", String(mode.opt)];
  if (!mode.native) args.push("--no-native");
  const [cmd, cmdArgs] = numblCommand(o, args);
  const env: NodeJS.ProcessEnv = { ...process.env, NUMBL_JIT_LOG: "1" };
  delete env.NUMBL_NO_NATIVE;
  // A fresh HOME gives the C-JIT an empty cache (~/.cache/numbl/c-jit), so
  // the first call at --opt 2 includes the cc compile. ~/.numbl is linked
  // in so the CLI does not reinstall the mip core package.
  const tmp = mkdtempSync(join(tmpdir(), "numbl-bench-proc-"));
  if (!o.warmCJit) {
    const home = join(tmp, "home");
    mkdirSync(home);
    const realNumbl = join(homedir(), ".numbl");
    if (existsSync(realNumbl)) symlinkSync(realNumbl, join(home, ".numbl"));
    env.HOME = home;
  }
  const logPath = join(tmp, "output.log");
  const fd = openSync(logPath, "w");
  const t0 = performance.now();
  return new Promise(res => {
    const p = spawn(cmd, cmdArgs, {
      cwd: ROOT,
      env,
      stdio: ["ignore", fd, fd],
    });
    closeSync(fd);
    let consumed = 0;
    let lastChange = performance.now();
    let lastSize = 0;
    const poll = () => {
      const size = statSync(logPath).size;
      if (size !== lastSize) {
        lastSize = size;
        lastChange = performance.now();
        const text = readFileSync(logPath, "utf-8");
        const end = text.lastIndexOf("\n") + 1;
        if (end > consumed) {
          for (const l of text.slice(consumed, end - 1).split("\n"))
            onLine?.(l);
          consumed = end;
        }
      } else if (performance.now() - lastChange > idleMs) {
        p.kill("SIGKILL");
      }
    };
    const timer = setInterval(poll, 500);
    p.on("close", code => {
      clearInterval(timer);
      poll();
      const output = readFileSync(logPath, "utf-8");
      rmSync(tmp, { recursive: true, force: true });
      res({ output, code, wallMs: performance.now() - t0 });
    });
  });
}

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

interface CaseOutput {
  entries: Map<number, Entry>;
  done: boolean;
  /** JIT declines and other diagnostics printed while the case ran. */
  diagnostics: string[];
}

/** Split a process's output by case. */
function parseOutput(output: string, mode: Mode): Map<string, CaseOutput> {
  const byCase = new Map<string, CaseOutput>();
  let current: CaseOutput | undefined;
  for (const raw of output.split("\n")) {
    const line = raw.trim();
    if (!line) continue;
    const parts = line.split(" ");
    if (parts[0] === "BENCH_CASE") {
      current = { entries: new Map(), done: false, diagnostics: [] };
      byCase.set(parts[1], current);
      continue;
    }
    if (parts[0] === "BENCH_DONE") {
      const c = byCase.get(parts[1]);
      if (c) c.done = true;
      current = undefined;
      continue;
    }
    if (/^BENCH_(RESULT|SKIP|ERROR) /.test(line)) {
      const c = byCase.get(parts[1]);
      if (!c) continue;
      const size = Number(parts[2]);
      const base = { case: parts[1], mode: mode.name, size };
      if (parts[0] === "BENCH_RESULT") {
        const [first, med, min, inner, reps, chk] = parts.slice(3).map(Number);
        c.entries.set(size, {
          ...base,
          status: "ok",
          median_ms: med * 1e3,
          min_ms: min * 1e3,
          first_ms: first * 1e3,
          inner,
          reps,
          checksum: chk,
        });
      } else if (parts[0] === "BENCH_SKIP") {
        c.entries.set(size, { ...base, status: "skipped" });
      } else {
        c.entries.set(size, {
          ...base,
          status: "error",
          error: parts.slice(3).join(" "),
        });
      }
      continue;
    }
    // Anything else (JIT declines, bail warnings, errors) belongs to the
    // case that was running. Declines in the generated drivers are harness
    // noise; for the kernel, drop the temp-dir path so messages are stable.
    if (current && !/\/bench_\w+\.m@\d+\]/.test(line)) {
      const msg = line.replace(/\[[^\]]*\/(bk_\w+\.m@\d+)\]/, "[$1]");
      if (!current.diagnostics.includes(msg) && current.diagnostics.length < 20)
        current.diagnostics.push(msg);
    }
  }
  return byCase;
}

// ── Run ──────────────────────────────────────────────────────────────────

function matches(name: string, patterns: string[]): boolean {
  return patterns.some(p =>
    new RegExp(
      "^" + p.replace(/[.+?^${}()|[\]\\]/g, "\\$&").replace(/\*/g, ".*") + "$"
    ).test(name)
  );
}

async function run(o: RunOptions): Promise<void> {
  const load0 = loadavg()[0];
  const machine = collectMachine(o);
  if (o.modes) {
    const unknown = o.modes.filter(n => !ALL_MODES.some(m => m.name === n));
    if (unknown.length) throw new Error(`unknown mode(s): ${unknown}`);
  }
  let modes = ALL_MODES.filter(m =>
    o.modes
      ? o.modes.includes(m.name)
      : !o.quick || QUICK_MODES.includes(m.name)
  );
  if (!machine.cc || !machine.koffi) {
    console.error("cc or koffi not available: skipping the --opt 2 modes");
    modes = modes.filter(m => m.opt !== 2);
  }
  if (!machine.nativeAddon) {
    console.error(
      "native addon not built (npx tsx src/cli.ts build-addon): running only the -nonative modes"
    );
    modes = modes.filter(m => !m.native);
  }

  const plan: { c: BenchCase; sizes: number[] }[] = [];
  for (const c of CASES) {
    if (o.cases ? !matches(c.name, o.cases) : o.quick && !c.quickSizes)
      continue;
    let sizes = o.quick && c.quickSizes ? c.quickSizes : c.sizes;
    if (o.sizes) sizes = sizes.filter(s => o.sizes!.includes(s));
    if (sizes.length) plan.push({ c, sizes });
  }
  if (!plan.length) throw new Error("nothing to run (check --case/--size)");
  // Pure scalar loops never reach the addon: skip -nonative unless asked.
  const casesFor = (m: Mode) =>
    plan.filter(p => m.native || !p.c.nativeIndependent || o.modes);
  modes = modes.filter(m => casesFor(m).length);
  if (!modes.length) throw new Error("no modes to run");

  const workDir = mkdtempSync(join(tmpdir(), "numbl-bench-"));
  const t0 = performance.now();
  const startup: Record<string, number> = {};
  const entries: Entry[] = [];
  const diagnostics: BenchResult["diagnostics"] = [];
  try {
    for (const { c, sizes } of plan) {
      writeFileSync(join(workDir, `bk_${c.name}.m`), genKernel(c));
      writeFileSync(join(workDir, `bench_${c.name}.m`), genDriver(c, sizes, o));
    }

    // Process start-up: a trivial script, median of 3 runs (1 with --quick).
    const startupDir = join(workDir, "startup");
    mkdirSync(startupDir);
    const startupFile = join(startupDir, "bench_startup.m");
    writeFileSync(startupFile, "x = 1;\n");
    for (const m of modes) {
      const walls: number[] = [];
      for (let i = 0; i < (o.quick ? 1 : 3); i++)
        walls.push((await runNumbl(o, m, startupFile, 60_000)).wallMs);
      startup[m.name] = median(walls);
      console.error(`start-up ${m.name}: ${startup[m.name].toFixed(0)} ms`);
    }

    const total = modes.reduce((n, m) => n + casesFor(m).length, 0);
    let k = 0;
    for (const m of modes) {
      let pending = casesFor(m);
      while (pending.length) {
        const main = join(workDir, "bench_main.m");
        writeFileSync(main, genMain(pending.map(p => p.c)));
        const r = await runNumbl(o, m, main, 10 * 60_000, line => {
          const parts = line.trim().split(" ");
          if (parts[0] === "BENCH_DONE")
            console.error(
              `[${++k}/${total}] ${m.name} ${parts[1]}: ${Number(parts[2]).toFixed(1)} s`
            );
        });
        const parsed = parseOutput(r.output, m);
        const next: typeof pending = [];
        let crashed = false;
        for (const p of pending) {
          const co = parsed.get(p.c.name);
          if (!co) {
            next.push(p); // never started: rerun in a new process
            continue;
          }
          if (!co.done) {
            crashed = true;
            k++;
            const tail = r.output.trim().split("\n").slice(-3).join(" | ");
            console.error(
              `[${k}/${total}] ${m.name} ${p.c.name}: FAILED (${tail})`
            );
            co.diagnostics.push(
              `process exited (code ${r.code}) while running this case: ${tail}`
            );
          }
          for (const s of p.sizes)
            entries.push(
              co.entries.get(s) ?? {
                case: p.c.name,
                mode: m.name,
                size: s,
                status: "error",
                error: "no result (process failed or timed out)",
              }
            );
          if (co.diagnostics.length)
            diagnostics.push({
              case: p.c.name,
              mode: m.name,
              lines: co.diagnostics,
            });
        }
        if (!crashed && next.length === pending.length) {
          // Nothing started at all: the process failed up front.
          throw new Error(`${m.name}: numbl failed to run:\n${r.output}`);
        }
        pending = next;
      }
    }
  } finally {
    rmSync(workDir, { recursive: true, force: true });
  }

  const result: BenchResult = {
    schema: 1,
    created: new Date().toISOString(),
    machine: { ...machine, loadavg: [load0, loadavg()[0]] },
    config: {
      quick: o.quick,
      reps: o.reps,
      minBatchS: o.minBatch,
      budgetS: o.budget,
      maxCallS: o.maxCall,
      coldCJitCache: !o.warmCJit,
      cli: o.cli ? "built" : "tsx src/cli.ts",
    },
    totalWallS: Math.round((performance.now() - t0) / 100) / 10,
    modes: modes.map(m => ({
      name: m.name,
      args: `--opt ${m.opt}${m.native ? "" : " --no-native"}`,
      startup_ms: startup[m.name],
    })),
    cases: plan.map(p => ({
      name: p.c.name,
      description: p.c.description,
      sizeLabel: p.c.sizeLabel,
    })),
    entries,
    diagnostics,
  };

  const out =
    o.out ??
    join(
      ROOT,
      "benchmarks",
      "results",
      "local",
      `${result.created.slice(0, 10)}-${machine.commit}${o.quick ? "-quick" : ""}.json`
    );
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, JSON.stringify(result, null, 1) + "\n");
  const md = out.replace(/\.json$/, "") + ".md";
  writeFileSync(md, renderMarkdown(result));
  console.error(`\nwrote ${out}\nwrote ${md} (${result.totalWallS} s total)`);
}

// ── Main ─────────────────────────────────────────────────────────────────

function loadResult(path: string): BenchResult {
  return JSON.parse(readFileSync(path, "utf-8")) as BenchResult;
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  if (argv[0] === "compare") {
    const files: string[] = [];
    let threshold = 0.15;
    for (let i = 1; i < argv.length; i++) {
      if (argv[i] === "--threshold") threshold = Number(argv[++i]);
      else files.push(argv[i]);
    }
    if (files.length !== 2 || !(threshold > 0)) usage();
    const { text, regressions } = compareResults(
      loadResult(files[0]),
      loadResult(files[1]),
      threshold
    );
    console.log(text);
    process.exit(regressions > 0 ? 1 : 0);
  } else if (argv[0] === "report") {
    if (!argv[1]) usage();
    const md = argv[1].replace(/\.json$/, "") + ".md";
    writeFileSync(md, renderMarkdown(loadResult(argv[1])));
    console.error(`wrote ${md}`);
  } else if (argv[0] === "--list") {
    for (const c of CASES)
      console.log(
        `${c.name.padEnd(20)} ${c.description}  [${c.sizes.join(", ")}]${c.quickSizes ? "  (quick)" : ""}`
      );
    console.log(`\nmodes: ${ALL_MODES.map(m => m.name).join(", ")}`);
  } else {
    await run(parseRunArgs(argv[0] === "run" ? argv.slice(1) : argv));
  }
}

main().catch(e => {
  console.error((e as Error).message);
  process.exit(2);
});
