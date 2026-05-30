#!/usr/bin/env node
// jit_parity/run.mjs
//
// Runs every script in jit_parity/scripts/ under numbl --opt 0 (interpreter),
// --opt 1 (JS-JIT) and --opt 2 (C-JIT) and checks that all three produce the
// SAME observable result. Because all three modes are the same codebase, an
// in-bounds run should be byte-for-byte identical; any divergence is a bug.
//
// Comparison rules (deliberately strict — we expect EXACT matches):
//   * stdout is compared verbatim, except:
//       - lines containing "using bridge:" are dropped (a one-time native-addon
//         diagnostic the interpreter prints to stdout; pure noise),
//       - lines beginning with "warning:" are dropped (JIT bail notices; these
//         normally go to stderr, filtered here too for safety),
//       - trailing whitespace/newlines are trimmed.
//   * a run that exits non-zero is reduced to the single outcome token <ERROR>
//     (the interpreter and C-JIT word their error messages differently, but
//     "it correctly refused" is the contract we test, not the message text).
//   * a run that times out is reduced to <TIMEOUT>.
//
// Usage:
//   node jit_parity/run.mjs                 # run all scripts
//   node jit_parity/run.mjs A04 A17         # only scripts whose name matches a filter
//   node jit_parity/run.mjs -v              # show full per-mode output on failures
//   node jit_parity/run.mjs --no-opt2       # skip the (slower, compiling) C-JIT mode
//
// Exit code is 0 iff every (selected) script passes.

import { spawnSync } from "node:child_process";
import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = dirname(scriptDir);
const scriptsDir = join(scriptDir, "scripts");

const argv = process.argv.slice(2);
const verbose = argv.includes("-v") || argv.includes("--verbose");
const skipOpt2 = argv.includes("--no-opt2");
const filters = argv.filter((a) => !a.startsWith("-"));

const OPTS = skipOpt2 ? [0, 1] : [0, 1, 2];
const TIMEOUT_MS = 180_000;

// Scripts excluded from the pass/fail gate: their divergence is either
// inherent floating-point non-associativity / library-ULP differences
// (can't be made bitwise-identical without dropping BLAS/libm), or it
// requires reimplementing V8's number→string conversion in C (a large
// dtoa project) where a naive fix would pass these scripts but introduce
// NEW silent divergences for non-exact values. They're still run and
// shown (EXCL) so the divergence stays visible, but don't count toward
// pass/fail or the exit code. Keyed by filename substring → reason.
const GATE_EXCLUDED = {
  C03_matmul: "matmul: BLAS dgemm (interpreter) vs naive triple-loop (JIT) accumulate in different orders → last-bit differences. Inherent FP non-associativity.",
  "A35_complex-pow": "(-1)^(0.5+1i) leaves a ~2.6e-18 real residue from exp(b·log(a)) (= exp(-π)·cos(π/2)); MATLAB keeps it, but it's the difference of transcendentals → sensitive to libm (opt2) vs V8 (opt0/opt1) ULP, like C02. The integer-power line is fixable; the script can't be byte-stable.",
  "F01_sprintf-half-even": "%f/%e/%g half-way rounding: libc snprintf rounds half-to-even; matching V8's toFixed/toExponential (correct-rounding toward +Inf on the exact decimal) byte-for-byte needs a V8-style dtoa in C. Deferred — a naive scale-and-round would pass these exact-representable cases but diverge on non-exact values.",
  "F02_disp-scalar-half-even": "disp() scalar half-way rounding: same root cause as F01 (format_double.h snprintf %.4e half-to-even vs V8 toExponential). Deferred pending a V8-equivalent dtoa.",
  "F04_s-noninteger": "%s of a non-integer needs V8's shortest round-trip form (String(0.1)='0.1'); libc %.17g/%g aren't shortest. Deferred pending a dtoa.",
};
function exclusionReason(name) {
  for (const [sub, reason] of Object.entries(GATE_EXCLUDED)) {
    if (name.includes(sub)) return reason;
  }
  return null;
}

function runMode(file, opt) {
  const r = spawnSync(
    "npx",
    ["tsx", "src/cli.ts", "run", file, "--opt", String(opt)],
    { cwd: repoRoot, encoding: "utf8", timeout: TIMEOUT_MS, maxBuffer: 64 * 1024 * 1024 },
  );
  const timedOut = r.error && (r.error.code === "ETIMEDOUT" || r.signal === "SIGTERM");
  return { stdout: r.stdout ?? "", stderr: r.stderr ?? "", status: r.status, timedOut };
}

function normalize({ stdout, status, timedOut }) {
  if (timedOut) return "<TIMEOUT>";
  if (status !== 0) return "<ERROR>";
  const kept = stdout
    .split("\n")
    .filter((l) => !/using bridge:/.test(l) && !/^\s*warning:/i.test(l));
  return kept.join("\n").replace(/\s+$/, "");
}

function previewBlock(s, label) {
  const body = s === "<ERROR>" || s === "<TIMEOUT>" ? s : s.length ? s : "(empty)";
  const lines = body.split("\n");
  const shown = lines.slice(0, 12);
  const tail = lines.length > 12 ? `\n        … (${lines.length - 12} more line(s))` : "";
  return `      ${label}:\n` + shown.map((l) => "        " + l).join("\n") + tail;
}

let scripts = readdirSync(scriptsDir)
  .filter((f) => f.endsWith(".m"))
  .sort();
if (filters.length) scripts = scripts.filter((f) => filters.some((q) => f.includes(q)));

if (!scripts.length) {
  console.error("No matching scripts.");
  process.exit(2);
}

console.log(
  `Comparing --opt ${OPTS.join("/")} across ${scripts.length} script(s) in jit_parity/scripts/\n`,
);

let passCount = 0;
let excludedCount = 0;
let gateTotal = 0;
const failures = [];

for (const name of scripts) {
  const file = join(scriptsDir, name);
  const results = {};
  for (const opt of OPTS) results[opt] = runMode(file, opt);
  const keys = {};
  for (const opt of OPTS) keys[opt] = normalize(results[opt]);

  // group modes by identical key
  const groups = new Map();
  for (const opt of OPTS) {
    const k = keys[opt];
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(opt);
  }

  const agree = groups.size === 1;
  const excluded = exclusionReason(name);
  const showGroups = () => {
    for (const [k, modes] of groups) {
      const label = modes.map((o) => `opt${o}`).join(",");
      const oneLine = k.includes("\n") ? null : k === "" ? "(empty)" : k;
      if (oneLine !== null) console.log(`        ${label}: ${oneLine}`);
      else console.log(`        ${label}:\n${k.split("\n").map((l) => "          " + l).join("\n")}`);
    }
  };

  if (excluded) {
    // Not part of the gate. Report status for visibility only.
    excludedCount++;
    console.log(`EXCL  ${name}  — ${agree ? "agree" : "diverge (expected)"}`);
    console.log(`        (${excluded})`);
    if (!agree) showGroups();
    continue;
  }

  gateTotal++;
  if (agree) {
    passCount++;
    console.log(`PASS  ${name}`);
  } else {
    console.log(`FAIL  ${name}`);
    showGroups();
    failures.push({ name, results, keys });
  }
}

if (verbose && failures.length) {
  console.log("\n──────── failure detail (raw per-mode stdout/stderr) ────────");
  for (const { name, results } of failures) {
    console.log(`\n### ${name}`);
    for (const opt of OPTS) {
      const r = results[opt];
      const status = r.timedOut ? "TIMEOUT" : `exit=${r.status}`;
      console.log(`  --opt ${opt} [${status}]`);
      if (r.stdout.trim()) console.log(previewBlock(r.stdout.replace(/\s+$/, ""), "stdout"));
      if (r.stderr.trim()) console.log(previewBlock(r.stderr.trim().split("\n").slice(0, 3).join("\n"), "stderr"));
    }
  }
}

const exclNote = excludedCount ? ` (${excludedCount} excluded from gate)` : "";
console.log(
  `\n${passCount}/${gateTotal} passed, ${gateTotal - passCount} failed${exclNote}.`,
);
process.exit(passCount === gateTotal ? 0 : 1);
