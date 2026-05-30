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

  const pass = groups.size === 1;
  if (pass) {
    passCount++;
    console.log(`PASS  ${name}`);
  } else {
    console.log(`FAIL  ${name}`);
    for (const [k, modes] of groups) {
      const label = modes.map((o) => `opt${o}`).join(",");
      const oneLine = k.includes("\n") ? null : k === "" ? "(empty)" : k;
      if (oneLine !== null) console.log(`        ${label}: ${oneLine}`);
      else console.log(`        ${label}:\n${k.split("\n").map((l) => "          " + l).join("\n")}`);
    }
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

console.log(
  `\n${passCount}/${scripts.length} passed, ${scripts.length - passCount} failed.`,
);
process.exit(passCount === scripts.length ? 0 : 1);
