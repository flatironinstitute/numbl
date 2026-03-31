#!/usr/bin/env npx tsx

import { discoverBackends } from "./matmul-bench-backends.js";
import {
  buildScenarios,
  runBenchmarks,
  formatResults,
  validateBackend,
} from "./matmul-bench-core.js";

const args = process.argv.slice(2);
const jsonOutput = args.includes("--json");
function getArg(name: string, fallback: number): number {
  const idx = args.indexOf(name);
  return idx >= 0 ? parseInt(args[idx + 1]) || fallback : fallback;
}
const warmup = getArg("--warmup", 3);
const iterations = getArg("--iterations", 10);

const backends = await discoverBackends();
const scenarios = buildScenarios();

console.error(`Backends: ${backends.map(b => b.id).join(", ")}\n`);

// Validate correctness
const ref = backends[0];
for (const b of backends.slice(1)) {
  const { ok, errors } = validateBackend(b, ref, scenarios);
  if (!ok) {
    console.error(`  ${b.id}: CORRECTNESS FAILED`);
    for (const e of errors) console.error(`    ${e}`);
  } else {
    console.error(`  ${b.id}: correctness OK`);
  }
}
console.error("");

const results = await runBenchmarks(scenarios, backends, {
  warmup,
  iterations,
  onProgress: msg => {
    if (!jsonOutput) process.stderr.write(`  ${msg}\r`);
  },
});

if (!jsonOutput) process.stderr.write("\n");

if (jsonOutput) {
  console.log(JSON.stringify(results, null, 2));
} else {
  console.log(formatResults(results, scenarios, backends));
}
