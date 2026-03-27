#!/usr/bin/env npx tsx

import { createRequire } from "module";
import { dirname, join } from "path";
import { fileURLToPath } from "url";
import {
  NATIVE_ADDON_EXPECTED_VERSION,
  type LapackBridge,
} from "../numbl-core/native/lapack-bridge.js";
import { getTsLapackBridge } from "../numbl-core/native/ts-lapack-bridge.js";
import {
  buildScenarios,
  runAllScenarios,
  formatResultsTable,
  type BenchBackend,
} from "./linalg-bench-core.js";

const args = process.argv.slice(2);
function getArg(name: string, fallback: number): number {
  const idx = args.indexOf(name);
  return idx >= 0 ? parseInt(args[idx + 1]) || fallback : fallback;
}
const jsonOutput = args.includes("--json");
const warmup = getArg("--warmup", 3);
const iterations = getArg("--iterations", 10);
const time = getArg("--time", 0);

const backends: BenchBackend[] = [
  {
    id: "ts-lapack",
    label: "TypeScript LAPACK (pure JS)",
    bridge: getTsLapackBridge(),
  },
];

const addonPath = join(
  dirname(fileURLToPath(import.meta.url)),
  "..",
  "..",
  "build",
  "Release",
  "numbl_addon.node"
);

try {
  const req = createRequire(import.meta.url);
  const addon = req(addonPath) as LapackBridge;
  const addonVer =
    typeof addon.addonVersion === "function" ? addon.addonVersion() : 0;
  if (addonVer === NATIVE_ADDON_EXPECTED_VERSION) {
    backends.push({
      id: "openblas",
      label: "Native addon (OpenBLAS/LAPACK)",
      bridge: addon,
    });
  } else {
    console.error(
      `Native addon version mismatch (got ${addonVer}, expected ${NATIVE_ADDON_EXPECTED_VERSION}), skipping.`
    );
  }
} catch {
  console.error(
    "Native addon not found, skipping. Run 'npx numbl build-addon' to build it."
  );
}

const scenarios = buildScenarios();

console.error(
  `Running ${scenarios.length} scenarios × ${backends.length} backend(s), ` +
    `warmup=${warmup}, iterations=${iterations}` +
    (time ? `, time=${time}ms` : "") +
    `\n`
);
console.error(`Backends: ${backends.map(b => b.id).join(", ")}\n`);

const results = await runAllScenarios(scenarios, backends, {
  warmup,
  iterations,
  time,
  onProgress: sid => {
    if (!jsonOutput) process.stderr.write(`  ${sid}...\r`);
  },
});

if (!jsonOutput) process.stderr.write("\n");

if (jsonOutput) {
  console.log(JSON.stringify(results, null, 2));
} else {
  console.log(formatResultsTable(results, backends));
}
