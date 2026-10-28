#!/usr/bin/env tsx
/**
 * Annotate integration test scripts with the strongest VERIFIED
 * top-level `%!numbl:assert_jit` directive:
 *
 *   - `%!numbl:assert_jit c`  if the whole script JITs at BOTH --opt 1
 *                             (JS) and --opt 2 (C),
 *   - `%!numbl:assert_jit`    else if it JITs at --opt 1 (JS only),
 *   - (nothing)               else.
 *
 * Idempotent: strips any existing *top-level* (unindented) directive
 * line first, then re-decides — so re-running upgrades plain->c as more
 * scripts gain C-JIT support (and downgrades if support regresses).
 * Indented (in-loop) directives are left untouched.
 *
 * "JITs" / "passes" uses the same runTestScript as `run-tests` and
 * src/__tests__/test-scripts.test.ts (workspace scan, searchPaths, SUCCESS
 * last-line check); the --opt 2 check additionally sets up the C-JIT
 * and koffi via initNodePlatform (as cli.ts does). Not wired into npm;
 * run manually:
 *   npx tsx scripts/annotate_assert_jit.ts
 */
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { findTestScripts } from "../src/cli-scan.js";
import { runTestScript } from "../src/cli-run-tests.js";
import { initNodePlatform } from "../src/node-platform.js";

const PLAIN = "%!numbl:assert_jit";
const CVAR = "%!numbl:assert_jit c";

const { nativeBridge } = await initNodePlatform({ native: false });
if (!nativeBridge) {
  console.error("koffi not available — cannot verify --opt 2; aborting.");
  process.exit(1);
}

/** Remove any existing top-level (unindented) assert_jit directive line. */
function stripTopLevelDirective(src: string): string {
  return src
    .split("\n")
    .filter(l => l.trimEnd() !== PLAIN && l.trimEnd() !== CVAR)
    .join("\n");
}

/** Insert `directive` after the leading single-line-comment / blank
 *  block, before the first code line; fall back to the very top if a
 *  block comment (`%{`) leads the file. */
function insertDirective(base: string, directive: string): string {
  const lines = base.split("\n");
  let at = 0;
  for (let i = 0; i < lines.length; i++) {
    const t = lines[i].trim();
    if (t === "") continue;
    if (t.startsWith("%{") || t.startsWith("%}")) {
      at = 0;
      break;
    }
    if (t.startsWith("%")) continue;
    at = i;
    break;
  }
  lines.splice(at, 0, directive);
  return lines.join("\n");
}

function passes(filepath: string, source: string, opt: "1" | "2"): boolean {
  try {
    const { lastLine } = runTestScript(filepath, source, {
      optimization: opt,
      nativeBridge,
    });
    return lastLine === "SUCCESS";
  } catch {
    return false;
  }
}

const thisDir = dirname(fileURLToPath(import.meta.url));
const testDir = resolve(thisDir, "..", "numbl_test_scripts");
const files = findTestScripts(testDir);

const cFiles: string[] = [];
const plainFiles: string[] = [];
let none = 0;
for (const f of files) {
  const original = readFileSync(f, "utf-8");
  const base = stripTopLevelDirective(original);
  const cVariant = insertDirective(base, CVAR);
  const plainVariant = insertDirective(base, PLAIN);

  let chosen: string;
  if (passes(f, cVariant, "1") && passes(f, cVariant, "2")) {
    chosen = cVariant;
    cFiles.push(f);
  } else if (passes(f, plainVariant, "1")) {
    chosen = plainVariant;
    plainFiles.push(f);
  } else {
    chosen = base;
    none++;
  }
  if (chosen !== original) writeFileSync(f, chosen);
}

console.log(`test files:    ${files.length}`);
console.log(`assert_jit c:  ${cFiles.length}`);
console.log(`assert_jit:    ${plainFiles.length}`);
console.log(`no directive:  ${none}`);
console.log("\nC-JIT (assert_jit c):");
for (const f of cFiles) console.log("  " + f.replace(testDir + "/", ""));
