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
 * "JITs" / "passes" mirrors src/__tests__/test-scripts.test.ts exactly:
 * same workspace scan, searchPaths, and the SUCCESS last-line check; the
 * --opt 2 check additionally wires koffi (as cli.ts does) so the C path
 * is exercised. Not wired into npm; run manually:
 *   npx tsx scripts/annotate_assert_jit.ts
 */
import { readFileSync, writeFileSync, readdirSync, statSync } from "node:fs";
import { join, dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
import { executeCode } from "../src/numbl-core/executeCode.js";
import type {
  NativeBridge,
  WorkspaceFile,
} from "../src/numbl-core/workspace/types.js";
import { NodeFileIOAdapter } from "../src/cli-fileio.js";
import { NodeSystemAdapter } from "../src/cli-system.js";
import { registerNodeCompileC } from "../src/numbl-core/executors/jit/compileC.node.js";

const PLAIN = "%!numbl:assert_jit";
const CVAR = "%!numbl:assert_jit c";

registerNodeCompileC();
let nativeBridge: NativeBridge | undefined;
try {
  const koffi = createRequire(import.meta.url)("koffi");
  nativeBridge = { load: (p: string) => koffi.load(p), koffi };
} catch {
  console.error("koffi not available — cannot verify --opt 2; aborting.");
  process.exit(1);
}

function scanMFiles(dirPath: string, excludeFile?: string): WorkspaceFile[] {
  const files: WorkspaceFile[] = [];
  let entries: string[];
  try {
    entries = readdirSync(dirPath);
  } catch {
    return files;
  }
  for (const entry of entries) {
    const fullPath = join(dirPath, entry);
    if (excludeFile && fullPath === excludeFile) continue;
    try {
      const stat = statSync(fullPath);
      if (stat.isDirectory()) {
        if (
          entry.startsWith("@") ||
          entry.startsWith("+") ||
          entry === "private"
        )
          files.push(...scanMFiles(fullPath, excludeFile));
      } else if (
        stat.isFile() &&
        (entry.endsWith(".m") || entry.endsWith(".numbl.js"))
      ) {
        files.push({ name: fullPath, source: readFileSync(fullPath, "utf-8") });
      } else if (stat.isFile() && entry.endsWith(".wasm")) {
        files.push({
          name: fullPath,
          source: "",
          data: new Uint8Array(readFileSync(fullPath)),
        });
      }
    } catch {
      continue;
    }
  }
  return files;
}

function findTestFiles(dir: string): string[] {
  const out: string[] = [];
  function walk(cur: string) {
    let entries: string[];
    try {
      entries = readdirSync(cur);
    } catch {
      return;
    }
    entries.sort();
    for (const entry of entries) {
      const p = join(cur, entry);
      let st;
      try {
        st = statSync(p);
      } catch {
        continue;
      }
      if (st.isDirectory()) {
        if (!entry.startsWith("@") && !entry.startsWith("+")) walk(p);
      } else if (st.isFile() && entry.endsWith(".m")) {
        if (readFileSync(p, "utf-8").includes("SUCCESS")) out.push(p);
      }
    }
  }
  walk(dir);
  return out;
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
  const scriptDir = dirname(filepath);
  try {
    const result = executeCode(
      source,
      {
        displayResults: true,
        optimization: opt,
        fileIO: new NodeFileIOAdapter(),
        system: new NodeSystemAdapter(),
      },
      scanMFiles(scriptDir, filepath),
      filepath,
      [scriptDir],
      nativeBridge
    );
    const lines = result.output
      .join("")
      .split("\n")
      .filter(l => l.length > 0);
    return lines[lines.length - 1] === "SUCCESS";
  } catch {
    return false;
  }
}

const thisDir = dirname(fileURLToPath(import.meta.url));
const testDir = resolve(thisDir, "..", "numbl_test_scripts");
const files = findTestFiles(testDir);

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
