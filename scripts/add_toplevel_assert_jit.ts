#!/usr/bin/env tsx
/**
 * One-shot helper: insert a top-level `%!numbl:assert_jit` directive into
 * every integration test script whose WHOLE body JITs at --opt 1, keeping
 * the directive only when verified (the script still ends in SUCCESS with
 * the directive present). Mirrors src/__tests__/test-scripts.test.ts so
 * "passes" means exactly what the suite means. Not wired into npm; run
 * manually: `npx tsx scripts/add_toplevel_assert_jit.ts`.
 */
import { readFileSync, writeFileSync, readdirSync, statSync } from "node:fs";
import { join, dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { executeCode } from "../src/numbl-core/executeCode.js";
import type { WorkspaceFile } from "../src/numbl-core/workspace/types.js";
import { NodeFileIOAdapter } from "../src/cli-fileio.js";
import { NodeSystemAdapter } from "../src/cli-system.js";

const DIRECTIVE = "%!numbl:assert_jit";

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

/** Insert the directive after the leading single-line-comment / blank
 *  block, before the first code line. If a block comment (`%{`) appears
 *  in the leading region, fall back to inserting at the very top so we
 *  never land inside a comment block. Idempotent. */
function withDirective(src: string): string | null {
  const lines = src.split("\n");
  let insertAt = 0;
  for (let i = 0; i < lines.length; i++) {
    const t = lines[i].trim();
    if (t === "") continue;
    if (t.startsWith("%{") || t.startsWith("%}")) {
      insertAt = 0;
      break;
    }
    if (t.startsWith("%")) continue;
    insertAt = i;
    break;
  }
  if (lines[insertAt]?.trim() === DIRECTIVE) return null; // already present
  lines.splice(insertAt, 0, DIRECTIVE);
  return lines.join("\n");
}

function passesWithSuccess(filepath: string, source: string): boolean {
  const scriptDir = dirname(filepath);
  try {
    const result = executeCode(
      source,
      {
        displayResults: true,
        fileIO: new NodeFileIOAdapter(),
        system: new NodeSystemAdapter(),
      },
      scanMFiles(scriptDir, filepath),
      filepath,
      [scriptDir]
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

const added: string[] = [];
let skipped = 0;
let notJit = 0;
for (const f of files) {
  const original = readFileSync(f, "utf-8");
  const modified = withDirective(original);
  if (modified === null) {
    skipped++;
    continue;
  }
  if (passesWithSuccess(f, modified)) {
    writeFileSync(f, modified);
    added.push(f);
  } else {
    notJit++;
  }
}

console.log(`test files:        ${files.length}`);
console.log(`directive ADDED:   ${added.length}`);
console.log(`already present:   ${skipped}`);
console.log(`not whole-scope:   ${notJit}`);
console.log("\nAdded to:");
for (const f of added) console.log("  " + f.replace(testDir + "/", ""));
