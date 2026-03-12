/**
 * Run all .m integration test scripts under vitest so their coverage is
 * collected by vitest's native V8 coverage provider — the same provider
 * used for unit tests.  This gives us a single, consistent coverage report
 * across both test suites.
 */

import { describe, it, expect } from "vitest";
import { readFileSync, readdirSync, statSync } from "fs";
import { join, dirname, relative, resolve } from "path";
import { executeCode } from "../numbl-core/executeCode.js";
import type { WorkspaceFile } from "../numbl-core/workspace/types.js";

// ── helpers (mirror cli.ts logic) ────────────────────────────────────

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
        ) {
          files.push(...scanMFiles(fullPath, excludeFile));
        }
      } else if (
        stat.isFile() &&
        (entry.endsWith(".m") || entry.endsWith(".js"))
      ) {
        files.push({ name: fullPath, source: readFileSync(fullPath, "utf-8") });
      }
    } catch {
      continue;
    }
  }
  return files;
}

function findTestFiles(dir: string): string[] {
  const results: string[] = [];
  function walk(current: string) {
    let entries: string[];
    try {
      entries = readdirSync(current);
    } catch {
      return;
    }
    entries.sort();
    for (const entry of entries) {
      const fullPath = join(current, entry);
      let stat;
      try {
        stat = statSync(fullPath);
      } catch {
        continue;
      }
      if (stat.isDirectory()) {
        if (
          !entry.startsWith("@") &&
          !entry.startsWith("+") &&
          entry !== "wasm"
        ) {
          walk(fullPath);
        }
      } else if (stat.isFile() && entry.endsWith(".m")) {
        const content = readFileSync(fullPath, "utf-8");
        if (content.includes("SUCCESS")) {
          results.push(fullPath);
        }
      }
    }
  }
  walk(dir);
  return results;
}

// ── discover and register tests ──────────────────────────────────────

const testDir = resolve(__dirname, "../../numbl_test_scripts");
const testFiles = findTestFiles(testDir);

describe("integration test scripts", () => {
  for (const filepath of testFiles) {
    const rel = relative(resolve(__dirname, "../.."), filepath);

    it(rel, () => {
      const source = readFileSync(filepath, "utf-8");
      const scriptDir = dirname(filepath);
      const workspaceFiles = scanMFiles(scriptDir, filepath);
      const searchPaths = [scriptDir];

      const result = executeCode(
        source,
        { displayResults: true },
        workspaceFiles,
        filepath,
        searchPaths
      );

      const outputText = result.output.join("");
      const lines = outputText.split("\n").filter(l => l.length > 0);
      const lastLine = lines.length > 0 ? lines[lines.length - 1] : "";
      expect(lastLine).toBe("SUCCESS");
    });
  }
});
