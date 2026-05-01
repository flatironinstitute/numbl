/**
 * Double-dispose safety tests — runs every integration test script at
 * --opt 0 (interpreter only) and asserts that no DoubleDisposeError is
 * thrown. These scripts exercise the full breadth of the interpreter's
 * value-semantics paths (struct mutation, cell arrays, class instances,
 * closures, varargout, etc.) and are the most thorough way to catch any
 * aliasing bugs that cause the same buffer to be disposed twice.
 *
 * Double-dispose detection is always-on: `disposeFloat64` / `disposeFloatX`
 * in `runtime/alloc.ts` throw unconditionally when a buffer is disposed
 * while it is still in the pool's WeakSet.
 */

import { describe, it } from "vitest";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { executeCode } from "../numbl-core/executeCode.js";
import type { WorkspaceFile } from "../numbl-core/workspace/types.js";
import { NodeFileIOAdapter } from "../cli-fileio.js";
import { NodeSystemAdapter } from "../cli-system.js";

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
        if (!entry.startsWith("@") && !entry.startsWith("+")) {
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

const thisDir = dirname(fileURLToPath(import.meta.url));
const testDir = resolve(thisDir, "../../numbl_test_scripts");
const testFiles = findTestFiles(testDir);

describe("double-dispose: integration scripts at --opt 0", () => {
  for (const filepath of testFiles) {
    const rel = relative(resolve(thisDir, "../.."), filepath);

    it(rel, () => {
      const source = readFileSync(filepath, "utf-8");
      const scriptDir = dirname(filepath);
      const workspaceFiles = scanMFiles(scriptDir, filepath);
      const searchPaths = [scriptDir];

      // Run at opt 0 (interpreter only) — no JIT dispose paths to worry about,
      // and the interpreter exercises the most dispose sites.
      try {
        executeCode(
          source,
          {
            optimization: "0",
            displayResults: true,
            fileIO: new NodeFileIOAdapter(),
            system: new NodeSystemAdapter(),
          },
          workspaceFiles,
          filepath,
          searchPaths
        );
      } catch (err: unknown) {
        // Only fail on double-dispose; other errors (network, expected
        // runtime errors, etc.) are the integration suite's concern.
        if (err instanceof Error && err.name === "DoubleDisposeError") {
          throw err;
        }
      }
    });
  }
});
