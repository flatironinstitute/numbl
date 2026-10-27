/**
 * Shared utilities for scanning directories for .m/.numbl.js/.wasm workspace
 * files and for discovering .m test scripts.
 * Extracted from cli.ts to avoid circular imports (cli-fileio → cli → main()).
 */

import { readFileSync, readdirSync, statSync } from "fs";
import { join } from "path";
import type { WorkspaceFile } from "./numbl-core/workspace/types.js";

export function scanMFiles(
  dirPath: string,
  excludeFile?: string
): WorkspaceFile[] {
  const files: WorkspaceFile[] = [];

  let entries: string[];
  try {
    entries = readdirSync(dirPath);
  } catch {
    // Can't read directory (e.g. permission denied) — skip it
    return files;
  }

  for (const entry of entries) {
    const fullPath = join(dirPath, entry);

    if (excludeFile && fullPath === excludeFile) {
      continue;
    }

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
        const source = readFileSync(fullPath, "utf-8");
        files.push({
          name: fullPath,
          source,
        });
      } else if (stat.isFile() && entry.endsWith(".wasm")) {
        const data = readFileSync(fullPath);
        files.push({
          name: fullPath,
          source: "",
          data: new Uint8Array(data),
        });
      }
    } catch (err) {
      console.warn(
        `Warning: could not read ${fullPath}: ${err instanceof Error ? err.message : String(err)}`
      );
    }
  }

  return files;
}

/**
 * Find integration test scripts under `dir`: `.m` files containing "SUCCESS",
 * in sorted depth-first order. Skips class (`@`), package (`+`) and `wasm`
 * directories. Shared by `numbl run-tests`, the vitest wrapper and scripts/.
 */
export function findTestScripts(dir: string): string[] {
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
        if (readFileSync(fullPath, "utf-8").includes("SUCCESS")) {
          results.push(fullPath);
        }
      }
    }
  }

  walk(dir);
  return results;
}
