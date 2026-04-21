// Shared manifest builder used by both the production copy step
// (scripts/copy-test-scripts.ts) and the dev middleware
// (vite.test-runner.config.ts).

import { existsSync, readdirSync, readFileSync, statSync } from "fs";
import { dirname, join, relative } from "path";

export interface TestEntry {
  path: string;
  workspace: string[];
  /** If set, runner skips with this reason instead of executing. */
  skip?: string;
}
export interface Manifest {
  tests: TestEntry[];
  allFiles: string[];
  /**
   * Text sources (.m, .numbl.js, .c), keyed by relative path. Inlined so
   * the runner can populate the in-browser VFS with the full tree before
   * each test runs — required for addpath/rmpath tests that scan
   * sub-library directories.
   */
  sources: Record<string, string>;
  /** Binary fixtures (.wasm, .zip, etc.) base64-encoded. */
  binaries: Record<string, string>;
}

const TEXT_EXTENSIONS = [".m", ".numbl.js", ".c", ".txt", ".json"];

function isTextFile(path: string): boolean {
  return TEXT_EXTENSIONS.some(ext => path.endsWith(ext));
}

function isWorkspaceFile(name: string): boolean {
  return (
    name.endsWith(".m") || name.endsWith(".numbl.js") || name.endsWith(".wasm")
  );
}

export function walkAllFiles(dir: string, out: string[] = []): string[] {
  for (const entry of readdirSync(dir).sort()) {
    const full = join(dir, entry);
    const st = statSync(full);
    if (st.isDirectory()) {
      if (!entry.startsWith(".")) walkAllFiles(full, out);
    } else if (st.isFile()) {
      // Skip shell scripts — they're build tooling, not test assets.
      if (entry.endsWith(".sh")) continue;
      out.push(full);
    }
  }
  return out;
}

export function walkMFiles(dir: string, out: string[] = []): string[] {
  for (const entry of readdirSync(dir).sort()) {
    const full = join(dir, entry);
    const st = statSync(full);
    if (st.isDirectory()) {
      if (!entry.startsWith(".")) walkMFiles(full, out);
    } else if (st.isFile() && entry.endsWith(".m")) {
      out.push(full);
    }
  }
  return out;
}

// Mirrors scanMFiles() in src/__tests__/test-scripts.test.ts — siblings of
// the test file in the same directory, plus @-dirs, +-dirs, and `private/`
// walked recursively. Includes .m, .numbl.js, and .wasm files.
export function scanWorkspace(
  scriptDir: string,
  excludeFile: string
): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(scriptDir).sort()) {
    const full = join(scriptDir, entry);
    if (full === excludeFile) continue;
    const st = statSync(full);
    if (st.isDirectory()) {
      if (
        entry.startsWith("@") ||
        entry.startsWith("+") ||
        entry === "private"
      ) {
        walkAllFiles(full).forEach(f => {
          if (isWorkspaceFile(f)) out.push(f);
        });
      }
    } else if (st.isFile() && isWorkspaceFile(entry)) {
      out.push(full);
    }
  }
  return out;
}

function toPosix(p: string): string {
  return p.split(/[\\/]/).join("/");
}

// Each line: `<path>[|<reason>]` or a `#` comment. Blank lines ignored.
function readSkipList(rootDir: string): Map<string, string> {
  const skipFile = join(rootDir, ".browser-skip");
  const skips = new Map<string, string>();
  if (!existsSync(skipFile)) return skips;
  const content = readFileSync(skipFile, "utf-8");
  for (const raw of content.split("\n")) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const pipe = line.indexOf("|");
    const path = pipe >= 0 ? line.slice(0, pipe).trim() : line;
    const reason =
      pipe >= 0 ? line.slice(pipe + 1).trim() : "(no reason given)";
    skips.set(path, reason);
  }
  return skips;
}

export function buildManifest(rootDir: string): Manifest {
  const all = walkAllFiles(rootDir);
  const allFiles = all.map(f => toPosix(relative(rootDir, f)));
  const sources: Record<string, string> = {};
  const binaries: Record<string, string> = {};
  for (const f of all) {
    const rel = toPosix(relative(rootDir, f));
    if (isTextFile(f)) {
      sources[rel] = readFileSync(f, "utf-8");
    } else {
      binaries[rel] = readFileSync(f).toString("base64");
    }
  }

  const mFiles = all.filter(f => f.endsWith(".m"));
  const skips = readSkipList(rootDir);

  const tests: TestEntry[] = mFiles
    .filter(f => sources[toPosix(relative(rootDir, f))]?.includes("SUCCESS"))
    .map(testAbs => {
      const path = toPosix(relative(rootDir, testAbs));
      const entry: TestEntry = {
        path,
        workspace: scanWorkspace(dirname(testAbs), testAbs).map(f =>
          toPosix(relative(rootDir, f))
        ),
      };
      const skip = skips.get(path);
      if (skip) entry.skip = skip;
      return entry;
    });

  return { tests, allFiles, sources, binaries };
}
