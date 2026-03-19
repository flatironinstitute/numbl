/**
 * Browser-side MIP package loading.
 * Fetches .mhl packages from CDN, caches in IndexedDB,
 * processes load_package.m, and returns workspace files.
 */

import { unzipSync } from "fflate";
import { db, type MipPackageCache } from "../db/schema.js";
import {
  compareVersions,
  type PackageIndex,
  type PackageIndexEntry,
} from "./types.js";
import { executeCode } from "../numbl-core/executeCode.js";
import { toString, RTV } from "../numbl-core/runtime/index.js";
import type { WorkspaceFile } from "../numbl-core/workspace/index.js";

const DEFAULT_INDEX_URL = "https://mip-org.github.io/mip-core/index.json";

// Session-level cache for the package index to avoid re-fetching
let cachedIndex: PackageIndex | null = null;

async function fetchIndex(): Promise<PackageIndex> {
  if (cachedIndex) return cachedIndex;
  const resp = await fetch(DEFAULT_INDEX_URL);
  if (!resp.ok) {
    throw new Error(`Failed to fetch MIP package index: ${resp.status}`);
  }
  cachedIndex = (await resp.json()) as PackageIndex;
  return cachedIndex;
}

function findEntry(
  index: PackageIndex,
  packageName: string
): PackageIndexEntry | undefined {
  // In the browser, prefer wasm packages, then fall back to "any"
  for (const candidate of ["numbl_wasm", "any"]) {
    const matches = index.packages.filter(
      p => p.name === packageName && p.architecture === candidate
    );
    if (matches.length > 0) {
      return matches.reduce((best, cur) =>
        compareVersions(cur.version, best.version) > 0 ? cur : best
      );
    }
  }
  return undefined;
}

/**
 * Resolve all transitive dependencies for a package.
 * Returns entries in dependency-first order.
 */
function resolveDeps(
  index: PackageIndex,
  packageName: string
): PackageIndexEntry[] {
  const resolved: PackageIndexEntry[] = [];
  const visited = new Set<string>();

  function visit(name: string): void {
    if (visited.has(name)) return;
    visited.add(name);
    const entry = findEntry(index, name);
    if (!entry) {
      throw new Error(`MIP package "${name}" not found (tried wasm, all)`);
    }
    for (const dep of entry.dependencies) {
      visit(dep);
    }
    resolved.push(entry);
  }

  visit(packageName);
  return resolved;
}

/**
 * Fetch and extract an .mhl package, returning all files.
 */
async function fetchAndExtractPackage(
  entry: PackageIndexEntry,
  onProgress?: (msg: string) => void
): Promise<{ path: string; source: string; data?: Uint8Array }[]> {
  onProgress?.(`Downloading ${entry.name}...`);
  const resp = await fetch(entry.mhl_url);
  if (!resp.ok) {
    throw new Error(`Failed to download package ${entry.name}: ${resp.status}`);
  }
  const buffer = await resp.arrayBuffer();
  const zipData = new Uint8Array(buffer);

  onProgress?.(`Extracting ${entry.name}...`);
  const extracted = unzipSync(zipData);

  const files: { path: string; source: string; data?: Uint8Array }[] = [];
  for (const [path, data] of Object.entries(extracted)) {
    // Skip directories (empty entries ending with /)
    if (path.endsWith("/")) continue;
    if (path.endsWith(".wasm")) {
      // Binary file — store raw bytes
      files.push({ path, source: "", data: new Uint8Array(data) });
    } else if (path.endsWith(".m") || path.endsWith(".js")) {
      const source = new TextDecoder().decode(data);
      files.push({ path, source });
    }
  }
  return files;
}

/**
 * Process load_package.m to determine which directories should be on the path.
 * Uses executeCode with custom builtins, mirroring the CLI implementation.
 */
function processLoadPackage(
  loadPackageSource: string,
  packageName: string
): string[] {
  const virtualPkgDir = `/mip/packages/${packageName}`;
  const virtualLoadScript = `${virtualPkgDir}/load_package.m`;
  const collectedPaths: string[] = [];

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const customBuiltins: Record<string, (nargout: number, args: any[]) => any> =
    {
      addpath: (_nargout, args) => {
        for (const arg of args) {
          const p = toString(arg);
          for (const part of p.split(":")) {
            if (part.length > 0) {
              collectedPaths.push(part);
            }
          }
        }
        return undefined;
      },
      mfilename: (_nargout, args) => {
        if (args.length > 0 && toString(args[0]) === "fullpath") {
          return RTV.char(virtualLoadScript.replace(/\.m$/, ""));
        }
        return RTV.char("load_package");
      },
      fileparts: (nargout, args) => {
        const p = toString(args[0]);
        const lastSlash = p.lastIndexOf("/");
        const dir = lastSlash >= 0 ? p.substring(0, lastSlash) : ".";
        const base = lastSlash >= 0 ? p.substring(lastSlash + 1) : p;
        const dotIdx = base.lastIndexOf(".");
        const name = dotIdx >= 0 ? base.substring(0, dotIdx) : base;
        const ext = dotIdx >= 0 ? base.substring(dotIdx) : "";
        if (nargout <= 1) return RTV.char(dir);
        if (nargout === 2) return [RTV.char(dir), RTV.char(name)];
        return [RTV.char(dir), RTV.char(name), RTV.char(ext)];
      },
      fullfile: (_nargout, args) => {
        const parts = args.map(a => toString(a));
        // Join with / and normalize double slashes
        return RTV.char(parts.join("/").replace(/\/+/g, "/"));
      },
    };

  executeCode(loadPackageSource, { customBuiltins }, [], virtualLoadScript);

  return collectedPaths;
}

/**
 * Check if a relative path (e.g., "file.m" or "@Foo/bar.m" or "+pkg/func.m")
 * would be visible on the search path. Direct files are visible, and files
 * inside @class, +package, or private directories are visible (recursively).
 */
function isOnSearchPath(relativePath: string): boolean {
  const parts = relativePath.split("/");
  // All intermediate directories (everything except the filename) must be @/+/private
  for (let i = 0; i < parts.length - 1; i++) {
    const dir = parts[i];
    if (!dir.startsWith("@") && !dir.startsWith("+") && dir !== "private") {
      return false;
    }
  }
  return true;
}

/**
 * Filter package files by the addpath paths, returning only files
 * under those directories as WorkspaceFile[].
 */
function filterFilesByPaths(
  files: { path: string; source: string; data?: Uint8Array }[],
  loadPaths: string[],
  packageName: string
): WorkspaceFile[] {
  const virtualPkgDir = `/mip/packages/${packageName}`;
  const result: WorkspaceFile[] = [];

  for (const loadPath of loadPaths) {
    // loadPath is like /mip/packages/pkg/subdir
    // files have paths like subdir/file.m (relative to package root)
    // We need to map between them
    const prefix = loadPath.startsWith(virtualPkgDir + "/")
      ? loadPath.substring(virtualPkgDir.length + 1)
      : loadPath;

    for (const file of files) {
      if (file.path === "load_package.m") continue;
      if (!file.path.startsWith(prefix + "/")) continue;
      const rest = file.path.substring(prefix.length + 1);
      // addpath only adds the specific directory, not arbitrary subdirs.
      // But @class, +package, and private dirs are always recursed into.
      if (!isOnSearchPath(rest)) continue;
      const wsFile: WorkspaceFile = {
        name: `${virtualPkgDir}/${file.path}`,
        source: file.source,
      };
      if (file.data) wsFile.data = file.data;
      result.push(wsFile);
    }
  }

  // Deduplicate (a file might match multiple paths)
  const seen = new Set<string>();
  return result.filter(f => {
    if (seen.has(f.name)) return false;
    seen.add(f.name);
    return true;
  });
}

/**
 * Load a single MIP package for use in the browser.
 * Checks IndexedDB cache first, fetches from CDN if needed.
 */
async function loadSinglePackage(
  entry: PackageIndexEntry,
  onProgress?: (msg: string) => void
): Promise<{ workspaceFiles: WorkspaceFile[]; searchPaths: string[] }> {
  // Check cache
  const cached = await db.mipPackages.get(entry.name);
  if (cached && cached.version === entry.version) {
    onProgress?.(`Using cached ${entry.name}`);
    const workspaceFiles = filterFilesByPaths(
      cached.files,
      cached.loadPaths,
      entry.name
    );
    return {
      workspaceFiles,
      searchPaths: cached.loadPaths,
    };
  }

  // Fetch and extract
  const files = await fetchAndExtractPackage(entry, onProgress);

  // Process load_package.m
  const loadPkgFile = files.find(
    f => f.path === "load_package.m" || f.path.endsWith("/load_package.m")
  );
  let loadPaths: string[];
  if (loadPkgFile) {
    onProgress?.(`Processing ${entry.name} load_package.m...`);
    loadPaths = processLoadPackage(loadPkgFile.source, entry.name);
  } else {
    // Fallback: expose all files at package root
    loadPaths = [`/mip/packages/${entry.name}`];
  }

  // Cache in IndexedDB
  const cacheEntry: MipPackageCache = {
    name: entry.name,
    version: entry.version,
    files,
    loadPaths,
    fetchedAt: Date.now(),
  };
  await db.mipPackages.put(cacheEntry);

  const workspaceFiles = filterFilesByPaths(files, loadPaths, entry.name);
  return { workspaceFiles, searchPaths: loadPaths };
}

/**
 * Load a MIP package and all its dependencies for use in the browser.
 * Returns combined workspace files and search paths.
 */
export async function loadMipPackageBrowser(
  packageName: string,
  onProgress?: (msg: string) => void
): Promise<{ workspaceFiles: WorkspaceFile[]; searchPaths: string[] }> {
  onProgress?.(`Resolving ${packageName}...`);
  const index = await fetchIndex();
  const entries = resolveDeps(index, packageName);

  const allWorkspaceFiles: WorkspaceFile[] = [];
  const allSearchPaths: string[] = [];

  for (const entry of entries) {
    const { workspaceFiles, searchPaths } = await loadSinglePackage(
      entry,
      onProgress
    );
    allWorkspaceFiles.push(...workspaceFiles);
    allSearchPaths.push(...searchPaths);
  }

  return { workspaceFiles: allWorkspaceFiles, searchPaths: allSearchPaths };
}
