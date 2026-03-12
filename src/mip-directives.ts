/**
 * MIP directive handling for the CLI (Node.js).
 * Re-exports pure parsing from mip-directives-core.ts and adds
 * the Node-specific processMipLoad() implementation.
 */

import { readFileSync, existsSync } from "fs";
import { join, dirname, basename, extname } from "path";
import { homedir } from "os";
import { executeCode } from "./numbl-core/executeCode.js";
import { toString, RTV } from "./numbl-core/runtime/index.js";

// Re-export pure parsing (no Node deps)
export {
  extractMipDirectives,
  type MipDirective,
  type MipDirectiveResult,
} from "./mip-directives-core.js";

export interface MipLoadResult {
  packageName: string;
  paths: string[];
}

export function processMipLoad(
  packageName: string,
  _visited?: Set<string>
): MipLoadResult[] {
  const visited = _visited ?? new Set<string>();
  if (visited.has(packageName)) return [];
  visited.add(packageName);

  const mipDir = process.env.MIP_DIR || join(homedir(), ".mip");
  const pkgDir = join(mipDir, "packages", packageName);
  const loadScript = join(pkgDir, "load_package.m");

  if (!existsSync(loadScript)) {
    throw new Error(
      `mip load ${packageName}: package not found (expected ${loadScript})`
    );
  }

  // Load dependencies first by reading mip.json
  const allResults: MipLoadResult[] = [];
  const mipJsonPath = join(pkgDir, "mip.json");
  if (existsSync(mipJsonPath)) {
    try {
      const mipJson = JSON.parse(readFileSync(mipJsonPath, "utf-8"));
      const deps: string[] = mipJson.dependencies ?? [];
      for (const dep of deps) {
        const depPkgDir = join(mipDir, "packages", dep);
        if (!existsSync(depPkgDir)) {
          console.warn(
            `Warning: dependency "${dep}" of "${packageName}" is not installed; skipping`
          );
          continue;
        }
        allResults.push(...processMipLoad(dep, visited));
      }
    } catch {
      // If mip.json is malformed, continue without dependencies
    }
  }

  const source = readFileSync(loadScript, "utf-8");
  const collectedPaths: string[] = [];

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const customBuiltins: Record<string, (nargout: number, args: any[]) => any> =
    {
      addpath: (_nargout, args) => {
        for (const arg of args) {
          const p = toString(arg);
          // addpath supports multiple dirs separated by pathsep
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
          // Return full path without .m extension
          return RTV.char(loadScript.replace(/\.m$/, ""));
        }
        return RTV.char("load_package");
      },
      fileparts: (nargout, args) => {
        const p = toString(args[0]);
        const dir = dirname(p);
        const ext = extname(p);
        const name = basename(p, ext);
        if (nargout <= 1) return RTV.char(dir);
        if (nargout === 2) return [RTV.char(dir), RTV.char(name)];
        return [RTV.char(dir), RTV.char(name), RTV.char(ext)];
      },
      fullfile: (_nargout, args) => {
        const parts = args.map(a => toString(a));
        return RTV.char(join(...parts));
      },
    };

  executeCode(source, { customBuiltins }, [], loadScript);

  // Dependencies first, then this package's paths
  allResults.push({ packageName, paths: collectedPaths });
  return allResults;
}
