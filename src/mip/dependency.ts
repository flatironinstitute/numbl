import { PackageIndex, PackageIndexEntry, MipArchitecture } from "./types.js";
import { findPackageEntry } from "./index-client.js";

/**
 * Resolve all dependencies for a package recursively.
 * Returns entries in topological order (dependencies first).
 */
export function resolveDependencies(
  index: PackageIndex,
  packageName: string,
  arch: MipArchitecture
): PackageIndexEntry[] {
  const resolved: PackageIndexEntry[] = [];
  const visited = new Set<string>();

  function visit(name: string): void {
    if (visited.has(name)) return;
    visited.add(name);

    const entry = findPackageEntry(index, name, arch);
    if (!entry) {
      throw new Error(
        `Package "${name}" not found for architecture "${arch}" (tried ${arch}, wasm, all)`
      );
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
 * Compute which packages are transitively required by the given
 * directly-installed packages. Used for pruning orphaned dependencies.
 */
export function computeRequiredPackages(
  directlyInstalled: string[],
  index: PackageIndex,
  arch: MipArchitecture
): Set<string> {
  const required = new Set<string>();

  function visit(name: string): void {
    if (required.has(name)) return;
    required.add(name);

    const entry = findPackageEntry(index, name, arch);
    if (!entry) return;

    for (const dep of entry.dependencies) {
      visit(dep);
    }
  }

  for (const name of directlyInstalled) {
    visit(name);
  }

  return required;
}
