import {
  PackageIndex,
  PackageIndexEntry,
  MipArchitecture,
  MipBackend,
  compareVersions,
} from "./types.js";

const DEFAULT_INDEX_URL = "https://mip-org.github.io/mip-core/index.json";

export async function fetchPackageIndex(
  backend: MipBackend,
  indexUrl: string = DEFAULT_INDEX_URL
): Promise<PackageIndex> {
  return backend.fetchJson<PackageIndex>(indexUrl);
}

/** Find the best entry for a package on the given architecture.
 *  Priority: exact native match → wasm → all */
export function findPackageEntry(
  index: PackageIndex,
  packageName: string,
  arch: MipArchitecture
): PackageIndexEntry | undefined {
  for (const candidate of ["numbl_" + arch, "numbl_wasm", "any"]) {
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

export function listAvailablePackages(
  index: PackageIndex,
  arch: MipArchitecture
): PackageIndexEntry[] {
  return index.packages.filter(
    p =>
      p.architecture === "numbl_" + arch ||
      p.architecture === "numbl_wasm" ||
      p.architecture === "any"
  );
}
