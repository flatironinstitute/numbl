import {
  PackageIndex,
  PackageIndexEntry,
  MipArchitecture,
  MipBackend,
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
    const entry = index.packages.find(
      p => p.name === packageName && p.architecture === candidate
    );
    if (entry) return entry;
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
