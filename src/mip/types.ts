/** A single entry from the remote package index */
export interface PackageIndexEntry {
  name: string;
  description: string;
  version: string;
  release_number: number;
  dependencies: string[];
  homepage: string;
  repository: string;
  license: string;
  architecture: string;
  build_on: string;
  usage_examples: string[];
  exposed_symbols: string[];
  source_hash?: string;
  timestamp: string;
  prepare_duration: number;
  compile_duration: number;
  mhl_url: string;
  mip_json_url: string;
}

/** The top-level structure of index.json */
export interface PackageIndex {
  packages: PackageIndexEntry[];
  total_packages: number;
  last_updated: string;
}

/** The mip.json that lives inside each installed package */
export interface MipJson {
  name: string;
  version: string;
  dependencies: string[];
  exposed_symbols: string[];
  description?: string;
  license?: string;
  homepage?: string;
  architecture?: string;
}

/**
 * Compare two version strings component-by-component.
 * Returns 1 if v1 > v2, -1 if v1 < v2, 0 if equal.
 */
export function compareVersions(v1: string, v2: string): number {
  const parts1 = v1.split(".").map(Number);
  const parts2 = v2.split(".").map(Number);
  const maxLen = Math.max(parts1.length, parts2.length);
  for (let i = 0; i < maxLen; i++) {
    const a = parts1[i] ?? 0;
    const b = parts2[i] ?? 0;
    if (a > b) return 1;
    if (a < b) return -1;
  }
  return 0;
}

export type MipArchitecture =
  | "linux_x86_64"
  | "macos_arm64"
  | "macos_x86_64"
  | "windows_x86_64";

/**
 * Abstraction for filesystem and network operations.
 * Node.js CLI provides one implementation; a future web app provides another.
 */
export interface MipBackend {
  getMipDir(): string;
  dirExists(path: string): Promise<boolean>;
  readTextFile(path: string): Promise<string | null>;
  writeTextFile(path: string, content: string): Promise<void>;
  readJsonFile<T>(path: string): Promise<T | null>;
  listDirs(path: string): Promise<string[]>;
  removeDir(path: string): Promise<void>;
  downloadAndExtractZip(url: string, targetDir: string): Promise<void>;
  fetchJson<T>(url: string): Promise<T>;
  getArchitecture(): MipArchitecture;
}
