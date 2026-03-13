import { MipBackend, MipJson, PackageIndexEntry } from "./types.js";
import {
  fetchPackageIndex,
  findPackageEntry,
  listAvailablePackages,
} from "./index-client.js";
import { resolveDependencies, computeRequiredPackages } from "./dependency.js";

function packagesDir(mipDir: string): string {
  return `${mipDir}/numbl_packages`;
}
function packageDir(mipDir: string, name: string): string {
  return `${mipDir}/numbl_packages/${name}`;
}
function directlyInstalledPath(mipDir: string): string {
  return `${mipDir}/numbl_packages/directly_installed.txt`;
}

export interface InstallResult {
  installed: string[];
  alreadyInstalled: string[];
}

export interface UninstallResult {
  removed: string[];
  pruned: string[];
  notInstalled: boolean;
}

export interface InstalledPackageInfo {
  name: string;
  version: string;
  isDirect: boolean;
  mipJson: MipJson;
}

export class MipManager {
  constructor(private backend: MipBackend) {}

  async install(
    packageName: string,
    log: (msg: string) => void = console.log
  ): Promise<InstallResult> {
    const mipDir = this.backend.getMipDir();
    const arch = this.backend.getArchitecture();

    log("Fetching package index...");
    const index = await fetchPackageIndex(this.backend);

    const toInstall = resolveDependencies(index, packageName, arch);

    const installed: string[] = [];
    const alreadyInstalled: string[] = [];

    for (const entry of toInstall) {
      const pkgDir = packageDir(mipDir, entry.name);
      if (await this.backend.dirExists(pkgDir)) {
        alreadyInstalled.push(entry.name);
        log(`  ${entry.name} already installed, skipping`);
        continue;
      }

      log(`  Installing ${entry.name} v${entry.version}...`);
      await this.backend.downloadAndExtractZip(entry.mhl_url, pkgDir);

      const mipJson = await this.backend.readJsonFile<MipJson>(
        `${pkgDir}/mip.json`
      );
      if (!mipJson) {
        await this.backend.removeDir(pkgDir);
        throw new Error(`Package "${entry.name}" is invalid: missing mip.json`);
      }

      installed.push(entry.name);
    }

    await this.addDirectlyInstalled(packageName);

    return { installed, alreadyInstalled };
  }

  async uninstall(
    packageName: string,
    log: (msg: string) => void = console.log
  ): Promise<UninstallResult> {
    const mipDir = this.backend.getMipDir();
    const arch = this.backend.getArchitecture();

    const pkgDir = packageDir(mipDir, packageName);
    if (!(await this.backend.dirExists(pkgDir))) {
      return { removed: [], pruned: [], notInstalled: true };
    }

    log(`Removing ${packageName}...`);
    await this.backend.removeDir(pkgDir);
    await this.removeDirectlyInstalled(packageName);

    log("Checking for orphaned dependencies...");
    const index = await fetchPackageIndex(this.backend);
    const directNames = await this.getDirectlyInstalled();
    const required = computeRequiredPackages(directNames, index, arch);

    const allInstalled = await this.backend.listDirs(packagesDir(mipDir));
    const pruned: string[] = [];

    for (const dirName of allInstalled) {
      if (!required.has(dirName) && !directNames.includes(dirName)) {
        log(`  Pruning orphaned dependency: ${dirName}`);
        await this.backend.removeDir(packageDir(mipDir, dirName));
        pruned.push(dirName);
      }
    }

    return { removed: [packageName], pruned, notInstalled: false };
  }

  async list(): Promise<InstalledPackageInfo[]> {
    const mipDir = this.backend.getMipDir();
    const pkgsDir = packagesDir(mipDir);

    if (!(await this.backend.dirExists(pkgsDir))) {
      return [];
    }

    const directNames = new Set(await this.getDirectlyInstalled());
    const dirs = await this.backend.listDirs(pkgsDir);
    const result: InstalledPackageInfo[] = [];

    for (const dirName of dirs) {
      const mipJson = await this.backend.readJsonFile<MipJson>(
        `${pkgsDir}/${dirName}/mip.json`
      );
      if (!mipJson) continue;

      result.push({
        name: mipJson.name,
        version: mipJson.version,
        isDirect: directNames.has(dirName),
        mipJson,
      });
    }

    return result;
  }

  async avail(): Promise<PackageIndexEntry[]> {
    const arch = this.backend.getArchitecture();
    const index = await fetchPackageIndex(this.backend);
    return listAvailablePackages(index, arch);
  }

  async info(packageName: string): Promise<{
    installed: InstalledPackageInfo | null;
    available: PackageIndexEntry | null;
  }> {
    const mipDir = this.backend.getMipDir();
    const arch = this.backend.getArchitecture();

    let installed: InstalledPackageInfo | null = null;
    const mipJson = await this.backend.readJsonFile<MipJson>(
      packageDir(mipDir, packageName) + "/mip.json"
    );
    if (mipJson) {
      const directNames = new Set(await this.getDirectlyInstalled());
      installed = {
        name: mipJson.name,
        version: mipJson.version,
        isDirect: directNames.has(packageName),
        mipJson,
      };
    }

    const index = await fetchPackageIndex(this.backend);
    const available = findPackageEntry(index, packageName, arch) ?? null;

    return { installed, available };
  }

  // ── directly_installed.txt helpers ──

  private async getDirectlyInstalled(): Promise<string[]> {
    const mipDir = this.backend.getMipDir();
    const content = await this.backend.readTextFile(
      directlyInstalledPath(mipDir)
    );
    if (!content) return [];
    return content
      .split("\n")
      .map(s => s.trim())
      .filter(s => s.length > 0);
  }

  private async addDirectlyInstalled(name: string): Promise<void> {
    const names = await this.getDirectlyInstalled();
    if (!names.includes(name)) {
      names.push(name);
      names.sort();
    }
    const mipDir = this.backend.getMipDir();
    await this.backend.writeTextFile(
      directlyInstalledPath(mipDir),
      names.join("\n") + "\n"
    );
  }

  private async removeDirectlyInstalled(name: string): Promise<void> {
    const names = await this.getDirectlyInstalled();
    const filtered = names.filter(n => n !== name);
    const mipDir = this.backend.getMipDir();
    await this.backend.writeTextFile(
      directlyInstalledPath(mipDir),
      filtered.join("\n") + "\n"
    );
  }
}
