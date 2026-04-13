import { readFileSync, writeFileSync } from "fs";
import { execSync } from "child_process";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const pkgPath = resolve(root, "package.json");
const versionTsPath = resolve(root, "src/numbl-core/version.ts");

const pkg = JSON.parse(readFileSync(pkgPath, "utf-8"));
const [major, minor, patch] = pkg.version.split(".").map(Number);
const newVersion = `${major}.${minor}.${patch + 1}`;

pkg.version = newVersion;
writeFileSync(pkgPath, JSON.stringify(pkg, null, 2) + "\n");

const versionTs = readFileSync(versionTsPath, "utf-8");
writeFileSync(
  versionTsPath,
  versionTs.replace(
    /export const NUMBL_VERSION = ".*"/,
    `export const NUMBL_VERSION = "${newVersion}"`
  )
);

console.log(`Bumped version: ${major}.${minor}.${patch} → ${newVersion}`);
console.log("Running npm install to sync package-lock.json...");
execSync("npm install", { cwd: root, stdio: "inherit" });
console.log("Done.");
