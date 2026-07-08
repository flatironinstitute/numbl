/**
 * Build the numbl/browser entry (dist-browser/):
 *   1. Bundle src/browser/worker.ts standalone (everything inlined) and
 *      write it as src/browser/generated/worker-code.txt.
 *   2. Bundle src/browser/index.ts -> dist-browser/browser.js, importing the
 *      worker bundle as text so the session can start it from a Blob URL.
 *   3. Emit declarations via tsconfig.browser.json.
 *
 * Run via: npm run build:browser
 */
import { build } from "esbuild";
import { execSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const generatedDir = path.join(root, "src", "browser", "generated");

async function main() {
  // 1. Self-contained worker bundle (no externals — it runs from a blob:
  // URL where nothing can be resolved at runtime).
  const workerResult = await build({
    entryPoints: [path.join(root, "src", "browser", "worker.ts")],
    bundle: true,
    platform: "browser",
    format: "esm",
    write: false,
    logLevel: "warning",
  });
  fs.mkdirSync(generatedDir, { recursive: true });
  fs.writeFileSync(
    path.join(generatedDir, "worker-code.txt"),
    workerResult.outputFiles[0].text
  );

  // 2. Host-side entry with the worker text inlined.
  await build({
    entryPoints: [path.join(root, "src", "browser", "index.ts")],
    bundle: true,
    platform: "browser",
    format: "esm",
    outfile: path.join(root, "dist-browser", "browser.js"),
    loader: { ".txt": "text" },
    logLevel: "warning",
  });

  // 3. Type declarations.
  execSync("npx tsc -p tsconfig.browser.json", { cwd: root, stdio: "inherit" });

  const size = fs.statSync(path.join(root, "dist-browser", "browser.js")).size;
  console.log(`dist-browser/browser.js  ${(size / 1024 / 1024).toFixed(1)}mb`);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
