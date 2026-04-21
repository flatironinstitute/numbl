#!/usr/bin/env tsx
// Copies numbl_test_scripts/ into dist/test-scripts/ and writes a manifest
// that the browser test-runner entry (src/test-runner/) consumes.
//
// Manifest shape is defined in ./test-scripts-manifest.ts.

import { mkdirSync, readFileSync, writeFileSync } from "fs";
import { dirname, join, resolve } from "path";
import { buildManifest, walkAllFiles } from "./test-scripts-manifest.js";

const ROOT = resolve(import.meta.dirname, "..");
const SRC_DIR = join(ROOT, "numbl_test_scripts");
const OUT_DIR = join(ROOT, "dist", "test-scripts");

mkdirSync(OUT_DIR, { recursive: true });

const files = walkAllFiles(SRC_DIR);
for (const src of files) {
  const rel = src.slice(SRC_DIR.length + 1);
  const dst = join(OUT_DIR, rel);
  mkdirSync(dirname(dst), { recursive: true });
  writeFileSync(dst, readFileSync(src));
}

const manifest = buildManifest(SRC_DIR);
writeFileSync(join(OUT_DIR, "manifest.json"), JSON.stringify(manifest));

console.log(
  `test-scripts copied: ${files.length} file(s), ${manifest.tests.length} test(s) (${manifest.tests.filter(t => t.skip).length} skipped)`
);
