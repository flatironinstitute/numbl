/**
 * Run all .m integration test scripts under vitest so their coverage is
 * collected by vitest's native V8 coverage provider — the same provider
 * used for unit tests.  This gives us a single, consistent coverage report
 * across both test suites.
 */

/// <reference types="node" />

import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { findTestScripts } from "../cli-scan.js";
import { runTestScript } from "../cli-run-tests.js";

const thisDir = dirname(fileURLToPath(import.meta.url));
const testDir = resolve(thisDir, "../../numbl_test_scripts");
const testFiles = findTestScripts(testDir);

describe("integration test scripts", () => {
  for (const filepath of testFiles) {
    const rel = relative(resolve(thisDir, "../.."), filepath);

    it(rel, () => {
      const source = readFileSync(filepath, "utf-8");
      expect(runTestScript(filepath, source).lastLine).toBe("SUCCESS");
    });
  }
});
