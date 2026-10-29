/**
 * `numbl run-tests`: run .m test scripts and check for a trailing SUCCESS line.
 * `runTestScript` is also used by the vitest wrapper and scripts/.
 */

import { readFileSync } from "fs";
import { dirname, relative, resolve } from "path";
import { executeCode } from "./numbl-core/executeCode.js";
import {
  diagnoseErrors,
  formatDiagnostics,
  getSourceForFile,
} from "./numbl-core/diagnostics";
import type { OptLevel } from "./numbl-core/executors/plugins.js";
import type { WorkspaceFile } from "./numbl-core/workspace/types.js";
import type { NumblHost } from "./numbl-core/host.js";
import { NodeFileIOAdapter } from "./cli-fileio.js";
import { NodeSystemAdapter } from "./cli-system.js";
import { findTestScripts, scanMFiles } from "./cli-scan.js";

/** Workspace for a test script: its directory, minus the script itself. */
export function testScriptWorkspace(filepath: string): WorkspaceFile[] {
  return scanMFiles(dirname(filepath), filepath);
}

/**
 * Run one test script with its directory as the search path. Returns the
 * output and its last non-empty line (a pass is "SUCCESS"); numbl errors
 * propagate.
 */
export function runTestScript(
  filepath: string,
  source: string,
  options: {
    optimization?: OptLevel;
    host?: NumblHost;
    workspaceFiles?: WorkspaceFile[];
  } = {}
): { output: string; lastLine: string } {
  const result = executeCode(
    source,
    {
      displayResults: true,
      optimization: options.optimization ?? "1",
      fileIO: new NodeFileIOAdapter(),
      system: new NodeSystemAdapter(),
      host: options.host,
    },
    options.workspaceFiles ?? testScriptWorkspace(filepath),
    filepath,
    [dirname(filepath)]
  );
  const output = result.output.join("");
  const lines = output.split("\n").filter(l => l.length > 0);
  return { output, lastLine: lines.length > 0 ? lines[lines.length - 1] : "" };
}

export function runTests(
  dir: string,
  optimization: OptLevel,
  host: NumblHost
): void {
  const testFiles = findTestScripts(resolve(process.cwd(), dir));

  if (testFiles.length === 0) {
    console.error(`No .m test files found in ${dir}`);
    process.exit(1);
  }

  let pass = 0;
  const failedScripts: string[] = [];

  for (const filepath of testFiles) {
    const rel = relative(process.cwd(), filepath);
    const source = readFileSync(filepath, "utf-8");
    const workspaceFiles = testScriptWorkspace(filepath);

    try {
      const { output, lastLine } = runTestScript(filepath, source, {
        optimization,
        host,
        workspaceFiles,
      });
      if (lastLine === "SUCCESS") {
        console.log(`PASS  ${rel}`);
        pass++;
      } else {
        console.log(`FAIL  ${rel}`);
        console.log(output.replace(/^/gm, "      "));
        failedScripts.push(rel);
      }
    } catch (error) {
      const diags = diagnoseErrors(error, source, filepath, workspaceFiles);
      const getSource = (file: string) =>
        getSourceForFile(file, filepath, source, workspaceFiles);
      console.log(`FAIL  ${rel}`);
      console.log(
        `      ${formatDiagnostics(diags, getSource).replace(/\n/g, "\n      ")}`
      );
      failedScripts.push(rel);
    }
  }

  console.log("");
  console.log(`Results: ${pass} passed, ${failedScripts.length} failed`);

  if (failedScripts.length > 0) {
    console.log("");
    console.log("Failed scripts:");
    for (const s of failedScripts) {
      console.log(`  ${s}`);
    }
    process.exit(1);
  }
}
