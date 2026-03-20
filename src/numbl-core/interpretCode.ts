/**
 * Entry point for interpreted execution.
 *
 * Parallel to executeCode.ts but walks the AST directly
 * instead of lowering → codegen → eval.
 */

import { type ExecOptions, type ExecResult } from "./executor/types.js";
import type { WorkspaceFile } from "../numbl-core/workspace/index.js";
import { Runtime } from "./runtime/runtime.js";
import { RTV } from "./runtime/constructors.js";
import { RuntimeError } from "../numbl-core/runtime/index.js";
import { parseMFile } from "./parser/index.js";
import { Interpreter } from "./interpreter/interpreter.js";

export function interpretCode(
  source: string,
  options: ExecOptions = {},
  workspaceFiles?: WorkspaceFile[],
  mainFileName: string = "script.m"
): ExecResult {
  // Parse
  const ast = parseMFile(source, mainFileName);

  if (options.log) {
    options.log("AST parsed, starting interpretation");
  }

  // Create runtime (same as codegen path — builtins, output, etc.)
  const rt = new Runtime(options, options.initialVariableValues);

  // Register .js user functions (same as codegen path)
  // Note: for now, skip jsUserFunctions — they require the codegen pipeline.
  // This can be added later if needed.

  // Apply custom builtins
  if (options.customBuiltins) {
    Object.assign(rt.builtins, options.customBuiltins);
  }

  // Wire up eval callback
  rt.evalLocalCallback = (code, initialVars, onOutput) => {
    const evalResult = interpretCode(code, {
      onOutput,
      displayResults: false,
      initialVariableValues: initialVars,
    });
    return {
      returnValue: evalResult.returnValue,
      variableValues: evalResult.variableValues,
    };
  };

  // Create and run the interpreter
  const interpreter = new Interpreter(
    rt,
    workspaceFiles,
    options.initialVariableValues
  );

  try {
    interpreter.run(ast);

    const result: ExecResult = {
      output: rt.outputLines,
      generatedJS: "// interpreted mode — no JS generated",
      plotInstructions: rt.plotInstructions,
      returnValue: interpreter.ans ?? RTV.num(0),
      variableValues: interpreter.getVariableValues(),
      holdState: rt.holdState,
    };

    return result;
  } catch (e) {
    if (e instanceof RuntimeError) {
      // Annotate with file/line info
      if (e.line === null && rt.$file && rt.$line > 0) {
        e.file = rt.$file;
        e.line = rt.$line;
      }
      throw e;
    }
    const re = new RuntimeError(e instanceof Error ? e.message : String(e));
    if (rt.$file && rt.$line > 0) {
      re.file = rt.$file;
      re.line = rt.$line;
    }
    throw re;
  }
}
