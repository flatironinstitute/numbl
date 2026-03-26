/**
 * Entry point for on-demand code execution.
 *
 * Drop-in replacement for 05_executor/index.ts:executeCode.
 * Pipeline: Parse → Lower (on demand) → Codegen (on demand) → Execute.
 */

import { type ExecOptions, type ExecResult } from "./executor/types.js";
import type {
  WorkspaceFile,
  NativeBridge,
} from "../numbl-core/workspace/index.js";
import { Runtime } from "./runtime/runtime.js";
import { wrapReturnValue } from "./executor/helpers.js";
import {
  RuntimeError,
  offsetToLine,
  offsetToColumn,
} from "../numbl-core/runtime/index.js";
import { ensureRuntimeValue } from "./runtime/runtimeHelpers.js";
import {
  registerDynamicIBuiltin,
  unregisterIBuiltin,
  inferJitType,
} from "./interpreter/builtins/types.js";
import { SemanticError } from "./lowering/errors.js";
import { JitCompiler } from "./executor/jitCompiler.js";
import { generateMainScriptCode } from "./codegen/generateMainScriptCode.js";
import { interpretCode } from "./interpretCode.js";

export { generateMainScriptCode as generateCode } from "./codegen/generateMainScriptCode.js";

export function executeCode(
  source: string,
  options: ExecOptions = {},
  workspaceFiles?: WorkspaceFile[],
  mainFileName: string = "script.m",
  searchPaths?: string[],
  nativeBridge?: NativeBridge
): ExecResult {
  if (options.interpret) {
    return interpretCode(
      source,
      options,
      workspaceFiles,
      mainFileName,
      searchPaths,
      nativeBridge
    );
  }

  const initialVariableNames = options.initialVariableValues
    ? Object.keys(options.initialVariableValues)
    : undefined;
  const codegenStart = performance.now();
  const {
    jsCode, // generated JavaScript for the main script body
    ctx, // lowering context (holds specialization cache, class/function registry)
    fileSources, // filename → source text, for error annotation and JIT codegen
    functionIndex, // upfront index of all available functions
    fileASTCache, // pre-parsed ASTs for all workspace files
    jsUserFunctions, // .js user functions loaded from workspace
    codegenBreakdown, // per-phase timing breakdown
  } = generateMainScriptCode(
    source,
    mainFileName,
    workspaceFiles,
    initialVariableNames,
    searchPaths,
    { noLineTracking: options.noLineTracking, nativeBridge }
  );
  const codegenTimeMs = performance.now() - codegenStart;

  if (options.log) {
    options.log("Generated JS:\n" + jsCode);
  }

  const rt = new Runtime(options, options.initialVariableValues);

  // Register .js user functions as IBuiltins (and on rt.builtins for codegen dispatch)
  for (const ib of jsUserFunctions) {
    registerDynamicIBuiltin(ib);
    rt.builtins[ib.name] = (nargout: number, args: unknown[]) => {
      const margs = args.map(a => ensureRuntimeValue(a));
      const argTypes = margs.map(inferJitType);
      const resolution = ib.resolve(argTypes, nargout);
      if (resolution) return resolution.apply(margs, nargout);
      throw new RuntimeError(
        `JS user function '${ib.name}' rejected arguments`
      );
    };
  }

  // Apply custom builtins (overrides defaults for this execution only)
  if (options.customBuiltins) {
    Object.assign(rt.builtins, options.customBuiltins);
  }

  // Wire up JIT compilation callbacks so the runtime can compile functions on demand
  const jit = new JitCompiler(
    ctx,
    fileSources,
    functionIndex,
    fileASTCache,
    rt,
    options.onJitCompile,
    options.noLineTracking
  );
  jit.install();

  // Wire up eval-with-local-vars callback
  rt.evalLocalCallback = (code, initialVars, onOutput) => {
    const evalResult = executeCode(code, {
      onOutput,
      displayResults: false,
      initialVariableValues: initialVars,
    });
    return {
      returnValue: evalResult.returnValue,
      variableValues: evalResult.variableValues,
    };
  };

  try {
    const syncCode = `${jsCode}`;
    const fn = new Function("$rt", syncCode);
    const execStart = performance.now();
    const r = fn(rt);
    const executionTimeMs = performance.now() - execStart;
    const result: ExecResult = {
      output: rt.outputLines,
      generatedJS: jsCode,
      plotInstructions: rt.plotInstructions,
      returnValue: wrapReturnValue(r),
      variableValues: rt.variableValues,
      holdState: rt.holdState,
    };
    const duc = rt.getDispatchUnknownCounts();
    if (Object.keys(duc).length > 0) {
      result.dispatchUnknownCounts = duc;
    }
    if (options.profile) {
      result.profileData = {
        codegenTimeMs,
        codegenBreakdown,
        executionTimeMs,
        jitCompileTimeMs: rt.getJitCompileTimeMs(),
        builtins: rt.getBuiltinProfile(),
        dispatches: rt.getDispatchProfile(),
      };
    }
    return result;
  } catch (e) {
    // Annotate runtime errors with file/line info
    const annotateError = (re: RuntimeError) => {
      if (re.span && re.line === null) {
        const src = fileSources.get(re.span.file);
        if (src) {
          re.file = re.span.file;
          re.line = offsetToLine(src, re.span.start);
          re.column = offsetToColumn(src, re.span.start);
        }
      }
      if (re.line === null && rt.$file && rt.$line > 0) {
        re.file = rt.$file;
        re.line = rt.$line;
      }
      if (re.line === null && rt.$line > 0) {
        re.line = rt.$line;
      }
      if (re.callStack === null && rt.$callStack.length > 0) {
        re.callStack = [...rt.$callStack];
      }
    };

    type ErrorWithDebugInfo = Error & {
      generatedJS?: string;
      jitFunctionCode?: Map<string, string>;
    };

    if (e instanceof RuntimeError) {
      annotateError(e);
      (e as ErrorWithDebugInfo).generatedJS = jsCode;
      (e as ErrorWithDebugInfo).jitFunctionCode = rt.jitFunctionCode;
      throw e;
    }
    const re = new RuntimeError(e instanceof Error ? e.message : String(e));
    if (e instanceof SemanticError && e.span) {
      re.span = e.span;
    }
    annotateError(re);
    (re as ErrorWithDebugInfo).generatedJS = jsCode;
    (re as ErrorWithDebugInfo).jitFunctionCode = rt.jitFunctionCode;
    throw re;
  } finally {
    // Unregister .js user function IBuiltins to avoid polluting the global registry
    for (const ib of jsUserFunctions) {
      unregisterIBuiltin(ib.name);
    }
  }
}
