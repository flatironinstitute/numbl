/**
 * Loader for .js user functions.
 *
 * Evaluates .js files that define branches via register({ check, apply }).
 */

import type { BuiltinFn, BuiltinFnBranch } from "./builtins/registry.js";
import { type ItemType, IType } from "./lowering/itemTypes.js";
import type { WorkspaceFile } from "./workspace/index.js";
import { RTV, RuntimeError } from "./runtime/index.js";
import { FloatXArray } from "./runtime/types.js";

/**
 * Derive a function name from a .js workspace file path.
 * E.g., "myadd.js" → "myadd", "/path/to/myadd.js" → "myadd"
 */
function funcNameFromFile(fileName: string): string {
  const base = fileName.split("/").pop()!;
  return base.replace(/\.js$/, "");
}

const defaultCheck = (_argTypes: ItemType[], nargout: number) => ({
  outputTypes: Array(Math.max(nargout, 1)).fill({
    kind: "Unknown",
  } as ItemType),
});

/**
 * Build a map of base name → Uint8Array from .wasm workspace files.
 * E.g., "/path/to/myfunc.wasm" → "myfunc" → Uint8Array
 */
function buildWasmMap(wasmFiles: WorkspaceFile[]): Map<string, Uint8Array> {
  const map = new Map<string, Uint8Array>();
  for (const f of wasmFiles) {
    if (!f.data) continue;
    const base = f.name
      .split("/")
      .pop()!
      .replace(/\.wasm$/, "");
    map.set(base, f.data);
  }
  return map;
}

/**
 * Load .js user function files and return them as a map of function name → BuiltinFn.
 * If wasmFiles are provided, matching .wasm modules are compiled and exposed as `wasm`
 * in the .js function's execution context.
 */
export function loadJsUserFunctions(
  jsFiles: WorkspaceFile[],
  wasmFiles?: WorkspaceFile[]
): Map<string, BuiltinFn> {
  const result = new Map<string, BuiltinFn>();
  const wasmMap = wasmFiles ? buildWasmMap(wasmFiles) : new Map();

  for (const file of jsFiles) {
    const funcName = funcNameFromFile(file.name);
    try {
      const branches: BuiltinFnBranch[] = [];

      const registerFn = (branch: {
        check?: BuiltinFnBranch["check"];
        apply: BuiltinFnBranch["apply"];
      }) => {
        if (typeof branch.apply !== "function") {
          throw new Error("register(): branch must have an apply function");
        }
        branches.push({
          check: branch.check ?? defaultCheck,
          apply: branch.apply,
        });
      };

      // Compile matching .wasm module if available
      let wasmInstance: WebAssembly.Instance | null = null;
      const wasmData = wasmMap.get(funcName);
      if (wasmData) {
        const wasmModule = new WebAssembly.Module(wasmData);
        // Check if module needs WASI imports (complex modules built with Emscripten)
        const moduleImports = WebAssembly.Module.imports(wasmModule);
        const importObject: WebAssembly.Imports = {};
        // Provide stubs for all imported modules (WASI + Emscripten env)
        const neededModules = new Set(moduleImports.map(i => i.module));
        if (neededModules.has("wasi_snapshot_preview1")) {
          importObject.wasi_snapshot_preview1 = {
            fd_write: () => 0,
            fd_read: () => 0,
            fd_close: () => 0,
            fd_seek: () => 0,
            fd_fdstat_get: () => 0,
            proc_exit: () => {},
            environ_sizes_get: () => 0,
            environ_get: () => 0,
            clock_time_get: () => 0,
            args_sizes_get: () => 0,
            args_get: () => 0,
          };
        }
        if (neededModules.has("env")) {
          importObject.env = {
            emscripten_notify_memory_growth: () => {},
          };
        }
        wasmInstance = new WebAssembly.Instance(wasmModule, importObject);
      }

      const factory = new Function(
        "RTV",
        "RuntimeError",
        "FloatXArray",
        "IType",
        "register",
        "wasm",
        file.source
      );
      factory(RTV, RuntimeError, FloatXArray, IType, registerFn, wasmInstance);

      if (branches.length === 0) {
        throw new Error(
          `JS user function '${funcName}' (${file.name}) must call register() at least once`
        );
      }

      result.set(funcName, branches);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      throw new Error(
        `Error loading JS user function '${funcName}' (${file.name}): ${msg}`
      );
    }
  }

  return result;
}
