/**
 * Loader for .js user functions.
 *
 * Evaluates .js files that define branches via register({ check, apply }).
 * Supports optional WASM and native shared library bindings via directives:
 *   // wasm: <name>
 *   // native: <name>
 */

import type { BuiltinFn, BuiltinFnBranch } from "./builtins/registry.js";
import { type ItemType, IType } from "./lowering/itemTypes.js";
import type { WorkspaceFile, NativeBridge } from "./workspace/index.js";
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

/** Parsed directives from the top of a .js user function file. */
interface JsDirectives {
  wasm?: string;
  native?: string;
}

/**
 * Parse YAML-compatible directives from consecutive comment lines at the top
 * of a .js source file. Supported keys: wasm, native.
 *
 * Example:
 *   // wasm: wadd
 *   // native: wadd
 */
function parseDirectives(source: string): JsDirectives {
  const directives: JsDirectives = {};
  const lines = source.split("\n");
  for (const line of lines) {
    const match = line.match(/^\s*\/\/\s*(\w+):\s*(\S+)/);
    if (!match) break; // stop at first non-directive line
    const key = match[1];
    const value = match[2];
    if (key === "wasm") directives.wasm = value;
    else if (key === "native") directives.native = value;
  }
  return directives;
}

/**
 * Resolve a platform-appropriate shared library filename from a base name.
 * E.g., "wadd" → "wadd.so" (Linux), "wadd.dll" (Windows), "wadd.dylib" (macOS)
 */
function nativeLibFilename(baseName: string): string {
  switch (process.platform) {
    case "win32":
      return `${baseName}.dll`;
    case "darwin":
      return `${baseName}.dylib`;
    default:
      return `${baseName}.so`;
  }
}

/**
 * Compile and instantiate a WASM module from raw bytes, providing
 * WASI and Emscripten stubs as needed.
 */
function instantiateWasm(wasmData: Uint8Array): WebAssembly.Instance {
  const wasmModule = new WebAssembly.Module(wasmData as BufferSource);
  const moduleImports = WebAssembly.Module.imports(wasmModule);
  const importObject: WebAssembly.Imports = {};
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
  return new WebAssembly.Instance(wasmModule, importObject);
}

/**
 * Resolve WASM and native bindings for a .js file based on its directives.
 */
function resolveBindings(
  file: WorkspaceFile,
  directives: JsDirectives,
  getWasmInstance: (name: string) => WebAssembly.Instance | undefined,
  nativeBridge?: NativeBridge
): { wasmInstance: WebAssembly.Instance | undefined; nativeLib: unknown } {
  const wasmInstance = directives.wasm
    ? getWasmInstance(directives.wasm)
    : undefined;

  let nativeLib: unknown;
  if (directives.native && nativeBridge) {
    const libFile = nativeLibFilename(directives.native);
    const dir = file.name.substring(0, file.name.lastIndexOf("/") + 1);
    const libPath = dir + libFile;
    try {
      nativeLib = nativeBridge.load(libPath);
    } catch {
      // Native library not found or failed to load — leave undefined
    }
  }
  return { wasmInstance, nativeLib };
}

/** Sentinel value indicating a library is currently being loaded. */
const LOADING = Symbol("loading");

/**
 * Load .js user function files and return them as a map of function name → BuiltinFn.
 *
 * A .js file can specify bindings via directives at the top of the file:
 *   // wasm: <name>    — load a WASM module (looked up by name in wasmFiles)
 *   // native: <name>  — load a native shared library (resolved relative to .js file)
 *
 * The `wasm` and `native` parameters are passed to the .js function context.
 * They are `undefined` when the corresponding directive is absent or the binary
 * is not found.
 *
 * Files whose basename starts with `_` are library files. They must not call
 * register() and instead export values via `return`. Other .js files can import
 * them with `importJS("_name")`.
 */
export function loadJsUserFunctions(
  jsFiles: WorkspaceFile[],
  wasmFiles?: WorkspaceFile[],
  nativeBridge?: NativeBridge
): Map<string, BuiltinFn> {
  const result = new Map<string, BuiltinFn>();
  const wasmMap = wasmFiles ? buildWasmMap(wasmFiles) : new Map();
  // Cache compiled WASM instances so multiple .js files can share one module
  const wasmInstanceCache = new Map<string, WebAssembly.Instance>();

  function getWasmInstance(name: string): WebAssembly.Instance | undefined {
    const cached = wasmInstanceCache.get(name);
    if (cached) return cached;
    const data = wasmMap.get(name);
    if (!data) return undefined;
    const instance = instantiateWasm(data);
    wasmInstanceCache.set(name, instance);
    return instance;
  }

  // Separate library files (_-prefixed) from function files
  const libraryFiles = new Map<string, WorkspaceFile>();
  const functionFiles: WorkspaceFile[] = [];
  for (const file of jsFiles) {
    const base = file.name.split("/").pop()!;
    if (base.startsWith("_")) {
      const libName = base.replace(/\.js$/, "");
      libraryFiles.set(libName, file);
    } else {
      functionFiles.push(file);
    }
  }

  // Library cache: LOADING sentinel for circular detection, or cached exports
  const libCache = new Map<string, typeof LOADING | unknown>();

  function importJS(name: string): unknown {
    const cached = libCache.get(name);
    if (cached === LOADING) {
      throw new RuntimeError(`Circular dependency detected: ${name}.js`);
    }
    if (libCache.has(name)) return cached;

    const libFile = libraryFiles.get(name);
    if (!libFile) {
      throw new RuntimeError(
        `importJS: library '${name}.js' not found in workspace`
      );
    }

    libCache.set(name, LOADING);

    const directives = parseDirectives(libFile.source);
    const { wasmInstance, nativeLib } = resolveBindings(
      libFile,
      directives,
      getWasmInstance,
      nativeBridge
    );

    const dummyRegister = () => {
      throw new RuntimeError(
        `Library file '${name}.js' must not call register(). ` +
          `Use return {...} to export values.`
      );
    };

    const factory = new Function(
      "RTV",
      "RuntimeError",
      "FloatXArray",
      "IType",
      "register",
      "wasm",
      "native",
      "importJS",
      libFile.source
    );
    const exports = factory(
      RTV,
      RuntimeError,
      FloatXArray,
      IType,
      dummyRegister,
      wasmInstance,
      nativeLib,
      importJS
    );

    libCache.set(name, exports);
    return exports;
  }

  for (const file of functionFiles) {
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

      // Parse directives from the top of the .js file
      const directives = parseDirectives(file.source);

      const { wasmInstance, nativeLib } = resolveBindings(
        file,
        directives,
        getWasmInstance,
        nativeBridge
      );

      const factory = new Function(
        "RTV",
        "RuntimeError",
        "FloatXArray",
        "IType",
        "register",
        "wasm",
        "native",
        "importJS",
        file.source
      );
      factory(
        RTV,
        RuntimeError,
        FloatXArray,
        IType,
        registerFn,
        wasmInstance,
        nativeLib,
        importJS
      );

      if (branches.length === 0) {
        throw new Error(
          `JS user function '${funcName}' (${file.name}) must call register() at least once`
        );
      }

      // Build binding status message (only if directives were present)
      if (directives.native || directives.wasm) {
        const parts: string[] = [];
        if (directives.native) {
          parts.push(nativeLib ? "native: loaded" : "native: not found");
        }
        if (directives.wasm) {
          parts.push(wasmInstance ? "wasm: loaded" : "wasm: not found");
        }
        const statusMsg = `${funcName}: ${parts.join(", ")}`;
        let logged = false;
        for (const branch of branches) {
          const origApply = branch.apply;
          branch.apply = (...applyArgs) => {
            if (!logged) {
              logged = true;
              console.log(statusMsg);
            }
            return origApply(...applyArgs);
          };
        }
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
