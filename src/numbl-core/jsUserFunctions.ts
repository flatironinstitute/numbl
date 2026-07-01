/**
 * Loader for .numbl.js user functions.
 *
 * Evaluates .numbl.js files that define IBuiltins via
 *   register({ resolve }).
 * Supports optional WASM and native shared library bindings via directives:
 *   // wasm: <name>
 *   // native: <name>
 *
 * The `.numbl.js` extension distinguishes numbl user functions from
 * unrelated `.js` files (config files, source maps, etc.) so that they can
 * be auto-discovered from the current working directory like `.m` files.
 */

import type { WorkspaceFile, NativeBridge } from "./workspace/index.js";
import { RTV, RuntimeError, toNumber } from "./runtime/index.js";
import { getCurrentRuntime } from "./runtime/refcount.js";
import type { IBuiltin } from "./interpreter/builtins/types.js";

/** Minimal runtime surface needed to invoke a function handle from JS. */
interface HandleCaller {
  index(base: unknown, indices: unknown[], nargout?: number): unknown;
}

/**
 * Invoke a numbl function handle from inside a `.numbl.js` apply.
 *
 * `handle` is a RuntimeFunction — the value a `function_handle` argument
 * arrives as. `args` are runtime values or raw JS scalars (raw numbers are
 * accepted, matching arrayfun's call path). Returns the handle's result: a
 * runtime value for `nargout <= 1`, or an array of them when `nargout > 1`.
 *
 * Injected into every `.numbl.js` file as the global `callHandle`. It binds
 * to the active runtime lazily via `getCurrentRuntime()`, so the single
 * shared function reference works regardless of which runtime is executing.
 */
function callHandle(handle: unknown, args: unknown[], nargout = 1): unknown {
  const rt = getCurrentRuntime() as HandleCaller | null;
  if (!rt) {
    throw new RuntimeError("callHandle: no active runtime to invoke handle");
  }
  return rt.index(handle, args, nargout);
}

/**
 * Per-WASM-instance registry that exposes numbl function handles to WASM as
 * integer ids. A `.numbl.js` apply registers a JS thunk (closing over a
 * handle + `callHandle`) with `add()`, passes the returned id into a WASM
 * export, and removes it once the export returns. Inside WASM the handle is
 * invoked by calling the host-provided `env.numbl_cb_d(id, x)` import.
 */
export interface WasmCallbackRegistry {
  /** Register a callback thunk; returns its integer id. */
  add(fn: (...args: number[]) => number): number;
  /** Drop a previously registered callback. */
  remove(id: number): void;
}

/** A `WebAssembly.Instance` augmented with the callback registry. */
type WasmInstanceWithCallbacks = WebAssembly.Instance & {
  callbacks: WasmCallbackRegistry;
};

/** A loaded JS user function ready for registration in the workspace. */
export interface LoadedJsUserFunction {
  name: string;
  fileName: string;
  builtin: IBuiltin;
}

/**
 * Derive a function name from a .numbl.js workspace file path.
 * Library files (basename starts with `_`) strip only `.js`; function files
 * strip the full `.numbl.js` suffix.
 *   "myadd.numbl.js"            → "myadd"
 *   "/path/to/myadd.numbl.js"   → "myadd"
 *   "_helpers.numbl.js"         → "_helpers"
 */
function funcNameFromFile(fileName: string): string {
  const base = fileName.split("/").pop()!;
  return base.replace(/\.numbl\.js$/, "");
}

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

/** Returns true if the file is a numbl JS user function file (`*.numbl.js`). */
export function isNumblJsFile(fileName: string): boolean {
  return fileName.endsWith(".numbl.js");
}

/** Returns true if the file is an mtoc2-only user function file (`*.mtoc2.js`).
 *  Numbl recognizes the extension for workspace discovery + function-name
 *  resolution, but never executes the body — mtoc2's loader does that. */
export function isMtoc2JsFile(fileName: string): boolean {
  return fileName.endsWith(".mtoc2.js");
}

/** Derive a function name from a `.mtoc2.js` workspace file path.
 *    "myadd.mtoc2.js"            → "myadd"
 *    "/path/to/myadd.mtoc2.js"   → "myadd" */
export function funcNameFromMtoc2JsFile(fileName: string): string {
  const base = fileName.split("/").pop()!;
  return base.replace(/\.mtoc2\.js$/, "");
}

/**
 * Resolve a platform-appropriate shared library filename from a base name.
 * E.g., "wadd" → "wadd.so" (Linux), "wadd.dll" (Windows), "wadd.dylib" (macOS)
 */
function nativeLibFilename(baseName: string): string {
  const platform = typeof process !== "undefined" ? process.platform : "linux";
  switch (platform) {
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
 *
 * The returned instance carries a `callbacks` registry (see
 * {@link WasmCallbackRegistry}) and an `env.numbl_cb_d(id, x) -> double`
 * import that routes a WASM-side callback to the registered handle. A module
 * only resolves imports it actually declares, so this is always provided and
 * costs nothing for modules that never call back.
 */
function instantiateWasm(wasmData: Uint8Array): WasmInstanceWithCallbacks {
  const wasmModule = new WebAssembly.Module(wasmData as BufferSource);
  const moduleImports = WebAssembly.Module.imports(wasmModule);
  const importObject: WebAssembly.Imports = {};
  const neededModules = new Set(moduleImports.map(i => i.module));

  // Per-instance callback registry: maps an integer id → JS thunk so WASM
  // can invoke a numbl function handle via the `numbl_cb_d` import below.
  const callbacks = new Map<number, (...args: number[]) => number>();
  let nextCbId = 1;

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
      // Scalar callback: WASM calls back into a registered handle with one
      // f64 and receives an f64. Exceptions thrown here (incl. a missing id)
      // propagate out through the WASM call into the apply.
      numbl_cb_d: (id: number, x: number): number => {
        const fn = callbacks.get(id);
        if (!fn) {
          throw new RuntimeError(
            `numbl_cb_d: no callback registered for id ${id}`
          );
        }
        return fn(x);
      },
    };
  }

  const instance = new WebAssembly.Instance(
    wasmModule,
    importObject
  ) as WasmInstanceWithCallbacks;
  instance.callbacks = {
    add(fn) {
      const id = nextCbId++;
      callbacks.set(id, fn);
      return id;
    },
    remove(id) {
      callbacks.delete(id);
    },
  };
  return instance;
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
 * Load .numbl.js user function files and return them as LoadedJsUserFunction
 * records. Each loaded record carries the function name, source file name,
 * and the IBuiltin object built from the file's `register()` call.
 *
 * Each .numbl.js file calls register({ resolve }) to define an
 * IBuiltin. resolve(argTypes, nargout) returns { outputTypes, apply } or null.
 *
 * A .numbl.js file can specify bindings via directives at the top of the file:
 *   // wasm: <name>    — load a WASM module (looked up by name in wasmFiles)
 *   // native: <name>  — load a native shared library (resolved relative to file)
 *
 * Files whose basename starts with `_` are library files. They must not call
 * register() and instead export values via `return`. Other .numbl.js files
 * can import them with `importJS("_name")`.
 */
export function loadJsUserFunctions(
  jsFiles: WorkspaceFile[],
  wasmFiles?: WorkspaceFile[],
  nativeBridge?: NativeBridge
): LoadedJsUserFunction[] {
  const result: LoadedJsUserFunction[] = [];
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
      const libName = base.replace(/\.numbl\.js$/, "");
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
      throw new RuntimeError(`Circular dependency detected: ${name}.numbl.js`);
    }
    if (libCache.has(name)) return cached;

    const libFile = libraryFiles.get(name);
    if (!libFile) {
      throw new RuntimeError(
        `importJS: library '${name}.numbl.js' not found in workspace`
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
        `Library file '${name}.numbl.js' must not call register(). ` +
          `Use return {...} to export values.`
      );
    };

    const factory = new Function(
      "RTV",
      "RuntimeError",
      "Float64Array",
      "register",
      "wasm",
      "native",
      "importJS",
      "callHandle",
      "toNumber",
      libFile.source
    );
    const exports = factory(
      RTV,
      RuntimeError,
      Float64Array,
      dummyRegister,
      wasmInstance,
      nativeLib,
      importJS,
      callHandle,
      toNumber
    );

    libCache.set(name, exports);
    return exports;
  }

  for (const file of functionFiles) {
    const funcName = funcNameFromFile(file.name);
    try {
      let builtin: IBuiltin | null = null;

      const registerFn = (spec: { resolve: IBuiltin["resolve"] }) => {
        if (typeof spec.resolve !== "function") {
          throw new Error("register(): spec must have a resolve function");
        }
        if (builtin) {
          throw new Error(
            "register() called more than once — only one registration per .numbl.js file"
          );
        }
        builtin = {
          name: funcName,
          resolve: spec.resolve as IBuiltin["resolve"],
        };
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
        "Float64Array",
        "register",
        "wasm",
        "native",
        "importJS",
        "callHandle",
        "toNumber",
        file.source
      );
      factory(
        RTV,
        RuntimeError,
        Float64Array,
        registerFn,
        wasmInstance,
        nativeLib,
        importJS,
        callHandle,
        toNumber
      );

      if (!builtin) {
        throw new Error(
          `JS user function '${funcName}' (${file.name}) must call register() once`
        );
      }

      // Wrap resolve with binding status logging (only if directives were present)
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
        const origResolve = (builtin as IBuiltin).resolve;
        (builtin as IBuiltin).resolve = (argTypes, nargout) => {
          const resolution = origResolve(argTypes, nargout);
          if (!resolution) return null;
          const origApply = resolution.apply;
          return {
            outputTypes: resolution.outputTypes,
            apply: (args, n) => {
              if (!logged) {
                logged = true;
                console.log(statusMsg);
              }
              return origApply(args, n);
            },
          };
        };
      }

      result.push({ name: funcName, fileName: file.name, builtin });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      throw new Error(
        `Error loading JS user function '${funcName}' (${file.name}): ${msg}`
      );
    }
  }

  return result;
}
