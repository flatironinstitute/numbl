/**
 * Type definitions for workspace multi-file support.
 */

export interface WorkspaceFile {
  name: string; // e.g., "test1.m"
  source: string;
  data?: Uint8Array; // binary content for .wasm files
}

/**
 * Bridge for loading native shared libraries (.so/.dll/.dylib).
 * Implemented outside numbl-core (e.g. via koffi in the CLI) since
 * native FFI is not available in browser environments.
 *
 * `koffi`, when present, exposes the root koffi module so the
 * mtoc2-backed C-JIT executors can call its struct / alloc / encode /
 * decode helpers. Browser deployments leave `koffi` undefined; the
 * C-JIT executors decline (numbl falls back to JS-JIT).
 */
export interface NativeBridge {
  load(libraryPath: string): unknown;
  /** koffi root module — `require("koffi")`'s exports. Only the
   *  C-JIT path consumes it. Optional so non-Node hosts can omit. */
  koffi?: unknown;
}
