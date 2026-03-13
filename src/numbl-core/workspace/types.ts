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
 */
export interface NativeBridge {
  load(libraryPath: string): unknown;
}
