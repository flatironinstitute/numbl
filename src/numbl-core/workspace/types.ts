/**
 * Type definitions for workspace multi-file support.
 */

export interface WorkspaceFile {
  name: string; // e.g., "test1.m"
  source: string;
  data?: Uint8Array; // binary content for .wasm files
}
