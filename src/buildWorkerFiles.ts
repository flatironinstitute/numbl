import type { FileEntry } from "./hooks/projectFilesState.js";
import type { WorkspaceFile } from "./numbl-core/workspace/index.js";
import type { WorkerVfsFile } from "./numblWorkerProtocol.js";

export interface WorkerFileSources {
  /** From useSystemFiles. */
  getSystemVfsFiles: () => Promise<WorkerVfsFile[]>;
  getSystemWorkspaceFiles: () => Promise<WorkspaceFile[]>;
  /** Project files and their content loader; omit for system files only. */
  projectFiles?: FileEntry[];
  loadAllContents?: () => Promise<Map<string, Uint8Array>>;
}

/**
 * Files to send numbl-worker: every project and system file goes into the
 * VFS; .m files (minus `excludeFileId`, e.g. the script being run) also go
 * as text workspace files. Binary blobs (.wasm) and .numbl.js mex-like files
 * reach the worker only via the VFS and are discovered through scanDirectory
 * once their directory is on the search path (see
 * useSystemFiles.getSystemWorkspaceFiles).
 */
export async function buildWorkerFiles(
  sources: WorkerFileSources,
  excludeFileId?: string
): Promise<{
  vfsFiles: WorkerVfsFile[];
  workspaceFiles: WorkspaceFile[];
  projectContents: Map<string, Uint8Array>;
}> {
  const projectFiles = sources.projectFiles ?? [];
  const [projectContents, systemVfs, systemWs] = await Promise.all([
    sources.loadAllContents?.() ?? new Map<string, Uint8Array>(),
    sources.getSystemVfsFiles(),
    sources.getSystemWorkspaceFiles(),
  ]);
  const content = (f: FileEntry) =>
    projectContents.get(f.id) ?? new Uint8Array(0);

  const vfsFiles = [
    ...projectFiles.map(f => ({ path: f.name, content: content(f) })),
    ...systemVfs,
  ];

  const decoder = new TextDecoder("utf-8");
  const workspaceFiles = [
    ...projectFiles
      .filter(f => f.id !== excludeFileId && f.name.endsWith(".m"))
      .map(f => ({ name: f.name, source: decoder.decode(content(f)) })),
    ...systemWs,
  ];

  return { vfsFiles, workspaceFiles, projectContents };
}
