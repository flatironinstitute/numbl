/**
 * Syncs VFS changes from a worker execution back to IndexedDB.
 */

import { db } from "../db/schema.js";
import type { VfsChanges } from "./VirtualFileSystem.js";
import type { WorkspaceFile } from "../hooks/useProjectFiles.js";

/** Strip the /project/ prefix from VFS paths to get project-relative paths. */
function toProjectPath(vfsPath: string): string {
  if (vfsPath.startsWith("/project/")) return vfsPath.slice("/project/".length);
  if (vfsPath.startsWith("/")) return vfsPath.slice(1);
  return vfsPath;
}

export interface VfsSyncResult {
  addedFiles: WorkspaceFile[];
  modifiedFiles: { path: string; data: Uint8Array }[];
  deletedPaths: string[];
}

export async function syncVfsChangesToProject(
  projectName: string,
  changes: VfsChanges
): Promise<VfsSyncResult | null> {
  const addedFiles: WorkspaceFile[] = [];
  const modifiedFiles: { path: string; data: Uint8Array }[] = [];
  const deletedPaths: string[] = [];

  await db.transaction("rw", db.files, db.projects, async () => {
    // Handle created files
    for (const { path, content } of changes.created) {
      const projectPath = toProjectPath(path);
      const now = Date.now();
      const id = crypto.randomUUID();
      await db.files.add({
        id,
        projectName,
        path: projectPath,
        data: content,
        createdAt: now,
        updatedAt: now,
      });
      addedFiles.push({ id, name: projectPath, data: content });
    }

    // Handle modified files
    for (const { path, content } of changes.modified) {
      const projectPath = toProjectPath(path);
      const existing = await db.files
        .where("[projectName+path]")
        .equals([projectName, projectPath])
        .first();
      if (existing) {
        await db.files.update(existing.id, {
          data: content,
          updatedAt: Date.now(),
        });
        modifiedFiles.push({ path: projectPath, data: content });
      }
    }

    // Handle deleted files
    for (const path of changes.deleted) {
      const projectPath = toProjectPath(path);
      const existing = await db.files
        .where("[projectName+path]")
        .equals([projectName, projectPath])
        .first();
      if (existing) {
        await db.files.delete(existing.id);
        deletedPaths.push(projectPath);
      }
    }

    // Update project timestamp
    if (
      addedFiles.length > 0 ||
      modifiedFiles.length > 0 ||
      deletedPaths.length > 0
    ) {
      await db.projects.update(projectName, { updatedAt: Date.now() });
    }
  });

  if (
    addedFiles.length === 0 &&
    modifiedFiles.length === 0 &&
    deletedPaths.length === 0
  ) {
    return null;
  }
  return { addedFiles, modifiedFiles, deletedPaths };
}
