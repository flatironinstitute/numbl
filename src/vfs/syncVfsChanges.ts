/**
 * Syncs VFS changes from a worker execution back to IndexedDB.
 */

import { db } from "../db/schema.js";
import { ensureSystemProject, SYSTEM_PROJECT_NAME } from "../db/operations.js";
import type { VfsChanges } from "./VirtualFileSystem.js";
import type { WorkspaceFile } from "../hooks/useProjectFiles.js";

const SYSTEM_PREFIX = "/system/";

/** Strip the /project/ prefix from VFS paths to get project-relative paths. */
function toProjectPath(vfsPath: string): string {
  if (vfsPath.startsWith("/project/")) return vfsPath.slice("/project/".length);
  if (vfsPath.startsWith("/")) return vfsPath.slice(1);
  return vfsPath;
}

/** Strip the /system/ prefix from VFS paths to get system-relative paths. */
function toSystemPath(vfsPath: string): string {
  if (vfsPath.startsWith(SYSTEM_PREFIX))
    return vfsPath.slice(SYSTEM_PREFIX.length);
  return vfsPath;
}

function isSystemPath(vfsPath: string): boolean {
  return vfsPath.startsWith(SYSTEM_PREFIX) || vfsPath === "/system";
}

/** Split VFS changes into project changes and system changes. */
function splitChanges(changes: VfsChanges): {
  projectChanges: VfsChanges;
  systemChanges: VfsChanges;
} {
  const projectChanges: VfsChanges = { created: [], modified: [], deleted: [] };
  const systemChanges: VfsChanges = { created: [], modified: [], deleted: [] };

  for (const entry of changes.created) {
    (isSystemPath(entry.path) ? systemChanges : projectChanges).created.push(
      entry
    );
  }
  for (const entry of changes.modified) {
    (isSystemPath(entry.path) ? systemChanges : projectChanges).modified.push(
      entry
    );
  }
  for (const path of changes.deleted) {
    (isSystemPath(path) ? systemChanges : projectChanges).deleted.push(path);
  }

  return { projectChanges, systemChanges };
}

export interface VfsSyncResult {
  addedFiles: WorkspaceFile[];
  modifiedFiles: { path: string; data: Uint8Array }[];
  deletedPaths: string[];
}

async function syncChangesToDb(
  projectName: string,
  changes: VfsChanges,
  toRelativePath: (vfsPath: string) => string
): Promise<VfsSyncResult | null> {
  const addedFiles: WorkspaceFile[] = [];
  const modifiedFiles: { path: string; data: Uint8Array }[] = [];
  const deletedPaths: string[] = [];

  await db.transaction(
    "rw",
    db.files,
    db.fileContents,
    db.projects,
    async () => {
      for (const { path, content } of changes.created) {
        const relPath = toRelativePath(path);
        const now = Date.now();
        const id = crypto.randomUUID();
        await db.files.add({
          id,
          projectName,
          path: relPath,
          createdAt: now,
          updatedAt: now,
        });
        await db.fileContents.add({ id, data: content });
        addedFiles.push({ id, name: relPath });
      }

      for (const { path, content } of changes.modified) {
        const relPath = toRelativePath(path);
        const existing = await db.files
          .where("[projectName+path]")
          .equals([projectName, relPath])
          .first();
        if (existing) {
          await db.fileContents.put({ id: existing.id, data: content });
          await db.files.update(existing.id, { updatedAt: Date.now() });
          modifiedFiles.push({ path: relPath, data: content });
        }
      }

      for (const path of changes.deleted) {
        const relPath = toRelativePath(path);
        const existing = await db.files
          .where("[projectName+path]")
          .equals([projectName, relPath])
          .first();
        if (existing) {
          await db.files.delete(existing.id);
          await db.fileContents.delete(existing.id);
          deletedPaths.push(relPath);
        }
      }

      if (
        addedFiles.length > 0 ||
        modifiedFiles.length > 0 ||
        deletedPaths.length > 0
      ) {
        await db.projects.update(projectName, { updatedAt: Date.now() });
      }
    }
  );

  if (
    addedFiles.length === 0 &&
    modifiedFiles.length === 0 &&
    deletedPaths.length === 0
  ) {
    return null;
  }
  return { addedFiles, modifiedFiles, deletedPaths };
}

export interface VfsSyncAllResult {
  projectResult: VfsSyncResult | null;
  systemResult: VfsSyncResult | null;
}

/**
 * Sync only the /system/ portion of VFS changes to the __system__ project in IndexedDB.
 * Returns true if any system files were changed.
 */
export async function syncSystemVfsChanges(
  changes: VfsChanges
): Promise<boolean> {
  const { systemChanges } = splitChanges(changes);
  if (
    systemChanges.created.length === 0 &&
    systemChanges.modified.length === 0 &&
    systemChanges.deleted.length === 0
  ) {
    return false;
  }
  await ensureSystemProject();
  const result = await syncChangesToDb(
    SYSTEM_PROJECT_NAME,
    systemChanges,
    toSystemPath
  );
  return result !== null;
}

export async function syncVfsChangesToProject(
  projectName: string,
  changes: VfsChanges
): Promise<VfsSyncAllResult> {
  const { projectChanges, systemChanges } = splitChanges(changes);

  const projectResult = await syncChangesToDb(
    projectName,
    projectChanges,
    toProjectPath
  );

  let systemResult: VfsSyncResult | null = null;
  if (
    systemChanges.created.length > 0 ||
    systemChanges.modified.length > 0 ||
    systemChanges.deleted.length > 0
  ) {
    await ensureSystemProject();
    systemResult = await syncChangesToDb(
      SYSTEM_PROJECT_NAME,
      systemChanges,
      toSystemPath
    );
  }

  return { projectResult, systemResult };
}
