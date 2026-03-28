/**
 * Syncs VFS changes from a worker execution back to IndexedDB.
 */

import { db } from "../db/schema.js";
import { ensureHomeProject } from "../db/operations.js";
import type { VfsChanges } from "./VirtualFileSystem.js";
import type { WorkspaceFile } from "../hooks/useProjectFiles.js";

const HOME_PREFIX = "/home/";
const HOME_PROJECT_NAME = "__home__";

/** Strip the /project/ prefix from VFS paths to get project-relative paths. */
function toProjectPath(vfsPath: string): string {
  if (vfsPath.startsWith("/project/")) return vfsPath.slice("/project/".length);
  if (vfsPath.startsWith("/")) return vfsPath.slice(1);
  return vfsPath;
}

/** Strip the /home/ prefix from VFS paths to get home-relative paths. */
function toHomePath(vfsPath: string): string {
  if (vfsPath.startsWith(HOME_PREFIX)) return vfsPath.slice(HOME_PREFIX.length);
  return vfsPath;
}

function isHomePath(vfsPath: string): boolean {
  return vfsPath.startsWith(HOME_PREFIX) || vfsPath === "/home";
}

/** Split VFS changes into project changes and home changes. */
function splitChanges(changes: VfsChanges): {
  projectChanges: VfsChanges;
  homeChanges: VfsChanges;
} {
  const projectChanges: VfsChanges = { created: [], modified: [], deleted: [] };
  const homeChanges: VfsChanges = { created: [], modified: [], deleted: [] };

  for (const entry of changes.created) {
    (isHomePath(entry.path) ? homeChanges : projectChanges).created.push(entry);
  }
  for (const entry of changes.modified) {
    (isHomePath(entry.path) ? homeChanges : projectChanges).modified.push(
      entry
    );
  }
  for (const path of changes.deleted) {
    (isHomePath(path) ? homeChanges : projectChanges).deleted.push(path);
  }

  return { projectChanges, homeChanges };
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

  await db.transaction("rw", db.files, db.projects, async () => {
    for (const { path, content } of changes.created) {
      const relPath = toRelativePath(path);
      const now = Date.now();
      const id = crypto.randomUUID();
      await db.files.add({
        id,
        projectName,
        path: relPath,
        data: content,
        createdAt: now,
        updatedAt: now,
      });
      addedFiles.push({ id, name: relPath, data: content });
    }

    for (const { path, content } of changes.modified) {
      const relPath = toRelativePath(path);
      const existing = await db.files
        .where("[projectName+path]")
        .equals([projectName, relPath])
        .first();
      if (existing) {
        await db.files.update(existing.id, {
          data: content,
          updatedAt: Date.now(),
        });
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

export interface VfsSyncAllResult {
  projectResult: VfsSyncResult | null;
  homeResult: VfsSyncResult | null;
}

export async function syncVfsChangesToProject(
  projectName: string,
  changes: VfsChanges
): Promise<VfsSyncAllResult> {
  const { projectChanges, homeChanges } = splitChanges(changes);

  const projectResult = await syncChangesToDb(
    projectName,
    projectChanges,
    toProjectPath
  );

  let homeResult: VfsSyncResult | null = null;
  if (
    homeChanges.created.length > 0 ||
    homeChanges.modified.length > 0 ||
    homeChanges.deleted.length > 0
  ) {
    await ensureHomeProject();
    homeResult = await syncChangesToDb(
      HOME_PROJECT_NAME,
      homeChanges,
      toHomePath
    );
  }

  return { projectResult, homeResult };
}
