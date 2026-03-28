import { useState, useEffect, useCallback, useMemo } from "react";
import { db } from "../db/schema";
import { ensureHomeProject, getHomeFiles } from "../db/operations";
import type { WorkspaceFile } from "./useProjectFiles";

const HOME_PROJECT_NAME = "__home__";
const HOME_PREFIX = "~/";

export function useHomeFiles() {
  const [homeFiles, setHomeFiles] = useState<WorkspaceFile[]>([]);

  const loadHomeFiles = useCallback(async () => {
    const files = await getHomeFiles();
    setHomeFiles(
      files.map(f => ({
        id: f.id,
        name: HOME_PREFIX + f.path,
        data: f.data,
      }))
    );
  }, []);

  useEffect(() => {
    // Initial async load from IndexedDB — setState is called after await, not synchronously
    // eslint-disable-next-line react-hooks/set-state-in-effect
    loadHomeFiles();
  }, [loadHomeFiles]);

  const updateFileContent = useCallback(
    async (fileId: string, data: Uint8Array) => {
      await db.files.update(fileId, { data, updatedAt: Date.now() });
      setHomeFiles(prev =>
        prev.map(f => (f.id === fileId ? { ...f, data } : f))
      );
    },
    []
  );

  const addFile = useCallback(
    async (folderPath?: string): Promise<string> => {
      await ensureHomeProject();
      // Strip ~/ prefix if present
      const homeFolder = folderPath?.startsWith(HOME_PREFIX)
        ? folderPath.slice(HOME_PREFIX.length)
        : folderPath;

      const baseName = generateUniqueName(homeFiles, folderPath);
      const name = homeFolder ? `${homeFolder}/${baseName}` : baseName;
      const now = Date.now();
      const id = crypto.randomUUID();
      const data = new Uint8Array(0);

      await db.files.add({
        id,
        projectName: HOME_PROJECT_NAME,
        path: name,
        data,
        createdAt: now,
        updatedAt: now,
      });

      setHomeFiles(prev => [...prev, { id, name: HOME_PREFIX + name, data }]);
      return id;
    },
    [homeFiles]
  );

  const addFolder = useCallback(
    async (parentPath?: string): Promise<string> => {
      await ensureHomeProject();
      const homeParent = parentPath?.startsWith(HOME_PREFIX)
        ? parentPath.slice(HOME_PREFIX.length)
        : parentPath;

      const folderName = generateUniqueFolderName(homeFiles, parentPath);
      const fullPath = homeParent ? `${homeParent}/${folderName}` : folderName;
      const fileName = generateUniqueName(homeFiles, HOME_PREFIX + fullPath);
      const name = `${fullPath}/${fileName}`;
      const now = Date.now();
      const id = crypto.randomUUID();
      const data = new Uint8Array(0);

      await db.files.add({
        id,
        projectName: HOME_PROJECT_NAME,
        path: name,
        data,
        createdAt: now,
        updatedAt: now,
      });

      setHomeFiles(prev => [...prev, { id, name: HOME_PREFIX + name, data }]);
      return HOME_PREFIX + fullPath;
    },
    [homeFiles]
  );

  const deleteFile = useCallback(async (fileId: string) => {
    await db.files.delete(fileId);
    setHomeFiles(prev => prev.filter(f => f.id !== fileId));
  }, []);

  const deleteFolder = useCallback(
    async (folderPath: string) => {
      // folderPath includes ~/ prefix
      const filesToDelete = homeFiles.filter(f =>
        f.name.startsWith(folderPath + "/")
      );
      await Promise.all(filesToDelete.map(f => db.files.delete(f.id)));
      setHomeFiles(prev =>
        prev.filter(f => !f.name.startsWith(folderPath + "/"))
      );
    },
    [homeFiles]
  );

  const renameFile = useCallback(async (fileId: string, newName: string) => {
    // newName includes ~/ prefix
    const homePath = newName.startsWith(HOME_PREFIX)
      ? newName.slice(HOME_PREFIX.length)
      : newName;
    await db.files.update(fileId, { path: homePath, updatedAt: Date.now() });
    setHomeFiles(prev =>
      prev.map(f => (f.id === fileId ? { ...f, name: newName } : f))
    );
  }, []);

  const renameFolder = useCallback(
    async (oldPath: string, newName: string) => {
      // oldPath includes ~/ prefix
      const parts = oldPath.split("/");
      parts[parts.length - 1] = newName;
      const newPath = parts.join("/");

      const filesToUpdate = homeFiles.filter(f =>
        f.name.startsWith(oldPath + "/")
      );
      await Promise.all(
        filesToUpdate.map(f => {
          const newFilePath = newPath + f.name.slice(oldPath.length);
          const homePath = newFilePath.startsWith(HOME_PREFIX)
            ? newFilePath.slice(HOME_PREFIX.length)
            : newFilePath;
          return db.files.update(f.id, {
            path: homePath,
            updatedAt: Date.now(),
          });
        })
      );
      setHomeFiles(prev =>
        prev.map(f => {
          if (f.name.startsWith(oldPath + "/")) {
            return { ...f, name: newPath + f.name.slice(oldPath.length) };
          }
          return f;
        })
      );
    },
    [homeFiles]
  );

  const moveFile = useCallback(
    async (fileId: string, targetFolder: string | null) => {
      const file = homeFiles.find(f => f.id === fileId);
      if (!file) return;
      const parts = file.name.split("/");
      const baseName = parts[parts.length - 1];
      const newName = targetFolder ? `${targetFolder}/${baseName}` : baseName;
      if (newName === file.name) return;
      const homePath = newName.startsWith(HOME_PREFIX)
        ? newName.slice(HOME_PREFIX.length)
        : newName;
      await db.files.update(fileId, { path: homePath, updatedAt: Date.now() });
      setHomeFiles(prev =>
        prev.map(f => (f.id === fileId ? { ...f, name: newName } : f))
      );
    },
    [homeFiles]
  );

  /** VFS-format files for sending to workers (with /home/ prefix paths). */
  const homeVfsFiles = useMemo(
    () =>
      homeFiles.map(f => ({
        path: "/home/" + f.name.slice(HOME_PREFIX.length),
        content: f.data,
      })),
    [homeFiles]
  );

  return {
    homeFiles,
    homeVfsFiles,
    reloadHomeFiles: loadHomeFiles,
    updateHomeFileContent: updateFileContent,
    addHomeFile: addFile,
    addHomeFolder: addFolder,
    deleteHomeFile: deleteFile,
    deleteHomeFolder: deleteFolder,
    renameHomeFile: renameFile,
    renameHomeFolder: renameFolder,
    moveHomeFile: moveFile,
  };
}

function generateUniqueName(
  files: WorkspaceFile[],
  folderPath?: string
): string {
  const existing = new Set(files.map(f => f.name));
  for (let i = 1; ; i++) {
    const baseName = `untitled${i === 1 ? "" : i}.m`;
    const fullName = folderPath ? `${folderPath}/${baseName}` : baseName;
    if (!existing.has(fullName)) return baseName;
  }
}

function generateUniqueFolderName(
  files: WorkspaceFile[],
  parentPath?: string
): string {
  const existingFolders = new Set<string>();
  for (const file of files) {
    if (parentPath) {
      if (file.name.startsWith(parentPath + "/")) {
        const relativeParts = file.name.slice(parentPath.length + 1).split("/");
        if (relativeParts.length > 1) {
          existingFolders.add(relativeParts[0]);
        }
      }
    } else {
      const parts = file.name.split("/");
      if (parts.length > 1) {
        existingFolders.add(parts[0]);
      }
    }
  }
  for (let i = 1; ; i++) {
    const name = `folder${i === 1 ? "" : i}`;
    if (!existingFolders.has(name)) return name;
  }
}
