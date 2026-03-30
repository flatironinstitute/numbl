import { useState, useEffect, useCallback, useRef } from "react";
import { db } from "../db/schema";
import {
  ensureHomeProject,
  getHomeFiles,
  getFileContent,
  getHomeFileContents,
  saveFileData,
} from "../db/operations";
import type { WorkspaceFile } from "./useProjectFiles";

const HOME_PROJECT_NAME = "__home__";
const HOME_PREFIX = "~/";

export function useHomeFiles() {
  const [homeFiles, setHomeFiles] = useState<WorkspaceFile[]>([]);
  const contentCacheRef = useRef(new Map<string, Uint8Array>());

  const loadHomeFiles = useCallback(async () => {
    const files = await getHomeFiles();
    setHomeFiles(
      files.map(f => ({
        id: f.id,
        name: HOME_PREFIX + f.path,
      }))
    );
  }, []);

  useEffect(() => {
    // Initial async load from IndexedDB — setState is called after await, not synchronously
    // eslint-disable-next-line react-hooks/set-state-in-effect
    loadHomeFiles();
  }, [loadHomeFiles]);

  const loadFileContent = useCallback(
    async (fileId: string): Promise<Uint8Array> => {
      const cached = contentCacheRef.current.get(fileId);
      if (cached !== undefined) return cached;
      const data = await getFileContent(fileId);
      contentCacheRef.current.set(fileId, data);
      return data;
    },
    []
  );

  /** Load all home file contents from DB. Used before code execution. */
  const loadAllHomeContents = useCallback(async (): Promise<
    Map<string, Uint8Array>
  > => {
    const map = await getHomeFileContents();
    for (const [id, data] of map) {
      contentCacheRef.current.set(id, data);
    }
    return map;
  }, []);

  /** Build VFS files for workers. Loads all content from DB. */
  const getHomeVfsFiles = useCallback(async (): Promise<
    { path: string; content: Uint8Array }[]
  > => {
    const contentsMap = await loadAllHomeContents();
    return homeFiles.map(f => ({
      path: "/home/" + f.name.slice(HOME_PREFIX.length),
      content: contentsMap.get(f.id) ?? new Uint8Array(0),
    }));
  }, [homeFiles, loadAllHomeContents]);

  /** Build workspace files (text) for workers. Loads all content from DB. */
  const getHomeWorkspaceFiles = useCallback(async (): Promise<
    { name: string; source: string }[]
  > => {
    const decoder = new TextDecoder("utf-8");
    const contentsMap = await loadAllHomeContents();
    return homeFiles.map(f => ({
      name: f.name,
      source: decoder.decode(contentsMap.get(f.id) ?? new Uint8Array(0)),
    }));
  }, [homeFiles, loadAllHomeContents]);

  const updateFileContent = useCallback(
    async (fileId: string, data: Uint8Array) => {
      contentCacheRef.current.set(fileId, data);
      await saveFileData(fileId, data);
    },
    []
  );

  const addFile = useCallback(
    async (folderPath?: string): Promise<string> => {
      await ensureHomeProject();
      const homeFolder = folderPath?.startsWith(HOME_PREFIX)
        ? folderPath.slice(HOME_PREFIX.length)
        : folderPath;

      const baseName = generateUniqueName(homeFiles, folderPath);
      const name = homeFolder ? `${homeFolder}/${baseName}` : baseName;
      const now = Date.now();
      const id = crypto.randomUUID();
      const data = new Uint8Array(0);

      await db.transaction("rw", db.files, db.fileContents, async () => {
        await db.files.add({
          id,
          projectName: HOME_PROJECT_NAME,
          path: name,
          createdAt: now,
          updatedAt: now,
        });
        await db.fileContents.add({ id, data });
      });

      contentCacheRef.current.set(id, data);
      setHomeFiles(prev => [...prev, { id, name: HOME_PREFIX + name }]);
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

      await db.transaction("rw", db.files, db.fileContents, async () => {
        await db.files.add({
          id,
          projectName: HOME_PROJECT_NAME,
          path: name,
          createdAt: now,
          updatedAt: now,
        });
        await db.fileContents.add({ id, data });
      });

      contentCacheRef.current.set(id, data);
      setHomeFiles(prev => [...prev, { id, name: HOME_PREFIX + name }]);
      return HOME_PREFIX + fullPath;
    },
    [homeFiles]
  );

  const deleteFile = useCallback(async (fileId: string) => {
    await db.transaction("rw", db.files, db.fileContents, async () => {
      await db.files.delete(fileId);
      await db.fileContents.delete(fileId);
    });
    contentCacheRef.current.delete(fileId);
    setHomeFiles(prev => prev.filter(f => f.id !== fileId));
  }, []);

  const deleteFolder = useCallback(
    async (folderPath: string) => {
      const filesToDelete = homeFiles.filter(f =>
        f.name.startsWith(folderPath + "/")
      );
      await db.transaction("rw", db.files, db.fileContents, async () => {
        for (const f of filesToDelete) {
          await db.files.delete(f.id);
          await db.fileContents.delete(f.id);
        }
      });
      for (const f of filesToDelete) {
        contentCacheRef.current.delete(f.id);
      }
      setHomeFiles(prev =>
        prev.filter(f => !f.name.startsWith(folderPath + "/"))
      );
    },
    [homeFiles]
  );

  const renameFile = useCallback(async (fileId: string, newName: string) => {
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

  return {
    homeFiles,
    reloadHomeFiles: loadHomeFiles,
    updateHomeFileContent: updateFileContent,
    addHomeFile: addFile,
    addHomeFolder: addFolder,
    deleteHomeFile: deleteFile,
    deleteHomeFolder: deleteFolder,
    renameHomeFile: renameFile,
    renameHomeFolder: renameFolder,
    moveHomeFile: moveFile,
    loadHomeFileContent: loadFileContent,
    loadAllHomeContents,
    getHomeVfsFiles,
    getHomeWorkspaceFiles,
    homeContentCache: contentCacheRef,
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
