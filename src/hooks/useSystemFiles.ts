import { useState, useEffect, useCallback, useRef } from "react";
import { db } from "../db/schema";
import {
  ensureSystemProject,
  getSystemFiles,
  getFileContent,
  getSystemFileContents,
  saveFileData,
  SYSTEM_PROJECT_NAME,
} from "../db/operations";
import type { WorkspaceFile } from "./useProjectFiles";

const SYSTEM_PREFIX = "system/";

export function useSystemFiles() {
  const [systemFiles, setSystemFiles] = useState<WorkspaceFile[]>([]);
  const contentCacheRef = useRef(new Map<string, Uint8Array>());

  const loadSystemFiles = useCallback(async () => {
    const files = await getSystemFiles();
    setSystemFiles(
      files.map(f => ({
        id: f.id,
        name: SYSTEM_PREFIX + f.path,
      }))
    );
  }, []);

  useEffect(() => {
    // Initial async load from IndexedDB — setState is called after await, not synchronously
    // eslint-disable-next-line react-hooks/set-state-in-effect
    loadSystemFiles();
  }, [loadSystemFiles]);

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

  /** Load all system file contents from DB. Used before code execution. */
  const loadAllSystemContents = useCallback(async (): Promise<
    Map<string, Uint8Array>
  > => {
    const map = await getSystemFileContents();
    for (const [id, data] of map) {
      contentCacheRef.current.set(id, data);
    }
    return map;
  }, []);

  /** Build VFS files for workers. Loads all content from DB. */
  const getSystemVfsFiles = useCallback(async (): Promise<
    { path: string; content: Uint8Array }[]
  > => {
    const contentsMap = await loadAllSystemContents();
    return systemFiles.map(f => ({
      path: "/system/" + f.name.slice(SYSTEM_PREFIX.length),
      content: contentsMap.get(f.id) ?? new Uint8Array(0),
    }));
  }, [systemFiles, loadAllSystemContents]);

  /** Build workspace files (text) for workers. Loads all content from DB. */
  const getSystemWorkspaceFiles = useCallback(async (): Promise<
    { name: string; source: string }[]
  > => {
    const decoder = new TextDecoder("utf-8");
    const contentsMap = await loadAllSystemContents();
    return systemFiles.map(f => ({
      name: f.name,
      source: decoder.decode(contentsMap.get(f.id) ?? new Uint8Array(0)),
    }));
  }, [systemFiles, loadAllSystemContents]);

  const updateFileContent = useCallback(
    async (fileId: string, data: Uint8Array) => {
      contentCacheRef.current.set(fileId, data);
      await saveFileData(fileId, data);
    },
    []
  );

  const addFile = useCallback(
    async (folderPath?: string): Promise<string> => {
      await ensureSystemProject();
      const systemFolder = folderPath?.startsWith(SYSTEM_PREFIX)
        ? folderPath.slice(SYSTEM_PREFIX.length)
        : folderPath;

      const baseName = generateUniqueName(systemFiles, folderPath);
      const name = systemFolder ? `${systemFolder}/${baseName}` : baseName;
      const now = Date.now();
      const id = crypto.randomUUID();
      const data = new Uint8Array(0);

      await db.transaction("rw", db.files, db.fileContents, async () => {
        await db.files.add({
          id,
          projectName: SYSTEM_PROJECT_NAME,
          path: name,
          createdAt: now,
          updatedAt: now,
        });
        await db.fileContents.add({ id, data });
      });

      contentCacheRef.current.set(id, data);
      setSystemFiles(prev => [...prev, { id, name: SYSTEM_PREFIX + name }]);
      return id;
    },
    [systemFiles]
  );

  const addFolder = useCallback(
    async (parentPath?: string): Promise<string> => {
      await ensureSystemProject();
      const systemParent = parentPath?.startsWith(SYSTEM_PREFIX)
        ? parentPath.slice(SYSTEM_PREFIX.length)
        : parentPath;

      const folderName = generateUniqueFolderName(systemFiles, parentPath);
      const fullPath = systemParent
        ? `${systemParent}/${folderName}`
        : folderName;
      const fileName = generateUniqueName(
        systemFiles,
        SYSTEM_PREFIX + fullPath
      );
      const name = `${fullPath}/${fileName}`;
      const now = Date.now();
      const id = crypto.randomUUID();
      const data = new Uint8Array(0);

      await db.transaction("rw", db.files, db.fileContents, async () => {
        await db.files.add({
          id,
          projectName: SYSTEM_PROJECT_NAME,
          path: name,
          createdAt: now,
          updatedAt: now,
        });
        await db.fileContents.add({ id, data });
      });

      contentCacheRef.current.set(id, data);
      setSystemFiles(prev => [...prev, { id, name: SYSTEM_PREFIX + name }]);
      return SYSTEM_PREFIX + fullPath;
    },
    [systemFiles]
  );

  const deleteFile = useCallback(async (fileId: string) => {
    await db.transaction("rw", db.files, db.fileContents, async () => {
      await db.files.delete(fileId);
      await db.fileContents.delete(fileId);
    });
    contentCacheRef.current.delete(fileId);
    setSystemFiles(prev => prev.filter(f => f.id !== fileId));
  }, []);

  const deleteFolder = useCallback(
    async (folderPath: string) => {
      const filesToDelete = systemFiles.filter(f =>
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
      setSystemFiles(prev =>
        prev.filter(f => !f.name.startsWith(folderPath + "/"))
      );
    },
    [systemFiles]
  );

  const renameFile = useCallback(async (fileId: string, newName: string) => {
    const systemPath = newName.startsWith(SYSTEM_PREFIX)
      ? newName.slice(SYSTEM_PREFIX.length)
      : newName;
    await db.files.update(fileId, { path: systemPath, updatedAt: Date.now() });
    setSystemFiles(prev =>
      prev.map(f => (f.id === fileId ? { ...f, name: newName } : f))
    );
  }, []);

  const renameFolder = useCallback(
    async (oldPath: string, newName: string) => {
      const parts = oldPath.split("/");
      parts[parts.length - 1] = newName;
      const newPath = parts.join("/");

      const filesToUpdate = systemFiles.filter(f =>
        f.name.startsWith(oldPath + "/")
      );
      await Promise.all(
        filesToUpdate.map(f => {
          const newFilePath = newPath + f.name.slice(oldPath.length);
          const systemPath = newFilePath.startsWith(SYSTEM_PREFIX)
            ? newFilePath.slice(SYSTEM_PREFIX.length)
            : newFilePath;
          return db.files.update(f.id, {
            path: systemPath,
            updatedAt: Date.now(),
          });
        })
      );
      setSystemFiles(prev =>
        prev.map(f => {
          if (f.name.startsWith(oldPath + "/")) {
            return { ...f, name: newPath + f.name.slice(oldPath.length) };
          }
          return f;
        })
      );
    },
    [systemFiles]
  );

  const moveFile = useCallback(
    async (fileId: string, targetFolder: string | null) => {
      const file = systemFiles.find(f => f.id === fileId);
      if (!file) return;
      const parts = file.name.split("/");
      const baseName = parts[parts.length - 1];
      const newName = targetFolder ? `${targetFolder}/${baseName}` : baseName;
      if (newName === file.name) return;
      const systemPath = newName.startsWith(SYSTEM_PREFIX)
        ? newName.slice(SYSTEM_PREFIX.length)
        : newName;
      await db.files.update(fileId, {
        path: systemPath,
        updatedAt: Date.now(),
      });
      setSystemFiles(prev =>
        prev.map(f => (f.id === fileId ? { ...f, name: newName } : f))
      );
    },
    [systemFiles]
  );

  return {
    systemFiles,
    reloadSystemFiles: loadSystemFiles,
    updateSystemFileContent: updateFileContent,
    addSystemFile: addFile,
    addSystemFolder: addFolder,
    deleteSystemFile: deleteFile,
    deleteSystemFolder: deleteFolder,
    renameSystemFile: renameFile,
    renameSystemFolder: renameFolder,
    moveSystemFile: moveFile,
    loadSystemFileContent: loadFileContent,
    loadAllSystemContents,
    getSystemVfsFiles,
    getSystemWorkspaceFiles,
    systemContentCache: contentCacheRef,
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
