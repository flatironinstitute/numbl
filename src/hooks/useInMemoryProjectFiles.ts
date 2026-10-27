import { useState, useCallback, useMemo, useReducer, useRef } from "react";
import type { UseProjectFilesResult } from "./useProjectFiles";
import {
  filesReducer,
  generateDuplicateName,
  generateUniqueFolderName,
  generateUniqueName,
  uploadOverwriteMessage,
  type FileEntry,
} from "./projectFilesState";

const textEncoder = new TextEncoder();

export interface InMemoryProjectFiles extends UseProjectFilesResult {
  /** Replace all files and their contents (initial load). */
  setProject: (
    files: FileEntry[],
    contents: Map<string, Uint8Array>,
    activeFileId: string
  ) => void;
  setLoading: (loading: boolean) => void;
}

/**
 * Project files held entirely in memory (no persistence). Shared by the
 * share-URL and static-site hooks, which differ only in how the project is
 * loaded and whether edits are mirrored somewhere (`onContentChange`).
 */
export function useInMemoryProjectFiles(
  onContentChange?: (
    files: FileEntry[],
    contents: Map<string, Uint8Array>,
    activeFileId: string
  ) => void
): InMemoryProjectFiles {
  const [files, dispatch] = useReducer(filesReducer, []);
  const [activeFileId, setActiveFileId] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const contentMapRef = useRef(new Map<string, Uint8Array>());

  const setProject = useCallback(
    (
      newFiles: FileEntry[],
      contents: Map<string, Uint8Array>,
      newActiveFileId: string
    ) => {
      contentMapRef.current = contents;
      dispatch({ type: "SET_FILES", files: newFiles });
      setActiveFileId(newActiveFileId);
    },
    []
  );

  const updateFileContent = useCallback(
    (content: string) => {
      if (!activeFileId) return;
      contentMapRef.current.set(activeFileId, textEncoder.encode(content));
      onContentChange?.(files, contentMapRef.current, activeFileId);
    },
    [activeFileId, files, onContentChange]
  );

  const emptyData = useMemo(() => new Uint8Array(0), []);

  const addFile = useCallback(
    async (folderPath?: string): Promise<string> => {
      const baseName = generateUniqueName(files, folderPath);
      const name = folderPath ? `${folderPath}/${baseName}` : baseName;
      const id = crypto.randomUUID();
      contentMapRef.current.set(id, emptyData);
      dispatch({ type: "ADD_FILE", file: { id, name } });
      setActiveFileId(id);
      return id;
    },
    [files, emptyData]
  );

  const addFolder = useCallback(
    async (parentPath?: string): Promise<string> => {
      const folderName = generateUniqueFolderName(files, parentPath);
      const fullPath = parentPath ? `${parentPath}/${folderName}` : folderName;
      const fileName = generateUniqueName(files, fullPath);
      const name = `${fullPath}/${fileName}`;
      const id = crypto.randomUUID();
      contentMapRef.current.set(id, emptyData);
      dispatch({ type: "ADD_FILE", file: { id, name } });
      setActiveFileId(id);
      return fullPath;
    },
    [files, emptyData]
  );

  const deleteFile = useCallback(
    async (fileId: string) => {
      contentMapRef.current.delete(fileId);
      if (activeFileId === fileId) {
        const newFiles = files.filter(f => f.id !== fileId);
        setActiveFileId(newFiles.length > 0 ? newFiles[0].id : "");
      }
      dispatch({ type: "DELETE_FILE", fileId });
    },
    [files, activeFileId]
  );

  const deleteFolder = useCallback(
    async (folderPath: string) => {
      const filesToDelete = files.filter(f =>
        f.name.startsWith(folderPath + "/")
      );
      if (filesToDelete.length === files.length) {
        alert("Cannot delete all files");
        return;
      }
      for (const f of filesToDelete) {
        contentMapRef.current.delete(f.id);
      }
      const newFiles = files.filter(f => !f.name.startsWith(folderPath + "/"));
      if (
        filesToDelete.some(f => f.id === activeFileId) &&
        newFiles.length > 0
      ) {
        setActiveFileId(newFiles[0].id);
      }
      filesToDelete.forEach(f => {
        dispatch({ type: "DELETE_FILE", fileId: f.id });
      });
    },
    [files, activeFileId]
  );

  const renameFile = useCallback(
    async (fileId: string, newName: string) => {
      if (files.some(f => f.id !== fileId && f.name === newName)) {
        alert("A file with this name already exists");
        return;
      }
      dispatch({ type: "RENAME_FILE", fileId, newName });
    },
    [files]
  );

  const renameFolder = useCallback(async (oldPath: string, newName: string) => {
    const parts = oldPath.split("/");
    parts[parts.length - 1] = newName;
    const newPath = parts.join("/");
    dispatch({ type: "RENAME_FOLDER", oldPath, newPath });
  }, []);

  const moveFile = useCallback(
    async (fileId: string, targetFolder: string | null) => {
      const file = files.find(f => f.id === fileId);
      if (!file) return;
      const parts = file.name.split("/");
      const baseName = parts[parts.length - 1];
      const newName = targetFolder ? `${targetFolder}/${baseName}` : baseName;
      if (newName === file.name) return;
      if (files.some(f => f.id !== fileId && f.name === newName)) {
        alert("A file with this name already exists in the target location");
        return;
      }
      dispatch({ type: "RENAME_FILE", fileId, newName });
    },
    [files]
  );

  const duplicateFile = useCallback(
    async (fileId: string): Promise<string> => {
      const source = files.find(f => f.id === fileId);
      if (!source) return "";
      const newName = generateDuplicateName(files, source.name);
      const sourceData = contentMapRef.current.get(fileId) ?? new Uint8Array(0);
      const dataCopy = new Uint8Array(sourceData);
      const id = crypto.randomUUID();
      contentMapRef.current.set(id, dataCopy);
      dispatch({ type: "ADD_FILE", file: { id, name: newName } });
      setActiveFileId(id);
      return id;
    },
    [files]
  );

  const uploadFiles = useCallback(
    async (
      entries: { path: string; content: string }[],
      targetFolder?: string
    ) => {
      if (entries.length === 0) return;
      const toUpload = entries.map(e => ({
        ...e,
        fullPath: targetFolder ? `${targetFolder}/${e.path}` : e.path,
      }));
      const existingByPath = new Map(files.map(f => [f.name, f]));
      const duplicates = toUpload.filter(e => existingByPath.has(e.fullPath));
      const newEntries = toUpload.filter(e => !existingByPath.has(e.fullPath));

      if (duplicates.length > 0) {
        const ok = window.confirm(
          uploadOverwriteMessage(
            duplicates.map(d => d.fullPath),
            newEntries.length
          )
        );
        if (!ok) return;
      }

      let firstNewId = "";
      for (const entry of duplicates) {
        const existing = existingByPath.get(entry.fullPath)!;
        contentMapRef.current.set(
          existing.id,
          textEncoder.encode(entry.content)
        );
        if (!firstNewId) firstNewId = existing.id;
      }
      for (const entry of newEntries) {
        const id = crypto.randomUUID();
        contentMapRef.current.set(id, textEncoder.encode(entry.content));
        dispatch({ type: "ADD_FILE", file: { id, name: entry.fullPath } });
        if (!firstNewId) firstNewId = id;
      }
      if (firstNewId) setActiveFileId(firstNewId);
    },
    [files]
  );

  const reload = useCallback(async () => {
    // Nothing to reload from.
  }, []);

  const loadFileContent = useCallback(
    async (fileId: string): Promise<Uint8Array> => {
      return contentMapRef.current.get(fileId) ?? new Uint8Array(0);
    },
    []
  );

  const loadAllContents = useCallback(async (): Promise<
    Map<string, Uint8Array>
  > => {
    return new Map(contentMapRef.current);
  }, []);

  const mergeVfsChanges = useCallback(() => {
    // Script-side file changes are not mirrored back in memory-only mode.
  }, []);

  return {
    files,
    activeFileId,
    loading,
    setActiveFileId,
    updateFileContent,
    addFile,
    addFolder,
    deleteFile,
    deleteFolder,
    renameFile,
    renameFolder,
    moveFile,
    duplicateFile,
    uploadFiles,
    reload,
    loadFileContent,
    loadAllContents,
    contentCache: contentMapRef,
    mergeVfsChanges,
    setProject,
    setLoading,
  };
}
