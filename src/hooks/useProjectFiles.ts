import {
  useState,
  useEffect,
  useCallback,
  useMemo,
  useReducer,
  useRef,
} from "react";
import {
  getProjectFiles,
  saveFileData,
  createFile,
  deleteFile,
  renameFile as renameFileInDb,
  getFileContent,
  getAllFileContents,
} from "../db/operations";
import {
  debounce,
  filesReducer,
  generateDuplicateName,
  generateUniqueFolderName,
  generateUniqueName,
  uploadOverwriteMessage,
  type FileEntry,
} from "./projectFilesState";

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder("utf-8");

/** Decode file data as UTF-8 text. */
export function fileText(data: Uint8Array): string {
  return textDecoder.decode(data);
}

/** Check if data contains binary (non-text) content. */
export function isBinaryData(data: Uint8Array): boolean {
  for (let i = 0; i < data.length; i++) {
    if (data[i] === 0) return true;
  }
  return false;
}

export interface UseProjectFilesResult {
  files: FileEntry[];
  activeFileId: string;
  loading: boolean;
  setActiveFileId: (id: string) => void;
  updateFileContent: (content: string) => void;
  addFile: (folderPath?: string) => Promise<string>;
  addFolder: (parentPath?: string) => Promise<string>;
  deleteFile: (fileId: string) => void;
  deleteFolder: (folderPath: string) => void;
  renameFile: (fileId: string, newName: string) => void;
  renameFolder: (oldPath: string, newName: string) => void;
  moveFile: (fileId: string, targetFolder: string | null) => void;
  duplicateFile: (fileId: string) => Promise<string>;
  uploadFiles: (
    entries: { path: string; content: string }[],
    targetFolder?: string
  ) => Promise<void>;
  reload: () => Promise<void>;
  /** Load content for a single file (cached). */
  loadFileContent: (fileId: string) => Promise<Uint8Array>;
  /** Load all file contents for this project. Returns map of id → data. */
  loadAllContents: () => Promise<Map<string, Uint8Array>>;
  /** Content cache — updated on edits and loads. */
  contentCache: React.RefObject<Map<string, Uint8Array>>;
  mergeVfsChanges: (result: {
    addedFiles: FileEntry[];
    modifiedFiles: { path: string; data: Uint8Array }[];
    deletedPaths: string[];
  }) => void;
}

function getStoredActiveFileId(projectName: string): string {
  try {
    return localStorage.getItem(`numbl_active_file_${projectName}`) || "";
  } catch {
    return "";
  }
}

function storeActiveFileId(projectName: string, fileId: string): void {
  try {
    localStorage.setItem(`numbl_active_file_${projectName}`, fileId);
  } catch {
    // Ignore storage errors
  }
}

export function useProjectFiles(projectName: string): UseProjectFilesResult {
  const [files, dispatch] = useReducer(filesReducer, []);
  const [activeFileId, setActiveFileIdRaw] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const contentCacheRef = useRef(new Map<string, Uint8Array>());

  const setActiveFileId = useCallback(
    (id: string) => {
      setActiveFileIdRaw(id);
      storeActiveFileId(projectName, id);
    },
    [projectName]
  );

  // Load file metadata from IndexedDB (no content)
  useEffect(() => {
    let cancelled = false;

    async function loadFilesFromDB() {
      try {
        setLoading(true);
        const projectFiles = await getProjectFiles(projectName);

        if (cancelled) return;

        const workspaceFiles: FileEntry[] = projectFiles.map(pf => ({
          id: pf.id,
          name: pf.path,
        }));

        dispatch({ type: "SET_FILES", files: workspaceFiles });

        setActiveFileIdRaw(prevId => {
          if (!prevId && workspaceFiles.length > 0) {
            const storedId = getStoredActiveFileId(projectName);
            if (storedId && workspaceFiles.some(f => f.id === storedId)) {
              return storedId;
            }
            return workspaceFiles[0].id;
          }
          return prevId;
        });
      } catch (error) {
        console.error("Failed to load files:", error);
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    loadFilesFromDB();

    return () => {
      cancelled = true;
    };
  }, [projectName]);

  const reload = useCallback(async () => {
    try {
      setLoading(true);
      const projectFiles = await getProjectFiles(projectName);

      const workspaceFiles: FileEntry[] = projectFiles.map(pf => ({
        id: pf.id,
        name: pf.path,
      }));

      dispatch({ type: "SET_FILES", files: workspaceFiles });
    } catch (error) {
      console.error("Failed to reload files:", error);
    } finally {
      setLoading(false);
    }
  }, [projectName]);

  // Content loading
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

  const loadAllContents = useCallback(async (): Promise<
    Map<string, Uint8Array>
  > => {
    const map = await getAllFileContents(projectName);
    // Update cache
    for (const [id, data] of map) {
      contentCacheRef.current.set(id, data);
    }
    return map;
  }, [projectName]);

  // Debounced save function
  const debouncedSave = useMemo(
    () =>
      debounce(async (fileId: string, data: Uint8Array) => {
        try {
          await saveFileData(fileId, data);
        } catch (error) {
          console.error("Failed to save file:", error);
        }
      }, 500),
    []
  );

  const updateFileContent = useCallback(
    (content: string) => {
      const data = textEncoder.encode(content);
      contentCacheRef.current.set(activeFileId, data);
      debouncedSave(activeFileId, data);
    },
    [activeFileId, debouncedSave]
  );

  const emptyData = useMemo(() => new Uint8Array(0), []);

  const addFile = useCallback(
    async (folderPath?: string): Promise<string> => {
      const baseName = generateUniqueName(files, folderPath);
      const name = folderPath ? `${folderPath}/${baseName}` : baseName;

      try {
        const file = await createFile(projectName, name, emptyData);
        dispatch({
          type: "ADD_FILE",
          file: { id: file.id, name },
        });
        contentCacheRef.current.set(file.id, emptyData);
        setActiveFileId(file.id);
        return file.id;
      } catch (error) {
        console.error("Failed to create file:", error);
        return "";
      }
    },
    [files, projectName, setActiveFileId, emptyData]
  );

  const addFolder = useCallback(
    async (parentPath?: string): Promise<string> => {
      const folderName = generateUniqueFolderName(files, parentPath);
      const fullPath = parentPath ? `${parentPath}/${folderName}` : folderName;
      const fileName = generateUniqueName(files, fullPath);
      const name = `${fullPath}/${fileName}`;

      try {
        const file = await createFile(projectName, name, emptyData);
        dispatch({
          type: "ADD_FILE",
          file: { id: file.id, name },
        });
        contentCacheRef.current.set(file.id, emptyData);
        setActiveFileId(file.id);
        return fullPath;
      } catch (error) {
        console.error("Failed to create folder:", error);
        return "";
      }
    },
    [files, projectName, setActiveFileId, emptyData]
  );

  const handleDeleteFile = useCallback(
    async (fileId: string) => {
      try {
        await deleteFile(fileId);
        contentCacheRef.current.delete(fileId);
        if (activeFileId === fileId) {
          const newFiles = files.filter(f => f.id !== fileId);
          setActiveFileId(newFiles.length > 0 ? newFiles[0].id : "");
        }
        dispatch({ type: "DELETE_FILE", fileId });
      } catch (error) {
        console.error("Failed to delete file:", error);
      }
    },
    [files, activeFileId, setActiveFileId]
  );

  const handleDeleteFolder = useCallback(
    async (folderPath: string) => {
      const filesToDelete = files.filter(f =>
        f.name.startsWith(folderPath + "/")
      );
      if (filesToDelete.length === files.length) {
        alert("Cannot delete all files");
        return;
      }

      try {
        await Promise.all(filesToDelete.map(f => deleteFile(f.id)));
        for (const f of filesToDelete) {
          contentCacheRef.current.delete(f.id);
        }
        const newFiles = files.filter(
          f => !f.name.startsWith(folderPath + "/")
        );
        if (
          filesToDelete.some(f => f.id === activeFileId) &&
          newFiles.length > 0
        ) {
          setActiveFileId(newFiles[0].id);
        }
        filesToDelete.forEach(f => {
          dispatch({ type: "DELETE_FILE", fileId: f.id });
        });
      } catch (error) {
        console.error("Failed to delete folder:", error);
      }
    },
    [files, activeFileId, setActiveFileId]
  );

  const handleRenameFile = useCallback(
    async (fileId: string, newName: string) => {
      if (files.some(f => f.id !== fileId && f.name === newName)) {
        alert("A file with this name already exists");
        return;
      }

      try {
        await renameFileInDb(fileId, newName);
        dispatch({ type: "RENAME_FILE", fileId, newName });
      } catch (error) {
        console.error("Failed to rename file:", error);
      }
    },
    [files]
  );

  const handleRenameFolder = useCallback(
    async (oldPath: string, newName: string) => {
      const parts = oldPath.split("/");
      parts[parts.length - 1] = newName;
      const newPath = parts.join("/");

      const filesToUpdate = files.filter(f => f.name.startsWith(oldPath + "/"));

      try {
        await Promise.all(
          filesToUpdate.map(f => {
            const newFilePath = newPath + f.name.slice(oldPath.length);
            return renameFileInDb(f.id, newFilePath);
          })
        );

        dispatch({ type: "RENAME_FOLDER", oldPath, newPath });
      } catch (error) {
        console.error("Failed to rename folder:", error);
      }
    },
    [files]
  );

  const handleMoveFile = useCallback(
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

      try {
        await renameFileInDb(fileId, newName);
        dispatch({ type: "RENAME_FILE", fileId, newName });
      } catch (error) {
        console.error("Failed to move file:", error);
      }
    },
    [files]
  );

  const handleDuplicateFile = useCallback(
    async (fileId: string): Promise<string> => {
      const source = files.find(f => f.id === fileId);
      if (!source) return "";
      const newName = generateDuplicateName(files, source.name);
      const sourceData =
        contentCacheRef.current.get(fileId) ?? (await getFileContent(fileId));
      const dataCopy = new Uint8Array(sourceData);
      try {
        const file = await createFile(projectName, newName, dataCopy);
        dispatch({ type: "ADD_FILE", file: { id: file.id, name: newName } });
        contentCacheRef.current.set(file.id, dataCopy);
        setActiveFileId(file.id);
        return file.id;
      } catch (error) {
        console.error("Failed to duplicate file:", error);
        return "";
      }
    },
    [files, projectName, setActiveFileId]
  );

  const handleUploadFiles = useCallback(
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

      try {
        let firstNewId = "";

        for (const entry of duplicates) {
          const existing = existingByPath.get(entry.fullPath)!;
          const data = textEncoder.encode(entry.content);
          await saveFileData(existing.id, data);
          contentCacheRef.current.set(existing.id, data);
          if (!firstNewId) firstNewId = existing.id;
        }

        for (const entry of newEntries) {
          const data = textEncoder.encode(entry.content);
          const file = await createFile(projectName, entry.fullPath, data);
          dispatch({
            type: "ADD_FILE",
            file: { id: file.id, name: entry.fullPath },
          });
          contentCacheRef.current.set(file.id, data);
          if (!firstNewId) firstNewId = file.id;
        }

        if (firstNewId) {
          setActiveFileId(firstNewId);
        }
      } catch (error) {
        console.error("Failed to upload files:", error);
      }
    },
    [files, projectName, setActiveFileId]
  );

  return {
    files,
    activeFileId,
    loading,
    setActiveFileId,
    updateFileContent,
    addFile,
    addFolder,
    deleteFile: handleDeleteFile,
    deleteFolder: handleDeleteFolder,
    renameFile: handleRenameFile,
    renameFolder: handleRenameFolder,
    moveFile: handleMoveFile,
    duplicateFile: handleDuplicateFile,
    uploadFiles: handleUploadFiles,
    reload,
    loadFileContent,
    loadAllContents,
    contentCache: contentCacheRef,
    mergeVfsChanges: useCallback(
      (result: {
        addedFiles: FileEntry[];
        modifiedFiles: { path: string; data: Uint8Array }[];
        deletedPaths: string[];
      }) => {
        // Update content cache for modified files
        for (const mod of result.modifiedFiles) {
          const file = files.find(f => f.name === mod.path);
          if (file) {
            contentCacheRef.current.set(file.id, mod.data);
          }
        }
        // Cache content for added files — find data from addedFiles
        // (addedFiles still carry data from syncVfsChanges)
        dispatch({
          type: "MERGE_VFS",
          added: result.addedFiles,
          deletedPaths: result.deletedPaths,
        });
      },
      [files]
    ),
  };
}
