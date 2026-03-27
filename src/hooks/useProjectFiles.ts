import { useState, useEffect, useCallback, useMemo, useReducer } from "react";
import {
  getProjectFiles,
  saveFileData,
  createFile,
  deleteFile,
  renameFile as renameFileInDb,
} from "../db/operations";

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder("utf-8");

export interface WorkspaceFile {
  id: string;
  name: string;
  data: Uint8Array; // All files stored as binary. Use textDecoder to get text.
}

/** Decode file data as UTF-8 text. */
export function fileText(f: WorkspaceFile): string {
  return textDecoder.decode(f.data);
}

/** Check if a file contains binary (non-text) data. */
export function isBinaryFile(f: WorkspaceFile): boolean {
  const data = f.data;
  for (let i = 0; i < data.length; i++) {
    if (data[i] === 0) return true;
  }
  return false;
}

// Actions for the files reducer
type FilesAction =
  | { type: "SET_FILES"; files: WorkspaceFile[] }
  | { type: "UPDATE_DATA"; fileId: string; data: Uint8Array }
  | { type: "ADD_FILE"; file: WorkspaceFile }
  | { type: "DELETE_FILE"; fileId: string }
  | { type: "RENAME_FILE"; fileId: string; newName: string }
  | { type: "RENAME_FOLDER"; oldPath: string; newPath: string }
  | { type: "MOVE_FILE"; fileId: string; newName: string }
  | {
      type: "MERGE_VFS";
      added: WorkspaceFile[];
      modified: { path: string; data: Uint8Array }[];
      deletedPaths: string[];
    };

// Reducer for managing files state
function filesReducer(
  state: WorkspaceFile[],
  action: FilesAction
): WorkspaceFile[] {
  switch (action.type) {
    case "SET_FILES":
      return action.files;

    case "UPDATE_DATA":
      return state.map(f =>
        f.id === action.fileId ? { ...f, data: action.data } : f
      );

    case "ADD_FILE":
      return [...state, action.file];

    case "DELETE_FILE":
      return state.filter(f => f.id !== action.fileId);

    case "RENAME_FILE":
      return state.map(f =>
        f.id === action.fileId ? { ...f, name: action.newName } : f
      );

    case "RENAME_FOLDER":
      return state.map(f => {
        if (f.name.startsWith(action.oldPath + "/")) {
          return {
            ...f,
            name: action.newPath + f.name.slice(action.oldPath.length),
          };
        }
        return f;
      });

    case "MOVE_FILE":
      return state.map(f =>
        f.id === action.fileId ? { ...f, name: action.newName } : f
      );

    case "MERGE_VFS": {
      let result = state;
      if (action.modified.length > 0) {
        result = result.map(f => {
          const mod = action.modified.find(m => m.path === f.name);
          return mod ? { ...f, data: mod.data } : f;
        });
      }
      if (action.deletedPaths.length > 0) {
        const deletedSet = new Set(action.deletedPaths);
        result = result.filter(f => !deletedSet.has(f.name));
      }
      if (action.added.length > 0) {
        result = [...result, ...action.added];
      }
      return result;
    }

    default:
      return state;
  }
}

export interface UseProjectFilesResult {
  files: WorkspaceFile[];
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
  uploadFiles: (
    entries: { path: string; content: string }[],
    targetFolder?: string
  ) => Promise<void>;
  reload: () => Promise<void>;
  mergeVfsChanges: (result: {
    addedFiles: WorkspaceFile[];
    modifiedFiles: { path: string; data: Uint8Array }[];
    deletedPaths: string[];
  }) => void;
}

// Debounce helper
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function debounce<T extends (...args: any[]) => any>(
  fn: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timeoutId: ReturnType<typeof setTimeout>;
  return (...args: Parameters<T>) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn(...args), delay);
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
    const parts = file.name.split("/");
    if (parentPath) {
      if (file.name.startsWith(parentPath + "/")) {
        const relativeParts = file.name.slice(parentPath.length + 1).split("/");
        if (relativeParts.length > 1) {
          existingFolders.add(relativeParts[0]);
        }
      }
    } else {
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

  const setActiveFileId = useCallback(
    (id: string) => {
      setActiveFileIdRaw(id);
      storeActiveFileId(projectName, id);
    },
    [projectName]
  );

  // Load files from IndexedDB
  useEffect(() => {
    let cancelled = false;

    async function loadFilesFromDB() {
      try {
        setLoading(true);
        const projectFiles = await getProjectFiles(projectName);

        if (cancelled) return;

        const workspaceFiles: WorkspaceFile[] = projectFiles.map(pf => ({
          id: pf.id,
          name: pf.path,
          data: pf.data,
        }));

        dispatch({ type: "SET_FILES", files: workspaceFiles });

        // Restore previously active file, or fall back to the first file
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
  }, [projectName]); // Only reload when project changes

  // Reload function for manual reloads
  const reload = useCallback(async () => {
    try {
      setLoading(true);
      const projectFiles = await getProjectFiles(projectName);

      const workspaceFiles: WorkspaceFile[] = projectFiles.map(pf => ({
        id: pf.id,
        name: pf.path,
        data: pf.data,
      }));

      dispatch({ type: "SET_FILES", files: workspaceFiles });
    } catch (error) {
      console.error("Failed to reload files:", error);
    } finally {
      setLoading(false);
    }
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
      dispatch({ type: "UPDATE_DATA", fileId: activeFileId, data });
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
          file: { id: file.id, name, data: emptyData },
        });
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
          file: { id: file.id, name, data: emptyData },
        });
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
      if (files.length === 1) {
        alert("Cannot delete the last file");
        return;
      }

      try {
        await deleteFile(fileId);
        if (activeFileId === fileId) {
          const newFiles = files.filter(f => f.id !== fileId);
          if (newFiles.length > 0) setActiveFileId(newFiles[0].id);
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

      // Check for duplicates
      const existingByPath = new Map(files.map(f => [f.name, f]));
      const duplicates = toUpload.filter(e => existingByPath.has(e.fullPath));
      const newEntries = toUpload.filter(e => !existingByPath.has(e.fullPath));

      if (duplicates.length > 0) {
        const names = duplicates.map(d => d.fullPath).join("\n");
        const ok = window.confirm(
          `The following ${duplicates.length} file(s) already exist and will be overwritten:\n\n${names}\n\n` +
            (newEntries.length > 0
              ? `${newEntries.length} new file(s) will also be added.\n\n`
              : "") +
            "Click OK to proceed, or Cancel to abort the upload."
        );
        if (!ok) return;
      }

      try {
        let firstNewId = "";

        // Overwrite existing files
        for (const entry of duplicates) {
          const existing = existingByPath.get(entry.fullPath)!;
          const data = textEncoder.encode(entry.content);
          await saveFileData(existing.id, data);
          dispatch({
            type: "UPDATE_DATA",
            fileId: existing.id,
            data,
          });
          if (!firstNewId) firstNewId = existing.id;
        }

        // Create new files
        for (const entry of newEntries) {
          const data = textEncoder.encode(entry.content);
          const file = await createFile(projectName, entry.fullPath, data);
          dispatch({
            type: "ADD_FILE",
            file: { id: file.id, name: entry.fullPath, data },
          });
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
    uploadFiles: handleUploadFiles,
    reload,
    mergeVfsChanges: useCallback(
      (result: {
        addedFiles: WorkspaceFile[];
        modifiedFiles: { path: string; data: Uint8Array }[];
        deletedPaths: string[];
      }) => {
        dispatch({
          type: "MERGE_VFS",
          added: result.addedFiles,
          modified: result.modifiedFiles,
          deletedPaths: result.deletedPaths,
        });
      },
      []
    ),
  };
}
