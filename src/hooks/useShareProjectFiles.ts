import { useState, useCallback, useMemo, useReducer, useEffect } from "react";
import type { WorkspaceFile } from "./useProjectFiles";
import type { UseProjectFilesResult } from "./useProjectFiles";
import {
  encodeShareData,
  decodeShareData,
  shareDataToWorkspaceFiles,
} from "../utils/shareUrl";

const textEncoder = new TextEncoder();

// Reuse the same reducer actions
type FilesAction =
  | { type: "SET_FILES"; files: WorkspaceFile[] }
  | { type: "UPDATE_DATA"; fileId: string; data: Uint8Array }
  | { type: "ADD_FILE"; file: WorkspaceFile }
  | { type: "DELETE_FILE"; fileId: string }
  | { type: "RENAME_FILE"; fileId: string; newName: string }
  | { type: "RENAME_FOLDER"; oldPath: string; newPath: string }
  | { type: "MOVE_FILE"; fileId: string; newName: string };

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
    default:
      return state;
  }
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

export function useShareProjectFiles(): UseProjectFilesResult & {
  urlSizeTooLarge: boolean;
} {
  const [files, dispatch] = useReducer(filesReducer, []);
  const [activeFileId, setActiveFileId] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [urlSizeTooLarge, setUrlSizeTooLarge] = useState(false);

  // Load initial state from URL hash on mount
  useEffect(() => {
    try {
      const hash = window.location.hash.slice(1); // remove #
      if (hash) {
        const data = decodeShareData(hash);
        const { files: wsFiles, activeFileId: aId } =
          shareDataToWorkspaceFiles(data);
        dispatch({ type: "SET_FILES", files: wsFiles });
        setActiveFileId(aId);
      } else {
        // Empty share - create a default file
        const defaultFile: WorkspaceFile = {
          id: crypto.randomUUID(),
          name: "script.m",
          data: textEncoder.encode("% Write your script here\n"),
        };
        dispatch({ type: "SET_FILES", files: [defaultFile] });
        setActiveFileId(defaultFile.id);
      }
    } catch (e) {
      console.error("Failed to decode share URL:", e);
      const defaultFile: WorkspaceFile = {
        id: crypto.randomUUID(),
        name: "script.m",
        data: textEncoder.encode("% Failed to load shared project\n"),
      };
      dispatch({ type: "SET_FILES", files: [defaultFile] });
      setActiveFileId(defaultFile.id);
    } finally {
      setLoading(false);
    }
  }, []); // Only on mount

  // Debounced URL update
  const debouncedUpdateUrl = useMemo(
    () =>
      debounce((currentFiles: WorkspaceFile[], currentActiveId: string) => {
        try {
          const encoded = encodeShareData(currentFiles, currentActiveId);
          const newUrl = `${window.location.pathname}#${encoded}`;
          if (newUrl.length > 64000) {
            setUrlSizeTooLarge(true);
          } else {
            setUrlSizeTooLarge(false);
          }
          window.history.replaceState(null, "", newUrl);
        } catch (e) {
          console.error("Failed to update share URL:", e);
        }
      }, 500),
    []
  );

  // Sync state to URL whenever files or activeFileId change
  useEffect(() => {
    if (!loading && files.length > 0) {
      debouncedUpdateUrl(files, activeFileId);
    }
  }, [files, activeFileId, loading, debouncedUpdateUrl]);

  const updateFileContent = useCallback(
    (content: string) => {
      dispatch({
        type: "UPDATE_DATA",
        fileId: activeFileId,
        data: textEncoder.encode(content),
      });
    },
    [activeFileId]
  );

  const emptyData = useMemo(() => new Uint8Array(0), []);

  const addFile = useCallback(
    async (folderPath?: string): Promise<string> => {
      const baseName = generateUniqueName(files, folderPath);
      const name = folderPath ? `${folderPath}/${baseName}` : baseName;
      const id = crypto.randomUUID();
      dispatch({
        type: "ADD_FILE",
        file: { id, name, data: emptyData },
      });
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
      dispatch({
        type: "ADD_FILE",
        file: { id, name, data: emptyData },
      });
      setActiveFileId(id);
      return fullPath;
    },
    [files, emptyData]
  );

  const handleDeleteFile = useCallback(
    async (fileId: string) => {
      if (files.length === 1) {
        alert("Cannot delete the last file");
        return;
      }
      if (activeFileId === fileId) {
        const newFiles = files.filter(f => f.id !== fileId);
        if (newFiles.length > 0) setActiveFileId(newFiles[0].id);
      }
      dispatch({ type: "DELETE_FILE", fileId });
    },
    [files, activeFileId]
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

  const handleRenameFile = useCallback(
    async (fileId: string, newName: string) => {
      if (files.some(f => f.id !== fileId && f.name === newName)) {
        alert("A file with this name already exists");
        return;
      }
      dispatch({ type: "RENAME_FILE", fileId, newName });
    },
    [files]
  );

  const handleRenameFolder = useCallback(
    async (oldPath: string, newName: string) => {
      const parts = oldPath.split("/");
      parts[parts.length - 1] = newName;
      const newPath = parts.join("/");
      dispatch({ type: "RENAME_FOLDER", oldPath, newPath });
    },
    []
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
      dispatch({ type: "RENAME_FILE", fileId, newName });
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

      let firstNewId = "";
      for (const entry of duplicates) {
        const existing = existingByPath.get(entry.fullPath)!;
        dispatch({
          type: "UPDATE_DATA",
          fileId: existing.id,
          data: textEncoder.encode(entry.content),
        });
        if (!firstNewId) firstNewId = existing.id;
      }
      for (const entry of newEntries) {
        const id = crypto.randomUUID();
        dispatch({
          type: "ADD_FILE",
          file: {
            id,
            name: entry.fullPath,
            data: textEncoder.encode(entry.content),
          },
        });
        if (!firstNewId) firstNewId = id;
      }
      if (firstNewId) {
        setActiveFileId(firstNewId);
      }
    },
    [files]
  );

  const reload = useCallback(async () => {
    // No-op for share mode — state is in memory
  }, []);

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
    urlSizeTooLarge,
    mergeVfsChanges: useCallback(() => {
      // No-op for share mode
    }, []),
  };
}
