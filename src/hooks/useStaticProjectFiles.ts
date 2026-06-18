import { useState, useCallback, useMemo, useReducer, useEffect } from "react";
import { useRef } from "react";
import { generateDuplicateName } from "./useProjectFiles";
import type { WorkspaceFile, UseProjectFilesResult } from "./useProjectFiles";
import { unzipToFiles } from "../vfs/unzipToFiles";

const textEncoder = new TextEncoder();

/**
 * Optional manifest baked into the project bundle. When present (as
 * `numbl-project.json` at the bundle root) it controls the entry file and
 * site title. The manifest itself is hidden from the file browser.
 */
interface ProjectManifest {
  entry?: string;
  title?: string;
  repository?: string;
}

const MANIFEST_NAME = "numbl-project.json";

/** Build id injected by `build-site` (a hash of the bundle). Used to
 *  cache-bust the project.zip fetch per build, without busting on every
 *  reload. Undefined in dev / older bundles. */
function getBuildId(): string | undefined {
  return typeof window !== "undefined"
    ? (window as unknown as { __NUMBL_BUILD_ID__?: string }).__NUMBL_BUILD_ID__
    : undefined;
}

/** Resolve the deploy base path (where `project.zip` lives). */
function getSiteBase(): string {
  // The build-site CLI injects an absolute base for GitHub Pages project
  // sites (served under /<repo>/). Falls back to Vite's BASE_URL ("./" for
  // the relative-base build, "/" in dev), which resolves against the
  // single-page document root.
  const injected =
    typeof window !== "undefined"
      ? (window as unknown as { __NUMBL_BASE__?: string }).__NUMBL_BASE__
      : undefined;
  const base = injected || import.meta.env.BASE_URL || "/";
  return base.endsWith("/") ? base : base + "/";
}

type FilesAction =
  | { type: "SET_FILES"; files: WorkspaceFile[] }
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

/** Choose a sensible landing file: manifest entry, else README, else a script. */
function pickActiveFile(files: WorkspaceFile[], entry?: string): string {
  if (entry) {
    const m = files.find(f => f.name === entry);
    if (m) return m.id;
  }
  const readme = files.find(f => /(^|\/)README\.md$/i.test(f.name));
  if (readme) return readme.id;
  const main = files.find(f => /(^|\/)main\.m$/i.test(f.name));
  if (main) return main.id;
  const firstScript = files.find(f => f.name.endsWith(".m"));
  if (firstScript) return firstScript.id;
  return files.length > 0 ? files[0].id : "";
}

export interface UseStaticProjectFilesResult extends UseProjectFilesResult {
  /** Title from the bundle manifest, if any. */
  title: string | null;
  /** Source repository URL from the bundle manifest, if any. */
  repository: string | null;
  /** True if the bundle failed to load. */
  loadError: string | null;
}

/**
 * File source for a statically-deployed numbl project. On mount it fetches
 * `project.zip` from the deploy base, unzips it (binary-safe) into an
 * in-memory content map, and exposes the standard project-files interface.
 *
 * Edits live in memory for the session and reset on reload — the deployed
 * bundle is the source of truth. Mirrors useShareProjectFiles, but loads from
 * a baked zip rather than the URL hash so binary files survive intact.
 */
export function useStaticProjectFiles(): UseStaticProjectFilesResult {
  const [files, dispatch] = useReducer(filesReducer, []);
  const [activeFileId, setActiveFileId] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [title, setTitle] = useState<string | null>(null);
  const [repository, setRepository] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const contentMapRef = useRef(new Map<string, Uint8Array>());

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const buildId = getBuildId();
        const zipUrl =
          getSiteBase() +
          "project.zip" +
          (buildId ? `?v=${encodeURIComponent(buildId)}` : "");
        const resp = await fetch(zipUrl);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const buf = new Uint8Array(await resp.arrayBuffer());
        if (cancelled) return;

        const extracted = unzipToFiles(buf);

        // Pull out the optional manifest before building the file list.
        let manifest: ProjectManifest = {};
        const manifestEntry = extracted.find(f => f.path === MANIFEST_NAME);
        if (manifestEntry) {
          try {
            manifest = JSON.parse(
              new TextDecoder().decode(manifestEntry.content)
            );
          } catch {
            // Malformed manifest is non-fatal; ignore it.
          }
        }

        const wsFiles: WorkspaceFile[] = [];
        const contentMap = new Map<string, Uint8Array>();
        for (const f of extracted) {
          if (f.path === MANIFEST_NAME) continue;
          const id = crypto.randomUUID();
          wsFiles.push({ id, name: f.path });
          contentMap.set(id, f.content);
        }
        wsFiles.sort((a, b) => a.name.localeCompare(b.name));

        contentMapRef.current = contentMap;
        dispatch({ type: "SET_FILES", files: wsFiles });
        setActiveFileId(pickActiveFile(wsFiles, manifest.entry));
        setTitle(manifest.title ?? null);
        setRepository(manifest.repository ?? null);
      } catch (e) {
        if (cancelled) return;
        console.error("Failed to load project bundle:", e);
        setLoadError(e instanceof Error ? e.message : "Failed to load project");
        const id = crypto.randomUUID();
        contentMapRef.current.set(
          id,
          textEncoder.encode("% Failed to load project bundle\n")
        );
        dispatch({ type: "SET_FILES", files: [{ id, name: "script.m" }] });
        setActiveFileId(id);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const updateFileContent = useCallback(
    (content: string) => {
      if (!activeFileId) return;
      contentMapRef.current.set(activeFileId, textEncoder.encode(content));
    },
    [activeFileId]
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

  const handleDeleteFile = useCallback(
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

  const handleDeleteFolder = useCallback(
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

  const handleDuplicateFile = useCallback(
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
    // No-op for static mode.
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
    contentCache: contentMapRef,
    title,
    repository,
    loadError,
    mergeVfsChanges: useCallback(() => {
      // No-op for static mode.
    }, []),
  };
}
