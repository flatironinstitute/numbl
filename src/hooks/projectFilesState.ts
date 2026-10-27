/** Metadata-only file reference shown in the IDE (no content loaded). */
export interface FileEntry {
  id: string;
  name: string;
}

export type FilesAction =
  | { type: "SET_FILES"; files: FileEntry[] }
  | { type: "ADD_FILE"; file: FileEntry }
  | { type: "DELETE_FILE"; fileId: string }
  | { type: "RENAME_FILE"; fileId: string; newName: string }
  | { type: "RENAME_FOLDER"; oldPath: string; newPath: string }
  | {
      type: "MERGE_VFS";
      added: FileEntry[];
      deletedPaths: string[];
    };

export function filesReducer(
  state: FileEntry[],
  action: FilesAction
): FileEntry[] {
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

    case "MERGE_VFS": {
      let result = state;
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

/** First free `untitled<N>.m` base name in `folderPath`. */
export function generateUniqueName(
  files: FileEntry[],
  folderPath?: string
): string {
  const existing = new Set(files.map(f => f.name));
  for (let i = 1; ; i++) {
    const baseName = `untitled${i === 1 ? "" : i}.m`;
    const fullName = folderPath ? `${folderPath}/${baseName}` : baseName;
    if (!existing.has(fullName)) return baseName;
  }
}

/** First free `folder<N>` name directly under `parentPath`. */
export function generateUniqueFolderName(
  files: FileEntry[],
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

/** Full path for a copy of `sourcePath` (`name_copy.m`, `name_copy2.m`, ...). */
export function generateDuplicateName(
  files: FileEntry[],
  sourcePath: string
): string {
  const slash = sourcePath.lastIndexOf("/");
  const folder = slash >= 0 ? sourcePath.slice(0, slash) : "";
  const base = slash >= 0 ? sourcePath.slice(slash + 1) : sourcePath;
  const dot = base.lastIndexOf(".");
  const stem = dot > 0 ? base.slice(0, dot) : base;
  const ext = dot > 0 ? base.slice(dot) : "";
  const existing = new Set(files.map(f => f.name));
  for (let i = 1; ; i++) {
    const suffix = i === 1 ? "_copy" : `_copy${i}`;
    const candidate = `${stem}${suffix}${ext}`;
    const fullName = folder ? `${folder}/${candidate}` : candidate;
    if (!existing.has(fullName)) return fullName;
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function debounce<T extends (...args: any[]) => any>(
  fn: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timeoutId: ReturnType<typeof setTimeout>;
  return (...args: Parameters<T>) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn(...args), delay);
  };
}

/** Confirmation text for an upload that overwrites existing files. */
export function uploadOverwriteMessage(
  duplicatePaths: string[],
  newCount: number
): string {
  return (
    `The following ${duplicatePaths.length} file(s) already exist and will be overwritten:\n\n${duplicatePaths.join("\n")}\n\n` +
    (newCount > 0 ? `${newCount} new file(s) will also be added.\n\n` : "") +
    "Click OK to proceed, or Cancel to abort the upload."
  );
}
