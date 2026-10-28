import { useEffect, useState } from "react";
import type { FileEntry } from "../../hooks/projectFilesState";
import type { useSystemFiles } from "../../hooks/useSystemFiles";
import { FileBrowser } from "../FileBrowser";

/** File operations on the project (everything outside system/). */
export interface ProjectFileOps {
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
}

const isSystemPath = (name: string) =>
  name === "system" || name.startsWith("system/");

/**
 * File browser over project and system files, routing each operation to
 * the project or to the system directory, and starting a rename on files
 * and folders it creates.
 */
export function WorkspaceFileBrowser({
  files,
  activeFileId,
  onSelectFile,
  project,
  system,
}: {
  /** Project files followed by system files. */
  files: FileEntry[];
  activeFileId: string;
  onSelectFile: (fileId: string) => void;
  project: ProjectFileOps;
  system: ReturnType<typeof useSystemFiles>;
}) {
  const [triggerRenameId, setTriggerRenameId] = useState<string | undefined>();
  // Clear the trigger shortly after so the same id can trigger again.
  useEffect(() => {
    if (!triggerRenameId) return;
    const timer = setTimeout(() => setTriggerRenameId(undefined), 100);
    return () => clearTimeout(timer);
  }, [triggerRenameId]);

  const isSystemFileId = (fileId: string) =>
    system.systemFiles.some(f => f.id === fileId);

  return (
    <FileBrowser
      files={files}
      activeFileId={activeFileId}
      onSelectFile={onSelectFile}
      onAddFile={async folderPath => {
        const id =
          folderPath && isSystemPath(folderPath)
            ? await system.addSystemFile(folderPath)
            : await project.addFile(folderPath);
        if (id) setTriggerRenameId(id);
      }}
      onAddFolder={async parentPath => {
        const folderPath =
          parentPath && isSystemPath(parentPath)
            ? await system.addSystemFolder(parentPath)
            : await project.addFolder(parentPath);
        if (folderPath) setTriggerRenameId(`folder:${folderPath}`);
      }}
      onDeleteFile={fileId =>
        isSystemFileId(fileId)
          ? system.deleteSystemFile(fileId)
          : project.deleteFile(fileId)
      }
      onDeleteFolder={folderPath =>
        isSystemPath(folderPath)
          ? system.deleteSystemFolder(folderPath)
          : project.deleteFolder(folderPath)
      }
      onRenameFile={(fileId, newName) =>
        isSystemFileId(fileId)
          ? system.renameSystemFile(fileId, newName)
          : project.renameFile(fileId, newName)
      }
      onRenameFolder={(oldPath, newName) =>
        isSystemPath(oldPath)
          ? system.renameSystemFolder(oldPath, newName)
          : project.renameFolder(oldPath, newName)
      }
      onMoveFile={(fileId, targetFolder) =>
        isSystemFileId(fileId)
          ? system.moveSystemFile(fileId, targetFolder)
          : project.moveFile(fileId, targetFolder)
      }
      onDuplicateFile={async fileId => {
        const id = isSystemFileId(fileId)
          ? await system.duplicateSystemFile(fileId)
          : await project.duplicateFile(fileId);
        if (id) setTriggerRenameId(id);
      }}
      onUploadFiles={project.uploadFiles}
      fileCount={files.length}
      triggerRenameId={triggerRenameId}
    />
  );
}
