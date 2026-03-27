import AddIcon from "@mui/icons-material/Add";
import CreateNewFolderIcon from "@mui/icons-material/CreateNewFolder";
import DeleteIcon from "@mui/icons-material/Delete";
import EditIcon from "@mui/icons-material/Edit";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import FolderIcon from "@mui/icons-material/Folder";
import FolderOpenIcon from "@mui/icons-material/FolderOpen";
import InsertDriveFileIcon from "@mui/icons-material/InsertDriveFile";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import { Box, IconButton, Menu, MenuItem, Typography } from "@mui/material";
import { useCallback, useEffect, useRef, useState } from "react";

// --- Types ---

export interface WorkspaceFile {
  id: string;
  name: string; // may include '/' for folder paths, e.g. "src/utils/helper.m"
  data: Uint8Array;
}

interface TreeNode {
  name: string; // segment name (e.g. "src" or "helper.m")
  path: string; // full path (e.g. "src/utils/helper.m")
  isFolder: boolean;
  children: TreeNode[];
  file?: WorkspaceFile; // only for leaf files
}

interface FileBrowserProps {
  files: WorkspaceFile[];
  activeFileId: string;
  onSelectFile: (fileId: string) => void;
  onAddFile: (folderPath?: string) => void;
  onAddFolder: (parentPath?: string) => void;
  onDeleteFile: (fileId: string) => void;
  onRenameFile: (fileId: string, newName: string) => void;
  onDeleteFolder: (folderPath: string) => void;
  onRenameFolder: (oldPath: string, newName: string) => void;
  onMoveFile: (fileId: string, targetFolder: string | null) => void;
  onUploadFiles: (
    files: { path: string; content: string }[],
    targetFolder?: string
  ) => void;
  fileCount: number;
  triggerRenameId?: string; // Set to file ID or "folder:path" to auto-trigger rename
}

// --- Tree building ---

function buildTree(files: WorkspaceFile[]): TreeNode[] {
  const root: TreeNode[] = [];

  for (const file of files) {
    const parts = file.name.split("/");
    let currentLevel = root;

    for (let i = 0; i < parts.length; i++) {
      const segment = parts[i];
      const isLast = i === parts.length - 1;
      const pathSoFar = parts.slice(0, i + 1).join("/");

      if (isLast) {
        // It's a file
        currentLevel.push({
          name: segment,
          path: pathSoFar,
          isFolder: false,
          children: [],
          file,
        });
      } else {
        // It's a folder segment
        let folder = currentLevel.find(n => n.isFolder && n.name === segment);
        if (!folder) {
          folder = {
            name: segment,
            path: pathSoFar,
            isFolder: true,
            children: [],
          };
          currentLevel.push(folder);
        }
        currentLevel = folder.children;
      }
    }
  }

  // Sort: folders first, then files, both alphabetically
  const sortNodes = (nodes: TreeNode[]) => {
    nodes.sort((a, b) => {
      if (a.isFolder !== b.isFolder) return a.isFolder ? -1 : 1;
      return a.name.localeCompare(b.name);
    });
    for (const node of nodes) {
      if (node.isFolder) sortNodes(node.children);
    }
  };
  sortNodes(root);

  return root;
}

// --- Inline rename input ---

function InlineInput({
  initialValue,
  onCommit,
  onCancel,
}: {
  initialValue: string;
  onCommit: (value: string) => void;
  onCancel: () => void;
}) {
  const [value, setValue] = useState(initialValue);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
    inputRef.current?.select();
  }, []);

  return (
    <input
      ref={inputRef}
      value={value}
      onChange={e => setValue(e.target.value)}
      onBlur={() => onCommit(value)}
      onKeyDown={e => {
        if (e.key === "Enter") onCommit(value);
        if (e.key === "Escape") onCancel();
      }}
      onClick={e => e.stopPropagation()}
      style={{
        background: "#3c3c3c",
        border: "1px solid #007acc",
        color: "#cccccc",
        fontSize: "13px",
        padding: "1px 4px",
        width: "100%",
        outline: "none",
        fontFamily: "inherit",
      }}
    />
  );
}

// --- Tree node component ---

function TreeNodeItem({
  node,
  depth,
  activeFileId,
  expandedFolders,
  renamingPath,
  dragOverFolder,
  onToggleFolder,
  onSelectFile,
  onStartRename,
  onCommitRename,
  onCancelRename,
  onDelete,
  onAddFile,
  onAddFolder,
  onDeleteFolder,
  onDragStart,
  onDragOver,
  onDragLeave,
  onDrop,
  onUploadToFolder,
  fileCount,
}: {
  node: TreeNode;
  depth: number;
  activeFileId: string;
  expandedFolders: Set<string>;
  renamingPath: string | null;
  dragOverFolder: string | null;
  onToggleFolder: (path: string) => void;
  onSelectFile: (fileId: string) => void;
  onStartRename: (path: string, currentName: string, isFolder: boolean) => void;
  onCommitRename: (value: string) => void;
  onCancelRename: () => void;
  onDelete: (fileId: string) => void;
  onAddFile: (folderPath?: string) => void;
  onAddFolder: (parentPath?: string) => void;
  onDeleteFolder: (folderPath: string) => void;
  onDragStart: (e: React.DragEvent, fileId: string) => void;
  onDragOver: (e: React.DragEvent, folderPath: string | null) => void;
  onDragLeave: (e: React.DragEvent) => void;
  onDrop: (e: React.DragEvent, folderPath: string | null) => void;
  onUploadToFolder: (folderPath: string) => void;
  fileCount: number;
}) {
  const isExpanded = expandedFolders.has(node.path);
  const isActive = !node.isFolder && node.file?.id === activeFileId;
  const isRenaming =
    renamingPath === (node.isFolder ? `folder:${node.path}` : node.path);
  const isDragOver = node.isFolder && dragOverFolder === node.path;
  const indent = depth * 16;

  if (node.isFolder) {
    return (
      <>
        <Box
          onClick={() => onToggleFolder(node.path)}
          onDragOver={e => onDragOver(e, node.path)}
          onDragLeave={onDragLeave}
          onDrop={e => onDrop(e, node.path)}
          sx={{
            display: "flex",
            alignItems: "center",
            pl: `${indent + 4}px`,
            pr: "4px",
            py: "2px",
            cursor: "pointer",
            userSelect: "none",
            bgcolor: isDragOver ? "#094771" : "transparent",
            "&:hover": { bgcolor: isDragOver ? "#094771" : "#2a2d2e" },
            "&:hover .folder-actions": { opacity: 1 },
          }}
        >
          {isExpanded ? (
            <ExpandMoreIcon sx={{ fontSize: 16, color: "#cccccc", mr: 0.25 }} />
          ) : (
            <ChevronRightIcon
              sx={{ fontSize: 16, color: "#cccccc", mr: 0.25 }}
            />
          )}
          {isExpanded ? (
            <FolderOpenIcon sx={{ fontSize: 16, color: "#dcb67a", mr: 0.75 }} />
          ) : (
            <FolderIcon sx={{ fontSize: 16, color: "#dcb67a", mr: 0.75 }} />
          )}
          {isRenaming ? (
            <Box sx={{ flex: 1 }}>
              <InlineInput
                initialValue={node.name}
                onCommit={onCommitRename}
                onCancel={onCancelRename}
              />
            </Box>
          ) : (
            <>
              <Typography
                sx={{
                  fontSize: "13px",
                  color: "#cccccc",
                  flex: 1,
                  whiteSpace: "nowrap",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  fontWeight: 600,
                }}
              >
                {node.name}
              </Typography>
              <Box
                className="folder-actions"
                sx={{ display: "flex", gap: 0, opacity: 0, ml: 0.5 }}
              >
                <IconButton
                  size="small"
                  onClick={e => {
                    e.stopPropagation();
                    onAddFile(node.path);
                  }}
                  title="New file"
                  sx={{
                    color: "#bbbbbb",
                    "&:hover": { color: "#ffffff" },
                    p: 0.25,
                  }}
                >
                  <AddIcon sx={{ fontSize: 14 }} />
                </IconButton>
                <IconButton
                  size="small"
                  onClick={e => {
                    e.stopPropagation();
                    onAddFolder(node.path);
                  }}
                  title="New folder"
                  sx={{
                    color: "#bbbbbb",
                    "&:hover": { color: "#ffffff" },
                    p: 0.25,
                  }}
                >
                  <CreateNewFolderIcon sx={{ fontSize: 14 }} />
                </IconButton>
                <IconButton
                  size="small"
                  onClick={e => {
                    e.stopPropagation();
                    onStartRename(`folder:${node.path}`, node.name, true);
                  }}
                  title="Rename folder"
                  sx={{
                    color: "#bbbbbb",
                    "&:hover": { color: "#ffffff" },
                    p: 0.25,
                  }}
                >
                  <EditIcon sx={{ fontSize: 14 }} />
                </IconButton>
                <IconButton
                  size="small"
                  onClick={e => {
                    e.stopPropagation();
                    onUploadToFolder(node.path);
                  }}
                  title="Upload files"
                  sx={{
                    color: "#bbbbbb",
                    "&:hover": { color: "#ffffff" },
                    p: 0.25,
                  }}
                >
                  <UploadFileIcon sx={{ fontSize: 14 }} />
                </IconButton>
                <IconButton
                  size="small"
                  onClick={e => {
                    e.stopPropagation();
                    onDeleteFolder(node.path);
                  }}
                  title="Delete folder"
                  sx={{
                    color: "#bbbbbb",
                    "&:hover": { color: "#f48771" },
                    p: 0.25,
                  }}
                >
                  <DeleteIcon sx={{ fontSize: 14 }} />
                </IconButton>
              </Box>
            </>
          )}
        </Box>
        {isExpanded &&
          node.children.map(child => (
            <TreeNodeItem
              key={
                child.isFolder
                  ? `folder:${child.path}`
                  : (child.file?.id ?? child.path)
              }
              node={child}
              depth={depth + 1}
              activeFileId={activeFileId}
              expandedFolders={expandedFolders}
              renamingPath={renamingPath}
              dragOverFolder={dragOverFolder}
              onToggleFolder={onToggleFolder}
              onSelectFile={onSelectFile}
              onStartRename={onStartRename}
              onCommitRename={onCommitRename}
              onCancelRename={onCancelRename}
              onDelete={onDelete}
              onAddFile={onAddFile}
              onAddFolder={onAddFolder}
              onDeleteFolder={onDeleteFolder}
              onDragStart={onDragStart}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              onDrop={onDrop}
              onUploadToFolder={onUploadToFolder}
              fileCount={fileCount}
            />
          ))}
      </>
    );
  }

  // File node
  return (
    <Box
      draggable
      onDragStart={e => {
        if (node.file) onDragStart(e, node.file.id);
      }}
      onClick={() => {
        if (node.file) onSelectFile(node.file.id);
      }}
      sx={{
        display: "flex",
        alignItems: "center",
        pl: `${indent + 24}px`,
        pr: "4px",
        py: "2px",
        cursor: "grab",
        userSelect: "none",
        bgcolor: isActive ? "#37373d" : "transparent",
        "&:hover": { bgcolor: isActive ? "#37373d" : "#2a2d2e" },
        "&:hover .file-actions": { opacity: 1 },
      }}
    >
      <InsertDriveFileIcon sx={{ fontSize: 16, color: "#519aba", mr: 0.75 }} />
      {isRenaming ? (
        <Box sx={{ flex: 1 }}>
          <InlineInput
            initialValue={node.name}
            onCommit={onCommitRename}
            onCancel={onCancelRename}
          />
        </Box>
      ) : (
        <>
          <Typography
            sx={{
              fontSize: "13px",
              color: isActive ? "#ffffff" : "#cccccc",
              flex: 1,
              whiteSpace: "nowrap",
              overflow: "hidden",
              textOverflow: "ellipsis",
            }}
          >
            {node.name}
          </Typography>
          <Box
            className="file-actions"
            sx={{ display: "flex", gap: 0, opacity: 0, ml: 0.5 }}
          >
            <IconButton
              size="small"
              onClick={e => {
                e.stopPropagation();
                if (node.file) onStartRename(node.path, node.name, false);
              }}
              title="Rename"
              sx={{
                color: "#bbbbbb",
                "&:hover": { color: "#ffffff" },
                p: 0.25,
              }}
            >
              <EditIcon sx={{ fontSize: 14 }} />
            </IconButton>
            {fileCount > 1 && (
              <IconButton
                size="small"
                onClick={e => {
                  e.stopPropagation();
                  if (node.file) onDelete(node.file.id);
                }}
                title="Delete"
                sx={{
                  color: "#bbbbbb",
                  "&:hover": { color: "#f48771" },
                  p: 0.25,
                }}
              >
                <DeleteIcon sx={{ fontSize: 14 }} />
              </IconButton>
            )}
          </Box>
        </>
      )}
    </Box>
  );
}

// --- Main FileBrowser component ---

// Read File objects into path/content entries, filtering to .m files only
async function readFiles(
  files: File[]
): Promise<{ path: string; content: string }[]> {
  const mFiles = files.filter(f => f.name.endsWith(".m") && f.size > 0);
  const results: { path: string; content: string }[] = [];
  for (const f of mFiles) {
    try {
      const content = await f.text();
      // For webkitdirectory uploads, webkitRelativePath includes the
      // top-level folder; for plain file uploads, use just the name
      const path = f.webkitRelativePath || f.name;
      results.push({ path, content });
    } catch (err) {
      console.warn("Failed to read file:", f.name, err);
    }
  }
  return results;
}

export function FileBrowser({
  files,
  activeFileId,
  onSelectFile,
  onAddFile,
  onAddFolder,
  onDeleteFile,
  onRenameFile,
  onDeleteFolder,
  onRenameFolder,
  onMoveFile,
  onUploadFiles,
  fileCount,
  triggerRenameId,
}: FileBrowserProps) {
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(() => {
    // Only expand top-level folders that have fewer than 4 direct children
    const folders = new Set<string>();
    const topLevelChildren: Record<string, Set<string>> = {};
    for (const file of files) {
      const parts = file.name.split("/");
      if (parts.length > 1) {
        const topFolder = parts[0];
        if (!topLevelChildren[topFolder]) {
          topLevelChildren[topFolder] = new Set();
        }
        topLevelChildren[topFolder].add(parts[1]);
      }
    }
    for (const [folder, children] of Object.entries(topLevelChildren)) {
      if (children.size < 4) {
        folders.add(folder);
      }
    }
    return folders;
  });

  const [renamingPath, setRenamingPath] = useState<string | null>(null);
  const [renamingIsFolder, setRenamingIsFolder] = useState(false);
  const [renamingOriginalPath, setRenamingOriginalPath] = useState("");

  // Upload state
  const [uploadMenuAnchor, setUploadMenuAnchor] = useState<HTMLElement | null>(
    null
  );
  const [uploadTargetFolder, setUploadTargetFolder] = useState<
    string | undefined
  >();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

  const handleUploadClick = useCallback(
    (e: React.MouseEvent<HTMLElement>, targetFolder?: string) => {
      setUploadMenuAnchor(e.currentTarget);
      setUploadTargetFolder(targetFolder);
    },
    []
  );

  const handleFileInputChange = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const fileList = e.target.files;
      if (!fileList || fileList.length === 0) return;
      const entries = await readFiles(Array.from(fileList));
      if (entries.length > 0) {
        onUploadFiles(entries, uploadTargetFolder);
      }
      e.target.value = "";
    },
    [onUploadFiles, uploadTargetFolder]
  );

  const handleFolderInputChange = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const fileList = e.target.files;
      if (!fileList || fileList.length === 0) return;
      const entries = await readFiles(Array.from(fileList));
      if (entries.length > 0) {
        onUploadFiles(entries, uploadTargetFolder);
      }
      e.target.value = "";
    },
    [onUploadFiles, uploadTargetFolder]
  );

  // Wrapped handlers that also expand the target folder
  const handleAddFile = useCallback(
    (folderPath?: string) => {
      if (folderPath) {
        setExpandedFolders(prev => {
          const next = new Set(prev);
          const parts = folderPath.split("/");
          for (let i = 1; i <= parts.length; i++) {
            next.add(parts.slice(0, i).join("/"));
          }
          return next;
        });
      }
      onAddFile(folderPath);
    },
    [onAddFile]
  );

  const handleAddFolder = useCallback(
    (parentPath?: string) => {
      if (parentPath) {
        setExpandedFolders(prev => {
          const next = new Set(prev);
          const parts = parentPath.split("/");
          for (let i = 1; i <= parts.length; i++) {
            next.add(parts.slice(0, i).join("/"));
          }
          return next;
        });
      }
      onAddFolder(parentPath);
    },
    [onAddFolder]
  );

  // --- Drag and drop state ---
  const [dragOverFolder, setDragOverFolder] = useState<string | null>(null);
  const draggingFileIdRef = useRef<string | null>(null);

  const handleDragStart = useCallback((e: React.DragEvent, fileId: string) => {
    draggingFileIdRef.current = fileId;
    e.dataTransfer.effectAllowed = "move";
    e.dataTransfer.setData("text/plain", fileId);
  }, []);

  const handleDragOver = useCallback(
    (e: React.DragEvent, folderPath: string | null) => {
      e.preventDefault();
      e.stopPropagation();
      e.dataTransfer.dropEffect = "move";
      setDragOverFolder(folderPath);
    },
    []
  );

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    // Only clear if leaving the element (not entering a child)
    const relatedTarget = e.relatedTarget as HTMLElement | null;
    if (!relatedTarget || !e.currentTarget.contains(relatedTarget)) {
      setDragOverFolder(null);
    }
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent, folderPath: string | null) => {
      e.preventDefault();
      e.stopPropagation();
      setDragOverFolder(null);
      const fileId = draggingFileIdRef.current;
      if (fileId) {
        onMoveFile(fileId, folderPath);
        draggingFileIdRef.current = null;
      }
    },
    [onMoveFile]
  );

  const tree = buildTree(files);

  const toggleFolder = useCallback((path: string) => {
    setExpandedFolders(prev => {
      const next = new Set(prev);
      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
      }
      return next;
    });
  }, []);

  const startRename = useCallback(
    (_path: string, _currentName: string, isFolder: boolean) => {
      const actualPath = isFolder ? _path.replace("folder:", "") : _path;
      setRenamingPath(_path);
      setRenamingIsFolder(isFolder);
      setRenamingOriginalPath(actualPath);
    },
    []
  );

  const commitRename = useCallback(
    (newName: string) => {
      const trimmed = newName.trim();
      if (!trimmed) {
        setRenamingPath(null);
        return;
      }

      if (renamingIsFolder) {
        onRenameFolder(renamingOriginalPath, trimmed);
      } else {
        // Find the file by path
        const file = files.find(f => f.name === renamingOriginalPath);
        if (file) {
          // Compute new full path
          const parts = renamingOriginalPath.split("/");
          parts[parts.length - 1] = trimmed;
          const newFullName = parts.join("/");
          onRenameFile(file.id, newFullName);
        }
      }
      setRenamingPath(null);
    },
    [
      renamingIsFolder,
      renamingOriginalPath,
      files,
      onRenameFile,
      onRenameFolder,
    ]
  );

  const cancelRename = useCallback(() => {
    setRenamingPath(null);
  }, []);

  // Handle F2 key to rename active file
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "F2" && activeFileId && !renamingPath) {
        e.preventDefault();
        const activeFile = files.find(f => f.id === activeFileId);
        if (activeFile) {
          startRename(
            activeFile.name,
            activeFile.name.split("/").pop() || "",
            false
          );
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [activeFileId, files, renamingPath, startRename]);

  // Handle auto-trigger rename when new file/folder is created
  useEffect(() => {
    if (triggerRenameId && !renamingPath) {
      if (triggerRenameId.startsWith("folder:")) {
        const folderPath = triggerRenameId.substring(7);
        const folderName = folderPath.split("/").pop() || "";
        // eslint-disable-next-line react-hooks/set-state-in-effect
        startRename(triggerRenameId, folderName, true);
      } else {
        const file = files.find(f => f.id === triggerRenameId);
        if (file) {
          const fileName = file.name.split("/").pop() || "";
          startRename(file.name, fileName, false);
        }
      }
    }
  }, [triggerRenameId, files, renamingPath, startRename]);

  return (
    <Box
      sx={{
        bgcolor: "#252526",
        borderRight: "1px solid #3c3c3c",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
        height: "100%",
      }}
    >
      <Box
        sx={{
          p: 1,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          borderBottom: "1px solid #3c3c3c",
        }}
      >
        <Typography
          variant="caption"
          sx={{
            color: "#bbbbbb",
            textTransform: "uppercase",
            fontWeight: 600,
            letterSpacing: "0.5px",
            fontSize: "11px",
          }}
        >
          Explorer
        </Typography>
        <Box sx={{ display: "flex", gap: 0 }}>
          <IconButton
            size="small"
            onClick={() => handleAddFile()}
            title="New file"
            sx={{ color: "#bbbbbb", "&:hover": { color: "#ffffff" }, p: 0.5 }}
          >
            <AddIcon fontSize="small" />
          </IconButton>
          <IconButton
            size="small"
            onClick={() => handleAddFolder()}
            title="New folder"
            sx={{ color: "#bbbbbb", "&:hover": { color: "#ffffff" }, p: 0.5 }}
          >
            <CreateNewFolderIcon fontSize="small" />
          </IconButton>
          <IconButton
            size="small"
            onClick={e => handleUploadClick(e)}
            title="Upload files"
            sx={{ color: "#bbbbbb", "&:hover": { color: "#ffffff" }, p: 0.5 }}
          >
            <UploadFileIcon fontSize="small" />
          </IconButton>
        </Box>
      </Box>
      <Box
        sx={{
          flex: 1,
          overflow: "auto",
          py: 0.5,
          bgcolor: dragOverFolder === "__root__" ? "#094771" : "transparent",
        }}
        onDragOver={e => {
          e.preventDefault();
          e.stopPropagation();
          e.dataTransfer.dropEffect = "move";
          setDragOverFolder("__root__");
        }}
        onDragLeave={e => {
          const rt = e.relatedTarget as HTMLElement | null;
          if (!rt || !e.currentTarget.contains(rt)) setDragOverFolder(null);
        }}
        onDrop={e => handleDrop(e, null)}
      >
        {tree.map(node => (
          <TreeNodeItem
            key={
              node.isFolder
                ? `folder:${node.path}`
                : (node.file?.id ?? node.path)
            }
            node={node}
            depth={0}
            activeFileId={activeFileId}
            expandedFolders={expandedFolders}
            renamingPath={renamingPath}
            dragOverFolder={dragOverFolder}
            onToggleFolder={toggleFolder}
            onSelectFile={onSelectFile}
            onStartRename={startRename}
            onCommitRename={commitRename}
            onCancelRename={cancelRename}
            onDelete={onDeleteFile}
            onAddFile={handleAddFile}
            onAddFolder={handleAddFolder}
            onDeleteFolder={onDeleteFolder}
            onDragStart={handleDragStart}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onUploadToFolder={folderPath => {
              setUploadTargetFolder(folderPath);
              fileInputRef.current?.click();
            }}
            fileCount={fileCount}
          />
        ))}
      </Box>
      <Menu
        anchorEl={uploadMenuAnchor}
        open={Boolean(uploadMenuAnchor)}
        onClose={() => setUploadMenuAnchor(null)}
      >
        <MenuItem
          onClick={() => {
            setUploadMenuAnchor(null);
            fileInputRef.current?.click();
          }}
        >
          Upload Files
        </MenuItem>
        <MenuItem
          onClick={() => {
            setUploadMenuAnchor(null);
            folderInputRef.current?.click();
          }}
        >
          Upload Folder
        </MenuItem>
      </Menu>
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".m"
        style={{ display: "none" }}
        onChange={handleFileInputChange}
      />
      <input
        ref={folderInputRef}
        type="file"
        // @ts-expect-error webkitdirectory is not in the type definitions
        webkitdirectory=""
        style={{ display: "none" }}
        onChange={handleFolderInputChange}
      />
    </Box>
  );
}
