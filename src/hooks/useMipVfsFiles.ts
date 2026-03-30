import { useEffect, useMemo, useState } from "react";
import { fetchMipCoreFiles, type VfsFile } from "./fetchMipCoreFiles.js";

const textDecoder = new TextDecoder();

interface MipFiles {
  /** VFS-format files (path: /home/..., content: Uint8Array) for worker VFS. */
  vfsFiles: VfsFile[];
  /** WorkspaceFile-format entries (name: ~/..., source: string) for function resolution. */
  workspaceFiles: { name: string; source: string }[];
}

/**
 * Lightweight hook that fetches mip core files into memory (no IndexedDB).
 * Returns both VFS-ready files and workspace files for function resolution.
 */
export function useMipVfsFiles(): MipFiles {
  const [vfsFiles, setVfsFiles] = useState<VfsFile[]>([]);

  useEffect(() => {
    let cancelled = false;
    fetchMipCoreFiles().then(
      result => {
        if (!cancelled) setVfsFiles(result);
      },
      err => console.warn("Failed to load mip core package:", err)
    );
    return () => {
      cancelled = true;
    };
  }, []);

  const workspaceFiles = useMemo(
    () =>
      vfsFiles
        .filter(f => f.path.endsWith(".m"))
        .map(f => ({
          // Convert /home/... to ~/... for search path matching
          name: "~/" + f.path.replace(/^\/home\//, ""),
          source: textDecoder.decode(f.content),
        })),
    [vfsFiles]
  );

  return { vfsFiles, workspaceFiles };
}
