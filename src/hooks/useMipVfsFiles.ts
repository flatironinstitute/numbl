import { useEffect, useState } from "react";
import { fetchMipCoreFiles, type VfsFile } from "./fetchMipCoreFiles.js";

/**
 * Lightweight hook that fetches mip core files into memory (no IndexedDB).
 * Returns VFS-ready files for passing to workers.
 */
export function useMipVfsFiles(): VfsFile[] {
  const [files, setFiles] = useState<VfsFile[]>([]);

  useEffect(() => {
    let cancelled = false;
    fetchMipCoreFiles().then(
      result => {
        if (!cancelled) setFiles(result);
      },
      err => console.warn("Failed to load mip core package:", err)
    );
    return () => {
      cancelled = true;
    };
  }, []);

  return files;
}
