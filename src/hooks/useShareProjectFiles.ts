import { useState, useMemo, useEffect } from "react";
import type { UseProjectFilesResult } from "./useProjectFiles";
import { debounce, type FileEntry } from "./projectFilesState";
import { useInMemoryProjectFiles } from "./useInMemoryProjectFiles";
import {
  encodeShareData,
  decodeShareData,
  shareDataToWorkspaceFiles,
} from "../utils/shareUrl";

const textEncoder = new TextEncoder();

/** In-memory project files loaded from, and mirrored to, the URL hash. */
export function useShareProjectFiles(): UseProjectFilesResult & {
  urlSizeTooLarge: boolean;
} {
  const [urlSizeTooLarge, setUrlSizeTooLarge] = useState(false);

  const debouncedUpdateUrl = useMemo(
    () =>
      debounce(
        (
          currentFiles: FileEntry[],
          contents: Map<string, Uint8Array>,
          currentActiveId: string
        ) => {
          try {
            const encoded = encodeShareData(
              currentFiles,
              contents,
              currentActiveId
            );
            const newUrl = `${window.location.pathname}#${encoded}`;
            setUrlSizeTooLarge(newUrl.length > 64000);
            window.history.replaceState(null, "", newUrl);
          } catch (e) {
            console.error("Failed to update share URL:", e);
          }
        },
        500
      ),
    []
  );

  const project = useInMemoryProjectFiles(debouncedUpdateUrl);
  const { files, activeFileId, loading, contentCache, setProject, setLoading } =
    project;

  useEffect(() => {
    const single = (text: string) => {
      const id = crypto.randomUUID();
      setProject(
        [{ id, name: "script.m" }],
        new Map([[id, textEncoder.encode(text)]]),
        id
      );
    };
    try {
      const hash = window.location.hash.slice(1);
      if (hash) {
        const data = decodeShareData(hash);
        const {
          files: wsFiles,
          contentMap,
          activeFileId: aId,
        } = shareDataToWorkspaceFiles(data);
        setProject(wsFiles, contentMap, aId);
      } else {
        single("% Write your script here\n");
      }
    } catch (e) {
      console.error("Failed to decode share URL:", e);
      single("% Failed to load shared project\n");
    } finally {
      setLoading(false);
    }
  }, [setProject, setLoading]);

  useEffect(() => {
    if (!loading && files.length > 0) {
      debouncedUpdateUrl(files, contentCache.current, activeFileId);
    }
  }, [files, activeFileId, loading, contentCache, debouncedUpdateUrl]);

  return { ...project, urlSizeTooLarge };
}
