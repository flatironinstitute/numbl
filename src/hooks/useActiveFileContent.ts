import { useCallback, useEffect, useState } from "react";

/**
 * Content of the active file: null until it has loaded (or when no file is
 * active). `setContent` records edits so the view stays in sync without a
 * reload.
 */
export function useActiveFileContent(
  fileId: string,
  load: (fileId: string) => Promise<Uint8Array>
): [Uint8Array | null, (content: Uint8Array) => void] {
  const [loaded, setLoaded] = useState<{
    fileId: string;
    content: Uint8Array;
  } | null>(null);

  useEffect(() => {
    if (!fileId) return;
    let cancelled = false;
    load(fileId).then(content => {
      if (!cancelled) setLoaded({ fileId, content });
    });
    return () => {
      cancelled = true;
    };
  }, [fileId, load]);

  const setContent = useCallback(
    (content: Uint8Array) => setLoaded({ fileId, content }),
    [fileId]
  );

  return [loaded?.fileId === fileId ? loaded.content : null, setContent];
}
