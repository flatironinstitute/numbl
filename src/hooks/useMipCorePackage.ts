import { useEffect } from "react";
import { db } from "../db/schema.js";
import { ensureHomeProject } from "../db/operations.js";
import { unzipToFiles } from "../vfs/unzipToFiles.js";

const MHL_URL =
  "https://github.com/mip-org/mip-core/releases/download/mip-main/mip-main-any.mhl";
const MIP_HOME_PREFIX = ".mip/packages/mip-org/core/mip/";
const HOME_PROJECT_NAME = "__home__";

function proxiedUrl(url: string): string {
  if (/^https:\/\/github\.com\/.+\/releases\/download\/.+/.test(url)) {
    return url.replace(
      "https://github.com/",
      "https://mip-cors-proxy.figurl.workers.dev/gh/"
    );
  }
  return url;
}

/**
 * On mount, fetches the mip core package, unzips it, and writes the files into
 * IndexedDB under the __home__ project at .mip/packages/mip-org/core/mip/,
 * overwriting any previous contents. Calls onInstalled() when done so the
 * caller can reload home files.
 */
export function useMipCorePackage(onInstalled: () => void): void {
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const resp = await fetch(proxiedUrl(MHL_URL));
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const buf = await resp.arrayBuffer();
        const extracted = unzipToFiles(new Uint8Array(buf));
        if (cancelled) return;

        await ensureHomeProject();

        // Delete existing mip package files
        const existing = await db.files
          .where("projectName")
          .equals(HOME_PROJECT_NAME)
          .filter(f => f.path.startsWith(MIP_HOME_PREFIX))
          .toArray();
        await db.files.bulkDelete(existing.map(f => f.id));

        // Write new files
        const now = Date.now();
        await db.files.bulkAdd(
          extracted.map(f => ({
            id: crypto.randomUUID(),
            projectName: HOME_PROJECT_NAME,
            path: MIP_HOME_PREFIX + f.path,
            data: f.content,
            createdAt: now,
            updatedAt: now,
          }))
        );

        if (!cancelled) {
          onInstalled();
        }
      } catch (e) {
        console.warn("Failed to load mip core package:", e);
      }
    })();
    return () => {
      cancelled = true;
    };
    // onInstalled is stable (useCallback) from the caller
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
}
