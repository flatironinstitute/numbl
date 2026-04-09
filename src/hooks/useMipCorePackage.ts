import { useEffect } from "react";
import { db } from "../db/schema.js";
import { ensureSystemProject, SYSTEM_PROJECT_NAME } from "../db/operations.js";
import { fetchMipCoreFiles } from "./fetchMipCoreFiles.js";

const MIP_SYSTEM_PREFIX = ".mip/packages/mip-org/core/mip/";

/**
 * On mount, fetches the mip core package, unzips it, and writes the files into
 * IndexedDB under the __system__ project at .mip/packages/mip-org/core/mip/,
 * overwriting any previous contents. Calls onInstalled() when done so the
 * caller can reload system files.
 */
export function useMipCorePackage(onInstalled: () => void): void {
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const vfsFiles = await fetchMipCoreFiles();
        if (cancelled) return;

        await ensureSystemProject();

        // Upsert mip files using deterministic IDs
        const now = Date.now();
        await db.transaction("rw", db.files, db.fileContents, async () => {
          const metaRecords = vfsFiles.map(f => {
            const path = f.path.replace(/^\/system\//, "");
            return {
              id: "mip:" + path,
              projectName: SYSTEM_PROJECT_NAME,
              path,
              createdAt: now,
              updatedAt: now,
            };
          });
          const contentRecords = vfsFiles.map(f => ({
            id: "mip:" + f.path.replace(/^\/system\//, ""),
            data: f.content,
          }));
          await db.files.bulkPut(metaRecords);
          await db.fileContents.bulkPut(contentRecords);
        });

        // Clean up stale mip files with old UUID keys
        const existingKeys = await db.files
          .where("[projectName+path]")
          .between(
            [SYSTEM_PROJECT_NAME, MIP_SYSTEM_PREFIX],
            [SYSTEM_PROJECT_NAME, MIP_SYSTEM_PREFIX + "\uffff"]
          )
          .primaryKeys();
        const staleKeys = existingKeys.filter(
          k => typeof k === "string" && !k.startsWith("mip:")
        );
        if (staleKeys.length > 0) {
          await db.transaction("rw", db.files, db.fileContents, async () => {
            await db.fileContents.bulkDelete(staleKeys);
            await db.files.bulkDelete(staleKeys);
          });
        }

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
