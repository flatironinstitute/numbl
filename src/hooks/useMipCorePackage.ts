import { useEffect } from "react";
import { db } from "../db/schema.js";
import {
  clearSystemFiles,
  ensureSystemProject,
  isMipCoreInstalled,
  SYSTEM_PROJECT_NAME,
} from "../db/operations.js";
import { fetchMipCoreFiles } from "./fetchMipCoreFiles.js";

// The system directory (mip core + any packages installed at runtime by
// `mip load --install`) lives in IndexedDB and persists across reloads. It is
// wiped completely after a period of inactivity so installs don't accumulate
// forever; mip core is then reinstalled on demand whenever it's found missing.
const SYSTEM_INACTIVITY_MS = 10 * 60 * 1000;
const SYSTEM_LAST_ACTIVITY_KEY = "numbl:systemLastActivity";

/** Record system-directory activity, resetting the inactivity-wipe timer. Call
 *  this whenever the system files are used (e.g. on each run) so the directory
 *  isn't wiped out from under an actively-used session. */
export function markSystemActivity(): void {
  try {
    localStorage.setItem(SYSTEM_LAST_ACTIVITY_KEY, String(Date.now()));
  } catch {
    // localStorage may be unavailable (private mode, etc.) — ignore.
  }
}

function lastSystemActivity(): number {
  try {
    const raw = localStorage.getItem(SYSTEM_LAST_ACTIVITY_KEY);
    const n = raw ? parseInt(raw, 10) : 0;
    return Number.isFinite(n) ? n : 0;
  } catch {
    return 0;
  }
}

/**
 * On mount, bring the system directory to a usable state, simply:
 *   1. If it's been inactive longer than SYSTEM_INACTIVITY_MS, wipe the whole
 *      system directory (mip core + any installed packages).
 *   2. If mip core is missing (first load, or right after a wipe), fetch and
 *      install it into IndexedDB.
 * Anything already present — including packages installed by earlier runs — is
 * left in place, so it persists across reloads. Calls onReady() when done so
 * the caller can (re)load the system files.
 */
export function useMipCorePackage(onReady: () => void): void {
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        if (Date.now() - lastSystemActivity() > SYSTEM_INACTIVITY_MS) {
          await clearSystemFiles();
        }
        markSystemActivity();
        if (cancelled) return;

        await ensureSystemProject();

        // Reinstall mip core only when it's missing.
        if (!(await isMipCoreInstalled())) {
          const vfsFiles = await fetchMipCoreFiles();
          if (cancelled) return;
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
        }

        if (!cancelled) onReady();
      } catch (e) {
        console.warn("Failed to ensure mip core package:", e);
      }
    })();
    return () => {
      cancelled = true;
    };
    // onReady is stable (useCallback) from the caller
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
}
