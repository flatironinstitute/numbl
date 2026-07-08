/**
 * Dependency-free IndexedDB store for the session's /system directory (mip
 * core plus packages installed at runtime by `mip load --install`), so
 * installs persist across page loads. Mirrors the IDE's __system__ project
 * (db/operations.ts) but is usable from a worker with no Dexie/React.
 *
 * Staleness policy matches the IDE's: the whole store is wiped after a
 * period of inactivity, and mip core is re-fetched on demand — packages are
 * never individually invalidated.
 */
import type { VfsChanges } from "../vfs/VirtualFileSystem.js";

const DB_NAME = "numbl-embed-system";
const DB_VERSION = 1;
const FILES = "files";
const META = "meta";
const LAST_ACTIVITY = "lastActivity";

export interface StoredFile {
  path: string;
  content: Uint8Array;
}

function requestAsPromise<T>(req: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function txDone(tx: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  });
}

export class SystemStore {
  private dbPromise: Promise<IDBDatabase>;

  constructor(dbName = DB_NAME) {
    const req = indexedDB.open(dbName, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(FILES)) {
        db.createObjectStore(FILES, { keyPath: "path" });
      }
      if (!db.objectStoreNames.contains(META)) {
        db.createObjectStore(META, { keyPath: "key" });
      }
    };
    this.dbPromise = requestAsPromise(
      req as IDBRequest<IDBDatabase>
    ) as Promise<IDBDatabase>;
  }

  /**
   * Load the stored /system files, first wiping the store if it has been
   * inactive longer than `inactivityMs`. Marks activity either way.
   */
  async loadValid(inactivityMs: number): Promise<StoredFile[]> {
    const last = await this.getMeta(LAST_ACTIVITY);
    if (typeof last === "number" && Date.now() - last > inactivityMs) {
      await this.clear();
    }
    await this.markActivity();
    const db = await this.dbPromise;
    const tx = db.transaction(FILES, "readonly");
    const all = await requestAsPromise(
      tx.objectStore(FILES).getAll() as IDBRequest<StoredFile[]>
    );
    return all;
  }

  /** Persist the /system subset of a run's VFS changes. */
  async applyChanges(changes: VfsChanges): Promise<void> {
    const written = [...changes.created, ...changes.modified].filter(f =>
      f.path.startsWith("/system/")
    );
    const deleted = changes.deleted.filter(p => p.startsWith("/system/"));
    if (written.length === 0 && deleted.length === 0) return;
    const db = await this.dbPromise;
    const tx = db.transaction(FILES, "readwrite");
    const store = tx.objectStore(FILES);
    for (const f of written) store.put({ path: f.path, content: f.content });
    for (const p of deleted) store.delete(p);
    await txDone(tx);
  }

  async markActivity(): Promise<void> {
    await this.setMeta(LAST_ACTIVITY, Date.now());
  }

  async clear(): Promise<void> {
    const db = await this.dbPromise;
    const tx = db.transaction([FILES, META], "readwrite");
    tx.objectStore(FILES).clear();
    tx.objectStore(META).clear();
    await txDone(tx);
  }

  private async getMeta(key: string): Promise<unknown> {
    const db = await this.dbPromise;
    const tx = db.transaction(META, "readonly");
    const row = await requestAsPromise(
      tx.objectStore(META).get(key) as IDBRequest<
        { key: string; value: unknown } | undefined
      >
    );
    return row?.value;
  }

  private async setMeta(key: string, value: unknown): Promise<void> {
    const db = await this.dbPromise;
    const tx = db.transaction(META, "readwrite");
    tx.objectStore(META).put({ key, value });
    await txDone(tx);
  }
}
