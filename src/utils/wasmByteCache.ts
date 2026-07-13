/**
 * Minimal, dependency-free IndexedDB byte cache for downloaded wasm
 * binaries, keyed by URL. Used by the browser worker so the (cross-origin)
 * accelerator wasm isn't re-downloaded on every worker start.
 *
 * Invalidation is by key: the manifest is always fetched fresh and should
 * name a content-hashed wasm filename, so a new build ⇒ a new key ⇒ a
 * natural cache miss (stale entries are harmless and can be pruned later).
 *
 * All operations are best-effort: if IndexedDB is unavailable or errors,
 * `get` resolves to `null` and `put` resolves silently, so caching never
 * breaks loading.
 */

import type { WasmByteCache } from "../numbl-core/native/wasm-lapack-browser.js";

const DB_NAME = "numbl-wasm-cache";
const STORE = "bytes";

function openDb(): Promise<IDBDatabase | null> {
  return new Promise(resolve => {
    if (typeof indexedDB === "undefined") {
      resolve(null);
      return;
    }
    let req: IDBOpenDBRequest;
    try {
      req = indexedDB.open(DB_NAME, 1);
    } catch {
      resolve(null);
      return;
    }
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => resolve(null);
  });
}

export class IdbWasmCache implements WasmByteCache {
  async get(key: string): Promise<Uint8Array | null> {
    const db = await openDb();
    if (!db) return null;
    try {
      return await new Promise<Uint8Array | null>(resolve => {
        const tx = db.transaction(STORE, "readonly");
        const req = tx.objectStore(STORE).get(key);
        req.onsuccess = () => {
          const v = req.result;
          resolve(v instanceof Uint8Array ? v : null);
        };
        req.onerror = () => resolve(null);
      });
    } catch {
      return null;
    } finally {
      db.close();
    }
  }

  async put(key: string, bytes: Uint8Array): Promise<void> {
    const db = await openDb();
    if (!db) return;
    try {
      await new Promise<void>(resolve => {
        const tx = db.transaction(STORE, "readwrite");
        tx.objectStore(STORE).put(bytes, key);
        tx.oncomplete = () => resolve();
        tx.onerror = () => resolve();
        tx.onabort = () => resolve();
      });
    } catch {
      /* best-effort */
    } finally {
      db.close();
    }
  }
}
