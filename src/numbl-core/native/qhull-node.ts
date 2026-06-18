/**
 * Node loader for the qhull WASM Delaunay backend.
 *
 * In Node the emscripten module reads its `.wasm` from disk on its own, so no
 * asset wiring is needed (unlike the browser; see qhull-browser.ts). This is
 * used by the CLI, the vitest setup, and is re-exported from the library entry
 * (lib.ts) so programmatic consumers can install the backend before calling
 * `delaunay` / `delaunayn`.
 *
 * Idempotent: the first call starts loading and caches the promise.
 */

import { setDelaunayBackend, setConvexHullBackend } from "./geometry-bridge.js";

let promise: Promise<void> | null = null;

/** Load the qhull WASM module and install it as the Delaunay backend (once). */
export function loadQhullNodeBackend(): Promise<void> {
  if (!promise) promise = load();
  return promise;
}

async function load(): Promise<void> {
  const { loadQhull } = await import("qhull-wasm");
  const qhull = await loadQhull();
  setDelaunayBackend((points, dim) => qhull.delaunay(points, dim).facets);
  setConvexHullBackend((points, dim) => qhull.convexHull(points, dim).facets);
}
