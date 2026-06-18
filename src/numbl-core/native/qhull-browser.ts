/**
 * Browser/worker loader for the qhull WASM Delaunay backend.
 *
 * The CLI installs the backend in cli.ts (reading the `.wasm` from disk). In
 * the browser there is no filesystem, so we let Vite emit the `.wasm` as an
 * asset (`?url`), fetch its bytes, and pass them to the emscripten module as
 * `wasmBinary` — this avoids relying on the glue locating the binary next to
 * itself at runtime.
 *
 * `ensureQhullBackend()` is idempotent and best-effort: on any failure it
 * resolves anyway (so it never crashes the worker), but then no backend is
 * installed and a later `delaunay`/`delaunayn` call throws (see geometry.ts).
 */

import { setDelaunayBackend, setConvexHullBackend } from "./geometry-bridge.js";
import { loadQhull } from "qhull-wasm";
import qhullWasmUrl from "qhull-wasm/dist/qhull.wasm?url";

let promise: Promise<void> | null = null;

/** Load and install the qhull backend (once). Safe to await before every run. */
export function ensureQhullBackend(): Promise<void> {
  if (!promise) promise = load();
  return promise;
}

async function load(): Promise<void> {
  try {
    const resp = await fetch(qhullWasmUrl);
    const wasmBinary = new Uint8Array(await resp.arrayBuffer());
    const qhull = await loadQhull({ wasmBinary });
    setDelaunayBackend((points, dim) => qhull.delaunay(points, dim).facets);
    setConvexHullBackend((points, dim) => qhull.convexHull(points, dim).facets);
  } catch (e) {
    console.error(
      "qhull WASM backend failed to load; delaunay/delaunayn/convhull/convhulln will be unavailable:",
      e instanceof Error ? e.message : String(e)
    );
  }
}
