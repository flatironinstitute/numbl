/**
 * Node loader for the qhull WASM Delaunay backend.
 *
 * In Node the emscripten module reads its `.wasm` from disk on its own, so no
 * asset wiring is needed (unlike the browser; see qhull-browser.ts). Used by
 * `initNodePlatform`, the vitest setup, and re-exported from the library entry
 * (lib.ts) as `loadDelaunayBackend`.
 *
 * The module is loaded once per process; each call installs it into `host`
 * (default: the process-default host).
 */

import { defaultHost, type NumblHost } from "../host.js";

type Qhull = Awaited<ReturnType<typeof import("qhull-wasm").loadQhull>>;
let promise: Promise<Qhull> | null = null;

/** Load the qhull WASM module (once) and install it as `host`'s Delaunay and
 *  convex-hull backends. */
export async function loadQhullNodeBackend(
  host: NumblHost = defaultHost()
): Promise<void> {
  promise ??= import("qhull-wasm").then(m => m.loadQhull());
  const qhull = await promise;
  host.geometry.delaunay = (points, dim) => qhull.delaunay(points, dim).facets;
  host.geometry.convexHull = (points, dim) =>
    qhull.convexHull(points, dim).facets;
}
