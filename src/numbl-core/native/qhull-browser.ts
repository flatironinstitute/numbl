/**
 * Browser/worker loader for the qhull WASM Delaunay backend.
 *
 * Node reads the `.wasm` from disk (qhull-node.ts). In the browser there is no
 * filesystem, so we let Vite emit the `.wasm` as an asset (`?url`), fetch its
 * bytes, and pass them to the emscripten module as `wasmBinary`; this avoids
 * relying on the glue locating the binary next to itself at runtime.
 *
 * `ensureQhullBackend(host)` is best-effort: on any failure it resolves anyway
 * (so it never crashes the worker), but then the host has no backend and a
 * later `delaunay`/`delaunayn` call throws (see geometry.ts). The module is
 * loaded once per worker.
 */

import { loadQhull } from "qhull-wasm";
import qhullWasmUrl from "qhull-wasm/dist/qhull.wasm?url";
import { defaultHost, type NumblHost } from "../host.js";

type Qhull = Awaited<ReturnType<typeof loadQhull>>;
let promise: Promise<Qhull | null> | null = null;

/** Load the qhull module (once) and install it into `host` (default: the
 *  process-default host). Safe to await before every run. */
export async function ensureQhullBackend(
  host: NumblHost = defaultHost()
): Promise<void> {
  promise ??= load();
  const qhull = await promise;
  if (!qhull) return;
  host.geometry.delaunay = (points, dim) => qhull.delaunay(points, dim).facets;
  host.geometry.convexHull = (points, dim) =>
    qhull.convexHull(points, dim).facets;
}

async function load(): Promise<Qhull | null> {
  try {
    const resp = await fetch(qhullWasmUrl);
    const wasmBinary = new Uint8Array(await resp.arrayBuffer());
    return await loadQhull({ wasmBinary });
  } catch (e) {
    console.error(
      "qhull WASM backend failed to load; delaunay/delaunayn/convhull/convhulln will be unavailable:",
      e instanceof Error ? e.message : String(e)
    );
    return null;
  }
}
