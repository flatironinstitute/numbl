/**
 * Geometry backends (Delaunay triangulation, convex hull) live on the host
 * (`NumblHost.geometry`, see host.ts). Startup loads the qhull WASM module
 * (exact, robust to cospherical/coplanar input) into the host:
 * `loadQhullNodeBackend` (qhull-node.ts) in Node, `ensureQhullBackend`
 * (qhull-browser.ts) in the browser worker. The builtins require it: with no
 * backend, delaunay/delaunayn/convhull throw.
 */

import {
  currentHost,
  defaultHost,
  type ConvexHullBackend,
  type DelaunayBackend,
} from "../host.js";

export type { ConvexHullBackend, DelaunayBackend } from "../host.js";

/** Install (or clear, with null) the process-default host's Delaunay
 *  backend.
 *  @deprecated Set `host.geometry.delaunay`, or load qhull into a host with
 *  `loadDelaunayBackend(host)`. */
export function setDelaunayBackend(fn: DelaunayBackend | null): void {
  defaultHost().geometry.delaunay = fn ?? undefined;
}

/** The running code's Delaunay backend, or null if its host has none. */
export function getDelaunayBackend(): DelaunayBackend | null {
  return currentHost().geometry.delaunay ?? null;
}

/** Install (or clear, with null) the process-default host's convex-hull
 *  backend.
 *  @deprecated Set `host.geometry.convexHull`. */
export function setConvexHullBackend(fn: ConvexHullBackend | null): void {
  defaultHost().geometry.convexHull = fn ?? undefined;
}

/** The running code's convex-hull backend, or null if its host has none. */
export function getConvexHullBackend(): ConvexHullBackend | null {
  return currentHost().geometry.convexHull ?? null;
}
