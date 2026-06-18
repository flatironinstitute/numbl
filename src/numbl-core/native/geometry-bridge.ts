/**
 * Geometry bridge — a module-level singleton that holds the Delaunay
 * triangulation backend.
 *
 * Startup (CLI, browser worker, test-runner) loads the qhull WASM module
 * (exact, robust to cospherical/coplanar input) and installs it here. The
 * builtins require it: if no backend is set, delaunay/delaunayn throw.
 *
 * Usage (startup):
 *   import { setDelaunayBackend } from './numbl-core/native/geometry-bridge.js';
 *   const qhull = await loadQhull();
 *   setDelaunayBackend((points, dim) => qhull.delaunay(points, dim).facets);
 *
 * Usage (builtin):
 *   import { getDelaunayBackend } from '../native/geometry-bridge.js';
 *   const backend = getDelaunayBackend();
 *   if (!backend) throw ...;
 *   const cells = backend(points, dim);
 */

/**
 * Compute a Delaunay triangulation of `points` (an array of dim-dimensional
 * tuples). Returns an array of simplices, each a list of `dim+1` 0-based input
 * point indices.
 */
export type DelaunayBackend = (points: number[][], dim: number) => number[][];

let backend: DelaunayBackend | null = null;

/** Install (or clear, with null) the Delaunay backend. */
export function setDelaunayBackend(fn: DelaunayBackend | null): void {
  backend = fn;
}

/** Get the current Delaunay backend, or null if none is installed. */
export function getDelaunayBackend(): DelaunayBackend | null {
  return backend;
}
