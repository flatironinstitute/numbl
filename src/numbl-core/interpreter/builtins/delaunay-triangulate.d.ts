// Type declaration for the untyped `delaunay-triangulate` npm package.
// https://github.com/mikolalysenko/delaunay-triangulate
declare module "delaunay-triangulate" {
  /**
   * Compute the Delaunay triangulation of points in arbitrary dimension.
   *
   * @param points Array of points (each `[x, y]` for 2-D, `[x, y, z]` for 3-D, ...).
   * @param pointAtInfinity If true, include unbounded cells using index -1.
   * @returns Array of cells, each a list of d+1 vertex indices into `points`
   *   (triangles in 2-D, tetrahedra in 3-D).
   */
  function triangulate(
    points: number[][],
    pointAtInfinity?: boolean
  ): number[][];
  export default triangulate;
}
