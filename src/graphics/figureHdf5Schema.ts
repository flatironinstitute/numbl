/** Shared constants for the "numbl figure HDF5 layout". The on-disk schema is
 *  documented in docs/figure-hdf5-format.md; the writer is exportFigureHdf5.ts
 *  and the reader is importFigureHdf5.ts. Keeping these tables in one place
 *  keeps the two directions in sync. */

/** Version written as the root `numbl_figure_version` attribute. Bump on
 *  incompatible schema changes. */
export const FIGURE_HDF5_VERSION = 1;

/** Per-kind fields that are flat column-major rows×cols grids — stored as
 *  row-major `[rows, cols]` datasets (transposed on write, untransposed on
 *  read). */
export const GRID_FIELDS: Record<string, string[]> = {
  surf: ["x", "y", "z", "c"],
  surface: ["x", "y", "z", "c"],
  mesh: ["x", "y", "z", "c"],
  pcolor: ["x", "y", "c"],
  contour: ["x", "y", "z"],
  heatmap: ["data"],
  imagesc: ["z"],
  bar3: ["z"],
  bar3h: ["z"],
};

/** Maps a trace's `kind` attribute to the AxesState field it belongs in.
 *  `single` marks the one-per-axes traces (imagesc/pie/heatmap). */
export const KIND_TO_CATEGORY: Record<
  string,
  { field: string; single?: boolean }
> = {
  plot: { field: "traces" },
  plot3: { field: "plot3Traces" },
  area: { field: "areaTraces" },
  patch: { field: "patchTraces" },
  surf: { field: "surfTraces" },
  surface: { field: "surfTraces" },
  mesh: { field: "surfTraces" },
  imagesc: { field: "imagescTrace", single: true },
  pcolor: { field: "pcolorTraces" },
  contour: { field: "contourTraces" },
  bar: { field: "barTraces" },
  barh: { field: "barhTraces" },
  bar3: { field: "bar3Traces" },
  bar3h: { field: "bar3hTraces" },
  errorbar: { field: "errorBarTraces" },
  boxchart: { field: "boxTraces" },
  piechart: { field: "pieTrace", single: true },
  heatmap: { field: "heatmapTrace", single: true },
  quiver: { field: "quiverTraces" },
  quiver3: { field: "quiver3Traces" },
};

/** Trace boolean fields, stored as 0/1 int attributes; coerced back on read. */
export const TRACE_BOOLEAN_FIELDS = new Set([
  "showArrowHead",
  "markerFilled",
  "autoScale",
  "is3D",
  "filled",
]);
