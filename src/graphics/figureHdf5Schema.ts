/** Shared constants for the "numbl figure HDF5 layout". The on-disk schema is
 *  documented in docs/figure-hdf5-format.md; the writer is exportFigureHdf5.ts
 *  and the reader is importFigureHdf5.ts. Keeping these tables in one place
 *  keeps the two directions in sync. */

import {
  TRACE_FIELDS,
  TRACE_FIELD_NAMES,
  type TraceField,
} from "../numbl-core/plotting/axesTraces.js";

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
 *  `single` marks the one-per-axes traces (imagesc/pie/heatmap). Derived from
 *  the per-field declarations in axesTraces.ts. */
export const KIND_TO_CATEGORY: Record<
  string,
  { field: TraceField; single?: boolean }
> = Object.fromEntries(
  TRACE_FIELD_NAMES.flatMap(field => {
    const spec = TRACE_FIELDS[field];
    const cat =
      spec.multiplicity === "one" ? { field, single: true } : { field };
    return spec.kinds.map(kind => [kind, cat]);
  })
);

/** The `kind` written for each AxesState trace field (its first listed kind). */
export function exportKind(field: TraceField): string {
  return TRACE_FIELDS[field].kinds[0];
}

/** Trace boolean fields, stored as 0/1 int attributes; coerced back on read. */
export const TRACE_BOOLEAN_FIELDS = new Set([
  "showArrowHead",
  "markerFilled",
  "autoScale",
  "is3D",
  "filled",
]);
