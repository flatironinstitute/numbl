/**
 * Shared graphics types used by both numbl-core and the rendering layer.
 *
 * This file is the single source of truth for plot trace interfaces and
 * PlotInstruction.  numbl-core re-exports these types so that internal
 * runtime code can import them without reaching into src/graphics/.
 *
 * The numbl-vscode extension syncs this file (and the rest of src/graphics/)
 * via devel/sync-graphics.sh.
 */

// ── PlotTrace type ──────────────────────────────────────────────────────

export interface PlotTrace {
  x: number[];
  y: number[];
  lineStyle?: string;
  marker?: string;
  color?: [number, number, number];
  lineWidth?: number;
  markerSize?: number;
  markerEdgeColor?: [number, number, number];
  markerFaceColor?: [number, number, number];
  markerIndices?: number[];
}

// ── Plot3Trace type ─────────────────────────────────────────────────────

export interface Plot3Trace {
  x: number[];
  y: number[];
  z: number[];
  lineStyle?: string;
  marker?: string;
  color?: [number, number, number];
  lineWidth?: number;
  markerSize?: number;
  markerEdgeColor?: [number, number, number];
  markerFaceColor?: [number, number, number];
  markerIndices?: number[];
}

// ── SurfTrace type ──────────────────────────────────────────────────────

export interface SurfTrace {
  /** X coordinates: flat array of length rows*cols (column-major) */
  x: number[];
  /** Y coordinates: flat array of length rows*cols (column-major) */
  y: number[];
  /** Z values: flat array of length rows*cols (column-major) */
  z: number[];
  /** Number of rows in the grid */
  rows: number;
  /** Number of columns in the grid */
  cols: number;
  /** Optional color data (same shape as Z) */
  c?: number[];
  edgeColor?: [number, number, number] | "none" | "flat" | "interp";
  faceColor?:
    | [number, number, number]
    | "flat"
    | "interp"
    | "none"
    | "texturemap";
  faceAlpha?: number;
}

// ── ImagescTrace type ────────────────────────────────────────────────────

export interface ImagescTrace {
  /** X limits [xmin, xmax] */
  x: [number, number];
  /** Y limits [ymin, ymax] */
  y: [number, number];
  /** Z data: flat array (column-major), rows × cols */
  z: number[];
  rows: number;
  cols: number;
}

// ── ContourTrace type ────────────────────────────────────────────────────

export interface ContourTrace {
  /** X coordinates: flat array (column-major) */
  x: number[];
  /** Y coordinates: flat array (column-major) */
  y: number[];
  /** Z values: flat array (column-major) */
  z: number[];
  rows: number;
  cols: number;
  /** Number of contour levels */
  nLevels: number;
  /** Whether this is a filled contour (contourf) */
  filled: boolean;
}

// ── Plot Instructions ───────────────────────────────────────────────────

export type PlotInstruction =
  | { type: "set_figure_handle"; handle: number }
  | { type: "plot"; traces: PlotTrace[] }
  | { type: "plot3"; traces: Plot3Trace[] }
  | { type: "surf"; trace: SurfTrace }
  | { type: "imagesc"; trace: ImagescTrace }
  | { type: "contour"; trace: ContourTrace }
  | { type: "mesh"; trace: SurfTrace }
  | { type: "set_hold"; value: boolean }
  | { type: "set_title"; text: string }
  | { type: "set_xlabel"; text: string }
  | { type: "set_ylabel"; text: string }
  | { type: "set_zlabel"; text: string }
  | { type: "set_shading"; shading: "faceted" | "flat" | "interp" }
  | { type: "close" }
  | { type: "close_all" }
  | { type: "clf" }
  | { type: "set_subplot"; rows: number; cols: number; index: number }
  | { type: "set_legend"; labels: string[] }
  | { type: "set_sgtitle"; text: string }
  | { type: "set_grid"; value: boolean }
  | { type: "set_colorbar"; value: string }
  | { type: "set_colormap"; name: string }
  | { type: "set_axis"; value: string }
  | { type: "set_view"; az: number; el: number };
