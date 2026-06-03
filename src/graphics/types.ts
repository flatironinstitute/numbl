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
  /** Stable identity for a trace returned as a graphics handle (e.g. from
   *  `line`). Lets `set(h,'XData',...)` / `update_trace` find and mutate this
   *  exact trace in the renderer's accumulated state across `drawnow`. */
  id?: number;
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
  /** See PlotTrace.id. */
  id?: number;
}

// ── PatchTrace type ─────────────────────────────────────────────────────

/** A patch object: one or more colored polygons sharing a single set of
 *  vertices. Mirrors MATLAB's Faces/Vertices model — `patch(X,Y,C)` and the
 *  name-value forms are all canonicalized to this shape at parse time. */
export interface PatchTrace {
  /** Unique vertices: `vertices[k] = [x, y]` (2-D) or `[x, y, z]` (3-D). */
  vertices: number[][];
  /** Faces as 0-based vertex-index arrays — one polygon per face. */
  faces: number[][];
  /** Single RGB for all faces, a colormap keyword, or 'none'. */
  faceColor?: [number, number, number] | "flat" | "interp" | "none";
  /** Edge color (default black [0,0,0]). */
  edgeColor?: [number, number, number] | "flat" | "interp" | "none";
  /** Face transparency in [0,1]. */
  faceAlpha?: number;
  lineWidth?: number;
  lineStyle?: string;
  marker?: string;
  markerFaceColor?: [number, number, number] | "flat" | "none";
  /** Color data driving 'flat'/'interp': one entry per face (flat) or per
   *  vertex (interp). Each entry is a scalar (mapped through the colormap) or
   *  an RGB triplet. */
  faceVertexCData?: (number | [number, number, number])[];
  /** True when vertices carry a z-coordinate (3-D patch). */
  is3D?: boolean;
  /** See PlotTrace.id — lets `set(p,...)` live-update this patch. */
  id?: number;
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

// ── PcolorTrace type ────────────────────────────────────────────────────

export interface PcolorTrace {
  /** X coordinates: flat array of length rows*cols (column-major) */
  x: number[];
  /** Y coordinates: flat array of length rows*cols (column-major) */
  y: number[];
  /** Color values: flat array of length rows*cols (column-major) */
  c: number[];
  /** Number of rows in the grid */
  rows: number;
  /** Number of columns in the grid */
  cols: number;
  edgeColor?: [number, number, number] | "none";
  faceAlpha?: number;
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
  /** Number of contour levels (used only when `levels` is not given) */
  nLevels: number;
  /** Explicit contour levels (from `contour(...,V)` or `LevelList`). When
   *  absent, levels are chosen automatically from the data range. */
  levels?: number[];
  /** Line width from a `'LineWidth'` name-value pair. */
  lineWidth?: number;
  /** Line style from a `'LineStyle'` name-value pair (e.g. '-', '--'). */
  lineStyle?: string;
  /** Line color: an RGB triple, or a colormap keyword like 'flat'. */
  lineColor?: number[] | string;
  /** Whether this is a filled contour (contourf) */
  filled: boolean;
}

// ── BarTrace type ───────────────────────────────────────────────────────

export interface BarTrace {
  /** X positions for each bar */
  x: number[];
  /** Y values (bar heights) for each bar */
  y: number[];
  /** Relative bar width (0–1, default 0.8) */
  width: number;
  /** Bar color as RGB triple [0–1] */
  color?: [number, number, number];
}

// ── Bar3Trace type ──────────────────────────────────────────────────────

export interface Bar3Trace {
  /** X positions for each bar (column index) */
  x: number[];
  /** Y positions for each bar (row index) */
  y: number[];
  /** Z values (bar heights) for each bar */
  z: number[];
  /** Number of rows in the grid */
  rows: number;
  /** Number of columns in the grid */
  cols: number;
  /** Relative bar width (0–1, default 0.8) */
  width: number;
  /** Bar color as RGB triple [0–1] */
  color?: [number, number, number];
}

// ── ErrorBarTrace type ──────────────────────────────────────────────────

export interface ErrorBarTrace {
  x: number[];
  y: number[];
  /** Error below each data point */
  yNeg: number[];
  /** Error above each data point */
  yPos: number[];
  /** Error left of each data point (horizontal error bars) */
  xNeg?: number[];
  /** Error right of each data point (horizontal error bars) */
  xPos?: number[];
  color?: [number, number, number];
  lineStyle?: string;
  marker?: string;
  lineWidth?: number;
}

// ── BoxTrace type ───────────────────────────────────────────────────────

export interface BoxTrace {
  /** X position for this box */
  x: number;
  /** Median value */
  median: number;
  /** Lower quartile (Q1, 25th percentile) */
  q1: number;
  /** Upper quartile (Q3, 75th percentile) */
  q3: number;
  /** Lower whisker end (min non-outlier) */
  whiskerLow: number;
  /** Upper whisker end (max non-outlier) */
  whiskerHigh: number;
  /** Outlier values */
  outliers: number[];
  /** Relative box width */
  width: number;
  /** Box color as RGB triple [0–1] */
  color?: [number, number, number];
}

// ── PieTrace type ───────────────────────────────────────────────────────

export interface PieTrace {
  /** Slice values (absolute, not percentages) */
  values: number[];
  /** Optional slice names */
  names?: string[];
  /** Inner radius as fraction of outer radius (0 = pie, >0 = donut) */
  innerRadius: number;
  /** Per-slice colors as RGB triples [0–1] */
  colors?: [number, number, number][];
}

// ── QuiverTrace type ────────────────────────────────────────────────────

export interface QuiverTrace {
  /** Tail x-coordinates (flat array) */
  x: number[];
  /** Tail y-coordinates (flat array) */
  y: number[];
  /** x-component of each arrow (already scaled to data units) */
  u: number[];
  /** y-component of each arrow (already scaled to data units) */
  v: number[];
  /** Whether to draw arrowheads */
  showArrowHead: boolean;
  /** Arrow color as RGB triple [0–1] */
  color?: [number, number, number];
  lineStyle?: string;
  lineWidth?: number;
  marker?: string;
  /** Whether the marker should be filled */
  markerFilled?: boolean;
}

// ── Quiver3Trace type ───────────────────────────────────────────────────

export interface Quiver3Trace {
  /** Tail coordinates (flat arrays) */
  x: number[];
  y: number[];
  z: number[];
  /** Directional components, already scaled to data units */
  u: number[];
  v: number[];
  w: number[];
  /** Whether to draw arrowheads */
  showArrowHead: boolean;
  /** Arrow color as RGB triple [0–1] */
  color?: [number, number, number];
  lineStyle?: string;
  lineWidth?: number;
  marker?: string;
  markerFilled?: boolean;
  /** Whether auto-scaling was applied (for handle queries) */
  autoScale?: boolean;
  autoScaleFactor?: number;
}

// ── HeatmapTrace type ───────────────────────────────────────────────────

export interface HeatmapTrace {
  /** Cell values: flat array (column-major), rows × cols */
  data: number[];
  rows: number;
  cols: number;
  /** X-axis labels (one per column) */
  xLabels?: string[];
  /** Y-axis labels (one per row) */
  yLabels?: string[];
}

// ── Axis limit spec ──────────────────────────────────────────────────────

/** A single axis's limit request from `axis([...])` / `axis auto`.
 *  - A `[lo, hi]` pair, where either bound may be `null` to mean "keep the
 *    automatically-chosen bound" (from `inf`/`-inf` in the limits vector).
 *  - The string `"auto"` clears any explicit limit so the axis refits to its
 *    data (from `axis auto` / `axis 'auto x'`). */
export type AxisLimitSpec = [number | null, number | null] | "auto";

// ── Plot Instructions ───────────────────────────────────────────────────

export type PlotInstruction =
  | { type: "set_figure_handle"; handle: number }
  | {
      /** An HTML UI component (MATLAB `uihtml`): renders self-contained HTML
       *  markup in an iframe, bypassing the axes/trace model. `html` is the
       *  full HTMLSource string; `id` is a stable per-component key. */
      type: "uihtml";
      id: string;
      html: string;
    }
  | { type: "plot"; traces: PlotTrace[] }
  | { type: "plot3"; traces: Plot3Trace[] }
  | { type: "line"; traces: PlotTrace[] }
  | { type: "line3"; traces: Plot3Trace[] }
  | { type: "patch"; trace: PatchTrace }
  | { type: "update_trace"; id: number; props: Record<string, unknown> }
  | { type: "surf"; trace: SurfTrace }
  | { type: "surface"; trace: SurfTrace }
  | { type: "imagesc"; trace: ImagescTrace }
  | { type: "pcolor"; trace: PcolorTrace }
  | { type: "contour"; trace: ContourTrace }
  | { type: "mesh"; trace: SurfTrace }
  | { type: "bar"; traces: BarTrace[] }
  | { type: "barh"; traces: BarTrace[] }
  | { type: "bar3"; trace: Bar3Trace }
  | { type: "bar3h"; trace: Bar3Trace }
  | { type: "errorbar"; traces: ErrorBarTrace[] }
  | { type: "area"; traces: PlotTrace[]; baseValue: number }
  | { type: "boxchart"; traces: BoxTrace[] }
  | { type: "piechart"; trace: PieTrace }
  | { type: "heatmap"; trace: HeatmapTrace }
  | { type: "quiver"; traces: QuiverTrace[] }
  | { type: "quiver3"; trace: Quiver3Trace }
  | { type: "set_hold"; value: boolean }
  | { type: "set_title"; text: string }
  | { type: "set_xlabel"; text: string }
  | { type: "set_ylabel"; text: string }
  | { type: "set_zlabel"; text: string }
  | { type: "set_shading"; shading: "faceted" | "flat" | "interp" }
  | { type: "close" }
  | { type: "close_all" }
  | { type: "clf" }
  | { type: "cla"; reset: boolean }
  | { type: "set_subplot"; rows: number; cols: number; index: number }
  | { type: "set_legend"; labels: string[] }
  | { type: "set_sgtitle"; text: string }
  | { type: "set_grid"; value: boolean }
  | { type: "set_box"; value: boolean }
  | { type: "set_colorbar"; value: string; location?: string }
  | { type: "set_colormap"; name: string; data?: number[][] }
  | { type: "set_axis"; value: string }
  | {
      type: "set_axis_limits";
      xlim?: AxisLimitSpec;
      ylim?: AxisLimitSpec;
      zlim?: AxisLimitSpec;
    }
  | { type: "set_axis_ydir"; dir: "normal" | "reverse" }
  | { type: "set_axis_visible"; value: boolean }
  | {
      type: "set_axis_scale";
      value: "linear" | "semilogx" | "semilogy" | "loglog";
    }
  | { type: "set_view"; az: number; el: number }
  | { type: "set_caxis"; limits: [number, number] };
