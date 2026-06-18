/**
 * Export a figure's data to a self-describing HDF5 file ("numbl figure HDF5
 * layout v1"). The file holds everything needed to recreate the figure:
 * numeric data arrays are stored as gzip-compressed datasets, and all styling /
 * scalar metadata as attributes. The schema is documented in
 * docs/figure-hdf5-format.md.
 *
 * h5wasm (the only mature in-browser HDF5 writer) is loaded lazily via dynamic
 * import so its ~MB of JS+wasm never enters the main bundle — it is fetched the
 * first time a user actually downloads a figure. h5wasm's browser build needs
 * no COOP/COEP (no SharedArrayBuffer), and in Node it reads its wasm from disk.
 *
 * Tensors in numbl are column-major; 2-D grids (surf/imagesc/pcolor/...) are
 * transposed to row-major [rows, cols] datasets so generic HDF5 viewers
 * (h5web, Panoply, h5py) render them correctly without numbl conventions.
 */

import type { FigureState, AxesState } from "./figuresReducer.js";
import { FIGURE_HDF5_VERSION, GRID_FIELDS } from "./figureHdf5Schema.js";

// ── h5wasm minimal typings (we only use a slice of its API) ────────────────

interface H5Dataset {
  create_attribute(
    name: string,
    data: unknown,
    shape?: number[],
    dtype?: string
  ): void;
}
interface H5File extends H5Dataset {
  file_id: bigint;
  create_group(name: string): H5Group;
  create_dataset(args: {
    name: string;
    data: ArrayBufferView;
    shape: number[];
    dtype: string;
    chunks?: number[];
    compression?: "gzip" | number;
    compression_opts?: number;
  }): void;
  get(path: string): H5Group;
  flush(): void;
  close(): void;
}
type H5Group = H5Dataset & { create_group(name: string): H5Group };

interface H5Module {
  File: new (name: string, mode: string) => H5File;
  ready: Promise<{
    FS: { readFile(p: string): Uint8Array; unlink(p: string): void };
  }>;
}

/** Trace-object fields that are always styling/scalars → stored as attributes,
 *  even when their value is a short numeric array (e.g. an [r,g,b] color). */
const STYLE_ATTRS = new Set([
  "color",
  "markerEdgeColor",
  "markerFaceColor",
  "faceColor",
  "edgeColor",
  "lineColor",
  "lineStyle",
  "marker",
  "lineWidth",
  "markerSize",
  "markerIndices",
  "faceAlpha",
  "width",
  "innerRadius",
  "showArrowHead",
  "markerFilled",
  "autoScale",
  "autoScaleFactor",
  "is3D",
  "filled",
  "id",
  "rows",
  "cols",
  "nLevels",
  "baseValue",
]);

/** number[][] fields handled specially (not as plain 1-D datasets). */
const MATRIX_FIELDS = new Set(["vertices", "faces", "faceVertexCData"]);

const GZIP_LEVEL = 4;

/** Flatten the per-category trace arrays of one axes into an ordered list,
 *  each tagged with its plot `kind`. */
function collectTraces(
  ax: AxesState
): { kind: string; trace: Record<string, unknown> }[] {
  const out: { kind: string; trace: Record<string, unknown> }[] = [];
  const push = (kind: string, t: unknown) =>
    out.push({ kind, trace: t as Record<string, unknown> });
  const each = (kind: string, arr?: unknown[]) =>
    (arr ?? []).forEach(t => push(kind, t));

  each("plot", ax.traces);
  each("plot3", ax.plot3Traces);
  each("area", ax.areaTraces);
  each("patch", ax.patchTraces);
  each("surf", ax.surfTraces);
  if (ax.imagescTrace) push("imagesc", ax.imagescTrace);
  each("pcolor", ax.pcolorTraces);
  each("contour", ax.contourTraces);
  each("bar", ax.barTraces);
  each("barh", ax.barhTraces);
  each("bar3", ax.bar3Traces);
  each("bar3h", ax.bar3hTraces);
  each("errorbar", ax.errorBarTraces);
  each("boxchart", ax.boxTraces);
  if (ax.pieTrace) push("piechart", ax.pieTrace);
  if (ax.heatmapTrace) push("heatmap", ax.heatmapTrace);
  each("quiver", ax.quiverTraces);
  each("quiver3", ax.quiver3Traces);
  return out;
}

function isNumberArray(v: unknown): v is number[] {
  return Array.isArray(v) && v.every(e => typeof e === "number");
}
function isStringArray(v: unknown): v is string[] {
  return (
    Array.isArray(v) && v.length > 0 && v.every(e => typeof e === "string")
  );
}

/** Column-major flat (element (i,j) at j*rows+i) → row-major [rows,cols] flat. */
function colMajorToRowMajor(
  flat: number[],
  rows: number,
  cols: number
): Float64Array {
  const out = new Float64Array(rows * cols);
  for (let i = 0; i < rows; i++)
    for (let j = 0; j < cols; j++) out[i * cols + j] = flat[j * rows + i];
  return out;
}

export async function exportFigureHdf5(
  figure: FigureState
): Promise<Uint8Array> {
  const h5wasm = ((await import("h5wasm")) as unknown as { default: H5Module })
    .default;
  const { FS } = await h5wasm.ready;

  const fname = `/numbl_fig_${Date.now()}_${Math.random().toString(36).slice(2)}.h5`;
  const f = new h5wasm.File(fname, "w");
  const created = new Set<string>();

  const ensureGroup = (path: string): H5Group => {
    const segs = path.split("/").filter(Boolean);
    let cur = "";
    for (const seg of segs) {
      cur = cur ? `${cur}/${seg}` : seg;
      if (!created.has(cur)) {
        f.create_group(cur);
        created.add(cur);
      }
    }
    return f.get(path);
  };

  const attr = (obj: H5Dataset, name: string, value: unknown): void => {
    if (value === null || value === undefined) return;
    if (typeof value === "boolean") obj.create_attribute(name, value ? 1 : 0);
    else if (typeof value === "number" || typeof value === "string")
      obj.create_attribute(name, value);
    else if (isStringArray(value)) obj.create_attribute(name, value);
    else if (isNumberArray(value))
      obj.create_attribute(
        name,
        new Float64Array(value),
        [value.length],
        "<f8"
      );
  };

  const dataset1D = (path: string, arr: number[]): void => {
    if (arr.length === 0) return;
    f.create_dataset({
      name: path,
      data: new Float64Array(arr),
      shape: [arr.length],
      dtype: "<f8",
      chunks: [arr.length],
      compression: "gzip",
      compression_opts: GZIP_LEVEL,
    });
  };

  const dataset2D = (
    path: string,
    data: ArrayBufferView,
    rows: number,
    cols: number,
    dtype = "<f8"
  ): void => {
    if (rows * cols === 0) return;
    f.create_dataset({
      name: path,
      data,
      shape: [rows, cols],
      dtype,
      chunks: [rows, cols],
      compression: "gzip",
      compression_opts: GZIP_LEVEL,
    });
  };

  // number[][] fields (vertices / faces / faceVertexCData / colormap_data).
  const matrixDataset = (path: string, rowsArr: unknown[]): void => {
    if (rowsArr.length === 0) return;
    // faces are ragged 0-based index lists → pad to a rectangular int matrix.
    const allNumberRows = rowsArr.every(r => isNumberArray(r));
    if (!allNumberRows) {
      // faceVertexCData: mix of scalar (colormap value) and [r,g,b]. Normalize
      // each entry to length 3 (scalar → [v,NaN,NaN]) so it is rectangular.
      const norm = rowsArr.map(r =>
        typeof r === "number" ? [r, NaN, NaN] : (r as number[])
      );
      flatRect(path, norm, "<f8");
      return;
    }
    flatRect(
      path,
      rowsArr as number[][],
      path.endsWith("/faces") ? "<i4" : "<f8"
    );
  };

  const flatRect = (path: string, rows: number[][], dtype: string): void => {
    const nRows = rows.length;
    const nCols = Math.max(...rows.map(r => r.length));
    const pad = dtype === "<i4" ? -1 : NaN;
    const buf =
      dtype === "<i4"
        ? new Int32Array(nRows * nCols)
        : new Float64Array(nRows * nCols);
    for (let i = 0; i < nRows; i++)
      for (let j = 0; j < nCols; j++)
        buf[i * nCols + j] = j < rows[i].length ? rows[i][j] : pad;
    dataset2D(path, buf, nRows, nCols, dtype);
  };

  // ── root ────────────────────────────────────────────────────────────────
  const root = f.get("/");
  attr(root, "numbl_figure_version", FIGURE_HDF5_VERSION);
  attr(root, "generator", "numbl");
  if (figure.sgtitle) attr(root, "sgtitle", figure.sgtitle);
  if (figure.subplotGrid) {
    attr(root, "subplot_rows", figure.subplotGrid.rows);
    attr(root, "subplot_cols", figure.subplotGrid.cols);
  }
  attr(root, "current_axes", figure.currentAxesIndex);

  // ── uihtml figures have no axes/traces ────────────────────────────────────
  if (figure.uihtml) {
    const g = ensureGroup("uihtml");
    attr(g, "id", figure.uihtml.id);
    attr(g, "html", figure.uihtml.html);
    if (figure.uihtml.data) attr(g, "data", figure.uihtml.data);
    f.flush();
    f.close();
    const bytes = FS.readFile(fname);
    FS.unlink(fname);
    return bytes;
  }

  // ── axes ──────────────────────────────────────────────────────────────────
  for (const [idxStr, ax] of Object.entries(figure.axes)) {
    const axPath = `axes/${idxStr}`;
    const g = ensureGroup(axPath);

    attr(g, "title", ax.title);
    attr(g, "xlabel", ax.xlabel);
    attr(g, "ylabel", ax.ylabel);
    attr(g, "zlabel", ax.zlabel);
    attr(g, "shading", ax.shading);
    attr(g, "legend", ax.legend);
    attr(g, "grid_on", ax.gridOn);
    attr(g, "box_on", ax.boxOn);
    attr(g, "hold_on", ax.holdOn);
    attr(g, "colorbar", ax.colorbar);
    attr(g, "colorbar_location", ax.colorbarLocation);
    attr(g, "colormap", ax.colormap);
    attr(g, "axis_mode", ax.axisMode);
    attr(g, "axis_scale", ax.axisScale);
    attr(g, "axis_visible", ax.axisVisible);
    attr(g, "y_dir", ax.yDir);
    attr(g, "area_base_value", ax.areaBaseValue);
    if (ax.view) {
      attr(g, "view_az", ax.view.az);
      attr(g, "view_el", ax.view.el);
    }
    if (ax.caxis) attr(g, "caxis", ax.caxis);
    // limits: a null bound means "auto" → encode as NaN.
    const lim = (v?: (number | null)[]) =>
      v ? v.map(x => (x === null ? NaN : x)) : undefined;
    attr(g, "xlim", lim(ax.xlim));
    attr(g, "ylim", lim(ax.ylim));
    attr(g, "zlim", lim(ax.zlim));
    if (ax.colormapData)
      matrixDataset(`${axPath}/colormap_data`, ax.colormapData);

    const traces = collectTraces(ax);
    for (let k = 0; k < traces.length; k++) {
      const { kind, trace } = traces[k];
      const tPath = `${axPath}/traces/${k}`;
      const tg = ensureGroup(tPath);
      attr(tg, "kind", kind);
      const gridFields = GRID_FIELDS[kind] ?? [];
      const rows = typeof trace.rows === "number" ? trace.rows : 0;
      const cols = typeof trace.cols === "number" ? trace.cols : 0;

      for (const [key, val] of Object.entries(trace)) {
        if (val === null || val === undefined) continue;
        if (
          STYLE_ATTRS.has(key) ||
          typeof val === "number" ||
          typeof val === "string" ||
          typeof val === "boolean"
        ) {
          attr(tg, key, val);
        } else if (MATRIX_FIELDS.has(key) && Array.isArray(val)) {
          matrixDataset(`${tPath}/${key}`, val as unknown[]);
        } else if (isStringArray(val)) {
          attr(tg, key, val);
        } else if (isNumberArray(val)) {
          if (
            gridFields.includes(key) &&
            rows * cols === val.length &&
            rows * cols > 0
          )
            dataset2D(
              `${tPath}/${key}`,
              colMajorToRowMajor(val, rows, cols),
              rows,
              cols
            );
          else dataset1D(`${tPath}/${key}`, val);
        } else if (Array.isArray(val) && val.every(r => Array.isArray(r))) {
          matrixDataset(`${tPath}/${key}`, val as unknown[]);
        }
      }
    }
  }

  f.flush();
  f.close();
  const bytes = FS.readFile(fname);
  FS.unlink(fname);
  return bytes;
}

/** Build the HDF5 bytes for a figure and trigger a browser download.
 *  Browser-only (uses the DOM). `handle` is used for the default filename. */
export async function downloadFigureHdf5(
  figure: FigureState,
  handle: number
): Promise<void> {
  const bytes = await exportFigureHdf5(figure);
  // Copy into a fresh ArrayBuffer: the bytes returned by the emscripten FS are
  // a view onto WASM heap memory, which Blob must not alias.
  const blob = new Blob([bytes.slice()], { type: "application/x-hdf5" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `figure_${handle}.h5`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
