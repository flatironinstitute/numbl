/**
 * Read a "numbl figure HDF5 layout v1" file (see docs/figure-hdf5-format.md)
 * back into a `FigureState` that `FigureView` can render. The inverse of
 * exportFigureHdf5.ts. h5wasm is loaded lazily via dynamic import, same as the
 * writer.
 */

import type { FigureState, AxesState } from "./figuresReducer.js";
import {
  GRID_FIELDS,
  KIND_TO_CATEGORY,
  TRACE_BOOLEAN_FIELDS,
} from "./figureHdf5Schema.js";

// ── h5wasm read-side typings (a slice of its API) ──────────────────────────

interface H5ReadNode {
  attrs: Record<string, { value: unknown }>;
  keys?(): string[];
  value?: ArrayLike<number | bigint>;
  shape?: number[];
}
interface H5ReadFile {
  get(path: string): H5ReadNode | null;
  close(): void;
}
interface H5ReadModule {
  File: new (name: string, mode: string) => H5ReadFile;
  ready: Promise<{
    FS: { writeFile(p: string, d: Uint8Array): void; unlink(p: string): void };
  }>;
}

const num = (v: unknown): number =>
  typeof v === "bigint" ? Number(v) : (v as number);
const isTyped = (v: unknown): v is ArrayLike<number | bigint> =>
  !!v && typeof v === "object" && "BYTES_PER_ELEMENT" in (v as object);
const toNumArray = (v: unknown): number[] =>
  Array.from(v as ArrayLike<number | bigint>, x => Number(x));

function freshAxes(): AxesState {
  return {
    holdOn: false,
    traces: [],
    plot3Traces: [],
    surfTraces: [],
    pcolorTraces: [],
    contourTraces: [],
    barTraces: [],
    barhTraces: [],
    bar3Traces: [],
    bar3hTraces: [],
    errorBarTraces: [],
    boxTraces: [],
    quiverTraces: [],
    quiver3Traces: [],
    areaTraces: [],
    areaBaseValue: 0,
    patchTraces: [],
  };
}

/** Reverse the row-major [rows,cols] dataset back to numbl's column-major flat. */
function rowMajorToColMajor(
  flat: number[],
  rows: number,
  cols: number
): number[] {
  const out = new Array<number>(rows * cols);
  for (let i = 0; i < rows; i++)
    for (let j = 0; j < cols; j++) out[j * rows + i] = flat[i * cols + j];
  return out;
}

/** Convert a 2-D dataset into the trace field shape implied by its name/kind. */
function readMatrix(
  flat: number[],
  rows: number,
  cols: number,
  kind: string,
  key: string
): unknown {
  if ((GRID_FIELDS[kind] ?? []).includes(key))
    return rowMajorToColMajor(flat, rows, cols);
  if (key === "faces") {
    // strip the -1 padding back to ragged 0-based index lists
    const faces: number[][] = [];
    for (let i = 0; i < rows; i++) {
      const row: number[] = [];
      for (let j = 0; j < cols; j++) {
        const v = flat[i * cols + j];
        if (v === -1) break;
        row.push(v);
      }
      faces.push(row);
    }
    return faces;
  }
  if (key === "faceVertexCData") {
    // each row is [v,NaN,NaN] (scalar) or [r,g,b]
    const out: (number | number[])[] = [];
    for (let i = 0; i < rows; i++) {
      const a = flat[i * cols],
        b = flat[i * cols + 1],
        c = flat[i * cols + 2];
      out.push(Number.isNaN(b) && Number.isNaN(c) ? a : [a, b, c]);
    }
    return out;
  }
  // vertices / colors / colormap_data → number[][]
  const out: number[][] = [];
  for (let i = 0; i < rows; i++)
    out.push(flat.slice(i * cols, i * cols + cols));
  return out;
}

function readDataset(node: H5ReadNode, kind: string, key: string): unknown {
  const flat = toNumArray(node.value);
  const shape = node.shape ?? [flat.length];
  if (shape.length <= 1) return flat;
  return readMatrix(flat, shape[0], shape[1], kind, key);
}

function buildTrace(
  f: H5ReadFile,
  tPath: string
): { kind: string; trace: Record<string, unknown> } {
  const g = f.get(tPath)!;
  const trace: Record<string, unknown> = {};
  const kind = String(g.attrs.kind?.value ?? "plot");

  for (const [name, a] of Object.entries(g.attrs)) {
    if (name === "kind") continue;
    const raw = a.value;
    if (TRACE_BOOLEAN_FIELDS.has(name)) trace[name] = Boolean(num(raw));
    else if (isTyped(raw)) trace[name] = toNumArray(raw);
    else if (typeof raw === "bigint") trace[name] = Number(raw);
    else trace[name] = raw;
  }
  for (const key of g.keys?.() ?? []) {
    const d = f.get(`${tPath}/${key}`);
    if (d) trace[key] = readDataset(d, kind, key);
  }
  return { kind, trace };
}

function buildAxes(f: H5ReadFile, axPath: string): AxesState {
  const g = f.get(axPath)!;
  const ax = freshAxes();
  const A = g.attrs;
  const has = (n: string) => n in A;
  const v = (n: string) => A[n]?.value;
  const str = (n: string) => (has(n) ? String(v(n)) : undefined);
  const lim = (n: string) =>
    has(n)
      ? (toNumArray(v(n)).map(x => (Number.isNaN(x) ? null : x)) as [
          number | null,
          number | null,
        ])
      : undefined;

  ax.title = str("title");
  ax.xlabel = str("xlabel");
  ax.ylabel = str("ylabel");
  ax.zlabel = str("zlabel");
  ax.shading = str("shading") as AxesState["shading"];
  ax.colormap = str("colormap");
  ax.colorbarLocation = str("colorbar_location");
  ax.axisMode = str("axis_mode");
  ax.axisScale = str("axis_scale") as AxesState["axisScale"];
  ax.yDir = str("y_dir") as AxesState["yDir"];
  if (has("legend")) ax.legend = v("legend") as string[];
  if (has("grid_on")) ax.gridOn = Boolean(num(v("grid_on")));
  if (has("box_on")) ax.boxOn = Boolean(num(v("box_on")));
  if (has("hold_on")) ax.holdOn = Boolean(num(v("hold_on")));
  if (has("colorbar")) ax.colorbar = Boolean(num(v("colorbar")));
  if (has("axis_visible")) ax.axisVisible = Boolean(num(v("axis_visible")));
  if (has("area_base_value")) ax.areaBaseValue = num(v("area_base_value"));
  if (has("caxis")) ax.caxis = toNumArray(v("caxis")) as [number, number];
  ax.xlim = lim("xlim");
  ax.ylim = lim("ylim");
  ax.zlim = lim("zlim");
  if (has("view_az") && has("view_el"))
    ax.view = { az: num(v("view_az")), el: num(v("view_el")) };

  const children = g.keys?.() ?? [];
  if (children.includes("colormap_data")) {
    const d = f.get(`${axPath}/colormap_data`);
    if (d) ax.colormapData = readDataset(d, "", "colormap_data") as number[][];
  }
  if (children.includes("traces")) {
    const tg = f.get(`${axPath}/traces`)!;
    const keys = (tg.keys?.() ?? []).sort((a, b) => Number(a) - Number(b));
    for (const k of keys) {
      const { kind, trace } = buildTrace(f, `${axPath}/traces/${k}`);
      const cat = KIND_TO_CATEGORY[kind];
      if (!cat) continue;
      if (cat.single) (ax as Record<string, unknown>)[cat.field] = trace;
      else (ax[cat.field as keyof AxesState] as unknown[]).push(trace);
    }
  }
  return ax;
}

function buildFigure(f: H5ReadFile): FigureState {
  const root = f.get("/")!;
  const A = root.attrs;
  const figure: FigureState = {
    currentAxesIndex: "current_axes" in A ? num(A.current_axes.value) : 1,
    axes: {},
  };
  if ("sgtitle" in A) figure.sgtitle = String(A.sgtitle.value);
  if ("subplot_rows" in A && "subplot_cols" in A)
    figure.subplotGrid = {
      rows: num(A.subplot_rows.value),
      cols: num(A.subplot_cols.value),
    };

  const rootKeys = root.keys?.() ?? [];
  if (rootKeys.includes("uihtml")) {
    const u = f.get("uihtml")!;
    figure.uihtml = {
      id: String(u.attrs.id?.value ?? ""),
      html: String(u.attrs.html?.value ?? ""),
      ...(u.attrs.data ? { data: String(u.attrs.data.value) } : {}),
    };
    return figure;
  }

  if (rootKeys.includes("axes")) {
    const axesGroup = f.get("axes")!;
    for (const idx of axesGroup.keys?.() ?? []) {
      figure.axes[Number(idx)] = buildAxes(f, `axes/${idx}`);
    }
  }
  return figure;
}

/** Parse a numbl-figure HDF5 file into a renderable `FigureState`. */
export async function importFigureHdf5(
  bytes: Uint8Array
): Promise<FigureState> {
  const h5wasm = (
    (await import("h5wasm")) as unknown as { default: H5ReadModule }
  ).default;
  const { FS } = await h5wasm.ready;
  const fname = `/numbl_import_${Date.now()}_${Math.random().toString(36).slice(2)}.h5`;
  FS.writeFile(fname, bytes);
  const f = new h5wasm.File(fname, "r");
  try {
    return buildFigure(f);
  } finally {
    f.close();
    FS.unlink(fname);
  }
}
