/**
 * The per-kind trace containers of an axes, declared once.
 *
 * `AxesTraces` lists the fields of `AxesState` that hold plotted data, and
 * `TRACE_FIELDS` describes each one: its trace `kind` names (used by the
 * figure HDF5 layout), whether it holds one trace or many, whether it makes
 * the axes 3-D, and how it contributes to the automatic axis limits. The
 * reducer's empty axes and hold logic, the axis-limit computation, the 2-D/3-D
 * routing, and the HDF5 writer and reader are all derived from this table, so
 * a new field cannot be forgotten in any of them. The type below makes the
 * table exhaustive: adding a field to `AxesTraces` without an entry here is a
 * compile error.
 *
 * Pure data and helpers: no rendering code, shared by the runtime and the
 * renderer in src/graphics/.
 */

import type {
  PlotTrace,
  Plot3Trace,
  PatchTrace,
  SurfTrace,
  ImagescTrace,
  PcolorTrace,
  ContourTrace,
  BarTrace,
  Bar3Trace,
  ErrorBarTrace,
  BoxTrace,
  PieTrace,
  HeatmapTrace,
  QuiverTrace,
  Quiver3Trace,
} from "./types.js";
import { imagescExtent } from "./imagescExtent.js";

export type AxesTraces = {
  traces: PlotTrace[];
  plot3Traces: Plot3Trace[];
  areaTraces: PlotTrace[];
  patchTraces: PatchTrace[];
  surfTraces: SurfTrace[];
  imagescTrace?: ImagescTrace;
  pcolorTraces: PcolorTrace[];
  contourTraces: ContourTrace[];
  barTraces: BarTrace[];
  barhTraces: BarTrace[];
  bar3Traces: Bar3Trace[];
  bar3hTraces: Bar3Trace[];
  errorBarTraces: ErrorBarTrace[];
  boxTraces: BoxTrace[];
  pieTrace?: PieTrace;
  heatmapTrace?: HeatmapTrace;
  quiverTraces: QuiverTrace[];
  quiver3Traces: Quiver3Trace[];
};

export type TraceField = keyof AxesTraces;

/** Fields drawn only by the 3-D view (SurfView). Every other field is drawn
 *  by the 2-D canvas renderer, and must have an entry in its renderer table. */
export type TraceField3D =
  | "plot3Traces"
  | "surfTraces"
  | "bar3Traces"
  | "bar3hTraces"
  | "quiver3Traces";
export type TraceField2D = Exclude<TraceField, TraceField3D>;

/** The non-empty value of a trace field: an array, or the single trace. */
export type TraceValue<F extends TraceField> = NonNullable<AxesTraces[F]>;

// ── Data extents ─────────────────────────────────────────────────────────

export type Extent = { min: number; max: number };
export type Extents = { x: Extent; y: Extent; z: Extent };

export function include(e: Extent, v: number): void {
  if (Number.isFinite(v)) {
    if (v < e.min) e.min = v;
    if (v > e.max) e.max = v;
  }
}

export function includeAll(e: Extent, vs: ArrayLike<number> | undefined): void {
  if (!vs) return;
  for (let i = 0; i < vs.length; i++) include(e, vs[i]);
}

// ── Per-field declarations ───────────────────────────────────────────────

export interface TraceFieldSpec<F extends TraceField> {
  /** Trace `kind` names stored in this field by the figure HDF5 layout. The
   *  first is written on export; all are accepted on import. */
  kinds: readonly [string, ...string[]];
  /** "many": an array of traces. "one": a single trace per axes (a new one
   *  replaces the old one). */
  multiplicity: AxesTraces[F] extends unknown[] ? "many" : "one";
  /** Whether this (non-empty) content makes the axes 3-D, so that it is
   *  drawn by SurfView instead of the 2-D canvas. True exactly for the
   *  `TraceField3D` fields, except patch, which decides per value. */
  is3D(v: TraceValue<F>): boolean;
  /** A single-trace field that a new plot with hold off leaves in place. Only
   *  imagesc sets this; it is the reducer's long-standing behavior. */
  keptWithoutHold?: true;
  /** Grow `ext` to cover this field's data (raw, unpadded). `axes` gives
   *  access to related non-trace state such as `areaBaseValue`. */
  extents(v: TraceValue<F>, ext: Extents, axes: Partial<AxesTraceState>): void;
}

/** The subset of AxesState the extents functions may read. */
export type AxesTraceState = AxesTraces & { areaBaseValue: number };

/** True when a patch has a non-trivial z extent. A flat patch (all z equal,
 *  including the 2-D case where z defaults to 0) renders in the 2-D canvas. */
export function patchHasZVariation(patch: PatchTrace): boolean {
  if (!patch.is3D) return false;
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of patch.vertices) {
    const z = v[2] ?? 0;
    if (z < lo) lo = z;
    if (z > hi) hi = z;
  }
  return hi - lo > 1e-9;
}

const xyTraces = (
  ts: { x: ArrayLike<number>; y: ArrayLike<number> }[],
  ext: Extents
) => {
  for (const t of ts) {
    includeAll(ext.x, t.x);
    includeAll(ext.y, t.y);
  }
};

const xyzTraces = (
  ts: {
    x: ArrayLike<number>;
    y: ArrayLike<number>;
    z: ArrayLike<number>;
  }[],
  ext: Extents
) => {
  for (const t of ts) {
    includeAll(ext.x, t.x);
    includeAll(ext.y, t.y);
    includeAll(ext.z, t.z);
  }
};

const bar3Extents = (ts: Bar3Trace[], { x, y, z }: Extents) => {
  for (const bt of ts) {
    includeAll(x, bt.x);
    includeAll(y, bt.y);
    includeAll(z, bt.z);
    include(z, 0);
  }
};

/** One entry per trace field. The order is the order traces are written to
 *  the figure HDF5 file. */
export const TRACE_FIELDS: { [F in TraceField]: TraceFieldSpec<F> } = {
  traces: {
    kinds: ["plot"],
    multiplicity: "many",
    is3D: () => false,
    extents: xyTraces,
  },
  plot3Traces: {
    kinds: ["plot3"],
    multiplicity: "many",
    is3D: () => true,
    extents: xyzTraces,
  },
  areaTraces: {
    kinds: ["area"],
    multiplicity: "many",
    is3D: () => false,
    extents(areaTraces, { x, y }, axes) {
      if (areaTraces.length === 0) return;
      const base = axes.areaBaseValue ?? 0;
      include(y, base);
      const n = areaTraces[0].x.length;
      for (let i = 0; i < n; i++) {
        let cum = base;
        for (const t of areaTraces) {
          if (i < t.x.length) {
            include(x, t.x[i]);
            cum += t.y[i] - base;
            include(y, cum);
          }
        }
      }
    },
  },
  patchTraces: {
    kinds: ["patch"],
    multiplicity: "many",
    is3D: patches => patches.some(patchHasZVariation),
    extents(patches, { x, y, z }) {
      for (const t of patches) {
        for (const v of t.vertices) {
          include(x, v[0]);
          include(y, v[1]);
          include(z, v[2] ?? 0);
        }
      }
    },
  },
  surfTraces: {
    kinds: ["surf", "surface", "mesh"],
    multiplicity: "many",
    is3D: () => true,
    extents: xyzTraces,
  },
  imagescTrace: {
    kinds: ["imagesc"],
    multiplicity: "one",
    is3D: () => false,
    keptWithoutHold: true,
    extents(trace, { x, y }) {
      const ext = imagescExtent(trace);
      include(x, ext.x[0]);
      include(x, ext.x[1]);
      include(y, ext.y[0]);
      include(y, ext.y[1]);
    },
  },
  pcolorTraces: {
    kinds: ["pcolor"],
    multiplicity: "many",
    is3D: () => false,
    extents: xyTraces,
  },
  contourTraces: {
    kinds: ["contour"],
    multiplicity: "many",
    is3D: () => false,
    extents: xyTraces,
  },
  barTraces: {
    kinds: ["bar"],
    multiplicity: "many",
    is3D: () => false,
    extents(bars, { x, y }) {
      for (const bt of bars) {
        const hw = bt.width / 2;
        for (let i = 0; i < bt.x.length; i++) {
          include(x, bt.x[i] - hw);
          include(x, bt.x[i] + hw);
          include(y, bt.y[i]);
        }
        include(y, 0);
      }
    },
  },
  barhTraces: {
    kinds: ["barh"],
    multiplicity: "many",
    is3D: () => false,
    extents(bars, { x, y }) {
      for (const bt of bars) {
        const hh = bt.width / 2;
        for (let i = 0; i < bt.x.length; i++) {
          include(y, bt.x[i] - hh);
          include(y, bt.x[i] + hh);
          include(x, bt.y[i]);
        }
        include(x, 0);
      }
    },
  },
  bar3Traces: {
    kinds: ["bar3"],
    multiplicity: "many",
    is3D: () => true,
    extents: bar3Extents,
  },
  bar3hTraces: {
    kinds: ["bar3h"],
    multiplicity: "many",
    is3D: () => true,
    extents: bar3Extents,
  },
  errorBarTraces: {
    kinds: ["errorbar"],
    multiplicity: "many",
    is3D: () => false,
    extents(ets, { x, y }) {
      for (const et of ets) {
        for (let i = 0; i < et.x.length; i++) {
          include(x, et.xNeg ? et.x[i] - et.xNeg[i] : et.x[i]);
          include(x, et.xPos ? et.x[i] + et.xPos[i] : et.x[i]);
          include(y, et.y[i] - et.yNeg[i]);
          include(y, et.y[i] + et.yPos[i]);
        }
      }
    },
  },
  boxTraces: {
    kinds: ["boxchart"],
    multiplicity: "many",
    is3D: () => false,
    extents(boxes, { x, y }) {
      for (const bt of boxes) {
        const hw = bt.width / 2;
        include(x, bt.x - hw);
        include(x, bt.x + hw);
        include(y, bt.whiskerLow);
        include(y, bt.whiskerHigh);
        for (const o of bt.outliers) include(y, o);
      }
    },
  },
  // Pie charts and heatmaps take over the whole canvas and have no data
  // axes, so they do not contribute to the axis limits.
  pieTrace: {
    kinds: ["piechart"],
    multiplicity: "one",
    is3D: () => false,
    extents() {},
  },
  heatmapTrace: {
    kinds: ["heatmap"],
    multiplicity: "one",
    is3D: () => false,
    extents() {},
  },
  quiverTraces: {
    kinds: ["quiver"],
    multiplicity: "many",
    is3D: () => false,
    extents(qts, { x, y }) {
      for (const qt of qts) {
        for (let i = 0; i < qt.x.length; i++) {
          include(x, qt.x[i]);
          include(x, qt.x[i] + qt.u[i]);
          include(y, qt.y[i]);
          include(y, qt.y[i] + qt.v[i]);
        }
      }
    },
  },
  quiver3Traces: {
    kinds: ["quiver3"],
    multiplicity: "many",
    is3D: () => true,
    extents(qts, { x, y, z }) {
      for (const qt of qts) {
        for (let i = 0; i < qt.x.length; i++) {
          include(x, qt.x[i]);
          include(x, qt.x[i] + qt.u[i]);
          include(y, qt.y[i]);
          include(y, qt.y[i] + qt.v[i]);
          include(z, qt.z[i]);
          include(z, qt.z[i] + qt.w[i]);
        }
      }
    },
  },
};

export const TRACE_FIELD_NAMES = Object.keys(TRACE_FIELDS) as TraceField[];

/** Typed accessor for one field's spec (indexing the mapped type with a
 *  union key loses the correlation between key and spec). */
export function traceFieldSpec<F extends TraceField>(
  field: F
): TraceFieldSpec<F> {
  return TRACE_FIELDS[field] as TraceFieldSpec<F>;
}

/** The field's value when it holds at least one trace, else undefined.
 *  Tolerates absent array fields (partial or externally built AxesState). */
export function traceFieldContent<F extends TraceField>(
  axes: Partial<AxesTraces>,
  field: F
): TraceValue<F> | undefined {
  const v = axes[field];
  if (v === undefined || v === null) return undefined;
  if (Array.isArray(v) && v.length === 0) return undefined;
  return v as TraceValue<F>;
}

/** Fresh empty trace containers: `[]` for array fields; single-trace fields
 *  are left absent. */
export function emptyAxesTraces(): AxesTraces {
  const out: Partial<Record<TraceField, unknown>> = {};
  for (const f of TRACE_FIELD_NAMES) {
    if (TRACE_FIELDS[f].multiplicity === "many") out[f] = [];
  }
  return out as AxesTraces;
}
