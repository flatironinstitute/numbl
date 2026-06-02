/**
 * Pure computation of the current axis limits for an `AxesState`.
 *
 * This is the source of truth for the query form `lim = axis`: the numbl
 * runtime reduces its accumulated plot instructions into an `AxesState` and
 * asks this module for the limit vector. It is deliberately rendering-free
 * (no canvas, no DOM) so numbl-core can import it.
 *
 * For an axis with an explicit limit, that limit is returned verbatim (a
 * `null` bound falls back to the data-derived bound). For a fully automatic
 * axis the data extent is computed across every trace type and padded the
 * same way `drawPlot` pads a default 2-D view, so the reported limits track
 * what is drawn.
 */

import type { AxesState } from "./figuresReducer.js";

type Extent = { min: number; max: number };

const EMPTY: Extent = { min: Infinity, max: -Infinity };

function include(e: Extent, v: number): void {
  if (Number.isFinite(v)) {
    if (v < e.min) e.min = v;
    if (v > e.max) e.max = v;
  }
}

function includeAll(e: Extent, vs: ArrayLike<number> | undefined): void {
  if (!vs) return;
  for (let i = 0; i < vs.length; i++) include(e, vs[i]);
}

/** Does this axes hold any 3-D content (so limits are a 6-vector)? */
export function axesIs3D(axes: AxesState): boolean {
  return (
    (axes.surfTraces?.length ?? 0) > 0 ||
    (axes.plot3Traces?.length ?? 0) > 0 ||
    (axes.bar3Traces?.length ?? 0) > 0 ||
    (axes.bar3hTraces?.length ?? 0) > 0 ||
    (axes.quiver3Traces?.length ?? 0) > 0
  );
}

/** Raw data extents (no padding) across every renderable trace type. */
function dataExtents(axes: AxesState): { x: Extent; y: Extent; z: Extent } {
  const x: Extent = { ...EMPTY };
  const y: Extent = { ...EMPTY };
  const z: Extent = { ...EMPTY };

  for (const t of axes.traces ?? []) {
    includeAll(x, t.x);
    includeAll(y, t.y);
  }
  for (const t of axes.plot3Traces ?? []) {
    includeAll(x, t.x);
    includeAll(y, t.y);
    includeAll(z, t.z);
  }
  for (const t of [...(axes.surfTraces ?? [])]) {
    includeAll(x, t.x);
    includeAll(y, t.y);
    includeAll(z, t.z);
  }
  if (axes.imagescTrace) {
    include(x, axes.imagescTrace.x[0]);
    include(x, axes.imagescTrace.x[1]);
    include(y, axes.imagescTrace.y[0]);
    include(y, axes.imagescTrace.y[1]);
  }
  for (const t of axes.pcolorTraces ?? []) {
    includeAll(x, t.x);
    includeAll(y, t.y);
  }
  for (const t of axes.contourTraces ?? []) {
    includeAll(x, t.x);
    includeAll(y, t.y);
  }
  for (const bt of axes.barTraces ?? []) {
    const hw = bt.width / 2;
    for (let i = 0; i < bt.x.length; i++) {
      include(x, bt.x[i] - hw);
      include(x, bt.x[i] + hw);
      include(y, bt.y[i]);
    }
    include(y, 0);
  }
  for (const bt of axes.barhTraces ?? []) {
    const hh = bt.width / 2;
    for (let i = 0; i < bt.x.length; i++) {
      include(y, bt.x[i] - hh);
      include(y, bt.x[i] + hh);
      include(x, bt.y[i]);
    }
    include(x, 0);
  }
  for (const bt of [...(axes.bar3Traces ?? []), ...(axes.bar3hTraces ?? [])]) {
    includeAll(x, bt.x);
    includeAll(y, bt.y);
    includeAll(z, bt.z);
    include(z, 0);
  }
  for (const et of axes.errorBarTraces ?? []) {
    for (let i = 0; i < et.x.length; i++) {
      include(x, et.xNeg ? et.x[i] - et.xNeg[i] : et.x[i]);
      include(x, et.xPos ? et.x[i] + et.xPos[i] : et.x[i]);
      include(y, et.y[i] - et.yNeg[i]);
      include(y, et.y[i] + et.yPos[i]);
    }
  }
  for (const bt of axes.boxTraces ?? []) {
    const hw = bt.width / 2;
    include(x, bt.x - hw);
    include(x, bt.x + hw);
    include(y, bt.whiskerLow);
    include(y, bt.whiskerHigh);
    for (const o of bt.outliers) include(y, o);
  }
  if (axes.areaTraces && axes.areaTraces.length > 0) {
    const base = axes.areaBaseValue ?? 0;
    include(y, base);
    const n = axes.areaTraces[0].x.length;
    for (let i = 0; i < n; i++) {
      let cum = base;
      for (const t of axes.areaTraces) {
        if (i < t.x.length) {
          include(x, t.x[i]);
          cum += t.y[i] - base;
          include(y, cum);
        }
      }
    }
  }
  for (const qt of axes.quiverTraces ?? []) {
    for (let i = 0; i < qt.x.length; i++) {
      include(x, qt.x[i]);
      include(x, qt.x[i] + qt.u[i]);
      include(y, qt.y[i]);
      include(y, qt.y[i] + qt.v[i]);
    }
  }
  for (const qt of axes.quiver3Traces ?? []) {
    for (let i = 0; i < qt.x.length; i++) {
      include(x, qt.x[i]);
      include(x, qt.x[i] + qt.u[i]);
      include(y, qt.y[i]);
      include(y, qt.y[i] + qt.v[i]);
      include(z, qt.z[i]);
      include(z, qt.z[i] + qt.w[i]);
    }
  }
  return { x, y, z };
}

/** Pad an automatic extent the way a default 2-D view is drawn. `tight`
 *  views use the bare data range. */
function padExtent(e: Extent, tight: boolean): [number, number] {
  if (!Number.isFinite(e.min) || !Number.isFinite(e.max)) return [0, 1];
  let { min, max } = e;
  if (max === min) {
    min -= 1;
    max += 1;
    return [min, max];
  }
  if (tight) return [min, max];
  const pad = (max - min) * 0.05;
  return [min - pad, max + pad];
}

/** Apply an explicit `[lo, hi]` spec (either bound may be `null`) on top of
 *  the auto-derived `[lo, hi]`. */
function applyExplicit(
  auto: [number, number],
  explicit: [number | null, number | null] | undefined
): [number, number] {
  if (!explicit) return auto;
  return [explicit[0] ?? auto[0], explicit[1] ?? auto[1]];
}

/**
 * Compute the limit vector for `lim = axis`.
 * Returns 4 elements `[xmin xmax ymin ymax]` for a 2-D view, or 6 elements
 * `[xmin xmax ymin ymax zmin zmax]` when the axes holds 3-D content.
 */
export function computeAxisLimits(axes: AxesState): number[] {
  const tight =
    axes.axisMode?.includes("tight") || axes.axisMode?.includes("image")
      ? true
      : false;
  const ext = dataExtents(axes);
  const xl = applyExplicit(padExtent(ext.x, tight), axes.xlim);
  const yl = applyExplicit(padExtent(ext.y, tight), axes.ylim);
  if (!axesIs3D(axes)) return [xl[0], xl[1], yl[0], yl[1]];
  const zl = applyExplicit(padExtent(ext.z, tight), axes.zlim);
  return [xl[0], xl[1], yl[0], yl[1], zl[0], zl[1]];
}
