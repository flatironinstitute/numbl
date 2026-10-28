/**
 * Pure computation of the current axis limits for an `AxesState`.
 *
 * This is the source of truth for the query form `lim = axis`: the numbl
 * runtime reduces its accumulated plot instructions into an `AxesState` and
 * asks this module for the limit vector. It is rendering-free (no canvas, no
 * DOM); the renderer in src/graphics/ uses it too.
 *
 * For an axis with an explicit limit, that limit is returned verbatim (a
 * `null` bound falls back to the data-derived bound). For a fully automatic
 * axis the data extent is computed across every trace type and padded the
 * same way `drawPlot` pads a default 2-D view, so the reported limits track
 * what is drawn.
 */

import type { AxesState } from "./figuresReducer.js";
import {
  TRACE_FIELD_NAMES,
  traceFieldContent,
  traceFieldSpec,
  type Extent,
  type Extents,
  type TraceField,
} from "./axesTraces.js";

const EMPTY: Extent = { min: Infinity, max: -Infinity };

function fieldIs3D<F extends TraceField>(axes: AxesState, field: F): boolean {
  const v = traceFieldContent(axes, field);
  return v !== undefined && traceFieldSpec(field).is3D(v);
}

/** Does this axes hold any 3-D content (so it is drawn by the 3-D view and
 *  its limits are a 6-vector)? A patch counts only when its vertices span a
 *  non-trivial z range; a flat patch (e.g. trimesh(...,0*x,...) under
 *  view(2)) stays 2-D. */
export function axesIs3D(axes: AxesState): boolean {
  return TRACE_FIELD_NAMES.some(f => fieldIs3D(axes, f));
}

function addFieldExtents<F extends TraceField>(
  axes: Partial<AxesState>,
  field: F,
  ext: Extents
): void {
  const v = axes[field];
  if (v === undefined || v === null) return;
  traceFieldSpec(field).extents(v, ext, axes);
}

/** Raw data extents (no padding) across every renderable trace type.
 *  Non-finite values are ignored; an empty axis has min > max. */
export function dataExtents(axes: Partial<AxesState>): Extents {
  const ext: Extents = { x: { ...EMPTY }, y: { ...EMPTY }, z: { ...EMPTY } };
  for (const f of TRACE_FIELD_NAMES) addFieldExtents(axes, f, ext);
  return ext;
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
  // As in drawPlot, axes holding an image are tight.
  const tight =
    axes.axisMode?.includes("tight") ||
    axes.axisMode?.includes("image") ||
    axes.imagescTrace !== undefined
      ? true
      : false;
  const ext = dataExtents(axes);
  const xl = applyExplicit(padExtent(ext.x, tight), axes.xlim);
  const yl = applyExplicit(padExtent(ext.y, tight), axes.ylim);
  if (!axesIs3D(axes)) return [xl[0], xl[1], yl[0], yl[1]];
  const zl = applyExplicit(padExtent(ext.z, tight), axes.zlim);
  return [xl[0], xl[1], yl[0], yl[1], zl[0], zl[1]];
}
