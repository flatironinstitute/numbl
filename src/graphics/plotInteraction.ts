/**
 * Interactive view changes on a 2-D axes (pan, zoom, color limits).
 *
 * The user's changes are kept as a `ViewOverride` beside the axes state, not
 * in it, and applied as explicit `xlim` / `ylim` / `caxis` at draw time. So
 * they persist while new data streams in, until the user resets them.
 */

import type { AxesState } from "../numbl-core/plotting/figuresReducer.js";

export interface ViewOverride {
  xlim?: [number, number];
  ylim?: [number, number];
  caxis?: [number, number];
}

/** Where drawPlot put things, in CSS pixels, and the limits it drew. */
export interface PlotLayout {
  plot: Rect;
  x: [number, number];
  y: [number, number];
  logX: boolean;
  logY: boolean;
  yReversed: boolean;
  /** Color limits in effect, if anything is drawn through the colormap. */
  colorRange?: [number, number];
  colorbar?: Rect & { vertical: boolean };
}

export interface Rect {
  left: number;
  top: number;
  width: number;
  height: number;
}

export function applyView(axes: AxesState, view: ViewOverride | null) {
  if (!view) return axes;
  return {
    ...axes,
    ...(view.xlim && { xlim: view.xlim }),
    ...(view.ylim && { ylim: view.ylim }),
    ...(view.caxis && { caxis: view.caxis }),
  };
}

export function inRect(r: Rect, px: number, py: number): boolean {
  return (
    px >= r.left &&
    px <= r.left + r.width &&
    py >= r.top &&
    py <= r.top + r.height
  );
}

const fwd = (v: number, log: boolean) => (log ? Math.log10(v) : v);
const inv = (v: number, log: boolean) => (log ? Math.pow(10, v) : v);

/** Scale `range` by `factor` about `center` (factor < 1 zooms in). Works in
 *  log space for a log axis. */
export function zoomRange(
  range: [number, number],
  center: number,
  factor: number,
  log = false
): [number, number] {
  const a = fwd(range[0], log);
  const b = fwd(range[1], log);
  const c = fwd(center, log);
  return [inv(c - (c - a) * factor, log), inv(c + (b - c) * factor, log)];
}

/** Shift `range` by `frac` of its width (in log space for a log axis). */
export function shiftRange(
  range: [number, number],
  frac: number,
  log = false
): [number, number] {
  const a = fwd(range[0], log);
  const b = fwd(range[1], log);
  const d = (b - a) * frac;
  return [inv(a + d, log), inv(b + d, log)];
}

/** The value at fraction `frac` (0..1) along `range`. */
export function valueAt(
  range: [number, number],
  frac: number,
  log = false
): number {
  const a = fwd(range[0], log);
  const b = fwd(range[1], log);
  return inv(a + frac * (b - a), log);
}

/** Fractions along the x and y axes of a canvas point (0 at xMin / yMin). */
export function plotFractions(
  layout: PlotLayout,
  px: number,
  py: number
): { fx: number; fy: number } {
  const { left, top, width, height } = layout.plot;
  const fx = (px - left) / width;
  const down = (py - top) / height;
  return { fx, fy: layout.yReversed ? down : 1 - down };
}

/** Fraction along the colorbar (0 at the low color limit). */
export function colorbarFraction(
  cb: NonNullable<PlotLayout["colorbar"]>,
  px: number,
  py: number
): number {
  return cb.vertical
    ? 1 - (py - cb.top) / cb.height
    : (px - cb.left) / cb.width;
}

/** Wheel delta in pixels, whatever the event's delta mode. */
export function wheelPixels(e: WheelEvent): number {
  const d = e.deltaY !== 0 ? e.deltaY : e.deltaX;
  return e.deltaMode === 1 ? d * 16 : e.deltaMode === 2 ? d * 400 : d;
}

/** Zoom factor for a wheel movement of `px` pixels (> 1 zooms out). */
export function wheelFactor(px: number): number {
  return Math.exp(Math.max(-300, Math.min(300, px)) * 0.002);
}
