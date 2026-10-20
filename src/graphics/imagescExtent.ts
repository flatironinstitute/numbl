import type { ImagescTrace } from "./types.js";

/**
 * Pixel geometry of an imagesc trace. As in MATLAB, `x` and `y` hold the
 * centers of the first and last columns/rows (XData/YData), so the image
 * extends half a pixel beyond them; a single column or row is one unit wide.
 * `dx`/`dy` may be negative (decreasing XData/YData).
 */
export function imagescExtent(t: ImagescTrace): {
  dx: number;
  dy: number;
  x: [number, number];
  y: [number, number];
} {
  const dx = t.cols > 1 ? (t.x[1] - t.x[0]) / (t.cols - 1) : 1;
  const dy = t.rows > 1 ? (t.y[1] - t.y[0]) / (t.rows - 1) : 1;
  const xa = t.x[0] - dx / 2;
  const xb = t.x[0] + (t.cols - 0.5) * dx;
  const ya = t.y[0] - dy / 2;
  const yb = t.y[0] + (t.rows - 0.5) * dy;
  return {
    dx,
    dy,
    x: [Math.min(xa, xb), Math.max(xa, xb)],
    y: [Math.min(ya, yb), Math.max(ya, yb)],
  };
}

/**
 * Color limits for an imagesc trace: the axes' caxis/clim when set, else the
 * finite data range. Values outside the limits clamp to the colormap ends.
 * Returns null when there is no finite data and no caxis.
 */
export function imagescColorLimits(
  t: ImagescTrace,
  caxis?: [number, number]
): [number, number] | null {
  if (caxis) return caxis;
  let lo = Infinity,
    hi = -Infinity;
  for (const v of t.z) {
    if (isFinite(v)) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  return isFinite(lo) ? [lo, hi] : null;
}
