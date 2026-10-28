import type { PlotTrace } from "../../numbl-core/plotting/types.js";
import type { DataRenderer } from "./types.js";

/** Stacked filled areas, drawn from `areaBaseValue` upward. */
function drawAreas(
  ctx: CanvasRenderingContext2D,
  areaTraces: PlotTrace[],
  areaBaseValue: number | undefined,
  toCanvasX: (v: number) => number,
  toCanvasY: (v: number) => number
) {
  const base = areaBaseValue ?? 0;
  const defaultColors: [number, number, number][] = [
    [0.0, 0.447, 0.741],
    [0.85, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933],
    [0.635, 0.078, 0.184],
  ];
  // Build cumulative stacks (bottom-up)
  const n = areaTraces[0].x.length;
  const bottoms: number[][] = [];
  const tops: number[][] = [];
  let prevTop = new Array(n).fill(base);
  for (let ti = 0; ti < areaTraces.length; ti++) {
    const t = areaTraces[ti];
    bottoms.push([...prevTop]);
    const top: number[] = [];
    for (let i = 0; i < n; i++) {
      top.push(prevTop[i] + (i < t.y.length ? t.y[i] - base : 0));
    }
    tops.push(top);
    prevTop = top;
  }
  // Draw from last (top) to first (bottom) so earlier areas are on top visually
  for (let ti = areaTraces.length - 1; ti >= 0; ti--) {
    const t = areaTraces[ti];
    const [r, g, b] = t.color ?? defaultColors[ti % defaultColors.length];
    // Fill
    ctx.fillStyle = `rgba(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)},0.6)`;
    ctx.beginPath();
    // Top edge (left to right)
    for (let i = 0; i < n; i++) {
      const cx = toCanvasX(t.x[i]);
      const cy = toCanvasY(tops[ti][i]);
      if (i === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    }
    // Bottom edge (right to left)
    for (let i = n - 1; i >= 0; i--) {
      ctx.lineTo(toCanvasX(t.x[i]), toCanvasY(bottoms[ti][i]));
    }
    ctx.closePath();
    ctx.fill();
    // Stroke top edge
    ctx.strokeStyle = `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const cx = toCanvasX(t.x[i]);
      const cy = toCanvasY(tops[ti][i]);
      if (i === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    }
    ctx.stroke();
  }
}

export const areaRenderer: DataRenderer<"areaTraces"> = {
  layer: "data",
  draw: ({ ctx, axes }, traces, { toCanvasX, toCanvasY }) =>
    drawAreas(ctx, traces, axes.areaBaseValue, toCanvasX, toCanvasY),
};
