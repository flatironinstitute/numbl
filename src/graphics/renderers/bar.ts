import type { BarTrace } from "../../numbl-core/plotting/types.js";
import type { DataRenderer } from "./types.js";

function drawBars(
  ctx: CanvasRenderingContext2D,
  barTraces: BarTrace[],
  toCanvasX: (v: number) => number,
  toCanvasY: (v: number) => number
) {
  const defaultColors: [number, number, number][] = [
    [0.0, 0.447, 0.741],
    [0.85, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
  ];
  for (let bi = 0; bi < barTraces.length; bi++) {
    const bt = barTraces[bi];
    const [r, g, b] = bt.color ?? defaultColors[bi % defaultColors.length];
    ctx.fillStyle = `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`;
    ctx.strokeStyle = `rgb(${Math.round(r * 180)},${Math.round(g * 180)},${Math.round(b * 180)})`;
    ctx.lineWidth = 1;
    const y0 = toCanvasY(0);
    for (let i = 0; i < bt.x.length; i++) {
      const cx = toCanvasX(bt.x[i]);
      const cy = toCanvasY(bt.y[i]);
      const halfW = (toCanvasX(bt.x[i] + bt.width) - toCanvasX(bt.x[i])) / 2;
      ctx.fillRect(cx - halfW, Math.min(cy, y0), halfW * 2, Math.abs(cy - y0));
      ctx.strokeRect(
        cx - halfW,
        Math.min(cy, y0),
        halfW * 2,
        Math.abs(cy - y0)
      );
    }
  }
}

/** Horizontal bars: `x` is the position on the y-axis, `y` the bar length. */
function drawBarsH(
  ctx: CanvasRenderingContext2D,
  barhTraces: BarTrace[],
  toCanvasX: (v: number) => number,
  toCanvasY: (v: number) => number
) {
  const defaultColors: [number, number, number][] = [
    [0.0, 0.447, 0.741],
    [0.85, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
  ];
  for (let bi = 0; bi < barhTraces.length; bi++) {
    const bt = barhTraces[bi];
    const [r, g, b] = bt.color ?? defaultColors[bi % defaultColors.length];
    ctx.fillStyle = `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`;
    ctx.strokeStyle = `rgb(${Math.round(r * 180)},${Math.round(g * 180)},${Math.round(b * 180)})`;
    ctx.lineWidth = 1;
    const x0 = toCanvasX(0);
    for (let i = 0; i < bt.x.length; i++) {
      const cy = toCanvasY(bt.x[i]); // position on y-axis
      const cx = toCanvasX(bt.y[i]); // bar length on x-axis
      const halfH = (toCanvasY(bt.x[i] - bt.width) - toCanvasY(bt.x[i])) / 2;
      ctx.fillRect(Math.min(x0, cx), cy - halfH, Math.abs(cx - x0), halfH * 2);
      ctx.strokeRect(
        Math.min(x0, cx),
        cy - halfH,
        Math.abs(cx - x0),
        halfH * 2
      );
    }
  }
}

export const barRenderer: DataRenderer<"barTraces"> = {
  layer: "data",
  draw: ({ ctx }, traces, { toCanvasX, toCanvasY }) =>
    drawBars(ctx, traces, toCanvasX, toCanvasY),
};

export const barhRenderer: DataRenderer<"barhTraces"> = {
  layer: "data",
  draw: ({ ctx }, traces, { toCanvasX, toCanvasY }) =>
    drawBarsH(ctx, traces, toCanvasX, toCanvasY),
};
