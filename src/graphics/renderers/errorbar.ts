import type { ErrorBarTrace } from "../../numbl-core/plotting/types.js";
import { traceColor } from "../plotHelpers.js";
import type { DataRenderer } from "./types.js";

function drawErrorBars(
  ctx: CanvasRenderingContext2D,
  errorBarTraces: ErrorBarTrace[],
  toCanvasX: (v: number) => number,
  toCanvasY: (v: number) => number,
  dpr: number
) {
  for (let ei = 0; ei < errorBarTraces.length; ei++) {
    const et = errorBarTraces[ei];
    const colorStr = et.color
      ? `rgb(${Math.round(et.color[0] * 255)},${Math.round(et.color[1] * 255)},${Math.round(et.color[2] * 255)})`
      : traceColor({ x: [], y: [] }, ei);
    ctx.strokeStyle = colorStr;
    ctx.fillStyle = colorStr;
    ctx.lineWidth = (et.lineWidth ?? 1) * dpr;
    const capHalf = 3 * dpr;

    // Draw the connecting line
    ctx.beginPath();
    for (let i = 0; i < et.x.length; i++) {
      const cx = toCanvasX(et.x[i]);
      const cy = toCanvasY(et.y[i]);
      if (i === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    }
    ctx.stroke();

    // Draw error bars
    for (let i = 0; i < et.x.length; i++) {
      const cx = toCanvasX(et.x[i]);
      const cy = toCanvasY(et.y[i]);

      // Vertical error bars
      const yLo = toCanvasY(et.y[i] - et.yNeg[i]);
      const yHi = toCanvasY(et.y[i] + et.yPos[i]);
      ctx.beginPath();
      ctx.moveTo(cx, yLo);
      ctx.lineTo(cx, yHi);
      ctx.stroke();
      // Caps
      ctx.beginPath();
      ctx.moveTo(cx - capHalf, yLo);
      ctx.lineTo(cx + capHalf, yLo);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(cx - capHalf, yHi);
      ctx.lineTo(cx + capHalf, yHi);
      ctx.stroke();

      // Horizontal error bars
      if (et.xNeg && et.xPos) {
        const xL = toCanvasX(et.x[i] - et.xNeg[i]);
        const xR = toCanvasX(et.x[i] + et.xPos[i]);
        ctx.beginPath();
        ctx.moveTo(xL, cy);
        ctx.lineTo(xR, cy);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(xL, cy - capHalf);
        ctx.lineTo(xL, cy + capHalf);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(xR, cy - capHalf);
        ctx.lineTo(xR, cy + capHalf);
        ctx.stroke();
      }

      // Draw marker at data point
      ctx.beginPath();
      ctx.arc(cx, cy, 2 * dpr, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
}

export const errorBarRenderer: DataRenderer<"errorBarTraces"> = {
  layer: "data",
  draw: ({ ctx, dpr }, traces, { toCanvasX, toCanvasY }) =>
    drawErrorBars(ctx, traces, toCanvasX, toCanvasY, dpr),
};
