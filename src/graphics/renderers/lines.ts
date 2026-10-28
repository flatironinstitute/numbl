import type { PlotTrace } from "../../numbl-core/plotting/types.js";
import { traceColor, getLineDash } from "../plotHelpers.js";
import { drawMarkers } from "../plotMarkers.js";
import type { DataRenderer } from "./types.js";

/** Line plots (plot, line, scatter, stairs, ...): polylines broken at
 *  non-finite points, plus markers. */
function drawLines(
  ctx: CanvasRenderingContext2D,
  traces: PlotTrace[],
  toCanvasX: (v: number) => number,
  toCanvasY: (v: number) => number
) {
  for (let ti = 0; ti < traces.length; ti++) {
    const t = traces[ti];
    const color = traceColor(t, ti);

    // Draw line (unless lineStyle is 'none')
    if (t.lineStyle !== "none") {
      ctx.strokeStyle = color;
      ctx.lineWidth = t.lineWidth ?? 2;
      ctx.lineJoin = "round";
      ctx.setLineDash(getLineDash(t.lineStyle));

      ctx.beginPath();
      let penDown = false;
      for (let i = 0; i < t.x.length; i++) {
        const vx = t.x[i];
        const vy = t.y[i];
        if (!isFinite(vx) || !isFinite(vy)) {
          // NaN/Inf creates a break in the line
          penDown = false;
          continue;
        }
        const cx = toCanvasX(vx);
        const cy = toCanvasY(vy);
        if (!penDown) {
          ctx.moveTo(cx, cy);
          penDown = true;
        } else {
          ctx.lineTo(cx, cy);
        }
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Draw markers
    if (t.marker) {
      drawMarkers(ctx, t, toCanvasX, toCanvasY, color);
    }
  }
}

export const linesRenderer: DataRenderer<"traces"> = {
  layer: "data",
  draw: ({ ctx }, traces, { toCanvasX, toCanvasY }) =>
    drawLines(ctx, traces, toCanvasX, toCanvasY),
};
