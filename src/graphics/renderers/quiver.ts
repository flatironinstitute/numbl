import type { QuiverTrace } from "../../numbl-core/plotting/types.js";
import { getLineDash } from "../plotHelpers.js";
import type { DataRenderer } from "./types.js";

const QUIVER_DEFAULT_COLORS: [number, number, number][] = [
  [0.0, 0.447, 0.741],
  [0.85, 0.325, 0.098],
  [0.929, 0.694, 0.125],
  [0.494, 0.184, 0.556],
  [0.466, 0.674, 0.188],
];

function rgbStr(c: [number, number, number]): string {
  return `rgb(${Math.round(c[0] * 255)},${Math.round(c[1] * 255)},${Math.round(c[2] * 255)})`;
}

function drawQuiver(
  ctx: CanvasRenderingContext2D,
  quiverTraces: QuiverTrace[],
  toCanvasX: (v: number) => number,
  toCanvasY: (v: number) => number,
  dpr: number
) {
  for (let qi = 0; qi < quiverTraces.length; qi++) {
    const qt = quiverTraces[qi];
    const color =
      qt.color ?? QUIVER_DEFAULT_COLORS[qi % QUIVER_DEFAULT_COLORS.length];
    const colorStr = rgbStr(color);
    const lineWidth = (qt.lineWidth ?? 0.5) * dpr * 2;

    ctx.strokeStyle = colorStr;
    ctx.fillStyle = colorStr;
    ctx.lineWidth = lineWidth;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.setLineDash(getLineDash(qt.lineStyle));

    const n = qt.x.length;
    for (let i = 0; i < n; i++) {
      const x0 = qt.x[i];
      const y0 = qt.y[i];
      const ux = qt.u[i];
      const vy = qt.v[i];
      if (!isFinite(x0) || !isFinite(y0) || !isFinite(ux) || !isFinite(vy))
        continue;

      const x1 = x0 + ux;
      const y1 = y0 + vy;
      const cx0 = toCanvasX(x0);
      const cy0 = toCanvasY(y0);
      const cx1 = toCanvasX(x1);
      const cy1 = toCanvasY(y1);

      // Shaft
      ctx.beginPath();
      ctx.moveTo(cx0, cy0);
      ctx.lineTo(cx1, cy1);
      ctx.stroke();

      // Arrowhead — pixel-space, ~1/3 of the shaft length but capped
      if (qt.showArrowHead) {
        const dxp = cx1 - cx0;
        const dyp = cy1 - cy0;
        const lenp = Math.sqrt(dxp * dxp + dyp * dyp);
        if (lenp > 1) {
          const headLen = Math.min(lenp * 0.3, 12 * dpr);
          const headAngle = Math.PI / 9; // 20°
          const ang = Math.atan2(dyp, dxp);
          const hx1 = cx1 - headLen * Math.cos(ang - headAngle);
          const hy1 = cy1 - headLen * Math.sin(ang - headAngle);
          const hx2 = cx1 - headLen * Math.cos(ang + headAngle);
          const hy2 = cy1 - headLen * Math.sin(ang + headAngle);
          ctx.beginPath();
          ctx.moveTo(hx1, hy1);
          ctx.lineTo(cx1, cy1);
          ctx.lineTo(hx2, hy2);
          ctx.stroke();
        }
      }

      // Tail marker
      if (qt.marker) {
        const r = 3 * dpr;
        ctx.beginPath();
        ctx.arc(cx0, cy0, r, 0, Math.PI * 2);
        if (qt.markerFilled) ctx.fill();
        else ctx.stroke();
      }
    }

    ctx.setLineDash([]);
  }
}

export const quiverRenderer: DataRenderer<"quiverTraces"> = {
  layer: "data",
  draw: ({ ctx, dpr }, traces, { toCanvasX, toCanvasY }) =>
    drawQuiver(ctx, traces, toCanvasX, toCanvasY, dpr),
};
