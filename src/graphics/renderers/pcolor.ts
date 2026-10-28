import type { PcolorTrace } from "../../numbl-core/plotting/types.js";
import {
  colormapLookup,
  type ColormapSpec,
} from "../../numbl-core/plotting/colormaps.js";
import type { DataRenderer } from "./types.js";

/**
 * Draw a single pcolor trace. Each cell (i,j) for i in [0,rows-1) and
 * j in [0,cols-1) is rendered as a quadrilateral with corners
 *   (x[i,j],     y[i,j]),
 *   (x[i,j+1],   y[i,j+1]),
 *   (x[i+1,j+1], y[i+1,j+1]),
 *   (x[i+1,j],   y[i+1,j]).
 *
 * MATLAB convention: the cell color comes from the (i,j) corner of C,
 * so the last row and last column of C are not displayed.
 *
 * Shading dispatch:
 *   - faceted (default): flat fill + edge stroke
 *   - flat:              flat fill, no edge
 *   - interp:            placeholder, currently calls flat (TODO)
 */
function drawPcolor(
  ctx: CanvasRenderingContext2D,
  trace: PcolorTrace,
  toCanvasX: (v: number) => number,
  toCanvasY: (v: number) => number,
  colormap: ColormapSpec,
  shading: "faceted" | "flat" | "interp" | undefined,
  caxis?: [number, number]
) {
  const mode = shading ?? "faceted";
  if (mode === "interp") {
    // TODO: implement bilinear interpolation across cell corners.
    drawPcolorFlat(ctx, trace, toCanvasX, toCanvasY, colormap, false, caxis);
    return;
  }
  drawPcolorFlat(
    ctx,
    trace,
    toCanvasX,
    toCanvasY,
    colormap,
    mode === "faceted",
    caxis
  );
}

function drawPcolorFlat(
  ctx: CanvasRenderingContext2D,
  trace: PcolorTrace,
  toCanvasX: (v: number) => number,
  toCanvasY: (v: number) => number,
  colormap: ColormapSpec,
  drawEdges: boolean,
  caxis?: [number, number]
) {
  const { rows, cols, x, y, c } = trace;
  if (rows < 2 || cols < 2) return;

  // Compute color range (use caxis if provided)
  let cMin: number, cMax: number;
  if (caxis) {
    [cMin, cMax] = caxis;
  } else {
    cMin = Infinity;
    cMax = -Infinity;
    for (const v of c) {
      if (isFinite(v)) {
        if (v < cMin) cMin = v;
        if (v > cMax) cMax = v;
      }
    }
  }
  if (!isFinite(cMin)) return;
  const cRange = cMax - cMin || 1;

  const alpha = trace.faceAlpha ?? 1;
  ctx.save();
  if (alpha < 1) ctx.globalAlpha = alpha;

  // Each cell (i,j) for i in [0..rows-2], j in [0..cols-2]
  for (let j = 0; j < cols - 1; j++) {
    for (let i = 0; i < rows - 1; i++) {
      const idx00 = j * rows + i;
      const idx10 = j * rows + (i + 1);
      const idx01 = (j + 1) * rows + i;
      const idx11 = (j + 1) * rows + (i + 1);

      const cv = c[idx00];
      if (!isFinite(cv)) continue;

      const t = (cv - cMin) / cRange;
      const [r, g, b] = colormapLookup(t, colormap);
      const fill = `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`;

      const x0 = toCanvasX(x[idx00]);
      const y0 = toCanvasY(y[idx00]);
      const x1 = toCanvasX(x[idx10]);
      const y1 = toCanvasY(y[idx10]);
      const x2 = toCanvasX(x[idx11]);
      const y2 = toCanvasY(y[idx11]);
      const x3 = toCanvasX(x[idx01]);
      const y3 = toCanvasY(y[idx01]);

      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.lineTo(x3, y3);
      ctx.closePath();
      ctx.fillStyle = fill;
      ctx.fill();

      if (drawEdges && trace.edgeColor !== "none") {
        if (Array.isArray(trace.edgeColor)) {
          const [er, eg, eb] = trace.edgeColor;
          ctx.strokeStyle = `rgb(${Math.round(er * 255)},${Math.round(eg * 255)},${Math.round(eb * 255)})`;
        } else {
          ctx.strokeStyle = "rgba(0,0,0,0.3)";
        }
        ctx.lineWidth = 0.5;
        ctx.stroke();
      } else if (trace.edgeColor === "none") {
        // Stroke with fill color to cover anti-aliasing seams between cells
        ctx.strokeStyle = fill;
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }
  }

  ctx.restore();
}

export const pcolorRenderer: DataRenderer<"pcolorTraces"> = {
  layer: "data",
  draw: ({ ctx, cmap, axes }, traces, { toCanvasX, toCanvasY }) => {
    for (const pt of traces)
      drawPcolor(ctx, pt, toCanvasX, toCanvasY, cmap, axes.shading, axes.caxis);
  },
  colorValues: (traces, consume) => {
    for (const pt of traces) pt.c.forEach(consume);
  },
};
