import type { HeatmapTrace } from "../../numbl-core/plotting/types.js";
import {
  colormapLookup,
  type ColormapSpec,
} from "../../numbl-core/plotting/colormaps.js";
import { formatCbVal } from "./colorbar.js";
import type { StandaloneRenderer } from "./types.js";

/** Heatmap with categorical axes, cell value labels and its own colorbar. */
function drawHeatmap(
  ctx: CanvasRenderingContext2D,
  cw: number,
  ch: number,
  hm: HeatmapTrace,
  title: string | undefined,
  colormap: ColormapSpec
) {
  const { data, rows, cols } = hm;
  if (rows === 0 || cols === 0) return;

  // Compute data range for coloring
  let dMin = Infinity;
  let dMax = -Infinity;
  for (const v of data) {
    if (isFinite(v)) {
      if (v < dMin) dMin = v;
      if (v > dMax) dMax = v;
    }
  }
  if (!isFinite(dMin)) return;
  const dRange = dMax - dMin || 1;

  // Measure label widths to compute margins
  ctx.font = "12px sans-serif";
  const yLabels =
    hm.yLabels ?? Array.from({ length: rows }, (_, i) => String(i + 1));
  const xLabels =
    hm.xLabels ?? Array.from({ length: cols }, (_, i) => String(i + 1));

  let maxYLabelW = 0;
  for (const lbl of yLabels) {
    const w = ctx.measureText(lbl).width;
    if (w > maxYLabelW) maxYLabelW = w;
  }

  const titleH = title ? 30 : 10;
  const xLabelH = 24;
  const yLabelW = maxYLabelW + 12;
  const colorbarW = 50;

  const plotL = yLabelW;
  const plotT = titleH;
  const plotW = cw - plotL - colorbarW;
  const plotH = ch - plotT - xLabelH;
  if (plotW <= 0 || plotH <= 0) return;

  const cellW = plotW / cols;
  const cellH = plotH / rows;

  // Title
  if (title) {
    ctx.font = "bold 14px sans-serif";
    ctx.fillStyle = "#000";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(title, cw / 2, 8);
  }

  // Draw cells (column-major: data[j * rows + i] is row i, col j)
  const fontSize = Math.max(8, Math.min(14, Math.min(cellW, cellH) * 0.35));
  ctx.font = `${fontSize}px sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  for (let j = 0; j < cols; j++) {
    for (let i = 0; i < rows; i++) {
      const v = data[j * rows + i];
      const cx = plotL + j * cellW;
      const cy = plotT + i * cellH;

      // Cell color
      const t = isFinite(v) ? (v - dMin) / dRange : 0;
      const [r, g, b] = colormapLookup(t, colormap);
      ctx.fillStyle = `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`;
      ctx.fillRect(cx, cy, cellW, cellH);

      // Cell border
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1;
      ctx.strokeRect(cx, cy, cellW, cellH);

      // Cell value text
      if (isFinite(v) && cellW > 16 && cellH > 12) {
        const lum = r * 0.299 + g * 0.587 + b * 0.114;
        ctx.fillStyle = lum > 0.5 ? "#000" : "#fff";
        const txt = Number.isInteger(v) ? String(v) : v.toFixed(2);
        ctx.fillText(txt, cx + cellW / 2, cy + cellH / 2);
      }
    }
  }

  // X-axis labels
  ctx.font = "11px sans-serif";
  ctx.fillStyle = "#333";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let j = 0; j < cols; j++) {
    ctx.fillText(
      xLabels[j] ?? "",
      plotL + j * cellW + cellW / 2,
      plotT + plotH + 4
    );
  }

  // Y-axis labels
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let i = 0; i < rows; i++) {
    ctx.fillText(yLabels[i] ?? "", plotL - 6, plotT + i * cellH + cellH / 2);
  }

  // Color bar
  const cbL = plotL + plotW + 10;
  const cbW = 16;
  const cbT = plotT;
  const cbH = plotH;
  const nSteps = Math.max(1, Math.round(cbH));
  for (let s = 0; s < nSteps; s++) {
    const t = 1 - s / nSteps; // top = max, bottom = min
    const [r, g, b] = colormapLookup(t, colormap);
    ctx.fillStyle = `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`;
    ctx.fillRect(cbL, cbT + (s / nSteps) * cbH, cbW, cbH / nSteps + 1);
  }
  ctx.strokeStyle = "#999";
  ctx.lineWidth = 1;
  ctx.strokeRect(cbL, cbT, cbW, cbH);

  // Color bar labels
  ctx.font = "10px sans-serif";
  ctx.fillStyle = "#333";
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText(formatCbVal(dMax), cbL + cbW + 4, cbT);
  ctx.textBaseline = "bottom";
  ctx.fillText(formatCbVal(dMin), cbL + cbW + 4, cbT + cbH);
}

export const heatmapRenderer: StandaloneRenderer<"heatmapTrace"> = {
  layer: "standalone",
  draw: ({ ctx, axes, cmap }, trace, width, height) =>
    drawHeatmap(ctx, width, height, trace, axes.title, cmap),
};
