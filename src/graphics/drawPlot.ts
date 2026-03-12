import type { PlotTrace } from "../numbl-core/runtime/plotUtils.js";
import {
  traceColor,
  getLineDash,
  generateTicks,
  formatTick,
} from "./plotHelpers.js";
import { drawMarkers } from "./plotMarkers.js";
import { drawLegend } from "./plotLegend.js";

export function drawPlot(
  canvas: HTMLCanvasElement,
  traces: PlotTrace[],
  title?: string,
  xlabel?: string,
  ylabel?: string,
  legend?: string[],
  gridOn?: boolean
) {
  const ctx = canvas.getContext("2d");
  if (!ctx || traces.length === 0) return;

  const dpr = window.devicePixelRatio || 1;
  const w = canvas.width;
  const h = canvas.height;

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const cw = w / dpr;
  const ch = h / dpr;

  // Clear with white background
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, cw, ch);

  // Margins — expand to accommodate labels
  const margin = {
    top: title ? 40 : 20,
    right: 20,
    bottom: xlabel ? 56 : 40,
    left: ylabel ? 76 : 60,
  };
  const plotW = cw - margin.left - margin.right;
  const plotH = ch - margin.top - margin.bottom;

  if (plotW <= 0 || plotH <= 0) return;

  // Compute data range across all traces
  let xMin = Infinity,
    xMax = -Infinity,
    yMin = Infinity,
    yMax = -Infinity;
  for (const t of traces) {
    for (const v of t.x) {
      if (isFinite(v)) {
        if (v < xMin) xMin = v;
        if (v > xMax) xMax = v;
      }
    }
    for (const v of t.y) {
      if (isFinite(v)) {
        if (v < yMin) yMin = v;
        if (v > yMax) yMax = v;
      }
    }
  }

  if (!isFinite(xMin)) return;

  // Add padding if range is zero
  if (xMax === xMin) {
    xMin -= 1;
    xMax += 1;
  }
  if (yMax === yMin) {
    yMin -= 1;
    yMax += 1;
  }

  // Small margin around data
  const xPad = (xMax - xMin) * 0.05;
  const yPad = (yMax - yMin) * 0.05;
  xMin -= xPad;
  xMax += xPad;
  yMin -= yPad;
  yMax += yPad;

  const toCanvasX = (v: number) =>
    margin.left + ((v - xMin) / (xMax - xMin)) * plotW;
  const toCanvasY = (v: number) =>
    margin.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

  // Grid (only when gridOn is true or undefined — default on for backward compat)
  if (gridOn !== false) {
    ctx.strokeStyle = "#ddd";
    ctx.lineWidth = 0.5;

    const xTicksGrid = generateTicks(
      xMin,
      xMax,
      Math.max(3, Math.floor(plotW / 80))
    );
    const yTicksGrid = generateTicks(
      yMin,
      yMax,
      Math.max(3, Math.floor(plotH / 50))
    );

    for (const tx of xTicksGrid) {
      const cx = toCanvasX(tx);
      ctx.beginPath();
      ctx.moveTo(cx, margin.top);
      ctx.lineTo(cx, margin.top + plotH);
      ctx.stroke();
    }
    for (const ty of yTicksGrid) {
      const cy = toCanvasY(ty);
      ctx.beginPath();
      ctx.moveTo(margin.left, cy);
      ctx.lineTo(margin.left + plotW, cy);
      ctx.stroke();
    }
  }

  // Plot border
  ctx.strokeStyle = "#333";
  ctx.lineWidth = 1;
  ctx.strokeRect(margin.left, margin.top, plotW, plotH);

  // Tick labels
  const xTicks = generateTicks(xMin, xMax, Math.max(3, Math.floor(plotW / 80)));
  const yTicks = generateTicks(yMin, yMax, Math.max(3, Math.floor(plotH / 50)));

  ctx.fillStyle = "#333";
  ctx.font = "11px monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (const tx of xTicks) {
    ctx.fillText(formatTick(tx), toCanvasX(tx), margin.top + plotH + 5);
  }
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (const ty of yTicks) {
    ctx.fillText(formatTick(ty), margin.left - 5, toCanvasY(ty));
  }

  // Labels
  ctx.fillStyle = "#222";
  if (title) {
    ctx.font = "bold 13px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(title, margin.left + plotW / 2, margin.top / 2);
  }
  if (xlabel) {
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText(xlabel, margin.left + plotW / 2, ch - 4);
  }
  if (ylabel) {
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.save();
    ctx.translate(14, margin.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(ylabel, 0, 0);
    ctx.restore();
  }

  // Data lines — clip to plot area
  ctx.save();
  ctx.beginPath();
  ctx.rect(margin.left, margin.top, plotW, plotH);
  ctx.clip();

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

  ctx.restore();

  // Legend
  if (legend && legend.length > 0) {
    drawLegend(ctx, traces, legend, margin.left + plotW, margin.top);
  }
}
