import type { AxesState } from "../numbl-core/plotting/figuresReducer.js";
import { traceFieldContent } from "../numbl-core/plotting/axesTraces.js";
import { generateTicks, formatTick } from "./plotHelpers.js";
import type { ColormapSpec } from "../numbl-core/plotting/colormaps.js";
import { dataExtents } from "../numbl-core/plotting/axisLimits.js";
import { drawLegend } from "./plotLegend.js";
import { drawColorbarAtLocation } from "./renderers/colorbar.js";
import { CANVAS_RENDER_ORDER, canvasRenderer } from "./renderers/index.js";
import type { RenderContext, PlotTransform } from "./renderers/types.js";

/** Generate tick values at powers of 10 for log-scale axes. */
function generateLogTicks(min: number, max: number): number[] {
  const safeMin = Math.max(min, Number.MIN_VALUE);
  const safeMax = Math.max(max, Number.MIN_VALUE);
  const logMin = Math.floor(Math.log10(safeMin));
  const logMax = Math.ceil(Math.log10(safeMax));
  const ticks: number[] = [];
  for (let p = logMin; p <= logMax; p++) {
    const v = Math.pow(10, p);
    if (v >= safeMin && v <= safeMax) ticks.push(v);
  }
  // If too few ticks, add intermediate values (2, 5) × 10^p
  if (ticks.length < 3) {
    for (let p = logMin; p <= logMax; p++) {
      for (const m of [2, 5]) {
        const v = m * Math.pow(10, p);
        if (v >= safeMin && v <= safeMax && !ticks.includes(v)) ticks.push(v);
      }
    }
    ticks.sort((a, b) => a - b);
  }
  return ticks;
}

/** Format a log-scale tick value (e.g., 10^3 → "1000", 10^-2 → "0.01"). */
function formatLogTick(v: number): string {
  if (v >= 1 && v < 1e6 && Number.isInteger(v)) return v.toString();
  return v.toExponential(0);
}

/** Render one 2-D axes to a canvas. The per-kind drawing lives in the
 *  renderer table (renderers/index.ts); this function owns what the kinds
 *  share: the data limits, axis modes and scales, grid, ticks, labels, legend
 *  and colorbar. 3-D axes are drawn by SurfView instead. */
export function drawPlot(canvas: HTMLCanvasElement, axes: AxesState) {
  const {
    traces,
    title,
    xlabel,
    ylabel,
    legend,
    gridOn,
    boxOn,
    imagescTrace,
    colormap,
    colormapData,
    axisMode,
    axisScale,
    colorbar,
    colorbarLocation,
    caxis,
    xlim,
    ylim,
    yDir,
    axisVisible,
  } = axes;
  const cmap: ColormapSpec = { name: colormap, data: colormapData };
  const ctx = canvas.getContext("2d");
  const hasContent = CANVAS_RENDER_ORDER.some(
    f => traceFieldContent(axes, f) !== undefined
  );
  if (!ctx || !hasContent) return;

  const dpr = window.devicePixelRatio || 1;
  const rc: RenderContext = { ctx, axes, cmap, dpr };
  const w = canvas.width;
  const h = canvas.height;

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const cw = w / dpr;
  const ch = h / dpr;

  // Clear with white background
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, cw, ch);

  // Pie chart / heatmap: a standalone renderer takes over the whole canvas
  // (no shared axes).
  for (const field of CANVAS_RENDER_ORDER) {
    const renderer = canvasRenderer(field);
    const value = traceFieldContent(axes, field);
    if (renderer.layer === "standalone" && value !== undefined) {
      renderer.draw(rc, value, cw, ch);
      return;
    }
  }

  // Margins — expand to accommodate labels
  const margin = {
    top: title ? 40 : 20,
    right: 20,
    bottom: xlabel ? 56 : 40,
    left: ylabel ? 76 : 60,
  };
  // Reserve space for an outside colorbar.
  const cbLoc = (colorbarLocation ?? "eastoutside").toLowerCase();
  if (colorbar) {
    if (cbLoc === "eastoutside") margin.right += 70;
    else if (cbLoc === "westoutside") margin.left += 70;
    else if (cbLoc === "northoutside") margin.top += 50;
    else if (cbLoc === "southoutside") margin.bottom += 50;
  }
  const plotW = cw - margin.left - margin.right;
  const plotH = ch - margin.top - margin.bottom;

  if (plotW <= 0 || plotH <= 0) return;

  // Data range across all traces (shared with `lim = axis`).
  const ext = dataExtents(axes);
  let xMin = ext.x.min,
    xMax = ext.x.max,
    yMin = ext.y.min,
    yMax = ext.y.max;

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

  // Parse axis mode flags. `image` is equal+tight; `square` makes the plot
  // box square (without forcing equal data units); `padded` widens the
  // default margin.
  const isImage = axisMode?.includes("image") ?? false;
  // As in MATLAB, axes holding an image get no margin around the data.
  const isTight =
    (axisMode?.includes("tight") ?? false) ||
    isImage ||
    imagescTrace !== undefined;
  const isEqual = (axisMode?.includes("equal") ?? false) || isImage;
  const isSquare = axisMode?.includes("square") ?? false;
  const isPadded = axisMode?.includes("padded") ?? false;
  const showAxes = axisVisible !== false;
  const yReversed = yDir === "reverse";

  const logX = axisScale === "semilogx" || axisScale === "loglog";
  const logY = axisScale === "semilogy" || axisScale === "loglog";

  // For log axes, clamp mins to positive values
  if (logX && xMin <= 0) xMin = xMax > 0 ? xMax * 1e-6 : 1;
  if (logY && yMin <= 0) yMin = yMax > 0 ? yMax * 1e-6 : 1;

  // Small margin around data (skip if tight). `padded` uses a wider margin.
  if (!isTight) {
    const frac = isPadded ? 0.07 : 0.05;
    if (logX) {
      // Pad in log space
      const logPad = (Math.log10(xMax) - Math.log10(xMin)) * frac;
      xMin = Math.pow(10, Math.log10(xMin) - logPad);
      xMax = Math.pow(10, Math.log10(xMax) + logPad);
    } else {
      const xPad = (xMax - xMin) * frac;
      xMin -= xPad;
      xMax += xPad;
    }
    if (logY) {
      const logPad = (Math.log10(yMax) - Math.log10(yMin)) * frac;
      yMin = Math.pow(10, Math.log10(yMin) - logPad);
      yMax = Math.pow(10, Math.log10(yMax) + logPad);
    } else {
      const yPad = (yMax - yMin) * frac;
      yMin -= yPad;
      yMax += yPad;
    }
  }

  // Explicit limits from `axis([...])` / `xlim` / `ylim` override the
  // computed bounds. A `null` bound keeps the automatic value.
  if (xlim) {
    if (xlim[0] != null) xMin = xlim[0];
    if (xlim[1] != null) xMax = xlim[1];
  }
  if (ylim) {
    if (ylim[0] != null) yMin = ylim[0];
    if (ylim[1] != null) yMax = ylim[1];
  }
  if (xMax === xMin) {
    xMin -= 1;
    xMax += 1;
  }
  if (yMax === yMin) {
    yMin -= 1;
    yMax += 1;
  }

  // axis equal: ensure 1 data unit = same pixel length on both axes
  // With tight: shrink plot area to fit data. Without tight: expand data range.
  let effPlotW = plotW;
  let effPlotH = plotH;
  let effMarginLeft = margin.left;
  let effMarginTop = margin.top;
  if (isEqual) {
    const xRange = xMax - xMin;
    const yRange = yMax - yMin;
    const dataPerPxX = xRange / plotW;
    const dataPerPxY = yRange / plotH;
    if (isTight) {
      // Shrink the plot area to preserve aspect ratio
      if (dataPerPxX > dataPerPxY) {
        // X needs more space per unit — shrink plot height
        effPlotH = yRange / dataPerPxX;
        effMarginTop = margin.top + (plotH - effPlotH) / 2;
      } else {
        // Y needs more space per unit — shrink plot width
        effPlotW = xRange / dataPerPxY;
        effMarginLeft = margin.left + (plotW - effPlotW) / 2;
      }
    } else {
      // Expand data range to fill plot area
      if (dataPerPxX > dataPerPxY) {
        const newYRange = dataPerPxX * plotH;
        const yCenter = (yMin + yMax) / 2;
        yMin = yCenter - newYRange / 2;
        yMax = yCenter + newYRange / 2;
      } else {
        const newXRange = dataPerPxY * plotW;
        const xCenter = (xMin + xMax) / 2;
        xMin = xCenter - newXRange / 2;
        xMax = xCenter + newXRange / 2;
      }
    }
  } else if (isSquare) {
    // axis square: make the plot box square (equal axis-line lengths),
    // without forcing equal data units. Shrink the longer side and center.
    const side = Math.min(plotW, plotH);
    effMarginLeft = margin.left + (plotW - side) / 2;
    effMarginTop = margin.top + (plotH - side) / 2;
    effPlotW = side;
    effPlotH = side;
  }

  const toCanvasX = logX
    ? (v: number) => {
        const logV = Math.log10(Math.max(v, Number.MIN_VALUE));
        const logMin = Math.log10(Math.max(xMin, Number.MIN_VALUE));
        const logMax = Math.log10(Math.max(xMax, Number.MIN_VALUE));
        return effMarginLeft + ((logV - logMin) / (logMax - logMin)) * effPlotW;
      }
    : (v: number) => effMarginLeft + ((v - xMin) / (xMax - xMin)) * effPlotW;
  // `axis ij` (yReversed) places the origin at the top-left: y increases
  // downward instead of upward.
  const toCanvasY = logY
    ? (v: number) => {
        const logV = Math.log10(Math.max(v, Number.MIN_VALUE));
        const logMin = Math.log10(Math.max(yMin, Number.MIN_VALUE));
        const logMax = Math.log10(Math.max(yMax, Number.MIN_VALUE));
        const frac = (logV - logMin) / (logMax - logMin);
        return yReversed
          ? effMarginTop + frac * effPlotH
          : effMarginTop + effPlotH - frac * effPlotH;
      }
    : (v: number) => {
        const frac = (v - yMin) / (yMax - yMin);
        return yReversed
          ? effMarginTop + frac * effPlotH
          : effMarginTop + effPlotH - frac * effPlotH;
      };

  // Grid (only when gridOn is true or undefined — default on for backward compat)
  if (gridOn !== false && showAxes) {
    ctx.strokeStyle = "#ddd";
    ctx.lineWidth = 0.5;

    const xTicksGrid = logX
      ? generateLogTicks(xMin, xMax)
      : generateTicks(xMin, xMax, Math.max(3, Math.floor(effPlotW / 80)));
    const yTicksGrid = logY
      ? generateLogTicks(yMin, yMax)
      : generateTicks(yMin, yMax, Math.max(3, Math.floor(effPlotH / 50)));

    for (const tx of xTicksGrid) {
      const cx = toCanvasX(tx);
      ctx.beginPath();
      ctx.moveTo(cx, effMarginTop);
      ctx.lineTo(cx, effMarginTop + effPlotH);
      ctx.stroke();
    }
    for (const ty of yTicksGrid) {
      const cy = toCanvasY(ty);
      ctx.beginPath();
      ctx.moveTo(effMarginLeft, cy);
      ctx.lineTo(effMarginLeft + effPlotW, cy);
      ctx.stroke();
    }
  }

  // Plot border + tick labels. Hidden by `axis off` (Visible='off'), which
  // suppresses the axes lines, ticks, and tick labels.
  if (showAxes) {
    ctx.strokeStyle = "#333";
    ctx.lineWidth = 1;
    // `box on` (default) draws the full rectangle; `box off` draws only the
    // left and bottom axis lines (MATLAB's L-shaped axes).
    if (boxOn === false) {
      ctx.beginPath();
      ctx.moveTo(effMarginLeft, effMarginTop);
      ctx.lineTo(effMarginLeft, effMarginTop + effPlotH);
      ctx.lineTo(effMarginLeft + effPlotW, effMarginTop + effPlotH);
      ctx.stroke();
    } else {
      ctx.strokeRect(effMarginLeft, effMarginTop, effPlotW, effPlotH);
    }

    const xTicks = logX
      ? generateLogTicks(xMin, xMax)
      : generateTicks(xMin, xMax, Math.max(3, Math.floor(effPlotW / 80)));
    const yTicks = logY
      ? generateLogTicks(yMin, yMax)
      : generateTicks(yMin, yMax, Math.max(3, Math.floor(effPlotH / 50)));

    ctx.fillStyle = "#333";
    ctx.font = "11px monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const tx of xTicks) {
      ctx.fillText(
        logX ? formatLogTick(tx) : formatTick(tx),
        toCanvasX(tx),
        effMarginTop + effPlotH + 5
      );
    }
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const ty of yTicks) {
      ctx.fillText(
        logY ? formatLogTick(ty) : formatTick(ty),
        effMarginLeft - 5,
        toCanvasY(ty)
      );
    }
  }

  // Labels. The title stays visible with `axis off` (it tracks the axes
  // title, not the axis lines); the x/y axis labels are hidden.
  ctx.fillStyle = "#222";
  if (title) {
    ctx.font = "bold 13px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(title, effMarginLeft + effPlotW / 2, effMarginTop / 2);
  }
  if (xlabel && showAxes) {
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText(xlabel, effMarginLeft + effPlotW / 2, ch - 4);
  }
  if (ylabel && showAxes) {
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.save();
    ctx.translate(14, effMarginTop + effPlotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(ylabel, 0, 0);
    ctx.restore();
  }

  // Data — clip to plot area
  ctx.save();
  ctx.beginPath();
  ctx.rect(effMarginLeft, effMarginTop, effPlotW, effPlotH);
  ctx.clip();

  const tf: PlotTransform = { toCanvasX, toCanvasY };
  for (const field of CANVAS_RENDER_ORDER) {
    const renderer = canvasRenderer(field);
    const value = traceFieldContent(axes, field);
    if (renderer.layer === "data" && value !== undefined)
      renderer.draw(rc, value, tf);
  }

  ctx.restore();

  // Legend
  if (legend && legend.length > 0) {
    drawLegend(ctx, traces, legend, effMarginLeft + effPlotW, effMarginTop);
  }

  // Colorbar (drawn after clip is restored so it can sit outside the plot area)
  if (colorbar) {
    const cbRange = caxis ?? computeColorbarRange(axes);
    if (cbRange) {
      drawColorbarAtLocation(
        ctx,
        cbLoc,
        effMarginLeft,
        effMarginTop,
        effPlotW,
        effPlotH,
        cbRange[0],
        cbRange[1],
        cmap
      );
    }
  }
}

/** The automatic colorbar range: the extent of the values that the drawn
 *  kinds map through the colormap. */
function computeColorbarRange(axes: AxesState): [number, number] | null {
  let dMin = Infinity;
  let dMax = -Infinity;
  const consume = (v: number) => {
    if (isFinite(v)) {
      if (v < dMin) dMin = v;
      if (v > dMax) dMax = v;
    }
  };
  for (const field of CANVAS_RENDER_ORDER) {
    const renderer = canvasRenderer(field);
    const value = traceFieldContent(axes, field);
    if (
      renderer.layer === "data" &&
      renderer.colorValues &&
      value !== undefined
    )
      renderer.colorValues(value, consume);
  }

  if (!isFinite(dMin) || !isFinite(dMax)) return null;
  if (dMin === dMax) {
    dMin -= 0.5;
    dMax += 0.5;
  }
  return [dMin, dMax];
}
