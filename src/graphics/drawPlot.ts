import type {
  PlotTrace,
  ImagescTrace,
  PcolorTrace,
  ContourTrace,
  BarTrace,
  ErrorBarTrace,
  BoxTrace,
  PieTrace,
  HeatmapTrace,
} from "./types.js";
import {
  traceColor,
  getLineDash,
  generateTicks,
  formatTick,
} from "./plotHelpers.js";
import { drawMarkers } from "./plotMarkers.js";
import { drawLegend } from "./plotLegend.js";

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

export function drawPlot(
  canvas: HTMLCanvasElement,
  traces: PlotTrace[],
  title?: string,
  xlabel?: string,
  ylabel?: string,
  legend?: string[],
  gridOn?: boolean,
  imagescTrace?: ImagescTrace,
  contourTraces?: ContourTrace[],
  colormap?: string,
  axisMode?: string,
  axisScale?: "linear" | "semilogx" | "semilogy" | "loglog",
  barTraces?: BarTrace[],
  barhTraces?: BarTrace[],
  errorBarTraces?: ErrorBarTrace[],
  boxTraces?: BoxTrace[],
  pieTrace?: PieTrace,
  heatmapTrace?: HeatmapTrace,
  areaTraces?: PlotTrace[],
  areaBaseValue?: number,
  pcolorTraces?: PcolorTrace[],
  shading?: "faceted" | "flat" | "interp",
  colorbar?: boolean,
  colorbarLocation?: string,
  caxis?: [number, number],
  colormapData?: number[][]
) {
  _activeColormapData = colormapData;
  const ctx = canvas.getContext("2d");
  const hasContent =
    traces.length > 0 ||
    imagescTrace !== undefined ||
    (pcolorTraces && pcolorTraces.length > 0) ||
    (contourTraces && contourTraces.length > 0) ||
    (barTraces && barTraces.length > 0) ||
    (barhTraces && barhTraces.length > 0) ||
    (errorBarTraces && errorBarTraces.length > 0) ||
    (boxTraces && boxTraces.length > 0) ||
    pieTrace !== undefined ||
    heatmapTrace !== undefined ||
    (areaTraces && areaTraces.length > 0);
  if (!ctx || !hasContent) return;

  const dpr = window.devicePixelRatio || 1;
  const w = canvas.width;
  const h = canvas.height;

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const cw = w / dpr;
  const ch = h / dpr;

  // Clear with white background
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, cw, ch);

  // Pie / donut chart — completely separate rendering path (no axes)
  if (pieTrace) {
    drawPieChart(ctx, cw, ch, pieTrace, title);
    return;
  }

  // Heatmap — completely separate rendering path (categorical axes)
  if (heatmapTrace) {
    drawHeatmap(ctx, cw, ch, heatmapTrace, title, colormap);
    return;
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

  // Include imagesc bounds
  if (imagescTrace) {
    if (imagescTrace.x[0] < xMin) xMin = imagescTrace.x[0];
    if (imagescTrace.x[1] > xMax) xMax = imagescTrace.x[1];
    if (imagescTrace.y[0] < yMin) yMin = imagescTrace.y[0];
    if (imagescTrace.y[1] > yMax) yMax = imagescTrace.y[1];
  }

  // Include pcolor bounds
  if (pcolorTraces) {
    for (const pt of pcolorTraces) {
      for (const v of pt.x) {
        if (isFinite(v)) {
          if (v < xMin) xMin = v;
          if (v > xMax) xMax = v;
        }
      }
      for (const v of pt.y) {
        if (isFinite(v)) {
          if (v < yMin) yMin = v;
          if (v > yMax) yMax = v;
        }
      }
    }
  }

  // Include contour bounds
  if (contourTraces) {
    for (const ct of contourTraces) {
      for (const v of ct.x) {
        if (isFinite(v)) {
          if (v < xMin) xMin = v;
          if (v > xMax) xMax = v;
        }
      }
      for (const v of ct.y) {
        if (isFinite(v)) {
          if (v < yMin) yMin = v;
          if (v > yMax) yMax = v;
        }
      }
    }
  }

  // Include bar bounds
  if (barTraces) {
    for (const bt of barTraces) {
      const halfW = bt.width / 2;
      for (let i = 0; i < bt.x.length; i++) {
        const bx = bt.x[i];
        if (isFinite(bx)) {
          if (bx - halfW < xMin) xMin = bx - halfW;
          if (bx + halfW > xMax) xMax = bx + halfW;
        }
        const by = bt.y[i];
        if (isFinite(by)) {
          if (by < yMin) yMin = by;
          if (by > yMax) yMax = by;
        }
      }
      // Bars extend to zero
      if (0 < yMin) yMin = 0;
      if (0 > yMax) yMax = 0;
    }
  }

  // Include barh bounds (horizontal bars: x=positions on y-axis, y=bar lengths on x-axis)
  if (barhTraces) {
    for (const bt of barhTraces) {
      const halfH = bt.width / 2;
      for (let i = 0; i < bt.x.length; i++) {
        const pos = bt.x[i]; // position on y-axis
        if (isFinite(pos)) {
          if (pos - halfH < yMin) yMin = pos - halfH;
          if (pos + halfH > yMax) yMax = pos + halfH;
        }
        const len = bt.y[i]; // bar length on x-axis
        if (isFinite(len)) {
          if (len < xMin) xMin = len;
          if (len > xMax) xMax = len;
        }
      }
      // Horizontal bars extend to zero on the x-axis
      if (0 < xMin) xMin = 0;
      if (0 > xMax) xMax = 0;
    }
  }

  // Include errorbar bounds
  if (errorBarTraces) {
    for (const et of errorBarTraces) {
      for (let i = 0; i < et.x.length; i++) {
        const ex = et.x[i];
        const ey = et.y[i];
        if (isFinite(ex)) {
          const xl = et.xNeg ? ex - et.xNeg[i] : ex;
          const xr = et.xPos ? ex + et.xPos[i] : ex;
          if (xl < xMin) xMin = xl;
          if (xr > xMax) xMax = xr;
        }
        if (isFinite(ey)) {
          const ylo = ey - et.yNeg[i];
          const yhi = ey + et.yPos[i];
          if (ylo < yMin) yMin = ylo;
          if (yhi > yMax) yMax = yhi;
        }
      }
    }
  }

  // Include boxchart bounds
  if (boxTraces) {
    for (const bt of boxTraces) {
      const halfW = bt.width / 2;
      if (bt.x - halfW < xMin) xMin = bt.x - halfW;
      if (bt.x + halfW > xMax) xMax = bt.x + halfW;
      if (bt.whiskerLow < yMin) yMin = bt.whiskerLow;
      if (bt.whiskerHigh > yMax) yMax = bt.whiskerHigh;
      for (const o of bt.outliers) {
        if (o < yMin) yMin = o;
        if (o > yMax) yMax = o;
      }
    }
  }

  // Include area bounds (stacked: sum y values for upper bound)
  if (areaTraces && areaTraces.length > 0) {
    const base = areaBaseValue ?? 0;
    if (base < yMin) yMin = base;
    if (base > yMax) yMax = base;
    // For stacked areas, the total height is the sum of all traces
    const n = areaTraces[0].x.length;
    for (let i = 0; i < n; i++) {
      let cumY = base;
      for (const t of areaTraces) {
        if (i < t.x.length) {
          const v = t.x[i];
          if (isFinite(v)) {
            if (v < xMin) xMin = v;
            if (v > xMax) xMax = v;
          }
          cumY += t.y[i] - base;
          if (isFinite(cumY)) {
            if (cumY < yMin) yMin = cumY;
            if (cumY > yMax) yMax = cumY;
          }
        }
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

  // Parse axis mode flags
  const isTight = axisMode?.includes("tight") ?? false;
  const isEqual = axisMode?.includes("equal") ?? false;

  const logX = axisScale === "semilogx" || axisScale === "loglog";
  const logY = axisScale === "semilogy" || axisScale === "loglog";

  // For log axes, clamp mins to positive values
  if (logX && xMin <= 0) xMin = xMax > 0 ? xMax * 1e-6 : 1;
  if (logY && yMin <= 0) yMin = yMax > 0 ? yMax * 1e-6 : 1;

  // Small margin around data (skip if tight)
  if (!isTight) {
    if (logX) {
      // Pad in log space
      const logPad = (Math.log10(xMax) - Math.log10(xMin)) * 0.05;
      xMin = Math.pow(10, Math.log10(xMin) - logPad);
      xMax = Math.pow(10, Math.log10(xMax) + logPad);
    } else {
      const xPad = (xMax - xMin) * 0.05;
      xMin -= xPad;
      xMax += xPad;
    }
    if (logY) {
      const logPad = (Math.log10(yMax) - Math.log10(yMin)) * 0.05;
      yMin = Math.pow(10, Math.log10(yMin) - logPad);
      yMax = Math.pow(10, Math.log10(yMax) + logPad);
    } else {
      const yPad = (yMax - yMin) * 0.05;
      yMin -= yPad;
      yMax += yPad;
    }
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
  }

  const toCanvasX = logX
    ? (v: number) => {
        const logV = Math.log10(Math.max(v, Number.MIN_VALUE));
        const logMin = Math.log10(Math.max(xMin, Number.MIN_VALUE));
        const logMax = Math.log10(Math.max(xMax, Number.MIN_VALUE));
        return effMarginLeft + ((logV - logMin) / (logMax - logMin)) * effPlotW;
      }
    : (v: number) => effMarginLeft + ((v - xMin) / (xMax - xMin)) * effPlotW;
  const toCanvasY = logY
    ? (v: number) => {
        const logV = Math.log10(Math.max(v, Number.MIN_VALUE));
        const logMin = Math.log10(Math.max(yMin, Number.MIN_VALUE));
        const logMax = Math.log10(Math.max(yMax, Number.MIN_VALUE));
        return (
          effMarginTop +
          effPlotH -
          ((logV - logMin) / (logMax - logMin)) * effPlotH
        );
      }
    : (v: number) =>
        effMarginTop + effPlotH - ((v - yMin) / (yMax - yMin)) * effPlotH;

  // Grid (only when gridOn is true or undefined — default on for backward compat)
  if (gridOn !== false) {
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

  // Plot border
  ctx.strokeStyle = "#333";
  ctx.lineWidth = 1;
  ctx.strokeRect(effMarginLeft, effMarginTop, effPlotW, effPlotH);

  // Tick labels
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

  // Labels
  ctx.fillStyle = "#222";
  if (title) {
    ctx.font = "bold 13px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(title, effMarginLeft + effPlotW / 2, effMarginTop / 2);
  }
  if (xlabel) {
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText(xlabel, effMarginLeft + effPlotW / 2, ch - 4);
  }
  if (ylabel) {
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

  // Imagesc rendering
  if (imagescTrace) {
    drawImagesc(ctx, imagescTrace, toCanvasX, toCanvasY, colormap);
  }

  // Pcolor rendering
  if (pcolorTraces) {
    for (const pt of pcolorTraces) {
      drawPcolor(ctx, pt, toCanvasX, toCanvasY, colormap, shading, caxis);
    }
  }

  // Contour rendering
  if (contourTraces) {
    for (const ct of contourTraces) {
      drawContour(ctx, ct, toCanvasX, toCanvasY, colormap);
    }
  }

  // Bar rendering
  if (barTraces) {
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
        ctx.fillRect(
          cx - halfW,
          Math.min(cy, y0),
          halfW * 2,
          Math.abs(cy - y0)
        );
        ctx.strokeRect(
          cx - halfW,
          Math.min(cy, y0),
          halfW * 2,
          Math.abs(cy - y0)
        );
      }
    }
  }

  // Barh rendering (horizontal bars)
  if (barhTraces) {
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
        ctx.fillRect(
          Math.min(x0, cx),
          cy - halfH,
          Math.abs(cx - x0),
          halfH * 2
        );
        ctx.strokeRect(
          Math.min(x0, cx),
          cy - halfH,
          Math.abs(cx - x0),
          halfH * 2
        );
      }
    }
  }

  // Box chart rendering
  if (boxTraces) {
    const defaultColors: [number, number, number][] = [
      [0.0, 0.447, 0.741],
      [0.85, 0.325, 0.098],
      [0.929, 0.694, 0.125],
      [0.494, 0.184, 0.556],
      [0.466, 0.674, 0.188],
    ];
    for (let bi = 0; bi < boxTraces.length; bi++) {
      const bt = boxTraces[bi];
      const [r, g, b] = bt.color ?? defaultColors[bi % defaultColors.length];
      const fillColor = `rgba(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)},0.3)`;
      const strokeColor = `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`;

      const cx = toCanvasX(bt.x);
      const halfW = Math.abs(toCanvasX(bt.x + bt.width) - toCanvasX(bt.x)) / 2;

      const yQ1 = toCanvasY(bt.q1);
      const yQ3 = toCanvasY(bt.q3);
      const yMed = toCanvasY(bt.median);
      const yWLo = toCanvasY(bt.whiskerLow);
      const yWHi = toCanvasY(bt.whiskerHigh);

      // Box (Q1 to Q3)
      ctx.fillStyle = fillColor;
      ctx.fillRect(
        cx - halfW,
        Math.min(yQ1, yQ3),
        halfW * 2,
        Math.abs(yQ3 - yQ1)
      );
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(
        cx - halfW,
        Math.min(yQ1, yQ3),
        halfW * 2,
        Math.abs(yQ3 - yQ1)
      );

      // Median line
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(cx - halfW, yMed);
      ctx.lineTo(cx + halfW, yMed);
      ctx.stroke();

      // Whiskers (vertical lines from box to whisker ends)
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 2]);
      ctx.beginPath();
      ctx.moveTo(cx, yQ3);
      ctx.lineTo(cx, yWHi);
      ctx.moveTo(cx, yQ1);
      ctx.lineTo(cx, yWLo);
      ctx.stroke();
      ctx.setLineDash([]);

      // Whisker caps (horizontal lines at whisker ends)
      const capW = halfW * 0.5;
      ctx.beginPath();
      ctx.moveTo(cx - capW, yWHi);
      ctx.lineTo(cx + capW, yWHi);
      ctx.moveTo(cx - capW, yWLo);
      ctx.lineTo(cx + capW, yWLo);
      ctx.stroke();

      // Outliers
      ctx.fillStyle = strokeColor;
      for (const o of bt.outliers) {
        const oy = toCanvasY(o);
        ctx.beginPath();
        ctx.arc(cx, oy, 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  // Area rendering (stacked filled areas)
  if (areaTraces && areaTraces.length > 0) {
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

  // Errorbar rendering
  if (errorBarTraces) {
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
    drawLegend(ctx, traces, legend, effMarginLeft + effPlotW, effMarginTop);
  }

  // Colorbar (drawn after clip is restored so it can sit outside the plot area)
  if (colorbar) {
    const cbRange =
      caxis ?? computeColorbarRange(pcolorTraces, imagescTrace, contourTraces);
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
        colormap
      );
    }
  }
}

// ── Colormap helpers ────────────────────────────────────────────────────

/** Module-scoped custom colormap data, set at the start of drawPlot. */
let _activeColormapData: number[][] | undefined;

function colormapLookup(t: number, name?: string): [number, number, number] {
  // t is 0..1
  const clamped = Math.max(0, Math.min(1, t));
  if (_activeColormapData && _activeColormapData.length > 0) {
    const data = _activeColormapData;
    const n = data.length;
    if (n === 1) return data[0] as [number, number, number];
    const idx = clamped * (n - 1);
    const lo = Math.floor(idx);
    const hi = Math.min(lo + 1, n - 1);
    const frac = idx - lo;
    return [
      data[lo][0] + frac * (data[hi][0] - data[lo][0]),
      data[lo][1] + frac * (data[hi][1] - data[lo][1]),
      data[lo][2] + frac * (data[hi][2] - data[lo][2]),
    ];
  }
  if (name === "jet") {
    return jetColormap(clamped);
  }
  if (name === "redblue") {
    return redblueColormap(clamped);
  }
  // Default: parula-like (blue → yellow)
  return parulaColormap(clamped);
}

function parulaColormap(t: number): [number, number, number] {
  // Simplified parula: dark blue → cyan → yellow
  if (t < 0.5) {
    const s = t * 2;
    return [0.2 * (1 - s), 0.1 + 0.6 * s, 0.9 - 0.3 * s];
  }
  const s = (t - 0.5) * 2;
  return [0.2 + 0.8 * s, 0.7 + 0.3 * s, 0.6 - 0.5 * s];
}

function jetColormap(t: number): [number, number, number] {
  let r: number, g: number, b: number;
  if (t < 0.125) {
    r = 0;
    g = 0;
    b = 0.5 + t * 4;
  } else if (t < 0.375) {
    r = 0;
    g = (t - 0.125) * 4;
    b = 1;
  } else if (t < 0.625) {
    r = (t - 0.375) * 4;
    g = 1;
    b = 1 - (t - 0.375) * 4;
  } else if (t < 0.875) {
    r = 1;
    g = 1 - (t - 0.625) * 4;
    b = 0;
  } else {
    r = 1 - (t - 0.875) * 4;
    g = 0;
    b = 0;
  }
  return [
    Math.max(0, Math.min(1, r)),
    Math.max(0, Math.min(1, g)),
    Math.max(0, Math.min(1, b)),
  ];
}

function redblueColormap(t: number): [number, number, number] {
  // Blue (0) → White (0.5) → Red (1)
  if (t < 0.5) {
    const s = t * 2; // 0..1
    return [s, s, 1];
  }
  const s = (t - 0.5) * 2; // 0..1
  return [1, 1 - s, 1 - s];
}

// ── Imagesc rendering ────────────────────────────────────────────────────

function drawImagesc(
  ctx: CanvasRenderingContext2D,
  trace: ImagescTrace,
  toCanvasX: (v: number) => number,
  toCanvasY: (v: number) => number,
  colormap?: string
) {
  const { rows, cols, z, x, y } = trace;
  if (rows === 0 || cols === 0) return;

  // Find z range
  let zMin = Infinity,
    zMax = -Infinity;
  for (const v of z) {
    if (isFinite(v)) {
      if (v < zMin) zMin = v;
      if (v > zMax) zMax = v;
    }
  }
  if (!isFinite(zMin)) return;
  const zRange = zMax - zMin || 1;

  // Cell size in data coordinates
  const dx = (x[1] - x[0]) / cols;
  const dy = (y[1] - y[0]) / rows;

  for (let j = 0; j < cols; j++) {
    for (let i = 0; i < rows; i++) {
      const val = z[j * rows + i]; // column-major
      const t = (val - zMin) / zRange;
      const [r, g, b] = colormapLookup(t, colormap);

      const cx1 = toCanvasX(x[0] + j * dx);
      const cy1 = toCanvasY(y[0] + (i + 1) * dy);
      const cx2 = toCanvasX(x[0] + (j + 1) * dx);
      const cy2 = toCanvasY(y[0] + i * dy);

      ctx.fillStyle = `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`;
      ctx.fillRect(cx1, cy1, cx2 - cx1, cy2 - cy1);
    }
  }
}

// ── Contour rendering ────────────────────────────────────────────────────

function drawContour(
  ctx: CanvasRenderingContext2D,
  trace: ContourTrace,
  toCanvasX: (v: number) => number,
  toCanvasY: (v: number) => number,
  colormap?: string
) {
  const { rows, cols, z, x, y, nLevels, filled } = trace;
  if (rows < 2 || cols < 2) return;

  // Find z range
  let zMin = Infinity,
    zMax = -Infinity;
  for (const v of z) {
    if (isFinite(v)) {
      if (v < zMin) zMin = v;
      if (v > zMax) zMax = v;
    }
  }
  if (!isFinite(zMin)) return;
  const zRange = zMax - zMin || 1;

  // Generate contour levels
  const levels: number[] = [];
  for (let i = 0; i <= nLevels; i++) {
    levels.push(zMin + (i / nLevels) * zRange);
  }

  // Helper to get z value at grid position (row i, col j)
  const getZ = (i: number, j: number) => z[j * rows + i]; // column-major
  const getX = (i: number, j: number) => x[j * rows + i];
  const getY = (i: number, j: number) => y[j * rows + i];

  if (filled) {
    // Filled contour: draw colored rectangles approximating the field
    for (let j = 0; j < cols - 1; j++) {
      for (let i = 0; i < rows - 1; i++) {
        const zAvg =
          (getZ(i, j) + getZ(i + 1, j) + getZ(i, j + 1) + getZ(i + 1, j + 1)) /
          4;
        const t = (zAvg - zMin) / zRange;
        const [r, g, b] = colormapLookup(t, colormap);

        const cx1 = toCanvasX(getX(i, j));
        const cy1 = toCanvasY(getY(i, j));
        const cx2 = toCanvasX(getX(i + 1, j + 1));
        const cy2 = toCanvasY(getY(i + 1, j + 1));

        ctx.fillStyle = `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`;
        ctx.fillRect(
          Math.min(cx1, cx2),
          Math.min(cy1, cy2),
          Math.abs(cx2 - cx1) || 1,
          Math.abs(cy2 - cy1) || 1
        );
      }
    }
  } else {
    // Line contour: use marching squares
    for (let li = 1; li < levels.length - 1; li++) {
      const level = levels[li];
      const t = (level - zMin) / zRange;
      const [r, g, b] = colormapLookup(t, colormap);
      ctx.strokeStyle = `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`;
      ctx.lineWidth = 1;

      for (let j = 0; j < cols - 1; j++) {
        for (let i = 0; i < rows - 1; i++) {
          // Marching squares for this cell
          const z00 = getZ(i, j);
          const z10 = getZ(i + 1, j);
          const z01 = getZ(i, j + 1);
          const z11 = getZ(i + 1, j + 1);

          const x00 = getX(i, j),
            y00 = getY(i, j);
          const x10 = getX(i + 1, j),
            y10 = getY(i + 1, j);
          const x01 = getX(i, j + 1),
            y01 = getY(i, j + 1);
          const x11 = getX(i + 1, j + 1),
            y11 = getY(i + 1, j + 1);

          const segments = marchingSquaresCell(
            z00,
            z10,
            z01,
            z11,
            x00,
            y00,
            x10,
            y10,
            x01,
            y01,
            x11,
            y11,
            level
          );

          for (const seg of segments) {
            ctx.beginPath();
            ctx.moveTo(toCanvasX(seg[0]), toCanvasY(seg[1]));
            ctx.lineTo(toCanvasX(seg[2]), toCanvasY(seg[3]));
            ctx.stroke();
          }
        }
      }
    }
  }
}

/** Returns line segments [x1,y1,x2,y2][] for a single marching squares cell */
function marchingSquaresCell(
  z00: number,
  z10: number,
  z01: number,
  z11: number,
  x00: number,
  y00: number,
  x10: number,
  y10: number,
  x01: number,
  y01: number,
  x11: number,
  y11: number,
  level: number
): [number, number, number, number][] {
  const b0 = z00 >= level ? 1 : 0;
  const b1 = z10 >= level ? 1 : 0;
  const b2 = z01 >= level ? 1 : 0;
  const b3 = z11 >= level ? 1 : 0;
  const code = b0 | (b1 << 1) | (b2 << 2) | (b3 << 3);

  if (code === 0 || code === 15) return [];

  // Interpolate along edges
  const lerp = (a: number, b: number, za: number, zb: number) => {
    const t = (level - za) / (zb - za || 1);
    return a + t * (b - a);
  };

  // Edge midpoints: bottom (0-1), right (1-3), top (2-3), left (0-2)
  const bx = lerp(x00, x10, z00, z10),
    by = lerp(y00, y10, z00, z10); // bottom
  const rx = lerp(x10, x11, z10, z11),
    ry = lerp(y10, y11, z10, z11); // right
  const tx = lerp(x01, x11, z01, z11),
    ty = lerp(y01, y11, z01, z11); // top
  const lx = lerp(x00, x01, z00, z01),
    ly = lerp(y00, y01, z00, z01); // left

  const segs: [number, number, number, number][] = [];
  // Lookup table for marching squares
  switch (code) {
    case 1:
    case 14:
      segs.push([bx, by, lx, ly]);
      break;
    case 2:
    case 13:
      segs.push([bx, by, rx, ry]);
      break;
    case 3:
    case 12:
      segs.push([lx, ly, rx, ry]);
      break;
    case 4:
    case 11:
      segs.push([lx, ly, tx, ty]);
      break;
    case 5:
    case 10:
      segs.push([bx, by, tx, ty]);
      break;
    case 6:
    case 9:
      segs.push([bx, by, lx, ly]);
      segs.push([tx, ty, rx, ry]);
      break;
    case 7:
    case 8:
      segs.push([tx, ty, rx, ry]);
      break;
  }
  return segs;
}

// ── Pie / Donut chart rendering ─────────────────────────────────────────

const PIE_COLORS: [number, number, number][] = [
  [0.0, 0.447, 0.741],
  [0.85, 0.325, 0.098],
  [0.929, 0.694, 0.125],
  [0.494, 0.184, 0.556],
  [0.466, 0.674, 0.188],
  [0.301, 0.745, 0.933],
  [0.635, 0.078, 0.184],
];

function drawPieChart(
  ctx: CanvasRenderingContext2D,
  cw: number,
  ch: number,
  pie: PieTrace,
  title?: string
) {
  const total = pie.values.reduce((a, b) => a + b, 0);
  if (total <= 0) return;

  const titleH = title ? 36 : 0;
  const labelMargin = 60;
  const cx = cw / 2;
  const cy = titleH + (ch - titleH) / 2;
  const outerR = Math.min(
    cw / 2 - labelMargin,
    (ch - titleH) / 2 - labelMargin
  );
  if (outerR <= 0) return;
  const innerR = outerR * pie.innerRadius;

  // Title
  if (title) {
    ctx.font = "bold 14px sans-serif";
    ctx.fillStyle = "#000";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(title, cx, 10);
  }

  let startAngle = -Math.PI / 2; // start at top

  for (let i = 0; i < pie.values.length; i++) {
    const sliceAngle = (pie.values[i] / total) * 2 * Math.PI;
    const endAngle = startAngle + sliceAngle;

    const [r, g, b] = pie.colors?.[i] ?? PIE_COLORS[i % PIE_COLORS.length];

    // Draw slice
    ctx.beginPath();
    if (innerR > 0) {
      ctx.arc(cx, cy, outerR, startAngle, endAngle);
      ctx.arc(cx, cy, innerR, endAngle, startAngle, true);
    } else {
      ctx.moveTo(cx, cy);
      ctx.arc(cx, cy, outerR, startAngle, endAngle);
      ctx.closePath();
    }
    ctx.fillStyle = `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`;
    ctx.fill();

    // Slice border
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Label
    const midAngle = startAngle + sliceAngle / 2;
    const pct = ((pie.values[i] / total) * 100).toFixed(1) + "%";
    const labelR = outerR + 16;
    const lx = cx + labelR * Math.cos(midAngle);
    const ly = cy + labelR * Math.sin(midAngle);

    ctx.font = "12px sans-serif";
    ctx.fillStyle = "#333";
    ctx.textAlign =
      midAngle > Math.PI / 2 || midAngle < -Math.PI / 2 ? "right" : "left";
    ctx.textBaseline = "middle";

    const label = pie.names?.[i] ? `${pie.names[i]} (${pct})` : pct;
    ctx.fillText(label, lx, ly);

    startAngle = endAngle;
  }
}

// ── Heatmap rendering ───────────────────────────────────────────────────

function drawHeatmap(
  ctx: CanvasRenderingContext2D,
  cw: number,
  ch: number,
  hm: HeatmapTrace,
  title?: string,
  colormap?: string
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

function formatCbVal(v: number): string {
  if (Number.isInteger(v)) return String(v);
  return v.toPrecision(3);
}

// ── Pcolor rendering ─────────────────────────────────────────────────────

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
  colormap: string | undefined,
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
  colormap: string | undefined,
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

// ── Colorbar rendering ──────────────────────────────────────────────────

/**
 * Determine the data range for the colorbar by inspecting whichever
 * color-mapped trace is present in the 2D pipeline.
 */
function computeColorbarRange(
  pcolorTraces: PcolorTrace[] | undefined,
  imagescTrace: ImagescTrace | undefined,
  contourTraces: ContourTrace[] | undefined
): [number, number] | null {
  let dMin = Infinity;
  let dMax = -Infinity;

  const consume = (arr: number[] | Float64Array) => {
    for (const v of arr) {
      if (isFinite(v)) {
        if (v < dMin) dMin = v;
        if (v > dMax) dMax = v;
      }
    }
  };

  if (pcolorTraces) {
    for (const pt of pcolorTraces) consume(pt.c);
  }
  if (imagescTrace) {
    consume(imagescTrace.z);
  }
  if (contourTraces) {
    for (const ct of contourTraces) consume(ct.z);
  }

  if (!isFinite(dMin) || !isFinite(dMax)) return null;
  if (dMin === dMax) {
    dMin -= 0.5;
    dMax += 0.5;
  }
  return [dMin, dMax];
}

/**
 * Draw a colorbar at one of the 8 MATLAB locations relative to the plot
 * area at (plotL, plotT, plotW, plotH).
 */
function drawColorbarAtLocation(
  ctx: CanvasRenderingContext2D,
  location: string,
  plotL: number,
  plotT: number,
  plotW: number,
  plotH: number,
  dMin: number,
  dMax: number,
  colormap: string | undefined
) {
  const barThickness = 16;
  const inset = 8; // gap between plot edge and colorbar for inside variants
  const gap = 10; // gap between plot edge and colorbar for outside variants

  switch (location) {
    case "eastoutside": {
      const x = plotL + plotW + gap;
      drawColorbarVertical(
        ctx,
        x,
        plotT,
        barThickness,
        plotH,
        dMin,
        dMax,
        colormap,
        "right"
      );
      break;
    }
    case "westoutside": {
      const x = plotL - gap - barThickness;
      drawColorbarVertical(
        ctx,
        x,
        plotT,
        barThickness,
        plotH,
        dMin,
        dMax,
        colormap,
        "left"
      );
      break;
    }
    case "northoutside": {
      const y = plotT - gap - barThickness;
      drawColorbarHorizontal(
        ctx,
        plotL,
        y,
        plotW,
        barThickness,
        dMin,
        dMax,
        colormap,
        "top"
      );
      break;
    }
    case "southoutside": {
      const y = plotT + plotH + gap;
      drawColorbarHorizontal(
        ctx,
        plotL,
        y,
        plotW,
        barThickness,
        dMin,
        dMax,
        colormap,
        "bottom"
      );
      break;
    }
    case "east": {
      const x = plotL + plotW - inset - barThickness;
      drawColorbarVertical(
        ctx,
        x,
        plotT + inset,
        barThickness,
        plotH - 2 * inset,
        dMin,
        dMax,
        colormap,
        "left"
      );
      break;
    }
    case "west": {
      const x = plotL + inset;
      drawColorbarVertical(
        ctx,
        x,
        plotT + inset,
        barThickness,
        plotH - 2 * inset,
        dMin,
        dMax,
        colormap,
        "right"
      );
      break;
    }
    case "north": {
      const y = plotT + inset;
      drawColorbarHorizontal(
        ctx,
        plotL + inset,
        y,
        plotW - 2 * inset,
        barThickness,
        dMin,
        dMax,
        colormap,
        "bottom"
      );
      break;
    }
    case "south": {
      const y = plotT + plotH - inset - barThickness;
      drawColorbarHorizontal(
        ctx,
        plotL + inset,
        y,
        plotW - 2 * inset,
        barThickness,
        dMin,
        dMax,
        colormap,
        "top"
      );
      break;
    }
  }
}

function drawColorbarVertical(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  dMin: number,
  dMax: number,
  colormap: string | undefined,
  labelSide: "left" | "right"
) {
  const nSteps = Math.max(1, Math.round(h));
  for (let s = 0; s < nSteps; s++) {
    const t = 1 - s / nSteps; // top = max, bottom = min
    const [r, g, b] = colormapLookup(t, colormap);
    ctx.fillStyle = `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`;
    ctx.fillRect(x, y + (s / nSteps) * h, w, h / nSteps + 1);
  }
  ctx.strokeStyle = "#999";
  ctx.lineWidth = 1;
  ctx.strokeRect(x, y, w, h);

  // Labels
  ctx.font = "10px sans-serif";
  ctx.fillStyle = "#333";
  if (labelSide === "right") {
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(formatCbVal(dMax), x + w + 4, y);
    ctx.textBaseline = "bottom";
    ctx.fillText(formatCbVal(dMin), x + w + 4, y + h);
  } else {
    ctx.textAlign = "right";
    ctx.textBaseline = "top";
    ctx.fillText(formatCbVal(dMax), x - 4, y);
    ctx.textBaseline = "bottom";
    ctx.fillText(formatCbVal(dMin), x - 4, y + h);
  }
}

function drawColorbarHorizontal(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  dMin: number,
  dMax: number,
  colormap: string | undefined,
  labelSide: "top" | "bottom"
) {
  const nSteps = Math.max(1, Math.round(w));
  for (let s = 0; s < nSteps; s++) {
    const t = s / nSteps; // left = min, right = max
    const [r, g, b] = colormapLookup(t, colormap);
    ctx.fillStyle = `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`;
    ctx.fillRect(x + (s / nSteps) * w, y, w / nSteps + 1, h);
  }
  ctx.strokeStyle = "#999";
  ctx.lineWidth = 1;
  ctx.strokeRect(x, y, w, h);

  // Labels
  ctx.font = "10px sans-serif";
  ctx.fillStyle = "#333";
  if (labelSide === "bottom") {
    ctx.textBaseline = "top";
    ctx.textAlign = "left";
    ctx.fillText(formatCbVal(dMin), x, y + h + 4);
    ctx.textAlign = "right";
    ctx.fillText(formatCbVal(dMax), x + w, y + h + 4);
  } else {
    ctx.textBaseline = "bottom";
    ctx.textAlign = "left";
    ctx.fillText(formatCbVal(dMin), x, y - 4);
    ctx.textAlign = "right";
    ctx.fillText(formatCbVal(dMax), x + w, y - 4);
  }
}
