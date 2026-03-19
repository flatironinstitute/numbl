import type { PlotTrace, ImagescTrace, ContourTrace } from "./types.js";
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
  gridOn?: boolean,
  imagescTrace?: ImagescTrace,
  contourTraces?: ContourTrace[],
  colormap?: string,
  axisMode?: string
) {
  const ctx = canvas.getContext("2d");
  const hasContent =
    traces.length > 0 ||
    imagescTrace !== undefined ||
    (contourTraces && contourTraces.length > 0);
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

  // Include imagesc bounds
  if (imagescTrace) {
    if (imagescTrace.x[0] < xMin) xMin = imagescTrace.x[0];
    if (imagescTrace.x[1] > xMax) xMax = imagescTrace.x[1];
    if (imagescTrace.y[0] < yMin) yMin = imagescTrace.y[0];
    if (imagescTrace.y[1] > yMax) yMax = imagescTrace.y[1];
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

  // Small margin around data (skip if tight)
  if (!isTight) {
    const xPad = (xMax - xMin) * 0.05;
    const yPad = (yMax - yMin) * 0.05;
    xMin -= xPad;
    xMax += xPad;
    yMin -= yPad;
    yMax += yPad;
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

  const toCanvasX = (v: number) =>
    effMarginLeft + ((v - xMin) / (xMax - xMin)) * effPlotW;
  const toCanvasY = (v: number) =>
    effMarginTop + effPlotH - ((v - yMin) / (yMax - yMin)) * effPlotH;

  // Grid (only when gridOn is true or undefined — default on for backward compat)
  if (gridOn !== false) {
    ctx.strokeStyle = "#ddd";
    ctx.lineWidth = 0.5;

    const xTicksGrid = generateTicks(
      xMin,
      xMax,
      Math.max(3, Math.floor(effPlotW / 80))
    );
    const yTicksGrid = generateTicks(
      yMin,
      yMax,
      Math.max(3, Math.floor(effPlotH / 50))
    );

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
  const xTicks = generateTicks(
    xMin,
    xMax,
    Math.max(3, Math.floor(effPlotW / 80))
  );
  const yTicks = generateTicks(
    yMin,
    yMax,
    Math.max(3, Math.floor(effPlotH / 50))
  );

  ctx.fillStyle = "#333";
  ctx.font = "11px monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (const tx of xTicks) {
    ctx.fillText(formatTick(tx), toCanvasX(tx), effMarginTop + effPlotH + 5);
  }
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (const ty of yTicks) {
    ctx.fillText(formatTick(ty), effMarginLeft - 5, toCanvasY(ty));
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

  // Contour rendering
  if (contourTraces) {
    for (const ct of contourTraces) {
      drawContour(ctx, ct, toCanvasX, toCanvasY, colormap);
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
}

// ── Colormap helpers ────────────────────────────────────────────────────

function colormapLookup(t: number, name?: string): [number, number, number] {
  // t is 0..1
  const clamped = Math.max(0, Math.min(1, t));
  if (name === "jet") {
    return jetColormap(clamped);
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
