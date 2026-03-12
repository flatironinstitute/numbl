import { useRef, useEffect, useCallback } from "react";
import type {
  PlotTrace,
  Plot3Trace,
  SurfTrace,
} from "../numbl-core/runtime/plotUtils.js";
import { SurfView } from "./SurfView.js";

interface FigureViewProps {
  traces: PlotTrace[];
  plot3Traces?: Plot3Trace[];
  surfTraces?: SurfTrace[];
  title?: string;
  xlabel?: string;
  ylabel?: string;
}

const TRACE_COLORS = [
  "#0072BD", // blue
  "#D95319", // red-orange
  "#EDB120", // yellow
  "#7E2F8E", // purple
  "#77AC30", // green
  "#4DBEEE", // cyan
  "#A2142F", // dark red
];

export function FigureView({
  traces,
  plot3Traces,
  surfTraces,
  title,
  xlabel,
  ylabel,
}: FigureViewProps) {
  const has3D =
    (surfTraces && surfTraces.length > 0) ||
    (plot3Traces && plot3Traces.length > 0);

  if (has3D) {
    return (
      <SurfView surfTraces={surfTraces ?? []} plot3Traces={plot3Traces ?? []} />
    );
  }

  return (
    <PlotCanvas traces={traces} title={title} xlabel={xlabel} ylabel={ylabel} />
  );
}

function PlotCanvas({
  traces,
  title,
  xlabel,
  ylabel,
}: {
  traces: PlotTrace[];
  title?: string;
  xlabel?: string;
  ylabel?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    drawPlot(canvas, traces, title, xlabel, ylabel);
  }, [traces, title, xlabel, ylabel]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const observer = new ResizeObserver(() => {
      const rect = container.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
      redraw();
    });
    observer.observe(container);

    return () => observer.disconnect();
  }, [redraw]);

  return (
    <div ref={containerRef} style={{ width: "100%", height: "100%" }}>
      <canvas ref={canvasRef} style={{ display: "block" }} />
    </div>
  );
}

function niceTickStep(range: number, maxTicks: number): number {
  const rough = range / maxTicks;
  const pow = Math.pow(10, Math.floor(Math.log10(rough)));
  const norm = rough / pow;
  let step: number;
  if (norm <= 1.5) step = 1;
  else if (norm <= 3.5) step = 2;
  else if (norm <= 7.5) step = 5;
  else step = 10;
  return step * pow;
}

function generateTicks(min: number, max: number, maxTicks: number): number[] {
  const step = niceTickStep(max - min, maxTicks);
  const ticks: number[] = [];
  const start = Math.ceil(min / step) * step;
  for (let v = start; v <= max + step * 0.001; v += step) {
    ticks.push(v);
  }
  return ticks;
}

function formatTick(v: number): string {
  if (Math.abs(v) < 1e-10) return "0";
  if (Math.abs(v) >= 1e4 || (Math.abs(v) < 0.01 && v !== 0))
    return v.toExponential(1);
  return parseFloat(v.toPrecision(6)).toString();
}

// ── Color helpers ───────────────────────────────────────────────────────

function rgbToCSS(rgb: [number, number, number]): string {
  return `rgb(${Math.round(rgb[0] * 255)}, ${Math.round(rgb[1] * 255)}, ${Math.round(rgb[2] * 255)})`;
}

function traceColor(trace: PlotTrace, index: number): string {
  if (trace.color) return rgbToCSS(trace.color);
  return TRACE_COLORS[index % TRACE_COLORS.length];
}

// ── Line dash patterns ──────────────────────────────────────────────────

function getLineDash(style: string | undefined): number[] {
  switch (style) {
    case "--":
      return [8, 4];
    case ":":
      return [2, 4];
    case "-.":
      return [8, 4, 2, 4];
    default:
      return [];
  }
}

// ── Marker rendering ────────────────────────────────────────────────────

function drawMarkers(
  ctx: CanvasRenderingContext2D,
  trace: PlotTrace,
  toCanvasX: (v: number) => number,
  toCanvasY: (v: number) => number,
  defaultColor: string
) {
  if (!trace.marker || trace.marker === "none") return;

  const halfSize = (trace.markerSize ?? 6) / 2;
  const edgeColor = trace.markerEdgeColor
    ? rgbToCSS(trace.markerEdgeColor)
    : defaultColor;
  const faceColor = trace.markerFaceColor
    ? rgbToCSS(trace.markerFaceColor)
    : undefined; // undefined = no fill

  // Determine which indices to draw markers at
  const indices = trace.markerIndices
    ? trace.markerIndices.map(i => i - 1) // 1-based → 0-based
    : Array.from({ length: trace.x.length }, (_, i) => i);

  ctx.lineWidth = 1.5;

  for (const i of indices) {
    if (i < 0 || i >= trace.x.length) continue;
    const vx = trace.x[i];
    const vy = trace.y[i];
    if (!isFinite(vx) || !isFinite(vy)) continue; // skip NaN/Inf
    const cx = toCanvasX(vx);
    const cy = toCanvasY(vy);
    drawSingleMarker(ctx, trace.marker, cx, cy, halfSize, edgeColor, faceColor);
  }
}

function drawSingleMarker(
  ctx: CanvasRenderingContext2D,
  marker: string,
  cx: number,
  cy: number,
  r: number,
  edgeColor: string,
  faceColor: string | undefined
) {
  ctx.strokeStyle = edgeColor;
  ctx.fillStyle = faceColor ?? "transparent";

  switch (marker) {
    case "o": {
      // Circle
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      if (faceColor) ctx.fill();
      ctx.stroke();
      break;
    }
    case "+": {
      // Plus sign
      ctx.beginPath();
      ctx.moveTo(cx - r, cy);
      ctx.lineTo(cx + r, cy);
      ctx.moveTo(cx, cy - r);
      ctx.lineTo(cx, cy + r);
      ctx.stroke();
      break;
    }
    case "*": {
      // Asterisk (6 lines)
      ctx.beginPath();
      for (let a = 0; a < 6; a++) {
        const angle = (a * Math.PI) / 3;
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx + r * Math.cos(angle), cy + r * Math.sin(angle));
      }
      ctx.stroke();
      break;
    }
    case ".": {
      // Small filled dot
      ctx.beginPath();
      ctx.arc(cx, cy, Math.max(r / 3, 1.5), 0, Math.PI * 2);
      ctx.fillStyle = edgeColor;
      ctx.fill();
      break;
    }
    case "x": {
      // X-cross
      const d = r * 0.707; // r * cos(45°)
      ctx.beginPath();
      ctx.moveTo(cx - d, cy - d);
      ctx.lineTo(cx + d, cy + d);
      ctx.moveTo(cx + d, cy - d);
      ctx.lineTo(cx - d, cy + d);
      ctx.stroke();
      break;
    }
    case "_": {
      // Horizontal line
      ctx.beginPath();
      ctx.moveTo(cx - r, cy);
      ctx.lineTo(cx + r, cy);
      ctx.stroke();
      break;
    }
    case "|": {
      // Vertical line
      ctx.beginPath();
      ctx.moveTo(cx, cy - r);
      ctx.lineTo(cx, cy + r);
      ctx.stroke();
      break;
    }
    case "s": {
      // Square
      ctx.beginPath();
      ctx.rect(cx - r, cy - r, r * 2, r * 2);
      if (faceColor) ctx.fill();
      ctx.stroke();
      break;
    }
    case "d": {
      // Diamond
      ctx.beginPath();
      ctx.moveTo(cx, cy - r);
      ctx.lineTo(cx + r, cy);
      ctx.lineTo(cx, cy + r);
      ctx.lineTo(cx - r, cy);
      ctx.closePath();
      if (faceColor) ctx.fill();
      ctx.stroke();
      break;
    }
    case "^": {
      // Upward triangle
      drawTriangle(ctx, cx, cy, r, 0, faceColor);
      break;
    }
    case "v": {
      // Downward triangle
      drawTriangle(ctx, cx, cy, r, Math.PI, faceColor);
      break;
    }
    case "<": {
      // Left triangle
      drawTriangle(ctx, cx, cy, r, -Math.PI / 2, faceColor);
      break;
    }
    case ">": {
      // Right triangle
      drawTriangle(ctx, cx, cy, r, Math.PI / 2, faceColor);
      break;
    }
    case "p": {
      // Pentagram (5-pointed star)
      drawStar(ctx, cx, cy, r, 5, faceColor);
      break;
    }
    case "h": {
      // Hexagram (6-pointed star)
      drawStar(ctx, cx, cy, r, 6, faceColor);
      break;
    }
  }
}

function drawTriangle(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  r: number,
  rotation: number,
  faceColor: string | undefined
) {
  ctx.beginPath();
  for (let i = 0; i < 3; i++) {
    const angle = rotation - Math.PI / 2 + (i * 2 * Math.PI) / 3;
    const px = cx + r * Math.cos(angle);
    const py = cy + r * Math.sin(angle);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.closePath();
  if (faceColor) ctx.fill();
  ctx.stroke();
}

function drawStar(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  r: number,
  points: number,
  faceColor: string | undefined
) {
  const innerR = r * 0.4;
  ctx.beginPath();
  for (let i = 0; i < points * 2; i++) {
    const angle = -Math.PI / 2 + (i * Math.PI) / points;
    const radius = i % 2 === 0 ? r : innerR;
    const px = cx + radius * Math.cos(angle);
    const py = cy + radius * Math.sin(angle);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.closePath();
  if (faceColor) ctx.fill();
  ctx.stroke();
}

// ── Main draw function ──────────────────────────────────────────────────

function drawPlot(
  canvas: HTMLCanvasElement,
  traces: PlotTrace[],
  title?: string,
  xlabel?: string,
  ylabel?: string
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

  // Grid
  ctx.strokeStyle = "#ddd";
  ctx.lineWidth = 0.5;

  const xTicks = generateTicks(xMin, xMax, Math.max(3, Math.floor(plotW / 80)));
  const yTicks = generateTicks(yMin, yMax, Math.max(3, Math.floor(plotH / 50)));

  for (const tx of xTicks) {
    const cx = toCanvasX(tx);
    ctx.beginPath();
    ctx.moveTo(cx, margin.top);
    ctx.lineTo(cx, margin.top + plotH);
    ctx.stroke();
  }
  for (const ty of yTicks) {
    const cy = toCanvasY(ty);
    ctx.beginPath();
    ctx.moveTo(margin.left, cy);
    ctx.lineTo(margin.left + plotW, cy);
    ctx.stroke();
  }

  // Plot border
  ctx.strokeStyle = "#333";
  ctx.lineWidth = 1;
  ctx.strokeRect(margin.left, margin.top, plotW, plotH);

  // Tick labels
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
}
