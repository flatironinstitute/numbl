import type { PlotTrace } from "./types.js";
import { rgbToCSS } from "./plotHelpers.js";

export function drawMarkers(
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
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      if (faceColor) ctx.fill();
      ctx.stroke();
      break;
    }
    case "+": {
      ctx.beginPath();
      ctx.moveTo(cx - r, cy);
      ctx.lineTo(cx + r, cy);
      ctx.moveTo(cx, cy - r);
      ctx.lineTo(cx, cy + r);
      ctx.stroke();
      break;
    }
    case "*": {
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
      ctx.beginPath();
      ctx.arc(cx, cy, Math.max(r / 3, 1.5), 0, Math.PI * 2);
      ctx.fillStyle = edgeColor;
      ctx.fill();
      break;
    }
    case "x": {
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
      ctx.beginPath();
      ctx.moveTo(cx - r, cy);
      ctx.lineTo(cx + r, cy);
      ctx.stroke();
      break;
    }
    case "|": {
      ctx.beginPath();
      ctx.moveTo(cx, cy - r);
      ctx.lineTo(cx, cy + r);
      ctx.stroke();
      break;
    }
    case "s": {
      ctx.beginPath();
      ctx.rect(cx - r, cy - r, r * 2, r * 2);
      if (faceColor) ctx.fill();
      ctx.stroke();
      break;
    }
    case "d": {
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
      drawTriangle(ctx, cx, cy, r, 0, faceColor);
      break;
    }
    case "v": {
      drawTriangle(ctx, cx, cy, r, Math.PI, faceColor);
      break;
    }
    case "<": {
      drawTriangle(ctx, cx, cy, r, -Math.PI / 2, faceColor);
      break;
    }
    case ">": {
      drawTriangle(ctx, cx, cy, r, Math.PI / 2, faceColor);
      break;
    }
    case "p": {
      drawStar(ctx, cx, cy, r, 5, faceColor);
      break;
    }
    case "h": {
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
