import type { PlotTrace } from "./types.js";
import { traceColor, getLineDash } from "./plotHelpers.js";

export function drawLegend(
  ctx: CanvasRenderingContext2D,
  traces: PlotTrace[],
  labels: string[],
  plotRight: number,
  plotTop: number
) {
  const fontSize = 11;
  const lineH = fontSize + 6;
  const sampleW = 20;
  const pad = 8;
  const gap = 6;

  ctx.font = `${fontSize}px sans-serif`;
  let maxLabelW = 0;
  for (const l of labels) {
    const w = ctx.measureText(l).width;
    if (w > maxLabelW) maxLabelW = w;
  }

  const boxW = pad + sampleW + gap + maxLabelW + pad;
  const boxH = pad + labels.length * lineH + pad;

  const bx = plotRight - boxW - 8;
  const by = plotTop + 8;

  // Background
  ctx.fillStyle = "rgba(255,255,255,0.9)";
  ctx.fillRect(bx, by, boxW, boxH);
  ctx.strokeStyle = "#999";
  ctx.lineWidth = 0.5;
  ctx.strokeRect(bx, by, boxW, boxH);

  for (let i = 0; i < labels.length; i++) {
    const color = i < traces.length ? traceColor(traces[i], i) : "#333";
    const lw = i < traces.length ? (traces[i].lineWidth ?? 2) : 2;
    const lineStyle = i < traces.length ? traces[i].lineStyle : undefined;
    const y = by + pad + i * lineH + lineH / 2;

    // Sample line
    ctx.strokeStyle = color;
    ctx.lineWidth = lw;
    ctx.setLineDash(getLineDash(lineStyle));
    ctx.beginPath();
    ctx.moveTo(bx + pad, y);
    ctx.lineTo(bx + pad + sampleW, y);
    ctx.stroke();
    ctx.setLineDash([]);

    // Label text
    ctx.fillStyle = "#333";
    ctx.font = `${fontSize}px sans-serif`;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(labels[i], bx + pad + sampleW + gap, y);
  }
}
