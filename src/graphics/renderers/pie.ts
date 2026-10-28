import type { PieTrace } from "../../numbl-core/plotting/types.js";
import type { StandaloneRenderer } from "./types.js";

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

export const pieRenderer: StandaloneRenderer<"pieTrace"> = {
  layer: "standalone",
  draw: ({ ctx, axes }, trace, width, height) =>
    drawPieChart(ctx, width, height, trace, axes.title),
};
