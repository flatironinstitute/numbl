import {
  colormapLookup,
  type ColormapSpec,
} from "../../numbl-core/plotting/colormaps.js";

export function formatCbVal(v: number): string {
  if (Number.isInteger(v)) return String(v);
  return v.toPrecision(3);
}

/** Where a colorbar was drawn, in CSS pixels. */
export interface ColorbarRect {
  left: number;
  top: number;
  width: number;
  height: number;
  vertical: boolean;
}

/**
 * Draw a colorbar at one of the 8 MATLAB locations relative to the plot
 * area at (plotL, plotT, plotW, plotH). Returns where it was drawn.
 */
export function drawColorbarAtLocation(
  ctx: CanvasRenderingContext2D,
  location: string,
  plotL: number,
  plotT: number,
  plotW: number,
  plotH: number,
  dMin: number,
  dMax: number,
  colormap: ColormapSpec
): ColorbarRect | undefined {
  const t = 16; // bar thickness
  const inset = 8; // gap between plot edge and colorbar for inside variants
  const gap = 10; // gap between plot edge and colorbar for outside variants

  // [left, top, width, height, label side]
  type Geom = [
    number,
    number,
    number,
    number,
    "left" | "right" | "top" | "bottom",
  ];
  const geoms: Record<string, Geom> = {
    eastoutside: [plotL + plotW + gap, plotT, t, plotH, "right"],
    westoutside: [plotL - gap - t, plotT, t, plotH, "left"],
    northoutside: [plotL, plotT - gap - t, plotW, t, "top"],
    southoutside: [plotL, plotT + plotH + gap, plotW, t, "bottom"],
    east: [
      plotL + plotW - inset - t,
      plotT + inset,
      t,
      plotH - 2 * inset,
      "left",
    ],
    west: [plotL + inset, plotT + inset, t, plotH - 2 * inset, "right"],
    north: [plotL + inset, plotT + inset, plotW - 2 * inset, t, "bottom"],
    south: [
      plotL + inset,
      plotT + plotH - inset - t,
      plotW - 2 * inset,
      t,
      "top",
    ],
  };
  const g = geoms[location];
  if (!g) return undefined;
  const [left, top, width, height, side] = g;
  const vertical = side === "left" || side === "right";
  if (vertical) {
    drawColorbarVertical(
      ctx,
      left,
      top,
      width,
      height,
      dMin,
      dMax,
      colormap,
      side
    );
  } else {
    drawColorbarHorizontal(
      ctx,
      left,
      top,
      width,
      height,
      dMin,
      dMax,
      colormap,
      side
    );
  }
  return { left, top, width, height, vertical };
}

function drawColorbarVertical(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  dMin: number,
  dMax: number,
  colormap: ColormapSpec,
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
  colormap: ColormapSpec,
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
