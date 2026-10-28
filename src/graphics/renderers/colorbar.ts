import {
  colormapLookup,
  type ColormapSpec,
} from "../../numbl-core/plotting/colormaps.js";

export function formatCbVal(v: number): string {
  if (Number.isInteger(v)) return String(v);
  return v.toPrecision(3);
}

/**
 * Draw a colorbar at one of the 8 MATLAB locations relative to the plot
 * area at (plotL, plotT, plotW, plotH).
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
