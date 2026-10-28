import type { ImagescTrace } from "../../numbl-core/plotting/types.js";
import {
  colormapLookup,
  type ColormapSpec,
} from "../../numbl-core/plotting/colormaps.js";
import {
  imagescColorLimits,
  imagescExtent,
} from "../../numbl-core/plotting/imagescExtent.js";
import type { DataRenderer } from "./types.js";

function drawImagesc(
  ctx: CanvasRenderingContext2D,
  trace: ImagescTrace,
  toCanvasX: (v: number) => number,
  toCanvasY: (v: number) => number,
  colormap: ColormapSpec,
  caxis?: [number, number]
) {
  const { rows, cols, z, x, y } = trace;
  if (rows === 0 || cols === 0) return;

  const limits = imagescColorLimits(trace, caxis);
  if (!limits) return;
  const [zMin, zMax] = limits;
  const zRange = zMax - zMin || 1;

  // Pixel (i, j) is centered at (x[0] + j*dx, y[0] + i*dy)
  const { dx, dy } = imagescExtent(trace);

  for (let j = 0; j < cols; j++) {
    for (let i = 0; i < rows; i++) {
      const val = z[j * rows + i]; // column-major
      // As in MATLAB, NaN takes the first colormap color; +-Inf clamp.
      const t = Number.isNaN(val) ? 0 : (val - zMin) / zRange;
      const [r, g, b] = colormapLookup(t, colormap);

      const cxa = toCanvasX(x[0] + (j - 0.5) * dx);
      const cxb = toCanvasX(x[0] + (j + 0.5) * dx);
      const cya = toCanvasY(y[0] + (i - 0.5) * dy);
      const cyb = toCanvasY(y[0] + (i + 0.5) * dy);

      ctx.fillStyle = `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`;
      ctx.fillRect(
        Math.min(cxa, cxb),
        Math.min(cya, cyb),
        Math.abs(cxb - cxa),
        Math.abs(cyb - cya)
      );
    }
  }
}

export const imagescRenderer: DataRenderer<"imagescTrace"> = {
  layer: "data",
  draw: ({ ctx, cmap, axes }, trace, { toCanvasX, toCanvasY }) =>
    drawImagesc(ctx, trace, toCanvasX, toCanvasY, cmap, axes.caxis),
  colorValues: (trace, consume) => trace.z.forEach(consume),
};
