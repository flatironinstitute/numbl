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

const LUT_SIZE = 1024;

/** Draw the image as one bitmap scaled onto the axes: fast, and without the
 *  seams that anti-aliased per-cell rectangles leave. Linear axes only. */
function drawImagescBitmap(
  ctx: CanvasRenderingContext2D,
  trace: ImagescTrace,
  toCanvasX: (v: number) => number,
  toCanvasY: (v: number) => number,
  colormap: ColormapSpec,
  zMin: number,
  zRange: number,
  dx: number,
  dy: number
) {
  const { rows, cols, z, x, y } = trace;
  const lut = new Uint8ClampedArray(LUT_SIZE * 3);
  for (let k = 0; k < LUT_SIZE; k++) {
    const [r, g, b] = colormapLookup(k / (LUT_SIZE - 1), colormap);
    lut[3 * k] = Math.round(r * 255);
    lut[3 * k + 1] = Math.round(g * 255);
    lut[3 * k + 2] = Math.round(b * 255);
  }
  // Canvas positions of the image's outer edges.
  const xa = toCanvasX(x[0] - 0.5 * dx);
  const xb = toCanvasX(x[0] + (cols - 0.5) * dx);
  const ya = toCanvasY(y[0] - 0.5 * dy);
  const yb = toCanvasY(y[0] + (rows - 0.5) * dy);
  const flipX = xb < xa;
  const flipY = yb < ya;

  const img = new ImageData(cols, rows);
  const px = img.data;
  for (let j = 0; j < cols; j++) {
    const col = flipX ? cols - 1 - j : j;
    for (let i = 0; i < rows; i++) {
      // As in MATLAB, NaN takes the first colormap color; +-Inf as in
      // colormapLookup.
      const t = (z[j * rows + i] - zMin) / zRange;
      const k = Number.isFinite(t)
        ? Math.round(Math.max(0, Math.min(1, t)) * (LUT_SIZE - 1))
        : 0;
      const o = 4 * ((flipY ? rows - 1 - i : i) * cols + col);
      px[o] = lut[3 * k];
      px[o + 1] = lut[3 * k + 1];
      px[o + 2] = lut[3 * k + 2];
      px[o + 3] = 255;
    }
  }
  const off = document.createElement("canvas");
  off.width = cols;
  off.height = rows;
  off.getContext("2d")?.putImageData(img, 0, 0);
  ctx.save();
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(
    off,
    Math.min(xa, xb),
    Math.min(ya, yb),
    Math.abs(xb - xa),
    Math.abs(yb - ya)
  );
  ctx.restore();
}

function drawImagesc(
  ctx: CanvasRenderingContext2D,
  trace: ImagescTrace,
  toCanvasX: (v: number) => number,
  toCanvasY: (v: number) => number,
  colormap: ColormapSpec,
  caxis: [number, number] | undefined,
  linearAxes: boolean
) {
  const { rows, cols, z, x, y } = trace;
  if (rows === 0 || cols === 0) return;

  const limits = imagescColorLimits(trace, caxis);
  if (!limits) return;
  const [zMin, zMax] = limits;
  const zRange = zMax - zMin || 1;

  // Pixel (i, j) is centered at (x[0] + j*dx, y[0] + i*dy)
  const { dx, dy } = imagescExtent(trace);

  if (linearAxes && typeof document !== "undefined") {
    drawImagescBitmap(
      ctx,
      trace,
      toCanvasX,
      toCanvasY,
      colormap,
      zMin,
      zRange,
      dx,
      dy
    );
    return;
  }

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
    drawImagesc(
      ctx,
      trace,
      toCanvasX,
      toCanvasY,
      cmap,
      axes.caxis,
      axes.axisScale === undefined || axes.axisScale === "linear"
    ),
  colorValues: (trace, consume) => trace.z.forEach(consume),
};
