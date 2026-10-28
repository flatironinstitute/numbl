import type { ContourTrace } from "../../numbl-core/plotting/types.js";
import {
  colormapLookup,
  type ColormapSpec,
} from "../../numbl-core/plotting/colormaps.js";
import type { DataRenderer } from "./types.js";

function drawContour(
  ctx: CanvasRenderingContext2D,
  trace: ContourTrace,
  toCanvasX: (v: number) => number,
  toCanvasY: (v: number) => number,
  colormap: ColormapSpec
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

  // Contour levels: use the explicit list when given, else evenly spaced
  // across the data range. The line-contour loop below skips the first and
  // last entries (open boundary contours), so pad an explicit list with
  // sentinel endpoints to keep every requested level drawn.
  let levels: number[];
  if (trace.levels && trace.levels.length > 0) {
    levels = [zMin - zRange, ...trace.levels, zMax + zRange];
  } else {
    levels = [];
    for (let i = 0; i <= nLevels; i++) {
      levels.push(zMin + (i / nLevels) * zRange);
    }
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
        // NaN regions stay empty, as in MATLAB's contourf
        if (Number.isNaN(zAvg)) continue;
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
      ctx.lineWidth = trace.lineWidth ?? 1;

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

export const contourRenderer: DataRenderer<"contourTraces"> = {
  layer: "data",
  draw: ({ ctx, cmap }, traces, { toCanvasX, toCanvasY }) => {
    for (const ct of traces) drawContour(ctx, ct, toCanvasX, toCanvasY, cmap);
  },
  colorValues: (traces, consume) => {
    for (const ct of traces) ct.z.forEach(consume);
  },
};
