/**
 * streamline / stream2: trace 2-D streamlines through a gridded vector field.
 */

import type { RuntimeValue } from "../runtime/index.js";
import type { PlotInstruction } from "./types.js";
import {
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeTensor,
  isRuntimeCell,
  type RuntimeTensor,
  type RuntimeCell,
} from "../runtime/types.js";
import { RTV } from "../runtime/constructors.js";
import { ensureRuntimeValue } from "../runtime/runtimeHelpers.js";
import { allocFloat64Array } from "../runtime/alloc.js";

// ── Streamline helpers ────────────────────────────────────────────────

/** Extract a flat number[] from a RuntimeValue that is a number or tensor. */
function toNumArr(v: RuntimeValue): number[] {
  if (isRuntimeNumber(v)) return [v];
  if (isRuntimeTensor(v)) return Array.from(v.data);
  if (isRuntimeLogical(v)) return [v ? 1 : 0];
  throw new Error("streamline: expected numeric array");
}

/**
 * Bilinear interpolation of a scalar field on a meshgrid.
 *
 * X, Y are the grid coordinates (column-major, rows × cols).
 * F is the field values on that grid.
 * Returns interpolated value at (px, py), or NaN if out of bounds.
 */
function interpGrid(
  xVec: number[],
  yVec: number[],
  F: number[],
  rows: number,
  cols: number,
  px: number,
  py: number
): number {
  // Find column index (x direction)
  let ci = -1;
  for (let j = 0; j < cols - 1; j++) {
    if (
      (xVec[j] <= px && px <= xVec[j + 1]) ||
      (xVec[j] >= px && px >= xVec[j + 1])
    ) {
      ci = j;
      break;
    }
  }
  if (ci < 0) return NaN;

  // Find row index (y direction)
  let ri = -1;
  for (let i = 0; i < rows - 1; i++) {
    if (
      (yVec[i] <= py && py <= yVec[i + 1]) ||
      (yVec[i] >= py && py >= yVec[i + 1])
    ) {
      ri = i;
      break;
    }
  }
  if (ri < 0) return NaN;

  // Bilinear weights
  const dx = xVec[ci + 1] - xVec[ci];
  const dy = yVec[ri + 1] - yVec[ri];
  if (dx === 0 || dy === 0) return NaN;
  const tx = (px - xVec[ci]) / dx;
  const ty = (py - yVec[ri]) / dy;

  // Column-major: index = row + col * rows
  const f00 = F[ri + ci * rows];
  const f10 = F[ri + 1 + ci * rows];
  const f01 = F[ri + (ci + 1) * rows];
  const f11 = F[ri + 1 + (ci + 1) * rows];

  return (
    f00 * (1 - tx) * (1 - ty) +
    f01 * tx * (1 - ty) +
    f10 * (1 - tx) * ty +
    f11 * tx * ty
  );
}

/**
 * Compute 2D streamline vertices from a meshgrid vector field.
 *
 * X, Y: grid coordinate arrays from meshgrid (rows × cols, column-major)
 * U, V: velocity component arrays (same layout)
 * startX, startY: starting position arrays
 * step: integration step size (default 0.1)
 * maxVert: max vertices per streamline (default 10000)
 *
 * Returns array of vertex lists, one per starting point.
 * Each vertex list is {x: number[], y: number[]}.
 */
function computeStreamlines2D(
  X: RuntimeTensor,
  Y: RuntimeTensor,
  U: RuntimeTensor,
  V: RuntimeTensor,
  startX: number[],
  startY: number[],
  step: number,
  maxVert: number
): { x: number[]; y: number[] }[] {
  const rows = X.shape[0];
  const cols = X.shape[1];

  // Extract unique x and y vectors from the meshgrid.
  // X varies along columns (same value in each row), Y varies along rows.
  const xVec: number[] = new Array(cols);
  for (let j = 0; j < cols; j++) xVec[j] = X.data[j * rows]; // first row
  const yVec: number[] = new Array(rows);
  for (let i = 0; i < rows; i++) yVec[i] = Y.data[i]; // first column

  const Udata = Array.from(U.data);
  const Vdata = Array.from(V.data);

  // Characteristic cell size for scaling the step
  const hx = Math.abs(xVec[cols - 1] - xVec[0]) / (cols - 1);
  const hy = Math.abs(yVec[rows - 1] - yVec[0]) / (rows - 1);
  const cellSize = Math.sqrt(hx * hx + hy * hy);

  const results: { x: number[]; y: number[] }[] = [];

  for (let k = 0; k < startX.length; k++) {
    const xs: number[] = [startX[k]];
    const ys: number[] = [startY[k]];
    let px = startX[k];
    let py = startY[k];

    for (let iter = 0; iter < maxVert - 1; iter++) {
      // RK4 integration
      const u1 = interpGrid(xVec, yVec, Udata, rows, cols, px, py);
      const v1 = interpGrid(xVec, yVec, Vdata, rows, cols, px, py);
      if (isNaN(u1) || isNaN(v1)) break;

      const speed1 = Math.sqrt(u1 * u1 + v1 * v1);
      if (speed1 === 0) break;
      const ds = step * cellSize;
      const dt1 = ds / speed1;

      const px2 = px + 0.5 * dt1 * u1;
      const py2 = py + 0.5 * dt1 * v1;
      const u2 = interpGrid(xVec, yVec, Udata, rows, cols, px2, py2);
      const v2 = interpGrid(xVec, yVec, Vdata, rows, cols, px2, py2);
      if (isNaN(u2) || isNaN(v2)) break;

      const px3 = px + 0.5 * dt1 * u2;
      const py3 = py + 0.5 * dt1 * v2;
      const u3 = interpGrid(xVec, yVec, Udata, rows, cols, px3, py3);
      const v3 = interpGrid(xVec, yVec, Vdata, rows, cols, px3, py3);
      if (isNaN(u3) || isNaN(v3)) break;

      const px4 = px + dt1 * u3;
      const py4 = py + dt1 * v3;
      const u4 = interpGrid(xVec, yVec, Udata, rows, cols, px4, py4);
      const v4 = interpGrid(xVec, yVec, Vdata, rows, cols, px4, py4);
      if (isNaN(u4) || isNaN(v4)) break;

      const dx = (dt1 / 6) * (u1 + 2 * u2 + 2 * u3 + u4);
      const dy = (dt1 / 6) * (v1 + 2 * v2 + 2 * v3 + v4);

      px += dx;
      py += dy;
      xs.push(px);
      ys.push(py);
    }

    results.push({ x: xs, y: ys });
  }

  return results;
}

/**
 * Parse streamline/stream2 arguments.
 *
 * Supported forms:
 *   streamline(X,Y,U,V,startX,startY)
 *   streamline(X,Y,U,V,startX,startY,options)
 *   streamline(verts)  — cell array of Nx2 vertex matrices
 */
function parseStreamlineArgs(args: RuntimeValue[]):
  | {
      mode: "field";
      X: RuntimeTensor;
      Y: RuntimeTensor;
      U: RuntimeTensor;
      V: RuntimeTensor;
      startX: number[];
      startY: number[];
      step: number;
      maxVert: number;
    }
  | {
      mode: "verts";
      verts: RuntimeCell;
    } {
  if (args.length === 1 && isRuntimeCell(args[0])) {
    return { mode: "verts", verts: args[0] };
  }

  if (args.length < 4) {
    throw new Error("streamline: expected (X,Y,U,V,startX,startY) or (verts)");
  }

  let step = 0.1;
  let maxVert = 10000;

  // Check for trailing options vector
  let nPositional = args.length;
  if (nPositional === 5 || nPositional === 7) {
    const last = args[nPositional - 1];
    if (
      isRuntimeTensor(last) &&
      last.data.length <= 2 &&
      last.data.length >= 1
    ) {
      step = last.data[0];
      if (last.data.length === 2) maxVert = last.data[1];
      nPositional--;
    }
  }

  if (nPositional === 4) {
    // streamline(U,V,startX,startY) — default grid coordinates
    const U = ensureRuntimeValue(args[0]) as RuntimeTensor;
    const V = ensureRuntimeValue(args[1]) as RuntimeTensor;
    if (!isRuntimeTensor(U) || !isRuntimeTensor(V)) {
      throw new Error("streamline: U and V must be numeric arrays");
    }
    const rows = U.shape[0];
    const cols = U.shape[1];
    // Default coordinates: X = 1:cols, Y = 1:rows (meshgrid convention)
    const xData = allocFloat64Array(rows * cols);
    const yData = allocFloat64Array(rows * cols);
    for (let j = 0; j < cols; j++) {
      for (let i = 0; i < rows; i++) {
        xData[i + j * rows] = j + 1;
        yData[i + j * rows] = i + 1;
      }
    }
    const X = RTV.tensor(xData, [rows, cols]);
    const Y = RTV.tensor(yData, [rows, cols]);
    const startX = toNumArr(args[2]);
    const startY = toNumArr(args[3]);
    return { mode: "field", X, Y, U, V, startX, startY, step, maxVert };
  }

  if (nPositional >= 6) {
    // streamline(X,Y,U,V,startX,startY)
    const X = ensureRuntimeValue(args[0]) as RuntimeTensor;
    const Y = ensureRuntimeValue(args[1]) as RuntimeTensor;
    const U = ensureRuntimeValue(args[2]) as RuntimeTensor;
    const V = ensureRuntimeValue(args[3]) as RuntimeTensor;
    if (
      !isRuntimeTensor(X) ||
      !isRuntimeTensor(Y) ||
      !isRuntimeTensor(U) ||
      !isRuntimeTensor(V)
    ) {
      throw new Error("streamline: X, Y, U, V must be numeric arrays");
    }
    const startX = toNumArr(args[4]);
    const startY = toNumArr(args[5]);
    return { mode: "field", X, Y, U, V, startX, startY, step, maxVert };
  }

  throw new Error("streamline: expected (X,Y,U,V,startX,startY) or (verts)");
}

/**
 * streamline(X,Y,U,V,startX,startY) — compute and plot 2D streamlines.
 * streamline(verts) — plot from pre-computed cell array of vertex matrices.
 * Returns a dummy graphics handle.
 */
export function streamlineCall(
  plotInstructions: PlotInstruction[],
  args: RuntimeValue[]
): RuntimeValue {
  const parsed = parseStreamlineArgs(args);

  if (parsed.mode === "verts") {
    // Plot from cell array of vertex matrices
    const traces = cellVertsToTraces(parsed.verts);
    if (traces.length > 0) {
      plotInstructions.push({ type: "plot", traces });
    }
    return RTV.dummyHandle();
  }

  // Compute streamlines
  const streamlines = computeStreamlines2D(
    parsed.X,
    parsed.Y,
    parsed.U,
    parsed.V,
    parsed.startX,
    parsed.startY,
    parsed.step,
    parsed.maxVert
  );

  const traces = streamlines
    .filter(s => s.x.length > 1)
    .map(s => ({ x: s.x, y: s.y }));

  if (traces.length > 0) {
    plotInstructions.push({ type: "plot", traces });
  }

  return RTV.dummyHandle();
}

/** Convert a cell array of Nx2 (or Nx3) vertex matrices into PlotTraces. */
function cellVertsToTraces(cell: RuntimeCell): { x: number[]; y: number[] }[] {
  const traces: { x: number[]; y: number[] }[] = [];
  for (const elem of cell.data) {
    if (isRuntimeTensor(elem) && elem.shape.length === 2) {
      const nRows = elem.shape[0];
      const nCols = elem.shape[1];
      if (nCols >= 2 && nRows > 0) {
        const x: number[] = new Array(nRows);
        const y: number[] = new Array(nRows);
        for (let i = 0; i < nRows; i++) {
          x[i] = elem.data[i]; // column 0
          y[i] = elem.data[i + nRows]; // column 1
        }
        traces.push({ x, y });
      }
    }
  }
  return traces;
}

/**
 * stream2(X,Y,U,V,startX,startY) — compute 2D streamline vertices.
 * Returns a cell array of Nx2 vertex matrices.
 */
export function stream2Call(args: RuntimeValue[]): RuntimeValue {
  const parsed = parseStreamlineArgs(args);

  if (parsed.mode === "verts") {
    // Already vertices — just return them
    return parsed.verts;
  }

  const streamlines = computeStreamlines2D(
    parsed.X,
    parsed.Y,
    parsed.U,
    parsed.V,
    parsed.startX,
    parsed.startY,
    parsed.step,
    parsed.maxVert
  );

  // Convert to cell array of Nx2 matrices
  const cellData: RuntimeValue[] = streamlines.map(s => {
    const n = s.x.length;
    const data = allocFloat64Array(n * 2);
    for (let i = 0; i < n; i++) {
      data[i] = s.x[i]; // column 0
      data[i + n] = s.y[i]; // column 1
    }
    return RTV.tensor(data, [n, 2]);
  });

  return RTV.cell(cellData, [1, cellData.length]);
}
