/**
 * Shared helpers for plot argument parsing: color specs, LineSpec strings,
 * Name-Value key detection, and conversion of runtime values to plain
 * number arrays and grids.
 */

import {
  type RuntimeValue,
  type RuntimeTensor,
  isRuntimeTensor,
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeString,
  isRuntimeChar,
} from "../runtime/types.js";
import { toString } from "../runtime/convert.js";
import { allocFloat64Array } from "../runtime/alloc.js";

// ── Color mapping ───────────────────────────────────────────────────────

export const COLOR_SHORT: Record<string, [number, number, number]> = {
  r: [1, 0, 0],
  g: [0, 1, 0],
  b: [0, 0, 1],
  c: [0, 1, 1],
  m: [1, 0, 1],
  y: [1, 1, 0],
  k: [0, 0, 0],
  w: [1, 1, 1],
};

const COLOR_NAMES: Record<string, [number, number, number]> = {
  red: [1, 0, 0],
  green: [0, 1, 0],
  blue: [0, 0, 1],
  cyan: [0, 1, 1],
  magenta: [1, 0, 1],
  yellow: [1, 1, 0],
  black: [0, 0, 0],
  white: [1, 1, 1],
};

export function resolveColor(
  v: RuntimeValue | string
): [number, number, number] | undefined {
  if (typeof v === "string") {
    const lower = v.toLowerCase();
    if (COLOR_SHORT[lower]) return COLOR_SHORT[lower];
    if (COLOR_NAMES[lower]) return COLOR_NAMES[lower];
    return undefined;
  }
  if (isRuntimeString(v) || isRuntimeChar(v)) {
    return resolveColor(isRuntimeString(v) ? v : v.value);
  }
  if (isRuntimeTensor(v) && v.data.length === 3) {
    return [v.data[0], v.data[1], v.data[2]];
  }
  return undefined;
}

// ── LineSpec parser ─────────────────────────────────────────────────────

const COLOR_CHARS = new Set(["r", "g", "b", "c", "m", "y", "k", "w"]);
const MARKER_CHARS = new Set([
  "o",
  "+",
  "*",
  ".",
  "x",
  "_",
  "|",
  "s",
  "d",
  "^",
  "v",
  "<",
  ">",
  "p",
  "h",
]);

export interface ParsedLineSpec {
  color?: string;
  lineStyle?: string;
  marker?: string;
}

export function parseLineSpec(s: string): ParsedLineSpec | null {
  const result: ParsedLineSpec = {};
  let pos = 0;

  while (pos < s.length) {
    // Try two-char line styles first
    if (pos + 1 < s.length) {
      const two = s.slice(pos, pos + 2);
      if (two === "--" || two === "-.") {
        if (result.lineStyle !== undefined) return null;
        result.lineStyle = two;
        pos += 2;
        continue;
      }
    }

    const ch = s[pos];

    // Single-char line style: '-' or ':'
    if (ch === "-" || ch === ":") {
      if (result.lineStyle !== undefined) return null;
      result.lineStyle = ch;
      pos++;
      continue;
    }

    // Color
    if (COLOR_CHARS.has(ch)) {
      if (result.color !== undefined) return null;
      result.color = ch;
      pos++;
      continue;
    }

    // Marker
    if (MARKER_CHARS.has(ch)) {
      // '.' could be ambiguous but since we checked '-.' above, standalone '.' is a marker
      if (result.marker !== undefined) return null;
      result.marker = ch;
      pos++;
      continue;
    }

    // Unrecognized character → not a LineSpec
    return null;
  }

  // Must have matched at least one thing
  if (
    result.color === undefined &&
    result.lineStyle === undefined &&
    result.marker === undefined
  )
    return null;

  return result;
}

// ── Name-Value key detection ────────────────────────────────────────────

const NAME_VALUE_KEYS = new Set([
  "color",
  "linestyle",
  "linewidth",
  "marker",
  "markersize",
  "markeredgecolor",
  "markerfacecolor",
  "markerindices",
]);

export function isNameValueKey(v: RuntimeValue): string | null {
  if (!isRuntimeString(v) && !isRuntimeChar(v)) return null;
  const lower = isRuntimeString(v)
    ? v.toLocaleLowerCase()
    : v.value.toLowerCase();
  if (NAME_VALUE_KEYS.has(lower)) return lower;
  return null;
}

// ── Numeric data helpers ────────────────────────────────────────────────

export function isNumericArg(v: unknown): boolean {
  if (typeof v === "number") return true;
  if (typeof v === "boolean") return true;
  if (v && typeof v === "object" && "kind" in v) {
    const mv = v as RuntimeValue;
    return isRuntimeNumber(mv) || isRuntimeTensor(mv) || isRuntimeLogical(mv);
  }
  return false;
}

export function isStringArg(v: unknown): boolean {
  if (typeof v === "string") return true;
  if (v && typeof v === "object" && "kind" in v) {
    const mv = v as RuntimeValue;
    return isRuntimeString(mv) || isRuntimeChar(mv);
  }
  return false;
}

export function getStringValue(v: RuntimeValue): string {
  if (isRuntimeString(v)) return v;
  if (isRuntimeChar(v)) return v.value;
  return toString(v);
}

/** Convert runtime value to a flat number array */
export function toNumberArray(v: RuntimeValue): number[] {
  if (isRuntimeNumber(v)) return [v];
  if (isRuntimeTensor(v)) return Array.from(v.data);
  if (isRuntimeLogical(v)) return [v ? 1 : 0];
  return [0];
}

/** Extract column j from a column-major tensor with shape [m, n] */
export function tensorColumn(tensor: RuntimeTensor, col: number): number[] {
  const m = tensor.shape[0];
  const offset = col * m;
  const result = new Array(m);
  for (let i = 0; i < m; i++) {
    result[i] = tensor.data[offset + i];
  }
  return result;
}

/**
 * Get the number of columns of a numeric value for matrix splitting.
 * Only returns > 1 for true matrices (nRows > 1 && nCols > 1).
 * Row vectors (1×N) and column vectors (N×1) return 1.
 */
export function numColumns(v: RuntimeValue): number {
  if (
    isRuntimeTensor(v) &&
    v.shape.length >= 2 &&
    v.shape[0] > 1 &&
    v.shape[1] > 1
  )
    return v.shape[1];
  return 1;
}

/** Generate 1-based index array [1, 2, ..., n] */
export function oneBasedIndices(n: number): number[] {
  const arr = new Array(n);
  for (let i = 0; i < n; i++) arr[i] = i + 1;
  return arr;
}

/**
 * Expand a numeric value into column-wise traces.
 * For a vector (row or column), returns one array.
 * For an m×n matrix (m>1, n>1), returns n arrays (one per column).
 */
export function expandColumns(v: RuntimeValue): number[][] {
  if (
    isRuntimeTensor(v) &&
    v.shape.length >= 2 &&
    v.shape[0] > 1 &&
    v.shape[1] > 1
  ) {
    const cols: number[][] = [];
    for (let j = 0; j < v.shape[1]; j++) {
      cols.push(tensorColumn(v, j));
    }
    return cols;
  }
  // Scalar or vector — single column
  return [toNumberArray(v)];
}

/** A leading argument that should be skipped as an axes handle. numbl has no
 *  real axes handles, so `line(ax, ___)` is accepted and the handle ignored. */
export function isAxesHandleArg(v: RuntimeValue): boolean {
  return (
    !!v &&
    typeof v === "object" &&
    "kind" in v &&
    ((v as RuntimeValue & { kind: string }).kind === "graphics_handle" ||
      (v as RuntimeValue & { kind: string }).kind === "dummy_handle")
  );
}

export function getMatrixInfo(v: RuntimeValue): {
  data: number[];
  rows: number;
  cols: number;
} {
  if (isRuntimeTensor(v) && v.shape.length >= 2) {
    return {
      data: Array.from(v.data),
      rows: v.shape[0],
      cols: v.shape[1],
    };
  }
  if (isRuntimeNumber(v)) {
    return { data: [v], rows: 1, cols: 1 };
  }
  // Fallback: treat as column vector
  const arr = toNumberArray(v);
  return { data: arr, rows: arr.length, cols: 1 };
}

/** Generate meshgrid X and Y for a rows×cols grid: X = 1:cols, Y = 1:rows (column-major) */
export function generateMeshgrid(
  rows: number,
  cols: number
): { x: number[]; y: number[] } {
  const n = rows * cols;
  const x = new Array(n);
  const y = new Array(n);
  for (let j = 0; j < cols; j++) {
    for (let i = 0; i < rows; i++) {
      const idx = j * rows + i; // column-major
      x[idx] = j + 1;
      y[idx] = i + 1;
    }
  }
  return { x, y };
}

/**
 * Expand X and Y to match the grid dimensions of Z.
 * Handles:
 *   - X and Y as full m×n matrices (pass through)
 *   - X as 1×n row vector, Y as m×1 column vector (meshgrid expansion)
 *   - X as n-element vector, Y as m-element vector (meshgrid expansion)
 */
export function expandXY(
  xVal: RuntimeValue,
  yVal: RuntimeValue,
  rows: number,
  cols: number
): { x: number[]; y: number[] } {
  const xArr = toNumberArray(xVal);
  const yArr = toNumberArray(yVal);
  const n = rows * cols;

  // If X and Y are already full matrices, use them directly
  if (xArr.length === n && yArr.length === n) {
    return { x: xArr, y: yArr };
  }

  // Meshgrid expansion: X has cols elements, Y has rows elements
  const xVec = xArr.length === cols ? xArr : xArr.slice(0, cols);
  const yVec = yArr.length === rows ? yArr : yArr.slice(0, rows);
  const x = new Array(n);
  const y = new Array(n);
  for (let j = 0; j < cols; j++) {
    for (let i = 0; i < rows; i++) {
      const idx = j * rows + i; // column-major
      x[idx] = xVec[j];
      y[idx] = yVec[i];
    }
  }
  return { x, y };
}

/** Get string value if the value is a string/char, otherwise undefined */
export function getStringValueIfString(v: RuntimeValue): string | undefined {
  if (isRuntimeString(v)) return v;
  if (isRuntimeChar(v)) return v.value;
  return undefined;
}

/** Create a runtime tensor from a plain number array (for synthetic args). */
export function tensorFromArray(arr: number[]): RuntimeValue {
  return {
    kind: "tensor",
    data: allocFloat64Array(arr),
    shape: [1, arr.length],
  } as RuntimeTensor;
}
