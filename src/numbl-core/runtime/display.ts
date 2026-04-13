/**
 * Value display/formatting.
 */

import {
  type RuntimeCell,
  type RuntimeClassInstance,
  type RuntimeDictionary,
  type RuntimeSparseMatrix,
  type RuntimeStruct,
  type RuntimeTensor,
  type RuntimeValue,
  isRuntimeChar,
  isRuntimeLogical,
  isRuntimeNumber,
  isRuntimeString,
  RuntimeStructArray,
} from "./types.js";
import { colMajorIndex, ind2sub } from "./utils.js";

/** Format a value for display */
export function displayValue(v: RuntimeValue): string {
  if (isRuntimeNumber(v)) {
    return formatNumber(v);
  }
  if (isRuntimeString(v)) {
    return v;
  }
  if (isRuntimeLogical(v)) {
    return v ? "1" : "0";
  }
  switch (v.kind) {
    case "char": {
      if (v.shape && v.shape[0] > 1) {
        // Multi-row char array: split into rows of shape[1] chars
        const rowWidth = v.shape[1];
        const rows: string[] = [];
        for (let r = 0; r < v.shape[0]; r++) {
          rows.push(v.value.slice(r * rowWidth, (r + 1) * rowWidth));
        }
        return rows.join("\n");
      }
      return v.value;
    }
    case "tensor":
      return formatTensor(v);
    case "cell":
      return formatCell(v);
    case "struct":
      return formatStruct(v);
    case "function":
      return `@${v.name}`;
    case "class_instance":
      return formatClassInstance(v);
    case "complex_number":
      return formatComplex(v.re, v.im);
    case "dummy_handle":
      return "[dummy_handle]";
    case "graphics_handle":
      return `[graphics_handle: ${v._traceType}]`;
    case "struct_array":
      return formatStructArray(v);
    case "sparse_matrix":
      return formatSparseMatrix(v);
    case "dictionary":
      return formatDictionary(v);
  }
}

const formatStructArray = (v: RuntimeStructArray): string => {
  const elements = v.elements;
  const fieldNames = v.fieldNames;
  const formattedElements = elements
    .map((element, index) => {
      const elementStr = fieldNames
        .map(fieldName => {
          const fieldValue = element.fields.get(fieldName);
          if (fieldValue == undefined) return `${fieldName}: <undefined>`;
          return `${fieldName}: ${displayValue(fieldValue)}`;
        })
        .join("\n");
      return `  ${index + 1}: [${elementStr}]`;
    })
    .join("\n");

  return `struct array with fields:\n${fieldNames.join("\n")}\n\n${formattedElements}`;
};

function formatComplex(re: number, im: number): string {
  if (im === 0) return formatNumber(re);
  if (re === 0) return `${formatNumber(im)}i`;
  if (im < 0) return `${formatNumber(re)} - ${formatNumber(-im)}i`;
  return `${formatNumber(re)} + ${formatNumber(im)}i`;
}

function formatNumber(n: number): string {
  if (Number.isInteger(n) && Math.abs(n) < 1e15) {
    return n.toString();
  }
  // 4 decimal places for most numbers
  const s = n.toPrecision(5);
  // Remove trailing zeros after decimal
  if (s.includes(".")) {
    return s.replace(/\.?0+$/, "") || "0";
  }
  return s;
}

/** Max rows/cols to display before truncating with "..." */
const MAX_DISPLAY_ROWS = 20;
const MAX_DISPLAY_COLS = 20;

function formatTensor(t: RuntimeTensor): string {
  if (t.data.length === 0) {
    return "[]";
  }
  const shape = t.shape;
  const rows = shape.length >= 1 ? shape[0] : 1;
  const cols = shape.length >= 2 ? shape[1] : 1;
  const isComplex = t.imag !== undefined;

  if (t.data.length === 1) {
    if (isComplex) {
      return formatComplex(t.data[0], t.imag![0]);
    }
    return formatNumber(t.data[0]);
  }

  // For 2D tensors (or fewer), display as a single matrix
  if (shape.length <= 2) {
    return format2DSlice(t.data, t.imag, rows, cols, isComplex);
  }

  // For 3D+ tensors, display page-by-page
  const higherDims = shape.slice(2);
  const pageSize = rows * cols;
  const numPages = higherDims.reduce((a, b) => a * b, 1);
  const lines: string[] = [];

  for (let p = 0; p < numPages; p++) {
    // Compute the higher-dim subscripts for this page
    const pageSubs = ind2sub(higherDims, p);
    const label = pageSubs.map(s => s + 1).join(",");
    if (lines.length > 0) lines.push("");
    lines.push(`(:,:,${label}) =\n`);

    // Extract this page's data
    const baseOffset = p * pageSize;
    const pageData = t.data.slice(baseOffset, baseOffset + pageSize);
    const pageImag = t.imag
      ? t.imag.slice(baseOffset, baseOffset + pageSize)
      : undefined;
    lines.push(format2DSlice(pageData, pageImag, rows, cols, isComplex));
  }

  return lines.join("\n");
}

function format2DSlice(
  data: ArrayLike<number>,
  imag: ArrayLike<number> | undefined,
  rows: number,
  cols: number,
  isComplex: boolean
): string {
  const truncateRows = rows > MAX_DISPLAY_ROWS;
  const truncateCols = cols > MAX_DISPLAY_COLS;

  // Which rows/cols to show
  const showRows = truncateRows
    ? [
        ...Array(Math.ceil(MAX_DISPLAY_ROWS / 2)).keys(),
        ...Array.from(
          { length: Math.floor(MAX_DISPLAY_ROWS / 2) },
          (_, i) => rows - Math.floor(MAX_DISPLAY_ROWS / 2) + i
        ),
      ]
    : Array.from({ length: rows }, (_, i) => i);
  const showCols = truncateCols
    ? [
        ...Array(Math.ceil(MAX_DISPLAY_COLS / 2)).keys(),
        ...Array.from(
          { length: Math.floor(MAX_DISPLAY_COLS / 2) },
          (_, i) => cols - Math.floor(MAX_DISPLAY_COLS / 2) + i
        ),
      ]
    : Array.from({ length: cols }, (_, i) => i);

  const lines: string[] = [];

  // Header for columns that are too wide
  if (truncateRows || truncateCols) {
    lines.push(`  Columns 1 through ${cols}`);
    lines.push("");
  }

  const formatted: string[][] = [];
  const colWidths: number[] = new Array(
    showCols.length + (truncateCols ? 1 : 0)
  ).fill(0);

  for (const r of showRows) {
    const row: string[] = [];
    let ci = 0;
    for (const c of showCols) {
      const idx = colMajorIndex(r, c, rows);
      const s = isComplex
        ? formatComplex(data[idx], imag![idx])
        : formatNumber(data[idx]);
      row.push(s);
      colWidths[ci] = Math.max(colWidths[ci], s.length);
      ci++;
      // Insert ellipsis column between first and last halves
      if (truncateCols && ci === Math.ceil(MAX_DISPLAY_COLS / 2)) {
        row.push("...");
        colWidths[ci] = Math.max(colWidths[ci], 3);
        ci++;
      }
    }
    formatted.push(row);
  }

  let fi = 0;
  for (let si = 0; si < showRows.length; si++) {
    // Insert ellipsis row between first and last halves
    if (truncateRows && si === Math.ceil(MAX_DISPLAY_ROWS / 2)) {
      const ellipsisRow = colWidths.map(w => "...".padStart(w));
      lines.push("   " + ellipsisRow.join("   "));
    }
    const parts = formatted[fi].map((s, c) => s.padStart(colWidths[c]));
    lines.push("   " + parts.join("   "));
    fi++;
  }

  return lines.join("\n");
}

function formatCell(c: RuntimeCell): string {
  const parts = c.data.map(v => {
    if (isRuntimeChar(v)) return `'${v.value}'`;
    if (isRuntimeString(v)) return `"${v}"`;
    return displayValue(v);
  });
  return `{${parts.join(", ")}}`;
}

function formatStruct(s: RuntimeStruct): string {
  const lines: string[] = [];
  for (const [key, val] of s.fields) {
    lines.push(`    ${key}: ${displayValue(val)}`);
  }
  return lines.join("\n");
}

function formatSparseMatrix(v: RuntimeSparseMatrix): string {
  const nnz = v.jc[v.n];
  if (nnz === 0) {
    return `   All zero sparse: ${v.m}x${v.n}`;
  }
  const lines: string[] = [];
  for (let col = 0; col < v.n; col++) {
    for (let k = v.jc[col]; k < v.jc[col + 1]; k++) {
      const row = v.ir[k] + 1; // 1-based
      const c = col + 1; // 1-based
      const val = v.pi
        ? formatComplex(v.pr[k], v.pi[k])
        : formatNumber(v.pr[k]);
      lines.push(
        `   (${row},${c})${" ".repeat(Math.max(1, 10 - `(${row},${c})`.length))}${val}`
      );
    }
  }
  return lines.join("\n");
}

function formatClassInstance(v: RuntimeClassInstance): string {
  const lines: string[] = [`  ${v.className} with properties:\n`];
  for (const [key, val] of v.fields) {
    lines.push(`    ${key}: ${displayValue(val)}`);
  }
  return lines.join("\n");
}

function formatDictionary(d: RuntimeDictionary): string {
  if (!d.keyType && !d.valueType && d.entries.size === 0) {
    return "  dictionary with unset key and value types.";
  }
  const kt = d.keyType ?? "unset";
  const vt = d.valueType ?? "unset";
  const n = d.entries.size;
  if (n === 0) {
    return `  dictionary (${kt} \u27FC ${vt}) with no entries.`;
  }
  const header = `  dictionary (${kt} \u27FC ${vt}) with ${n} ${n === 1 ? "entry" : "entries"}:\n`;
  const lines: string[] = [header];

  // Format keys and values, compute alignment
  const formatted: { keyStr: string; valStr: string }[] = [];
  let maxKeyLen = 0;
  for (const { key, value } of d.entries.values()) {
    const keyStr = formatDictKey(key);
    const valStr = displayValue(value);
    formatted.push({ keyStr, valStr });
    maxKeyLen = Math.max(maxKeyLen, keyStr.length);
  }
  for (const { keyStr, valStr } of formatted) {
    lines.push(`    ${keyStr.padEnd(maxKeyLen)} \u27FC ${valStr}`);
  }
  return lines.join("\n");
}

function formatDictKey(v: RuntimeValue): string {
  if (isRuntimeString(v)) return `"${v}"`;
  if (isRuntimeChar(v)) return `'${v.value}'`;
  return displayValue(v);
}
