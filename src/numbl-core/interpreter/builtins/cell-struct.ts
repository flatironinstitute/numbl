/**
 * Cell/struct operation builtins: cell2mat, cell2struct, mat2cell, num2cell,
 * struct2cell, namedargs2cell, rmfield.
 */

import {
  FloatXArray,
  isRuntimeCell,
  isRuntimeChar,
  isRuntimeComplexNumber,
  isRuntimeLogical,
  isRuntimeNumber,
  isRuntimeStruct,
  isRuntimeStructArray,
  isRuntimeTensor,
} from "../../runtime/types.js";
import type {
  RuntimeValue,
  RuntimeTensor,
  RuntimeStruct,
} from "../../runtime/types.js";
import { RTV, RuntimeError } from "../../runtime/index.js";
import { toNumber, toString } from "../../runtime/convert.js";
import { registerIBuiltin } from "./types.js";

// ── helpers ─────────────────────────────────────────────────────────────

function toNumericVector(v: RuntimeValue, name: string): number[] {
  if (isRuntimeNumber(v)) return [v as number];
  if (isRuntimeLogical(v)) return [v ? 1 : 0];
  if (isRuntimeTensor(v)) return Array.from(v.data);
  throw new RuntimeError(`${name}: argument must be numeric`);
}

function coerceToTensor(v: RuntimeValue, name: string): RuntimeTensor {
  if (isRuntimeTensor(v)) return v;
  if (isRuntimeNumber(v))
    return RTV.tensor(new FloatXArray([v as number]), [1, 1]);
  if (isRuntimeLogical(v))
    return RTV.tensor(new FloatXArray([v ? 1 : 0]), [1, 1]);
  if (isRuntimeComplexNumber(v))
    return RTV.tensor(new FloatXArray([v.re]), [1, 1], new FloatXArray([v.im]));
  throw new RuntimeError(`${name}: argument must be numeric`);
}

// ── cell2mat ────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "cell2mat",
  resolve: argTypes => {
    if (argTypes.length !== 1) return null;
    return {
      outputTypes: [{ kind: "tensor", isComplex: false }],
      apply: args => {
        const C = args[0];
        if (!isRuntimeCell(C))
          throw new RuntimeError("cell2mat: argument must be a cell array");

        const cellRows = C.shape[0];
        const cellCols = C.shape.length >= 2 ? C.shape[1] : 1;

        const asTensor = (v: RuntimeValue) => {
          if (isRuntimeNumber(v))
            return {
              data: new FloatXArray([v as number]),
              shape: [1, 1],
              imag: undefined,
            };
          if (isRuntimeComplexNumber(v))
            return {
              data: new FloatXArray([v.re]),
              shape: [1, 1],
              imag: new FloatXArray([v.im]),
            };
          if (isRuntimeTensor(v))
            return { data: v.data, shape: v.shape, imag: v.imag };
          if (isRuntimeLogical(v))
            return {
              data: new FloatXArray([v ? 1 : 0]),
              shape: [1, 1],
              imag: undefined,
            };
          throw new RuntimeError(
            "cell2mat: cell contents must be numeric arrays"
          );
        };

        const rowHeights: number[] = [];
        for (let ci = 0; ci < cellRows; ci++) {
          const t = asTensor(C.data[0 * cellRows + ci]);
          rowHeights.push(t.shape[0]);
        }
        const colWidths: number[] = [];
        for (let cj = 0; cj < cellCols; cj++) {
          const t = asTensor(C.data[cj * cellRows + 0]);
          colWidths.push(t.shape.length >= 2 ? t.shape[1] : 1);
        }

        const totalRows = rowHeights.reduce((a, b) => a + b, 0);
        const totalCols = colWidths.reduce((a, b) => a + b, 0);
        const resultData = new FloatXArray(totalRows * totalCols);
        let hasImag = false;
        for (let k = 0; k < C.data.length; k++) {
          if (asTensor(C.data[k]).imag) {
            hasImag = true;
            break;
          }
        }
        const resultImag = hasImag
          ? new FloatXArray(totalRows * totalCols)
          : undefined;

        let colOffset = 0;
        for (let cj = 0; cj < cellCols; cj++) {
          let rowOffset = 0;
          for (let ci = 0; ci < cellRows; ci++) {
            const t = asTensor(C.data[cj * cellRows + ci]);
            const bRows = t.shape[0];
            const bCols = t.shape.length >= 2 ? t.shape[1] : 1;
            for (let j = 0; j < bCols; j++) {
              for (let i = 0; i < bRows; i++) {
                const srcIdx = j * bRows + i;
                const dstIdx = (colOffset + j) * totalRows + (rowOffset + i);
                resultData[dstIdx] = t.data[srcIdx];
                if (resultImag)
                  resultImag[dstIdx] = t.imag ? t.imag[srcIdx] : 0;
              }
            }
            rowOffset += rowHeights[ci];
          }
          colOffset += colWidths[cj];
        }
        return RTV.tensor(resultData, [totalRows, totalCols], resultImag);
      },
    };
  },
});

// ── mat2cell ────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "mat2cell",
  resolve: argTypes => {
    if (argTypes.length < 2) return null;
    return {
      outputTypes: [{ kind: "unknown" }],
      apply: args => {
        const distVec = (v: RuntimeValue) =>
          toNumericVector(v, "mat2cell").map(x => Math.round(x));

        // Handle char arrays
        if (isRuntimeChar(args[0])) {
          const str = (args[0] as { kind: "char"; value: string }).value;
          distVec(args[1]);
          const colDist = args.length >= 3 ? distVec(args[2]) : [str.length];
          const nCellCols = colDist.length;
          const cellData: RuntimeValue[] = [];
          let pos = 0;
          for (let cj = 0; cj < nCellCols; cj++) {
            cellData.push(RTV.char(str.substring(pos, pos + colDist[cj])));
            pos += colDist[cj];
          }
          return RTV.cell(cellData, [1, nCellCols]);
        }

        const A = coerceToTensor(args[0], "mat2cell");
        const rows = A.shape[0];
        const cols = A.shape.length >= 2 ? A.shape[1] : 1;
        const rowDist = distVec(args[1]);
        const colDist = args.length === 2 ? [cols] : distVec(args[2]);

        const rowSum = rowDist.reduce((a, b) => a + b, 0);
        const colSum = colDist.reduce((a, b) => a + b, 0);
        if (rowSum !== rows)
          throw new RuntimeError(
            `mat2cell: row distribution sums to ${rowSum}, but array has ${rows} rows`
          );
        if (colSum !== cols)
          throw new RuntimeError(
            `mat2cell: column distribution sums to ${colSum}, but array has ${cols} columns`
          );

        const nCellRows = rowDist.length;
        const nCellCols = colDist.length;
        const cellData: RuntimeValue[] = new Array(nCellRows * nCellCols);
        const hasImag = A.imag !== undefined;

        let colStart = 0;
        for (let cj = 0; cj < nCellCols; cj++) {
          let rowStart = 0;
          for (let ci = 0; ci < nCellRows; ci++) {
            const subRows = rowDist[ci];
            const subCols = colDist[cj];
            if (subRows === 1 && subCols === 1) {
              const idx = colStart * rows + rowStart;
              if (hasImag && A.imag![idx] !== 0) {
                cellData[cj * nCellRows + ci] = RTV.complex(
                  A.data[idx],
                  A.imag![idx]
                );
              } else {
                cellData[cj * nCellRows + ci] = RTV.num(A.data[idx]);
              }
            } else {
              const subData = new FloatXArray(subRows * subCols);
              const subImag = hasImag
                ? new FloatXArray(subRows * subCols)
                : undefined;
              for (let j = 0; j < subCols; j++) {
                for (let i = 0; i < subRows; i++) {
                  const srcIdx = (colStart + j) * rows + (rowStart + i);
                  const dstIdx = j * subRows + i;
                  subData[dstIdx] = A.data[srcIdx];
                  if (subImag) subImag[dstIdx] = A.imag![srcIdx];
                }
              }
              cellData[cj * nCellRows + ci] = RTV.tensor(
                subData,
                [subRows, subCols],
                subImag
              );
            }
            rowStart += subRows;
          }
          colStart += colDist[cj];
        }
        return RTV.cell(cellData, [nCellRows, nCellCols]);
      },
    };
  },
});

// ── num2cell ────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "num2cell",
  resolve: argTypes => {
    if (argTypes.length < 1 || argTypes.length > 2) return null;
    return {
      outputTypes: [{ kind: "unknown" }],
      apply: args => {
        let A = args[0];
        if (isRuntimeNumber(A)) {
          if (args.length === 1) return RTV.cell([A], [1, 1]);
          A = RTV.tensor(new FloatXArray([A as number]), [1, 1]);
        }
        if (isRuntimeLogical(A)) {
          if (args.length === 1) return RTV.cell([A], [1, 1]);
          A = RTV.tensor(new FloatXArray([A ? 1 : 0]), [1, 1]);
        }
        if (!isRuntimeTensor(A))
          throw new RuntimeError(
            "num2cell: first argument must be a numeric array"
          );

        const shape = A.shape;
        const rows = shape[0];
        const cols = shape.length >= 2 ? shape[1] : 1;

        if (args.length === 1) {
          const totalElems = A.data.length;
          const cellData: RuntimeValue[] = new Array(totalElems);
          for (let i = 0; i < totalElems; i++) cellData[i] = RTV.num(A.data[i]);
          return RTV.cell(cellData, [...shape]);
        }

        const dim = Math.round(toNumber(args[1]));
        if (dim === 1) {
          const cellData: RuntimeValue[] = new Array(cols);
          for (let j = 0; j < cols; j++) {
            const colData = new FloatXArray(rows);
            for (let i = 0; i < rows; i++) colData[i] = A.data[j * rows + i];
            cellData[j] = RTV.tensor(colData, [rows, 1]);
          }
          return RTV.cell(cellData, [1, cols]);
        } else if (dim === 2) {
          const cellData: RuntimeValue[] = new Array(rows);
          for (let i = 0; i < rows; i++) {
            const rowData = new FloatXArray(cols);
            for (let j = 0; j < cols; j++) rowData[j] = A.data[j * rows + i];
            cellData[i] = RTV.tensor(rowData, [1, cols]);
          }
          return RTV.cell(cellData, [rows, 1]);
        }
        throw new RuntimeError(`num2cell: unsupported dimension ${dim}`);
      },
    };
  },
});

// ── cell2struct ─────────────────────────────────────────────────────────

registerIBuiltin({
  name: "cell2struct",
  resolve: argTypes => {
    if (argTypes.length < 2 || argTypes.length > 3) return null;
    return {
      outputTypes: [{ kind: "struct", fields: {} }],
      apply: args => {
        const cellArg = args[0];
        if (!isRuntimeCell(cellArg))
          throw new RuntimeError(
            "cell2struct: first argument must be a cell array"
          );
        const fieldsArg = args[1];
        let fieldNames: string[];
        if (isRuntimeCell(fieldsArg)) {
          fieldNames = fieldsArg.data.map(v => toString(v));
        } else {
          throw new RuntimeError(
            "cell2struct: field names must be a cell array of strings"
          );
        }
        const nFields = fieldNames.length;
        const dim = args.length >= 3 ? toNumber(args[2]) : 1;
        const shape = cellArg.shape;

        if (dim === 1) {
          const nRows = shape[0];
          if (nRows !== nFields)
            throw new RuntimeError(
              "cell2struct: number of cells along dim must match number of field names"
            );
          const nCols = shape.length >= 2 ? shape[1] : 1;
          if (nCols === 1) {
            const fields: Record<string, RuntimeValue> = {};
            for (let i = 0; i < nFields; i++)
              fields[fieldNames[i]] = cellArg.data[i];
            return RTV.struct(fields);
          }
          const elements: RuntimeStruct[] = [];
          for (let col = 0; col < nCols; col++) {
            const fields: Record<string, RuntimeValue> = {};
            for (let row = 0; row < nRows; row++) {
              fields[fieldNames[row]] = cellArg.data[row + col * nRows];
            }
            elements.push(RTV.struct(fields));
          }
          if (elements.length === 1) return elements[0];
          return RTV.structArray(fieldNames, elements);
        } else if (dim === 2) {
          const nRows = shape[0];
          const nCols = shape.length >= 2 ? shape[1] : 1;
          if (nCols !== nFields)
            throw new RuntimeError(
              "cell2struct: number of cells along dim must match number of field names"
            );
          const elements: RuntimeStruct[] = [];
          for (let row = 0; row < nRows; row++) {
            const fields: Record<string, RuntimeValue> = {};
            for (let col = 0; col < nCols; col++) {
              fields[fieldNames[col]] = cellArg.data[row + col * nRows];
            }
            elements.push(RTV.struct(fields));
          }
          if (elements.length === 1) return elements[0];
          return RTV.structArray(fieldNames, elements);
        }
        throw new RuntimeError("cell2struct: dim must be 1 or 2");
      },
    };
  },
});

// ── struct2cell ─────────────────────────────────────────────────────────

registerIBuiltin({
  name: "struct2cell",
  resolve: argTypes => {
    if (argTypes.length !== 1) return null;
    return {
      outputTypes: [{ kind: "unknown" }],
      apply: args => {
        const s = args[0];
        if (!isRuntimeStruct(s))
          throw new RuntimeError("struct2cell: argument must be a struct");
        const keys = Array.from(s.fields.keys());
        const cellData: RuntimeValue[] = keys.map(k => s.fields.get(k)!);
        return RTV.cell(cellData, [keys.length, 1]);
      },
    };
  },
});

// ── namedargs2cell ──────────────────────────────────────────────────────

registerIBuiltin({
  name: "namedargs2cell",
  resolve: argTypes => {
    if (argTypes.length !== 1) return null;
    return {
      outputTypes: [{ kind: "unknown" }],
      apply: args => {
        const v = args[0];
        if (!isRuntimeStruct(v))
          throw new RuntimeError("namedargs2cell: argument must be a struct");
        const names = [...v.fields.keys()];
        const result: RuntimeValue[] = [];
        for (const name of names) {
          result.push(RTV.string(name));
          result.push(v.fields.get(name)!);
        }
        return RTV.cell(result, [1, 2 * names.length]);
      },
    };
  },
});

// ── rmfield ─────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "rmfield",
  resolve: argTypes => {
    if (argTypes.length !== 2) return null;
    return {
      outputTypes: [{ kind: "struct", fields: {} }],
      apply: args => {
        const v = args[0];
        if (isRuntimeStructArray(v)) {
          const name = toString(args[1]);
          if (!v.fieldNames.includes(name))
            throw new RuntimeError(`rmfield: field '${name}' does not exist`);
          const newFieldNames = v.fieldNames.filter(n => n !== name);
          const newElements = v.elements.map(el => {
            const newFields = new Map(el.fields);
            newFields.delete(name);
            return RTV.struct(newFields);
          });
          return RTV.structArray(newFieldNames, newElements);
        }
        if (!isRuntimeStruct(v))
          throw new RuntimeError("rmfield: first argument must be a struct");
        const name = toString(args[1]);
        if (!v.fields.has(name))
          throw new RuntimeError(`rmfield: field '${name}' does not exist`);
        const newFields = new Map(v.fields);
        newFields.delete(name);
        return RTV.struct(newFields);
      },
    };
  },
});
