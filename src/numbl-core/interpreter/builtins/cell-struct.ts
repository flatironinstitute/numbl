/**
 * Cell/struct operation builtins: cell2mat, cell2struct, mat2cell, num2cell,
 * struct2cell, namedargs2cell, rmfield.
 */

import {
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
import { allocFloat64Array } from "../../runtime/alloc.js";

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
    return RTV.tensor(allocFloat64Array([v as number]), [1, 1]);
  if (isRuntimeLogical(v))
    return RTV.tensor(allocFloat64Array([v ? 1 : 0]), [1, 1]);
  if (isRuntimeComplexNumber(v))
    return RTV.tensor(
      allocFloat64Array([v.re]),
      [1, 1],
      allocFloat64Array([v.im])
    );
  throw new RuntimeError(`${name}: argument must be numeric`);
}

// ── cell2mat ────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "cell2mat",
  resolve: argTypes => {
    if (argTypes.length !== 1) return null;
    return {
      outputTypes: [{ kind: "unknown" }],
      apply: args => {
        const C = args[0];
        if (!isRuntimeCell(C))
          throw new RuntimeError("cell2mat: argument must be a cell array");

        const cellRows = C.shape[0];
        const cellCols = C.shape.length >= 2 ? C.shape[1] : 1;

        // Char contents: block-concatenate into a char array
        if (C.data.length > 0 && C.data.some(v => isRuntimeChar(v))) {
          const asCharBlock = (v: RuntimeValue) => {
            if (!isRuntimeChar(v))
              throw new RuntimeError(
                "cell2mat: cannot mix char and non-char cell contents"
              );
            const rows = v.shape ? v.shape[0] : v.value.length === 0 ? 0 : 1;
            const cols = v.shape ? v.shape[1] : v.value.length;
            const rowStrings: string[] = [];
            for (let r = 0; r < rows; r++)
              rowStrings.push(v.value.slice(r * cols, (r + 1) * cols));
            return { rows, cols, rowStrings };
          };
          const outRows: string[] = [];
          for (let ci = 0; ci < cellRows; ci++) {
            const blockHeight = asCharBlock(C.data[ci]).rows;
            const lines: string[] = new Array(blockHeight).fill("");
            for (let cj = 0; cj < cellCols; cj++) {
              const b = asCharBlock(C.data[cj * cellRows + ci]);
              if (b.rows !== blockHeight)
                throw new RuntimeError(
                  "cell2mat: inconsistent cell content sizes"
                );
              for (let r = 0; r < blockHeight; r++) lines[r] += b.rowStrings[r];
            }
            outRows.push(...lines);
          }
          const width = outRows.length > 0 ? outRows[0].length : 0;
          if (outRows.some(r => r.length !== width))
            throw new RuntimeError("cell2mat: inconsistent cell content sizes");
          if (outRows.length <= 1) return RTV.char(outRows[0] ?? "");
          const rc = RTV.char(outRows.join(""));
          rc.shape = [outRows.length, width];
          return rc;
        }

        const asTensor = (v: RuntimeValue) => {
          if (isRuntimeNumber(v))
            return {
              data: allocFloat64Array([v as number]),
              shape: [1, 1],
              imag: undefined,
            };
          if (isRuntimeComplexNumber(v))
            return {
              data: allocFloat64Array([v.re]),
              shape: [1, 1],
              imag: allocFloat64Array([v.im]),
            };
          if (isRuntimeTensor(v))
            return { data: v.data, shape: v.shape, imag: v.imag };
          if (isRuntimeLogical(v))
            return {
              data: allocFloat64Array([v ? 1 : 0]),
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
        const resultData = allocFloat64Array(totalRows * totalCols);
        let hasImag = false;
        for (let k = 0; k < C.data.length; k++) {
          if (asTensor(C.data[k]).imag) {
            hasImag = true;
            break;
          }
        }
        const resultImag = hasImag
          ? allocFloat64Array(totalRows * totalCols)
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
              const subData = allocFloat64Array(subRows * subCols);
              const subImag = hasImag
                ? allocFloat64Array(subRows * subCols)
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
          A = RTV.tensor(allocFloat64Array([A as number]), [1, 1]);
        }
        if (isRuntimeLogical(A)) {
          if (args.length === 1) return RTV.cell([A], [1, 1]);
          A = RTV.tensor(allocFloat64Array([A ? 1 : 0]), [1, 1]);
        }
        // MATLAB num2cell wraps each element of ANY array — including struct
        // arrays — in a 1x1 cell. A scalar struct -> a 1x1 cell; a struct array
        // (modelled as 1xN in numbl, matching size()) -> a 1xN cell of scalar
        // structs. RTV.cell increfs its contents, so sharing the existing
        // struct elements is safe.
        if (isRuntimeStruct(A)) {
          return RTV.cell([A], [1, 1]);
        }
        if (isRuntimeStructArray(A)) {
          const elems = A.elements as RuntimeValue[];
          return RTV.cell([...elems], [1, elems.length]);
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
            const colData = allocFloat64Array(rows);
            for (let i = 0; i < rows; i++) colData[i] = A.data[j * rows + i];
            cellData[j] = RTV.tensor(colData, [rows, 1]);
          }
          return RTV.cell(cellData, [1, cols]);
        } else if (dim === 2) {
          const cellData: RuntimeValue[] = new Array(rows);
          for (let i = 0; i < rows; i++) {
            const rowData = allocFloat64Array(cols);
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

// ── getfield / setfield ─────────────────────────────────────────────────

registerIBuiltin({
  name: "getfield",
  resolve: argTypes => {
    if (argTypes.length < 2) return null;
    return {
      outputTypes: [{ kind: "unknown" }],
      apply: args => {
        // getfield(S, field1, ..., fieldN): walk a chain of field names.
        // Cell-array {idx} subscripts are not supported.
        let v = args[0];
        for (let i = 1; i < args.length; i++) {
          if (isRuntimeCell(args[i]))
            throw new RuntimeError(
              "getfield: index ({}) subscripts are not supported"
            );
          if (!isRuntimeStruct(v))
            throw new RuntimeError("getfield: argument must be a structure");
          const name = toString(args[i]);
          if (!v.fields.has(name))
            throw new RuntimeError(
              `Reference to non-existent field '${name}'.`
            );
          v = v.fields.get(name)!;
        }
        return v;
      },
    };
  },
});

registerIBuiltin({
  name: "setfield",
  resolve: argTypes => {
    if (argTypes.length < 3) return null;
    return {
      outputTypes: [{ kind: "struct", fields: {} }],
      apply: args => {
        // setfield(S, field1, ..., fieldN, value): set a (possibly nested)
        // field, returning a new struct. Cell-array {idx} subscripts are
        // not supported.
        const value = args[args.length - 1];
        const fieldArgs = args.slice(1, args.length - 1);
        const setChain = (s: RuntimeValue, depth: number): RuntimeValue => {
          const fa = fieldArgs[depth];
          if (isRuntimeCell(fa))
            throw new RuntimeError(
              "setfield: index ({}) subscripts are not supported"
            );
          const name = toString(fa);
          // An empty [] target becomes a fresh struct (matches MATLAB).
          const base = isRuntimeStruct(s)
            ? s.fields
            : new Map<string, RuntimeValue>();
          if (
            !isRuntimeStruct(s) &&
            !(isRuntimeTensor(s) && s.data.length === 0)
          )
            throw new RuntimeError("setfield: argument must be a structure");
          const newFields = new Map(base);
          if (depth === fieldArgs.length - 1) {
            newFields.set(name, value);
          } else {
            const child = newFields.get(name);
            newFields.set(
              name,
              setChain(child ?? RTV.struct(new Map()), depth + 1)
            );
          }
          return RTV.struct(newFields);
        };
        return setChain(args[0], 0);
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

/** Field-name argument to rmfield: a char/string or a cell array of them. */
function rmfieldNames(arg: RuntimeValue): string[] {
  if (isRuntimeCell(arg)) return arg.data.map(toString);
  return [toString(arg)];
}

registerIBuiltin({
  name: "rmfield",
  resolve: argTypes => {
    if (argTypes.length !== 2) return null;
    return {
      outputTypes: [{ kind: "struct", fields: {} }],
      apply: args => {
        const v = args[0];
        const names = rmfieldNames(args[1]);
        if (isRuntimeStructArray(v)) {
          for (const name of names)
            if (!v.fieldNames.includes(name))
              throw new RuntimeError(`rmfield: field '${name}' does not exist`);
          const newFieldNames = v.fieldNames.filter(n => !names.includes(n));
          const newElements = v.elements.map(el => {
            const newFields = new Map(el.fields);
            for (const name of names) newFields.delete(name);
            return RTV.struct(newFields);
          });
          return RTV.structArray(newFieldNames, newElements);
        }
        if (!isRuntimeStruct(v))
          throw new RuntimeError("rmfield: first argument must be a struct");
        for (const name of names)
          if (!v.fields.has(name))
            throw new RuntimeError(`rmfield: field '${name}' does not exist`);
        const newFields = new Map(v.fields);
        for (const name of names) newFields.delete(name);
        return RTV.struct(newFields);
      },
    };
  },
});
