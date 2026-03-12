/* eslint-disable @typescript-eslint/no-explicit-any */
/**
 * Miscellaneous builtin functions (I/O, type conversion, cell/struct, error handling, timing, etc.)
 */

import { executeCode } from "../executeCode.js";
import {
  RuntimeValue,
  RTV,
  toNumber,
  toBool,
  toString,
  RuntimeError,
} from "../runtime/index.js";
import {
  FloatXArray,
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeTensor,
  isRuntimeChar,
  isRuntimeStruct,
  isRuntimeClassInstance,
  isRuntimeComplexNumber,
  isRuntimeCell,
  isRuntimeFunction,
  isRuntimeStructArray,
  RuntimeStruct,
} from "../runtime/types.js";
import { register, builtinSingle } from "./registry.js";
import { parseShapeArgs } from "./shape-utils.js";
import { sprintfFormat } from "./string.js";
import {
  mAdd,
  mSub,
  mMul,
  mDiv,
  mLeftDiv,
  mPow,
  mElemMul,
  mElemDiv,
  mElemLeftDiv,
  mElemPow,
  mNeg,
  mEqual,
  mNotEqual,
  mLess,
  mLessEqual,
  mGreater,
  mGreaterEqual,
} from "./arithmetic.js";

export function registerMiscFunctions(): void {
  // ── I/O ─────────────────────────────────────────────────────────────────

  // disp is handled specially by the interpreter to capture output
  register(
    "disp",
    builtinSingle(() => {
      // This is a placeholder; the interpreter hooks into this
      return RTV.num(0);
    })
  );

  register(
    "fprintf",
    builtinSingle(() => {
      // Placeholder: interpreter handles output capture
      return RTV.num(0);
    })
  );

  // ── Type conversion ─────────────────────────────────────────────────────

  register(
    "logical",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("logical requires 1 argument");
      const v = args[0];
      if (isRuntimeTensor(v)) {
        const result = new FloatXArray(v.data.length);
        for (let i = 0; i < v.data.length; i++) {
          result[i] = v.data[i] !== 0 ? 1 : 0;
        }
        const t = RTV.tensor(result, v.shape);
        t._isLogical = true;
        return t;
      }
      return RTV.logical(toBool(v));
    })
  );

  // ── Cell/Struct ─────────────────────────────────────────────────────────

  register(
    "cell",
    builtinSingle(args => {
      if (args.length === 0) return RTV.cell([], [0, 0]);
      if (args.length === 1) {
        const arg = args[0];
        // If the argument is a non-scalar tensor (vector), treat elements as dimensions
        if (
          typeof arg === "object" &&
          arg !== null &&
          "kind" in arg &&
          arg.kind === "tensor" &&
          arg.data.length > 1
        ) {
          const dims = Array.from(arg.data).map(d => Math.round(d));
          const total = dims.reduce((a, b) => a * b, 1);
          const data: RuntimeValue[] = new Array(total).fill(
            RTV.tensor(new FloatXArray(0), [0, 0])
          );
          return RTV.cell(data, dims);
        }
        const n = Math.round(toNumber(arg));
        const data: RuntimeValue[] = new Array(n * n).fill(
          RTV.tensor(new FloatXArray(0), [0, 0])
        );
        return RTV.cell(data, [n, n]);
      }
      if (args.length === 2) {
        const r = Math.round(toNumber(args[0]));
        const c = Math.round(toNumber(args[1]));
        const data: RuntimeValue[] = new Array(r * c).fill(
          RTV.tensor(new FloatXArray(0), [0, 0])
        );
        return RTV.cell(data, [r, c]);
      }
      throw new RuntimeError("cell: too many arguments");
    })
  );

  register(
    "struct",
    builtinSingle(args => {
      if (args.length === 0) return RTV.struct(new Map());
      if (args.length % 2 !== 0)
        throw new RuntimeError("struct: requires field-value pairs");
      // Check if any value is a cell array — if so, create a struct array
      const fieldNames: string[] = [];
      const fieldValues: RuntimeValue[] = [];
      let hasCell = false;
      let arrayLen = -1;
      for (let i = 0; i < args.length; i += 2) {
        const name = toString(args[i]);
        const val = args[i + 1];
        fieldNames.push(name);
        fieldValues.push(val);
        if (isRuntimeCell(val)) {
          hasCell = true;
          const len = val.data.length;
          if (arrayLen === -1) arrayLen = len;
          else if (len !== arrayLen)
            throw new RuntimeError(
              "struct: cell array values must have the same length"
            );
        }
      }
      if (!hasCell) {
        const fields = new Map<string, RuntimeValue>();
        for (let i = 0; i < fieldNames.length; i++) {
          fields.set(fieldNames[i], fieldValues[i] as RuntimeValue);
        }
        return RTV.struct(fields);
      }
      // Create struct array: each element gets one value from each cell
      const elements = [];
      for (let k = 0; k < arrayLen; k++) {
        const fields = new Map<string, RuntimeValue>();
        for (let i = 0; i < fieldNames.length; i++) {
          const val = fieldValues[i];
          if (isRuntimeCell(val)) {
            fields.set(fieldNames[i], val.data[k] as RuntimeValue);
          } else {
            // Non-cell values are shared across all elements
            fields.set(fieldNames[i], val as RuntimeValue);
          }
        }
        elements.push(RTV.struct(fields));
      }
      return RTV.structArray(fieldNames, elements);
    })
  );

  const fieldnamesApply = builtinSingle(args => {
    if (args.length !== 1)
      throw new RuntimeError("fieldnames requires 1 argument");
    const v = args[0];
    if (isRuntimeStructArray(v)) {
      const names = v.fieldNames;
      return RTV.cell(
        names.map(n => RTV.string(n)),
        [names.length, 1]
      );
    }
    if (!isRuntimeStruct(v) && !isRuntimeClassInstance(v))
      throw new RuntimeError("fieldnames: argument must be a struct");
    const names = [...v.fields.keys()];
    return RTV.cell(
      names.map(n => RTV.string(n)),
      [names.length, 1]
    );
  });
  register("fieldnames", fieldnamesApply);
  register("fields", fieldnamesApply);

  register(
    "namedargs2cell",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("namedargs2cell requires 1 argument");
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
    }),
    1
  );

  register(
    "isfield",
    builtinSingle(args => {
      if (args.length !== 2)
        throw new RuntimeError("isfield requires 2 arguments");
      const v = args[0];
      if (isRuntimeStructArray(v))
        return RTV.logical(v.fieldNames.includes(toString(args[1])));
      if (!isRuntimeStruct(v) && !isRuntimeClassInstance(v))
        return RTV.logical(false);
      return RTV.logical(v.fields.has(toString(args[1])));
    })
  );

  register(
    "rmfield",
    builtinSingle(args => {
      if (args.length !== 2)
        throw new RuntimeError("rmfield requires 2 arguments");
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
    })
  );

  register(
    "mat2cell",
    builtinSingle(args => {
      if (args.length < 2)
        throw new RuntimeError("mat2cell requires at least 2 arguments");

      let A = args[0];

      // Handle char arrays: split into cell of char substrings
      if (isRuntimeChar(A)) {
        const str = A.value;
        const getDistVector = (v: RuntimeValue): number[] => {
          if (isRuntimeNumber(v)) return [Math.round(v)];
          if (isRuntimeTensor(v))
            return Array.from(v.data).map(x => Math.round(x));
          throw new RuntimeError(
            "mat2cell: distribution arguments must be numeric vectors"
          );
        };
        // rowDist (args[1]) is consumed but not used for 1-row char arrays
        getDistVector(args[1]);
        const colDist =
          args.length >= 3 ? getDistVector(args[2]) : [str.length];
        const nCellCols = colDist.length;
        const cellData: RuntimeValue[] = [];
        let pos = 0;
        for (let cj = 0; cj < nCellCols; cj++) {
          cellData.push(RTV.char(str.substring(pos, pos + colDist[cj])));
          pos += colDist[cj];
        }
        return RTV.cell(cellData, [1, nCellCols]);
      }

      // Convert scalar to 1x1 tensor
      if (isRuntimeNumber(A)) {
        A = RTV.tensor(new FloatXArray([A]), [1, 1]);
      }
      if (!isRuntimeTensor(A))
        throw new RuntimeError(
          "mat2cell: first argument must be a numeric array"
        );

      const rows = A.shape[0];
      const cols = A.shape.length >= 2 ? A.shape[1] : 1;

      // Parse a distribution vector from a runtime value
      const getDistVector = (v: RuntimeValue): number[] => {
        if (isRuntimeNumber(v)) return [Math.round(v)];
        if (isRuntimeTensor(v))
          return Array.from(v.data).map(x => Math.round(x));
        throw new RuntimeError(
          "mat2cell: distribution arguments must be numeric vectors"
        );
      };

      const rowDist = getDistVector(args[1]);

      // If only rowDist provided, split by rows only (n-by-1 cell)
      let colDist: number[];
      if (args.length === 2) {
        colDist = [cols];
      } else {
        colDist = getDistVector(args[2]);
      }

      // Validate distributions sum to array dimensions
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

      // Build cell array (column-major order)
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
            // Single element — return as scalar or complex
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
    })
  );

  register(
    "cell2mat",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("cell2mat requires exactly 1 argument");
      const C = args[0];
      if (!isRuntimeCell(C))
        throw new RuntimeError("cell2mat: argument must be a cell array");

      const cellRows = C.shape[0];
      const cellCols = C.shape.length >= 2 ? C.shape[1] : 1;

      // Helper: get a cell element as a tensor (scalars → 1×1)
      const asTensor = (
        v: RuntimeValue
      ): {
        data: InstanceType<typeof FloatXArray>;
        shape: number[];
        imag?: InstanceType<typeof FloatXArray>;
      } => {
        if (isRuntimeNumber(v))
          return {
            data: new FloatXArray([v]),
            shape: [1, 1],
          };
        if (isRuntimeComplexNumber(v))
          return {
            data: new FloatXArray([v.re]),
            shape: [1, 1],
            imag: new FloatXArray([v.im]),
          };
        if (isRuntimeTensor(v))
          return {
            data: v.data as InstanceType<typeof FloatXArray>,
            shape: v.shape,
            imag: v.imag as InstanceType<typeof FloatXArray> | undefined,
          };
        if (isRuntimeLogical(v))
          return {
            data: new FloatXArray([v ? 1 : 0]),
            shape: [1, 1],
          };
        throw new RuntimeError(
          "cell2mat: cell contents must be numeric arrays"
        );
      };

      // Determine row heights (from first column) and column widths (from first row)
      const rowHeights: number[] = [];
      for (let ci = 0; ci < cellRows; ci++) {
        const t = asTensor(C.data[0 * cellRows + ci]); // column-major: col 0, row ci
        rowHeights.push(t.shape[0]);
      }
      const colWidths: number[] = [];
      for (let cj = 0; cj < cellCols; cj++) {
        const t = asTensor(C.data[cj * cellRows + 0]); // col cj, row 0
        colWidths.push(t.shape.length >= 2 ? t.shape[1] : 1);
      }

      const totalRows = rowHeights.reduce((a, b) => a + b, 0);
      const totalCols = colWidths.reduce((a, b) => a + b, 0);
      const resultData = new FloatXArray(totalRows * totalCols);
      let hasImag = false;

      // Check if any element has imaginary part
      for (let k = 0; k < C.data.length; k++) {
        const t = asTensor(C.data[k]);
        if (t.imag) {
          hasImag = true;
          break;
        }
      }
      const resultImag = hasImag
        ? new FloatXArray(totalRows * totalCols)
        : undefined;

      // Copy blocks into result (column-major layout)
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
              if (resultImag) {
                resultImag[dstIdx] = t.imag ? t.imag[srcIdx] : 0;
              }
            }
          }
          rowOffset += rowHeights[ci];
        }
        colOffset += colWidths[cj];
      }

      return RTV.tensor(resultData, [totalRows, totalCols], resultImag);
    })
  );

  // ── num2cell ──────────────────────────────────────────────────────────

  register(
    "num2cell",
    builtinSingle(args => {
      if (args.length < 1 || args.length > 2)
        throw new RuntimeError("num2cell requires 1 or 2 arguments");
      let A = args[0];

      // Convert scalar to 1x1 tensor
      if (isRuntimeNumber(A)) {
        if (args.length === 1) {
          return RTV.cell([A], [1, 1]);
        }
        A = RTV.tensor(new FloatXArray([A]), [1, 1]);
      }
      if (isRuntimeLogical(A)) {
        if (args.length === 1) {
          return RTV.cell([A], [1, 1]);
        }
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
        // No dim: each element becomes a cell
        const totalElems = A.data.length;
        const cellData: RuntimeValue[] = new Array(totalElems);
        for (let i = 0; i < totalElems; i++) {
          cellData[i] = RTV.num(A.data[i]);
        }
        return RTV.cell(cellData, [...shape]);
      }

      // With dim argument
      const dim = Math.round(toNumber(args[1]));
      if (dim === 1) {
        // Each column becomes a cell element -> 1 x cols cell
        const cellData: RuntimeValue[] = new Array(cols);
        for (let j = 0; j < cols; j++) {
          const colData = new FloatXArray(rows);
          for (let i = 0; i < rows; i++) {
            colData[i] = A.data[j * rows + i];
          }
          cellData[j] = RTV.tensor(colData, [rows, 1]);
        }
        return RTV.cell(cellData, [1, cols]);
      } else if (dim === 2) {
        // Each row becomes a cell element -> rows x 1 cell
        const cellData: RuntimeValue[] = new Array(rows);
        for (let i = 0; i < rows; i++) {
          const rowData = new FloatXArray(cols);
          for (let j = 0; j < cols; j++) {
            rowData[j] = A.data[j * rows + i];
          }
          cellData[i] = RTV.tensor(rowData, [1, cols]);
        }
        return RTV.cell(cellData, [rows, 1]);
      }
      throw new RuntimeError(`num2cell: unsupported dimension ${dim}`);
    })
  );

  // ── cell2struct ──────────────────────────────────────────────────────

  register(
    "cell2struct",
    builtinSingle(args => {
      if (args.length < 2 || args.length > 3)
        throw new RuntimeError("cell2struct requires 2 or 3 arguments");

      const cellArg = args[0];
      if (!isRuntimeCell(cellArg))
        throw new RuntimeError(
          "cell2struct: first argument must be a cell array"
        );

      // Get field names
      const fieldsArg = args[1];
      let fieldNames: string[];
      if (isRuntimeCell(fieldsArg)) {
        fieldNames = fieldsArg.data.map(v => toString(v));
      } else if (isRuntimeTensor(fieldsArg)) {
        // Could be a char array, but unlikely in practice
        throw new RuntimeError(
          "cell2struct: field names must be a cell array of strings"
        );
      } else {
        throw new RuntimeError(
          "cell2struct: field names must be a cell array of strings"
        );
      }

      const nFields = fieldNames.length;
      const dim = args.length >= 3 ? toNumber(args[2]) : 1;
      const shape = cellArg.shape;

      if (dim === 1) {
        // dim=1 (default): field names correspond to rows
        // For a column cell {a; b; c} with shape [3,1], each row is a field
        const nRows = shape[0];
        if (nRows !== nFields)
          throw new RuntimeError(
            "cell2struct: number of cells along dim must match number of field names"
          );
        const nCols = shape.length >= 2 ? shape[1] : 1;
        if (nCols === 1) {
          // Simple case: column cell -> single struct
          const fields: Record<string, RuntimeValue> = {};
          for (let i = 0; i < nFields; i++) {
            fields[fieldNames[i]] = cellArg.data[i];
          }
          return RTV.struct(fields);
        }
        // Multiple columns: each column becomes a struct element
        const elements: RuntimeStruct[] = [];
        for (let col = 0; col < nCols; col++) {
          const fields: Record<string, RuntimeValue> = {};
          for (let row = 0; row < nRows; row++) {
            // column-major: index = row + col * nRows
            fields[fieldNames[row]] = cellArg.data[row + col * nRows];
          }
          elements.push(RTV.struct(fields));
        }
        if (elements.length === 1) return elements[0];
        return RTV.structArray(fieldNames, elements);
      } else if (dim === 2) {
        // dim=2: field names correspond to columns
        // Each row becomes a struct element
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
            // column-major: index = row + col * nRows
            fields[fieldNames[col]] = cellArg.data[row + col * nRows];
          }
          elements.push(RTV.struct(fields));
        }
        if (elements.length === 1) return elements[0];
        return RTV.structArray(fieldNames, elements);
      } else {
        throw new RuntimeError("cell2struct: dim must be 1 or 2");
      }
    })
  );

  // ── struct2cell ─────────────────────────────────────────────────────

  register(
    "struct2cell",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("struct2cell requires exactly 1 argument");

      const s = args[0];
      if (!isRuntimeStruct(s))
        throw new RuntimeError("struct2cell: argument must be a struct");

      const keys = Array.from(s.fields.keys());
      const cellData: RuntimeValue[] = keys.map(k => s.fields.get(k)!);
      return RTV.cell(cellData, [keys.length, 1]);
    })
  );

  // ── Error handling ──────────────────────────────────────────────────────

  register(
    "error",
    builtinSingle(args => {
      if (args.length === 0)
        throw new RuntimeError("error called with no arguments");
      const first = toString(args[0]);
      // If first arg contains ':' and there are more args, treat as error(id, fmt, ...)
      if (args.length >= 2 && first.includes(":")) {
        const identifier = first;
        const msg =
          args.length === 2
            ? toString(args[1])
            : sprintfFormat(toString(args[1]), args.slice(2));
        const err = new RuntimeError(msg);
        err.identifier = identifier;
        throw err;
      }
      const msg =
        args.length === 1 ? first : sprintfFormat(first, args.slice(1));
      throw new RuntimeError(msg);
    })
  );

  register(
    "warning",
    builtinSingle(() => {
      // Silently ignore warnings for now
      return RTV.num(0);
    })
  );

  register(
    "assert",
    builtinSingle(args => {
      if (args.length < 1)
        throw new RuntimeError("assert requires at least 1 argument");
      if (!toBool(args[0])) {
        const msg = args.length > 1 ? toString(args[1]) : "Assertion failed";
        throw new RuntimeError(msg);
      }
      return RTV.num(0);
    })
  );

  // ── Timing ──────────────────────────────────────────────────────────────

  let ticTime = 0;
  register(
    "tic",
    builtinSingle(
      () => {
        ticTime = performance.now();
        return RTV.num(ticTime / 1000);
      },
      { outputType: { kind: "Number" } }
    )
  );

  register(
    "toc",
    builtinSingle(
      () => {
        const elapsed = (performance.now() - ticTime) / 1000;
        return RTV.num(elapsed);
      },
      { outputType: { kind: "Number" } }
    )
  );

  // ── Misc ────────────────────────────────────────────────────────────────

  register(
    "true",
    builtinSingle(args => {
      if (args.length === 0) return RTV.logical(true);
      const shape = parseShapeArgs(args);
      const rows = shape[0];
      const cols = shape[1] ?? rows;
      const t = RTV.tensor(new FloatXArray(rows * cols).fill(1), [rows, cols]);
      t._isLogical = true;
      return t;
    })
  );
  register(
    "false",
    builtinSingle(args => {
      if (args.length === 0) return RTV.logical(false);
      const shape = parseShapeArgs(args);
      const rows = shape[0];
      const cols = shape[1] ?? rows;
      const t = RTV.tensor(new FloatXArray(rows * cols), [rows, cols]);
      t._isLogical = true;
      return t;
    })
  );

  register(
    "clear",
    builtinSingle(() => {
      // No-op: variable clearing is not meaningful in compiled mode
      return RTV.num(0);
    })
  );

  register(
    "clc",
    builtinSingle(() => {
      // No-op: console clearing is not meaningful in compiled mode
      return RTV.num(0);
    })
  );

  register(
    "clf",
    builtinSingle(() => {
      // No-op: figure clearing is not meaningful in this context
      return RTV.num(0);
    })
  );

  register(
    "exist",
    builtinSingle(args => {
      if (
        args.length >= 2 &&
        isRuntimeChar(args[1]) &&
        args[1].value === "var"
      ) {
        throw new RuntimeError(
          "exist('name', 'var') requires a literal variable name (dynamic names not supported)"
        );
      }
      return RTV.num(0);
    })
  );

  // nargin: when called with no args inside a function, codegen emits `arguments.length` directly.
  // When called with a function handle arg, return the stored nargin.
  register(
    "nargin",
    builtinSingle(
      args => {
        if (args.length === 1 && isRuntimeFunction(args[0])) {
          const handle = args[0];
          if (handle.nargin === undefined && handle.impl === "builtin") {
            throw new RuntimeError(
              `nargin not available for builtin function '${handle.name}' (not yet registered)`
            );
          }
          return RTV.num(handle.nargin ?? 0);
        }
        return RTV.num(0);
      },
      { outputType: { kind: "Number" } }
    )
  );
  register(
    "nargout",
    builtinSingle(() => RTV.num(0), { outputType: { kind: "Number" } })
  );

  register(
    "eval",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("eval requires exactly 1 argument");
      const code = toString(args[0]);
      const result = executeCode(code, {
        onOutput: (text: string) => {
          console.log(text); // For now, just log eval output to console
        },
        displayResults: false, // Don't display results of eval statements
      });
      return result.returnValue;
    })
  );

  // ── Internal testing utilities ──────────────────────────────────────────

  // __inferred_type_str: Internal-only function for testing type inference.
  // Returns a string representation of the inferred type of its argument.
  // The actual implementation is handled specially during code generation
  // (see genFuncCall in codegen/genExprs.ts), which resolves the type at
  // compile-time and generates a string literal.
  // arrayfun / cellfun — actual implementation is in Runtime.call() where raw
  // JS function args are still accessible.  These registrations let the
  // codegen know the functions exist.
  register(
    "feval",
    builtinSingle(() => {
      throw new RuntimeError("feval: should be handled by runtime");
    })
  );

  register(
    "deal",
    builtinSingle((args, nargout) => {
      if (args.length === 0)
        throw new RuntimeError("deal requires at least 1 argument");
      if (args.length === 1) {
        // [B1,...,Bn] = deal(A) — replicate single input to all outputs
        if (nargout <= 1) return args[0];
        return Array(nargout).fill(args[0]);
      }
      // [B1,...,Bn] = deal(A1,...,An) — number of inputs must match nargout
      if (args.length !== nargout)
        throw new RuntimeError(
          `deal: number of inputs (${args.length}) must match number of outputs (${nargout})`
        );
      if (nargout <= 1) return args[0];
      return args;
    })
  );

  register(
    "func2str",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("func2str requires 1 argument");
      const v = args[0];
      if (!isRuntimeFunction(v))
        throw new RuntimeError("func2str: argument must be a function handle");
      return RTV.string(v.name);
    })
  );

  register(
    "arrayfun",
    builtinSingle(() => {
      throw new RuntimeError("arrayfun: should be handled by runtime");
    })
  );

  register(
    "cellfun",
    builtinSingle(() => {
      throw new RuntimeError("cellfun: should be handled by runtime");
    })
  );

  register(
    "structfun",
    builtinSingle(() => {
      throw new RuntimeError("structfun: should be handled by runtime");
    })
  );

  register(
    "bsxfun",
    builtinSingle(() => {
      throw new RuntimeError("bsxfun: should be handled by runtime");
    })
  );

  // Operator-name builtins — allow @plus, @minus, etc. as function handles
  const opMap: [string, (a: RuntimeValue, b: RuntimeValue) => RuntimeValue][] =
    [
      ["plus", mAdd],
      ["minus", mSub],
      ["times", mElemMul],
      ["rdivide", mElemDiv],
      ["power", mElemPow],
      ["mtimes", mMul],
      ["mrdivide", mDiv],
      ["mldivide", mLeftDiv],
      ["mpower", mPow],
      ["ldivide", mElemLeftDiv],
      ["eq", mEqual],
      ["ne", mNotEqual],
      ["lt", mLess],
      ["le", mLessEqual],
      ["gt", mGreater],
      ["ge", mGreaterEqual],
    ];
  for (const [name, op] of opMap) {
    register(
      name,
      builtinSingle(args => {
        if (args.length !== 2)
          throw new RuntimeError(`${name} requires exactly 2 arguments`);
        return op(args[0], args[1]);
      }),
      2
    );
  }

  // Unary operator-name builtins
  register(
    "uminus",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("uminus requires exactly 1 argument");
      return mNeg(args[0]);
    }),
    1
  );

  register(
    "uplus",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("uplus requires exactly 1 argument");
      return args[0];
    })
  );

  // narginchk / nargoutchk — these are handled specially by codegen which
  // passes the actual argument count. The builtin registration just lets
  // the compiler know they exist.
  register(
    "narginchk",
    builtinSingle(() => {
      // Handled by codegen/runtime
      return RTV.num(0);
    })
  );

  register(
    "nargoutchk",
    builtinSingle(() => {
      // Handled by codegen/runtime
      return RTV.num(0);
    })
  );

  // This runtime implementation is just a placeholder that should never be called.
  register(
    "__inferred_type_str",
    builtinSingle(
      () => {
        throw new RuntimeError(
          "__inferred_type_str should be resolved at compile-time"
        );
      },
      { outputType: { kind: "String" } }
    )
  );

  // cart2sph: Transform Cartesian coordinates to spherical
  register(
    "cart2sph",
    builtinSingle((args, nargout) => {
      if (args.length !== 3)
        throw new RuntimeError("cart2sph requires 3 arguments");
      const x = args[0];
      const y = args[1];
      const z = args[2];
      const xIsT = isRuntimeTensor(x);
      const yIsT = isRuntimeTensor(y);
      const zIsT = isRuntimeTensor(z);
      if (xIsT || yIsT || zIsT) {
        // Tensor case: element-wise
        const xd = xIsT ? x.data : null;
        const yd = yIsT ? y.data : null;
        const zd = zIsT ? z.data : null;
        const shape = xIsT ? x.shape : yIsT ? y.shape : (z as any).shape;
        const len = xIsT ? xd!.length : yIsT ? yd!.length : zd!.length;
        const azData = new FloatXArray(len);
        const elData = new FloatXArray(len);
        const rData = new FloatXArray(len);
        for (let i = 0; i < len; i++) {
          const xi = xd ? xd[i] : toNumber(x);
          const yi = yd ? yd[i] : toNumber(y);
          const zi = zd ? zd[i] : toNumber(z);
          const hypotxy = Math.sqrt(xi * xi + yi * yi);
          azData[i] = Math.atan2(yi, xi);
          elData[i] = Math.atan2(zi, hypotxy);
          rData[i] = Math.sqrt(xi * xi + yi * yi + zi * zi);
        }
        if (nargout <= 1) return RTV.tensor(azData, shape);
        return [
          RTV.tensor(azData, shape),
          RTV.tensor(elData, shape),
          RTV.tensor(rData, shape),
        ];
      }
      // Scalar case
      const xv = toNumber(x);
      const yv = toNumber(y);
      const zv = toNumber(z);
      const hypotxy = Math.sqrt(xv * xv + yv * yv);
      const az = Math.atan2(yv, xv);
      const el = Math.atan2(zv, hypotxy);
      const r = Math.sqrt(xv * xv + yv * yv + zv * zv);
      if (nargout <= 1) return RTV.num(az);
      return [RTV.num(az), RTV.num(el), RTV.num(r)];
    }),
    3
  );

  // sph2cart: Transform spherical coordinates to Cartesian
  register(
    "sph2cart",
    builtinSingle((args, nargout) => {
      if (args.length !== 3)
        throw new RuntimeError("sph2cart requires 3 arguments");
      const az = args[0];
      const el = args[1];
      const r = args[2];
      const azIsT = isRuntimeTensor(az);
      const elIsT = isRuntimeTensor(el);
      const rIsT = isRuntimeTensor(r);
      if (azIsT || elIsT || rIsT) {
        const azd = azIsT ? az.data : null;
        const eld = elIsT ? el.data : null;
        const rd = rIsT ? r.data : null;
        const shape = azIsT ? az.shape : elIsT ? el.shape : (r as any).shape;
        const len = azIsT ? azd!.length : elIsT ? eld!.length : rd!.length;
        const xData = new FloatXArray(len);
        const yData = new FloatXArray(len);
        const zData = new FloatXArray(len);
        for (let i = 0; i < len; i++) {
          const a = azd ? azd[i] : toNumber(az);
          const e = eld ? eld[i] : toNumber(el);
          const rv = rd ? rd[i] : toNumber(r);
          const rcosel = rv * Math.cos(e);
          xData[i] = rcosel * Math.cos(a);
          yData[i] = rcosel * Math.sin(a);
          zData[i] = rv * Math.sin(e);
        }
        if (nargout <= 1) return RTV.tensor(xData, shape);
        return [
          RTV.tensor(xData, shape),
          RTV.tensor(yData, shape),
          RTV.tensor(zData, shape),
        ];
      }
      const av = toNumber(az);
      const ev = toNumber(el);
      const rv = toNumber(r);
      const rcosel = rv * Math.cos(ev);
      const xv = rcosel * Math.cos(av);
      const yv = rcosel * Math.sin(av);
      const zv = rv * Math.sin(ev);
      if (nargout <= 1) return RTV.num(xv);
      return [RTV.num(xv), RTV.num(yv), RTV.num(zv)];
    }),
    3
  );

  // cart2pol: Transform Cartesian coordinates to polar/cylindrical
  register(
    "cart2pol",
    builtinSingle((args, nargout) => {
      if (args.length < 2 || args.length > 3)
        throw new RuntimeError("cart2pol requires 2 or 3 arguments");
      const x = args[0];
      const y = args[1];
      const hasZ = args.length === 3;
      const z = hasZ ? args[2] : undefined;
      const xIsT = isRuntimeTensor(x);
      const yIsT = isRuntimeTensor(y);
      const zIsT = z !== undefined && isRuntimeTensor(z);
      if (xIsT || yIsT || zIsT) {
        const xd = xIsT ? x.data : null;
        const yd = yIsT ? y.data : null;
        const zd = zIsT ? (z as any).data : null;
        const shape = xIsT ? x.shape : yIsT ? y.shape : (z as any).shape;
        const len = xIsT ? xd!.length : yIsT ? yd!.length : zd!.length;
        const thData = new FloatXArray(len);
        const rhoData = new FloatXArray(len);
        for (let i = 0; i < len; i++) {
          const xi = xd ? xd[i] : toNumber(x);
          const yi = yd ? yd[i] : toNumber(y);
          thData[i] = Math.atan2(yi, xi);
          rhoData[i] = Math.sqrt(xi * xi + yi * yi);
        }
        if (!hasZ) {
          if (nargout <= 1) return RTV.tensor(thData, shape);
          return [RTV.tensor(thData, shape), RTV.tensor(rhoData, shape)];
        }
        const zOutData = new FloatXArray(len);
        for (let i = 0; i < len; i++) {
          zOutData[i] = zd ? zd[i] : toNumber(z!);
        }
        if (nargout <= 1) return RTV.tensor(thData, shape);
        if (nargout === 2)
          return [RTV.tensor(thData, shape), RTV.tensor(rhoData, shape)];
        return [
          RTV.tensor(thData, shape),
          RTV.tensor(rhoData, shape),
          RTV.tensor(zOutData, shape),
        ];
      }
      const xv = toNumber(x);
      const yv = toNumber(y);
      const th = Math.atan2(yv, xv);
      const rho = Math.sqrt(xv * xv + yv * yv);
      if (!hasZ) {
        if (nargout <= 1) return RTV.num(th);
        return [RTV.num(th), RTV.num(rho)];
      }
      const zv = toNumber(z!);
      if (nargout <= 1) return RTV.num(th);
      if (nargout === 2) return [RTV.num(th), RTV.num(rho)];
      return [RTV.num(th), RTV.num(rho), RTV.num(zv)];
    })
  );

  // pol2cart: Transform polar/cylindrical coordinates to Cartesian
  register(
    "pol2cart",
    builtinSingle((args, nargout) => {
      if (args.length < 2 || args.length > 3)
        throw new RuntimeError("pol2cart requires 2 or 3 arguments");
      const theta = args[0];
      const rho = args[1];
      const hasZ = args.length === 3;
      const z = hasZ ? args[2] : undefined;
      const thIsT = isRuntimeTensor(theta);
      const rhoIsT = isRuntimeTensor(rho);
      const zIsT = z !== undefined && isRuntimeTensor(z);
      if (thIsT || rhoIsT || zIsT) {
        const thd = thIsT ? theta.data : null;
        const rhod = rhoIsT ? rho.data : null;
        const zd = zIsT ? (z as any).data : null;
        const shape = thIsT
          ? theta.shape
          : rhoIsT
            ? rho.shape
            : (z as any).shape;
        const len = thIsT ? thd!.length : rhoIsT ? rhod!.length : zd!.length;
        const xData = new FloatXArray(len);
        const yData = new FloatXArray(len);
        for (let i = 0; i < len; i++) {
          const t = thd ? thd[i] : toNumber(theta);
          const rv = rhod ? rhod[i] : toNumber(rho);
          xData[i] = rv * Math.cos(t);
          yData[i] = rv * Math.sin(t);
        }
        if (!hasZ) {
          if (nargout <= 1) return RTV.tensor(xData, shape);
          return [RTV.tensor(xData, shape), RTV.tensor(yData, shape)];
        }
        const zOutData = new FloatXArray(len);
        for (let i = 0; i < len; i++) {
          zOutData[i] = zd ? zd[i] : toNumber(z!);
        }
        if (nargout <= 1) return RTV.tensor(xData, shape);
        if (nargout === 2)
          return [RTV.tensor(xData, shape), RTV.tensor(yData, shape)];
        return [
          RTV.tensor(xData, shape),
          RTV.tensor(yData, shape),
          RTV.tensor(zOutData, shape),
        ];
      }
      const tv = toNumber(theta);
      const rv = toNumber(rho);
      const xv = rv * Math.cos(tv);
      const yv = rv * Math.sin(tv);
      if (!hasZ) {
        if (nargout <= 1) return RTV.num(xv);
        return [RTV.num(xv), RTV.num(yv)];
      }
      const zv = toNumber(z!);
      if (nargout <= 1) return RTV.num(xv);
      if (nargout === 2) return [RTV.num(xv), RTV.num(yv)];
      return [RTV.num(xv), RTV.num(yv), RTV.num(zv)];
    })
  );
}
