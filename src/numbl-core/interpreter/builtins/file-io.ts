/**
 * File I/O builtins on the runtime's FileTable (fopen, fclose, fgetl,
 * fscanf, fread, fwrite, fseek, ...), including the binary precision
 * codec used by fread/fwrite.
 */

import {
  type RuntimeValue,
  RTV,
  toString,
  toNumber,
  RuntimeError,
} from "../../runtime/index.js";
import {
  isRuntimeTensor,
  isRuntimeChar,
  isRuntimeString,
  isRuntimeNumber,
  isRuntimeLogical,
  RuntimeChar,
} from "../../runtime/types.js";
import type { Runtime } from "../../runtime/runtime.js";
import type { FileTable } from "../../fileTable.js";
import { allocFloat64Array } from "../../runtime/alloc.js";
import { scanFormat } from "../../helpers/scanFormat.js";
import {
  DEFAULT_TEXT_ENCODING,
  lookupTextEncoding,
} from "../../helpers/textCodec.js";
import { builtin } from "./builtin.js";

/** The runtime's open-file table, or an error when file I/O is not
 *  available. */
function requireFiles(rt: Runtime): FileTable {
  const files = rt.files;
  if (!files)
    throw new RuntimeError("File I/O is not available in this environment");
  return files;
}

const isTextArg = (v: RuntimeValue) => isRuntimeChar(v) || isRuntimeString(v);

/** Open fids as fopen('all') / openedFiles return them: a row vector, or
 *  0x0 when none are open. */
function openFidsValue(files: FileTable): RuntimeValue {
  const fids = files.openFids();
  return fids.length === 0
    ? RTV.tensor(allocFloat64Array(0), [0, 0])
    : RTV.tensor(allocFloat64Array(fids), [1, fids.length]);
}

builtin({
  name: "fopen",
  nargin: [1, 4],
  nargout: 4,
  apply: (args, nargout, ctx) => {
    const files = requireFiles(ctx.rt);
    const first = args[0];
    if (!isTextArg(first)) {
      // [filename, permission, machinefmt, encoding] = fopen(fid)
      const info = files.info(toNumber(first));
      const out = info
        ? [
            RTV.char(info.name),
            RTV.char(info.permission),
            RTV.char(info.machinefmt),
            RTV.char(info.encoding),
          ]
        : [RTV.char(""), RTV.char(""), RTV.char(""), RTV.char("")];
      return nargout <= 1 ? out[0] : out.slice(0, nargout);
    }
    const filename = toString(first);
    if (filename === "all" && args.length === 1) return openFidsValue(files);
    const opt = (i: number) =>
      args.length > i ? toString(args[i]) : undefined;
    const { fid, message } = files.open(filename, opt(1), opt(2), opt(3));
    return nargout <= 1 ? RTV.num(fid) : [RTV.num(fid), RTV.char(message)];
  },
});

builtin({
  name: "openedFiles",
  nargin: 0,
  apply: (_args, _nargout, ctx) => openFidsValue(requireFiles(ctx.rt)),
});

builtin({
  name: "fclose",
  nargin: 1,
  apply: (args, _nargout, ctx) => {
    const files = requireFiles(ctx.rt);
    const arg = args[0];
    if (isTextArg(arg)) {
      if (toString(arg) === "all") return RTV.num(files.closeAll());
      throw new RuntimeError("fclose: invalid argument");
    }
    return RTV.num(files.close(toNumber(arg)));
  },
});

// fgetl is an M-file in MATLAB; fgets is built in and takes an optional
// character limit.
for (const [name, keepNewline] of [
  ["fgetl", false],
  ["fgets", true],
] as const) {
  builtin({
    name,
    nargin: keepNewline ? [1, 2] : 1,
    mfile: !keepNewline,
    apply: (args, _nargout, ctx) => {
      const files = requireFiles(ctx.rt);
      const maxChars = args.length > 1 ? toNumber(args[1]) : undefined;
      const result = files.readLine(toNumber(args[0]), keepNewline, maxChars);
      return result === -1 ? RTV.num(-1) : RTV.char(result);
    },
  });
}

builtin({
  name: "fscanf",
  nargin: [2, 3],
  nargout: 2,
  apply: (args, nargout, ctx) => {
    const files = requireFiles(ctx.rt);
    const fid = toNumber(args[0]);
    const fmt = toString(args[1]);

    // Optional sizeA: scalar count (possibly Inf) or [m, n] (n possibly Inf).
    let maxCount = Infinity;
    let rows: number | null = null;
    if (args.length >= 3) {
      const sz = args[2];
      if (isRuntimeTensor(sz) && sz.data.length >= 2) {
        rows = sz.data[0];
        maxCount = sz.data[0] * sz.data[1];
      } else {
        maxCount = toNumber(sz);
      }
    }

    // Scan the rest of the file, then move past what the scan consumed.
    const rest = files.scanText(fid);
    const { results, consumed } = scanFormat(rest.text, fmt, maxCount);
    rest.consume(consumed);

    const n = results.length;
    let out: RuntimeValue;
    if (rows !== null && Number.isFinite(rows) && rows > 0) {
      // Column-major fill with `rows` rows; a partial final column is
      // zero-padded.
      const cols = Math.ceil(n / rows);
      const data = allocFloat64Array(rows * cols);
      for (let i = 0; i < n; i++) data[i] = results[i];
      out = RTV.tensor(data, [rows, cols]);
    } else if (n === 1) {
      out = RTV.num(results[0]);
    } else {
      out = RTV.tensor(allocFloat64Array(results), [n, 1]);
    }
    if (nargout >= 2) return [out, RTV.num(n)];
    return out;
  },
});

builtin({
  name: "fileread",
  // fileread(filename, Encoding=enc)
  nargin: [1, 3],
  mfile: true,
  apply: (args, _nargout, ctx) => {
    const files = requireFiles(ctx.rt);
    const filename = toString(args[0]);
    let encoding = DEFAULT_TEXT_ENCODING;
    for (let i = 1; i + 1 < args.length; i += 2) {
      if (toString(args[i]).toLowerCase() !== "encoding") continue;
      const name = toString(args[i + 1]);
      const found = lookupTextEncoding(name);
      if (!found)
        throw new RuntimeError(`fileread: unsupported encoding '${name}'`);
      encoding = found;
    }
    let bytes: Uint8Array;
    try {
      bytes = files.readFileBytes(filename);
    } catch {
      const err = new RuntimeError(
        `Could not open file ${filename}. No such file or directory.`
      );
      err.identifier = "MATLAB:fileread:cannotOpenFile";
      throw err;
    }
    return RTV.char(encoding.decode(bytes, 0, bytes.length).text);
  },
});

builtin({
  name: "feof",
  nargin: 1,
  apply: (args, _nargout, ctx) =>
    RTV.num(requireFiles(ctx.rt).atEof(toNumber(args[0])) ? 1 : 0),
});

builtin({
  name: "ferror",
  nargin: [1, 2],
  nargout: 2,
  apply: (args, nargout, ctx) => {
    const files = requireFiles(ctx.rt);
    const clear =
      args.length >= 2 && toString(args[1]).toLowerCase() === "clear";
    const { message, errnum } = files.error(toNumber(args[0]), clear);
    return nargout <= 1
      ? RTV.char(message)
      : [RTV.char(message), RTV.num(errnum)];
  },
});

// ── Binary fread / fwrite / frewind / fseek / ftell ──────────────

/** Parse a MATLAB precision string into source/output type info. */
function parsePrecision(prec: string): {
  repeat: number;
  sourceType: string;
  outputType: string;
  sourceBytes: number;
} {
  let repeat = 0;
  let rest = prec;
  // Handle N*type form
  const starIdx = rest.indexOf("*");
  if (starIdx > 0 && /^\d+$/.test(rest.slice(0, starIdx))) {
    repeat = parseInt(rest.slice(0, starIdx), 10);
    rest = rest.slice(starIdx + 1);
  } else if (starIdx === 0) {
    // *type means source=>source
    rest = rest.slice(1);
    const info = precisionBytes(rest);
    return {
      repeat: 0,
      sourceType: rest,
      outputType: rest,
      sourceBytes: info,
    };
  }
  // Handle source=>output form
  const arrowIdx = rest.indexOf("=>");
  let sourceType: string;
  let outputType: string;
  if (arrowIdx !== -1) {
    sourceType = rest.slice(0, arrowIdx);
    outputType = rest.slice(arrowIdx + 2);
  } else {
    sourceType = rest;
    outputType = "double";
  }
  return {
    repeat,
    sourceType,
    outputType,
    sourceBytes: precisionBytes(sourceType),
  };
}

function precisionBytes(type: string): number {
  switch (type) {
    case "uint8":
    case "int8":
    case "uchar":
    case "unsigned char":
    case "schar":
    case "signed char":
    case "integer*1":
    case "char":
    case "char*1":
      return 1;
    case "uint16":
    case "int16":
    case "ushort":
    case "short":
    case "integer*2":
      return 2;
    case "uint32":
    case "int32":
    case "uint":
    case "int":
    case "ulong":
    case "long":
    case "single":
    case "float":
    case "float32":
    case "real*4":
    case "integer*4":
      return 4;
    case "uint64":
    case "int64":
    case "double":
    case "float64":
    case "real*8":
    case "integer*8":
      return 8;
    default:
      throw new RuntimeError(`fread: unsupported precision '${type}'`);
  }
}

function readValue(
  dv: DataView,
  offset: number,
  sourceType: string,
  le: boolean
): number {
  switch (sourceType) {
    case "uint8":
    case "uchar":
    case "unsigned char":
    case "char":
    case "char*1":
      return dv.getUint8(offset);
    case "int8":
    case "schar":
    case "signed char":
    case "integer*1":
      return dv.getInt8(offset);
    case "uint16":
    case "ushort":
      return dv.getUint16(offset, le);
    case "int16":
    case "short":
    case "integer*2":
      return dv.getInt16(offset, le);
    case "uint32":
    case "uint":
    case "ulong":
      return dv.getUint32(offset, le);
    case "int32":
    case "int":
    case "long":
    case "integer*4":
      return dv.getInt32(offset, le);
    case "uint64":
      return Number(dv.getBigUint64(offset, le));
    case "int64":
    case "integer*8":
      return Number(dv.getBigInt64(offset, le));
    case "single":
    case "float":
    case "float32":
    case "real*4":
      return dv.getFloat32(offset, le);
    case "double":
    case "float64":
    case "real*8":
      return dv.getFloat64(offset, le);
    default:
      throw new RuntimeError(`fread: unsupported source type '${sourceType}'`);
  }
}

/** Round and saturate to an integer range (NaN -> 0), as fwrite does. */
function saturate(value: number, lo: number, hi: number): number {
  if (Number.isNaN(value)) return 0;
  return Math.min(hi, Math.max(lo, Math.round(value)));
}

function writeValue(
  dv: DataView,
  offset: number,
  value: number,
  sourceType: string,
  le: boolean
): void {
  switch (sourceType) {
    case "uint8":
    case "uchar":
    case "unsigned char":
    case "char":
    case "char*1":
      dv.setUint8(offset, saturate(value, 0, 0xff));
      break;
    case "int8":
    case "schar":
    case "signed char":
    case "integer*1":
      dv.setInt8(offset, saturate(value, -0x80, 0x7f));
      break;
    case "uint16":
    case "ushort":
      dv.setUint16(offset, saturate(value, 0, 0xffff), le);
      break;
    case "int16":
    case "short":
    case "integer*2":
      dv.setInt16(offset, saturate(value, -0x8000, 0x7fff), le);
      break;
    case "uint32":
    case "uint":
    case "ulong":
      dv.setUint32(offset, saturate(value, 0, 0xffffffff), le);
      break;
    case "int32":
    case "int":
    case "long":
    case "integer*4":
      dv.setInt32(offset, saturate(value, -0x80000000, 0x7fffffff), le);
      break;
    case "uint64":
      dv.setBigUint64(offset, BigInt(saturate(value, 0, 2 ** 64 - 1)), le);
      break;
    case "int64":
    case "integer*8":
      dv.setBigInt64(
        offset,
        BigInt(saturate(value, -(2 ** 63), 2 ** 63 - 1)),
        le
      );
      break;
    case "single":
    case "float":
    case "float32":
    case "real*4":
      dv.setFloat32(offset, value, le);
      break;
    case "double":
    case "float64":
    case "real*8":
      dv.setFloat64(offset, value, le);
      break;
    default:
      throw new RuntimeError(`fwrite: unsupported precision '${sourceType}'`);
  }
}

/** Whether a machine format names little-endian byte order. */
function isLittleEndian(machinefmt: string): boolean {
  switch (machinefmt.toLowerCase()) {
    case "b":
    case "ieee-be":
    case "s":
    case "ieee-be.l64":
      return false;
    default:
      return true;
  }
}

/** Precisions whose char data is read and written as encoded text. */
const TEXT_PRECISIONS = new Set(["char", "char*1"]);
const SIGNED_BYTE = new Set(["int8", "schar", "signed char", "integer*1"]);
const BYTE_PRECISIONS = new Set([
  "uint8",
  "uchar",
  "unsigned char",
  "char",
  "char*1",
]);

builtin({
  name: "fread",
  nargin: [1, 5],
  nargout: 2,
  apply: (args, nargout, ctx) => {
    const files = requireFiles(ctx.rt);
    const margs = args;
    const fid = toNumber(margs[0]);

    // Parse arguments: fread(fid), fread(fid,sizeA), fread(fid,precision),
    // fread(fid,sizeA,precision), fread(...,skip), fread(...,machinefmt)
    let sizeA: number[] = [Infinity, 1]; // default: column vector, read all
    let scalarSize = true;
    let precision = "uint8=>double";
    let skip = 0;
    let machinefmt = files.machineFormat(fid);
    let argIdx = 1;

    if (argIdx < margs.length) {
      const a = margs[argIdx];
      // Could be sizeA (number/tensor) or precision (string/char)
      if (isTextArg(a)) {
        precision = toString(a);
        argIdx++;
      } else {
        // It's sizeA
        if (isRuntimeTensor(a)) {
          sizeA = Array.from(a.data);
          scalarSize = sizeA.length < 2;
          argIdx++;
        } else if (isRuntimeNumber(a)) {
          sizeA = [toNumber(a), 1];
          scalarSize = true;
          argIdx++;
        }
        // Check for precision as next arg
        if (argIdx < margs.length && isTextArg(margs[argIdx])) {
          precision = toString(margs[argIdx]);
          argIdx++;
        }
      }
    }
    // Optional skip
    if (argIdx < margs.length) {
      const a = margs[argIdx];
      if (isRuntimeNumber(a) || (isRuntimeTensor(a) && a.data.length === 1)) {
        skip = toNumber(a);
        argIdx++;
      }
    }
    // Optional machinefmt
    if (argIdx < margs.length && isTextArg(margs[argIdx])) {
      machinefmt = toString(margs[argIdx]);
    }

    const parsed = parsePrecision(precision);
    const le = isLittleEndian(machinefmt);
    const bytesPerVal = parsed.sourceBytes;

    // Compute total elements to read
    const m = sizeA[0];
    const n = sizeA.length > 1 ? sizeA[1] : 1;
    const totalElems = m === Infinity || n === Infinity ? Infinity : m * n;

    // Values accumulate in a growable Float64Array.
    let values = new Float64Array(
      Number.isFinite(totalElems) ? Math.min(totalElems, 1 << 24) : 1 << 16
    );
    let count = 0;
    const reserve = (extra: number) => {
      if (count + extra <= values.length) return;
      const grown = new Float64Array(
        Math.max(count + extra, values.length * 2)
      );
      grown.set(values.subarray(0, count));
      values = grown;
    };
    const decodeInto = (bytes: Uint8Array) => {
      const elems = Math.floor(bytes.length / bytesPerVal);
      reserve(elems);
      if (bytesPerVal === 1 && !SIGNED_BYTE.has(parsed.sourceType)) {
        // uint8 and friends: bytes are the values
        values.set(bytes.subarray(0, elems), count);
      } else {
        const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.length);
        for (let i = 0; i < elems; i++)
          values[count + i] = readValue(
            dv,
            i * bytesPerVal,
            parsed.sourceType,
            le
          );
      }
      count += elems;
      return elems;
    };

    if (skip > 0) {
      // Read `repeat` values (default 1), then skip `skip` bytes; repeat.
      const batchSize = parsed.repeat || 1;
      while (count < totalElems) {
        const batch = Math.min(batchSize, totalElems - count);
        const got = decodeInto(files.readBytes(fid, batch * bytesPerVal));
        if (got < batch) break;
        files.readBytes(fid, skip);
      }
    } else if (TEXT_PRECISIONS.has(parsed.sourceType)) {
      // Characters are decoded with the file's encoding.
      const text = files.readChars(fid, totalElems);
      reserve(text.length);
      for (let i = 0; i < text.length; i++)
        values[count++] = text.charCodeAt(i);
    } else {
      decodeInto(files.readBytes(fid, totalElems * bytesPerVal));
    }

    // A scalar size gives a count-by-1 column; [m, n] gives m rows and as
    // many columns as the data fills, the last one zero-padded.
    const rows = scalarSize || m === Infinity ? count : m;
    const cols = count === 0 ? 0 : Math.ceil(count / rows);
    const shape = count === 0 ? [0, 0] : [rows, cols];
    let data: Float64Array = values;
    if (data.length !== rows * cols) {
      data = allocFloat64Array(rows * cols);
      data.set(values.subarray(0, count));
    }

    let result: RuntimeValue;
    if (parsed.outputType === "char" || parsed.outputType === "char*1") {
      // RuntimeChar stores rows one after another.
      let str = "";
      for (let r = 0; r < rows; r++)
        for (let c = 0; c < cols; c++)
          str += String.fromCharCode(data[c * rows + r]);
      result = new RuntimeChar(str, shape);
    } else {
      result = RTV.tensor(data, shape);
    }

    if (nargout >= 2) {
      return [result, RTV.num(count)];
    }
    return result;
  },
});

builtin({
  name: "fwrite",
  nargin: [2, 5],
  apply: (args, _nargout, ctx) => {
    const files = requireFiles(ctx.rt);
    const margs = args;

    const fid = toNumber(margs[0]);
    const data = margs[1];
    const precision = margs.length >= 3 ? toString(margs[2]) : "uint8";
    const skip = margs.length >= 4 ? toNumber(margs[3]) : 0;
    const machinefmt =
      margs.length >= 5 ? toString(margs[4]) : files.machineFormat(fid);

    const parsed = parsePrecision(precision);
    const le = isLittleEndian(machinefmt);

    // Char data written as bytes is encoded with the file's encoding.
    if (
      (isRuntimeChar(data) || isRuntimeString(data)) &&
      BYTE_PRECISIONS.has(parsed.sourceType) &&
      skip === 0
    ) {
      const str = toString(data);
      const written = files.writeBytes(fid, files.encodeText(fid, str));
      return RTV.num(written > 0 ? str.length : 0);
    }

    // Extract values from the data argument
    let values: number[];
    if (isRuntimeTensor(data)) {
      values = Array.from(data.data);
    } else if (isRuntimeNumber(data)) {
      values = [toNumber(data)];
    } else if (isRuntimeLogical(data)) {
      values = [data ? 1 : 0];
    } else if (isRuntimeChar(data) || isRuntimeString(data)) {
      const str = toString(data);
      values = [];
      for (let i = 0; i < str.length; i++) values.push(str.charCodeAt(i));
    } else {
      throw new RuntimeError("fwrite: data must be numeric or char");
    }

    const bytesPerVal = parsed.sourceBytes;
    if (skip === 0) {
      const buf = new Uint8Array(values.length * bytesPerVal);
      const dv = new DataView(buf.buffer);
      for (let i = 0; i < values.length; i++) {
        writeValue(dv, i * bytesPerVal, values[i], parsed.sourceType, le);
      }
      return RTV.num(files.writeBytes(fid, buf) / bytesPerVal);
    }
    // Skip `skip` bytes before each value (zero-filling past the end).
    const valBuf = new Uint8Array(bytesPerVal);
    const dv = new DataView(valBuf.buffer);
    let count = 0;
    for (const v of values) {
      files.skipForWrite(fid, skip);
      writeValue(dv, 0, v, parsed.sourceType, le);
      if (files.writeBytes(fid, valBuf) === 0) break;
      count++;
    }
    return RTV.num(count);
  },
});

// frewind has no outputs in MATLAB (`x = frewind(fid)` errors). It still
// accepts one here because the interpreter requests an output from a
// statement call through a cell element (`c{i}()`); it returns 0.
builtin({
  name: "frewind",
  nargin: 1,
  mfile: true,
  apply: (args, nargout, ctx) => {
    requireFiles(ctx.rt).seek(toNumber(args[0]), 0, -1);
    return nargout >= 1 ? RTV.num(0) : undefined;
  },
});

builtin({
  name: "fseek",
  nargin: 3,
  apply: (args, _nargout, ctx) => {
    const files = requireFiles(ctx.rt);
    const fid = toNumber(args[0]);
    const offset = toNumber(args[1]);
    const o = args[2];
    const name = isTextArg(o) ? toString(o) : String(toNumber(o));
    const origins: Record<string, number> = {
      bof: -1,
      "-1": -1,
      cof: 0,
      "0": 0,
      eof: 1,
      "1": 1,
    };
    if (!(name in origins))
      throw new RuntimeError(`fseek: invalid origin '${name}'`);
    return RTV.num(files.seek(fid, offset, origins[name]));
  },
});

builtin({
  name: "ftell",
  nargin: 1,
  apply: (args, _nargout, ctx) =>
    RTV.num(requireFiles(ctx.rt).tell(toNumber(args[0]))),
});
