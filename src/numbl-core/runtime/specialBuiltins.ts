/**
 * Special builtin functions that require direct runtime access.
 *
 * These builtins are registered as IBuiltins (via registerDynamicIBuiltin)
 * that close over the Runtime instance, unifying them with the standard
 * IBuiltin dispatch path.
 */

import {
  type RuntimeValue,
  RTV,
  toString,
  toNumber,
  displayValue,
  RuntimeError,
} from "../runtime/index.js";
import {
  isRuntimeTensor,
  isRuntimeChar,
  isRuntimeString,
  isRuntimeFunction,
  isRuntimeNumber,
  FloatXArray,
} from "../runtime/types.js";
import { sprintfFormat } from "../../numbl-core/helpers/string.js";
import { ensureRuntimeValue } from "./runtimeHelpers.js";
import {
  arrayfunImpl as _arrayfunImpl,
  cellfunImpl as _cellfunImpl,
  structfunImpl as _structfunImpl,
  bsxfunImpl as _bsxfunImpl,
  subsrefBuiltin as _subsrefBuiltin,
  subsasgnBuiltin as _subsasgnBuiltin,
} from "./runtimeDispatch.js";
import type { Runtime } from "./runtime.js";
import { registerDynamicIBuiltin } from "../interpreter/builtins/types.js";
import { convertJsonValue } from "../interpreter/builtins/misc.js";

/** Names of all special builtins (needed at lowering time before runtime exists). */
export const SPECIAL_BUILTIN_NAMES: readonly string[] = [
  "disp",
  "fprintf",
  "arrayfun",
  "cellfun",
  "structfun",
  "feval",
  "bsxfun",
  "subsref",
  "subsasgn",
  "builtin",
  "fopen",
  "fclose",
  "fgetl",
  "fgets",
  "fileread",
  "feof",
  "ferror",
  "fread",
  "fwrite",
  "frewind",
  "fseek",
  "ftell",
  "fileparts",
  "fullfile",
  "assignin",
  "evalin",
  "drawnow",
  "pause",
  "plot",
  "plot3",
  "surf",
  "scatter",
  "imagesc",
  "contour",
  "contourf",
  "mesh",
  "waterfall",
  "colormap",
  "view",
  "zlabel",
  "colorbar",
  "axis",
  "mfilename",
  "addpath",
  "rmpath",
  "path",
  "mkdir",
  "websave",
  "webread",
  "delete",
  "rmdir",
  "unzip",
  "dir",
  "warning",
  "input",
];

/** Helper: register a special builtin as an IBuiltin that accepts any args. */
function registerSpecial(
  name: string,
  fn: (nargout: number, args: RuntimeValue[]) => unknown
): void {
  registerDynamicIBuiltin({
    name,
    resolve: () => ({
      outputTypes: [{ kind: "unknown" as const }],
      apply: (args: RuntimeValue[], nargout: number) =>
        fn(nargout, args) as RuntimeValue | RuntimeValue[],
    }),
  });
}

/**
 * Register all special builtins as IBuiltins closing over the runtime instance.
 */
export function registerSpecialBuiltins(rt: Runtime): void {
  registerSpecial("disp", (_nargout, args) => {
    if (args.length >= 1) {
      const mv = ensureRuntimeValue(args[0]);
      if (isRuntimeTensor(mv) && mv.data.length === 0) return 0;
      rt.output(displayValue(mv) + "\n");
    }
    return 0;
  });

  registerSpecial("warning", (_nargout, args) => {
    if (args.length === 0) return RTV.num(0);
    const margs = args.map(a => ensureRuntimeValue(a));
    // warning('on'/'off', id) — state query/set form
    if (
      margs.length === 2 &&
      isRuntimeChar(margs[0]) &&
      isRuntimeChar(margs[1])
    ) {
      const state = toString(margs[0]);
      if (state === "on" || state === "off") {
        return RTV.struct(
          new Map<string, RuntimeValue>([
            ["state", RTV.char("on")],
            ["identifier", margs[1]],
          ])
        );
      }
    }
    // Detect warning(msgID, msg, ...) form — msgID contains a ':'
    let fmtIdx = 0;
    if (
      margs.length >= 2 &&
      isRuntimeChar(margs[0]) &&
      toString(margs[0]).includes(":")
    ) {
      fmtIdx = 1;
    }
    const fmt = toString(margs[fmtIdx]);
    const fmtArgs: RuntimeValue[] = [];
    for (let i = fmtIdx + 1; i < margs.length; i++) {
      const a = margs[i];
      if (isRuntimeTensor(a)) {
        for (let j = 0; j < a.data.length; j++)
          fmtArgs.push(RTV.num(a.data[j]));
      } else {
        fmtArgs.push(a);
      }
    }
    if (fmtArgs.length === 0) {
      rt.output("Warning: " + fmt + "\n");
    } else {
      rt.output("Warning: " + sprintfFormat(fmt, fmtArgs) + "\n");
    }
    return RTV.num(0);
  });

  registerSpecial("fprintf", (_nargout, args) => {
    let output = "";
    if (args.length >= 1) {
      const margs = args.map(a => ensureRuntimeValue(a));

      // Detect optional leading fid argument (numeric first arg with >= 2 args)
      let fid = 1; // default stdout
      let fmtIdx = 0;
      if (margs.length >= 2 && isRuntimeNumber(margs[0])) {
        fid = toNumber(margs[0]);
        fmtIdx = 1;
      }

      const fmt = toString(margs[fmtIdx]);
      const scalarArgs: RuntimeValue[] = [];
      for (let i = fmtIdx + 1; i < margs.length; i++) {
        const a = margs[i];
        if (isRuntimeTensor(a)) {
          for (let j = 0; j < a.data.length; j++)
            scalarArgs.push(RTV.num(a.data[j]));
        } else {
          scalarArgs.push(a);
        }
      }
      const specCount = (fmt.match(/%[^%]/g) || []).length;
      if (specCount === 0 || scalarArgs.length === 0) {
        output = sprintfFormat(fmt, scalarArgs);
      } else {
        let idx = 0;
        while (idx < scalarArgs.length) {
          const batch = scalarArgs.slice(idx, idx + specCount);
          output += sprintfFormat(fmt, batch);
          idx += specCount;
        }
      }

      if (fid === 1 || fid === 2) {
        rt.output(output);
      } else {
        if (!rt.fileIO)
          throw new RuntimeError(
            "File I/O is not available in this environment"
          );
        rt.fileIO.fwrite(fid, output);
      }
    }
    return output.length;
  });

  registerSpecial("arrayfun", (nargout, args) => {
    return _arrayfunImpl(rt, nargout, args);
  });

  registerSpecial("cellfun", (nargout, args) => {
    return _cellfunImpl(rt, nargout, args);
  });

  registerSpecial("structfun", (nargout, args) => {
    return _structfunImpl(rt, nargout, args);
  });

  registerSpecial("feval", (nargout, args) => {
    if (args.length < 1)
      throw new RuntimeError("feval requires at least 1 argument");
    const fn = ensureRuntimeValue(args[0]);
    if (isRuntimeFunction(fn)) {
      if (fn.jsFn) {
        return fn.jsFnExpectsNargout
          ? fn.jsFn(nargout, ...args.slice(1))
          : fn.jsFn(...args.slice(1));
      }
      return rt.dispatch(fn.name, nargout, args.slice(1));
    }
    if (isRuntimeChar(fn) || isRuntimeString(fn)) {
      const name = typeof fn === "string" ? fn : fn.value;
      return rt.dispatch(name, nargout, args.slice(1));
    }
    throw new RuntimeError(
      "feval: first argument must be a function handle or name"
    );
  });

  registerSpecial("bsxfun", (nargout, args) => {
    return _bsxfunImpl(rt, nargout, args);
  });

  registerSpecial("subsref", (nargout, args) => {
    return _subsrefBuiltin(rt, nargout, args);
  });

  registerSpecial("subsasgn", (nargout, args) => {
    return _subsasgnBuiltin(rt, nargout, args);
  });

  registerSpecial("builtin", (nargout, args) => {
    if (args.length < 1)
      throw new RuntimeError("builtin requires at least 1 argument");
    const fnNameArg = ensureRuntimeValue(args[0]);
    let fnName: string;
    if (isRuntimeFunction(fnNameArg)) {
      fnName = fnNameArg.name;
    } else if (isRuntimeChar(fnNameArg) || isRuntimeString(fnNameArg)) {
      fnName = isRuntimeString(fnNameArg) ? fnNameArg : fnNameArg.value;
    } else {
      throw new RuntimeError(
        "builtin: first argument must be a function name or handle"
      );
    }
    return rt.callBuiltin(fnName, nargout, args.slice(1));
  });

  // ── File I/O builtins ──────────────────────────────────────────────

  const requireFileIO = () => {
    if (!rt.fileIO)
      throw new RuntimeError("File I/O is not available in this environment");
    return rt.fileIO;
  };

  registerSpecial("fopen", (_nargout, args) => {
    const io = requireFileIO();
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length < 1)
      throw new RuntimeError("fopen requires at least 1 argument");
    const filename = toString(margs[0]);
    const permission = margs.length >= 2 ? toString(margs[1]) : "r";
    return RTV.num(io.fopen(filename, permission));
  });

  registerSpecial("fclose", (_nargout, args) => {
    const io = requireFileIO();
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length < 1) throw new RuntimeError("fclose requires 1 argument");
    const arg = margs[0];
    if (isRuntimeChar(arg) || isRuntimeString(arg)) {
      const s = toString(arg);
      if (s === "all") return RTV.num(io.fclose("all"));
      throw new RuntimeError("fclose: invalid argument");
    }
    return RTV.num(io.fclose(toNumber(arg)));
  });

  registerSpecial("fgetl", (_nargout, args) => {
    const io = requireFileIO();
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length < 1) throw new RuntimeError("fgetl requires 1 argument");
    const result = io.fgetl(toNumber(margs[0]));
    return typeof result === "number" ? RTV.num(result) : RTV.char(result);
  });

  registerSpecial("fgets", (_nargout, args) => {
    const io = requireFileIO();
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length < 1) throw new RuntimeError("fgets requires 1 argument");
    const result = io.fgets(toNumber(margs[0]));
    return typeof result === "number" ? RTV.num(result) : RTV.char(result);
  });

  registerSpecial("fileread", (_nargout, args) => {
    const io = requireFileIO();
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length < 1)
      throw new RuntimeError("fileread requires 1 argument");
    return RTV.char(io.fileread(toString(margs[0])));
  });

  registerSpecial("feof", (_nargout, args) => {
    const io = requireFileIO();
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length < 1) throw new RuntimeError("feof requires 1 argument");
    return RTV.num(io.feof(toNumber(margs[0])));
  });

  registerSpecial("ferror", (_nargout, args) => {
    const io = requireFileIO();
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length < 1) throw new RuntimeError("ferror requires 1 argument");
    return RTV.char(io.ferror(toNumber(margs[0])));
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
        throw new RuntimeError(
          `fread: unsupported source type '${sourceType}'`
        );
    }
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
        dv.setUint8(offset, value);
        break;
      case "int8":
      case "schar":
      case "signed char":
      case "integer*1":
        dv.setInt8(offset, value);
        break;
      case "uint16":
      case "ushort":
        dv.setUint16(offset, value, le);
        break;
      case "int16":
      case "short":
      case "integer*2":
        dv.setInt16(offset, value, le);
        break;
      case "uint32":
      case "uint":
      case "ulong":
        dv.setUint32(offset, value, le);
        break;
      case "int32":
      case "int":
      case "long":
      case "integer*4":
        dv.setInt32(offset, value, le);
        break;
      case "uint64":
        dv.setBigUint64(offset, BigInt(Math.round(value)), le);
        break;
      case "int64":
      case "integer*8":
        dv.setBigInt64(offset, BigInt(Math.round(value)), le);
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

  registerSpecial("fread", (nargout, args) => {
    const io = requireFileIO();
    if (!io.freadBytes)
      throw new RuntimeError("fread is not available in this environment");
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length < 1)
      throw new RuntimeError("fread requires at least 1 argument");
    const fid = toNumber(margs[0]);

    // Parse arguments: fread(fid), fread(fid,sizeA), fread(fid,precision),
    // fread(fid,sizeA,precision), fread(...,skip), fread(...,machinefmt)
    let sizeA: number[] = [Infinity, 1]; // default: column vector, read all
    let precision = "uint8=>double";
    let skip = 0;
    let machinefmt = "n";
    let argIdx = 1;

    if (argIdx < margs.length) {
      const a = margs[argIdx];
      // Could be sizeA (number/tensor) or precision (string/char)
      if (isRuntimeChar(a) || isRuntimeString(a)) {
        precision = toString(a);
        argIdx++;
      } else {
        // It's sizeA
        if (isRuntimeTensor(a)) {
          sizeA = Array.from(a.data);
          argIdx++;
        } else if (isRuntimeNumber(a)) {
          sizeA = [toNumber(a), 1];
          argIdx++;
        }
        // Check for precision as next arg
        if (argIdx < margs.length) {
          const b = margs[argIdx];
          if (isRuntimeChar(b) || isRuntimeString(b)) {
            precision = toString(b);
            argIdx++;
          }
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
    if (argIdx < margs.length) {
      const a = margs[argIdx];
      if (isRuntimeChar(a) || isRuntimeString(a)) {
        machinefmt = toString(a);
      }
    }

    const parsed = parsePrecision(precision);
    const le =
      machinefmt === "l" ||
      machinefmt === "ieee-le" ||
      machinefmt === "ieee-le.l64" ||
      machinefmt === "a" ||
      machinefmt === "n" ||
      machinefmt === "native"; // assume native = little-endian

    const bytesPerVal = parsed.sourceBytes;
    const repeatCount = parsed.repeat || 0;

    // Compute total elements to read
    let totalElems: number;
    const m = sizeA[0];
    const n = sizeA.length > 1 ? sizeA[1] : 1;
    if (m === Infinity && n === 1) {
      totalElems = Infinity;
    } else if (n === Infinity) {
      totalElems = Infinity;
    } else {
      totalElems = m * n;
    }

    // Read data
    const values: number[] = [];
    let count = 0;

    if (repeatCount > 0 && skip > 0) {
      // N*type with skip: read N values, skip `skip` bytes, repeat
      while (count < totalElems) {
        const batch = Math.min(repeatCount, totalElems - count);
        const bytes = io.freadBytes(fid, batch * bytesPerVal);
        if (bytes.length === 0) break;
        const dv = new DataView(
          bytes.buffer,
          bytes.byteOffset,
          bytes.byteLength
        );
        const elemsRead = Math.floor(bytes.length / bytesPerVal);
        for (let i = 0; i < elemsRead; i++) {
          values.push(readValue(dv, i * bytesPerVal, parsed.sourceType, le));
          count++;
        }
        if (elemsRead < batch) break;
        // Skip bytes
        if (count < totalElems) {
          io.freadBytes(fid, skip);
        }
      }
    } else {
      // Simple read: read all at once (or in chunks for Inf)
      if (totalElems === Infinity) {
        // Read in chunks until EOF
        const chunkSize = 65536;
        for (;;) {
          const bytes = io.freadBytes(fid, chunkSize);
          if (bytes.length === 0) break;
          const dv = new DataView(
            bytes.buffer,
            bytes.byteOffset,
            bytes.byteLength
          );
          const elemsRead = Math.floor(bytes.length / bytesPerVal);
          for (let i = 0; i < elemsRead; i++) {
            values.push(readValue(dv, i * bytesPerVal, parsed.sourceType, le));
            count++;
          }
          if (bytes.length < chunkSize) break;
        }
      } else {
        const totalBytes = totalElems * bytesPerVal;
        const bytes = io.freadBytes(fid, totalBytes);
        const dv = new DataView(
          bytes.buffer,
          bytes.byteOffset,
          bytes.byteLength
        );
        const elemsRead = Math.floor(bytes.length / bytesPerVal);
        for (let i = 0; i < elemsRead; i++) {
          values.push(readValue(dv, i * bytesPerVal, parsed.sourceType, le));
          count++;
        }
      }
    }

    // If output type is char, return a char array instead of numeric tensor
    const isCharOutput =
      parsed.outputType === "char" || parsed.outputType === "char*1";

    // Shape the output
    let result: RuntimeValue;
    if (isCharOutput) {
      const str = String.fromCharCode(...values);
      result = RTV.char(str);
    } else if (count === 0) {
      result = RTV.tensor(new FloatXArray(0), [0, 0]);
    } else if (count === 1 && m !== Infinity && n !== Infinity) {
      result = RTV.tensor(new FloatXArray(values), [m, n]);
    } else if (m === Infinity || (m !== Infinity && n === Infinity)) {
      if (m === Infinity) {
        result = RTV.tensor(new FloatXArray(values), [count, 1]);
      } else {
        const cols = Math.ceil(count / m);
        while (values.length < m * cols) values.push(0);
        result = RTV.tensor(new FloatXArray(values), [m, cols]);
      }
    } else {
      while (values.length < m * n) values.push(0);
      result = RTV.tensor(new FloatXArray(values.slice(0, m * n)), [m, n]);
    }

    if (nargout >= 2) {
      return [result, RTV.num(count)];
    }
    return result;
  });

  registerSpecial("fwrite", (_nargout, args) => {
    const io = requireFileIO();
    if (!io.fwriteBytes)
      throw new RuntimeError("fwrite is not available in this environment");
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length < 2)
      throw new RuntimeError("fwrite requires at least 2 arguments");

    const fid = toNumber(margs[0]);
    const data = margs[1];
    let precision = "uint8";
    let skip = 0;
    let machinefmt = "n";

    if (margs.length >= 3) {
      precision = toString(margs[2]);
    }
    if (margs.length >= 4) {
      skip = toNumber(margs[3]);
    }
    if (margs.length >= 5) {
      machinefmt = toString(margs[4]);
    }

    const parsed = parsePrecision(precision);
    const le =
      machinefmt === "l" ||
      machinefmt === "ieee-le" ||
      machinefmt === "ieee-le.l64" ||
      machinefmt === "a" ||
      machinefmt === "n" ||
      machinefmt === "native";

    // Extract values from the data argument
    let values: number[];
    if (isRuntimeTensor(data)) {
      values = Array.from(data.data);
    } else if (isRuntimeNumber(data)) {
      values = [toNumber(data)];
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
      io.fwriteBytes(fid, buf);
    } else {
      // Write with skip bytes between values
      const valBuf = new Uint8Array(bytesPerVal);
      const skipBuf = new Uint8Array(skip);
      const dv = new DataView(valBuf.buffer);
      for (let i = 0; i < values.length; i++) {
        writeValue(dv, 0, values[i], parsed.sourceType, le);
        io.fwriteBytes(fid, valBuf);
        if (i < values.length - 1) {
          io.fwriteBytes(fid, skipBuf);
        }
      }
    }
    return RTV.num(values.length);
  });

  registerSpecial("frewind", (_nargout, args) => {
    const io = requireFileIO();
    if (!io.fseek)
      throw new RuntimeError("frewind is not available in this environment");
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length < 1) throw new RuntimeError("frewind requires 1 argument");
    io.fseek(toNumber(margs[0]), 0, -1);
    return RTV.num(0);
  });

  registerSpecial("fseek", (_nargout, args) => {
    const io = requireFileIO();
    if (!io.fseek)
      throw new RuntimeError("fseek is not available in this environment");
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length < 2)
      throw new RuntimeError("fseek requires at least 2 arguments");
    const fid = toNumber(margs[0]);
    const offset = toNumber(margs[1]);
    let origin = -1; // default: bof
    if (margs.length >= 3) {
      const o = margs[2];
      if (isRuntimeChar(o) || isRuntimeString(o)) {
        const s = toString(o);
        if (s === "bof") origin = -1;
        else if (s === "cof") origin = 0;
        else if (s === "eof") origin = 1;
        else throw new RuntimeError(`fseek: invalid origin '${s}'`);
      } else {
        origin = toNumber(o);
      }
    }
    return RTV.num(io.fseek(fid, offset, origin));
  });

  registerSpecial("ftell", (_nargout, args) => {
    const io = requireFileIO();
    if (!io.ftell)
      throw new RuntimeError("ftell is not available in this environment");
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length < 1) throw new RuntimeError("ftell requires 1 argument");
    return RTV.num(io.ftell(toNumber(margs[0])));
  });

  registerSpecial("mkdir", (nargout, args) => {
    const io = requireFileIO();
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length < 1)
      throw new RuntimeError("mkdir requires at least 1 argument");
    if (!io.mkdir)
      throw new RuntimeError("mkdir is not available in this environment");
    let dirPath: string;
    if (margs.length === 1) {
      dirPath = toString(margs[0]);
    } else {
      // mkdir(parent, newDir) form
      dirPath = toString(margs[0]) + "/" + toString(margs[1]);
    }
    const ok = io.mkdir(dirPath);
    if (nargout === 0) {
      if (!ok)
        throw new RuntimeError(`mkdir: cannot create directory '${dirPath}'`);
      return undefined;
    }
    return nargout <= 1
      ? RTV.char(ok ? "true" : "false")
      : [
          RTV.char(ok ? "true" : "false"),
          RTV.char(ok ? "" : `Cannot create directory '${dirPath}'`),
          RTV.char(""),
        ];
  });

  // ── websave ─────────────────────────────────────────────────────────

  registerSpecial("websave", (_nargout, args) => {
    const io = requireFileIO();
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length < 2)
      throw new RuntimeError("websave requires at least 2 arguments");
    if (!io.websave)
      throw new RuntimeError("websave is not available in this environment");
    const filename = toString(margs[0]);
    let url = toString(margs[1]);
    // Append query parameters (name-value pairs)
    const queryParts: string[] = [];
    for (let i = 2; i + 1 < margs.length; i += 2) {
      const name = encodeURIComponent(toString(margs[i]));
      const value = encodeURIComponent(toString(margs[i + 1]));
      queryParts.push(`${name}=${value}`);
    }
    if (queryParts.length > 0) {
      const sep = url.includes("?") ? "&" : "?";
      url += sep + queryParts.join("&");
    }
    io.websave(url, filename);
    return RTV.char(filename);
  });

  // ── webread ────────────────────────────────────────────────────────

  registerSpecial("webread", (_nargout, args) => {
    const io = requireFileIO();
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length < 1)
      throw new RuntimeError("webread requires at least 1 argument");
    if (!io.webread)
      throw new RuntimeError("webread is not available in this environment");
    let url = toString(margs[0]);
    // Append query parameters (name-value pairs)
    const queryParts: string[] = [];
    for (let i = 1; i + 1 < margs.length; i += 2) {
      const name = encodeURIComponent(toString(margs[i]));
      const value = encodeURIComponent(toString(margs[i + 1]));
      queryParts.push(`${name}=${value}`);
    }
    if (queryParts.length > 0) {
      const sep = url.includes("?") ? "&" : "?";
      url += sep + queryParts.join("&");
    }
    const text = io.webread(url);
    // Try to parse as JSON (MATLAB auto-decodes JSON responses)
    try {
      const parsed = JSON.parse(text);
      return convertJsonValue(parsed);
    } catch {
      // Not JSON — return as char array
      return RTV.char(text);
    }
  });

  // ── delete (file deletion) ──────────────────────────────────────────

  registerSpecial("delete", (_nargout, args) => {
    const io = requireFileIO();
    if (!io.deleteFile)
      throw new RuntimeError("delete is not available in this environment");
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length < 1)
      throw new RuntimeError("delete requires at least 1 argument");
    for (const arg of margs) {
      io.deleteFile(toString(arg));
    }
    return 0;
  });

  // ── rmdir (directory removal) ────────────────────────────────────────

  registerSpecial("rmdir", (nargout, args) => {
    const io = requireFileIO();
    if (!io.rmdir)
      throw new RuntimeError("rmdir is not available in this environment");
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length < 1)
      throw new RuntimeError("rmdir requires at least 1 argument");
    const dirPath = toString(margs[0]);
    const recursive =
      margs.length >= 2 && toString(margs[1]).toLowerCase() === "s";
    const ok = io.rmdir(dirPath, recursive);
    if (nargout === 0) {
      if (!ok)
        throw new RuntimeError(`rmdir: cannot remove directory '${dirPath}'`);
      return undefined;
    }
    return nargout <= 1
      ? RTV.num(ok ? 1 : 0)
      : [
          RTV.num(ok ? 1 : 0),
          RTV.char(ok ? "" : `Cannot remove directory '${dirPath}'`),
          RTV.char(""),
        ];
  });

  // ── unzip (ZIP extraction) ──────────────────────────────────────────

  registerSpecial("unzip", (nargout, args) => {
    const io = requireFileIO();
    if (!io.unzip)
      throw new RuntimeError("unzip is not available in this environment");
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length < 1)
      throw new RuntimeError("unzip requires at least 1 argument");

    // Parse name-value pairs (Password=...)
    let zipfilename = toString(margs[0]);
    let outputfolder = ".";
    let nextArg = 1;

    // Second positional arg is outputfolder (if not a name-value pair)
    if (margs.length >= 2 && toString(margs[1]) !== "Password") {
      outputfolder = toString(margs[1]);
      nextArg = 2;
    }

    // Check for Password name-value pair (not supported, but give clear error)
    for (let i = nextArg; i < margs.length; i += 2) {
      const name = toString(margs[i]);
      if (name === "Password") {
        throw new RuntimeError(
          "unzip: Password-protected ZIP files are not supported"
        );
      }
    }

    // If no extension, try appending .zip
    if (!zipfilename.includes(".")) {
      zipfilename = zipfilename + ".zip";
    }

    // Support URL input: download to a temp file first
    let tempFile: string | null = null;
    if (
      zipfilename.startsWith("http://") ||
      zipfilename.startsWith("https://")
    ) {
      if (!io.websave)
        throw new RuntimeError(
          "unzip: URL download is not available in this environment"
        );
      // Create a temp file path in the output folder
      if (io.mkdir) io.mkdir(outputfolder);
      tempFile = outputfolder + "/.numbl_unzip_tmp_" + Date.now() + ".zip";
      io.websave(zipfilename, tempFile);
      zipfilename = tempFile;
    }

    let extracted: string[];
    try {
      extracted = io.unzip(zipfilename, outputfolder);
    } finally {
      // Clean up temp file
      if (tempFile && io.deleteFile) {
        try {
          io.deleteFile(tempFile);
        } catch {
          // ignore cleanup errors
        }
      }
    }

    if (nargout >= 1) {
      // Return cell array of extracted file names
      const cellData: RuntimeValue[] = extracted.map(f => RTV.char(f));
      return RTV.cell(cellData, [1, cellData.length]);
    }
    return 0;
  });

  // ── dir (directory listing) ──────────────────────────────────────────

  registerSpecial("dir", (nargout, args) => {
    const io = requireFileIO();
    if (!io.listDir)
      throw new RuntimeError("dir is not available in this environment");
    const margs = args.map(a => ensureRuntimeValue(a));
    const pattern = margs.length >= 1 ? toString(margs[0]) : ".";

    const entries = io.listDir(pattern);

    if (nargout === 0) {
      // Display mode: print names separated by spaces
      const names = entries.map(e => e.name);
      if (names.length > 0) {
        rt.output(names.join("  ") + "\n");
      }
      return undefined;
    }

    // Return struct array with fields: name, folder, date, bytes, isdir, datenum
    const fieldNames = ["name", "folder", "date", "bytes", "isdir", "datenum"];
    const elements = entries.map(e => {
      const d = new Date(e.mtimeMs);
      // Format date like MATLAB: dd-Mon-yyyy HH:MM:SS
      const months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
      ];
      const dd = String(d.getDate()).padStart(2, "0");
      const mon = months[d.getMonth()];
      const yyyy = d.getFullYear();
      const HH = String(d.getHours()).padStart(2, "0");
      const MM = String(d.getMinutes()).padStart(2, "0");
      const SS = String(d.getSeconds()).padStart(2, "0");
      const dateStr = `${dd}-${mon}-${yyyy} ${HH}:${MM}:${SS}`;
      // datenum: MATLAB serial date number (days since 0-Jan-0000)
      // MATLAB epoch offset: 719529 days from 0-Jan-0000 to 1-Jan-1970
      const datenum = 719529 + e.mtimeMs / 86400000;

      return RTV.struct(
        new Map<string, RuntimeValue>([
          ["name", RTV.char(e.name)],
          ["folder", RTV.char(e.folder)],
          ["date", RTV.char(dateStr)],
          ["bytes", RTV.num(e.bytes)],
          ["isdir", RTV.logical(e.isdir)],
          ["datenum", RTV.num(datenum)],
        ])
      );
    });

    if (elements.length === 0) {
      // Return empty struct array
      return RTV.structArray(fieldNames, []);
    }
    return RTV.structArray(fieldNames, elements);
  });

  // ── Path utility builtins (pure string operations, no fs needed) ──

  registerSpecial("fileparts", (nargout, args) => {
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length < 1)
      throw new RuntimeError("fileparts requires 1 argument");
    const p = toString(margs[0]);
    // Find the last path separator
    const lastSep = Math.max(p.lastIndexOf("/"), p.lastIndexOf("\\"));
    const dir = lastSep >= 0 ? p.slice(0, lastSep) : "";
    const rest = lastSep >= 0 ? p.slice(lastSep + 1) : p;
    // Find the extension
    const dotIdx = rest.lastIndexOf(".");
    const name = dotIdx >= 0 ? rest.slice(0, dotIdx) : rest;
    const ext = dotIdx >= 0 ? rest.slice(dotIdx) : "";
    if (nargout <= 1) return RTV.char(dir);
    if (nargout === 2) return [RTV.char(dir), RTV.char(name)];
    return [RTV.char(dir), RTV.char(name), RTV.char(ext)];
  });

  registerSpecial("fullfile", (_nargout, args) => {
    const margs = args.map(a => ensureRuntimeValue(a));
    const parts = margs.map(a => toString(a));
    return RTV.char(parts.join("/"));
  });

  // ── Workspace builtins ───────────────────────────────────────────

  registerSpecial("assignin", (_nargout, args) => {
    if (args.length < 3)
      throw new RuntimeError("assignin requires 3 arguments");
    const margs = args.map(a => ensureRuntimeValue(a));
    const ws = toString(margs[0]);
    if (ws !== "base" && ws !== "caller" && ws !== "workspace")
      throw new RuntimeError(
        "assignin: first argument must be 'base', 'caller', or 'workspace'"
      );
    const varName = toString(margs[1]);
    if (ws === "caller") {
      rt.setCallerVariable(varName, args[2]);
    } else {
      rt.setWorkspaceVariable(varName, args[2]);
    }
    return 0;
  });

  registerSpecial("evalin", (_nargout, args) => {
    if (args.length < 2)
      throw new RuntimeError("evalin requires at least 2 arguments");
    const margs = args.map(a => ensureRuntimeValue(a));
    const ws = toString(margs[0]);
    if (ws !== "base" && ws !== "caller" && ws !== "workspace")
      throw new RuntimeError(
        "evalin: first argument must be 'base', 'caller', or 'workspace'"
      );
    const varName = toString(margs[1]);
    const val =
      ws === "caller"
        ? rt.getCallerVariable(varName)
        : rt.getWorkspaceVariable(varName);
    if (val === undefined) {
      if (args.length >= 3) return args[2];
      throw new RuntimeError(
        `evalin: variable '${varName}' does not exist in ${ws}`
      );
    }
    return val;
  });

  // ── Plot builtins ────────────────────────────────────────────────

  registerSpecial("drawnow", () => {
    rt.drawnow();
    return 0;
  });

  registerSpecial("pause", (_nargout, args) => {
    rt.pause(args[0] ?? 0);
    return 0;
  });

  // ── mfilename builtin ────────────────────────────────────────────

  registerSpecial("mfilename", (_nargout, args) => {
    const file = rt.$file ?? "";
    if (args.length > 0) {
      const margs = args.map(a => ensureRuntimeValue(a));
      const opt = toString(margs[0]);
      if (opt === "fullpath") {
        // Return full path without .m extension
        return RTV.char(file.replace(/\.m$/, ""));
      }
    }
    // Return just the base name without path or extension
    const lastSep = Math.max(file.lastIndexOf("/"), file.lastIndexOf("\\"));
    const baseName = lastSep >= 0 ? file.slice(lastSep + 1) : file;
    return RTV.char(baseName.replace(/\.m$/, ""));
  });

  // ── Path management builtins ──────────────────────────────────────

  registerSpecial("addpath", (nargout, args) => {
    if (!rt.onPathChange) {
      throw new RuntimeError("addpath is not available in this environment");
    }
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length < 1)
      throw new RuntimeError("addpath requires at least 1 argument");

    // Detect '-end' or '-begin' flag as last argument
    let position: "begin" | "end" = "begin";
    const lastArg = margs[margs.length - 1];
    if (isRuntimeChar(lastArg) || isRuntimeString(lastArg)) {
      const flag = toString(lastArg);
      if (flag === "-end") {
        position = "end";
        margs.pop();
      } else if (flag === "-begin") {
        margs.pop();
      }
    }

    for (const arg of margs) {
      const dirStr = toString(arg);
      // MATLAB supports pathsep-separated dirs in a single string
      const dirs = dirStr.split(";");
      for (const d of dirs) {
        const trimmed = d.trim();
        if (trimmed) {
          rt.onPathChange("add", trimmed, position);
        }
      }
    }

    if (nargout >= 1) return RTV.char(rt.searchPaths.join(";"));
    return 0;
  });

  registerSpecial("rmpath", (nargout, args) => {
    if (!rt.onPathChange) {
      throw new RuntimeError("rmpath is not available in this environment");
    }
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length < 1)
      throw new RuntimeError("rmpath requires at least 1 argument");

    for (const arg of margs) {
      const dirStr = toString(arg);
      const dirs = dirStr.split(";");
      for (const d of dirs) {
        const trimmed = d.trim();
        if (trimmed) {
          rt.onPathChange("remove", trimmed, "begin");
        }
      }
    }

    if (nargout >= 1) return RTV.char(rt.searchPaths.join(";"));
    return 0;
  });

  // ── input builtin ────────────────────────────────────────────────

  registerSpecial("input", (_nargout, args) => {
    const margs = args.map(a => ensureRuntimeValue(a));
    const prompt = margs.length >= 1 ? toString(margs[0]) : "";
    const isStringMode =
      margs.length >= 2 &&
      (isRuntimeChar(margs[1]) || isRuntimeString(margs[1])) &&
      toString(margs[1]) === "s";

    const line = rt.readInput(prompt);

    if (isStringMode) {
      return RTV.char(line);
    }

    // Numeric/expression mode: empty input returns []
    if (line.trim() === "") {
      return RTV.tensor(new Float64Array(0), [0, 0]);
    }

    // Evaluate the expression in the current workspace
    if (!rt.evalLocalCallback) {
      throw new RuntimeError(
        "input: expression evaluation is not available in this environment"
      );
    }
    // Gather current workspace variables for evaluation context
    const vars: Record<string, RuntimeValue> = {};
    for (const [name, acc] of rt.workspaceAccessors) {
      const val = acc.get();
      if (val !== undefined) vars[name] = ensureRuntimeValue(val);
    }
    for (const [name, val] of rt.dynamicWorkspaceVars) {
      if (val !== undefined) vars[name] = ensureRuntimeValue(val);
    }
    const result = rt.evalLocalCallback(line, vars, text => rt.output(text));
    return ensureRuntimeValue(result.returnValue);
  });

  registerSpecial("path", (nargout, args) => {
    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length === 0) {
      // path() — return current path
      return RTV.char(rt.searchPaths.join(";"));
    }
    // path(newpath) — set the entire path (not commonly used, but supported)
    if (margs.length >= 1 && rt.onPathChange) {
      // Remove all current paths then add the new ones
      for (const sp of [...rt.searchPaths]) {
        rt.onPathChange("remove", sp, "begin");
      }
      const newPath = toString(margs[0]);
      const dirs = newPath.split(";");
      for (const d of dirs) {
        const trimmed = d.trim();
        if (trimmed) {
          rt.onPathChange("add", trimmed, "end");
        }
      }
    }
    if (nargout >= 1) return RTV.char(rt.searchPaths.join(";"));
    return 0;
  });
}
