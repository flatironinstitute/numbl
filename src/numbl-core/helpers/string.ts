/**
 * String manipulation builtin functions
 */

import {
  RuntimeValue,
  RTV,
  toNumber,
  toString,
  RuntimeError,
  displayValue,
} from "../runtime/index.js";
import {
  FloatXArray,
  isRuntimeCell,
  isRuntimeChar,
  isRuntimeLogical,
  isRuntimeNumber,
  isRuntimeString,
  isRuntimeTensor,
  RuntimeCell,
  RuntimeChar,
} from "../runtime/types.js";
import { register, builtinSingle, isBuiltin } from "./registry.js";
import { parseMFile } from "../parser/index.js";
import type { Expr, Stmt } from "../parser/types.js";
import { BUILTIN_CONSTANTS } from "../lowering/constants.js";

/** Convert a number to string.
 *  Uses short-g style: ~5 significant digits,
 *  switching to scientific notation when exponent < -4 or >= 5. */
function numStr(n: number): string {
  if (n === Infinity) return "Inf";
  if (n === -Infinity) return "-Inf";
  if (isNaN(n)) return "NaN";
  if (n === 0) return "0";
  const prec = 5;
  const exp = Math.floor(Math.log10(Math.abs(n)));
  let s: string;
  if (exp < -4 || exp >= prec) {
    // Scientific notation
    s = n.toExponential(prec - 1);
    // Strip trailing zeros from mantissa
    const ePos = s.indexOf("e");
    let mantissa = s.slice(0, ePos);
    const expPart0 = s.slice(ePos);
    if (mantissa.includes(".")) mantissa = mantissa.replace(/\.?0+$/, "");
    // Pad exponent to at least 2 digits: e+5 -> e+05
    const expPart = expPart0.replace(/([eE][+-])(\d)$/, "$1" + "0$2");
    s = mantissa + expPart;
  } else {
    // Fixed notation
    if (Number.isInteger(n)) return String(n);
    s = n.toPrecision(prec);
    // Strip trailing zeros after decimal point
    if (s.includes(".")) s = s.replace(/\.?0+$/, "");
  }
  return s;
}

function applyWidth(spec: string, str: string): string {
  // Parse flags and width from format spec like "%04", "%-10", "%010"
  // Flags: '-', '+', '0', ' ', '#'. Then width digits follow.
  // A leading '0' before width means zero-pad flag, e.g. %04x -> flag='0', width=4
  // But %10f -> no flag, width=10
  const m = spec.match(/^%([-+ #]*)0?(\d+)?/);
  if (!m) return str;
  const explicitFlags = m[1] || "";
  const leftAlign = explicitFlags.includes("-");
  // Check if there's a '0' flag: it appears between explicit flags and width digits
  const afterPercent = spec.slice(1);
  const flagAndWidth = afterPercent.match(/^([-+ #]*)(0?)(\d+)?/);
  const zeroFlag = flagAndWidth ? flagAndWidth[2] === "0" : false;
  const width = flagAndWidth && flagAndWidth[3] ? parseInt(flagAndWidth[3]) : 0;
  if (width <= str.length) return str;
  const zeroPad = !leftAlign && zeroFlag;
  const padLen = width - str.length;
  if (leftAlign) return str + " ".repeat(padLen);
  if (zeroPad) {
    if (str[0] === "-" || str[0] === "+") {
      return str[0] + "0".repeat(padLen) + str.slice(1);
    }
    return "0".repeat(padLen) + str;
  }
  return " ".repeat(padLen) + str;
}

export function sprintfFormat(fmt: string, args: RuntimeValue[]): string {
  // Flatten all args into scalar RuntimeValues (expands arrays element-by-element, column-major)
  const flatArgs: RuntimeValue[] = [];
  for (const arg of args) {
    if (isRuntimeTensor(arg)) {
      // tensor.data is stored in column-major order
      for (let k = 0; k < arg.data.length; k++) {
        flatArgs.push(arg.data[k] as RuntimeValue);
      }
    } else {
      flatArgs.push(arg);
    }
  }

  let result = "";
  let argIdx = 0;

  // Process format string, repeating if needed to consume all args
  do {
    const startArgIdx = argIdx;
    let outOfArgs = false;
    let i = 0;
    while (i < fmt.length && !outOfArgs) {
      if (fmt[i] === "%" && i + 1 < fmt.length) {
        i++;
        // Parse format specifier
        let spec = "%";
        while (i < fmt.length && !"dfigeEsoxXuc%".includes(fmt[i])) {
          spec += fmt[i];
          i++;
        }
        if (i < fmt.length) {
          const ch = fmt[i];
          i++;
          if (ch === "%") {
            result += "%";
          } else if (argIdx >= flatArgs.length) {
            outOfArgs = true;
          } else if (ch === "d" || ch === "i" || ch === "u") {
            const raw = toNumber(flatArgs[argIdx++]);
            const isInt = Number.isInteger(raw);
            const canPrintAsInt = ch === "u" ? isInt && raw >= 0 : isInt;
            if (!canPrintAsInt) {
              // Non-integer (or negative for %u) falls back to %e
              let eStr = raw.toExponential(6);
              eStr = eStr.replace(/e([+-])(\d)$/, "e$1" + "0$2");
              result += applyWidth(spec, eStr);
            } else {
              const n = raw;
              const flags = spec.slice(1); // everything between % and the type char
              const hasPlus = flags.includes("+");
              const leftAlign = flags.includes("-");
              const widthMatch = spec.match(/^%[^0-9]*(\d+)/);
              const width = widthMatch ? parseInt(widthMatch[1]) : 0;
              const zeroPad = !leftAlign && /^[-+ ]*0/.test(spec.slice(1));
              const s = String(Math.abs(n));
              const sign = n < 0 ? "-" : hasPlus ? "+" : "";
              if (width > 0) {
                const padChar = zeroPad ? "0" : " ";
                const padLen = Math.max(0, width - sign.length - s.length);
                const pad = padChar.repeat(padLen);
                result += leftAlign
                  ? sign + s + " ".repeat(padLen)
                  : zeroPad
                    ? sign + pad + s
                    : pad + sign + s;
              } else {
                result += sign + s;
              }
            }
          } else if (ch === "f") {
            const n = toNumber(flatArgs[argIdx++]);
            if (!isFinite(n) || isNaN(n)) {
              result += applyWidth(spec, numStr(n));
            } else {
              const fFlags = spec.slice(1);
              const fHasPlus = fFlags.includes("+");
              const precMatch = spec.match(/\.(\d+)/);
              const prec = precMatch ? parseInt(precMatch[1]) : 6;
              const formatted = n.toFixed(prec);
              const fSign = n < 0 ? "" : fHasPlus ? "+" : "";
              result += applyWidth(spec, fSign + formatted);
            }
          } else if (ch === "e" || ch === "E") {
            const n = toNumber(flatArgs[argIdx++]);
            if (!isFinite(n) || isNaN(n)) {
              result += applyWidth(spec, numStr(n));
            } else {
              const precMatch = spec.match(/\.(\d+)/);
              const prec = precMatch ? parseInt(precMatch[1]) : 6;
              let eStr = n.toExponential(prec);
              // Pads exponent to at least 2 digits: e+3 -> e+03
              eStr = eStr.replace(/e([+-])(\d)$/, "e$1" + "0$2");
              if (ch === "E") eStr = eStr.toUpperCase();
              result += applyWidth(spec, eStr);
            }
          } else if (ch === "x" || ch === "X") {
            const n = Math.round(toNumber(flatArgs[argIdx++]));
            let s = Math.abs(n).toString(16);
            if (ch === "X") s = s.toUpperCase();
            result += applyWidth(spec, s);
          } else if (ch === "o") {
            const n = Math.round(toNumber(flatArgs[argIdx++]));
            result += applyWidth(spec, Math.abs(n).toString(8));
          } else if (ch === "g" || ch === "G") {
            const gVal = toNumber(flatArgs[argIdx++]);
            if (!isFinite(gVal) || isNaN(gVal)) {
              result += applyWidth(spec, numStr(gVal));
            } else {
              const precMatch = spec.match(/\.(\d+)/);
              const gPrec = precMatch ? parseInt(precMatch[1]) : 6;
              // C %g: use %e if exponent < -4 or >= precision, else %f
              let gStr: string;
              if (gVal === 0) {
                gStr = "0";
              } else {
                const exp = Math.floor(Math.log10(Math.abs(gVal)));
                if (exp < -4 || exp >= gPrec) {
                  // Use scientific notation with (prec-1) decimal places
                  gStr = gVal.toExponential(gPrec - 1);
                  // Strip trailing zeros from mantissa
                  const ePos = gStr.indexOf("e");
                  let mantissa = gStr.slice(0, ePos);
                  let expPart = gStr.slice(ePos);
                  if (mantissa.includes(".")) {
                    mantissa = mantissa.replace(/\.?0+$/, "");
                  }
                  // Pad exponent to at least 2 digits
                  expPart = expPart.replace(/e([+-])(\d)$/, "e$1" + "0$2");
                  gStr = mantissa + expPart;
                } else {
                  // Use fixed notation with enough significant digits
                  gStr = gVal.toPrecision(gPrec);
                  if (gStr.includes(".")) {
                    gStr = gStr.replace(/\.?0+$/, "");
                  }
                  // toPrecision may still produce e notation for large numbers
                  if (gStr.includes("e")) {
                    gStr = String(parseFloat(gStr));
                  }
                }
              }
              if (ch === "G") gStr = gStr.toUpperCase();
              result += applyWidth(spec, gStr);
            }
          } else if (ch === "s") {
            const sVal = toString(flatArgs[argIdx++]);
            const sFlags = spec.slice(1);
            const sLeftAlign = sFlags.includes("-");
            const sWidthMatch = spec.match(/^%[^0-9]*(\d+)/);
            const sWidth = sWidthMatch ? parseInt(sWidthMatch[1]) : 0;
            if (sWidth > sVal.length) {
              const sPad = " ".repeat(sWidth - sVal.length);
              result += sLeftAlign ? sVal + sPad : sPad + sVal;
            } else {
              result += sVal;
            }
          } else if (ch === "c") {
            result += String.fromCharCode(
              Math.round(toNumber(flatArgs[argIdx++]))
            );
          } else {
            result += spec + ch;
            argIdx++;
          }
        }
      } else if (fmt[i] === "\\" && i + 1 < fmt.length) {
        i++;
        switch (fmt[i]) {
          case "n":
            result += "\n";
            break;
          case "t":
            result += "\t";
            break;
          case "\\":
            result += "\\";
            break;
          default:
            result += "\\" + fmt[i];
        }
        i++;
      } else {
        result += fmt[i];
        i++;
      }
    }
    // If no args were consumed in this pass, stop to avoid infinite loop
    if (argIdx === startArgIdx) break;
  } while (argIdx < flatArgs.length);

  return result;
}

export function registerStringFunctions(): void {
  register(
    "int2str",
    builtinSingle(args => {
      if (args.length < 1)
        throw new RuntimeError("int2str requires at least 1 argument");
      const v = args[0];
      const roundInt = (x: number) => Math.sign(x) * Math.round(Math.abs(x));
      if (isRuntimeTensor(v)) {
        const rows = v.shape[0] || 1;
        const cols = v.shape.length >= 2 ? v.shape[1] : v.data.length;
        const rowStrs: string[] = [];
        for (let r = 0; r < rows; r++) {
          const vals: string[] = [];
          for (let c = 0; c < cols; c++) {
            vals.push(String(roundInt(v.data[r + c * rows])));
          }
          rowStrs.push(vals.join("  "));
        }
        return RTV.char(rowStrs.join("\n"));
      }
      const n = toNumber(v);
      // Rounds half away from zero
      return RTV.char(String(roundInt(n)));
    })
  );

  register(
    "num2str",
    builtinSingle(args => {
      if (args.length < 1)
        throw new RuntimeError("num2str requires at least 1 argument");
      const v = args[0];
      // Format string provided
      if (
        args.length >= 2 &&
        (isRuntimeString(args[1]) || isRuntimeChar(args[1]))
      ) {
        return RTV.char(
          sprintfFormat(isRuntimeString(args[1]) ? args[1] : args[1].value, [v])
        );
      }
      // Numeric precision (significant digits) provided
      const precision =
        args.length >= 2 && isRuntimeNumber(args[1])
          ? (args[1] as number)
          : undefined;
      const fmtNum = (x: number): string => {
        if (precision !== undefined) {
          let s = x.toPrecision(precision);
          // Remove trailing zeros after decimal point
          if (s.includes(".")) {
            const eIdx = s.search(/[eE]/);
            if (eIdx === -1) {
              s = s.replace(/\.?0+$/, "");
            } else {
              // Strip trailing zeros from mantissa before exponent
              const mantissa = s.slice(0, eIdx).replace(/\.?0+$/, "");
              let expPart = s.slice(eIdx);
              // Pad exponent to at least 2 digits: e+5 -> e+05
              expPart = expPart.replace(/([eE][+-])(\d)$/, "$1" + "0$2");
              s = mantissa + expPart;
            }
          }
          return s;
        }
        return numStr(x);
      };
      // Scalar
      if (isRuntimeNumber(v) || isRuntimeLogical(v)) {
        return RTV.char(fmtNum(toNumber(v)));
      }
      // Tensor: convert each row, right-align columns
      if (isRuntimeTensor(v)) {
        const nRows = v.shape[0] || 1;
        const nCols = v.data.length / nRows;
        // Convert all elements to strings
        const strs: string[][] = [];
        for (let r = 0; r < nRows; r++) {
          const row: string[] = [];
          for (let c = 0; c < nCols; c++) {
            row.push(fmtNum(v.data[c * nRows + r]));
          }
          strs.push(row);
        }
        // Find max width per column
        const colWidths: number[] = [];
        for (let c = 0; c < nCols; c++) {
          let maxW = 0;
          for (let r = 0; r < nRows; r++) {
            if (strs[r][c].length > maxW) maxW = strs[r][c].length;
          }
          colWidths.push(maxW);
        }
        // Build rows with right-aligned columns, separated by spaces
        const rowStrs = strs.map(row =>
          row.map((s, c) => s.padStart(colWidths[c])).join("  ")
        );
        if (nRows === 1) {
          return RTV.char(rowStrs[0]);
        }
        // Multi-row char array: all rows padded to same width, concatenated
        const rowWidth = rowStrs[0].length;
        const paddedRows = rowStrs.map(r => r.padEnd(rowWidth));
        const result: RuntimeChar = {
          kind: "char",
          value: paddedRows.join(""),
          shape: [nRows, rowWidth],
        };
        return result;
      }
      return RTV.char(String(toNumber(v)));
    })
  );

  register(
    "str2num",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("str2num requires 1 argument");
      const s = toString(args[0]);
      const n = Number(s);
      return Number.isNaN(n)
        ? RTV.tensor(new FloatXArray(0), [0, 0])
        : RTV.num(n);
    })
  );

  register(
    "str2double",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("str2double requires 1 argument");
      const s = toString(args[0]).trim();
      return RTV.num(s === "" ? NaN : Number(s));
    })
  );

  register(
    "sprintf",
    builtinSingle(args => {
      if (args.length < 1)
        throw new RuntimeError("sprintf requires at least 1 argument");
      const fmtArg = args[0];
      const fmt = toString(fmtArg);
      const result = sprintfFormat(fmt, args.slice(1));
      return isRuntimeChar(fmtArg) ? RTV.char(result) : RTV.string(result);
    })
  );

  // Helper: check if a value is a text type (char or string)
  const isText = (v: RuntimeValue) => isRuntimeChar(v) || isRuntimeString(v);

  // Helper: try to get string value, return null for non-text types
  const tryToString = (v: RuntimeValue): string | null => {
    if (isRuntimeChar(v) || isRuntimeString(v)) return toString(v);
    return null;
  };

  // Shared implementation for strcmp/strcmpi
  const strcmpImpl = (
    args: RuntimeValue[],
    name: string,
    caseInsensitive: boolean
  ): RuntimeValue => {
    if (args.length !== 2)
      throw new RuntimeError(`${name} requires 2 arguments`);
    const [a, b] = args;
    const norm = caseInsensitive
      ? (s: string) => s.toLowerCase()
      : (s: string) => s;

    // Cell vs cell: element-wise comparison (same size or one is scalar)
    if (isRuntimeCell(a) && isRuntimeCell(b)) {
      const aScalar = a.data.length === 1;
      const bScalar = b.data.length === 1;
      const refCell = aScalar ? b : a;
      const data = new FloatXArray(refCell.data.length);
      for (let i = 0; i < refCell.data.length; i++) {
        const ai = aScalar ? a.data[0] : a.data[i];
        const bi = bScalar ? b.data[0] : b.data[i];
        const sa = tryToString(ai);
        const sb = tryToString(bi);
        data[i] = sa !== null && sb !== null && norm(sa) === norm(sb) ? 1 : 0;
      }
      return RTV.tensor(data, [...refCell.shape]);
    }

    // Cell vs scalar text: compare each cell element with the scalar
    if (isRuntimeCell(a) || isRuntimeCell(b)) {
      const cellArg = isRuntimeCell(a) ? a : (b as RuntimeCell);
      const otherArg = isRuntimeCell(a) ? b : a;
      const otherStr = tryToString(otherArg);
      const data = new FloatXArray(cellArg.data.length);
      for (let i = 0; i < cellArg.data.length; i++) {
        const s = tryToString(cellArg.data[i]);
        data[i] =
          s !== null && otherStr !== null && norm(s) === norm(otherStr) ? 1 : 0;
      }
      return RTV.tensor(data, [...cellArg.shape]);
    }

    // Scalar vs scalar: unsupported types return 0
    if (!isText(a) || !isText(b)) return RTV.logical(false);
    return RTV.logical(norm(toString(a)) === norm(toString(b)));
  };

  register(
    "strcmp",
    builtinSingle(args => strcmpImpl(args, "strcmp", false))
  );

  register(
    "strcmpi",
    builtinSingle(args => strcmpImpl(args, "strcmpi", true))
  );

  register(
    "strncmp",
    builtinSingle(args => {
      if (args.length !== 3)
        throw new RuntimeError("strncmp requires 3 arguments");
      const n = Math.round(toNumber(args[2]));
      const s1 = tryToString(args[0]);
      const s2 = tryToString(args[1]);
      if (s1 === null || s2 === null) return RTV.logical(false);
      return RTV.logical(
        s1.substring(0, n) === s2.substring(0, n) &&
          s1.length >= n &&
          s2.length >= n
      );
    })
  );

  register(
    "strncmpi",
    builtinSingle(args => {
      if (args.length !== 3)
        throw new RuntimeError("strncmpi requires 3 arguments");
      const n = Math.round(toNumber(args[2]));
      const s1 = tryToString(args[0])?.toLowerCase() ?? null;
      const s2 = tryToString(args[1])?.toLowerCase() ?? null;
      if (s1 === null || s2 === null) return RTV.logical(false);
      return RTV.logical(
        s1.substring(0, n) === s2.substring(0, n) &&
          s1.length >= n &&
          s2.length >= n
      );
    })
  );

  register(
    "lower",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("lower requires 1 argument");
      const v = args[0];
      if (isRuntimeChar(v)) return RTV.char(v.value.toLowerCase());
      return RTV.string(toString(v).toLowerCase());
    })
  );

  register(
    "upper",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("upper requires 1 argument");
      const v = args[0];
      if (isRuntimeChar(v)) return RTV.char(v.value.toUpperCase());
      return RTV.string(toString(v).toUpperCase());
    })
  );

  register(
    "strlength",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("strlength requires 1 argument");
      const v = args[0];
      if (isRuntimeChar(v) || isRuntimeString(v))
        return RTV.num(isRuntimeString(v) ? v.length : v.value.length);
      throw new RuntimeError("strlength: argument must be a string or char");
    })
  );

  register(
    "strtrim",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("strtrim requires 1 argument");
      const v = args[0];
      if (isRuntimeChar(v)) return RTV.char(v.value.trim());
      return RTV.string(toString(v).trim());
    })
  );

  register(
    "char",
    builtinSingle(args => {
      if (args.length === 1 && isRuntimeChar(args[0])) return args[0];
      if (args.length === 1 && isRuntimeString(args[0]))
        return RTV.char(args[0]);
      if (
        args.length === 1 &&
        (isRuntimeNumber(args[0]) || isRuntimeTensor(args[0]))
      ) {
        // char(65) → 'A'
        const v = args[0];
        if (isRuntimeNumber(v))
          return RTV.char(String.fromCharCode(Math.round(v)));
        const chars: string[] = [];
        for (let i = 0; i < v.data.length; i++) {
          chars.push(String.fromCharCode(Math.round(v.data[i])));
        }
        return RTV.char(chars.join(""));
      }
      throw new RuntimeError("char: unsupported arguments");
    })
  );

  register(
    "string",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("string requires 1 argument");
      if (isRuntimeString(args[0])) return args[0];
      if (isRuntimeChar(args[0])) return RTV.string(args[0].value);
      return RTV.string(displayValue(args[0]));
    })
  );

  register(
    "strcat",
    builtinSingle(args => {
      if (args.length < 1)
        throw new RuntimeError("strcat requires at least 1 argument");
      // If any input is a string, result is a string (trailing whitespace preserved)
      const anyString = args.some(a => isRuntimeString(a));
      if (anyString) {
        return RTV.string(args.map(a => toString(a)).join(""));
      }
      // All char inputs: strip trailing ASCII whitespace from each arg
      const parts = args.map(a => toString(a).replace(/[ \t\v\n\r\f]+$/, ""));
      return RTV.char(parts.join(""));
    })
  );

  register(
    "strsplit",
    builtinSingle(args => {
      if (args.length < 1)
        throw new RuntimeError("strsplit requires at least 1 argument");
      const s = toString(args[0]);
      let parts: string[];
      if (args.length < 2) {
        // Split on any whitespace, collapsing consecutive whitespace
        const trimmed = s.trim();
        parts = trimmed.length === 0 ? [""] : trimmed.split(/[ \f\n\r\t\v]+/);
      } else {
        // Delimiter can be a string/char or a cell array of strings
        let delims: string[];
        if (isRuntimeCell(args[1])) {
          delims = args[1].data.map(d => toString(d));
        } else {
          delims = [toString(args[1])];
        }
        const escaped = delims
          .map(d => d.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
          .join("|");
        parts = s.split(new RegExp("(?:" + escaped + ")+"));
      }
      return RTV.cell(
        parts.map(p => RTV.string(p)),
        [1, parts.length]
      );
    })
  );

  register(
    "strjoin",
    builtinSingle(args => {
      if (args.length < 1)
        throw new RuntimeError("strjoin requires at least 1 argument");
      if (!isRuntimeCell(args[0]))
        throw new RuntimeError("strjoin: first argument must be a cell array");
      const elements = args[0].data.map(v => toString(v));
      const delim = args.length >= 2 ? toString(args[1]) : " ";
      return RTV.string(elements.join(delim));
    })
  );

  register(
    "strfind",
    builtinSingle(args => {
      if (args.length !== 2)
        throw new RuntimeError("strfind requires 2 arguments");
      const s = toString(args[0]);
      const pattern = toString(args[1]);
      const indices: number[] = [];
      let pos = 0;
      while (pos <= s.length - pattern.length) {
        const idx = s.indexOf(pattern, pos);
        if (idx === -1) break;
        indices.push(idx + 1); // 1-based
        pos = idx + 1;
      }
      if (indices.length === 0) {
        return RTV.tensor(new FloatXArray(0), [1, 0]);
      }
      const data = new FloatXArray(indices.length);
      indices.forEach((v, i) => (data[i] = v));
      return RTV.tensor(data, [1, indices.length]);
    })
  );

  register(
    "contains",
    builtinSingle(args => {
      if (args.length < 2)
        throw new RuntimeError("contains requires at least 2 arguments");
      const s = toString(args[0]);
      const pat = args[1];
      // Pattern can be a single string or a cell array of strings
      if (isRuntimeCell(pat)) {
        for (let i = 0; i < pat.data.length; i++) {
          if (s.includes(toString(pat.data[i]))) return RTV.logical(true);
        }
        return RTV.logical(false);
      }
      return RTV.logical(s.includes(toString(pat)));
    })
  );

  register(
    "strrep",
    builtinSingle(args => {
      if (args.length !== 3)
        throw new RuntimeError("strrep requires 3 arguments");
      const v = args[0];
      const s = toString(v);
      const old = toString(args[1]);
      const rep = toString(args[2]);
      const result = s.split(old).join(rep);
      return isRuntimeChar(v) ? RTV.char(result) : RTV.string(result);
    })
  );

  // ── regexp / regexpi / regexprep ──────────────────────────────────────

  register(
    "regexp",
    builtinSingle((args, nargout) => {
      if (args.length < 2)
        throw new RuntimeError("regexp requires at least 2 arguments");
      const str = toString(args[0]);
      const pat = toString(args[1]);

      // Parse option flags
      let matchOnce = false;
      const outModes: string[] = [];
      for (let i = 2; i < args.length; i++) {
        const opt = toString(args[i]).toLowerCase();
        if (opt === "once") matchOnce = true;
        else outModes.push(opt);
      }
      // Default output mode is 'start' if no mode specified
      if (outModes.length === 0 && nargout <= 1) outModes.push("start");
      // With multiple nargout and no explicit modes: start, end, ...
      if (outModes.length === 0) {
        outModes.push(
          "start",
          "end",
          "tokenextents",
          "match",
          "tokens",
          "names"
        );
      }

      const re = new RegExp(pat, "g");
      const starts: number[] = [];
      const ends: number[] = [];
      const matches: string[] = [];
      const tokensList: string[][] = [];
      let m: RegExpExecArray | null;
      while ((m = re.exec(str)) !== null) {
        starts.push(m.index + 1); // 1-based
        ends.push(m.index + m[0].length);
        matches.push(m[0]);
        const toks: string[] = [];
        for (let g = 1; g < m.length; g++) toks.push(m[g] ?? "");
        tokensList.push(toks);
        if (matchOnce) break;
        if (m[0].length === 0) re.lastIndex++;
      }

      function buildOutput(mode: string): RuntimeValue {
        if (mode === "start") {
          if (matchOnce)
            return starts.length > 0
              ? RTV.num(starts[0])
              : RTV.tensor(new FloatXArray(0), [1, 0]);
          const d = new FloatXArray(starts);
          return RTV.tensor(d, [1, starts.length]);
        }
        if (mode === "end") {
          if (matchOnce)
            return ends.length > 0
              ? RTV.num(ends[0])
              : RTV.tensor(new FloatXArray(0), [1, 0]);
          const d = new FloatXArray(ends);
          return RTV.tensor(d, [1, ends.length]);
        }
        if (mode === "match") {
          if (matchOnce)
            return matches.length > 0 ? RTV.char(matches[0]) : RTV.char("");
          return RTV.cell(
            matches.map(s => RTV.char(s)),
            [1, matches.length]
          );
        }
        if (mode === "tokens") {
          if (matchOnce) {
            if (tokensList.length === 0) return RTV.cell([], [1, 0]);
            return RTV.cell(
              tokensList[0].map(s => RTV.char(s)),
              [1, tokensList[0].length]
            );
          }
          return RTV.cell(
            tokensList.map(toks =>
              RTV.cell(
                toks.map(s => RTV.char(s)),
                [1, toks.length]
              )
            ),
            [1, tokensList.length]
          );
        }
        if (mode === "names") {
          // Build struct from named capture groups
          if (starts.length === 0) {
            // No match — return empty struct array (0x0)
            return RTV.tensor(new FloatXArray(0), [0, 0]);
          }
          // Use the first match's named groups for 'once', or first match for default
          // JavaScript RegExp stores named groups in m.groups
          // We need to re-run the regex to get named groups
          const namedRe = new RegExp(pat, "g");
          const namedMatches: Record<string, string>[] = [];
          let nm: RegExpExecArray | null;
          while ((nm = namedRe.exec(str)) !== null) {
            namedMatches.push(
              Object.fromEntries(
                Object.entries(nm.groups ?? {}).map(([k, v]) => [k, v ?? ""])
              )
            );
            if (matchOnce) break;
            if (nm[0].length === 0) namedRe.lastIndex++;
          }
          if (matchOnce) {
            // Return a single struct
            const fields: Record<string, RuntimeValue> = {};
            for (const [k, v] of Object.entries(namedMatches[0] ?? {})) {
              fields[k] = RTV.char(v);
            }
            return RTV.struct(fields);
          }
          // Multiple matches: return 1xN struct array
          if (namedMatches.length === 0) {
            return RTV.tensor(new FloatXArray(0), [0, 0]);
          }
          // For multiple matches, return a 1xN struct array
          const keys = Object.keys(namedMatches[0]);
          const elements = namedMatches.map(nm2 => {
            const fields: Record<string, RuntimeValue> = {};
            for (const k of keys) fields[k] = RTV.char(nm2[k] ?? "");
            return RTV.struct(fields);
          });
          return RTV.structArray(keys, elements);
        }
        // tokenextents — return empty for now
        if (matchOnce) return RTV.tensor(new FloatXArray(0), [1, 0]);
        return RTV.tensor(new FloatXArray(0), [1, 0]);
      }

      if (outModes.length === 1) return buildOutput(outModes[0]);
      const results: RuntimeValue[] = outModes.map(m => buildOutput(m));
      return results.slice(0, nargout);
    })
  );

  register(
    "regexpi",
    builtinSingle((args, nargout) => {
      if (args.length < 2)
        throw new RuntimeError("regexpi requires at least 2 arguments");
      const str = toString(args[0]);
      const pat = toString(args[1]);

      let matchOnce = false;
      const outModes: string[] = [];
      for (let i = 2; i < args.length; i++) {
        const opt = toString(args[i]).toLowerCase();
        if (opt === "once") matchOnce = true;
        else outModes.push(opt);
      }
      if (outModes.length === 0 && nargout <= 1) outModes.push("start");
      if (outModes.length === 0) {
        outModes.push(
          "start",
          "end",
          "tokenextents",
          "match",
          "tokens",
          "names"
        );
      }

      const re = new RegExp(pat, "gi");
      const starts: number[] = [];
      const ends: number[] = [];
      const matches: string[] = [];
      const tokensList: string[][] = [];
      let m: RegExpExecArray | null;
      while ((m = re.exec(str)) !== null) {
        starts.push(m.index + 1);
        ends.push(m.index + m[0].length);
        matches.push(m[0]);
        const toks: string[] = [];
        for (let g = 1; g < m.length; g++) toks.push(m[g] ?? "");
        tokensList.push(toks);
        if (matchOnce) break;
        if (m[0].length === 0) re.lastIndex++;
      }

      function buildOutput(mode: string): RuntimeValue {
        if (mode === "start") {
          if (matchOnce)
            return starts.length > 0
              ? RTV.num(starts[0])
              : RTV.tensor(new FloatXArray(0), [1, 0]);
          const d = new FloatXArray(starts);
          return RTV.tensor(d, [1, starts.length]);
        }
        if (mode === "end") {
          if (matchOnce)
            return ends.length > 0
              ? RTV.num(ends[0])
              : RTV.tensor(new FloatXArray(0), [1, 0]);
          const d = new FloatXArray(ends);
          return RTV.tensor(d, [1, ends.length]);
        }
        if (mode === "match") {
          if (matchOnce)
            return matches.length > 0 ? RTV.char(matches[0]) : RTV.char("");
          return RTV.cell(
            matches.map(s => RTV.char(s)),
            [1, matches.length]
          );
        }
        if (mode === "tokens") {
          if (matchOnce) {
            if (tokensList.length === 0) return RTV.cell([], [1, 0]);
            return RTV.cell(
              tokensList[0].map(s => RTV.char(s)),
              [1, tokensList[0].length]
            );
          }
          return RTV.cell(
            tokensList.map(toks =>
              RTV.cell(
                toks.map(s => RTV.char(s)),
                [1, toks.length]
              )
            ),
            [1, tokensList.length]
          );
        }
        if (mode === "names") {
          if (starts.length === 0) {
            return RTV.tensor(new FloatXArray(0), [0, 0]);
          }
          const namedRe = new RegExp(pat, "gi");
          const namedMatches: Record<string, string>[] = [];
          let nm: RegExpExecArray | null;
          while ((nm = namedRe.exec(str)) !== null) {
            namedMatches.push(
              Object.fromEntries(
                Object.entries(nm.groups ?? {}).map(([k, v]) => [k, v ?? ""])
              )
            );
            if (matchOnce) break;
            if (nm[0].length === 0) namedRe.lastIndex++;
          }
          if (matchOnce) {
            const fields: Record<string, RuntimeValue> = {};
            for (const [k, v] of Object.entries(namedMatches[0] ?? {})) {
              fields[k] = RTV.char(v);
            }
            return RTV.struct(fields);
          }
          if (namedMatches.length === 0) {
            return RTV.tensor(new FloatXArray(0), [0, 0]);
          }
          const keys = Object.keys(namedMatches[0]);
          const elements = namedMatches.map(nm2 => {
            const fields: Record<string, RuntimeValue> = {};
            for (const k of keys) fields[k] = RTV.char(nm2[k] ?? "");
            return RTV.struct(fields);
          });
          return RTV.structArray(keys, elements);
        }
        // tokenextents — return empty for now
        if (matchOnce) return RTV.tensor(new FloatXArray(0), [1, 0]);
        return RTV.tensor(new FloatXArray(0), [1, 0]);
      }

      if (outModes.length === 1) return buildOutput(outModes[0]);
      const results: RuntimeValue[] = outModes.map(m => buildOutput(m));
      return results.slice(0, nargout);
    })
  );

  register(
    "regexprep",
    builtinSingle(args => {
      if (args.length < 3)
        throw new RuntimeError("regexprep requires at least 3 arguments");
      const str = toString(args[0]);
      const pat = toString(args[1]);
      const rep = toString(args[2]);
      let flags = "g";
      for (let i = 3; i < args.length; i++) {
        const opt = toString(args[i]).toLowerCase();
        if (opt === "ignorecase") flags += "i";
      }
      const result = str.replace(new RegExp(pat, flags), rep);
      return isRuntimeChar(args[0]) ? RTV.char(result) : RTV.string(result);
    })
  );

  // ── String utility functions ────────────────────────────────────────

  register(
    "deblank",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("deblank requires 1 argument");
      const v = args[0];
      const s = toString(v);
      const result = s.replace(/\s+$/, "");
      return isRuntimeChar(v) ? RTV.char(result) : RTV.string(result);
    })
  );

  register(
    "blanks",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("blanks requires 1 argument");
      const n = Math.round(toNumber(args[0]));
      return RTV.char(" ".repeat(Math.max(0, n)));
    })
  );

  register(
    "strip",
    builtinSingle(args => {
      if (args.length < 1 || args.length > 2)
        throw new RuntimeError("strip requires 1 or 2 arguments");
      const v = args[0];
      const s = toString(v);
      let side = "both";
      if (args.length >= 2) side = toString(args[1]).toLowerCase();
      let result: string;
      if (side === "left") result = s.replace(/^\s+/, "");
      else if (side === "right") result = s.replace(/\s+$/, "");
      else result = s.trim();
      return isRuntimeChar(v) ? RTV.char(result) : RTV.string(result);
    })
  );

  register(
    "pad",
    builtinSingle(args => {
      if (args.length < 2)
        throw new RuntimeError("pad requires at least 2 arguments");
      const v = args[0];
      const s = toString(v);
      const n = Math.round(toNumber(args[1]));
      let side = "right";
      if (args.length >= 3) side = toString(args[2]).toLowerCase();
      if (s.length >= n) return isRuntimeChar(v) ? RTV.char(s) : RTV.string(s);
      const padding = " ".repeat(n - s.length);
      const result = side === "left" ? padding + s : s + padding;
      return isRuntimeChar(v) ? RTV.char(result) : RTV.string(result);
    })
  );

  register(
    "startsWith",
    builtinSingle(args => {
      if (args.length < 2)
        throw new RuntimeError("startsWith requires 2 arguments");
      return RTV.logical(toString(args[0]).startsWith(toString(args[1])));
    })
  );

  register(
    "endsWith",
    builtinSingle(args => {
      if (args.length < 2)
        throw new RuntimeError("endsWith requires 2 arguments");
      return RTV.logical(toString(args[0]).endsWith(toString(args[1])));
    })
  );

  register(
    "mat2str",
    builtinSingle(args => {
      if (args.length < 1)
        throw new RuntimeError("mat2str requires at least 1 argument");
      const v = args[0];
      const prec = args.length >= 2 ? Math.round(toNumber(args[1])) : -1;
      const fmt = (n: number) =>
        prec >= 0 ? Number(n.toPrecision(prec)).toString() : String(n);
      if (isRuntimeNumber(v)) return RTV.char(fmt(v));
      if (isRuntimeTensor(v)) {
        const nRows = v.shape[0] || 1;
        const nCols = v.data.length / nRows;
        if (nRows === 1 && nCols === 1) return RTV.char(fmt(v.data[0]));
        const rows: string[] = [];
        for (let r = 0; r < nRows; r++) {
          const elems: string[] = [];
          for (let c = 0; c < nCols; c++) {
            elems.push(fmt(v.data[c * nRows + r]));
          }
          rows.push(elems.join(" "));
        }
        return RTV.char("[" + rows.join(";") + "]");
      }
      return RTV.char(String(toNumber(v)));
    })
  );

  // ── count ────────────────────────────────────────────────────────────
  register(
    "count",
    builtinSingle(args => {
      if (args.length < 2) throw new RuntimeError("count requires 2 arguments");
      const s = toString(args[0]);
      const pat = toString(args[1]);
      if (pat.length === 0) return RTV.num(s.length + 1);
      let n = 0;
      let idx = 0;
      while ((idx = s.indexOf(pat, idx)) !== -1) {
        n++;
        idx += pat.length; // non-overlapping
      }
      return RTV.num(n);
    })
  );

  // ── replace (same as strrep) ────────────────────────────────────────
  register(
    "replace",
    builtinSingle(args => {
      if (args.length !== 3)
        throw new RuntimeError("replace requires 3 arguments");
      const v = args[0];
      const s = toString(v);
      const old = toString(args[1]);
      const rep = toString(args[2]);
      const result = s.split(old).join(rep);
      return isRuntimeChar(v) ? RTV.char(result) : RTV.string(result);
    })
  );

  // ── sscanf ──────────────────────────────────────────────────────────
  register(
    "sscanf",
    builtinSingle((args, nargout) => {
      if (args.length < 2)
        throw new RuntimeError("sscanf requires at least 2 arguments");
      const str = toString(args[0]);
      const fmt = toString(args[1]);
      const maxCount = args.length >= 3 ? toNumber(args[2]) : Infinity;

      const results: number[] = [];
      let strPos = 0;
      let fmtPos = 0;

      while (
        fmtPos < fmt.length &&
        strPos < str.length &&
        results.length < maxCount
      ) {
        if (fmt[fmtPos] === "%") {
          fmtPos++;
          if (fmtPos >= fmt.length) break;
          const spec = fmt[fmtPos];
          fmtPos++;

          // Skip whitespace before numeric reads
          if (spec !== "c" && spec !== "s") {
            while (strPos < str.length && /\s/.test(str[strPos])) strPos++;
          }

          if (spec === "d" || spec === "i") {
            const m = str.slice(strPos).match(/^[+-]?\d+/);
            if (!m) break;
            results.push(parseInt(m[0], 10));
            strPos += m[0].length;
          } else if (spec === "f" || spec === "e" || spec === "g") {
            const m = str
              .slice(strPos)
              .match(/^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?/);
            if (!m) break;
            results.push(parseFloat(m[0]));
            strPos += m[0].length;
          } else if (spec === "x") {
            const m = str.slice(strPos).match(/^[+-]?[0-9a-fA-F]+/);
            if (!m) break;
            results.push(parseInt(m[0], 16));
            strPos += m[0].length;
          } else if (spec === "o") {
            const m = str.slice(strPos).match(/^[+-]?[0-7]+/);
            if (!m) break;
            results.push(parseInt(m[0], 8));
            strPos += m[0].length;
          } else if (spec === "c") {
            results.push(str.charCodeAt(strPos));
            strPos++;
          } else if (spec === "s") {
            // Read non-whitespace characters, push each as char code
            const m = str.slice(strPos).match(/^\S+/);
            if (!m) break;
            for (
              let ci = 0;
              ci < m[0].length && results.length < maxCount;
              ci++
            ) {
              results.push(m[0].charCodeAt(ci));
            }
            strPos += m[0].length;
          }
        } else if (/\s/.test(fmt[fmtPos])) {
          // Whitespace in format matches any whitespace in string
          fmtPos++;
          while (strPos < str.length && /\s/.test(str[strPos])) strPos++;
        } else {
          // Literal character match
          if (str[strPos] !== fmt[fmtPos]) break;
          strPos++;
          fmtPos++;
        }

        // If format is exhausted but we haven't hit maxCount, restart format
        if (
          fmtPos >= fmt.length &&
          results.length < maxCount &&
          strPos < str.length
        ) {
          fmtPos = 0;
        }
      }

      const vals =
        results.length === 1
          ? RTV.num(results[0])
          : RTV.tensor(new FloatXArray(results), [results.length, 1]);
      if (nargout >= 2) {
        return [vals, RTV.num(results.length)];
      }
      return vals;
    })
  );

  // ── strtok ───────────────────────────────────────────────────────────
  register(
    "strtok",
    builtinSingle((args, nargout) => {
      if (args.length < 1 || args.length > 2)
        throw new RuntimeError("strtok requires 1 or 2 arguments");
      const s = toString(args[0]);
      const delims = args.length >= 2 ? toString(args[1]) : " \t\n\r\f\v";

      // Skip leading delimiters
      let start = 0;
      while (start < s.length && delims.includes(s[start])) start++;

      if (start >= s.length) {
        // All delimiters or empty
        if (nargout <= 1) return RTV.char("");
        return [RTV.char(""), RTV.char("")];
      }

      // Find end of token
      let end = start;
      while (end < s.length && !delims.includes(s[end])) end++;

      const token = s.substring(start, end);
      if (nargout <= 1) return RTV.char(token);
      return [RTV.char(token), RTV.char(s.substring(end))];
    })
  );

  // ── erase ───────────────────────────────────────────────────────────
  register(
    "erase",
    builtinSingle(args => {
      if (args.length !== 2)
        throw new RuntimeError("erase requires 2 arguments");
      const v = args[0];
      const s = toString(v);
      const pat = toString(args[1]);
      const result = s.split(pat).join("");
      return isRuntimeChar(v) ? RTV.char(result) : RTV.string(result);
    })
  );

  // ── reverse ─────────────────────────────────────────────────────────
  register(
    "reverse",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("reverse requires 1 argument");
      const v = args[0];
      const s = toString(v);
      const result = s.split("").reverse().join("");
      return isRuntimeChar(v) ? RTV.char(result) : RTV.string(result);
    })
  );

  // ── extractBefore / extractAfter ────────────────────────────────────
  register(
    "extractBefore",
    builtinSingle(args => {
      if (args.length !== 2)
        throw new RuntimeError("extractBefore requires 2 arguments");
      const v = args[0];
      const s = toString(v);
      const arg = args[1];
      let pos: number;
      if (
        isRuntimeNumber(arg) ||
        (isRuntimeTensor(arg) && arg.data.length === 1)
      ) {
        pos = Math.round(toNumber(arg)) - 1; // 1-based to 0-based
      } else {
        const pat = toString(arg);
        pos = s.indexOf(pat);
        if (pos === -1)
          throw new RuntimeError("extractBefore: pattern not found");
      }
      const result = s.substring(0, pos);
      return isRuntimeChar(v) ? RTV.char(result) : RTV.string(result);
    })
  );

  register(
    "extractAfter",
    builtinSingle(args => {
      if (args.length !== 2)
        throw new RuntimeError("extractAfter requires 2 arguments");
      const v = args[0];
      const s = toString(v);
      const arg = args[1];
      let pos: number;
      if (
        isRuntimeNumber(arg) ||
        (isRuntimeTensor(arg) && arg.data.length === 1)
      ) {
        pos = Math.round(toNumber(arg)); // 1-based, extractAfter starts after this position
      } else {
        const pat = toString(arg);
        const idx = s.indexOf(pat);
        if (idx === -1)
          throw new RuntimeError("extractAfter: pattern not found");
        pos = idx + pat.length;
      }
      const result = s.substring(pos);
      return isRuntimeChar(v) ? RTV.char(result) : RTV.string(result);
    })
  );

  // ── extractBetween ──────────────────────────────────────────────────
  register(
    "extractBetween",
    builtinSingle(args => {
      if (args.length !== 3)
        throw new RuntimeError("extractBetween requires 3 arguments");
      const v = args[0];
      const s = toString(v);
      const a = args[1];
      const b = args[2];
      let startPos: number;
      let endPos: number;
      if (isRuntimeNumber(a) || (isRuntimeTensor(a) && a.data.length === 1)) {
        startPos = Math.round(toNumber(a)) - 1; // 1-based to 0-based
      } else {
        const pat = toString(a);
        const idx = s.indexOf(pat);
        if (idx === -1)
          throw new RuntimeError("extractBetween: start pattern not found");
        startPos = idx + pat.length;
      }
      if (isRuntimeNumber(b) || (isRuntimeTensor(b) && b.data.length === 1)) {
        endPos = Math.round(toNumber(b)); // 1-based inclusive end -> 0-based exclusive
      } else {
        const pat = toString(b);
        const idx = s.indexOf(pat, startPos);
        if (idx === -1)
          throw new RuntimeError("extractBetween: end pattern not found");
        endPos = idx;
      }
      const result = s.substring(startPos, endPos);
      return isRuntimeChar(v) ? RTV.char(result) : RTV.string(result);
    })
  );

  // ── insertBefore / insertAfter ──────────────────────────────────────
  register(
    "insertBefore",
    builtinSingle(args => {
      if (args.length !== 3)
        throw new RuntimeError("insertBefore requires 3 arguments");
      const v = args[0];
      const s = toString(v);
      const arg = args[1];
      const ins = toString(args[2]);
      let pos: number;
      if (
        isRuntimeNumber(arg) ||
        (isRuntimeTensor(arg) && arg.data.length === 1)
      ) {
        pos = Math.round(toNumber(arg)) - 1;
      } else {
        const pat = toString(arg);
        pos = s.indexOf(pat);
        if (pos === -1)
          throw new RuntimeError("insertBefore: pattern not found");
      }
      const result = s.substring(0, pos) + ins + s.substring(pos);
      return isRuntimeChar(v) ? RTV.char(result) : RTV.string(result);
    })
  );

  register(
    "insertAfter",
    builtinSingle(args => {
      if (args.length !== 3)
        throw new RuntimeError("insertAfter requires 3 arguments");
      const v = args[0];
      const s = toString(v);
      const arg = args[1];
      const ins = toString(args[2]);
      let pos: number;
      if (
        isRuntimeNumber(arg) ||
        (isRuntimeTensor(arg) && arg.data.length === 1)
      ) {
        pos = Math.round(toNumber(arg));
      } else {
        const pat = toString(arg);
        const idx = s.indexOf(pat);
        if (idx === -1)
          throw new RuntimeError("insertAfter: pattern not found");
        pos = idx + pat.length;
      }
      const result = s.substring(0, pos) + ins + s.substring(pos);
      return isRuntimeChar(v) ? RTV.char(result) : RTV.string(result);
    })
  );

  // ── dec2hex / hex2dec / dec2bin / bin2dec ────────────────────────────
  register(
    "dec2hex",
    builtinSingle(args => {
      if (args.length < 1 || args.length > 2)
        throw new RuntimeError("dec2hex requires 1 or 2 arguments");
      const n = Math.round(toNumber(args[0]));
      let hex = n.toString(16).toUpperCase();
      if (args.length === 2) {
        const minDigits = Math.round(toNumber(args[1]));
        while (hex.length < minDigits) hex = "0" + hex;
      }
      return RTV.char(hex);
    })
  );

  register(
    "hex2dec",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("hex2dec requires 1 argument");
      return RTV.num(parseInt(toString(args[0]), 16));
    })
  );

  register(
    "dec2bin",
    builtinSingle(args => {
      if (args.length < 1 || args.length > 2)
        throw new RuntimeError("dec2bin requires 1 or 2 arguments");
      const n = Math.round(toNumber(args[0]));
      let bin = n.toString(2);
      if (args.length === 2) {
        const minDigits = Math.round(toNumber(args[1]));
        while (bin.length < minDigits) bin = "0" + bin;
      }
      return RTV.char(bin);
    })
  );

  register(
    "bin2dec",
    builtinSingle(args => {
      if (args.length !== 1)
        throw new RuntimeError("bin2dec requires 1 argument");
      return RTV.num(parseInt(toString(args[0]), 2));
    })
  );

  // symvar: extract variable names from expression string using the numbl parser
  registerSymvar();
}

function walkExpr(e: Expr, found: Set<string>): void {
  switch (e.type) {
    case "Ident":
      found.add(e.name);
      break;
    case "FuncCall":
      // Don't add function name; walk arguments only
      for (const arg of e.args) walkExpr(arg, found);
      break;
    case "Binary":
      walkExpr(e.left, found);
      walkExpr(e.right, found);
      break;
    case "Unary":
      walkExpr(e.operand, found);
      break;
    case "Index":
    case "IndexCell":
      walkExpr(e.base, found);
      for (const idx of e.indices) walkExpr(idx, found);
      break;
    case "Range":
      walkExpr(e.start, found);
      if (e.step) walkExpr(e.step, found);
      walkExpr(e.end, found);
      break;
    case "Member":
      walkExpr(e.base, found);
      break;
    case "MemberDynamic":
      walkExpr(e.base, found);
      walkExpr(e.nameExpr, found);
      break;
    case "MethodCall":
      walkExpr(e.base, found);
      for (const arg of e.args) walkExpr(arg, found);
      break;
    case "AnonFunc":
      // Don't descend — anonymous function params are bound
      break;
    case "Tensor":
    case "Cell":
      for (const row of e.rows) for (const el of row) walkExpr(el, found);
      break;
    case "ClassInstantiation":
      for (const arg of e.args) walkExpr(arg, found);
      break;
    case "SuperMethodCall":
      for (const arg of e.args) walkExpr(arg, found);
      break;
    // Leaf nodes with no identifiers
    case "Number":
    case "Char":
    case "String":
    case "ImagUnit":
    case "EndKeyword":
    case "Colon":
    case "FuncHandle":
    case "MetaClass":
      break;
  }
}

function walkStmt(s: Stmt, found: Set<string>): void {
  switch (s.type) {
    case "Assign":
      walkExpr(s.expr, found);
      break;
    case "ExprStmt":
      walkExpr(s.expr, found);
      break;
    default:
      break;
  }
}

function registerSymvar(): void {
  register(
    "symvar",
    builtinSingle(args => {
      if (args.length < 1 || args.length > 2)
        throw new RuntimeError("symvar requires 1 or 2 arguments");
      const expr = toString(args[0]);

      // Parse the expression wrapped as a dummy assignment
      let ast;
      try {
        ast = parseMFile(`__symvar_out__ = ${expr};`, "__symvar__.m");
      } catch {
        return RTV.cell([], [1, 0]);
      }

      // Walk AST collecting Ident nodes (free variables)
      const found = new Set<string>();
      for (const stmt of ast.body) {
        walkStmt(stmt, found);
      }

      // Remove the dummy variable
      found.delete("__symvar_out__");

      // Remove builtin constants and builtin functions
      for (const name of found) {
        if (BUILTIN_CONSTANTS.has(name) || isBuiltin(name)) {
          found.delete(name);
        }
      }

      // Sort: uppercase letters before lowercase, then alphabetical
      const sorted = [...found].sort((a, b) => {
        const aUpper = a[0] >= "A" && a[0] <= "Z";
        const bUpper = b[0] >= "A" && b[0] <= "Z";
        if (aUpper && !bUpper) return -1;
        if (!aUpper && bUpper) return 1;
        return a < b ? -1 : a > b ? 1 : 0;
      });

      let result = sorted;
      // symvar(s, n): select n variables closest to 'x'
      if (args.length === 2) {
        const n = Math.round(toNumber(args[1]));
        const byDist = [...sorted].sort((a, b) => {
          const distA = Math.abs(a.charCodeAt(0) - "x".charCodeAt(0));
          const distB = Math.abs(b.charCodeAt(0) - "x".charCodeAt(0));
          if (distA !== distB) return distA - distB;
          return a < b ? -1 : a > b ? 1 : 0;
        });
        const selected = new Set(byDist.slice(0, n));
        result = sorted.filter(v => selected.has(v));
      }

      return RTV.cell(
        result.map(v => RTV.char(v)),
        [1, result.length]
      );
    })
  );
}
