/**
 * String manipulation builtin functions
 */

import { type RuntimeValue, toNumber, toString } from "../runtime/index.js";
import { isRuntimeTensor } from "../runtime/types.js";

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
        // Parse format specifier. A '*' for width or precision consumes the
        // next arg as an integer and substitutes its digits into the spec.
        let spec = "%";
        while (i < fmt.length && !"dfigeEsoxXuc%".includes(fmt[i])) {
          if (fmt[i] === "*") {
            if (argIdx >= flatArgs.length) {
              outOfArgs = true;
              break;
            }
            spec += String(Math.round(toNumber(flatArgs[argIdx++])));
            i++;
          } else {
            spec += fmt[i];
            i++;
          }
        }
        if (outOfArgs) break;
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
