// JS sibling of `format_engine.h`. Numbl-compatible printf-style
// formatter, ported from `numbl/src/numbl-core/helpers/string.ts`'s
// `sprintfFormat`. Shared by fprintf / sprintf / error / assert on
// the JS side.
//
// Spec set: d i u f e E g G x X o s c %
// Flags: - + 0 # space. Precision `.N`. Width digits. `*` consumes the
// next numeric arg as a width.
// Escapes interpreted AT FORMAT TIME: \n \t \r \a \b \f \v \0 \\.
// Numeric tensors flatten column-major into the scalar stream; the
// format string cycles through args until consumed (with a "no arg
// consumed this pass" guard against infinite output).

function isTensor(v) {
  return typeof v === "object" && v !== null && v.mtoc2Tag === "tensor";
}
function isChar(v) {
  return typeof v === "object" && v !== null && v.mtoc2Tag === "char";
}

function toNumber(v) {
  if (typeof v === "number") return v;
  if (typeof v === "boolean") return v ? 1 : 0;
  if (typeof v === "string") return Number(v);
  if (isChar(v)) {
    // Match the interpreter's toNumber (runtime/convert.ts): a single
    // char yields its code point; a multi-char value has no scalar
    // numeric form. `Number("A")` would wrongly give NaN.
    if (v.value.length === 1) return v.value.charCodeAt(0);
    throw new Error("Cannot convert multi-char to number");
  }
  if (isTensor(v) && v.data.length === 1) return v.data[0];
  return Number(v);
}

function toString(v) {
  if (typeof v === "string") return v;
  if (typeof v === "number") return String(v);
  if (typeof v === "boolean") return v ? "1" : "0";
  if (isChar(v)) return v.value;
  return String(v);
}

function numStr(n) {
  if (n === Infinity) return "Inf";
  if (n === -Infinity) return "-Inf";
  if (Number.isNaN(n)) return "NaN";
  if (n === 0) return "0";
  const prec = 5;
  const exp = Math.floor(Math.log10(Math.abs(n)));
  let s;
  if (exp < -4 || exp >= prec) {
    s = n.toExponential(prec - 1);
    const ePos = s.indexOf("e");
    let mantissa = s.slice(0, ePos);
    const expPart0 = s.slice(ePos);
    if (mantissa.includes(".")) mantissa = mantissa.replace(/\.?0+$/, "");
    const expPart = expPart0.replace(/([eE][+-])(\d)$/, "$1" + "0$2");
    s = mantissa + expPart;
  } else {
    if (Number.isInteger(n)) return String(n);
    s = n.toPrecision(prec);
    if (s.includes(".")) s = s.replace(/\.?0+$/, "");
  }
  return s;
}

function applyWidth(spec, str) {
  const m = spec.match(/^%([-+ #]*)0?(\d+)?/);
  if (!m) return str;
  const explicitFlags = m[1] || "";
  const leftAlign = explicitFlags.includes("-");
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

export function mtoc2_sprintf_format(fmt, args) {
  // Char literals (`'foo'`) come through wrapped as
  // `{mtoc2Tag:"char", value:"..."}` to match the interpreter's
  // RuntimeValue convention; unwrap so the format walker indexes
  // a plain JS string.
  if (isChar(fmt)) fmt = fmt.value;
  // Flatten args column-major (tensors expand element-by-element).
  const flat = [];
  for (const arg of args) {
    if (isTensor(arg)) {
      for (let k = 0; k < arg.data.length; k++) flat.push(arg.data[k]);
    } else {
      flat.push(arg);
    }
  }

  let result = "";
  let argIdx = 0;

  do {
    const startArgIdx = argIdx;
    let outOfArgs = false;
    let i = 0;
    while (i < fmt.length && !outOfArgs) {
      if (fmt[i] === "%" && i + 1 < fmt.length) {
        i++;
        let spec = "%";
        while (i < fmt.length && !"dfigeEsoxXuc%".includes(fmt[i])) {
          if (fmt[i] === "*") {
            if (argIdx >= flat.length) {
              outOfArgs = true;
              break;
            }
            spec += String(Math.round(toNumber(flat[argIdx++])));
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
          } else if (argIdx >= flat.length) {
            outOfArgs = true;
          } else if (ch === "d" || ch === "i" || ch === "u") {
            const raw = toNumber(flat[argIdx++]);
            const isInt = Number.isInteger(raw);
            const canPrintAsInt = ch === "u" ? isInt && raw >= 0 : isInt;
            if (!canPrintAsInt) {
              let eStr = raw.toExponential(6);
              eStr = eStr.replace(/e([+-])(\d)$/, "e$1" + "0$2");
              result += applyWidth(spec, eStr);
            } else {
              const n = raw;
              const flags = spec.slice(1);
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
            const n = toNumber(flat[argIdx++]);
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
            const n = toNumber(flat[argIdx++]);
            if (!isFinite(n) || isNaN(n)) {
              result += applyWidth(spec, numStr(n));
            } else {
              const precMatch = spec.match(/\.(\d+)/);
              const prec = precMatch ? parseInt(precMatch[1]) : 6;
              let eStr = n.toExponential(prec);
              eStr = eStr.replace(/e([+-])(\d)$/, "e$1" + "0$2");
              if (ch === "E") eStr = eStr.toUpperCase();
              result += applyWidth(spec, eStr);
            }
          } else if (ch === "x" || ch === "X") {
            const n = Math.round(toNumber(flat[argIdx++]));
            let s = Math.abs(n).toString(16);
            if (ch === "X") s = s.toUpperCase();
            result += applyWidth(spec, s);
          } else if (ch === "o") {
            const n = Math.round(toNumber(flat[argIdx++]));
            result += applyWidth(spec, Math.abs(n).toString(8));
          } else if (ch === "g" || ch === "G") {
            const gVal = toNumber(flat[argIdx++]);
            if (!isFinite(gVal) || isNaN(gVal)) {
              result += applyWidth(spec, numStr(gVal));
            } else {
              const precMatch = spec.match(/\.(\d+)/);
              const gPrec = precMatch ? parseInt(precMatch[1]) : 6;
              let gStr;
              if (gVal === 0) {
                gStr = "0";
              } else {
                const exp = Math.floor(Math.log10(Math.abs(gVal)));
                if (exp < -4 || exp >= gPrec) {
                  gStr = gVal.toExponential(gPrec - 1);
                  const ePos = gStr.indexOf("e");
                  let mantissa = gStr.slice(0, ePos);
                  let expPart = gStr.slice(ePos);
                  if (mantissa.includes(".")) {
                    mantissa = mantissa.replace(/\.?0+$/, "");
                  }
                  expPart = expPart.replace(/e([+-])(\d)$/, "e$1" + "0$2");
                  gStr = mantissa + expPart;
                } else {
                  gStr = gVal.toPrecision(gPrec);
                  if (gStr.includes(".")) {
                    gStr = gStr.replace(/\.?0+$/, "");
                  }
                  if (gStr.includes("e")) {
                    gStr = String(parseFloat(gStr));
                  }
                }
              }
              if (ch === "G") gStr = gStr.toUpperCase();
              result += applyWidth(spec, gStr);
            }
          } else if (ch === "s") {
            const sArg = flat[argIdx++];
            // A numeric/logical argument to %s is a character code (MATLAB:
            // sprintf('%s', 65) -> 'A').
            const sVal =
              typeof sArg === "number" || typeof sArg === "boolean"
                ? String.fromCharCode(Math.round(toNumber(sArg)))
                : toString(sArg);
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
            result += String.fromCharCode(Math.round(toNumber(flat[argIdx++])));
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
          case "r":
            result += "\r";
            break;
          case "a":
            result += "\x07";
            break;
          case "b":
            result += "\b";
            break;
          case "f":
            result += "\f";
            break;
          case "v":
            result += "\v";
            break;
          case "0":
            result += "\0";
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
    if (argIdx === startArgIdx) break;
  } while (argIdx < flat.length);

  return result;
}
