// JS sibling of `format_double.h` — same numbl-compatible formatter,
// in JS. Used by `emitJs` for the `disp` scalar path and (later) by
// the interpreter via the re-export in `snippets.gen.ts`.

export function mtoc2_format_double(x) {
  if (Number.isNaN(x)) return "NaN";
  if (x === Infinity) return "Infinity";
  if (x === -Infinity) return "-Infinity";
  if (x === 0) x = 0;
  if (Math.abs(x) < 1e15 && x === Math.trunc(x)) return String(x);

  const buf = x.toExponential(4);
  const ePos = buf.indexOf("e");
  if (ePos < 0) return buf;
  const decExp = parseInt(buf.slice(ePos + 1), 10);

  if (decExp < -6 || decExp >= 5) {
    let mantissaEnd = ePos - 1;
    while (mantissaEnd > 0 && buf[mantissaEnd] === "0") mantissaEnd--;
    if (mantissaEnd >= 0 && buf[mantissaEnd] === ".") mantissaEnd--;
    let expPart = buf.slice(ePos + 1);
    let sign = "+";
    if (expPart[0] === "+" || expPart[0] === "-") {
      sign = expPart[0];
      expPart = expPart.slice(1);
    }
    while (expPart.length > 1 && expPart[0] === "0") expPart = expPart.slice(1);
    return buf.slice(0, mantissaEnd + 1) + "e" + sign + expPart;
  }

  let fracDigits = 4 - decExp;
  if (fracDigits < 0) fracDigits = 0;
  let s = x.toFixed(fracDigits);
  if (s.includes(".")) {
    while (s.endsWith("0")) s = s.slice(0, -1);
    if (s.endsWith(".")) s = s.slice(0, -1);
  }
  return s;
}
