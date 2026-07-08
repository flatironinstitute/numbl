/**
 * Shared scanning core for sscanf/fscanf: apply a MATLAB format string
 * cyclically to input text, collecting numeric results.
 */

export interface ScanResult {
  results: number[];
  /** Number of characters consumed from the input. */
  consumed: number;
  matchFailure: boolean;
}

export function scanFormat(
  str: string,
  fmt: string,
  maxCount: number
): ScanResult {
  const results: number[] = [];
  let strPos = 0;
  let fmtPos = 0;
  let matchFailure = false;

  while (
    fmtPos < fmt.length &&
    strPos < str.length &&
    results.length < maxCount
  ) {
    if (fmt[fmtPos] === "%") {
      fmtPos++;
      if (fmtPos >= fmt.length) break;
      // Skip width/precision modifiers, e.g. %10f or %5.2f
      while (fmtPos < fmt.length && /[\d.]/.test(fmt[fmtPos])) fmtPos++;
      if (fmtPos >= fmt.length) break;
      const spec = fmt[fmtPos];
      fmtPos++;
      if (spec !== "c" && spec !== "s") {
        while (strPos < str.length && /\s/.test(str[strPos])) strPos++;
      }
      if (spec === "d" || spec === "i" || spec === "u") {
        const m = str.slice(strPos).match(/^[+-]?\d+/);
        if (!m) {
          matchFailure = true;
          break;
        }
        results.push(parseInt(m[0], 10));
        strPos += m[0].length;
      } else if (spec === "f" || spec === "e" || spec === "g") {
        const m = str
          .slice(strPos)
          .match(/^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?/);
        if (!m) {
          matchFailure = true;
          break;
        }
        results.push(parseFloat(m[0]));
        strPos += m[0].length;
      } else if (spec === "x") {
        const m = str.slice(strPos).match(/^[+-]?[0-9a-fA-F]+/);
        if (!m) {
          matchFailure = true;
          break;
        }
        results.push(parseInt(m[0], 16));
        strPos += m[0].length;
      } else if (spec === "o") {
        const m = str.slice(strPos).match(/^[+-]?[0-7]+/);
        if (!m) {
          matchFailure = true;
          break;
        }
        results.push(parseInt(m[0], 8));
        strPos += m[0].length;
      } else if (spec === "c") {
        results.push(str.charCodeAt(strPos));
        strPos++;
      } else if (spec === "s") {
        const m = str.slice(strPos).match(/^\S+/);
        if (!m) {
          matchFailure = true;
          break;
        }
        for (let ci = 0; ci < m[0].length && results.length < maxCount; ci++) {
          results.push(m[0].charCodeAt(ci));
        }
        strPos += m[0].length;
      }
    } else if (/\s/.test(fmt[fmtPos])) {
      fmtPos++;
      while (strPos < str.length && /\s/.test(str[strPos])) strPos++;
    } else {
      if (str[strPos] !== fmt[fmtPos]) {
        matchFailure = true;
        break;
      }
      strPos++;
      fmtPos++;
    }
    if (
      fmtPos >= fmt.length &&
      results.length < maxCount &&
      strPos < str.length
    ) {
      fmtPos = 0;
    }
  }

  return { results, consumed: strPos, matchFailure };
}
